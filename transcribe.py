"""
transcribe.py
-------------
Speech-to-text with pluggable backends.
FIXED: Aggressive repetition suppression for Hindi/music, quality gating,
music/noise detection, and timestamp validation.
"""

from pathlib import Path
from typing import Optional
import gc
import re
import subprocess
import tempfile

import requests

import config
from quality_validation import (
    validate_asr_segment,
    validate_timestamps,
    detect_music_noise,
    get_validator,
)

log = config.get_logger(__name__)

_MAX_BYTES = 25 * 1024 * 1024

# Quality thresholds (configurable via .env)
MAX_REPETITION_RATIO = config.ASR_MAX_REPETITION_RATIO
MIN_LANGUAGE_PROB = config.ASR_MIN_LANGUAGE_PROB
MIN_AVG_LOGPROB = config.ASR_MIN_AVG_LOGPROB
MIN_SEGMENT_CONFIDENCE = 0.6
MAX_MUSIC_SEGMENTS_RATIO = 0.3


def unload_asr_resources() -> None:
    """Release ASR-side resources between stages on low-VRAM systems."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def _validate_audio(audio_path: Path) -> int:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path.stat().st_size


def _resolve_device_and_compute_type() -> tuple[str, str]:
    device = config.FASTER_WHISPER_DEVICE
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    compute_type = config.FASTER_WHISPER_COMPUTE_TYPE
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    if device == "cpu" and compute_type in {"float16", "bfloat16"}:
        log.warning("compute_type=%s not suitable for CPU. Falling back to int8.", compute_type)
        compute_type = "int8"

    return device, compute_type


def _calculate_repetition_score(text: str) -> float:
    """
    Calculate repetition ratio in text.
    Returns 0.0 (no repetition) to 1.0 (all repetition).
    """
    if not text or len(text) < 20:
        return 0.0
    
    words = text.split()
    if len(words) < 5:
        return 0.0
    
    # Count repeated consecutive tokens
    repeated_count = 0
    total_tokens = len(words)
    
    i = 0
    while i < len(words):
        word = words[i]
        # Look for runs of same/similar words
        run_length = 1
        j = i + 1
        while j < len(words):
            # Normalize for comparison
            w1 = re.sub(r"[^\w]", "", word.lower())
            w2 = re.sub(r"[^\w]", "", words[j].lower())
            if w1 and w2 and (w1 == w2 or w1 in w2 or w2 in w1):
                run_length += 1
                j += 1
            else:
                break
        
        if run_length >= 2:  # Count as repetition if repeated
            repeated_count += run_length
        
        i = j if j > i + 1 else i + 1
    
    return min(repeated_count / total_tokens, 1.0)


def _aggressive_squash_repetition(text: str) -> str:
    """
    Aggressive deduplication for pathological repeats like 'protty protty protty'.
    Handles Hindi-English mixed content and music-induced hallucinations.
    """
    if not text:
        return text
    
    # First pass: collapse immediate word repeats
    words = text.split()
    if not words:
        return text
    
    collapsed = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        # Normalize for comparison (keep original for output)
        normalized = re.sub(r"[^\w\u0900-\u097f]", "", word.lower())
        
        if normalized and normalized == prev_word:
            repeat_count += 1
            if repeat_count <= 1:  # Allow max 1 repeat
                collapsed.append(word)
        else:
            prev_word = normalized
            repeat_count = 0
            collapsed.append(word)
    
    text = " ".join(collapsed)
    
    # Second pass: collapse phrase repeats (2-4 word phrases)
    for phrase_len in range(4, 1, -1):
        pattern = r"(\b(?:\w+\s+){" + str(phrase_len-1) + r"}\w+\b)(?:\s+\1)+"
        text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
    
    # Third pass: clean up music/noise artifacts
    noise_patterns = [
        r"\b(protty|prottty|prr+|t+\s*t+|d+\s*d+)\b",
        r"\b(m+m+|h+h+|a+a+|o+o+)\b",
        r"\d{4,}",  # Long number sequences
        r"\b(music|instrumental|bgm)\b",  # ASR hallucinations for music
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def _detect_music_heavy_audio(segments: list[dict]) -> tuple[bool, list[int]]:
    """
    Detect if audio is music-heavy or has significant non-speech content.
    
    Returns:
        Tuple of (is_music_heavy, suspicious_segment_indices)
    """
    if not segments:
        return False, []
    
    suspicious_indices = detect_music_noise(segments)
    music_ratio = len(suspicious_indices) / len(segments) if segments else 0
    
    is_music_heavy = music_ratio > MAX_MUSIC_SEGMENTS_RATIO
    
    if is_music_heavy:
        log.warning("Detected music-heavy audio: %.1f%% of segments flagged as music/noise", 
                   music_ratio * 100)
    
    return is_music_heavy, suspicious_indices


def _split_overlong_segments(segments: list[dict]) -> list[dict]:
    """
    Split very long ASR segments into smaller pieces for better MT/TTS stability.
    """
    max_dur = max(2.0, float(config.ASR_MAX_SEGMENT_DURATION))
    out: list[dict] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        dur = end - start
        if dur <= max_dur or not text:
            out.append(seg)
            continue

        parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()]
        if len(parts) < 2:
            words = text.split()
            if len(words) >= 8:
                mid = len(words) // 2
                parts = [" ".join(words[:mid]), " ".join(words[mid:])]
            else:
                out.append(seg)
                continue

        total_chars = sum(max(1, len(p)) for p in parts)
        cur = start
        for i, part in enumerate(parts):
            frac = max(1, len(part)) / total_chars
            part_dur = dur * frac
            part_end = end if i == len(parts) - 1 else min(end, cur + part_dur)
            seg_new = dict(seg)
            seg_new["start"] = float(cur)
            seg_new["end"] = float(part_end)
            seg_new["text"] = part
            out.append(seg_new)
            cur = part_end
    return out


def _extract_audio_slice(
    audio_path: Path,
    start: float,
    end: float,
    out_path: Path,
) -> Path:
    duration = max(0.05, end - start)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def _refine_low_confidence_segments(
    model,
    audio_path: Path,
    segments: list[dict],
    *,
    language: Optional[str],
) -> list[dict]:
    """
    Re-transcribe low-confidence segments with relaxed decoding to improve accuracy
    on noisy/music-heavy windows.
    """
    if not config.ASR_REFINE_LOW_CONF_SEGMENTS_ENABLED:
        return segments

    candidates: list[int] = []
    for idx, seg in enumerate(segments):
        avg_logprob = float(seg.get("avg_logprob", 0.0))
        if avg_logprob < MIN_AVG_LOGPROB or seg.get("is_suspicious"):
            candidates.append(idx)

    if not candidates:
        return segments

    candidates = candidates[: max(0, config.ASR_REFINE_MAX_SEGMENTS)]
    log.info("Refining %d low-confidence ASR segments...", len(candidates))

    refined = [dict(s) for s in segments]
    margin = max(0.0, float(config.ASR_REFINE_MARGIN_SEC))

    with tempfile.TemporaryDirectory(prefix="asr_refine_") as td:
        tmp_dir = Path(td)
        for idx in candidates:
            seg = refined[idx]
            start = max(0.0, float(seg.get("start", 0.0)) - margin)
            end = float(seg.get("end", start)) + margin
            clip = tmp_dir / f"seg_{idx:04d}.wav"
            try:
                _extract_audio_slice(audio_path, start, end, clip)
                alt_segments, _ = model.transcribe(
                    str(clip),
                    language=language,
                    task="transcribe",
                    vad_filter=False,
                    beam_size=max(6, config.FASTER_WHISPER_BEAM_SIZE + 2),
                    best_of=max(5, config.FASTER_WHISPER_BEST_OF),
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.2,
                    compression_ratio_threshold=2.4,
                    repetition_penalty=1.1,
                )
                alt_text = " ".join(
                    _aggressive_squash_repetition((s.text or "").strip())
                    for s in alt_segments
                    if (s.text or "").strip()
                ).strip()
                if alt_text and len(alt_text) >= max(6, len(str(seg.get("text", "")).strip()) // 3):
                    refined[idx]["text"] = alt_text
                    refined[idx]["refined"] = True
            except Exception as exc:
                log.debug("Refinement skipped for segment %d: %s", idx, exc)

    return refined


def _refine_large_gaps(
    model,
    audio_path: Path,
    segments: list[dict],
    *,
    language: Optional[str],
) -> list[dict]:
    """
    Recover missed speech in large timestamp gaps by re-transcribing gap windows
    with VAD disabled. Useful when aggressive VAD drops short/low-energy speech.
    """
    if not segments:
        return segments

    enabled = bool(getattr(config, "ASR_GAP_REFINE_ENABLED", True))
    if not enabled:
        return segments

    min_gap_sec = float(getattr(config, "ASR_GAP_REFINE_MIN_GAP_SEC", 2.5))
    max_gap_sec = float(getattr(config, "ASR_GAP_REFINE_MAX_GAP_SEC", 15.0))
    max_gaps = int(getattr(config, "ASR_GAP_REFINE_MAX_GAPS", 6))
    margin_sec = max(0.0, float(getattr(config, "ASR_GAP_REFINE_MARGIN_SEC", 0.05)))

    candidates: list[tuple[int, float, float]] = []
    for i in range(1, len(segments)):
        prev_end = float(segments[i - 1].get("end", 0.0))
        next_start = float(segments[i].get("start", prev_end))
        gap = next_start - prev_end
        if min_gap_sec <= gap <= max_gap_sec:
            candidates.append((i, prev_end, next_start))

    if not candidates:
        return segments

    candidates = candidates[: max(0, max_gaps)]
    log.info("Refining %d large ASR gaps...", len(candidates))

    inserted: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="asr_gap_refine_") as td:
        tmp_dir = Path(td)
        for gap_idx, gap_start, gap_end in candidates:
            clip_start = max(0.0, gap_start + margin_sec)
            clip_end = max(clip_start + 0.10, gap_end - margin_sec)
            clip = tmp_dir / f"gap_{gap_idx:04d}.wav"
            try:
                _extract_audio_slice(audio_path, clip_start, clip_end, clip)
                alt_segments, _ = model.transcribe(
                    str(clip),
                    language=language,
                    task="transcribe",
                    vad_filter=False,
                    beam_size=max(6, config.FASTER_WHISPER_BEAM_SIZE + 1),
                    best_of=max(5, config.FASTER_WHISPER_BEST_OF),
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.8,
                    log_prob_threshold=-1.4,
                    compression_ratio_threshold=2.6,
                    repetition_penalty=1.1,
                )

                recovered_here = 0
                for s in alt_segments:
                    t = _aggressive_squash_repetition((s.text or "").strip())
                    if not t:
                        continue
                    abs_start = clip_start + float(getattr(s, "start", 0.0))
                    abs_end = clip_start + float(getattr(s, "end", abs_start))
                    if abs_end - abs_start < 0.20:
                        continue
                    inserted.append(
                        {
                            "start": float(abs_start),
                            "end": float(abs_end),
                            "text": t,
                            "speaker_id": "spk_00",
                            "gap_refined": True,
                        }
                    )
                    recovered_here += 1
                if recovered_here > 0:
                    log.info(
                        "Recovered %d segment(s) in gap %.2fs-%.2fs",
                        recovered_here,
                        gap_start,
                        gap_end,
                    )
            except Exception as exc:
                log.debug(
                    "Gap refinement skipped for %.2fs-%.2fs: %s",
                    gap_start,
                    gap_end,
                    exc,
                )

    if not inserted:
        return segments

    merged = [dict(s) for s in segments] + inserted
    merged.sort(key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))))
    merged = validate_timestamps(merged)
    merged = _split_overlong_segments(merged)
    return merged


def _check_quality_gate(
    text: str,
    detected_lang: str,
    expected_lang: Optional[str],
    lang_prob: float,
    avg_logprob: float = 0.0,
    segment_confidence: float = 0.0,
) -> tuple[bool, str]:
    """
    Quality gate: returns (passed, reason) tuple.
    Hard stop if quality is too low.
    """
    # Check 1: Repetition score
    rep_score = _calculate_repetition_score(text)
    if rep_score > MAX_REPETITION_RATIO:
        return False, f"Repetition score {rep_score:.2f} exceeds threshold {MAX_REPETITION_RATIO}"
    
    # Check 2: Language mismatch with higher threshold
    if expected_lang and expected_lang != "auto":
        if detected_lang != expected_lang and lang_prob < MIN_LANGUAGE_PROB:
            return False, f"Language mismatch: detected {detected_lang} (prob {lang_prob:.2f}) vs expected {expected_lang}"
    
    # Check 3: ASR confidence
    if avg_logprob < MIN_AVG_LOGPROB:
        return False, f"Low ASR confidence: avg_logprob={avg_logprob:.2f} (min={MIN_AVG_LOGPROB})"
    
    if segment_confidence > 0 and segment_confidence < MIN_SEGMENT_CONFIDENCE:
        return False, f"Low segment confidence: {segment_confidence:.2f} (min={MIN_SEGMENT_CONFIDENCE})"
    
    # Check 4: Empty or too short
    clean_text = _aggressive_squash_repetition(text)
    if len(clean_text.strip()) < 10:
        return False, "Transcript too short after cleaning"
    
    # Check 5: Garbage ratio (non-word characters)
    if len(text) > 0:
        garbage_ratio = len(re.findall(r"[^\w\s\u0900-\u097f\u0A80-\u0AFF\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F]", text)) / len(text)
        if garbage_ratio > 0.25:  # Reduced from 0.3
            return False, f"Garbage character ratio {garbage_ratio:.2f} too high"
    
    return True, "Quality checks passed"


def _transcribe_nvidia(
    audio_path: Path,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    file_size = _validate_audio(audio_path)
    if file_size > _MAX_BYTES:
        raise ValueError(f"Audio file is {file_size / 1e6:.1f} MB and exceeds the 25 MB limit.")

    if not config.NVIDIA_API_KEY:
        raise EnvironmentError("NVIDIA_API_KEY is not set for nvidia transcription backend.")

    log.info(
        "Transcribing '%s' (%.1f MB) via NVIDIA model '%s'...",
        audio_path.name,
        file_size / 1e6,
        config.NVIDIA_WHISPER_MODEL,
    )

    url = f"{config.NVIDIA_BASE_URL.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {config.NVIDIA_API_KEY}"}
    data: dict[str, str] = {
        "model": config.NVIDIA_WHISPER_MODEL,
        "response_format": "text",
        "temperature": str(temperature),
    }
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "audio/wav")}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=600)

    if response.status_code >= 400:
        raise RuntimeError(f"NVIDIA transcription failed ({response.status_code}): {response.text}")

    transcript = response.text.strip()
    if not transcript:
        raise RuntimeError("NVIDIA transcription returned empty text.")

    log.info("NVIDIA transcription complete. chars=%d", len(transcript))
    return transcript


def _transcribe_faster_whisper(
    audio_path: Path,
    *,
    language: Optional[str] = None,
    quality_gate: bool = True,
) -> str:
    _validate_audio(audio_path)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError("faster-whisper is not installed.") from exc

    device, compute_type = _resolve_device_and_compute_type()

    log.info(
        "Transcribing '%s' locally with faster-whisper model='%s' device='%s' compute_type='%s'...",
        audio_path.name,
        config.FASTER_WHISPER_MODEL,
        device,
        compute_type,
    )
    
    try:
        model = WhisperModel(
            config.FASTER_WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )
    except ValueError as exc:
        if device == "cuda" and "float16" in str(exc).lower():
            log.warning("float16 failed on CUDA runtime. Falling back to float32.")
            model = WhisperModel(
                config.FASTER_WHISPER_MODEL,
                device=device,
                compute_type="float32",
            )
        else:
            raise

    # ENHANCED VAD for music/noise suppression
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        vad_filter=config.FASTER_WHISPER_VAD_FILTER,
        vad_parameters={
            "threshold": 0.6,  # Increased from 0.5 for stricter speech detection
            "min_speech_duration_ms": 300,  # Increased from 250
            "max_speech_duration_s": 30,
            "min_silence_duration_ms": 500,  # Added: require longer silence gaps
        } if config.FASTER_WHISPER_VAD_FILTER else None,
        beam_size=config.FASTER_WHISPER_BEAM_SIZE,
        best_of=config.FASTER_WHISPER_BEST_OF,
        condition_on_previous_text=config.FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        temperature=0.0,
        no_speech_threshold=0.7,  # Increased from 0.6
        # Keep decoder threshold looser than quality gate to avoid premature drop.
        log_prob_threshold=min(-1.0, MIN_AVG_LOGPROB),
        compression_ratio_threshold=2.2,  # Tightened from 2.4
        repetition_penalty=1.3,  # Increased from 1.2
    )

    # Process segments with aggressive cleaning
    raw_text_parts = []
    segment_data = []
    total_logprob = 0.0
    segment_count = 0
    
    for seg in segments:
        text = seg.text.strip() if seg.text else ""
        if text:
            raw_text_parts.append(text)
            segment_data.append({
                "text": text,
                "start": seg.start,
                "end": seg.end,
                "avg_logprob": seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0,
                "no_speech_prob": seg.no_speech_prob if hasattr(seg, 'no_speech_prob') else 0.0,
            })
            if hasattr(seg, 'avg_logprob'):
                total_logprob += seg.avg_logprob
                segment_count += 1
    
    raw_text = " ".join(raw_text_parts).strip()
    
    # AGGRESSIVE repetition suppression
    text = _aggressive_squash_repetition(raw_text)
    
    if not text:
        raise RuntimeError("faster-whisper produced empty text after cleaning.")

    detected = getattr(info, "language", "unknown")
    prob = getattr(info, "language_probability", 0.0)
    avg_logprob = total_logprob / segment_count if segment_count > 0 else 0.0
    
    # Detect music-heavy content
    is_music_heavy, suspicious_indices = _detect_music_heavy_audio(segment_data)
    if is_music_heavy and quality_gate:
        log.warning("Music-heavy content detected - applying stricter quality filters")
    
    # QUALITY GATE
    if quality_gate:
        passed, reason = _check_quality_gate(text, detected, language, prob, avg_logprob)
        if not passed:
            log.error("QUALITY GATE FAILED: %s", reason)
            log.error("Raw transcript: %s", raw_text[:200])
            raise RuntimeError(f"Transcript quality too low: {reason}")
        log.info("Quality gate passed: %s", reason)
    
    rep_score = _calculate_repetition_score(text)
    log.info("Transcription complete. lang=%s prob=%.3f chars=%d rep_score=%.2f avg_logprob=%.2f", 
             detected, prob, len(text), rep_score, avg_logprob)
    
    return text


def _transcribe_with_timestamps_whisperx(
    audio_path: Path,
    *,
    language: Optional[str] = None,
    quality_gate: bool = True,
) -> dict:
    """
    WhisperX backend with optional alignment.
    Returns payload compatible with transcribe_with_timestamps().
    """
    _validate_audio(audio_path)

    try:
        import whisperx
        import torch
    except ImportError as exc:
        raise ImportError("whisperx is not installed. Run: pip install whisperx") from exc

    device = config.FASTER_WHISPER_DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = config.WHISPERX_COMPUTE_TYPE
    if device == "cpu" and compute_type in {"float16", "bfloat16"}:
        compute_type = "int8"

    log.info(
        "Transcribing '%s' with whisperx model='%s' device='%s' compute_type='%s'...",
        audio_path.name,
        config.WHISPERX_MODEL,
        device,
        compute_type,
    )

    try:
        model = whisperx.load_model(
            config.WHISPERX_MODEL,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    except Exception as exc:
        msg = str(exc)
        # Common Windows dependency breakage:
        # - torchvision::nms operator mismatch
        # - transformers Pipeline import failure
        if (
            "torchvision::nms" in msg
            or "Could not import module 'Pipeline'" in msg
            or "No module named 'lightning'" in msg
        ):
            log.warning(
                "WhisperX dependency issue detected (%s). "
                "Falling back to faster-whisper timestamp backend.",
                msg.splitlines()[0],
            )
            return _transcribe_with_timestamps_faster_whisper(
                audio_path,
                language=language,
                quality_gate=quality_gate,
            )
        raise
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=max(1, config.WHISPERX_BATCH_SIZE))

    segments = result.get("segments") or []
    detected_lang = result.get("language")

    if config.WHISPERX_DIARIZE_ENABLED and segments:
        if not config.HF_TOKEN:
            log.warning("WHISPERX_DIARIZE_ENABLED=true but HF_TOKEN is not set; using single-speaker fallback.")
        else:
            try:
                diarization_pipeline_cls = getattr(whisperx, "DiarizationPipeline", None)
                if diarization_pipeline_cls is None:
                    log.warning("WhisperX DiarizationPipeline is unavailable in this install; using single-speaker fallback.")
                else:
                    diarize_model = diarization_pipeline_cls(
                        use_auth_token=config.HF_TOKEN,
                        device=device,
                    )
                    diarize_segments = diarize_model(audio)
                    assigned = whisperx.assign_word_speakers(diarize_segments, result)
                    segments = assigned.get("segments") or segments
                    log.info("WhisperX diarization applied to %d segments.", len(segments))
            except Exception as exc:
                log.warning("WhisperX diarization failed: %s", exc)

    # Optional alignment pass (usually improves timestamp quality).
    if config.WHISPERX_ALIGN_ENABLED and segments:
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=(language or detected_lang or "en"),
                device=device,
            )
            aligned = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            segments = aligned.get("segments") or segments
        except Exception as exc:
            log.warning("WhisperX alignment skipped: %s", exc)

    out_segments: list[dict] = []
    for seg in segments:
        text = _aggressive_squash_repetition(str(seg.get("text", "")).strip())
        if not text:
            continue
        out = {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", seg.get("start", 0.0))),
            "text": text,
            "speaker_id": str(seg.get("speaker", "spk_00")),
        }
        # Some whisperx backends expose avg_logprob; keep it if available.
        if "avg_logprob" in seg:
            out["avg_logprob"] = float(seg.get("avg_logprob", 0.0))
        out_segments.append(out)

    out_segments = validate_timestamps(out_segments)
    out_segments = _split_overlong_segments(out_segments)
    is_music_heavy, suspicious_indices = _detect_music_heavy_audio(out_segments)
    for idx in suspicious_indices:
        if idx < len(out_segments):
            out_segments[idx]["is_suspicious"] = True
            out_segments[idx]["suspicion_reason"] = "music_or_noise"

    text = " ".join(s["text"] for s in out_segments if s.get("text")).strip()
    if not text:
        raise RuntimeError("WhisperX returned empty transcript.")

    avg_logprob = 0.0
    logprob_count = 0
    low_confidence_segments = 0
    for s in out_segments:
        if "avg_logprob" in s:
            lp = float(s["avg_logprob"])
            avg_logprob += lp
            logprob_count += 1
            if lp < MIN_AVG_LOGPROB:
                low_confidence_segments += 1
    avg_logprob = (avg_logprob / logprob_count) if logprob_count else 0.0

    if quality_gate:
        # WhisperX may not always provide language_probability; assume high if lang is stable.
        lang_prob = 1.0 if (detected_lang and (not language or detected_lang == language)) else 0.6
        passed, reason = _check_quality_gate(
            text,
            detected_lang or "unknown",
            language,
            lang_prob,
            avg_logprob,
        )
        if not passed:
            raise RuntimeError(f"Segment transcription quality too low: {reason}")

    return {
        "text": text,
        "language": detected_lang,
        "language_probability": None,
        "segments": out_segments,
        "avg_logprob": avg_logprob,
        "low_confidence_segments": low_confidence_segments,
        "music_heavy": is_music_heavy,
    }


def _transcribe_whisperx(
    audio_path: Path,
    *,
    language: Optional[str] = None,
    quality_gate: bool = True,
) -> str:
    payload = _transcribe_with_timestamps_whisperx(
        audio_path,
        language=language,
        quality_gate=quality_gate,
    )
    detected = payload.get("language") or "unknown"
    log.info(
        "WhisperX transcription complete. lang=%s chars=%d",
        detected,
        len(payload["text"]),
    )
    return payload["text"]


def transcribe(
    audio_path: str | Path,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    quality_gate: bool = True,
) -> str:
    audio_path = Path(audio_path).resolve()

    backend = config.TRANSCRIBE_BACKEND
    if backend == "whisperx":
        return _transcribe_whisperx(audio_path, language=language, quality_gate=quality_gate)
    if backend == "faster_whisper":
        return _transcribe_faster_whisper(audio_path, language=language, quality_gate=quality_gate)
    if backend == "nvidia":
        return _transcribe_nvidia(
            audio_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

    raise ValueError(f"Unknown TRANSCRIBE_BACKEND='{backend}'.")


def transcribe_with_timestamps(
    audio_path: str | Path,
    *,
    language: Optional[str] = None,
    quality_gate: bool = True,
) -> dict:
    """
    Returns segment-level timestamps with quality gating and validation.
    """
    audio_path = Path(audio_path).resolve()

    if config.TRANSCRIBE_BACKEND == "whisperx":
        return _transcribe_with_timestamps_whisperx(
            audio_path,
            language=language,
            quality_gate=quality_gate, 
        )

    if config.TRANSCRIBE_BACKEND != "faster_whisper":
        raise NotImplementedError("transcribe_with_timestamps requires whisperx or faster_whisper backend.")
    return _transcribe_with_timestamps_faster_whisper(
        audio_path,
        language=language,
        quality_gate=quality_gate,
    )


def _transcribe_with_timestamps_faster_whisper(
    audio_path: Path,
    *,
    language: Optional[str] = None,
    quality_gate: bool = True,
) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError("faster-whisper is not installed.") from exc

    device, compute_type = _resolve_device_and_compute_type()

    model = WhisperModel(
        config.FASTER_WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        vad_filter=config.FASTER_WHISPER_VAD_FILTER,
        vad_parameters={
            "threshold": 0.6,
            "min_speech_duration_ms": 300,
            "max_speech_duration_s": 30,
            "min_silence_duration_ms": 500,
        } if config.FASTER_WHISPER_VAD_FILTER else None,
        beam_size=config.FASTER_WHISPER_BEAM_SIZE,
        best_of=config.FASTER_WHISPER_BEST_OF,
        condition_on_previous_text=config.FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        temperature=0.0,
        no_speech_threshold=0.7,
        log_prob_threshold=min(-1.0, MIN_AVG_LOGPROB),
        compression_ratio_threshold=2.2,
        repetition_penalty=1.3,
    )
# Process segments with aggressive cleaning and quality checks
    out_segments = []
    total_logprob = 0.0
    segment_count = 0
    low_confidence_segments = []

    for seg in segments:
        text = _aggressive_squash_repetition(seg.text.strip()) if seg.text else ""
        if text:
            segment_info = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "speaker_id": "spk_00",
            }
            if hasattr(seg, "avg_logprob"):
                segment_info["avg_logprob"] = seg.avg_logprob
                total_logprob += seg.avg_logprob
                segment_count += 1
                if seg.avg_logprob < MIN_AVG_LOGPROB:
                    low_confidence_segments.append(
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "logprob": seg.avg_logprob,
                            "text": text[:50],
                        }
                    )
            if hasattr(seg, "no_speech_prob"):
                segment_info["no_speech_prob"] = seg.no_speech_prob
            out_segments.append(segment_info)

    out_segments = validate_timestamps(out_segments)
    out_segments = _split_overlong_segments(out_segments)
    is_music_heavy, suspicious_indices = _detect_music_heavy_audio(out_segments)
    for idx in suspicious_indices:
        if idx < len(out_segments):
            out_segments[idx]["is_suspicious"] = True
            out_segments[idx]["suspicion_reason"] = "music_or_noise"

    out_segments = _refine_low_confidence_segments(
        model,
        audio_path,
        out_segments,
        language=language,
    )
    out_segments = _refine_large_gaps(
        model,
        audio_path,
        out_segments,
        language=language,
    )

    text = " ".join(s["text"] for s in out_segments if s["text"]).strip()

    if quality_gate:
        detected = getattr(info, "language", "unknown")
        prob = getattr(info, "language_probability", 0.0)
        avg_logprob = total_logprob / segment_count if segment_count > 0 else 0.0
        passed, reason = _check_quality_gate(text, detected, language, prob, avg_logprob)
        if not passed:
            raise RuntimeError(f"Segment transcription quality too low: {reason}")
        if low_confidence_segments:
            log.warning("Found %d low-confidence segments", len(low_confidence_segments))
            for seg in low_confidence_segments[:3]:
                log.warning(
                    "  Low confidence: [%.2f-%.2f] logprob=%.2f",
                    seg["start"],
                    seg["end"],
                    seg["logprob"],
                )

    return {
        "text": text,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": out_segments,
        "avg_logprob": total_logprob / segment_count if segment_count > 0 else 0.0,
        "low_confidence_segments": len(low_confidence_segments),
        "music_heavy": is_music_heavy,
    }


def validate_segment_quality(segment: dict, expected_lang: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate a single ASR segment for quality.
    
    Returns:
        Tuple of (passed, reason)
    """
    text = segment.get("text", "")
    
    # Use the quality validation module
    result = validate_asr_segment(text, expected_lang)
    
    # Additional checks for timestamped segments
    duration = segment.get("end", 0) - segment.get("start", 0)
    if duration < 0.5:
        return False, f"Segment too short: {duration:.2f}s"
    
    if duration > 15.0:
        return False, f"Segment too long: {duration:.2f}s"
    
    # Check for suspicious flag
    if segment.get("is_suspicious"):
        return False, f"Suspicious segment flagged: {segment.get('suspicion_reason', 'unknown')}"
    
    return result.passed, result.reason
