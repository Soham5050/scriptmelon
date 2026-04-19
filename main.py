"""
main.py
-------
Adaptive video dubbing pipeline tuned for low/high VRAM GPUs.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import re
import subprocess
import shutil
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional, List, Tuple

import config
from audio_utils import convert_audio_format, extract_audio, get_video_duration
from merge import merge_audio
from separation import separate_dialogue_and_bed
from transcribe import transcribe, transcribe_with_timestamps, unload_asr_resources
from translate import translate, translate_texts_batch, unload_translation_model
try:
    from tts_optimized import synthesize, synthesize_timed_segments, unload_tts_model, _concat_wavs as _tts_concat_wavs
except ImportError:
    from tts import synthesize, synthesize_timed_segments, unload_tts_model, _concat_wavs as _tts_concat_wavs

log = config.get_logger("main")


# Video duration thresholds
SHORT_VIDEO_THRESHOLD = 60  # 1 minute - process normally
MEDIUM_VIDEO_THRESHOLD = 300  # 5 minutes - use smaller batches
LONG_VIDEO_THRESHOLD = 600  # 10 minutes - use CPU fallback for TTS


def _cleanup_memory(force: bool = False):
    """Aggressive memory cleanup."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    if force:
        for _ in range(3):
            gc.collect()


def _get_video_category(duration_sec: float) -> str:
    """Categorize video by duration for processing strategy."""
    if duration_sec <= SHORT_VIDEO_THRESHOLD:
        return "short"  # < 1 min
    elif duration_sec <= MEDIUM_VIDEO_THRESHOLD:
        return "medium"  # 1-5 min
    elif duration_sec <= LONG_VIDEO_THRESHOLD:
        return "long"  # 5-10 min
    else:
        return "very_long"  # > 10 min


def _chunk_video_segments(segments: List[dict], max_chunk_duration: float = 300.0) -> List[List[dict]]:
    """
    Split segments into time-based chunks for memory-efficient processing.
    Each chunk is processed independently to avoid OOM.
    """
    if not segments:
        return []
    
    chunks = []
    current_chunk = []
    chunk_start = segments[0].get("start", 0)
    
    for seg in segments:
        seg_start = seg.get("start", 0)
        
        # Start new chunk if duration exceeds limit
        if seg_start - chunk_start > max_chunk_duration and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [seg]
            chunk_start = seg_start
        else:
            current_chunk.append(seg)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def _normalize_chunk_segments(chunk_segments: List[dict]) -> Tuple[List[dict], float, float]:
    """
    Normalize chunk segment timestamps to local (chunk-relative) time.
    Returns (normalized_segments, chunk_start_abs, chunk_end_abs).
    """
    if not chunk_segments:
        return [], 0.0, 0.0

    chunk_start_abs = float(chunk_segments[0].get("start", 0.0))
    chunk_end_abs = float(chunk_segments[-1].get("end", chunk_start_abs))
    normalized: List[dict] = []

    for seg in chunk_segments:
        seg_start_abs = float(seg.get("start", chunk_start_abs))
        seg_end_abs = float(seg.get("end", seg_start_abs))
        seg_new = dict(seg)
        seg_new["start"] = max(0.0, seg_start_abs - chunk_start_abs)
        seg_new["end"] = max(seg_new["start"], seg_end_abs - chunk_start_abs)
        normalized.append(seg_new)

    return normalized, chunk_start_abs, chunk_end_abs


def _make_silence_wav(path: Path, duration_sec: float) -> Path:
    """Create a mono 24kHz PCM silence wav of the requested duration."""
    if duration_sec <= 0.0:
        return path
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{duration_sec:.6f}",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return path


def _checkpoint_path(work_dir: Path, step: str) -> Path:
    return work_dir / f"checkpoint_{step}.json"


def _save_checkpoint(work_dir: Path, step: str, data: dict, run_key: Optional[str] = None) -> None:
    path = _checkpoint_path(work_dir, step)
    payload = {"step": step, "saved_at": time.time(), "run_key": run_key, "data": data}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("      Checkpoint saved: %s", path.name)


def _load_checkpoint(work_dir: Path, step: str, run_key: Optional[str] = None) -> Optional[dict]:
    path = _checkpoint_path(work_dir, step)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_run_key = payload.get("run_key")
        if run_key and saved_run_key and saved_run_key != run_key:
            log.info("      Ignoring checkpoint %s (run key mismatch).", path.name)
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            log.info("      Resuming from checkpoint: %s", path.name)
            return data
    except Exception as exc:
        log.warning("      Failed to load checkpoint '%s': %s", path.name, exc)
    return None


def _estimate_timings_from_text(full_text: str, total_duration: float) -> List[dict]:
    """
    Build timing estimates from sentence word ratios when ASR timestamps are unavailable.
    This keeps timed TTS path active with lower confidence labels.
    """
    text = re.sub(r"\s+", " ", (full_text or "").strip())
    if not text or total_duration <= 0:
        return []

    sentence_splits = re.split(r"(?<=[.!?।॥])\s+", text)
    sentences = [s.strip() for s in sentence_splits if s and s.strip()]
    if not sentences:
        return []

    word_counts = [max(1, len(s.split())) for s in sentences]
    total_words = max(1, sum(word_counts))
    cursor = 0.0
    out: List[dict] = []
    for sent, wc in zip(sentences, word_counts):
        dur = max(0.05, (wc / total_words) * total_duration)
        start = cursor
        end = min(total_duration, cursor + dur)
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": sent,
                "estimated": True,
            }
        )
        cursor = end

    if out:
        out[-1]["end"] = round(total_duration, 3)
    return out


def _concat_audio_chunks(parts: List[Path], out_wav: Path, crossfade_ms: int = 100) -> None:
    """Concatenate chunk wavs using TTS helper when available, else ffmpeg concat fallback."""
    try:
        _tts_concat_wavs(parts, out_wav, crossfade_ms=crossfade_ms)
        return
    except Exception as exc:
        log.warning("TTS chunk concat helper failed, using ffmpeg concat fallback: %s", exc)

    list_file = out_wav.parent / "concat_list.txt"
    list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _format_timecode(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss for review display."""
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"


def _review_line_from_segment(seg: dict) -> str:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    text = re.sub(r"\s+", " ", str(seg.get("text", "") or "")).strip()
    return f"{_format_timecode(start)} - {_format_timecode(end)}  {text}"


def _write_review_timeline(review_file: Path, segments: List[dict]) -> None:
    lines = [
        "# Edit only the transcript text after the time range.",
        "# Keep one segment per line. Timeline format:",
        "# 0:00.00 - 0:03.20  text",
        "",
    ]
    lines.extend(_review_line_from_segment(seg) for seg in segments)
    review_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_review_timeline(review_file: Path, original_segments: List[dict]) -> List[dict]:
    """
    Apply user edits from review file.
    Keeps original timestamps and segment count for sync stability.
    """
    if not review_file.exists():
        return original_segments

    edited_lines: List[str] = []
    for raw in review_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Prefer parsing timeline format, fallback to raw line as text.
        m = re.match(r"^\s*[\d:.]+\s*-\s*[\d:.]+\s+(.*)$", line)
        text = (m.group(1) if m else line).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            text = text[1:-1].strip()
        edited_lines.append(text)

    if not edited_lines:
        return original_segments

    out: List[dict] = []
    for idx, seg in enumerate(original_segments):
        seg_new = dict(seg)
        if idx < len(edited_lines) and edited_lines[idx]:
            seg_new["text"] = edited_lines[idx]
        out.append(seg_new)

    if len(edited_lines) != len(original_segments):
        log.warning(
            "Review lines (%d) != ASR segments (%d). Applied edits by order and kept original timings.",
            len(edited_lines),
            len(original_segments),
        )
    return out


def _call_synthesize_timed_segments(
    segments: List[dict],
    out_dir: Path,
    *,
    target_lang: str,
    tts_backend: Optional[str],
    ref_audio: Optional[str | Path],
    ref_text: Optional[str],
    output_stem: Optional[str] = None,
    no_duration_match: bool,
    use_cpu_for_tts: bool,
) -> Path:
    """Call timed synthesis while remaining compatible with older TTS signatures."""
    def _is_cuda_failure(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            ("out of memory" in msg and ("cuda" in msg or "gpu" in msg))
            or "cuda error" in msg
            or "cublas_status_execution_failed" in msg
            or "cudnn_status" in msg
        )

    kwargs = {
        "segments": segments,
        "out_dir": out_dir,
        "target_lang": target_lang,
        "tts_backend": tts_backend,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "no_duration_match": no_duration_match,
    }
    if output_stem is not None:
        kwargs["output_stem"] = output_stem
    supports_cpu_retry = "use_cpu" in inspect.signature(synthesize_timed_segments).parameters
    if supports_cpu_retry:
        kwargs["use_cpu"] = use_cpu_for_tts
    try:
        return synthesize_timed_segments(**kwargs)
    except Exception as exc:
        if supports_cpu_retry and not use_cpu_for_tts and _is_cuda_failure(exc):
            log.warning("CUDA failure during timed TTS. Retrying on CPU fallback...")
            unload_tts_model()
            _cleanup_memory(force=True)
            kwargs["use_cpu"] = True
            return synthesize_timed_segments(**kwargs)
        raise


def _existing_path(value: object) -> Optional[Path]:
    """Return an existing Path from a checkpoint value, else None."""
    if value in (None, ""):
        return None
    try:
        path = Path(str(value))
    except Exception:
        return None
    return path if path.exists() else None


def run_pipeline(args):
    """Execute pipeline with automatic optimization based on video length + GPU tier."""
    config.validate()
    config.setup_gpu_memory_limit()
    
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        raise SystemExit(1)
    
    # Get video duration and determine processing strategy
    try:
        vid_dur = get_video_duration(input_path)
        category = _get_video_category(vid_dur)
        
        log.info("=" * 64)
        log.info("VIDEO ANALYSIS")
        log.info("=" * 64)
        log.info("Duration: %.1f seconds (%.1f minutes)", vid_dur, vid_dur / 60)
        log.info("Category: %s", category.upper())
        
        profile = config.get_runtime_profile()
        log.info(
            "GPU Profile: tier=%s vram=%.1fGB cuda=%s",
            profile.get("tier", "unknown"),
            float(profile.get("vram_gb", 0.0)),
            profile.get("cuda_available", False),
        )

        prefer_cpu_long = bool(profile.get("prefer_cpu_for_long_video", False))
        prefer_cpu_very_long = bool(profile.get("prefer_cpu_for_very_long_video", True))
        if not bool(profile.get("cuda_available", False)):
            prefer_cpu_long = True
            prefer_cpu_very_long = True

        # Set processing strategy based on video length + GPU tier
        if category == "short":
            log.info("Strategy: Standard processing")
            use_cpu_for_tts = False
            segment_chunk_duration = float(profile.get("segment_chunk_seconds_short", 60))
        elif category == "medium":
            log.info("Strategy: Chunked GPU processing")
            use_cpu_for_tts = False
            segment_chunk_duration = float(profile.get("segment_chunk_seconds_medium", 120))
        elif category == "long":
            log.info("Strategy: Aggressive cleanup mode")
            use_cpu_for_tts = prefer_cpu_long
            segment_chunk_duration = float(profile.get("segment_chunk_seconds_long", 180))
        else:  # very_long
            log.info("Strategy: Long-form memory-safe mode")
            use_cpu_for_tts = prefer_cpu_very_long
            segment_chunk_duration = float(profile.get("segment_chunk_seconds_very_long", 300))

        if args.force_cpu:
            use_cpu_for_tts = True

        if use_cpu_for_tts:
            log.info("TTS will use CPU (memory-safe mode)")
        
        log.info("=" * 64)
        
    except Exception as e:
        log.warning("Could not determine video duration: %s", e)
        profile = config.get_runtime_profile()
        use_cpu_for_tts = not bool(profile.get("cuda_available", False))
        segment_chunk_duration = float(profile.get("segment_chunk_seconds_long", 180))
    
    if args.keep_temp:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem).strip("._-") or "video"
        work_dir = output_path.parent / f"temp_dubbing_{safe_stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        _run_chunked(input_path, output_path, work_dir, args, use_cpu_for_tts, segment_chunk_duration)
    else:
        with tempfile.TemporaryDirectory(prefix="ai_dub_") as tmp:
            _run_chunked(input_path, output_path, Path(tmp), args, use_cpu_for_tts, segment_chunk_duration)


def _run_chunked(input_path, output_path, work_dir, args, use_cpu_for_tts: bool, segment_chunk_duration: float):
    """Run pipeline with chunked processing for memory efficiency."""
    separator = "=" * 64
    start_time = time.time()
    input_stat = input_path.stat()
    resume_key = "|".join(
        [
            str(input_path.resolve()),
            str(int(input_stat.st_size)),
            str(int(input_stat.st_mtime)),
            str(args.src_lang),
            str(args.target_lang),
            str(args.backend or config.TRANSLATION_BACKEND),
            str(args.tts_backend or config.TTS_BACKEND),
            str(bool(getattr(args, "preserve_bgm", False))),
            str(bool(getattr(args, "separate_audio", False))),
        ]
    )
    run_report: dict = {
        "input": str(input_path),
        "output": str(output_path),
        "src_lang": args.src_lang,
        "target_lang": args.target_lang,
        "translation_backend": args.backend or config.TRANSLATION_BACKEND,
        "tts_backend": args.tts_backend or config.TTS_BACKEND,
        "duration_match_enabled": not args.no_duration_match,
        "preserve_bgm": bool(getattr(args, "preserve_bgm", False)),
        "resume_enabled": bool(getattr(args, "resume", False)),
        "timestamps": {},
    }
    
    log.info(separator)
    log.info("AI VIDEO DUBBING PIPELINE - Adaptive")
    log.info(separator)
    log.info("Input   : %s", input_path)
    log.info("Output  : %s", output_path)
    log.info("Language: %s -> %s", args.src_lang, args.target_lang)
    log.info("Backend : %s", args.backend or config.TRANSLATION_BACKEND)
    log.info("TTS     : %s (CPU=%s)", args.tts_backend or config.TTS_BACKEND, use_cpu_for_tts)
    log.info("Duration Match: %s", "disabled" if args.no_duration_match else "enabled")
    log.info(separator)
    
    try:
        # =====================================================================
        # Step 1: Prepare Audio
        # =====================================================================
        log.info("[1/5] Preparing audio...")
        step_start = time.time()
        wav_path = work_dir / "extracted_audio.wav"
        original_mix_path: Optional[Path] = None
        speech_stem_path: Optional[Path] = None
        background_bed_path: Optional[Path] = None
        separation_requested = bool(getattr(args, "separate_audio", False) or getattr(args, "preserve_bgm", False))
        separation_used = False
        separation_meta: dict = {
            "requested": separation_requested,
            "used": False,
            "backend": getattr(config, "SEPARATION_BACKEND", "demucs"),
            "model": getattr(config, "SEPARATION_MODEL", "htdemucs"),
        }
        audio_ckpt = _load_checkpoint(work_dir, "audio_extract", run_key=resume_key) if args.resume else None
        if args.resume and audio_ckpt:
            ckpt_wav = _existing_path(audio_ckpt.get("wav_path"))
            ckpt_mix = _existing_path(audio_ckpt.get("mix_wav_path"))
            ckpt_speech = _existing_path(audio_ckpt.get("speech_stem_path"))
            ckpt_bed = _existing_path(audio_ckpt.get("background_bed_path"))
            ckpt_sep = audio_ckpt.get("separation")
            if isinstance(ckpt_sep, dict):
                separation_meta.update(ckpt_sep)
            can_reuse = bool(ckpt_wav)
            if separation_requested:
                can_reuse = can_reuse and bool(ckpt_speech and ckpt_bed)
            if can_reuse:
                wav_path = ckpt_wav or wav_path
                original_mix_path = ckpt_mix
                speech_stem_path = ckpt_speech
                background_bed_path = ckpt_bed
                separation_used = bool(speech_stem_path and background_bed_path)
                log.info("      Reusing prepared audio: %s", wav_path.name)

        if not wav_path.exists() or (separation_requested and not separation_used):
            if separation_requested:
                original_mix_path = extract_audio(
                    input_path,
                    work_dir,
                    sample_rate=48_000,
                    channels=2,
                    stem="original_mix",
                    apply_filter=False,
                )
                try:
                    sep_result = separate_dialogue_and_bed(
                        original_mix_path,
                        work_dir,
                        stem_prefix="separated",
                    )
                    speech_stem_path = sep_result.speech_path
                    background_bed_path = sep_result.bed_path
                    convert_audio_format(
                        speech_stem_path,
                        wav_path,
                        sample_rate=16_000,
                        channels=1,
                    )
                    separation_used = True
                    separation_meta.update(
                        {
                            "used": True,
                            "device": sep_result.device,
                            "seconds": round(sep_result.seconds, 3),
                            "speech_stem_path": str(speech_stem_path),
                            "background_bed_path": str(background_bed_path),
                        }
                    )
                    log.info("      Separation enabled: ASR will use '%s'", speech_stem_path.name)
                except Exception as exc:
                    log.warning("      Audio separation failed. Continuing with mixed audio: %s", exc)
                    wav_path = extract_audio(input_path, work_dir)
                    separation_meta.update({"used": False, "error": str(exc)})
                    speech_stem_path = None
                    background_bed_path = None
            else:
                wav_path = extract_audio(input_path, work_dir)
        audio_extract_sec = time.time() - step_start
        run_report["audio_extract_sec"] = round(audio_extract_sec, 3)
        run_report["audio_preparation"] = {
            "seconds": round(audio_extract_sec, 3),
            "wav_path": str(wav_path),
            "separation_requested": separation_requested,
            "separation_used": separation_used,
        }
        run_report["separation"] = separation_meta
        log.info("      Audio prepared in %.1fs", audio_extract_sec)
        _save_checkpoint(
            work_dir,
            "audio_extract",
            {
                "wav_path": str(wav_path),
                "mix_wav_path": str(original_mix_path) if original_mix_path else None,
                "speech_stem_path": str(speech_stem_path) if speech_stem_path else None,
                "background_bed_path": str(background_bed_path) if background_bed_path else None,
                "audio_extract_sec": round(audio_extract_sec, 3),
                "separation": separation_meta,
            },
            run_key=resume_key,
        )
        _cleanup_memory()
        
        # =====================================================================
        # Step 2: Transcribe with Quality Gate
        # =====================================================================
        log.info("[2/5] Transcribing with quality gate...")
        step_start = time.time()
        src_lang_hint = None if args.src_lang == "auto" else args.src_lang
        segments_payload = None
        transcript = ""
        transcribe_ckpt = _load_checkpoint(work_dir, "transcription", run_key=resume_key) if args.resume else None
        if transcribe_ckpt:
            transcript = str(transcribe_ckpt.get("transcript") or "").strip()
            sp = transcribe_ckpt.get("segments_payload")
            segments_payload = sp if isinstance(sp, dict) else None
            log.info(
                "      Loaded transcription checkpoint (chars=%d, has_timestamps=%s)",
                len(transcript),
                bool(segments_payload),
            )
        else:
            try:
                segments_payload = transcribe_with_timestamps(
                    wav_path, 
                    language=src_lang_hint,
                    quality_gate=not args.skip_quality_gate
                )
                transcript = (segments_payload.get("text") or "").strip()
            except RuntimeError as e:
                if "quality too low" in str(e).lower():
                    log.warning("=" * 64)
                    log.warning("TRANSCRIPTION QUALITY LOW - retrying with relaxed gate")
                    log.warning("=" * 64)
                    log.warning("Suggestions:")
                    log.warning("  1. Use cleaner audio (less music/background noise)")
                    log.warning("  2. Try --src_lang hi (explicit language)")
                    log.warning("  3. Pre-process audio to remove music")
                    try:
                        segments_payload = transcribe_with_timestamps(
                            wav_path,
                            language=src_lang_hint,
                            quality_gate=False,
                        )
                        transcript = (segments_payload.get("text") or "").strip()
                        log.warning("Proceeding with relaxed ASR quality gate.")
                    except Exception as relaxed_exc:
                        log.warning(
                            "Relaxed timestamped transcription failed: %s. Falling back to plain transcription.",
                            relaxed_exc,
                        )
                        segments_payload = None
                        transcript = transcribe(wav_path, language=src_lang_hint, quality_gate=False)
                        log.warning("Proceeding with plain transcription (no timestamps quality gate).")
                else:
                    log.warning("Timestamped transcription failed: %s", e)
                    transcript = transcribe(
                        wav_path,
                        language=src_lang_hint,
                        quality_gate=not args.skip_quality_gate,
                    )
        
        # Save transcription
        transcript_file = work_dir / "transcript.txt"
        transcript_file.write_text(transcript, encoding="utf-8")
        
        if segments_payload:
            (work_dir / "segments_asr.json").write_text(
                json.dumps(segments_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        transcribe_sec = time.time() - step_start
        log.info("      Transcription: %d chars in %.1fs", len(transcript), transcribe_sec)
        run_report["transcription"] = {
            "chars": len(transcript),
            "seconds": round(transcribe_sec, 3),
            "has_timestamps": bool(segments_payload),
            "segment_count": len(segments_payload.get("segments", []) if segments_payload else []),
            "detected_language": (segments_payload or {}).get("language"),
        }
        _save_checkpoint(
            work_dir,
            "transcription",
            {
                "transcript": transcript,
                "segments_payload": segments_payload,
                "transcription_meta": run_report["transcription"],
            },
            run_key=resume_key,
        )

        # Optional human review step before translation/TTS.
        if segments_payload and segments_payload.get("segments"):
            review_file = work_dir / "review_transcript.txt"
            asr_segments = list(segments_payload.get("segments") or [])
            _write_review_timeline(review_file, asr_segments)
            log.info("      Review timeline saved: %s", review_file)
            for preview in asr_segments[:5]:
                log.info("      %s", _review_line_from_segment(preview))
            if len(asr_segments) > 5:
                log.info("      ... (%d more segments)", len(asr_segments) - 5)

            if args.interactive:
                try:
                    if sys.platform == "win32":
                        subprocess.Popen(["notepad", str(review_file)])
                        log.info("      Opened review file in Notepad.")
                except Exception as exc:
                    log.warning("      Could not auto-open review file: %s", exc)

                if sys.stdin and sys.stdin.isatty():
                    input("\nEdit review_transcript.txt, then press Enter to continue...")
                else:
                    log.warning("--interactive enabled but stdin is non-interactive; continuing without pause.")

                reviewed_segments = _apply_review_timeline(review_file, asr_segments)
                segments_payload["segments"] = reviewed_segments
                transcript = " ".join(
                    str(seg.get("text", "")).strip()
                    for seg in reviewed_segments
                    if str(seg.get("text", "")).strip()
                ).strip()
                transcript_file.write_text(transcript, encoding="utf-8")
                (work_dir / "segments_asr_reviewed.json").write_text(
                    json.dumps({"segments": reviewed_segments}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                log.info("      Applied review edits. Updated transcript chars=%d", len(transcript))
        elif args.interactive:
            # Fallback path when timestamped segments are unavailable.
            review_file = work_dir / "review_transcript.txt"
            review_file.write_text(transcript + "\n", encoding="utf-8")
            log.info("      Review text saved: %s", review_file)
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["notepad", str(review_file)])
                    log.info("      Opened review file in Notepad.")
            except Exception as exc:
                log.warning("      Could not auto-open review file: %s", exc)
            if sys.stdin and sys.stdin.isatty():
                input("\nEdit review_transcript.txt, then press Enter to continue...")
                revised = review_file.read_text(encoding="utf-8").strip()
                if revised:
                    transcript = revised
                    transcript_file.write_text(transcript, encoding="utf-8")
                    log.info("      Applied review edits. Updated transcript chars=%d", len(transcript))
            else:
                log.warning("--interactive enabled but stdin is non-interactive; continuing without pause.")
        
        if "protty" in transcript.lower() or len(set(transcript.split())) < len(transcript.split()) * 0.3:
            log.warning("Possible repetition detected in transcript. Check quality.")
        
        _cleanup_memory()
        unload_asr_resources()
        _cleanup_memory()
        
        # =====================================================================
        # Step 3: Translate
        # =====================================================================
        log.info("[3/5] Translating [%s -> %s]...", args.src_lang, args.target_lang)
        step_start = time.time()
        translation_ckpt = _load_checkpoint(work_dir, "translation", run_key=resume_key) if args.resume else None
        if args.src_lang == "auto":
            detected_src = None
            if segments_payload:
                detected_src = segments_payload.get("language")
            src = detected_src or "en"
            log.info("      Auto source language resolved to: %s", src)
        else:
            src = args.src_lang
        
        if translation_ckpt:
            translation = str(translation_ckpt.get("translation") or "").strip()
            ts = translation_ckpt.get("translated_segments")
            translated_segments = ts if isinstance(ts, list) else []
            src = str(translation_ckpt.get("source_lang") or src)
            log.info(
                "      Loaded translation checkpoint (chars=%d, segments=%d)",
                len(translation),
                len(translated_segments),
            )
        else:
            segments = segments_payload.get("segments", []) if segments_payload else []
            skip_suspicious_segments = bool(getattr(config, "ASR_SKIP_SUSPICIOUS_SEGMENTS", False))
            if segments and skip_suspicious_segments:
                suspicious_count = sum(1 for seg in segments if seg.get("is_suspicious"))
                suspicious_ratio = suspicious_count / max(1, len(segments))
                max_skip_ratio = float(getattr(config, "ASR_SKIP_SUSPICIOUS_MAX_RATIO", 0.25))
                if suspicious_ratio > max_skip_ratio:
                    log.warning(
                        "Suspicious segment ratio too high (%.1f%% > %.1f%%). "
                        "Keeping suspicious segments to avoid silent gaps.",
                        suspicious_ratio * 100.0,
                        max_skip_ratio * 100.0,
                    )
                    skip_suspicious_segments = False

            if src == args.target_lang:
                log.info("      Source and target language match; skipping translation step.")
                translation = transcript
                translated_segments = []
                for seg in segments:
                    src_text = (seg.get("text") or "").strip()
                    if not src_text:
                        continue
                    if skip_suspicious_segments and seg.get("is_suspicious"):
                        continue
                    translated_segments.append(
                        {
                            "start": float(seg["start"]),
                            "end": float(seg["end"]),
                            "text": src_text,
                        }
                    )
            elif segments and not args.no_duration_match:
                # Batch translate segments for throughput.
                candidate_segments = []
                source_texts = []
                for seg in segments:
                    src_text = (seg.get("text") or "").strip()
                    if not src_text:
                        continue
                    if skip_suspicious_segments and seg.get("is_suspicious"):
                        log.debug("Skipping suspicious segment (likely music/noise)")
                        continue
                    candidate_segments.append(seg)
                    source_texts.append(src_text)

                translated_texts = translate_texts_batch(
                    source_texts,
                    tgt_lang=args.target_lang,
                    src_lang=src,
                    backend=args.backend or None,
                    use_glossary=True,
                )

                translated_segments = []
                for seg, tgt_text in zip(candidate_segments, translated_texts):
                    if not (tgt_text or "").strip():
                        continue
                    translated_segments.append({
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": tgt_text.strip(),
                    })

                if translated_segments:
                    translation = " ".join(s["text"] for s in translated_segments).strip()
                else:
                    log.warning(
                        "No translated timed segments after filtering; falling back to full-text translation."
                    )
                    translation = translate(
                        transcript,
                        tgt_lang=args.target_lang,
                        src_lang=src,
                        backend=args.backend or None,
                    )
            else:
                # Full text translation
                translation = translate(
                    transcript,
                    tgt_lang=args.target_lang,
                    src_lang=src,
                    backend=args.backend or None,
                )
                translated_segments = []

        if not translated_segments and not args.no_duration_match and translation.strip():
            try:
                total_dur = get_video_duration(input_path)
                estimated_segments = _estimate_timings_from_text(translation, total_dur)
                if estimated_segments:
                    translated_segments = estimated_segments
                    log.warning(
                        "No ASR timestamp segments available. Using %d estimated timing segments (lower sync confidence).",
                        len(translated_segments),
                    )
            except Exception as exc:
                log.warning("Could not estimate fallback timings: %s", exc)
        
        # Save translation
        translation_file = work_dir / "translation.txt"
        translation_file.write_text(translation, encoding="utf-8")
        
        if translated_segments:
            (work_dir / "segments_translated.json").write_text(
                json.dumps(translated_segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        
        log.info("      Translation: %d chars in %.1fs", len(translation), time.time() - step_start)
        run_report["translation"] = {
            "chars": len(translation),
            "seconds": round(time.time() - step_start, 3),
            "segment_count": len(translated_segments),
            "source_lang": src,
            "target_lang": args.target_lang,
        }
        _save_checkpoint(
            work_dir,
            "translation",
            {
                "translation": translation,
                "translated_segments": translated_segments,
                "source_lang": src,
                "target_lang": args.target_lang,
                "translation_meta": run_report["translation"],
            },
            run_key=resume_key,
        )
        unload_translation_model()
        _cleanup_memory()
        unload_asr_resources()
        _cleanup_memory()
        
        # =====================================================================
        # Step 4: TTS with Chunked Processing
        # =====================================================================
        log.info("[4/5] Synthesizing dubbed audio...")
        if args.no_duration_match:
            log.info("      (Duration matching DISABLED - natural speech speed)")
        if use_cpu_for_tts:
            log.info("      (Using CPU for TTS - memory-safe mode)")
        
        step_start = time.time()
        
        tts_backend = (args.tts_backend or config.TTS_BACKEND).lower()
        ref_audio = args.ref_audio or config.TTS_REF_AUDIO or None
        # Avoid implicit fallback ref_text to prevent prompt-text leakage into speech.
        ref_text = args.ref_text or None
        auto_ref_audio = speech_stem_path or wav_path
        
        if args.no_clone:
            ref_audio = auto_ref_audio
        elif not ref_audio or not Path(str(ref_audio)).exists():
            ref_audio = auto_ref_audio
        
        # Process TTS in chunks for long videos
        if translated_segments and len(translated_segments) > 20:
            # Chunk segments for memory efficiency
            segment_chunks = _chunk_video_segments(translated_segments, segment_chunk_duration)
            
            if len(segment_chunks) > 1:
                log.info("      Processing %d segments in %d chunks...", 
                        len(translated_segments), len(segment_chunks))
                unload_between_chunks = bool(getattr(config, "TTS_UNLOAD_MODEL_BETWEEN_CHUNKS", False))
                if unload_between_chunks:
                    log.info("      Chunk mode: unloading TTS model between chunks is enabled.")

                # Clean stale chunk artifacts from previous --keep_temp runs.
                for stale in work_dir.glob("dubbed_chunk_*.wav"):
                    try:
                        stale.unlink()
                    except Exception as exc:
                        log.debug("Could not remove stale chunk '%s': %s", stale, exc)
                
                chunk_wavs = []
                timeline_cursor = 0.0
                for i, chunk_segments in enumerate(segment_chunks):
                    log.info("      Processing chunk %d/%d (%d segments)...", 
                            i + 1, len(segment_chunks), len(chunk_segments))
                    
                    chunk_wav = work_dir / f"dubbed_chunk_{i:03d}.wav"
                    normalized_segments, chunk_start_abs, chunk_end_abs = _normalize_chunk_segments(chunk_segments)

                    # Preserve timeline gaps between chunks (including leading silence before first chunk).
                    gap = max(0.0, chunk_start_abs - timeline_cursor)
                    if gap > 0.03:
                        gap_wav = work_dir / f"dubbed_gap_{i:03d}.wav"
                        _make_silence_wav(gap_wav, gap)
                        chunk_wavs.append(gap_wav)
                    
                    _call_synthesize_timed_segments(
                        normalized_segments,
                        work_dir,
                        target_lang=args.target_lang,
                        tts_backend=tts_backend,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        output_stem=f"chunk_{i:03d}",
                        no_duration_match=args.no_duration_match,
                        use_cpu_for_tts=use_cpu_for_tts,
                    )
                    
                    # Move output to expected location
                    expected_output = work_dir / f"chunk_{i:03d}.wav"
                    if expected_output.exists():
                        # Use replace so reruns can overwrite existing files on Windows.
                        expected_output.replace(chunk_wav)
                        chunk_wavs.append(chunk_wav)
                        timeline_cursor = max(timeline_cursor, chunk_end_abs)
                    
                    # Cleanup between chunks
                    _cleanup_memory(force=True)
                    if unload_between_chunks:
                        unload_tts_model()
                
                # Concatenate all chunks
                dubbed_wav = work_dir / "dubbed_audio.wav"
                # Keep exact timing across chunk boundaries; chunk-level crossfade can shrink total duration.
                _concat_audio_chunks(chunk_wavs, dubbed_wav, crossfade_ms=0)
                
            else:
                # Single chunk - process normally
                dubbed_wav = _call_synthesize_timed_segments(
                    translated_segments,
                    work_dir,
                    target_lang=args.target_lang,
                    tts_backend=tts_backend,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    no_duration_match=args.no_duration_match,
                    use_cpu_for_tts=use_cpu_for_tts,
                )
        elif translated_segments:
            # Few segments - process normally
            dubbed_wav = _call_synthesize_timed_segments(
                translated_segments,
                work_dir,
                target_lang=args.target_lang,
                tts_backend=tts_backend,
                ref_audio=ref_audio,
                ref_text=ref_text,
                no_duration_match=args.no_duration_match,
                use_cpu_for_tts=use_cpu_for_tts,
            )
        else:
            # No segments - synthesize full text
            synth_kwargs = {
                "text": translation,
                "out_dir": work_dir,
                "target_lang": args.target_lang,
                "tts_backend": tts_backend,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
            }
            # Only pass use_cpu when supported by the selected TTS module.
            supports_cpu_retry = "use_cpu" in inspect.signature(synthesize).parameters
            if supports_cpu_retry:
                synth_kwargs["use_cpu"] = use_cpu_for_tts
            try:
                dubbed_wav = synthesize(**synth_kwargs)
            except Exception as exc:
                err = str(exc).lower()
                is_cuda_failure = (
                    ("out of memory" in err and ("cuda" in err or "gpu" in err))
                    or "cuda error" in err
                    or "cublas_status_execution_failed" in err
                    or "cudnn_status" in err
                )
                if supports_cpu_retry and not use_cpu_for_tts and is_cuda_failure:
                    log.warning("CUDA failure during full-text TTS. Retrying on CPU fallback...")
                    unload_tts_model()
                    _cleanup_memory(force=True)
                    synth_kwargs["use_cpu"] = True
                    dubbed_wav = synthesize(**synth_kwargs)
                else:
                    raise
        
        log.info("      Synthesis complete in %.1fs", time.time() - step_start)
        run_report["tts"] = {
            "seconds": round(time.time() - step_start, 3),
            "use_cpu": bool(use_cpu_for_tts),
            "segment_count": len(translated_segments),
            "chunk_seconds": float(segment_chunk_duration),
        }
        _cleanup_memory(force=True)
        unload_tts_model()
        
        # =====================================================================
        # Step 5: Merge
        # =====================================================================
        log.info("[5/5] Merging dubbed audio into video...")
        step_start = time.time()
        merged_path = work_dir / f"merged_{input_path.stem}.mp4"
        merge_audio(
            input_path,
            dubbed_wav,
            merged_path,
            background_audio_path=background_bed_path if separation_used else None,
            pad_audio=not args.no_pad,
            preserve_bgm=bool(args.preserve_bgm),
            original_audio_gain_db=float(args.bgm_gain_db),
            dubbed_audio_gain_db=float(args.dub_gain_db),
        )
        log.info("      Merge complete in %.1fs", time.time() - step_start)
        run_report["merge"] = {
            "seconds": round(time.time() - step_start, 3),
            "pad_audio": bool(not args.no_pad),
            "preserve_bgm": bool(getattr(args, "preserve_bgm", False)),
            "used_separated_bed": bool(separation_used and background_bed_path),
        }
        
        # Optional Step 6: LipSync
        if args.lipsync:
            log.info("[6/6] Applying LipSync...")
            from lipsync import apply_lipsync
            lipsync_path = work_dir / f"lipsync_{input_path.stem}.mp4"
            apply_lipsync(merged_path, dubbed_wav, lipsync_path)
            merged_path = lipsync_path
        
        shutil.copy2(merged_path, output_path)
        
        if args.keep_temp:
            art_dir = output_path.parent / "artifacts"
            art_dir.mkdir(exist_ok=True)
            for filename in (
                "transcript.txt",
                "translation.txt",
                "review_transcript.txt",
                "segments_asr.json",
                "segments_translated.json",
                "pipeline_report.json",
            ):
                src_file = work_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, art_dir / filename)
            log.info("Intermediate files kept at: %s", work_dir)
        
        total_time = time.time() - start_time
        run_report["total_seconds"] = round(total_time, 3)
        run_report["status"] = "ok"
        (work_dir / "pipeline_report.json").write_text(
            json.dumps(run_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(separator)
        log.info("PIPELINE COMPLETE")
        log.info("Output: %s", output_path)
        log.info("Total time: %.1f minutes", total_time / 60)
        log.info(separator)
        
    except KeyboardInterrupt:
        log.info("Interrupted.")
        run_report["status"] = "interrupted"
        run_report["total_seconds"] = round(time.time() - start_time, 3)
        try:
            (work_dir / "pipeline_report.json").write_text(
                json.dumps(run_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        unload_translation_model()
        unload_asr_resources()
        unload_tts_model()
        raise SystemExit(0)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        run_report["status"] = "failed"
        run_report["error"] = str(exc)
        run_report["total_seconds"] = round(time.time() - start_time, 3)
        try:
            (work_dir / "pipeline_report.json").write_text(
                json.dumps(run_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        unload_translation_model()
        unload_asr_resources()
        unload_tts_model()
        raise SystemExit(1)


def build_parser():
    p = argparse.ArgumentParser(
        description="AI Video Dubbing Pipeline - Adaptive GPU tiers (6GB+)",
        epilog=textwrap.dedent("""\
            Examples:
              # Short video (15-30s) - fast GPU processing
              python main.py --input short.mp4 --target_lang hi
              
              # Medium video (1-5 min) - GPU with cleanup
              python main.py --input medium.mp4 --target_lang mr
              
              # Long video (1 hour) - automatic CPU fallback
              python main.py --input long.mp4 --target_lang hi --force_cpu
              
              # Disable duration matching for natural speech
              python main.py --input video.mp4 --target_lang hi --no_duration_match

              # Human review step before translation/TTS
              python main.py --input video.mp4 --target_lang en --interactive

              # Preserve background music bed and mix dubbed voice
              python main.py --input video.mp4 --target_lang en --preserve_bgm

              # Run speech/background separation before ASR for music-heavy videos
              python main.py --input video.mp4 --target_lang en --separate_audio
        """),
    )
    
    p.add_argument("--input", "-i", required=True, help="Source video file")
    p.add_argument("--target_lang", "-t", required=True, help="Target language code")
    p.add_argument("--output", "-o", default=None, help="Output path")
    p.add_argument("--src_lang", "-s", default="en", help="Source language code")
    p.add_argument("--backend", "-b", default=None, choices=["indictrans2", "marian", "nllb", "google"])
    p.add_argument("--tts_backend", default=None, choices=["qwen3"], help="TTS backend")
    
    p.add_argument("--no_duration_match", action="store_true", 
                   help="Disable time-stretching to ASR timestamps")
    p.add_argument("--skip_quality_gate", action="store_true",
                   help="Skip ASR quality gate (not recommended)")
    p.add_argument("--force_cpu", action="store_true",
                   help="Force CPU for TTS (slower but memory-safe)")
    p.add_argument("--interactive", action="store_true",
                   help="Pause after transcription for manual timeline review/edit")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing checkpoints in temp working directory")
    
    p.add_argument("--ref_audio", help="Reference voice WAV")
    p.add_argument("--ref_text", help="Transcript of reference audio")
    p.add_argument("--no_clone", action="store_true", help="Disable voice cloning")
    p.add_argument("--no_pad", action="store_true", help="Do not pad audio")
    p.add_argument("--preserve_bgm", action="store_true",
                   help="Preserve original music/ambience bed and mix dubbed speech on top")
    p.add_argument("--separate_audio", action="store_true",
                   help="Run speech/background separation before ASR; implied by --preserve_bgm in practice")
    p.add_argument("--bgm_gain_db", type=float, default=-8.0,
                   help="Gain in dB for original audio bed when --preserve_bgm is enabled")
    p.add_argument("--dub_gain_db", type=float, default=3.0,
                   help="Gain in dB for dubbed voice when --preserve_bgm is enabled")
    p.add_argument("--lipsync", action="store_true", help="Apply lip sync")
    p.add_argument("--keep_temp", action="store_true", help="Keep intermediate files")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")
    
    return p


if __name__ == "__main__":
    import logging
    
    parser = build_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.output is None:
        stem = Path(args.input).stem
        out_dir = Path(config.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{stem}_{args.target_lang}.mp4")
    
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        log.info("Interrupted.")
        unload_translation_model()
        unload_asr_resources()
        unload_tts_model()
        raise SystemExit(0)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        unload_translation_model()
        unload_asr_resources()
        unload_tts_model()
        raise SystemExit(1)
