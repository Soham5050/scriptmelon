"""
tts.py
------
Memory-adaptive TTS for mixed GPU tiers (6GB to high-VRAM systems).
"""

from __future__ import annotations

import base64
import gc
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import config

log = config.get_logger(__name__)

# Qwen3 model singleton with lazy loading
_QWEN3_MODEL = None
_QWEN3_MODEL_CPU = None  # CPU fallback model

# Memory heuristics (adaptive per GPU tier)
GPU_MEMORY_WARNING_RATIO = 0.82
GPU_MEMORY_CRITICAL_RATIO = 0.92
MIN_FREE_VRAM_GB_BEFORE_CPU_FALLBACK = float(os.environ.get("TTS_MIN_FREE_VRAM_GB", "1.0"))
MAX_TTS_CHUNK_CHARS = 80
MIN_TTS_CHUNK_CHARS = 10
CPU_TTS_CHUNK_CHARS = max(40, int(os.environ.get("QWEN3_CPU_MAX_CHARS_PER_CHUNK", "120")))
QWEN3_MAX_NEW_TOKENS = max(128, int(os.environ.get("QWEN3_LOCAL_MAX_NEW_TOKENS", "512")))
QWEN3_USE_REF_TEXT = os.environ.get("QWEN3_USE_REF_TEXT", "false").lower() in {"1", "true", "yes", "on"}
QWEN3_RETRY_SILENT_SEGMENTS = os.environ.get("QWEN3_RETRY_SILENT_SEGMENTS", "true").lower() in {"1", "true", "yes", "on"}
QWEN3_MIN_SEGMENT_RMS = max(1.0, float(os.environ.get("QWEN3_MIN_SEGMENT_RMS", "80")))


@dataclass
class VRAMStatus:
    """GPU memory status tracker."""
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    free_gb: float
    warning_gb: float
    critical_gb: float
    
    @property
    def is_critical(self) -> bool:
        return self.allocated_gb > self.critical_gb
    
    @property
    def should_cleanup(self) -> bool:
        return self.allocated_gb > self.warning_gb


def _memory_thresholds_gb(total_gb: float) -> tuple[float, float]:
    """Compute warning/critical thresholds from effective memory budget."""
    profile = config.get_runtime_profile()
    fraction = float(profile.get("memory_fraction", 0.8))
    budget_gb = max(1.0, total_gb * min(max(fraction, 0.5), 0.95))
    warning = max(0.8, budget_gb * GPU_MEMORY_WARNING_RATIO)
    critical = max(1.0, budget_gb * GPU_MEMORY_CRITICAL_RATIO)
    return warning, critical


def _get_vram_status() -> Optional[VRAMStatus]:
    """Get current GPU memory status."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        warning, critical = _memory_thresholds_gb(total)
        
        return VRAMStatus(
            allocated_gb=allocated,
            reserved_gb=reserved,
            total_gb=total,
            free_gb=total - reserved,
            warning_gb=warning,
            critical_gb=critical,
        )
    except Exception:
        return None


def _log_vram_status(prefix: str = ""):
    """Log current VRAM status."""
    status = _get_vram_status()
    if status:
        msg = f"{prefix}VRAM: {status.allocated_gb:.2f}GB allocated, {status.free_gb:.2f}GB free"
        if status.is_critical:
            log.warning(f"⚠️ CRITICAL: {msg}")
        elif status.should_cleanup:
            log.info(f"⚡ {msg}")
        else:
            log.debug(msg)
    return status


def _aggressive_gpu_cleanup(force: bool = False):
    """Aggressive GPU memory cleanup."""
    try:
        import torch
        if torch.cuda.is_available():
            if force:
                log.info("Forcing aggressive GPU cleanup...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            if force:
                torch.cuda.empty_cache()
                _log_vram_status("After cleanup: ")
    except Exception as e:
        log.debug(f"GPU cleanup error: {e}")


def _clean_text(text: str) -> str:
    """Clean text for TTS."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?,;:]){4,}", r"\1\1\1", text)
    text = re.sub(r"[<>\"\'\`\\]", "", text)
    return text


def _effective_generation_temperature(text: str, language: str) -> float:
    """
    Mild expressive boost for English lines based on punctuation cues.
    Keeps range conservative to avoid instability.
    """
    base = float(getattr(config, "QWEN3_LOCAL_TEMPERATURE", 0.7))
    if not bool(getattr(config, "QWEN3_EXPRESSIVE_ENGLISH_ENABLED", True)):
        return base
    if (language or "").strip().lower() != "english":
        return base

    boost_cap = max(0.0, float(getattr(config, "QWEN3_EXPRESSIVE_ENGLISH_BOOST", 0.08)))
    exclaim = text.count("!")
    question = text.count("?")
    comma = text.count(",")
    semicolon = text.count(";") + text.count(":")

    cue = 0.03 * exclaim + 0.025 * question + 0.004 * min(8, comma) + 0.003 * min(6, semicolon)
    boost = min(boost_cap, cue)
    return max(0.55, min(0.95, base + boost))


def _safe_ref_text(text: str) -> str:
    """Windows-safe reference text handling."""
    text = (text or "").strip()
    if not text:
        return "Reference speech sample."
    
    if sys.platform == "win32":
        try:
            import ctypes
            codepage = ctypes.windll.kernel32.GetConsoleOutputCP()
            text.encode(f"cp{codepage}", errors="strict")
            return text
        except (UnicodeEncodeError, LookupError):
            return text[:100] if len(text) > 100 else "Reference speech sample."
    
    return text


def _wav_duration(wav_path: Path) -> float:
    """Get WAV file duration."""
    try:
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _wav_peak_rms(wav_path: Path) -> float:
    """
    Peak RMS over a WAV file.
    Used as a lightweight guard against near-silent synthesis artifacts.
    """
    try:
        import audioop
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            peak_rms = 0.0
            while True:
                frames = wf.readframes(8192)
                if not frames:
                    break
                mono = audioop.tomono(frames, sample_width, 0.5, 0.5) if channels > 1 else frames
                peak_rms = max(peak_rms, float(audioop.rms(mono, sample_width)))
            return peak_rms
    except Exception:
        return 0.0


def _resolve_reference_audio(ref_audio_path: str | Path | None) -> Path:
    """Resolve reference audio path."""
    if ref_audio_path:
        p = Path(ref_audio_path).resolve()
        if p.exists():
            return p
    raise FileNotFoundError("Reference audio not found.")


def _prepare_reference_audio(ref_path: Path, out_dir: Path, max_seconds: float = 6.0) -> Path:
    """Prepare short reference clip for voice cloning."""
    duration = _wav_duration(ref_path)
    if 0 < duration <= max_seconds:
        return ref_path
    
    clipped = out_dir / "ref_clip.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(ref_path),
        "-ss", "0", "-t", str(max_seconds),
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
        str(clipped),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if clipped.exists() and clipped.stat().st_size > 0:
            log.info("Reference clip: %.1fs -> %.1fs", duration, _wav_duration(clipped))
            return clipped
    except Exception as exc:
        log.warning("Failed to clip reference: %s", exc)
    return ref_path


def _chunk_text_for_tts(text: str, max_chars: int = MAX_TTS_CHUNK_CHARS) -> List[str]:
    """Split text into small chunks for stable memory usage."""
    text = _clean_text(text)
    if len(text) <= max_chars:
        return [text] if len(text) >= MIN_TTS_CHUNK_CHARS else []
    
    # Split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: List[str] = []
    current = ""
    
    for sent in sentences:
        if len(sent) > max_chars:
            # Long sentence - split on clauses
            if current:
                chunks.append(current.strip())
                current = ""
            
            clauses = re.split(r'(?<=[,;])\s+', sent)
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                if len(clause) > max_chars:
                    # Very long clause - split on words
                    words = clause.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 <= max_chars:
                            temp += " " + word if temp else word
                        else:
                            if temp:
                                chunks.append(temp)
                            temp = word
                    if temp:
                        chunks.append(temp)
                else:
                    if len(current) + len(clause) + 1 <= max_chars:
                        current += " " + clause if current else clause
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = clause
        else:
            if len(current) + len(sent) + 1 <= max_chars:
                current += " " + sent if current else sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
    
    if current and len(current) >= MIN_TTS_CHUNK_CHARS:
        chunks.append(current.strip())
    
    return chunks


def _concat_wavs(parts: List[Path], out_wav: Path, crossfade_ms: int = 50) -> Path:
    """Concatenate WAVs with crossfade."""
    if len(parts) == 1:
        import shutil
        shutil.copy(parts[0], out_wav)
        return out_wav
    
    try:
        import numpy as np
        import wave
        
        audio_segments = []
        sample_rate = None
        n_channels = None
        sampwidth = None
        
        for p in parts:
            with wave.open(str(p), 'rb') as wf:
                if sample_rate is None:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                if n_channels == 2:
                    audio = audio.reshape(-1, 2)
                audio_segments.append(audio)
        
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        result = audio_segments[0]

        for i in range(1, len(audio_segments)):
            next_seg = audio_segments[i]
            if crossfade_samples <= 0:
                result = np.concatenate([result, next_seg])
                continue
            if len(result) > crossfade_samples and len(next_seg) > crossfade_samples:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                if n_channels == 2:
                    fade_out = fade_out.reshape(-1, 1)
                    fade_in = fade_in.reshape(-1, 1)
                result_tail = result[-crossfade_samples:] * fade_out
                next_head = next_seg[:crossfade_samples] * fade_in
                crossfaded = (result_tail + next_head).astype(np.int16)
                result = np.concatenate([result[:-crossfade_samples], crossfaded, next_seg[crossfade_samples:]])
            else:
                result = np.concatenate([result, next_seg])
        
        with wave.open(str(out_wav), 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(result.tobytes())
        
        return out_wav
        
    except ImportError:
        # Fallback to ffmpeg
        list_file = out_wav.parent / "concat_list.txt"
        list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_wav)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_wav


def _make_silence(dst: Path, duration: float) -> Path:
    """Create silence WAV."""
    if duration <= 0.0:
        raise ValueError("Silence duration must be > 0")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", f"{duration:.6f}", "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dst


def _ffmpeg_time_stretch(src: Path, dst: Path, speed: float) -> Path:
    """Time-stretch audio preserving pitch."""
    if speed <= 0 or abs(speed - 1.0) < 0.05:
        import shutil
        shutil.copy(src, dst)
        return dst
    
    factors = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    
    filter_expr = ",".join(f"atempo={f:.6f}" for f in factors)
    
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter:a", filter_expr,
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dst


def _ffmpeg_fit_duration(src: Path, dst: Path, target_duration: float) -> Path:
    """
    Fit audio to an exact duration using mild pad/trim after stretching.
    This avoids extreme playback-rate artifacts while keeping timeline sync.
    """
    if target_duration <= 0:
        import shutil
        shutil.copy(src, dst)
        return dst

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", f"apad=pad_dur={target_duration:.6f}",
        "-t", f"{target_duration:.6f}",
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dst


# =============================================================================
# Qwen3-TTS Optimized Implementation
# =============================================================================

_QWEN3_LANG_MAP = {
    # Use lower-case names accepted by Qwen3-TTS validator.
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
    # Marathi is not in supported explicit language list; use auto detection.
    "mr": "auto",
}
_QWEN3_UNSUPPORTED_WARNED: set[str] = set()


def _qwen3_language_name(lang: Optional[str]) -> str:
    if not lang:
        return "auto"
    code = lang.lower()
    mapped = _QWEN3_LANG_MAP.get(code)
    if mapped is not None:
        return mapped
    if code not in _QWEN3_UNSUPPORTED_WARNED:
        log.warning(
            "Qwen3 language hint '%s' is not explicitly supported. Falling back to 'auto'.",
            code,
        )
        _QWEN3_UNSUPPORTED_WARNED.add(code)
    return "auto"


def _resolve_qwen3_model_source(model_id_raw: str) -> tuple[str, bool]:
    """
    Resolve local-vs-hub model source robustly on Windows.
    Returns: (source, is_local_path)
    """
    model_id = (model_id_raw or "").strip().strip('"').strip("'")
    if not model_id:
        raise ValueError("QWEN3_LOCAL_MODEL_ID is empty.")

    project_root = Path(__file__).resolve().parent
    is_path_like = (
        model_id.startswith((".", "~"))
        or "\\" in model_id
        or "/" in model_id
        or bool(re.match(r"^[A-Za-z]:[\\/]", model_id))
    )

    if is_path_like:
        raw_path = Path(model_id).expanduser()
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append((Path.cwd() / raw_path).resolve())
            candidates.append((project_root / raw_path).resolve())

        seen = set()
        for candidate in candidates:
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists() and candidate.is_dir():
                return str(candidate), True

        available_local = sorted(
            p.name for p in project_root.iterdir()
            if p.is_dir() and p.name.startswith("Qwen3-TTS-")
        )
        # Safety fallback: if configured local dir is missing, prefer the installed 1.7B dir.
        preferred = project_root / "Qwen3-TTS-12Hz-1.7B-Base"
        if preferred.exists() and preferred.is_dir():
            log.warning(
                "Configured Qwen3 model path '%s' not found. Falling back to '%s'.",
                model_id,
                preferred,
            )
            return str(preferred), True

        available_hint = ", ".join(available_local) if available_local else "none"
        raise FileNotFoundError(
            f"Local Qwen3 model directory not found: '{model_id}'. "
            f"Available local model dirs: {available_hint}"
        )

    return model_id, False


def _configure_qwen_generation_defaults(wrapper_model) -> None:
    """
    Configure generation token defaults once to avoid repeated Transformers warnings
    about auto-setting pad_token_id to eos_token_id.
    """
    try:
        model = getattr(wrapper_model, "model", None)
        if model is None:
            return

        talker = getattr(model, "talker", None)
        model_cfg = getattr(model, "config", None)
        talker_cfg = getattr(model_cfg, "talker_config", None)
        if talker is None or talker_cfg is None:
            return

        pad_id = getattr(talker_cfg, "codec_pad_id", None)
        eos_id = getattr(talker_cfg, "codec_eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        if pad_id is None:
            return

        if getattr(talker.config, "pad_token_id", None) is None:
            talker.config.pad_token_id = int(pad_id)
        if getattr(talker.config, "eos_token_id", None) is None and eos_id is not None:
            talker.config.eos_token_id = int(eos_id)

        gen_cfg = getattr(talker, "generation_config", None)
        if gen_cfg is not None:
            if getattr(gen_cfg, "pad_token_id", None) is None:
                gen_cfg.pad_token_id = int(pad_id)
            if getattr(gen_cfg, "eos_token_id", None) is None and eos_id is not None:
                gen_cfg.eos_token_id = int(eos_id)
    except Exception as exc:
        log.debug("Could not configure Qwen generation defaults: %s", exc)


def _unload_qwen3_model():
    """Unload Qwen3 model to free GPU memory."""
    global _QWEN3_MODEL, _QWEN3_MODEL_CPU
    
    if _QWEN3_MODEL is not None:
        log.info("Unloading Qwen3 model from GPU...")
        try:
            import torch
            del _QWEN3_MODEL
            _QWEN3_MODEL = None
            _aggressive_gpu_cleanup(force=True)
        except Exception as e:
            log.warning(f"Error unloading model: {e}")
    
    if _QWEN3_MODEL_CPU is not None:
        log.info("Unloading Qwen3 CPU model...")
        try:
            del _QWEN3_MODEL_CPU
            _QWEN3_MODEL_CPU = None
            gc.collect()
        except Exception as e:
            log.warning(f"Error unloading CPU model: {e}")


def _get_qwen3_model(use_cpu: bool = False):
    """Get or load Qwen3 model with memory management."""
    global _QWEN3_MODEL, _QWEN3_MODEL_CPU
    
    # Check VRAM before loading
    vram = _get_vram_status()
    if vram and vram.is_critical and not use_cpu:
        log.warning("VRAM critical! Forcing CPU mode...")
        use_cpu = True
    
    if use_cpu:
        if _QWEN3_MODEL_CPU is not None:
            return _QWEN3_MODEL_CPU, "cpu"
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise ImportError("qwen-tts not installed. Run: pip install -U qwen-tts") from exc
        
        model_source, is_local = _resolve_qwen3_model_source(config.QWEN3_LOCAL_MODEL_ID)
        log.info("Loading Qwen3-TTS on CPU (slower but memory-safe)...")
        if is_local:
            log.info("  source=local path: %s", model_source)

        kwargs = {}
        if is_local:
            kwargs["local_files_only"] = True

        _QWEN3_MODEL_CPU = Qwen3TTSModel.from_pretrained(
            model_source,
            device_map="cpu",
            dtype=torch.float32,
            **kwargs,
        )
        _configure_qwen_generation_defaults(_QWEN3_MODEL_CPU)
        return _QWEN3_MODEL_CPU, "cpu"
    
    # GPU mode
    if _QWEN3_MODEL is not None:
        return _QWEN3_MODEL, "cuda"
    
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError("qwen-tts not installed. Run: pip install -U qwen-tts") from exc
    
    # Use sdpa attention instead of flash_attn (avoids build issues on Windows)
    attn_impl = "sdpa"  # Safer than flash_attention_2 on Windows
    
    dtype_name = (config.QWEN3_LOCAL_DTYPE or "float16").lower()
    dtype = getattr(torch, dtype_name, torch.float16)
    
    model_source, is_local = _resolve_qwen3_model_source(config.QWEN3_LOCAL_MODEL_ID)
    log.info(f"Loading Qwen3-TTS on GPU: {model_source}")
    log.info(f"  dtype={dtype_name}, attention={attn_impl}")

    kwargs = {}
    if is_local:
        kwargs["local_files_only"] = True

    _QWEN3_MODEL = Qwen3TTSModel.from_pretrained(
        model_source,
        device_map="cuda:0",
        dtype=dtype,
        attn_implementation=attn_impl,
        **kwargs,
    )
    _configure_qwen_generation_defaults(_QWEN3_MODEL)
    
    _log_vram_status("After model load: ")
    return _QWEN3_MODEL, "cuda"


def _qwen3_generate_single(
    model,
    text: str,
    language: str,
    ref_audio: Path,
    ref_text: str,
    device: str = "cuda"
) -> Tuple[bytes, int]:
    """Generate single audio chunk with memory monitoring."""
    try:
        import soundfile as sf
        import io
    except ImportError as exc:
        raise ImportError("soundfile required. Run: pip install soundfile") from exc
    
    _log_vram_status("Before generate: ")
    use_ref_text = bool(ref_text.strip()) and QWEN3_USE_REF_TEXT
    effective_ref_text = ref_text.strip() if use_ref_text else None
    use_xvec = not use_ref_text
    # Guardrail against runaway long generations: scale token budget by chunk text length.
    # 12Hz codec means token count maps roughly to audio duration; this keeps short text short.
    text_len = max(1, len((text or "").strip()))
    dynamic_max_new_tokens = max(96, min(QWEN3_MAX_NEW_TOKENS, int(text_len * 2.5)))
    
    try:
        temperature = _effective_generation_temperature(text, language)
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio),
            ref_text=effective_ref_text,
            x_vector_only_mode=use_xvec,
            non_streaming_mode=True,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=dynamic_max_new_tokens,
        )
        
        if not wavs:
            raise RuntimeError("Qwen3 returned empty waveform")
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], int(sample_rate), format="WAV")
        buffer.seek(0)
        
        return buffer.read(), int(sample_rate)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _log_vram_status("OOM detected: ")
            raise
        raise


def _generate_with_oom_fallback(
    model,
    device: str,
    *,
    text: str,
    language: str,
    ref_audio: Path,
    ref_text: str,
) -> tuple[bytes, int, object, str]:
    """
    Generate TTS audio and switch to CPU model on CUDA failures.
    Returns: (audio_bytes, sample_rate, model, device)
    """
    def _is_cuda_failure(exc: BaseException) -> bool:
        msg = str(exc).lower()
        if "out of memory" in msg:
            return True
        # Driver reset / GEMM failures / generic CUDA runtime failures
        if "cuda error" in msg:
            return True
        if "cublas_status_execution_failed" in msg:
            return True
        if "cudnn_status" in msg:
            return True
        if "device-side assert" in msg:
            return True
        return False

    def _is_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower()

    # Proactive switch: if VRAM is already near exhaustion, move to CPU before hard OOM.
    if device != "cpu":
        status = _get_vram_status()
        if status and (status.is_critical or status.free_gb < MIN_FREE_VRAM_GB_BEFORE_CPU_FALLBACK):
            log.warning(
                "Low VRAM before generation (alloc=%.2fGB free=%.2fGB). Switching to CPU fallback.",
                status.allocated_gb,
                status.free_gb,
            )
            _unload_qwen3_model()
            model, device = _get_qwen3_model(use_cpu=True)

    try:
        audio_bytes, sample_rate = _qwen3_generate_single(
            model=model,
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            device=device,
        )
        return audio_bytes, sample_rate, model, device
    except RuntimeError as e:
        if not _is_cuda_failure(e):
            raise

        if device != "cpu":
            # OOM can be retried once on GPU; non-OOM CUDA failures should jump straight to CPU.
            if _is_oom(e):
                log.warning("OOM during generation. Retrying once on GPU after cleanup...")
                _aggressive_gpu_cleanup(force=True)
                try:
                    audio_bytes, sample_rate = _qwen3_generate_single(
                        model=model,
                        text=text,
                        language=language,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        device=device,
                    )
                    return audio_bytes, sample_rate, model, device
                except RuntimeError as retry_err:
                    if not _is_cuda_failure(retry_err):
                        raise
                    log.warning("GPU retry failed with CUDA error (%s). Switching to CPU fallback.", retry_err)
            else:
                log.warning("CUDA execution failure during generation (%s). Switching to CPU fallback.", e)

            _unload_qwen3_model()
            model, device = _get_qwen3_model(use_cpu=True)
            audio_bytes, sample_rate = _qwen3_generate_single(
                model=model,
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                device=device,
            )
            return audio_bytes, sample_rate, model, device

        # Already on CPU and still OOM: bubble up.
        raise


def _synthesize_qwen3_chunked(
    text: str,
    out_dir: Path,
    *,
    output_stem: str,
    target_lang: Optional[str],
    ref_audio: Optional[str | Path],
    ref_text: Optional[str],
    use_cpu: bool = False,
) -> Path:
    """
    Synthesize with chunked processing and memory management.
    For 1hr videos, processes in batches with cleanup between.
    """
    text = _clean_text(text)
    if not text:
        raise ValueError("Empty text for synthesis")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare reference
    ref_max_seconds = max(1.0, float(getattr(config, "TTS_SPEAKER_REF_MAX_SECONDS", 6.0)))
    ref_path = _resolve_reference_audio(ref_audio or config.TTS_REF_AUDIO or None)
    ref_path = _prepare_reference_audio(ref_path, out_dir, max_seconds=ref_max_seconds)
    safe_ref_text = _clean_text(ref_text or "")
    language = _qwen3_language_name(target_lang)
    
    # Chunk text (adaptive by GPU tier)
    profile = config.get_runtime_profile()
    if use_cpu:
        max_chars = CPU_TTS_CHUNK_CHARS
    else:
        adaptive_chars = int(profile.get("tts_max_chars_per_chunk", config.QWEN3_LOCAL_MAX_CHARS_PER_CHUNK))
        max_chars = max(30, min(config.QWEN3_LOCAL_MAX_CHARS_PER_CHUNK, adaptive_chars))
    chunks = _chunk_text_for_tts(text, max_chars=max_chars)
    
    if not chunks:
        raise ValueError("No valid chunks after cleaning")
    
    log.info(f"Synthesizing {len(chunks)} chunks (max {max_chars} chars each)...")
    
    # Get model
    model, device = _get_qwen3_model(use_cpu=use_cpu)
    
    out_parts: List[Path] = []
    batch_size = int(profile.get("tts_batch_size", 2)) if device == "cuda" else 1
    batch_size = max(1, batch_size)
    
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        
        for idx in range(batch_start, batch_end):
            chunk = chunks[idx]
            wav_part = out_dir / f"{output_stem}_{idx:04d}.wav"

            audio_bytes, sample_rate, model, device = _generate_with_oom_fallback(
                model,
                device,
                text=chunk,
                language=language,
                ref_audio=ref_path,
                ref_text=safe_ref_text,
            )

            wav_part.write_bytes(audio_bytes)
            out_parts.append(wav_part)
            log.debug(f"Chunk {idx+1}/{len(chunks)}: {len(chunk)} chars -> {wav_part.name}")
        
        # Cleanup after each batch
        if device == "cuda":
            _aggressive_gpu_cleanup()
            _log_vram_status(f"Batch {batch_start//batch_size + 1} complete: ")
    
    # Concatenate
    out_wav = out_dir / f"{output_stem}.wav"
    _concat_wavs(out_parts, out_wav, crossfade_ms=config.TTS_CROSSFADE_MS)
    
    duration = _wav_duration(out_wav)
    log.info(f"Synthesis complete: {out_wav.name} ({duration:.1f}s, {len(chunks)} chunks)")
    
    return out_wav


def _synthesize_qwen3_timed_segments(
    segments: List[dict],
    out_dir: Path,
    *,
    output_stem: str,
    target_lang: Optional[str],
    ref_audio: Optional[str | Path],
    ref_text: Optional[str],
    no_duration_match: bool = False,
    use_cpu: bool = False,
) -> Path:
    """Timed segment synthesis for Qwen3 with memory management."""
    out_dir = Path(out_dir)
    seg_dir = out_dir / "tts_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    parts: List[Path] = []
    cursor = 0.0
    
    # Prepare reference once
    ref_max_seconds = max(1.0, float(getattr(config, "TTS_SPEAKER_REF_MAX_SECONDS", 6.0)))
    ref_path = _resolve_reference_audio(ref_audio or config.TTS_REF_AUDIO or None)
    ref_path = _prepare_reference_audio(ref_path, seg_dir, max_seconds=ref_max_seconds)
    safe_ref_text = _clean_text(ref_text or "")
    language = _qwen3_language_name(target_lang)
    
    # Get model
    model, device = _get_qwen3_model(use_cpu=use_cpu)
    profile = config.get_runtime_profile()
    cleanup_every = max(1, int(profile.get("tts_cleanup_every", 3)))
    
    for i, seg in enumerate(segments):
        text = _clean_text(str(seg.get("text", "")))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        target_dur = max(0.05, end - start)
        
        if not text:
            continue
        
        # Add silence for gaps
        if start > cursor:
            gap = start - cursor
            silence = seg_dir / f"{i:04d}_silence.wav"
            _make_silence(silence, gap)
            parts.append(silence)
            cursor = start
        
        # Generate audio for segment (split long segment text to reduce VRAM spikes)
        raw_wav = seg_dir / f"{i:04d}_raw.wav"
        segment_max_chars = CPU_TTS_CHUNK_CHARS if device == "cpu" else max(
            30, int(profile.get("tts_max_chars_per_chunk", config.QWEN3_LOCAL_MAX_CHARS_PER_CHUNK))
        )
        text_chunks = _chunk_text_for_tts(text, max_chars=segment_max_chars)
        if not text_chunks:
            text_chunks = [text]
        if len(text_chunks) > 1:
            log.debug(
                "Segment %d split into %d chunks for memory stability.",
                i,
                len(text_chunks),
            )

        segment_parts: List[Path] = []
        for j, chunk_text in enumerate(text_chunks):
            part_wav = seg_dir / f"{i:04d}_part_{j:02d}.wav"
            last_rms = 0.0
            last_dur = 0.0
            for attempt in range(2):
                audio_bytes, sample_rate, model, device = _generate_with_oom_fallback(
                    model,
                    device,
                    text=chunk_text,
                    language=language,
                    ref_audio=ref_path,
                    ref_text=safe_ref_text,
                )
                part_wav.write_bytes(audio_bytes)
                last_dur = _wav_duration(part_wav)
                last_rms = _wav_peak_rms(part_wav)

                if not QWEN3_RETRY_SILENT_SEGMENTS:
                    break

                min_expected_dur = max(0.05, min(0.15, target_dur * 0.15))
                too_short = last_dur <= min_expected_dur
                too_quiet = target_dur >= 0.20 and last_rms < QWEN3_MIN_SEGMENT_RMS
                if not (too_short or too_quiet):
                    break

                if attempt == 0:
                    log.warning(
                        "Segment %d chunk %d low-energy output (dur=%.2fs, rms=%.1f). Retrying on CPU.",
                        i,
                        j,
                        last_dur,
                        last_rms,
                    )
                    if device != "cpu":
                        _unload_qwen3_model()
                        model, device = _get_qwen3_model(use_cpu=True)
                else:
                    log.warning(
                        "Segment %d chunk %d remained low-energy after retry (dur=%.2fs, rms=%.1f). Keeping best effort output.",
                        i,
                        j,
                        last_dur,
                        last_rms,
                    )
            segment_parts.append(part_wav)

        if len(segment_parts) == 1:
            raw_wav = segment_parts[0]
        else:
            _concat_wavs(segment_parts, raw_wav, crossfade_ms=min(30, config.TTS_CROSSFADE_MS))
        
        produced_dur = _wav_duration(raw_wav)
        if produced_dur <= 0.0:
            continue
        
        # Apply duration matching if enabled
        duration_lock_enabled = bool(getattr(config, "TTS_DURATION_LOCK_ENABLED", True))
        duration_lock_tolerance = max(
            0.0, float(getattr(config, "TTS_DURATION_LOCK_TOLERANCE_SEC", 0.06))
        )
        if no_duration_match or not duration_lock_enabled:
            final_wav = raw_wav
        else:
            raw_speed = produced_dur / target_dur
            min_speed = float(getattr(config, "TTS_DURATION_MATCH_MIN_SPEED", 0.80))
            max_speed = float(getattr(config, "TTS_DURATION_MATCH_MAX_SPEED", 1.25))
            lang_code = (target_lang or "").strip().lower()
            if lang_code == "en":
                en_min = getattr(config, "TTS_DURATION_MATCH_EN_MIN_SPEED", None)
                en_max = getattr(config, "TTS_DURATION_MATCH_EN_MAX_SPEED", None)
                if isinstance(en_min, (int, float)):
                    min_speed = float(en_min)
                if isinstance(en_max, (int, float)):
                    max_speed = float(en_max)
            if min_speed > max_speed:
                min_speed, max_speed = max_speed, min_speed
            speed = max(0.5, min(2.0, max(min_speed, min(max_speed, raw_speed))))
            force_exact_fit = bool(getattr(config, "TTS_DURATION_FORCE_EXACT_FIT", True))
            candidate_wav = raw_wav

            if abs(produced_dur - target_dur) > duration_lock_tolerance:
                stretched_wav = seg_dir / f"{i:04d}_matched_tmp.wav"
                _ffmpeg_time_stretch(raw_wav, stretched_wav, speed)
                stretched_dur = _wav_duration(stretched_wav)
                candidate_wav = stretched_wav if stretched_dur > 0 else raw_wav

            if force_exact_fit:
                final_wav = seg_dir / f"{i:04d}_matched.wav"
                _ffmpeg_fit_duration(candidate_wav, final_wav, target_dur)
            else:
                if candidate_wav is raw_wav:
                    final_wav = raw_wav
                else:
                    candidate_dur = _wav_duration(candidate_wav)
                    if abs(candidate_dur - target_dur) > duration_lock_tolerance:
                        final_wav = seg_dir / f"{i:04d}_matched.wav"
                        _ffmpeg_fit_duration(candidate_wav, final_wav, target_dur)
                    else:
                        final_wav = candidate_wav
        
        parts.append(final_wav)
        if no_duration_match or not duration_lock_enabled:
            cursor = start + _wav_duration(final_wav)
        else:
            # Anchor to transcript timeline to prevent cumulative drift.
            cursor = end
        
        # Periodic cleanup
        if (i + 1) % cleanup_every == 0 and device == "cuda":
            _aggressive_gpu_cleanup()
    
    if not parts:
        raise RuntimeError("No TTS segments generated")
    
    out_wav = out_dir / f"{output_stem}.wav"
    _concat_wavs(parts, out_wav, crossfade_ms=config.TTS_CROSSFADE_MS)
    
    return out_wav


# =============================================================================
# Public API
# =============================================================================

def synthesize(
    text: str,
    out_dir: str | Path,
    *,
    target_lang: Optional[str] = None,
    tts_backend: Optional[str] = None,
    ref_audio: Optional[str | Path] = None,
    ref_text: Optional[str] = None,
    output_stem: str = "dubbed_audio",
    use_cpu: bool = False,
) -> Path:
    """
    Standard synthesis without timestamp matching.
    
    Args:
        use_cpu: Force CPU processing (slower but memory-safe for long videos)
    """
    backend = (tts_backend or config.TTS_BACKEND).lower()
    
    if backend == "qwen3":
        return _synthesize_qwen3_chunked(
            text,
            out_dir,
            output_stem=output_stem,
            target_lang=target_lang,
            ref_audio=ref_audio,
            ref_text=ref_text,
            use_cpu=use_cpu,
        )
    
    raise ValueError(f"Backend '{backend}' not supported in optimized mode. Use 'qwen3'.")


def synthesize_timed_segments(
    segments: List[dict],
    out_dir: str | Path,
    *,
    target_lang: Optional[str] = None,
    tts_backend: Optional[str] = None,
    ref_audio: Optional[str | Path] = None,
    ref_text: Optional[str] = None,
    output_stem: str = "dubbed_audio",
    no_duration_match: bool = False,
    use_cpu: bool = False,
) -> Path:
    """Timed segment synthesis."""
    backend = (tts_backend or config.TTS_BACKEND).lower()
    
    if backend == "qwen3":
        return _synthesize_qwen3_timed_segments(
            segments,
            out_dir,
            output_stem=output_stem,
            target_lang=target_lang,
            ref_audio=ref_audio,
            ref_text=ref_text,
            no_duration_match=no_duration_match,
            use_cpu=use_cpu,
        )
    
    raise ValueError(f"Backend '{backend}' not supported. Use 'qwen3'.")


def unload_tts_model():
    """Public function to unload TTS model and free memory."""
    _unload_qwen3_model()
