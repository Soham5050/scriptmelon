"""
config.py
---------
Adaptive runtime configuration for mixed GPU tiers (6GB to 96GB+).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Any

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# =============================================================================
# Runtime / Platform Settings
# =============================================================================

# Safe default across Windows/Linux builds. Users can still override in .env.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# Disable flash-attn on Windows (requires MSVC build tools)
# Use sdpa (scaled dot product attention) instead - nearly as fast, no build needed
os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "sdpa")

# Disable torch compile on Windows (can cause issues)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# =============================================================================
# Transcription Settings
# =============================================================================

TRANSCRIBE_BACKEND: str = os.environ.get("TRANSCRIBE_BACKEND", "faster_whisper").lower()

# Options: tiny, base, small, medium, large-v1, large-v2, large-v3
FASTER_WHISPER_MODEL: str = os.environ.get("FASTER_WHISPER_MODEL", "small")
FASTER_WHISPER_DEVICE: str = os.environ.get("FASTER_WHISPER_DEVICE", "auto")

# INT8 default is safer for low-VRAM cards
FASTER_WHISPER_COMPUTE_TYPE: str = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
FASTER_WHISPER_BEAM_SIZE: int = int(os.environ.get("FASTER_WHISPER_BEAM_SIZE", "5"))
FASTER_WHISPER_BEST_OF: int = int(os.environ.get("FASTER_WHISPER_BEST_OF", "3"))
FASTER_WHISPER_VAD_FILTER: bool = os.environ.get("FASTER_WHISPER_VAD_FILTER", "true").lower() in {
    "1", "true", "yes", "on",
}
FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = os.environ.get(
    "FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false"
).lower() in {"1", "true", "yes", "on"}

# ASR Quality thresholds
ASR_MIN_LANGUAGE_PROB: float = float(os.environ.get("ASR_MIN_LANGUAGE_PROB", "0.75"))
ASR_MIN_AVG_LOGPROB: float = float(os.environ.get("ASR_MIN_AVG_LOGPROB", "-1.1"))
ASR_MAX_REPETITION_RATIO: float = float(os.environ.get("ASR_MAX_REPETITION_RATIO", "0.25"))
ASR_MAX_SEGMENT_DURATION: float = float(os.environ.get("ASR_MAX_SEGMENT_DURATION", "15.0"))
ASR_AUDIO_PREPROCESS_ENABLED: bool = os.environ.get("ASR_AUDIO_PREPROCESS_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
ASR_AUDIO_FILTER: str = os.environ.get("ASR_AUDIO_FILTER", "")
ASR_REFINE_LOW_CONF_SEGMENTS_ENABLED: bool = os.environ.get(
    "ASR_REFINE_LOW_CONF_SEGMENTS_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
ASR_REFINE_MAX_SEGMENTS: int = int(os.environ.get("ASR_REFINE_MAX_SEGMENTS", "8"))
ASR_REFINE_MARGIN_SEC: float = float(os.environ.get("ASR_REFINE_MARGIN_SEC", "0.25"))
ASR_RETRY_ON_LOW_QUALITY_ENABLED: bool = os.environ.get(
    "ASR_RETRY_ON_LOW_QUALITY_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
ASR_RETRY_MODEL: str = os.environ.get("ASR_RETRY_MODEL", "").strip()
ASR_RETRY_BEAM_SIZE: int = int(os.environ.get("ASR_RETRY_BEAM_SIZE", "8"))
ASR_RETRY_BEST_OF: int = int(os.environ.get("ASR_RETRY_BEST_OF", "5"))
ASR_RETRY_VAD_FILTER: bool = os.environ.get("ASR_RETRY_VAD_FILTER", "true").lower() in {
    "1", "true", "yes", "on",
}
ASR_RETRY_CONDITION_ON_PREVIOUS_TEXT: bool = os.environ.get(
    "ASR_RETRY_CONDITION_ON_PREVIOUS_TEXT", "false"
).lower() in {"1", "true", "yes", "on"}
ASR_SKIP_SUSPICIOUS_SEGMENTS: bool = os.environ.get(
    "ASR_SKIP_SUSPICIOUS_SEGMENTS", "false"
).lower() in {"1", "true", "yes", "on"}
ASR_SKIP_SUSPICIOUS_MAX_RATIO: float = float(
    os.environ.get("ASR_SKIP_SUSPICIOUS_MAX_RATIO", "0.25")
)

# WhisperX compatibility knobs (used when TRANSCRIBE_BACKEND=whisperx)
WHISPERX_MODEL: str = os.environ.get("WHISPERX_MODEL", "large-v3")
WHISPERX_COMPUTE_TYPE: str = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16")
WHISPERX_BATCH_SIZE: int = int(os.environ.get("WHISPERX_BATCH_SIZE", "4"))
WHISPERX_DIARIZE_ENABLED: bool = os.environ.get("WHISPERX_DIARIZE_ENABLED", "false").lower() in {
    "1", "true", "yes", "on",
}
WHISPERX_ALIGN_ENABLED: bool = os.environ.get("WHISPERX_ALIGN_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# NVIDIA API settings
NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_WHISPER_MODEL: str = os.environ.get("NVIDIA_WHISPER_MODEL", "openai/whisper-large-v3")

# =============================================================================
# Translation Settings
# =============================================================================

TRANSLATION_BACKEND: str = os.environ.get("TRANSLATION_BACKEND", "indictrans2")
TRANSLATION_ENABLE_SMART_ROUTING: bool = os.environ.get(
    "TRANSLATION_ENABLE_SMART_ROUTING", "false"
).lower() in {"1", "true", "yes", "on"}
INDICTRANS2_EN_INDIC_MODEL: str = os.environ.get(
    "INDICTRANS2_EN_INDIC_MODEL", "ai4bharat/indictrans2-en-indic-1B"
)
INDICTRANS2_INDIC_EN_MODEL: str = os.environ.get(
    "INDICTRANS2_INDIC_EN_MODEL", "ai4bharat/indictrans2-indic-en-1B"
)

# Translation chunking
TRANSLATION_MAX_CHUNK_CHARS: int = int(os.environ.get("TRANSLATION_MAX_CHUNK_CHARS", "800"))

# =============================================================================
# TTS Settings
# =============================================================================

TTS_BACKEND: str = os.environ.get("TTS_BACKEND", "qwen3").lower()
if TTS_BACKEND != "qwen3":
    TTS_BACKEND = "qwen3"

# Generic reference voice settings.
TTS_REF_AUDIO: str = os.environ.get("TTS_REF_AUDIO", "ref_voice.wav")
TTS_REF_TEXT: str = os.environ.get("TTS_REF_TEXT", "")

# TTS chunking
TTS_MAX_CHARS_PER_CHUNK: int = int(os.environ.get("TTS_MAX_CHARS_PER_CHUNK", "80"))
TTS_CROSSFADE_MS: int = int(os.environ.get("TTS_CROSSFADE_MS", "50"))

# Speech/background separation for BGM-safe dubbing
SEPARATION_BACKEND: str = os.environ.get("SEPARATION_BACKEND", "demucs").lower()
SEPARATION_MODEL: str = os.environ.get("SEPARATION_MODEL", "htdemucs")
SEPARATION_DEVICE: str = os.environ.get("SEPARATION_DEVICE", "auto").lower()

# Duration matching
TTS_DURATION_MATCH_MIN_SPEED: float = float(os.environ.get("TTS_DURATION_MATCH_MIN_SPEED", "0.80"))
TTS_DURATION_MATCH_MAX_SPEED: float = float(os.environ.get("TTS_DURATION_MATCH_MAX_SPEED", "1.25"))
_tts_duration_match_en_min_raw = os.environ.get("TTS_DURATION_MATCH_EN_MIN_SPEED", "").strip()
_tts_duration_match_en_max_raw = os.environ.get("TTS_DURATION_MATCH_EN_MAX_SPEED", "").strip()
TTS_DURATION_MATCH_EN_MIN_SPEED: Optional[float] = (
    float(_tts_duration_match_en_min_raw) if _tts_duration_match_en_min_raw else None
)
TTS_DURATION_MATCH_EN_MAX_SPEED: Optional[float] = (
    float(_tts_duration_match_en_max_raw) if _tts_duration_match_en_max_raw else None
)
TTS_DURATION_LOCK_ENABLED: bool = os.environ.get("TTS_DURATION_LOCK_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
TTS_DURATION_LOCK_TOLERANCE_SEC: float = float(os.environ.get("TTS_DURATION_LOCK_TOLERANCE_SEC", "0.06"))
TTS_DURATION_FORCE_EXACT_FIT: bool = os.environ.get("TTS_DURATION_FORCE_EXACT_FIT", "true").lower() in {
    "1", "true", "yes", "on",
}
TTS_SPEAKER_REF_MAX_SECONDS: float = float(os.environ.get("TTS_SPEAKER_REF_MAX_SECONDS", "6.0"))
TTS_UNLOAD_MODEL_BETWEEN_CHUNKS: bool = os.environ.get(
    "TTS_UNLOAD_MODEL_BETWEEN_CHUNKS", "false"
).lower() in {"1", "true", "yes", "on"}

# =============================================================================
# Qwen3-TTS Settings (Local)
# =============================================================================

QWEN3_LOCAL_MODEL_ID: str = os.environ.get("QWEN3_LOCAL_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
QWEN3_LOCAL_DEVICE: str = os.environ.get("QWEN3_LOCAL_DEVICE", "cuda:0")
QWEN3_LOCAL_DTYPE: str = os.environ.get("QWEN3_LOCAL_DTYPE", "float16")

# Use sdpa instead of flash_attn on Windows (no build required)
# Set to "true" only if you have MSVC 14.0+ installed and want to build flash-attn
QWEN3_LOCAL_USE_FLASH_ATTN: bool = os.environ.get("QWEN3_LOCAL_USE_FLASH_ATTN", "false").lower() in {
    "1", "true", "yes", "on",
}

# Qwen runtime controls
QWEN3_LOCAL_MAX_CHARS_PER_CHUNK: int = int(os.environ.get("QWEN3_LOCAL_MAX_CHARS_PER_CHUNK", "80"))
QWEN3_LOCAL_TEMPERATURE: float = float(os.environ.get("QWEN3_LOCAL_TEMPERATURE", "0.7"))
QWEN3_EXPRESSIVE_ENGLISH_ENABLED: bool = os.environ.get(
    "QWEN3_EXPRESSIVE_ENGLISH_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
QWEN3_EXPRESSIVE_ENGLISH_BOOST: float = float(
    os.environ.get("QWEN3_EXPRESSIVE_ENGLISH_BOOST", "0.08")
)
QWEN3_LOCAL_OUTPUT_SR: int = int(os.environ.get("QWEN3_LOCAL_OUTPUT_SR", "24000"))
QWEN3_LOW_VRAM_MAX_CHARS_PER_CHUNK: int = int(
    os.environ.get("QWEN3_LOW_VRAM_MAX_CHARS_PER_CHUNK", "50")
)
QWEN3_MID_VRAM_MAX_CHARS_PER_CHUNK: int = int(
    os.environ.get("QWEN3_MID_VRAM_MAX_CHARS_PER_CHUNK", "80")
)
QWEN3_HIGH_VRAM_MAX_CHARS_PER_CHUNK: int = int(
    os.environ.get("QWEN3_HIGH_VRAM_MAX_CHARS_PER_CHUNK", "120")
)
QWEN3_LOW_VRAM_BATCH_SIZE: int = int(os.environ.get("QWEN3_LOW_VRAM_BATCH_SIZE", "1"))
QWEN3_MID_VRAM_BATCH_SIZE: int = int(os.environ.get("QWEN3_MID_VRAM_BATCH_SIZE", "2"))
QWEN3_HIGH_VRAM_BATCH_SIZE: int = int(os.environ.get("QWEN3_HIGH_VRAM_BATCH_SIZE", "3"))
QWEN3_LOW_VRAM_CLEANUP_EVERY: int = int(os.environ.get("QWEN3_LOW_VRAM_CLEANUP_EVERY", "1"))
QWEN3_MID_VRAM_CLEANUP_EVERY: int = int(os.environ.get("QWEN3_MID_VRAM_CLEANUP_EVERY", "2"))
QWEN3_HIGH_VRAM_CLEANUP_EVERY: int = int(os.environ.get("QWEN3_HIGH_VRAM_CLEANUP_EVERY", "4"))

# =============================================================================
# Cloud TTS (Sarvam) - Alternative to local
# =============================================================================

SARVAM_API_KEY: str = os.environ.get("SARVAM_API_KEY", "")
SARVAM_BASE_URL: str = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai")
SARVAM_TTS_MODEL: str = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER: str = os.environ.get("SARVAM_TTS_SPEAKER", "Shubh")
SARVAM_TTS_PACE: float = float(os.environ.get("SARVAM_TTS_PACE", "1.0"))
SARVAM_TTS_TEMPERATURE: float = float(os.environ.get("SARVAM_TTS_TEMPERATURE", "0.6"))
SARVAM_TTS_OUTPUT_CODEC: str = os.environ.get("SARVAM_TTS_OUTPUT_CODEC", "wav")
SARVAM_TTS_SAMPLE_RATE: int = int(os.environ.get("SARVAM_TTS_SAMPLE_RATE", "24000"))

# =============================================================================
# GPU Memory Management
# =============================================================================

# If set to 0, memory fraction is auto-selected from GPU tier.
GPU_MEMORY_FRACTION: float = float(os.environ.get("GPU_MEMORY_FRACTION", "0"))
ADAPTIVE_RUNTIME_ENABLED: bool = os.environ.get("ADAPTIVE_RUNTIME_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
LOW_VRAM_THRESHOLD_GB: float = float(os.environ.get("LOW_VRAM_THRESHOLD_GB", "7"))
MID_VRAM_THRESHOLD_GB: float = float(os.environ.get("MID_VRAM_THRESHOLD_GB", "14"))
HIGH_VRAM_THRESHOLD_GB: float = float(os.environ.get("HIGH_VRAM_THRESHOLD_GB", "32"))

# Force CPU for specific components (for very long videos)
FORCE_CPU_FOR_TTS: bool = os.environ.get("FORCE_CPU_FOR_TTS", "").lower() in ("true", "1", "yes")
FORCE_CPU_FOR_TRANSLATION: bool = os.environ.get("FORCE_CPU_FOR_TRANSLATION", "").lower() in ("true", "1", "yes")

# =============================================================================
# Paths
# =============================================================================

DEFAULT_TARGET_LANG: str = os.environ.get("DEFAULT_TARGET_LANG", "hi")
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output")
TEMP_DIR: str = os.environ.get("TEMP_DIR", "temp")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
BENCHMARK_DIR: str = os.environ.get("BENCHMARK_DIR", "benchmarks")
BENCHMARK_AUTO_SAVE: bool = os.environ.get("BENCHMARK_AUTO_SAVE", "true").lower() in {
    "1", "true", "yes", "on",
}

# =============================================================================
# Maxine (Optional)
# =============================================================================

NVIDIA_MAXINE_API_KEY: str = os.environ.get("NVIDIA_MAXINE_API_KEY", NVIDIA_API_KEY)
NVIDIA_MAXINE_BASE_URL: str = os.environ.get("NVIDIA_MAXINE_BASE_URL", "https://grpc.nvcf.nvidia.com:443")

_RUNTIME_PROFILE_CACHE: dict[str, Any] | None = None


# =============================================================================
# Functions
# =============================================================================

def _recommended_memory_fraction(total_gb: float) -> float:
    """Pick safe per-process memory fraction by GPU tier."""
    if total_gb <= LOW_VRAM_THRESHOLD_GB:
        return 0.80
    if total_gb <= MID_VRAM_THRESHOLD_GB:
        return 0.86
    if total_gb <= HIGH_VRAM_THRESHOLD_GB:
        return 0.90
    return 0.92


def get_runtime_profile(force_refresh: bool = False) -> dict[str, Any]:
    """
    Return adaptive runtime profile based on detected GPU memory.
    Used by main/tts to tune chunking and CPU fallback.
    """
    global _RUNTIME_PROFILE_CACHE
    if _RUNTIME_PROFILE_CACHE is not None and not force_refresh:
        return _RUNTIME_PROFILE_CACHE

    profile: dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": "cpu",
        "vram_gb": 0.0,
        "tier": "cpu",
        "memory_fraction": 0.0,
        "tts_batch_size": 1,
        "tts_max_chars_per_chunk": max(30, min(QWEN3_LOCAL_MAX_CHARS_PER_CHUNK, 50)),
        "tts_cleanup_every": 1,
        "prefer_cpu_for_long_video": True,
        "prefer_cpu_for_very_long_video": True,
        "segment_chunk_seconds_short": 45.0,
        "segment_chunk_seconds_medium": 90.0,
        "segment_chunk_seconds_long": 120.0,
        "segment_chunk_seconds_very_long": 180.0,
    }

    if not ADAPTIVE_RUNTIME_ENABLED:
        _RUNTIME_PROFILE_CACHE = profile
        return profile

    try:
        import torch

        if torch.cuda.is_available():
            total_gb = float(torch.cuda.get_device_properties(0).total_memory / 1024**3)
            gpu_name = str(torch.cuda.get_device_name(0))
            tier: str
            if total_gb <= LOW_VRAM_THRESHOLD_GB:
                tier = "low"
                profile.update(
                    tts_batch_size=max(1, QWEN3_LOW_VRAM_BATCH_SIZE),
                    tts_max_chars_per_chunk=max(30, min(QWEN3_LOCAL_MAX_CHARS_PER_CHUNK, QWEN3_LOW_VRAM_MAX_CHARS_PER_CHUNK)),
                    tts_cleanup_every=max(1, QWEN3_LOW_VRAM_CLEANUP_EVERY),
                    prefer_cpu_for_long_video=True,
                    prefer_cpu_for_very_long_video=True,
                    segment_chunk_seconds_short=45.0,
                    segment_chunk_seconds_medium=75.0,
                    segment_chunk_seconds_long=90.0,
                    segment_chunk_seconds_very_long=120.0,
                )
            elif total_gb <= MID_VRAM_THRESHOLD_GB:
                tier = "mid"
                profile.update(
                    tts_batch_size=max(1, QWEN3_MID_VRAM_BATCH_SIZE),
                    tts_max_chars_per_chunk=max(40, min(QWEN3_LOCAL_MAX_CHARS_PER_CHUNK, QWEN3_MID_VRAM_MAX_CHARS_PER_CHUNK)),
                    tts_cleanup_every=max(1, QWEN3_MID_VRAM_CLEANUP_EVERY),
                    prefer_cpu_for_long_video=False,
                    prefer_cpu_for_very_long_video=True,
                    segment_chunk_seconds_short=60.0,
                    segment_chunk_seconds_medium=120.0,
                    segment_chunk_seconds_long=150.0,
                    segment_chunk_seconds_very_long=210.0,
                )
            else:
                tier = "high"
                profile.update(
                    tts_batch_size=max(1, QWEN3_HIGH_VRAM_BATCH_SIZE),
                    tts_max_chars_per_chunk=max(60, min(QWEN3_LOCAL_MAX_CHARS_PER_CHUNK, QWEN3_HIGH_VRAM_MAX_CHARS_PER_CHUNK)),
                    tts_cleanup_every=max(1, QWEN3_HIGH_VRAM_CLEANUP_EVERY),
                    prefer_cpu_for_long_video=False,
                    prefer_cpu_for_very_long_video=False,
                    segment_chunk_seconds_short=60.0,
                    segment_chunk_seconds_medium=150.0,
                    segment_chunk_seconds_long=240.0,
                    segment_chunk_seconds_very_long=360.0,
                )

            fraction = GPU_MEMORY_FRACTION if GPU_MEMORY_FRACTION > 0 else _recommended_memory_fraction(total_gb)
            tier_fraction_cap = 0.84 if tier == "low" else (0.90 if tier == "mid" else 0.94)
            fraction = min(float(fraction), tier_fraction_cap)
            profile.update(
                cuda_available=True,
                gpu_name=gpu_name,
                vram_gb=total_gb,
                tier=tier,
                memory_fraction=max(0.5, min(0.95, float(fraction))),
            )
    except Exception:
        pass

    _RUNTIME_PROFILE_CACHE = profile
    return profile

def validate() -> None:
    """Validate configuration."""
    missing = []
    if TRANSCRIBE_BACKEND == "nvidia" and not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY")
    if TTS_BACKEND == "sarvam" and not SARVAM_API_KEY:
        missing.append("SARVAM_API_KEY")
    
    if missing:
        raise EnvironmentError(f"Missing: {', '.join(missing)}")


def get_logger(name: str) -> logging.Logger:
    """Get configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def setup_gpu_memory_limit():
    """Set adaptive GPU memory guardrails and log active profile."""
    try:
        import torch
        if torch.cuda.is_available():
            profile = get_runtime_profile()
            fraction = float(profile.get("memory_fraction", 0.8))
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            log = get_logger(__name__)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            limited_gb = total_gb * fraction
            log.info(
                "[GPU] %s | Total: %.1fGB | Tier: %s | Limited to: %.1fGB (%.0f%%)",
                profile.get("gpu_name", "cuda"),
                total_gb,
                profile.get("tier", "unknown"),
                limited_gb,
                fraction * 100.0,
            )
            
            # Log Windows-specific info
            if sys.platform == "win32":
                log.info("[Windows] Using sdpa attention (flash-attn disabled)")
                log.info("[Windows] To enable flash-attn, install MSVC 14.0+ and set QWEN3_LOCAL_USE_FLASH_ATTN=true")
                
    except Exception as e:
        log = get_logger(__name__)
        log.debug(f"GPU setup warning: {e}")


def print_vram_status():
    """Print VRAM status for debugging."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            print(f"\n[VRAM] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Free: {free:.2f}GB")
            return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}
    except Exception:
        pass
    return None
