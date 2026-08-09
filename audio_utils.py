"""
audio_utils.py
--------------
FFmpeg-based audio extraction helpers.
No Python audio library is required — all work is delegated to FFmpeg,
which handles virtually every container / codec combination.
"""

import subprocess
import shutil
from pathlib import Path

import config

log = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_ffmpeg() -> str:
    """Return the absolute path to the ffmpeg binary, or raise."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise EnvironmentError(
            "ffmpeg not found in PATH.\n"
            "  macOS  : brew install ffmpeg\n"
            "  Ubuntu : sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    return binary


def _run(cmd: list[str], step_name: str) -> None:
    """Execute *cmd* and raise RuntimeError with stderr on failure."""
    log.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_audio(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    stem: str = "extracted_audio",
    apply_filter: bool = True,
) -> Path:
    """
    Extract and normalise audio from *video_path* to a WAV file.

    Parameters
    ----------
    video_path  : Source video (any format FFmpeg understands).
    out_dir     : Directory where the WAV file will be written.
    sample_rate : Target sample rate in Hz (default 16 000 — optimal for Whisper).
    channels    : Number of output channels (default 1 = mono).
    stem        : Filename stem for the output file.

    Returns
    -------
    Absolute Path to the extracted WAV file.
    """
    ffmpeg = _require_ffmpeg()
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    out_wav = out_dir / f"{stem}.wav"
    log.info("Extracting audio from '%s' → %s", video_path.name, out_wav.name)

    cmd = [
        ffmpeg,
        "-y",                          # overwrite without prompt
        "-i", str(video_path),
        "-vn",                         # drop video stream
    ]
    if apply_filter and config.ASR_AUDIO_PREPROCESS_ENABLED and config.ASR_AUDIO_FILTER.strip():
        cmd += ["-af", config.ASR_AUDIO_FILTER.strip()]
    cmd += [
        "-acodec", "pcm_s16le",        # PCM 16-bit little-endian (uncompressed WAV)
        "-ar", str(sample_rate),       # resample to target rate
        "-ac", str(channels),          # mono/stereo
        str(out_wav),
    ]
    _run(cmd, step_name="Audio extraction")

    size_kb = out_wav.stat().st_size // 1024
    log.info("Audio extracted  (%d KB, %d Hz, %d ch)", size_kb, sample_rate, channels)
    return out_wav


def get_video_duration(video_path: str | Path) -> float:
    """
    Return the duration of *video_path* in seconds using ffprobe.
    Useful for progress estimation and audio-length checks.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise EnvironmentError("ffprobe not found — install ffmpeg (includes ffprobe).")

    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{proc.stderr.strip()}")

    return float(proc.stdout.strip())


def convert_audio_format(
    src: str | Path,
    dst: str | Path,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path:
    """
    Generic audio format conversion using FFmpeg.
    Output format is inferred from *dst*'s extension (.wav, .mp3, .flac …).
    """
    ffmpeg = _require_ffmpeg()
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-y", "-i", str(src)]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd.append(str(dst))

    _run(cmd, step_name="Audio conversion")
    log.info("Converted %s → %s", src.name, dst.name)
    return dst
