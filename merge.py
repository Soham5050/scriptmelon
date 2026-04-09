"""
merge.py
--------
Merge dubbed audio into the source video.

Modes:
1) Replace mode (default): discard original audio, keep dubbed only.
2) Preserve-BGM mode: keep original bed (music/ambience), duck it under dubbed speech, and mix.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import config

log = config.get_logger(__name__)


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise EnvironmentError("ffmpeg not found in PATH. Install from https://ffmpeg.org/download.html")
    return binary


def _ffprobe() -> str:
    binary = shutil.which("ffprobe")
    if binary is None:
        raise EnvironmentError("ffprobe not found in PATH. Install from https://ffmpeg.org/download.html")
    return binary


def _run(cmd: list[str], step_name: str) -> None:
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{step_name} failed (exit {proc.returncode}):\n{proc.stderr.strip()}")


def _video_has_audio_stream(video_path: Path) -> bool:
    cmd = [
        _ffprobe(),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def merge_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    background_audio_path: str | Path | None = None,
    audio_bitrate: str = "192k",
    pad_audio: bool = True,
    preserve_bgm: bool = False,
    original_audio_gain_db: float = -8.0,
    dubbed_audio_gain_db: float = 3.0,
) -> Path:
    """
    Merge audio/video into output.

    preserve_bgm=True will keep original audio bed and mix dubbed voice on top.
    """
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()
    background_audio = Path(background_audio_path).resolve() if background_audio_path else None

    paths_to_check = [video_path, audio_path]
    if background_audio is not None:
        paths_to_check.append(background_audio)

    for p in paths_to_check:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Merging: video='%s' audio='%s' -> '%s' (preserve_bgm=%s, separated_bed=%s)",
        video_path.name,
        audio_path.name,
        output_path.name,
        preserve_bgm,
        bool(background_audio),
    )

    if preserve_bgm and background_audio is None and not _video_has_audio_stream(video_path):
        log.warning("Input video has no audio stream. Falling back to replace-audio merge.")
        preserve_bgm = False

    source_audio = _pad_audio_to_video(audio_path, video_path) if pad_audio else audio_path
    _merge(
        audio_path=source_audio,
        video_path=video_path,
        output_path=output_path,
        audio_bitrate=audio_bitrate,
        background_audio_path=background_audio,
        preserve_bgm=preserve_bgm,
        original_audio_gain_db=original_audio_gain_db,
        dubbed_audio_gain_db=dubbed_audio_gain_db,
        extra_flags=None if pad_audio else ["-shortest"],
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Merge produced no output at {output_path}")

    size_mb = output_path.stat().st_size / 1e6
    log.info("Merge complete - %s (%.1f MB)", output_path.name, size_mb)
    return output_path


def _merge(
    *,
    audio_path: Path,
    video_path: Path,
    output_path: Path,
    audio_bitrate: str,
    background_audio_path: Path | None,
    preserve_bgm: bool,
    original_audio_gain_db: float,
    dubbed_audio_gain_db: float,
    extra_flags: list[str] | None = None,
) -> None:
    cmd = [_ffmpeg(), "-y", "-i", str(video_path)]

    if preserve_bgm and background_audio_path is not None:
        cmd.extend(["-i", str(background_audio_path), "-i", str(audio_path)])
    else:
        cmd.extend(["-i", str(audio_path)])

    if preserve_bgm:
        # Mix strategy:
        # - Lower original bed a bit
        # - Duck original when dubbed voice is active
        # - Mix bed + dubbed and limit peaks
        if background_audio_path is not None:
            filter_complex = (
                f"[1:a]volume={original_audio_gain_db:.2f}dB[orig];"
                f"[2:a]volume={dubbed_audio_gain_db:.2f}dB[dub];"
                "[orig][dub]sidechaincompress=threshold=0.02:ratio=10:attack=6:release=280[ducked];"
                "[ducked][dub]amix=inputs=2:duration=first:normalize=0[mix];"
                "[mix]alimiter=limit=0.95[outa]"
            )
        else:
            filter_complex = (
                f"[0:a]volume={original_audio_gain_db:.2f}dB[orig];"
                f"[1:a]volume={dubbed_audio_gain_db:.2f}dB[dub];"
                "[orig][dub]sidechaincompress=threshold=0.02:ratio=10:attack=6:release=280[ducked];"
                "[ducked][dub]amix=inputs=2:duration=first:normalize=0[mix];"
                "[mix]alimiter=limit=0.95[outa]"
            )
        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v:0",
                "-map",
                "[outa]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-shortest",
            ]
        )
    else:
        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-shortest",
            ]
        )

    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(str(output_path))
    _run(cmd, step_name="Video/audio merge")


def _pad_audio_to_video(audio_path: Path, video_path: Path) -> Path:
    def _duration(path: Path) -> float:
        cmd = [
            _ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True)
        return float(out.stdout.strip())

    video_dur = _duration(video_path)
    audio_dur = _duration(audio_path)
    if audio_dur >= video_dur:
        return audio_path

    pad_secs = video_dur - audio_dur
    log.info("Padding dubbed audio by %.2f s to match video length (%.2f s).", pad_secs, video_dur)
    padded = audio_path.parent / f"{audio_path.stem}_padded{audio_path.suffix}"
    cmd = [
        _ffmpeg(),
        "-y",
        "-i",
        str(audio_path),
        "-af",
        f"apad=pad_dur={pad_secs:.4f}",
        "-c:a",
        "pcm_s16le",
        str(padded),
    ]
    _run(cmd, step_name="Audio padding")
    return padded


def burn_subtitles(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Optional: burn SRT subtitles into video."""
    video_path = Path(video_path).resolve()
    srt_path = Path(srt_path).resolve()
    output_path = Path(output_path).resolve()

    for p in (video_path, srt_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Burning subtitles from '%s' onto video...", srt_path.name)
    cmd = [
        _ffmpeg(),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles={srt_path}",
        "-c:a",
        "copy",
        str(output_path),
    ]
    _run(cmd, step_name="Subtitle burn-in")
    log.info("Subtitled video saved: %s", output_path.name)
    return output_path
