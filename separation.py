"""
separation.py
-------------
Dialogue/background separation helpers for BGM-safe dubbing.

Current backend:
- demucs (via CLI/module invocation)
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config

log = config.get_logger(__name__)


@dataclass
class SeparationResult:
    speech_path: Path
    bed_path: Path
    backend: str
    model: str
    device: str
    seconds: float


def _demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def _resolve_device(device: Optional[str]) -> str:
    requested = (device or getattr(config, "SEPARATION_DEVICE", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested
    profile = config.get_runtime_profile()
    return "cuda" if bool(profile.get("cuda_available", False)) else "cpu"


def _demucs_output_paths(out_dir: Path, model: str, input_audio: Path) -> tuple[Path, Path]:
    base = out_dir / "demucs" / model / input_audio.stem
    return base / "vocals.wav", base / "no_vocals.wav"


def _run_demucs(input_audio: Path, out_dir: Path, model: str, device: str) -> tuple[Path, Path]:
    if not _demucs_available():
        raise RuntimeError("demucs is not installed. Run: pip install demucs")

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-o",
        str(out_dir / "demucs"),
        "-n",
        model,
        "--two-stems",
        "vocals",
        "--device",
        device,
        str(input_audio),
    ]
    log.info("Running Demucs separation: model=%s device=%s", model, device)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "demucs separation failed")

    vocals, no_vocals = _demucs_output_paths(out_dir, model, input_audio)
    if not vocals.exists() or not no_vocals.exists():
        raise RuntimeError("demucs completed but expected vocals/no_vocals stems were not created")
    return vocals, no_vocals


def separate_dialogue_and_bed(
    input_audio: str | Path,
    out_dir: str | Path,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    stem_prefix: str = "separated",
) -> SeparationResult:
    """
    Separate an input mix into a speech/dialogue stem and a background bed.

    The speech stem is sourced from the Demucs vocals stem. The background bed
    is sourced from the complementary no_vocals stem.
    """
    backend_name = (backend or getattr(config, "SEPARATION_BACKEND", "demucs") or "demucs").strip().lower()
    if backend_name not in {"demucs"}:
        raise ValueError(f"Unsupported separation backend: {backend_name}")

    input_audio = Path(input_audio).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    model_name = (model or getattr(config, "SEPARATION_MODEL", "htdemucs") or "htdemucs").strip()
    resolved_device = _resolve_device(device)

    started = time.time()
    try:
        vocals_path, bed_path = _run_demucs(input_audio, out_dir, model_name, resolved_device)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if resolved_device != "cpu" and any(token in msg for token in ("cuda", "cublas", "out of memory")):
            log.warning("Demucs failed on %s. Retrying separation on CPU...", resolved_device)
            vocals_path, bed_path = _run_demucs(input_audio, out_dir, model_name, "cpu")
            resolved_device = "cpu"
        else:
            raise

    speech_out = out_dir / f"{stem_prefix}_speech.wav"
    bed_out = out_dir / f"{stem_prefix}_bed.wav"
    shutil.copy2(vocals_path, speech_out)
    shutil.copy2(bed_path, bed_out)

    elapsed = time.time() - started
    log.info(
        "Separation complete: speech='%s' bed='%s' in %.1fs",
        speech_out.name,
        bed_out.name,
        elapsed,
    )
    return SeparationResult(
        speech_path=speech_out,
        bed_path=bed_out,
        backend=backend_name,
        model=model_name,
        device=resolved_device,
        seconds=elapsed,
    )
