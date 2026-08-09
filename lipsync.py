"""
lipsync.py
----------
Optional Step 6: NVIDIA Maxine LipSync — retarget mouth movements in the
original video to match the synthesised dubbed audio track.

NVIDIA Maxine Audio2Face / LipSync is available as a cloud function via
NVIDIA's Cloud Function (NVCF) platform.  The function ingests:
  • A video file (or portrait image sequence)
  • An audio WAV

And returns a modified video with lip movements synchronised to the new audio.

SDK / Docs: https://developer.nvidia.com/maxine
NVCF Docs : https://docs.api.nvidia.com/

THIS MODULE IS OPTIONAL.  The main pipeline skips it unless --lipsync is
passed on the CLI.  It is designed to be easily replaced by any other
lipsync implementation (e.g., Wav2Lip, SadTalker) by swapping this module.

Current implementation uses NVIDIA's NVCF gRPC-based invocation pattern
via the `requests` library (REST polling variant — simpler than gRPC for
most users without the NVCF gRPC SDK installed).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

import config

log = config.get_logger(__name__)

# NVCF endpoints (check NVIDIA's latest docs for exact function IDs)
_NVCF_INVOKE_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions"
_LIPSYNC_FUNCTION_ID = "c9114669-a13b-4683-9e44-7e2fbc4a7e7e"   # Maxine Audio2Face

# Polling config
_POLL_INTERVAL_S = 5
_POLL_TIMEOUT_S  = 600


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lipsync(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    api_key: Optional[str] = None,
    function_id: str = _LIPSYNC_FUNCTION_ID,
    max_wait: int = _POLL_TIMEOUT_S,
) -> Path:
    """
    Retarget lip movements in *video_path* to match *audio_path* using
    NVIDIA Maxine LipSync via NVCF.

    Parameters
    ----------
    video_path   : Path to the dubbed video (after merge step).
    audio_path   : Path to the dubbed audio WAV used for synthesis.
    output_path  : Destination for the lipsync-corrected video.
    api_key      : NVIDIA Maxine API key (defaults to NVIDIA_MAXINE_API_KEY).
    function_id  : NVCF function ID for the Maxine LipSync function.
    max_wait     : Maximum seconds to poll for completion.

    Returns
    -------
    Absolute Path to the lipsync-corrected video.

    Raises
    ------
    EnvironmentError : API key missing.
    RuntimeError     : NVCF invocation or polling failed.

    Notes
    -----
    This requires your NVIDIA Maxine subscription to include the LipSync /
    Audio2Face function.  Contact NVIDIA if you need access.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed — pip install requests")

    key = api_key or config.NVIDIA_MAXINE_API_KEY
    if not key:
        raise EnvironmentError(
            "NVIDIA_MAXINE_API_KEY is not set.  "
            "Set it in .env or pass api_key= explicitly."
        )

    video_path  = Path(video_path).resolve()
    audio_path  = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()

    for p in (video_path, audio_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Submitting lipsync job to NVIDIA Maxine NVCF (function: %s)…",
        function_id,
    )

    # ---- Step 1: Submit the job ----------------------------------------
    video_b64 = base64.b64encode(video_path.read_bytes()).decode()
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

    payload = {
        "requestBody": {
            "video": video_b64,
            "audio": audio_b64,
            "output_format": "mp4",
        }
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "NVCF-POLL-SECONDS": "60",
    }

    invoke_url = f"{_NVCF_INVOKE_URL}/{function_id}"
    response = requests.post(invoke_url, json=payload, headers=headers, timeout=60)

    if response.status_code == 200:
        # Synchronous response — result is immediately available
        result_data = response.json()
        return _save_result(result_data, output_path)

    if response.status_code == 202:
        # Asynchronous — poll the status endpoint
        request_id = response.headers.get("NVCF-REQID")
        if not request_id:
            raise RuntimeError(
                "NVCF returned 202 but no NVCF-REQID in headers.\n"
                f"Response: {response.text}"
            )
        log.info("Job queued (request id: %s) — polling…", request_id)
        return _poll_result(request_id, output_path, headers, max_wait)

    raise RuntimeError(
        f"NVCF submission failed ({response.status_code}):\n{response.text}"
    )


def _poll_result(
    request_id: str,
    output_path: Path,
    headers: dict,
    max_wait: int,
) -> Path:
    """Poll the NVCF status endpoint until the job completes."""
    import requests

    status_url = (
        f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}"
    )
    deadline = time.time() + max_wait

    while time.time() < deadline:
        resp = requests.get(status_url, headers=headers, timeout=30)

        if resp.status_code == 200:
            log.info("Lipsync job complete.")
            return _save_result(resp.json(), output_path)

        if resp.status_code == 202:
            pct = resp.headers.get("NVCF-PCT-COMPLETE", "?")
            log.info("Still processing… (%s%% complete)", pct)
            time.sleep(_POLL_INTERVAL_S)
            continue

        raise RuntimeError(
            f"NVCF status poll failed ({resp.status_code}):\n{resp.text}"
        )

    raise RuntimeError(
        f"Lipsync job timed out after {max_wait} s (request id: {request_id})."
    )


def _save_result(result: dict, output_path: Path) -> Path:
    """Decode the base64 video payload and write it to *output_path*."""
    video_b64 = (
        result.get("responseBody", {}).get("video")
        or result.get("video")
    )
    if not video_b64:
        raise RuntimeError(
            f"Could not find 'video' key in NVCF response.\n"
            f"Keys received: {list(result.keys())}"
        )

    output_path.write_bytes(base64.b64decode(video_b64))
    size_mb = output_path.stat().st_size / 1e6
    log.info("Lipsync video saved: %s (%.1f MB)", output_path.name, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# Stub for local / offline lipsync (e.g. Wav2Lip)
# ---------------------------------------------------------------------------

def apply_lipsync_wav2lip(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint: str = "checkpoints/wav2lip_gan.pth",
) -> Path:
    """
    Alternative lipsync using the local Wav2Lip model.

    Wav2Lip GitHub : https://github.com/Rudrabha/Wav2Lip
    Install        : Follow the Wav2Lip repo README (separate environment recommended)

    This function calls inference.py as a subprocess to avoid dependency
    conflicts between Wav2Lip's requirements and the rest of the pipeline.
    """
    import subprocess

    video_path  = Path(video_path).resolve()
    audio_path  = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Running Wav2Lip lipsync…")

    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint,
        "--face",  str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="Wav2Lip")
    if proc.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed:\n{proc.stderr}")

    log.info("Wav2Lip complete: %s", output_path.name)
    return output_path
