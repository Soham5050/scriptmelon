from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Qwen3-TTS model locally.")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-1.7B",
        help="Hugging Face model id (default: Qwen/Qwen3-TTS-1.7B)",
    )
    p.add_argument(
        "--out",
        default="models/qwen3_tts_1_7b",
        help="Local output directory",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model} -> {out_dir}")
    local_path = snapshot_download(
        repo_id=args.model,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=(args.token or None),
    )
    print(f"Done: {local_path}")


if __name__ == "__main__":
    main()
