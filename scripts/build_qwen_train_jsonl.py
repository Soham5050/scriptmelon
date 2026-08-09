#!/usr/bin/env python3
"""
Build Qwen3-TTS finetuning JSONL from a CommonVoice-style Marathi folder.

Expected input directory contents:
- data.json with items: {"audioFilename": "...wav", "duration": float, "text": "..."}
- wav files referenced by audioFilename

Output JSONL format (required by Qwen3-TTS finetuning):
{"audio":"...wav","text":"...","ref_audio":"...wav"}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train_raw.jsonl for Qwen3-TTS finetuning.")
    p.add_argument("--dataset-dir", required=True, help="Folder containing data.json and wav files")
    p.add_argument("--output-jsonl", required=True, help="Output JSONL path")
    p.add_argument("--ref-audio", required=True, help="Reference speaker wav to use for all rows")
    p.add_argument("--min-duration", type=float, default=2.0, help="Min clip duration in seconds")
    p.add_argument("--max-duration", type=float, default=12.0, help="Max clip duration in seconds")
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--shuffle", action="store_true", help="Shuffle rows before truncation")
    p.add_argument("--seed", type=int, default=42, help="Random seed when --shuffle is used")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_path = dataset_dir / "data.json"
    out_path = Path(args.output_jsonl).resolve()
    ref_audio = Path(args.ref_audio).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    skipped_missing = 0
    skipped_empty = 0
    skipped_dur = 0

    for item in records:
        text = str(item.get("text", "")).strip()
        if not text:
            skipped_empty += 1
            continue

        dur = float(item.get("duration", 0.0) or 0.0)
        if dur < args.min_duration or dur > args.max_duration:
            skipped_dur += 1
            continue

        fn = str(item.get("audioFilename", "")).strip()
        if not fn:
            skipped_missing += 1
            continue
        audio_path = (dataset_dir / fn).resolve()
        if not audio_path.exists():
            skipped_missing += 1
            continue

        rows.append(
            {
                "audio": str(audio_path),
                "text": text,
                "ref_audio": str(ref_audio),
            }
        )

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)

    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote={len(rows)}")
    print(f"skipped_missing_audio={skipped_missing}")
    print(f"skipped_empty_text={skipped_empty}")
    print(f"skipped_duration={skipped_dur}")
    print(f"output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

