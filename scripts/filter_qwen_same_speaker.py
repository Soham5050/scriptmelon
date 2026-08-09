#!/usr/bin/env python3
"""
Filter a Qwen TTS training JSONL down to clips that are most similar to a
reference speaker, using Qwen's built-in speaker encoder.

Input JSONL rows are expected to contain at least:
{"audio":"...","text":"...","ref_audio":"..."}

Outputs:
- scored_all.jsonl
- filtered_all.jsonl
- train_raw.jsonl
- val_raw.jsonl
- test_raw.jsonl
- speaker_filter_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter JSONL to same-speaker clips using Qwen speaker encoder.")
    p.add_argument("--input-jsonl", required=True, help="Prepared raw JSONL input.")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--model-path", default="Qwen3-TTS-12Hz-1.7B-Base", help="Qwen model path.")
    p.add_argument("--ref-audio", default="", help="Override reference audio path. Defaults to first row ref_audio.")
    p.add_argument("--device", default="cuda:0", help="Model device.")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    p.add_argument("--min-sim", type=float, default=0.0, help="Minimum cosine similarity to keep.")
    p.add_argument("--top-k", type=int, default=250, help="Keep top K rows after threshold. 0 means keep all.")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--speaker-name", default="speaker_target", help="Speaker label to write in output rows.")
    p.add_argument("--max-samples", type=int, default=0, help="Debug limit for scoring. 0 means all.")
    return p.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(rows: list[dict], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rnd = random.Random(seed)
    items = list(rows)
    rnd.shuffle(items)
    n = len(items)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(torch.dot(a, b).item())


def _load_embedding(tts, audio_path: str | Path) -> torch.Tensor:
    audio_path = str(audio_path)
    normalized = tts._normalize_audio_inputs(audio_path)
    wav, sr = normalized[0]
    target_sr = int(tts.model.speaker_encoder_sample_rate)
    if int(sr) != target_sr:
        wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
    return tts.model.extract_speaker_embedding(audio=wav, sr=target_sr)


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    ref_audio = args.ref_audio.strip() or str(rows[0].get("ref_audio", "")).strip()
    if not ref_audio:
        raise ValueError("No reference audio found. Pass --ref-audio explicitly.")

    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, args.dtype)
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    ref_embedding = _load_embedding(tts, ref_audio)
    scored_rows: list[dict] = []

    for idx, row in enumerate(rows):
        audio = str(row.get("audio", "")).strip()
        if not audio:
            continue
        sim = _cosine_similarity(_load_embedding(tts, audio), ref_embedding)
        updated = dict(row)
        updated["speaker"] = args.speaker_name
        updated["speaker_similarity"] = round(sim, 6)
        scored_rows.append(updated)
        if (idx + 1) % 50 == 0:
            print(f"scored={idx + 1}/{len(rows)}")

    scored_rows.sort(key=lambda x: float(x.get("speaker_similarity", 0.0)), reverse=True)

    filtered_rows = [r for r in scored_rows if float(r.get("speaker_similarity", 0.0)) >= args.min_sim]
    if args.top_k > 0:
        filtered_rows = filtered_rows[: args.top_k]

    train, val, test = _split_rows(filtered_rows, args.val_ratio, args.test_ratio, args.seed)

    _write_jsonl(scored_rows, output_dir / "scored_all.jsonl")
    _write_jsonl(filtered_rows, output_dir / "filtered_all.jsonl")
    _write_jsonl(train, output_dir / "train_raw.jsonl")
    _write_jsonl(val, output_dir / "val_raw.jsonl")
    _write_jsonl(test, output_dir / "test_raw.jsonl")

    sims = [float(r["speaker_similarity"]) for r in scored_rows]
    report = {
        "input_jsonl": str(input_jsonl),
        "ref_audio": str(Path(ref_audio).resolve()),
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "rows_scored": len(scored_rows),
        "rows_kept": len(filtered_rows),
        "top_k": args.top_k,
        "min_sim": args.min_sim,
        "speaker_name": args.speaker_name,
        "similarity": {
            "max": max(sims) if sims else None,
            "min": min(sims) if sims else None,
            "avg": (sum(sims) / len(sims)) if sims else None,
        },
        "split_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }
    (output_dir / "speaker_filter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"rows_scored={len(scored_rows)}")
    print(f"rows_kept={len(filtered_rows)}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
