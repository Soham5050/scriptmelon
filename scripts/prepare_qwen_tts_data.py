#!/usr/bin/env python3
"""
Prepare Qwen3-TTS fine-tuning manifests from local speech datasets.

Supported input manifests:
1) data_json: array of objects with keys like:
   - audioFilename or audio
   - text
   - duration (optional)
   - speaker / speaker_id / client_id (optional)

2) jsonl: one JSON object per line with keys:
   - audio
   - text
   - duration (optional)
   - speaker / speaker_id / client_id (optional)

3) commonvoice_tsv: Common Voice TSV with columns:
   - path
   - sentence
   - client_id (optional in some exports)

Outputs:
- train_raw.jsonl
- val_raw.jsonl
- test_raw.jsonl
- all_raw.jsonl
- data_report.json

Each JSONL row format:
{"audio":"...wav/mp3","text":"...","ref_audio":"...wav","lang":"mr","speaker":"..."}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Row:
    audio: Path
    text: str
    duration: Optional[float]
    speaker: str
    lang: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Qwen3-TTS training manifests.")
    p.add_argument("--dataset-dir", required=True, help="Dataset root directory.")
    p.add_argument(
        "--manifest",
        required=True,
        help="Manifest file path (json/jsonl/tsv). Relative paths resolve from --dataset-dir.",
    )
    p.add_argument(
        "--manifest-format",
        default="auto",
        choices=["auto", "data_json", "jsonl", "commonvoice_tsv"],
        help="Manifest format. auto infers from extension/name.",
    )
    p.add_argument(
        "--audio-root",
        default="",
        help="Audio root folder (relative to dataset-dir or absolute). Default: dataset-dir.",
    )
    p.add_argument("--lang", required=True, choices=["mr", "hi"], help="Language code.")
    p.add_argument("--ref-audio", required=True, help="Reference WAV used for all rows.")
    p.add_argument("--output-dir", required=True, help="Output directory for manifests.")
    p.add_argument(
        "--normalize-audio-dir",
        default="",
        help="Optional directory to store normalized 24kHz mono WAV copies for training.",
    )
    p.add_argument("--min-duration", type=float, default=1.5)
    p.add_argument("--max-duration", type=float, default=15.0)
    p.add_argument("--min-text-chars", type=int, default=3)
    p.add_argument("--max-text-chars", type=int, default=280)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-by-speaker",
        action="store_true",
        help="Split by speaker groups to reduce leakage.",
    )
    p.add_argument(
        "--allow-non-wav",
        action="store_true",
        help="Allow non-wav input clips (duration probing for them will be skipped if metadata missing).",
    )
    return p.parse_args()


def _resolve_path(dataset_dir: Path, path_value: str) -> Path:
    p = Path(path_value)
    return p.resolve() if p.is_absolute() else (dataset_dir / p).resolve()


def _infer_format(manifest: Path) -> str:
    name = manifest.name.lower()
    suffix = manifest.suffix.lower()
    if name.endswith(".tsv"):
        return "commonvoice_tsv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "data_json"
    raise ValueError(f"Cannot infer manifest format from: {manifest}")


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text).strip()
    # Keep common punctuation that helps prosody.
    text = re.sub(r"[\"`]+", "", text)
    return text


def _duration_wav(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return float(wf.getnframes()) / float(rate)
    except Exception:
        return None


def _normalized_audio_path(src: Path, normalize_dir: Path) -> Path:
    token = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    name = f"{src.stem}_{token}.wav"
    return (normalize_dir / name).resolve()


def _normalize_to_24k_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _load_data_json(manifest_path: Path, dataset_dir: Path, audio_root: Path, lang: str) -> list[Row]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("data_json manifest must contain a JSON array.")

    rows: list[Row] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        text = _norm_text(str(item.get("text", "")))
        audio_name = str(item.get("audioFilename") or item.get("audio") or "").strip()
        if not audio_name:
            continue

        audio_path = _resolve_path(audio_root, audio_name)
        duration = item.get("duration")
        try:
            duration_f = float(duration) if duration is not None else None
        except Exception:
            duration_f = None

        speaker = (
            str(item.get("speaker_id") or item.get("speaker") or item.get("client_id") or f"spk_{i % 1000}")
            .strip()
            or f"spk_{i % 1000}"
        )

        rows.append(Row(audio=audio_path, text=text, duration=duration_f, speaker=speaker, lang=lang))
    return rows


def _load_jsonl(manifest_path: Path, dataset_dir: Path, audio_root: Path, lang: str) -> list[Row]:
    rows: list[Row] = []
    for i, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            continue
        text = _norm_text(str(item.get("text", "")))
        audio_name = str(item.get("audio") or item.get("audioFilename") or "").strip()
        if not audio_name:
            continue
        audio_path = _resolve_path(audio_root, audio_name)
        duration = item.get("duration")
        try:
            duration_f = float(duration) if duration is not None else None
        except Exception:
            duration_f = None
        speaker = (
            str(item.get("speaker_id") or item.get("speaker") or item.get("client_id") or f"spk_{i % 1000}")
            .strip()
            or f"spk_{i % 1000}"
        )
        rows.append(Row(audio=audio_path, text=text, duration=duration_f, speaker=speaker, lang=lang))
    return rows


def _load_commonvoice_tsv(manifest_path: Path, dataset_dir: Path, audio_root: Path, lang: str) -> list[Row]:
    rows: list[Row] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, item in enumerate(reader):
            path_value = str(item.get("path") or "").strip()
            text = _norm_text(str(item.get("sentence") or ""))
            if not path_value or not text:
                continue
            audio_path = _resolve_path(audio_root, path_value)
            speaker = (str(item.get("client_id") or "").strip() or f"spk_{i % 1000}")
            rows.append(Row(audio=audio_path, text=text, duration=None, speaker=speaker, lang=lang))
    return rows


def _load_rows(
    manifest_format: str,
    manifest_path: Path,
    dataset_dir: Path,
    audio_root: Path,
    lang: str,
) -> list[Row]:
    if manifest_format == "data_json":
        return _load_data_json(manifest_path, dataset_dir, audio_root, lang)
    if manifest_format == "jsonl":
        return _load_jsonl(manifest_path, dataset_dir, audio_root, lang)
    if manifest_format == "commonvoice_tsv":
        return _load_commonvoice_tsv(manifest_path, dataset_dir, audio_root, lang)
    raise ValueError(f"Unsupported manifest format: {manifest_format}")


def _split_rows(rows: list[Row], val_ratio: float, test_ratio: float, seed: int, split_by_speaker: bool) -> tuple[list[Row], list[Row], list[Row]]:
    rnd = random.Random(seed)
    rows = list(rows)
    if not rows:
        return [], [], []

    val_ratio = max(0.0, min(0.4, val_ratio))
    test_ratio = max(0.0, min(0.4, test_ratio))
    if val_ratio + test_ratio >= 0.8:
        raise ValueError("val_ratio + test_ratio must be < 0.8")

    if not split_by_speaker:
        rnd.shuffle(rows)
        n = len(rows)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test = rows[:n_test]
        val = rows[n_test:n_test + n_val]
        train = rows[n_test + n_val:]
        return train, val, test

    by_speaker: dict[str, list[Row]] = {}
    for row in rows:
        by_speaker.setdefault(row.speaker, []).append(row)
    speakers = list(by_speaker.keys())
    rnd.shuffle(speakers)

    target_test = int(round(len(rows) * test_ratio))
    target_val = int(round(len(rows) * val_ratio))

    test: list[Row] = []
    val: list[Row] = []
    train: list[Row] = []
    count_test = 0
    count_val = 0

    for spk in speakers:
        group = by_speaker[spk]
        if count_test < target_test:
            test.extend(group)
            count_test += len(group)
        elif count_val < target_val:
            val.extend(group)
            count_val += len(group)
        else:
            train.extend(group)

    return train, val, test


def _write_jsonl(rows: Iterable[Row], path: Path, ref_audio: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            row = {
                "audio": str(r.audio),
                "text": r.text,
                "ref_audio": str(ref_audio),
                "lang": r.lang,
                "speaker": r.speaker,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_path = _resolve_path(dataset_dir, args.manifest)
    if args.manifest_format == "auto":
        manifest_format = _infer_format(manifest_path)
    else:
        manifest_format = args.manifest_format

    audio_root = _resolve_path(dataset_dir, args.audio_root) if args.audio_root else dataset_dir
    ref_audio = _resolve_path(dataset_dir, args.ref_audio)
    output_dir = Path(args.output_dir).resolve()
    normalize_dir = Path(args.normalize_audio_dir).resolve() if args.normalize_audio_dir else None

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset-dir not found: {dataset_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not audio_root.exists():
        raise FileNotFoundError(f"audio-root not found: {audio_root}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"ref-audio not found: {ref_audio}")

    loaded = _load_rows(manifest_format, manifest_path, dataset_dir, audio_root, args.lang)

    kept: list[Row] = []
    skipped = {
        "missing_audio": 0,
        "non_wav_audio": 0,
        "duration_out_of_range": 0,
        "empty_or_short_text": 0,
        "long_text": 0,
        "duration_unavailable": 0,
        "audio_normalize_failed": 0,
    }

    normalized_cache: dict[Path, Path] = {}
    normalized_ref_audio = ref_audio
    if normalize_dir is not None:
        normalized_ref_audio = _normalized_audio_path(ref_audio, normalize_dir)
        if not normalized_ref_audio.exists():
            _normalize_to_24k_wav(ref_audio, normalized_ref_audio)

    for r in loaded:
        if len(r.text) < args.min_text_chars:
            skipped["empty_or_short_text"] += 1
            continue
        if len(r.text) > args.max_text_chars:
            skipped["long_text"] += 1
            continue
        if not r.audio.exists():
            skipped["missing_audio"] += 1
            continue
        if not args.allow_non_wav and r.audio.suffix.lower() != ".wav":
            skipped["non_wav_audio"] += 1
            continue

        if normalize_dir is not None:
            normalized_audio = normalized_cache.get(r.audio)
            if normalized_audio is None:
                normalized_audio = _normalized_audio_path(r.audio, normalize_dir)
                if not normalized_audio.exists():
                    try:
                        _normalize_to_24k_wav(r.audio, normalized_audio)
                    except Exception:
                        skipped["audio_normalize_failed"] += 1
                        continue
                normalized_cache[r.audio] = normalized_audio
            r.audio = normalized_audio

        dur = r.duration
        if dur is None and r.audio.suffix.lower() == ".wav":
            dur = _duration_wav(r.audio)
        if dur is None and (args.min_duration > 0 or args.max_duration > 0):
            skipped["duration_unavailable"] += 1
            continue
        if dur is not None:
            if dur < args.min_duration or dur > args.max_duration:
                skipped["duration_out_of_range"] += 1
                continue
        r.duration = dur
        kept.append(r)

    rnd = random.Random(args.seed)
    rnd.shuffle(kept)
    if args.max_samples and args.max_samples > 0:
        kept = kept[: args.max_samples]

    train, val, test = _split_rows(
        kept,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by_speaker=args.split_by_speaker,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    n_all = _write_jsonl(kept, output_dir / "all_raw.jsonl", normalized_ref_audio)
    n_train = _write_jsonl(train, output_dir / "train_raw.jsonl", normalized_ref_audio)
    n_val = _write_jsonl(val, output_dir / "val_raw.jsonl", normalized_ref_audio)
    n_test = _write_jsonl(test, output_dir / "test_raw.jsonl", normalized_ref_audio)

    report = {
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "manifest_format": manifest_format,
        "audio_root": str(audio_root),
        "lang": args.lang,
        "ref_audio": str(ref_audio),
        "normalized_ref_audio": str(normalized_ref_audio),
        "normalize_audio_dir": str(normalize_dir) if normalize_dir is not None else None,
        "seed": args.seed,
        "split_by_speaker": bool(args.split_by_speaker),
        "loaded_rows": len(loaded),
        "kept_rows": len(kept),
        "counts": {
            "all_raw": n_all,
            "train_raw": n_train,
            "val_raw": n_val,
            "test_raw": n_test,
        },
        "filters": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "min_text_chars": args.min_text_chars,
            "max_text_chars": args.max_text_chars,
            "max_samples": args.max_samples,
        },
        "skipped": skipped,
    }
    (output_dir / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"loaded_rows={len(loaded)}")
    print(f"kept_rows={len(kept)}")
    print(f"train={n_train} val={n_val} test={n_test}")
    for k, v in skipped.items():
        print(f"skipped_{k}={v}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
