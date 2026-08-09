# Qwen3-TTS Data Prep (Hindi + Marathi)

This repo now includes `scripts/prepare_qwen_tts_data.py` to build clean Qwen fine-tuning manifests.

## Output files

The script writes:

- `all_raw.jsonl`
- `train_raw.jsonl`
- `val_raw.jsonl`
- `test_raw.jsonl`
- `data_report.json`

Each row is:

```json
{"audio":"...","text":"...","ref_audio":"...","lang":"mr","speaker":"spk_001"}
```

## 1) Common Voice TSV input

Use this when you have `validated.tsv` + `clips/`:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_qwen_tts_data.py `
  --dataset-dir "D:\datasets\cv-mr" `
  --manifest "validated.tsv" `
  --manifest-format commonvoice_tsv `
  --audio-root "clips" `
  --lang mr `
  --ref-audio "ref.wav" `
  --output-dir "training_data\mr" `
  --min-duration 1.5 `
  --max-duration 12 `
  --split-by-speaker
```

For Hindi, run the same command with `--lang hi` and Hindi dataset paths.

## 2) data.json input

Use this when manifest rows look like:
`{"audioFilename":"x.wav","text":"...","duration":4.2}`

```powershell
.\.venv\Scripts\python.exe scripts\prepare_qwen_tts_data.py `
  --dataset-dir "D:\New folder\commonvoice_marathi\commonvoice_marathi" `
  --manifest "data.json" `
  --manifest-format data_json `
  --lang mr `
  --ref-audio "ref.wav" `
  --output-dir "training_data\mr" `
  --normalize-audio-dir "training_data\mr\audio_24k" `
  --min-duration 1.5 `
  --max-duration 12
```

## 3) Continue Qwen fine-tuning pipeline

After generating `train_raw.jsonl`:

```powershell
cd Qwen3-TTS\finetuning

..\..\.venv\Scripts\python.exe prepare_data.py `
  --device cuda:0 `
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz `
  --input_jsonl "..\..\training_data\mr\train_raw.jsonl" `
  --output_jsonl "..\..\training_data\mr\train_with_codes.jsonl" `
  --batch_size 4

..\..\.venv\Scripts\python.exe sft_12hz.py `
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base `
  --output_model_path "..\..\training_runs\mr_v1" `
  --train_jsonl "..\..\training_data\mr\train_with_codes.jsonl" `
  --batch_size 2 `
  --lr 2e-5 `
  --num_epochs 3 `
  --speaker_name mr_speaker_v1
```

## 4) Strongly recommended for anonymous multi-speaker data

If your source manifest has no true speaker IDs, do not fine-tune directly.
Filter clips against your reference speaker first:

```powershell
.\.venv\Scripts\python.exe scripts\filter_qwen_same_speaker.py `
  --input-jsonl "training_data\mr_full_24k\all_raw.jsonl" `
  --output-dir "training_data\mr_same_speaker" `
  --model-path "Qwen3-TTS-12Hz-1.7B-Base" `
  --ref-audio "training_data\mr_full_24k\audio_24k\common_voice_mr_27761945_2e28ceeb6e.wav" `
  --device cuda:0 `
  --dtype bfloat16 `
  --top-k 250 `
  --speaker-name mr_speaker_v1
```

Then build codes from the filtered training set:

```powershell
.\.venv\Scripts\python.exe Qwen3-TTS\finetuning\prepare_data.py `
  --device cuda:0 `
  --tokenizer_model_path Qwen3-TTS-Tokenizer-12Hz `
  --input_jsonl training_data\mr_same_speaker\train_raw.jsonl `
  --output_jsonl training_data\mr_same_speaker\train_with_codes.jsonl `
  --batch_size 4
```

## Notes

- Keep a **single stable** `ref_audio` per speaker during single-speaker fine-tuning.
- For 8GB GPUs, start with `batch_size=1` or `2`.
- If your source clips are MP3, either convert to WAV first or pass `--allow-non-wav`.
- Check `data_report.json` and fix high skip counts before training.
- If the source dataset has no real speaker IDs, use `filter_qwen_same_speaker.py` before SFT.
