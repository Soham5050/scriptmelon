# ScriptMelon — AI Video Dubbing Pipeline

A modular, adaptive Python pipeline that takes any video file and outputs a
dubbed version in a different language. Runs entirely locally on consumer
GPU hardware (RTX 5060 8GB and up), with an optional cloud transcription
backend if you'd rather not run ASR locally.

```
video.mp4
   │
   ▼  FFmpeg                        audio_utils.py
extracted.wav (16kHz mono)
   │
   ▼  faster-whisper / WhisperX /   transcribe.py
   │  NVIDIA NIM Whisper (choose one via TRANSCRIBE_BACKEND)
transcript.txt  ("Hello, welcome to…")
   │
   ▼  Google / IndicTrans2 /        translate.py
   │  MarianMT / NLLB
translation.txt ("नमस्ते, स्वागत है…")
   │
   ▼  Qwen3-TTS (local, w/ voice    tts.py
   │  cloning support)
dubbed_audio.wav
   │
   ▼  FFmpeg                        merge.py
output_hi.mp4  ← final dubbed video
   │
   ▼  [optional --lipsync]          lipsync.py
   │  NVIDIA Maxine
output_hi_lipsync.mp4
```

A desktop GUI (`studio_gui.py`, built with Dear PyGui) wraps the same CLI so
you don't have to remember flags.

---

## Project Structure

```
scriptmelon/
├── main.py                  ← CLI entry point + pipeline orchestration
├── studio_gui.py             ← Desktop GUI (Dear PyGui) — wraps main.py
├── config.py                  ← All config via environment variables (see .env.example)
├── audio_utils.py             ← Step 1: FFmpeg audio extraction
├── separation.py               ← Optional: Demucs speech/background separation (--separate_audio)
├── transcribe.py                ← Step 2: ASR (faster-whisper / WhisperX / NVIDIA NIM)
├── translate.py                  ← Step 3: Google / IndicTrans2 / MarianMT / NLLB
├── glossary.py                    ← Domain glossary enforcement for translation
├── quality_metrics.py             ← ASR/translation/TTS quality scoring
├── quality_validation.py          ← Quality gate logic used mid-pipeline
├── tts.py                          ← Step 4: Qwen3-TTS local speech synthesis + voice cloning
├── merge.py                         ← Step 5: FFmpeg audio-video merge
├── lipsync.py                        ← Step 6 (optional, --lipsync): NVIDIA Maxine LipSync
├── scripts/                           ← Qwen3-TTS fine-tuning data prep utilities
├── Qwen3-TTS/                          ← Local Qwen3-TTS package (installed via -e ./Qwen3-TTS)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 1. System Prerequisites

### FFmpeg (required)
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add bin/ to PATH

# Verify
ffmpeg -version && ffprobe -version
```

### Python 3.10+
Qwen3-TTS requires Python 3.10 or later.
```bash
python3 --version   # must be 3.10+
```

### GPU (strongly recommended)
- `requirements.txt` pins CUDA 12.8 PyTorch wheels by default (RTX 50-series).
  For a different CUDA version or CPU-only, replace the three `torch*` lines
  with the matching install command from pytorch.org.
- The pipeline auto-profiles your GPU at runtime (`ADAPTIVE_RUNTIME_ENABLED`)
  and adjusts chunk sizes / CPU fallback based on `LOW_VRAM_THRESHOLD_GB` and
  `MID_VRAM_THRESHOLD_GB` in `.env`.
- Google Translate backend (`--backend google`) needs no GPU.
- `--force_cpu` forces CPU-only TTS if you hit CUDA OOM.

---

## 2. Environment Setup

```bash
cd scriptmelon

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the local `Qwen3-TTS` package in editable mode
(`-e ./Qwen3-TTS`) — that folder must exist alongside `main.py`.

---

## 3. Configuration

```bash
cp .env.example .env
```

`.env` is git-ignored — never commit it. Every setting has a working default
in `config.py`; you only need to touch what you want to change. The ones
people actually need to look at:

| Variable | Required | Description |
|---|---|---|
| `TRANSCRIBE_BACKEND` | No (default `faster_whisper`) | `faster_whisper` (local) \| `whisperx` (local, alignment+diarization) \| `nvidia` (cloud) |
| `NVIDIA_API_KEY` | Only if `TRANSCRIBE_BACKEND=nvidia` | NIM API key from https://ngc.nvidia.com |
| `HF_TOKEN` | Only if `WHISPERX_DIARIZE_ENABLED=true` | HuggingFace token for gated pyannote diarization models |
| `TRANSLATION_BACKEND` | No (default `indictrans2`) | `indictrans2` \| `marian` \| `nllb` \| `google` |
| `TTS_REF_AUDIO` / `TTS_REF_TEXT` | No | Default voice-cloning reference (overridable per-run with `--ref_audio`/`--ref_text`) |
| `NVIDIA_MAXINE_API_KEY` | Only with `--lipsync` | Maxine cloud key (falls back to `NVIDIA_API_KEY` if unset) |

---

## 4. First-time Model Downloads

On first run, models download automatically and get cached in
`~/.cache/huggingface/` (set `HF_HOME` to change the location):

| Model | Backend | Trigger |
|---|---|---|
| `ai4bharat/indictrans2-en-indic-1B` (~4 GB) | Translation | `TRANSLATION_BACKEND=indictrans2` |
| Qwen3-TTS 1.7B (~4-5 GB) | TTS | Always (default TTS backend) |
| faster-whisper `large-v3` | ASR | `TRANSCRIBE_BACKEND=faster_whisper` (default) |
| Demucs `htdemucs` | Separation | `--separate_audio` / `--preserve_bgm` |

---

## 5. Usage

### Basic — English video → Hindi dub
```bash
python main.py --input lecture.mp4 --target_lang hi
```

### With voice cloning
```bash
python main.py \
  --input film.mkv \
  --target_lang es \
  --ref_audio my_voice.wav \
  --ref_text "This is a sample of my voice for cloning."
```

### Keep intermediate files + custom output path
```bash
python main.py \
  --input doc.mp4 \
  --target_lang te \
  --output out/doc_telugu.mp4 \
  --keep_temp
```

### Fast translation using Google Translate (no GPU needed)
```bash
python main.py --input video.mp4 --target_lang fr --backend google
```

### Preserve background music while dubbing over dialogue
```bash
python main.py --input video.mp4 --target_lang hi --preserve_bgm --separate_audio
```

### Resume an interrupted run from checkpoint
```bash
python main.py --input video.mp4 --target_lang hi --keep_temp --resume
```

### With optional LipSync
```bash
python main.py --input talk.mp4 --target_lang hi --lipsync
```

### Or just use the GUI
```bash
python studio_gui.py
```

### All flags
```
--input        / -i    Source video file (required)
--target_lang  / -t    Target language code (required)
--output       / -o    Output path (default: output/<stem>_<lang>.mp4)
--src_lang     / -s    Source language code, or 'auto' (default: en)
--backend      / -b    Translation backend: indictrans2 | marian | nllb | google
--tts_backend           TTS backend: qwen3 (only option currently)

--no_duration_match     Disable time-stretching to ASR timestamps
--skip_quality_gate     Skip ASR quality gate (not recommended)
--force_cpu             Force CPU for TTS (slower but memory-safe)
--interactive           Pause after transcription for manual timeline review/edit
--resume                Resume from checkpoints in the temp working directory (needs --keep_temp on the original run)

--ref_audio             Reference voice WAV for cloning
--ref_text              Transcript of the reference audio
--no_clone              Disable voice cloning
--no_pad                Do not pad audio to video length

--preserve_bgm          Preserve original music/ambience bed, mix dubbed speech on top
--separate_audio        Run speech/background separation before ASR (implied in practice by --preserve_bgm)
--bgm_gain_db           Gain (dB) for the original bed when --preserve_bgm is set (default -8.0)
--dub_gain_db           Gain (dB) for the dubbed voice when --preserve_bgm is set (default 3.0)

--lipsync               Apply NVIDIA Maxine LipSync
--keep_temp             Keep intermediate files (temp_dubbing_<name>/)
--verbose      / -v     DEBUG logging
```

---

## 6. Language Code Reference

| Code | Language  | IndicTrans2 | MarianMT | Google |
|------|-----------|:-----------:|:--------:|:------:|
| hi   | Hindi     | ✓           | ✓        | ✓      |
| bn   | Bengali   | ✓           | ✓        | ✓      |
| te   | Telugu    | ✓           | ✓        | ✓      |
| ta   | Tamil     | ✓           | ✓        | ✓      |
| mr   | Marathi   | ✓           | ✓        | ✓      |
| gu   | Gujarati  | ✓           | ✓        | ✓      |
| kn   | Kannada   | ✓           | ✓        | ✓      |
| ml   | Malayalam | ✓           | ✓        | ✓      |
| pa   | Punjabi   | ✓           | ✓        | ✓      |
| es   | Spanish   | —           | ✓        | ✓      |
| fr   | French    | —           | ✓        | ✓      |
| de   | German    | —           | ✓        | ✓      |
| ja   | Japanese  | —           | ✓        | ✓      |

For pairs `indictrans2` doesn't cover, use `--backend marian`, `--backend nllb`,
or `--backend google`.

---

## 7. Output Files

```
output/
└── lecture_hi.mp4          ← final dubbed video

# With --keep_temp:
temp_dubbing_lecture/
├── extracted_audio.wav     ← 16kHz mono source audio
├── transcript.txt          ← original speech text
├── translation.txt         ← translated text
├── segments_asr.json       ← ASR segments with timestamps
├── segments_translated.json ← translated segments with timestamps
├── review_transcript.txt   ← editable timeline (with --interactive)
├── checkpoint_*.json       ← per-step checkpoints (for --resume)
└── dubbed_audio.wav        ← Qwen3-TTS synthesised voice
```

---

## 8. Swapping Components

The modular design makes each step replaceable:

| Step | Current | How to swap |
|---|---|---|
| Transcription | faster-whisper (default) | Set `TRANSCRIBE_BACKEND=whisperx` or `nvidia` in `.env`, or edit `transcribe.py` |
| Translation | IndicTrans2 (default) | Set `TRANSLATION_BACKEND` in `.env`, or add a new backend in `translate.py` |
| TTS | Qwen3-TTS | Edit `tts.py` — swap Qwen3-TTS for another backend if needed |
| LipSync | NVIDIA Maxine | Edit `lipsync.py` — `apply_lipsync_wav2lip()` is provided as an offline alternative |

---

## 9. Troubleshooting

| Error | Fix |
|---|---|
| `ffmpeg: command not found` | Install FFmpeg and add to PATH |
| `NVIDIA_API_KEY is not set for nvidia transcription backend` | Only relevant if `TRANSCRIBE_BACKEND=nvidia`; set the key in `.env` or switch to `faster_whisper` |
| Audio > 25 MB (NVIDIA backend only) | Split with `ffmpeg -i v.mp4 -segment_time 600 -f segment seg_%03d.mp4` |
| `No Helsinki-NLP model for pair` | Use `--backend google` or `--backend nllb` for that language pair |
| Qwen3-TTS `ImportError` | Confirm `Qwen3-TTS/` exists and reinstall with `pip install -e ./Qwen3-TTS` |
| Dubbed audio length mismatch | Use `--no_pad` to trim, or omit to pad silence; `--no_duration_match` disables time-stretching entirely |
| CUDA out of memory | Use `--force_cpu`, or lower `QWEN3_LOCAL_MAX_CHARS_PER_CHUNK` / `TTS_MIN_FREE_VRAM_GB` in `.env` |
| Interrupted a long run | Rerun with `--keep_temp --resume` to pick up from the last checkpoint |
