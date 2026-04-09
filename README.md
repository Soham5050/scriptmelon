# AI Video Dubbing Pipeline

A modular, production-oriented Python pipeline that takes any video file and
outputs a dubbed version in a different language using entirely local or
self-hosted models (except for the NVIDIA NIM Whisper transcription API).

```
video.mp4
   │
   ▼  FFmpeg                  audio_utils.py
extracted.wav (16kHz mono)
   │
   ▼  NVIDIA NIM Whisper      transcribe.py
transcript.txt  ("Hello, welcome to…")
   │
   ▼  IndicTrans2 / MarianMT  translate.py
translation.txt ("नमस्ते, स्वागत है…")
   │
   ▼  Qwen3-TTS (local)       tts.py
dubbed_audio.wav
   │
   ▼  FFmpeg                  merge.py
output_hi.mp4  ← final dubbed video
   │
   ▼  [optional] NVIDIA Maxine lipsync.py
output_hi_lipsync.mp4
```

---

## Project Structure

```
ai_video_dubber/
├── main.py            ← CLI entry point + pipeline orchestration
├── config.py          ← All config via environment variables
├── audio_utils.py     ← Step 1: FFmpeg audio extraction
├── transcribe.py      ← Step 2: NVIDIA NIM Whisper ASR
├── translate.py       ← Step 3: IndicTrans2 / MarianMT / Google Translate
├── tts.py             ← Step 4: Qwen3-TTS local speech synthesis + voice cloning
├── merge.py           ← Step 5: FFmpeg audio-video merge
├── lipsync.py         ← Step 6 (optional): NVIDIA Maxine LipSync
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

### GPU (strongly recommended for IndicTrans2 and Qwen3-TTS)
- IndicTrans2 runs on CPU but is ~10× slower than on a GPU.
- Qwen3-TTS with CPU only is usable for short clips (< 2 min); GPU is preferred.
- Google Translate backend (`--backend google`) has no GPU requirement.

---

## 2. Environment Setup

```bash
# Clone / copy project
cd ai_video_dubber

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# For CUDA 12.x GPU support (replace the torch line in requirements.txt):
# pip install torch>=2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | **Yes** | NIM API key from https://ngc.nvidia.com |
| `NVIDIA_WHISPER_MODEL` | No | Default: `nvidia/canary-1b` |
| `TRANSLATION_BACKEND` | No | `indictrans2` \| `marian` \| `google` |
| `TTS_REF_AUDIO` | No | Path to reference voice WAV for cloning |
| `TTS_REF_TEXT` | No | Transcript of reference audio |
| `NVIDIA_MAXINE_API_KEY` | Only with `--lipsync` | Maxine cloud key |

---

## 4. First-time Model Downloads

On the first run, two large models are downloaded automatically:

| Model | Size | Backend | One-time |
|---|---|---|---|
| `ai4bharat/indictrans2-en-indic-1B` | ~4 GB | IndicTrans2 | Yes |
| Qwen3-TTS 1.7B local model | ~4-5 GB | Qwen3-TTS | Yes |

To avoid re-downloading, HuggingFace caches models in `~/.cache/huggingface/`.
Set `HF_HOME` to change this location.

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

### With optional LipSync
```bash
python main.py --input talk.mp4 --src_lang en --target_lang hi --lipsync
```

### All flags
```
  --input       / -i    Source video file (required)
  --target_lang / -t    Target language ISO-639-1 code (required)
  --output      / -o    Output path (default: output/<stem>_<lang>.mp4)
  --src_lang    / -s    Source language or 'auto' (default: en)
  --backend     / -b    Translation backend: indictrans2|marian|google
  --ref_audio           Reference WAV for voice cloning
  --ref_text            Transcript of reference audio
  --no_clone            Disable voice cloning
  --no_pad              Don't pad audio to video length
  --lipsync             Apply NVIDIA Maxine LipSync (optional)
  --keep_temp           Keep intermediate files
  --verbose     / -v    DEBUG logging
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

For language pairs not supported by IndicTrans2, set `--backend marian` or `--backend google`.

---

## 7. Output Files

```
output/
└── lecture_hi.mp4        ← final dubbed video

# With --keep_temp:
temp_dubbing/
├── extracted_audio.wav   ← 16kHz mono source audio
├── transcript.txt        ← original speech text
├── translation.txt       ← translated text
└── dubbed_audio.wav      ← Qwen3-TTS synthesised voice
artifacts/
├── transcript.txt
└── translation.txt
```

---

## 8. Swapping Components

The modular design makes each step replaceable:

| Step | Current | How to swap |
|---|---|---|
| Transcription | NVIDIA NIM Whisper | Edit `transcribe.py` — use `openai` Whisper, local `faster-whisper`, or AssemblyAI |
| Translation | IndicTrans2 | Add a new `BaseTranslator` subclass in `translate.py`; set `TRANSLATION_BACKEND` |
| TTS | Qwen3-TTS | Edit `tts.py` — swap Qwen3-TTS for another backend if needed |
| LipSync | NVIDIA Maxine | Edit `lipsync.py` — `apply_lipsync_wav2lip()` is already provided as an alternative |

---

## 9. Troubleshooting

| Error | Fix |
|---|---|
| `ffmpeg: command not found` | Install FFmpeg and add to PATH |
| `NVIDIA_API_KEY is not set` | Copy `.env.example → .env` and fill in key |
| Audio > 25 MB | Split with `ffmpeg -i v.mp4 -segment_time 600 -f segment seg_%03d.mp4` |
| `No Helsinki-NLP model for pair` | Use `--backend google` for that language pair |
| Qwen3-TTS `ImportError` | `pip install -U qwen-tts` in your venv |
| Dubbed audio length mismatch | Use `--no_pad` to trim, or omit to pad silence |
| IndicTrans2 slow on CPU | Add GPU or use `--backend google` |
