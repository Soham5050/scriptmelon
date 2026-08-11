# ScriptMelon — AI Video Dubbing Pipeline

A modular, adaptive Python pipeline that takes any video file and outputs a
dubbed version in a different language. Runs entirely locally on consumer
GPU hardware — as low as 6GB VRAM via automatic quantization — with an
optional cloud transcription backend if you'd rather not run ASR locally.

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
   ▼  IndicF5 / Qwen3-TTS (local,  tts.py
   │  w/ voice cloning; routed by
   │  target language)
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
├── speakers.py                     ← Per-speaker reference-clip selection for --multi_voice
├── tts.py                          ← Step 4: local speech synthesis + voice cloning (IndicF5 / Qwen3-TTS)
├── merge.py                         ← Step 5: FFmpeg audio-video merge
├── lipsync.py                        ← Step 6 (optional, --lipsync): NVIDIA Maxine LipSync
├── cleanup.py                         ← Housekeeping: delete leftover temp_dubbing_* directories
├── tests/                              ← Regression tests (pytest)
├── scripts/                             ← Qwen3-TTS fine-tuning data prep utilities
├── Qwen3-TTS/                            ← Local Qwen3-TTS package (installed via -e ./Qwen3-TTS)
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
  and picks a VRAM tier — `ultra_low` / `low` / `mid` / `high` — based on
  `ULTRA_LOW_VRAM_THRESHOLD_GB`, `LOW_VRAM_THRESHOLD_GB`, and
  `MID_VRAM_THRESHOLD_GB` in `.env`. Each tier adjusts chunk sizes, batch
  sizes, and CPU fallback automatically.
- **6GB cards (RTX 2060, 3050, similar) land in the `ultra_low` tier**,
  which additionally applies quantization automatically: `int8_float16` ASR
  compute (near-fp16 quality at roughly half the memory, so a 6GB card still
  runs the full `large-v3` Whisper model instead of a weaker one), and
  8-bit translation. The TTS model is intentionally **not** quantized below
  8-bit — 4-bit measurably degrades prosody, so the tradeoff isn't taken
  even on the smallest tier. Controlled by `ASR_COMPUTE_TYPE`,
  `TRANSLATION_QUANTIZATION`, and `TTS_QUANTIZATION` in `.env` — `auto`
  (the default) picks per-tier automatically; explicit values always win.
- Google Translate backend (`--backend google`) needs no GPU.
- `--force_cpu` forces CPU-only TTS if you hit CUDA OOM.

---

## 2. Environment Setup

```bash
git clone --recurse-submodules https://github.com/Soham5050/scriptmelon.git
cd scriptmelon

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps git+https://github.com/ai4bharat/IndicF5.git
```

`requirements.txt` installs the local `Qwen3-TTS` package in editable mode
(`-e ./Qwen3-TTS`), so that folder has to be populated before you install. It is
a git submodule — if you already cloned without `--recurse-submodules`, the
directory exists but is empty and the install fails on it:

```bash
git submodule update --init --recursive
```

IndicF5 is a separate line because it needs `--no-deps`, and pip requirements
files can't carry that flag. Its own metadata pins `numpy<=1.26.4` and
`transformers<4.50`; both are stale, and honouring them downgrades the
environment out from under Qwen3-TTS (`transformers==4.57.3`), torch 2.8,
whisperx and demucs. Its actual import closure is pinned in `requirements.txt`
instead, so `--no-deps` leaves nothing missing. Skip this step and Indic
languages fail at the TTS stage with an install hint rather than silently
producing bad audio.

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
| `HF_TOKEN` | For IndicF5 TTS, and if `WHISPERX_DIARIZE_ENABLED=true` | HuggingFace token. Both `ai4bharat/IndicF5` and the pyannote diarization models are gated — accept their terms on the model page first |
| `TRANSLATION_BACKEND` | No (default `indictrans2`) | `indictrans2` \| `marian` \| `nllb` \| `google` |
| `TTS_BACKEND` | No (default `auto`) | `auto` routes by target language (see §6) \| `qwen3` \| `indicf5` |
| `TTS_REF_AUDIO` / `TTS_REF_TEXT` | No | Default voice-cloning reference (overridable per-run with `--ref_audio`/`--ref_text`). IndicF5 needs the *text* to match the audio — supply both or neither |
| `NVIDIA_MAXINE_API_KEY` | Only with `--lipsync` | Maxine cloud key (falls back to `NVIDIA_API_KEY` if unset) |

---

## 4. First-time Model Downloads

On first run, models download automatically and get cached in
`~/.cache/huggingface/` (set `HF_HOME` to change the location):

| Model | Backend | Trigger |
|---|---|---|
| `ai4bharat/indictrans2-en-indic-1B` (~4 GB) | Translation | `TRANSLATION_BACKEND=indictrans2` |
| `ai4bharat/IndicF5` (0.4B) | TTS | Indic target language (see §6). Gated — needs `HF_TOKEN` |
| Qwen3-TTS 1.7B (~4-5 GB) | TTS | Non-Indic target language, or `--tts_backend qwen3` |
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

### Multiple speakers, each dubbed in their own cloned voice
```bash
python main.py --input meeting.mp4 --target_lang hi --multi_voice
```

Needs a HuggingFace token — see [Multi-Voice Dubbing](#multi-voice-dubbing)
below. Without one, `--multi_voice` logs a warning and falls back to
dubbing everyone with a single voice rather than failing the run.

If you already know how many people are speaking, pass it — it measurably
improves diarization accuracy over letting it guess:

```bash
python main.py --input meeting.mp4 --target_lang hi --multi_voice --num_speakers 3
```

### Resume an interrupted run from checkpoint
```bash
python main.py --input video.mp4 --target_lang hi --keep_temp --resume
```

### With optional LipSync
```bash
python main.py --input talk.mp4 --target_lang hi --lipsync
```

### Clean up leftover temp directories after a successful run
```bash
python main.py --input video.mp4 --target_lang hi --cleanup_temp
```

`--keep_temp` runs leave their working directory behind, and those add up
fast. `--cleanup_temp` sweeps old ones once the run succeeds, keeping any
directory that still holds resumable checkpoints. To inspect or clean up
manually:

```bash
python cleanup.py                 # list what exists, delete nothing
python cleanup.py --delete        # delete, keeping resumable directories
python cleanup.py --delete --all  # delete everything, checkpoints included
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
--tts_backend           TTS backend: auto | qwen3 | indicf5 (default auto — routes by target language)

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

--multi_voice           Detect each speaker and dub them with their own cloned voice.
                        Implies --separate_audio; needs HF_TOKEN (see Multi-Voice Dubbing below)
--num_speakers          Exact speaker count, when known. Only used with --multi_voice;
                        improves diarization accuracy over auto-detection

--lipsync               Apply NVIDIA Maxine LipSync
--keep_temp             Keep intermediate files (temp_dubbing_<name>/)
--cleanup_temp          After a successful run, delete leftover temp_dubbing_* directories
                        from earlier --keep_temp runs (resumable ones are kept)
--verbose      / -v     DEBUG logging
```

---

## Multi-Voice Dubbing

For source video with more than one speaker — interviews, panels, meetings —
`--multi_voice` detects each speaker, clones each of their voices separately
from clean reference clips, and dubs each person in their own voice rather
than one shared voice for everyone. Overlapping speech (interruptions, cross-
talk) is mixed at the correct timestamps instead of concatenated sequentially,
so simultaneous speech doesn't push every later line out of sync.

### Setup (one-time)

Diarization (figuring out *who* is speaking *when*) uses a HuggingFace model
that's gated behind accepting its terms:

1. Create a token: https://huggingface.co/settings/tokens
2. Accept the model terms: https://hf.co/pyannote/speaker-diarization-community-1
3. Set `HF_TOKEN=<your token>` in `.env`

Without a valid `HF_TOKEN`, `--multi_voice` logs a warning and falls back to
single-voice dubbing — it will not fail the run outright.

### How it works

1. `--multi_voice` implies `--separate_audio` — the source is split into a
   vocals stem and a background stem before anything else happens.
2. WhisperX (forced automatically, regardless of `TRANSCRIBE_BACKEND`) runs
   diarization on the vocals stem, tagging each transcript segment with a
   speaker ID.
3. For each speaker, `speakers.py` selects their cleanest reference clip —
   longest continuous run, filtered for ASR confidence and crosstalk, pulled
   from the vocals stem so background noise never contaminates the cloned
   voice. A speaker needs `DIARIZATION_MIN_SPEAKER_SECONDS` (default 1.5s)
   of clean speech to get a cloned voice; shorter appearances fall back to
   the default voice rather than failing the run.
4. Each speaker's lines are synthesized by the routed TTS backend, conditioned
   on their own reference clip — the model loads once for the whole run, only
   the per-line voice conditioning changes. IndicF5 additionally uses the
   transcript of that clip, taken from the ASR segments it was cut from.
5. The dubbed lines are placed back on the original timeline and mixed
   against the separated background stem, so music and ambience from the
   source are preserved underneath the new dialogue.

### Relevant `.env` settings

| Variable | Default | Description |
|---|---|---|
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | HuggingFace diarization model |
| `DIARIZATION_MAX_SPEAKERS` | `8` | Upper bound on distinct cloned voices; extra speakers reuse the default voice |
| `DIARIZATION_MIN_SPEAKER_SECONDS` | `1.5` | Minimum clean speech needed before a speaker gets their own cloned voice |
| `DIARIZATION_FILL_NEAREST` | `true` | Assigns the nearest speaker to segments with no diarization overlap, so every segment gets dubbed rather than silently dropped |

---

## 6. Language Support

Translation and speech synthesis do **not** cover the same languages, and the gap
used to be invisible. A language with a translation backend but no TTS model
still produced a finished video — it just did not sound like the language. So the
two are listed separately.

### 6.1 Text-to-speech

`--tts_backend auto` (the default) picks the engine from the target language.

| Engine | Languages | Notes |
|--------|-----------|-------|
| **IndicF5** | `as` `bn` `gu` `hi` `kn` `ml` `mr` `or` `pa` `ta` `te` | 11 Indian languages. Gated on Hugging Face — accept the terms and set `HF_TOKEN`. |
| **Qwen3-TTS** | `zh` `en` `ja` `ko` `de` `fr` `ru` `pt` `es` `it` | 10 languages. No Indian language among them. |
| *neither* | everything else, including Urdu (`ur`) | `auto` sends these to Qwen3-TTS, which does not refuse — it falls back to language auto-detect and produces fluent-sounding nonsense. |

Both engines clone the reference voice, so `--multi_voice` and `--ref_audio` work
either way. Naming a backend explicitly (`--tts_backend qwen3`) overrides the
routing even when the language is a bad fit — that is how you A/B the two.

IndicF5 also wants the *transcript* of the reference clip, not just the clip.
The pipeline derives it from the ASR segments the clip was cut from. If it cannot
(`--no_clone`, or an externally supplied `--ref_audio` with no `--ref_text`), it
falls back to a bundled Punjabi prompt and warns: pronunciation stays correct,
but the original voice is lost for that speaker.

### 6.2 Translation

`--backend` defaults to auto-routing by language pair. Pairs that aren't in the
routing table fall through to `TRANSLATION_BACKEND` (default `indictrans2`),
which handles en↔Indic only — so for the non-Indic languages you have to name a
backend yourself. It fails loudly (`Unknown language code for IndicTrans2`)
rather than mistranslating, but it does fail.

| Code | Language   | Auto-routed to | `--backend nllb` |
|------|------------|----------------|:----------------:|
| hi   | Hindi      | IndicTrans2    | ✓ |
| bn   | Bengali    | IndicTrans2    | ✓ |
| te   | Telugu     | IndicTrans2    | ✓ |
| ta   | Tamil      | IndicTrans2    | ✓ |
| mr   | Marathi    | IndicTrans2    | ✓ |
| gu   | Gujarati   | IndicTrans2    | ✓ |
| kn   | Kannada    | IndicTrans2    | ✓ |
| ml   | Malayalam  | IndicTrans2    | ✓ |
| pa   | Punjabi    | IndicTrans2    | ✓ |
| ur   | Urdu       | IndicTrans2    | ✓ |
| or   | Odia       | default backend handles it | ✓ |
| as   | Assamese   | default backend handles it | ✓ |
| es   | Spanish    | MarianMT       | ✓ |
| fr   | French     | MarianMT       | ✓ |
| de   | German     | MarianMT       | ✓ |
| zh   | Chinese    | nothing — pass `--backend` | ✓ |
| ja   | Japanese   | nothing — pass `--backend` | ✓ |
| ko   | Korean     | nothing — pass `--backend` | ✓ |
| ru   | Russian    | nothing — pass `--backend` | ✓ |
| ar   | Arabic     | nothing — pass `--backend` | ✓ |
| pt   | Portuguese | nothing — pass `--backend` | — |
| it   | Italian    | nothing — pass `--backend` | — |

What each backend can actually do:

- **`indictrans2`** — en↔Indic only. Any other pair raises
  `IndicTrans2 supports en<->Indic only`.
- **`nllb`** — the ✓ column above; it needs a FLORES code for both sides, and
  `pt`/`it` don't have one, so those two are the gaps. en↔X only.
- **`marian`** — resolves `Helsinki-NLP/opus-mt-<src>-<tgt>` at runtime and does
  not check the pair against a list first. If that model exists on the Hub it
  works; if not you get `No Marian model for <src>-><tgt>`, and for non-English
  pairs it retries by pivoting through English. Coverage is whatever
  Helsinki-NLP published, so treat anything outside `es`/`fr`/`de` as worth a
  test run rather than assumed.
- **`google`** — every code above, no local model, no GPU. The reliable fallback
  when a local backend has no path for your pair.

### 6.3 Where the two line up

The languages that work end to end today are the intersection: the 11 IndicF5
languages and `en` `es` `fr` `de` `zh` `ja` `ko` `ru` `pt` `it` on Qwen3-TTS.
Urdu is the notable hole — translation is solid, TTS has nothing.

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
└── dubbed_audio.wav        ← synthesised voice track
```

---

## 8. Swapping Components

The modular design makes each step replaceable:

| Step | Current | How to swap |
|---|---|---|
| Transcription | faster-whisper (default) | Set `TRANSCRIBE_BACKEND=whisperx` or `nvidia` in `.env`, or edit `transcribe.py` |
| Translation | IndicTrans2 (default) | Set `TRANSLATION_BACKEND` in `.env`, or add a new backend in `translate.py` |
| TTS | IndicF5 / Qwen3-TTS, routed by target language | Set `TTS_BACKEND` in `.env` or pass `--tts_backend`. To add an engine, subclass `_TTSEngine` in `tts.py` and register it in `_ENGINE_FACTORIES` — chunking, duration lock, timeline placement and the OOM/CPU fallback ladder are shared, so a new engine only implements load/generate/unload |
| LipSync | NVIDIA Maxine | Edit `lipsync.py` — `apply_lipsync_wav2lip()` is provided as an offline alternative |

---

## 9. Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests that shell out to FFmpeg skip themselves automatically when it isn't on
PATH. `tests/test_merge.py` covers the audio mix with tone signals — the bed
at 440 Hz and the dub at 660 Hz — so "did the dubbed voice survive the mix"
is a spectral-energy assertion rather than a listening judgment.

---

## 10. Troubleshooting

| Error | Fix |
|---|---|
| `ffmpeg: command not found` | Install FFmpeg and add to PATH |
| `NVIDIA_API_KEY is not set for nvidia transcription backend` | Only relevant if `TRANSCRIBE_BACKEND=nvidia`; set the key in `.env` or switch to `faster_whisper` |
| Audio > 25 MB (NVIDIA backend only) | Split with `ffmpeg -i v.mp4 -segment_time 600 -f segment seg_%03d.mp4` |
| `No Marian model for <src>-><tgt>` | Helsinki-NLP never published that pair. Use `--backend google`, or `--backend nllb` if both sides are in §6.2's ✓ column |
| Qwen3-TTS `ImportError` | Confirm `Qwen3-TTS/` exists and reinstall with `pip install -e ./Qwen3-TTS` |
| `No module named 'f5_tts'` on an Indic dub | IndicF5 wasn't installed — run the `pip install --no-deps git+...IndicF5.git` step from §2. Do **not** drop `--no-deps` |
| `401` / gated-repo error loading IndicF5 | Accept the terms at https://huggingface.co/ai4bharat/IndicF5 and set `HF_TOKEN` in `.env`, or force `--tts_backend qwen3` (wrong-language audio, but it runs) |
| Indic dub sounds fluent but is not the language | Backend routed to Qwen3-TTS, which has no Indian language. Check the `TTS :` banner line at run start; `auto` should show `indicf5` for those targets |
| Indic dub loses the speaker's voice, warning about a bundled prompt | No transcript for the reference clip. IndicF5 needs `ref_audio` *and* matching `ref_text`; supply both, or let the pipeline derive them from ASR instead of passing `--ref_audio` alone |
| Dubbed audio length mismatch | Use `--no_pad` to trim, or omit to pad silence; `--no_duration_match` disables time-stretching entirely |
| CUDA out of memory | Use `--force_cpu`, or lower `QWEN3_LOCAL_MAX_CHARS_PER_CHUNK` / `TTS_MIN_FREE_VRAM_GB` in `.env` |
| Interrupted a long run | Rerun with `--keep_temp --resume` to pick up from the last checkpoint |
| `--multi_voice` dubs everyone in one voice, warning in logs | `HF_TOKEN` is unset or invalid, or the pyannote model terms weren't accepted — see [Multi-Voice Dubbing](#multi-voice-dubbing) setup steps |
| `--preserve_bgm` / `--separate_audio` fails, Demucs error | Demucs failed to install correctly — check `pip show demucs`; usually a dependency version conflict, reinstalling in a clean venv is the most reliable fix |
| Translation/ASR quantization not applying on a small GPU | Confirm `ASR_COMPUTE_TYPE` / `TRANSLATION_QUANTIZATION` are `auto` (or explicitly set) in `.env`, and that `bitsandbytes` installed correctly (`pip show bitsandbytes`) |
| Dubbed video plays the original language | Fixed in the `asplit` merge fix — update to the latest `merge.py`. Older builds silently emitted source audio when `--preserve_bgm` was set |
