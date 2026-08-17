# Security Policy

## Supported Versions

ScriptMelon is developed on `main` with no versioned releases yet. Security
fixes are applied to `main` only — there are no maintained older branches.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Instead, report it privately through GitHub's Security Advisories:

1. Go to the [Security tab](https://github.com/Soham5050/scriptmelon/security) of this repo.
2. Click **"Report a vulnerability"**.
3. Include a description of the issue, steps to reproduce, and the
   potential impact.

You should get an acknowledgment within a few days. This is a solo-maintained
project, so response and fix timelines depend on severity and availability —
critical issues (e.g. anything that could leak credentials or allow remote
code execution) will be prioritized.

## Scope & Known Sensitive Areas

ScriptMelon runs entirely locally, but a few areas are worth flagging for
anyone auditing or reporting issues:

- **`.env` / HuggingFace token (`HF_TOKEN`):** Used for gated model access
  (diarization, IndicF5). `.env` is gitignored and must never be committed.
  If you find a code path that logs, prints, or otherwise leaks `HF_TOKEN`
  or other `.env` values, treat it as a vulnerability and report it
  privately.
- **Subprocess / FFmpeg calls:** The pipeline shells out to FFmpeg
  (`audio_utils.py`, `merge.py`). Any path where untrusted filenames or
  user input could enable command injection is in scope.
- **Model/config loading (`config.py`, `tts.py`, `translate.py`):**
  Deserialization or arbitrary file-path handling that could load unsafe
  content is in scope.
- **Cloud transcription backend (NVIDIA NIM):** If you find an issue with
  how credentials or audio data are handled when using the cloud backend
  instead of local ASR, that's in scope too.

Out of scope: vulnerabilities in third-party dependencies themselves
(report those upstream — e.g. to `faster-whisper`, `IndicF5`, `transformers`)
unless ScriptMelon's usage of them introduces the issue.

## Disclosure

Once a reported vulnerability is fixed, it will be disclosed via a GitHub
Security Advisory on this repo, crediting the reporter unless they request
otherwise.
