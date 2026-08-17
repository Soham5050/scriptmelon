# Contributing to ScriptMelon

Thanks for taking a look at ScriptMelon. This doc covers how to get set up,
the conventions the codebase follows, and how to submit changes.

## Getting Set Up

```bash
git clone https://github.com/Soham5050/scriptmelon.git
cd scriptmelon
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps git+https://github.com/ai4bharat/IndicF5.git
```

You'll also need FFmpeg on `PATH`, and a `.env` file based on `.env.example`
(never commit your real `.env` — it's already gitignored).

See the README's Project Structure section for what each module
(`transcribe.py`, `translate.py`, `tts.py`, `merge.py`, etc.) is responsible
for before touching the pipeline.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests in `tests/` must pass before a PR is merged. If you add a feature
or fix a bug, add or update a test that covers it — don't rely on manual
verification alone.

## Workflow Rules

- **Never do a full-file replace without seeing the current file first.**
  Read the file, then submit a scoped diff. Large unreviewed rewrites are
  not accepted.
- **Branch naming:** use a short descriptive prefix, e.g.
  `feature/multi-voice-routing`, `fix/temperature-prosody`,
  `chore/repo-hygiene`.
- **Commit messages:** keep them specific to what changed and why — avoid
  vague messages like "fixes" or "update".
- If a change risks breaking existing pipeline behavior (transcription,
  translation, TTS routing, merge), call that out explicitly in the PR
  description, and confirm the regression suite still passes.

## Submitting a Pull Request

1. Fork the repo and create a branch off `main`.
2. Make your changes as a scoped diff — avoid touching unrelated files.
3. Run `python -m pytest tests/ -v` and confirm everything passes.
4. Update the README if your change affects setup, config flags
   (`config.py` / `.env.example`), or the pipeline structure.
5. Open a PR with a clear description of what changed and why. Link any
   related issue.

## Reporting Bugs / Requesting Features

Open a GitHub issue with:
- What you expected vs. what happened
- Steps to reproduce (input video type, target language, flags used)
- Relevant logs or stack trace, and your GPU/VRAM tier if it's a
  performance or quantization-related issue

## Code Style

- Match the existing style in the file you're editing (naming, docstrings,
  logging patterns) rather than introducing a new convention.
- Keep pipeline stages decoupled — e.g. `tts.py` shouldn't reach into
  `translate.py` internals directly; go through the interfaces `main.py`
  already uses to orchestrate stages.

## License

ScriptMelon is licensed under the MIT License (see `LICENSE`). By
contributing, you agree your contributions will be licensed under the same
terms.
