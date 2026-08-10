"""
speakers.py
-----------
Pick one voice-cloning reference clip per detected speaker.

Reference clips are cut from the **separated vocals stem**, never the raw mix.
Cloning from the mix teaches Qwen3-TTS the room — applause, music, ambience —
instead of the person, and that contamination survives into every dubbed line.

A speaker with too little clean audio simply gets no entry in the returned map;
the caller falls back to the default/global voice for that speaker alone. A
two-second bit-part must never take down the run.
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path
from typing import Optional

import config
from audio_utils import slice_audio

log = config.get_logger(__name__)


# Consecutive turns by the same speaker separated by less than this are treated
# as one continuous stretch of speech. ASR splits on sentences, so without this
# almost every candidate would be a single short clause. Natural within-turn
# pauses run 0.3-0.7s, hence the threshold sits above that.
_MAX_JOIN_GAP_SECONDS = 0.75

# A reference this quiet is silence or breath, not speech.
_MIN_REFERENCE_RMS = 220.0

# Reference clips for Qwen3-TTS: mono, 24 kHz.
_REF_SAMPLE_RATE = 24_000
_REF_CHANNELS = 1


def _clip_rms(wav_path: Path) -> float:
    """RMS of a small PCM16 wav. Stdlib only — these clips are a few seconds."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getsampwidth() != 2:
                return _MIN_REFERENCE_RMS  # unknown format; don't reject it
            frames = wf.readframes(wf.getnframes())
        if not frames:
            return 0.0
        samples = array("h")
        samples.frombytes(frames)
        if not samples:
            return 0.0
        total = sum(float(s) * float(s) for s in samples)
        return (total / len(samples)) ** 0.5
    except Exception as exc:
        log.debug("Could not measure RMS for %s: %s", wav_path.name, exc)
        return _MIN_REFERENCE_RMS


def _overlapping_speaker_spans(segments: list[dict]) -> list[tuple[float, float]]:
    """
    Time spans where two *different* speakers are talking at once.

    pyannote's DataFrame carries no overlap flag (columns are only
    segment/label/speaker/start/end), so crosstalk has to be derived: any two
    segments with different labels whose intervals intersect. Clips taken from
    these spans clone a blend of two people and must be avoided.
    """
    spans: list[tuple[float, float]] = []
    ordered = sorted(
        segments, key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0)))
    )
    for i, seg in enumerate(ordered):
        a_start = float(seg.get("start", 0.0))
        a_end = float(seg.get("end", a_start))
        a_spk = str(seg.get("speaker_id") or seg.get("speaker") or "")
        for other in ordered[i + 1 :]:
            b_start = float(other.get("start", 0.0))
            if b_start >= a_end:
                break  # sorted by start: nothing later can overlap
            b_spk = str(other.get("speaker_id") or other.get("speaker") or "")
            if b_spk and a_spk and b_spk != a_spk:
                b_end = float(other.get("end", b_start))
                spans.append((max(a_start, b_start), min(a_end, b_end)))
    return spans


def _overlap_seconds(start: float, end: float, spans: list[tuple[float, float]]) -> float:
    total = 0.0
    for s_start, s_end in spans:
        lo, hi = max(start, s_start), min(end, s_end)
        if hi > lo:
            total += hi - lo
    return total


def _build_runs(segments: list[dict]) -> dict[str, list[dict]]:
    """
    Group each speaker's segments into continuous runs of speech.

    Two turns only merge when the gap between them is short *and* nobody else
    speaks inside it. Merging across another speaker's turn would produce a
    reference window containing two voices — exactly the contamination this
    module exists to avoid.

    Returns speaker_id -> list of {start, end, logprob, n} candidate runs.
    """
    by_speaker: dict[str, list[dict]] = {}
    ordered = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    def _speaker_of(seg: dict) -> str:
        return str(seg.get("speaker_id") or seg.get("speaker") or "spk_00")

    for seg in ordered:
        speaker = _speaker_of(seg)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            continue

        logprob = seg.get("avg_logprob")
        logprob = float(logprob) if isinstance(logprob, (int, float)) else None

        runs = by_speaker.setdefault(speaker, [])
        can_join = False
        if runs and start - runs[-1]["end"] <= _MAX_JOIN_GAP_SECONDS:
            gap_start, gap_end = runs[-1]["end"], start
            can_join = not any(
                _speaker_of(other) != speaker
                and float(other.get("end", 0.0)) > gap_start
                and float(other.get("start", 0.0)) < gap_end
                for other in ordered
            )

        if can_join:
            run = runs[-1]
            run["end"] = max(run["end"], end)
            run["n"] += 1
            if logprob is not None:
                run["logprob_sum"] += logprob
                run["logprob_n"] += 1
        else:
            runs.append(
                {
                    "start": start,
                    "end": end,
                    "n": 1,
                    "logprob_sum": logprob if logprob is not None else 0.0,
                    "logprob_n": 1 if logprob is not None else 0,
                }
            )

    return by_speaker


def _score_run(run: dict, min_seconds: float, max_seconds: float, overlap_spans: list) -> float:
    """Higher is better. Duration first, then confidence, minus crosstalk."""
    duration = run["end"] - run["start"]
    if duration < min_seconds:
        return float("-inf")

    # Prefer runs long enough to fill the reference window; past max_seconds
    # extra length buys nothing since we trim to max_seconds anyway.
    score = min(duration, max_seconds) * 10.0

    if run["logprob_n"]:
        # avg_logprob is negative; closer to 0 is a cleaner transcription and
        # therefore usually cleaner audio.
        score += (run["logprob_sum"] / run["logprob_n"]) * 4.0

    overlap = _overlap_seconds(run["start"], run["end"], overlap_spans)
    if overlap > 0:
        score -= (overlap / max(0.01, duration)) * 40.0

    return score


def _trim_to_window(run: dict, max_seconds: float) -> tuple[float, float]:
    """
    Take up to max_seconds from the middle of a run.

    The middle avoids turn-boundary artefacts: the start of a turn often
    catches the tail of the previous speaker, and the end often trails off.
    """
    start, end = float(run["start"]), float(run["end"])
    duration = end - start
    if duration <= max_seconds:
        return start, end
    excess = duration - max_seconds
    lead = min(excess, excess * 0.5 + 0.10)
    return start + lead, start + lead + max_seconds


def select_speaker_reference_clips(
    segments: list[dict],
    vocals_wav_path: str | Path,
    out_dir: str | Path,
    *,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
    max_speakers: Optional[int] = None,
) -> dict[str, Path]:
    """
    Extract one reference clip per speaker from the separated vocals stem.

    Parameters
    ----------
    segments        : ASR segments carrying `speaker_id` (post-diarization).
    vocals_wav_path : Demucs vocals stem — NOT the original mix.
    out_dir         : Run temp dir; clips are written as `ref_{speaker_id}.wav`.
    min_seconds     : Least usable reference length
                      (default config.DIARIZATION_MIN_SPEAKER_SECONDS).
    max_seconds     : Reference window cap
                      (default config.TTS_SPEAKER_REF_MAX_SECONDS).
    max_speakers    : Keep only this many speakers, ranked by total speech
                      (default config.DIARIZATION_MAX_SPEAKERS). The rest fall
                      back to the default voice.

    Returns
    -------
    speaker_id -> clip path. Speakers without enough clean audio are simply
    absent; callers treat a missing key as "use the default voice".
    """
    vocals_wav_path = Path(vocals_wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if min_seconds is None:
        min_seconds = float(config.DIARIZATION_MIN_SPEAKER_SECONDS)
    if max_seconds is None:
        max_seconds = float(config.TTS_SPEAKER_REF_MAX_SECONDS)
    if max_speakers is None:
        max_speakers = int(config.DIARIZATION_MAX_SPEAKERS)
    min_seconds = max(0.5, float(min_seconds))
    max_seconds = max(min_seconds, float(max_seconds))

    if not segments:
        return {}

    if not vocals_wav_path.exists():
        log.warning(
            "Vocals stem not found at %s; cannot build per-speaker references. "
            "Falling back to a single voice.",
            vocals_wav_path,
        )
        return {}

    overlap_spans = _overlapping_speaker_spans(segments)
    if overlap_spans:
        log.info(
            "Detected %d crosstalk span(s); those regions are avoided when picking references.",
            len(overlap_spans),
        )

    by_speaker = _build_runs(segments)

    # Rank speakers by total speech so the dominant voices win the budget.
    totals = {
        spk: sum(r["end"] - r["start"] for r in runs) for spk, runs in by_speaker.items()
    }
    ranked = sorted(totals, key=lambda s: totals[s], reverse=True)
    if max_speakers > 0 and len(ranked) > max_speakers:
        dropped = ranked[max_speakers:]
        log.info(
            "Detected %d speakers but DIARIZATION_MAX_SPEAKERS=%d; %s will use the default voice.",
            len(ranked),
            max_speakers,
            ", ".join(dropped),
        )
        ranked = ranked[:max_speakers]

    refs: dict[str, Path] = {}
    for speaker in ranked:
        runs = sorted(
            by_speaker[speaker],
            key=lambda r: _score_run(r, min_seconds, max_seconds, overlap_spans),
            reverse=True,
        )

        chosen: Optional[Path] = None
        # Walk the ranked candidates so a near-silent best pick can be replaced.
        for attempt, run in enumerate(runs[:3]):
            if _score_run(run, min_seconds, max_seconds, overlap_spans) == float("-inf"):
                break
            start, end = _trim_to_window(run, max_seconds)
            out_path = out_dir / f"ref_{speaker}.wav"
            try:
                slice_audio(
                    vocals_wav_path,
                    start,
                    end,
                    out_path,
                    sample_rate=_REF_SAMPLE_RATE,
                    channels=_REF_CHANNELS,
                )
            except Exception as exc:
                log.warning("Could not cut reference for %s: %s", speaker, exc)
                continue

            rms = _clip_rms(out_path)
            if rms < _MIN_REFERENCE_RMS and attempt < len(runs[:3]) - 1:
                log.debug(
                    "Reference candidate for %s is near-silent (rms=%.0f); trying next candidate.",
                    speaker,
                    rms,
                )
                continue

            chosen = out_path
            log.info(
                "Reference for %s: %.2fs–%.2fs (%.1fs, rms=%.0f) → %s",
                speaker,
                start,
                end,
                end - start,
                rms,
                out_path.name,
            )
            break

        if chosen is None:
            log.info(
                "Speaker %s has under %.1fs of clean audio (%.1fs total); using the default voice for them.",
                speaker,
                min_seconds,
                totals.get(speaker, 0.0),
            )
            continue

        refs[speaker] = chosen

    if refs:
        log.info("Built %d per-speaker reference clip(s).", len(refs))
    else:
        log.warning("No per-speaker references could be built; falling back to a single voice.")
    return refs
