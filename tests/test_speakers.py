"""
Tests for speakers.py reference selection.

What these exist for: a voice-cloning reference is a clip *plus* the transcript
of that clip. Reference-conditioned models of the F5 family use the transcript to
work out what the audio says before infilling a new line in the same voice, so a
transcript that does not match the audio is actively harmful — worse than none.

Two ways to get that wrong, both covered here:
  1. Cutting the clip at an arbitrary timestamp, leaving half-words at the edges
     while the transcript still claims the whole sentence.
  2. Quoting the *translated* text, which pairs Gujarati words with English audio.

The clips are 440 Hz tone, not speech. Nothing here judges audio quality; the
assertions are about which time window was cut and which text came back with it.
"""

from __future__ import annotations

import math
import shutil
import struct
import wave

import pytest

from speakers import (
    SpeakerRef,
    _segment_text,
    _select_reference_window,
    select_speaker_reference_clips,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)

SR = 8000


def _tone_wav(path, seconds: float, hz: float = 440.0, amplitude: float = 12000.0):
    """A loud mono tone — loud enough to clear speakers.py's near-silence floor."""
    frames = bytearray()
    for n in range(int(SR * seconds)):
        value = int(amplitude * math.sin(2.0 * math.pi * hz * n / SR))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(bytes(frames))
    return path


def _seg(start, end, text, speaker="spk_00", **extra):
    seg = {"start": start, "end": end, "text": text, "speaker_id": speaker}
    seg.update(extra)
    return seg


def _run(segments):
    return {
        "start": float(segments[0]["start"]),
        "end": float(segments[-1]["end"]),
        "n": len(segments),
        "logprob_sum": 0.0,
        "logprob_n": 0,
        "segments": list(segments),
    }


# --- _segment_text: which text belongs to the reference audio ----------------


def test_source_text_wins_over_translation():
    """The clip is original speech, so the transcript must be the source text."""
    seg = _seg(0.0, 2.0, "નમસ્તે મિત્રો", source_text="hello friends")
    assert _segment_text(seg) == "hello friends"


def test_absent_source_text_falls_back_to_text():
    """Raw ASR segments have no source_text key; their `text` *is* the source."""
    assert _segment_text(_seg(0.0, 2.0, "hello friends")) == "hello friends"


def test_empty_source_text_is_honoured_not_overridden():
    """
    Present-but-empty means the pipeline knows it cannot supply source text
    (timings estimated from a full-text translation). Falling back to `text`
    there would quote the target language.
    """
    seg = _seg(0.0, 2.0, "નમસ્તે મિત્રો", source_text="")
    assert _segment_text(seg) == ""


# --- _select_reference_window: cut on segment boundaries ---------------------


def test_short_run_returns_every_segment_and_its_full_text():
    segments = [_seg(1.0, 2.5, "one two"), _seg(2.5, 4.0, "three four")]
    start, end, text = _select_reference_window(_run(segments), max_seconds=6.0)
    assert (start, end) == (1.0, 4.0)
    assert text == "one two three four"


def test_window_never_cuts_mid_segment():
    """
    Four 2.5s segments, 6s budget. Any honest window is a whole-segment
    boundary, so both edges must coincide with a segment edge -- never 6.0s
    measured off the run start.
    """
    segments = [
        _seg(0.0, 2.5, "alpha"),
        _seg(2.5, 5.0, "bravo"),
        _seg(5.0, 7.5, "charlie"),
        _seg(7.5, 10.0, "delta"),
    ]
    start, end, text = _select_reference_window(_run(segments), max_seconds=6.0)

    edges = {0.0, 2.5, 5.0, 7.5, 10.0}
    assert start in edges and end in edges
    assert end - start <= 6.0
    # Two segments fit (5.0s); a third would be 7.5s and overrun.
    assert end - start == pytest.approx(5.0)

    kept = [s for s in segments if s["start"] >= start and s["end"] <= end]
    assert text == " ".join(s["text"] for s in kept)


def test_window_prefers_the_middle_of_a_long_run():
    """Turn edges catch the previous speaker's tail, so favour the middle."""
    segments = [_seg(float(i) * 2.0, float(i) * 2.0 + 2.0, f"s{i}") for i in range(6)]
    start, end, _ = _select_reference_window(_run(segments), max_seconds=4.0)
    run_middle = 6.0
    assert abs((start + end) / 2.0 - run_middle) <= 1.0


def test_oversized_single_segment_yields_no_transcript():
    """
    One 12s segment against a 6s budget: the clip has to be cut mid-sentence.
    A clip is still useful as a voice sample, but no transcript honestly
    describes it, so none is returned rather than an over-claiming one.
    """
    segments = [_seg(0.0, 12.0, "a very long uninterrupted sentence")]
    start, end, text = _select_reference_window(_run(segments), max_seconds=6.0)
    assert end - start == pytest.approx(6.0)
    assert text == ""


def test_transcript_covers_exactly_the_audio_that_was_cut():
    """The invariant that matters: text and window describe the same speech."""
    segments = [
        _seg(0.0, 2.0, "first"),
        _seg(2.0, 4.0, "second"),
        _seg(4.0, 6.0, "third"),
    ]
    start, end, text = _select_reference_window(_run(segments), max_seconds=4.0)
    inside = [s for s in segments if s["start"] >= start and s["end"] <= end]
    outside = [s for s in segments if s not in inside]
    for seg in inside:
        assert seg["text"] in text
    for seg in outside:
        assert seg["text"] not in text


# --- end to end through ffmpeg ----------------------------------------------


def test_returns_speaker_ref_with_clip_and_transcript(tmp_path):
    vocals = _tone_wav(tmp_path / "vocals.wav", seconds=12.0)
    segments = [
        _seg(0.5, 3.0, "કેમ છો", source_text="how are you", speaker="spk_00"),
        _seg(3.0, 5.5, "મજામાં", source_text="doing well", speaker="spk_00"),
        _seg(6.0, 9.5, "સરસ", source_text="very nice", speaker="spk_01"),
    ]

    refs = select_speaker_reference_clips(
        segments, vocals, tmp_path / "refs", min_seconds=1.0, max_seconds=6.0
    )

    assert set(refs) == {"spk_00", "spk_01"}
    for speaker, ref in refs.items():
        assert isinstance(ref, SpeakerRef)
        assert ref.audio.exists() and ref.audio.stat().st_size > 0

    # Source text, never the Gujarati translation.
    assert refs["spk_00"].text == "how are you doing well"
    assert refs["spk_01"].text == "very nice"


def test_translated_text_never_reaches_the_transcript(tmp_path):
    """The regression guard: ref audio is English, so ref text must be too."""
    vocals = _tone_wav(tmp_path / "vocals.wav", seconds=8.0)
    segments = [_seg(0.5, 4.0, "નમસ્તે મિત્રો", source_text="hello friends")]

    refs = select_speaker_reference_clips(
        segments, vocals, tmp_path / "refs", min_seconds=1.0, max_seconds=6.0
    )

    assert refs["spk_00"].text == "hello friends"
    assert "નમસ્તે" not in refs["spk_00"].text


def test_missing_vocals_stem_degrades_to_no_references(tmp_path):
    """A missing stem must fall back to a single voice, not raise."""
    refs = select_speaker_reference_clips(
        [_seg(0.0, 4.0, "hello")], tmp_path / "absent.wav", tmp_path / "refs"
    )
    assert refs == {}
