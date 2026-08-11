"""
Tests for the tts.py engine seam.

What these exist for: the synthesis orchestration — text chunking, the duration
lock, timeline mixing, the silent-segment retry, the OOM/CPU fallback ladder —
must stay single-sourced across TTS backends. A synthesis path per backend means
every fix has to be made twice and will eventually be made once; merge.py's
preserve_bgm bug survived as long as it did for exactly that reason.

So these drive the real orchestration with a fake engine that has nothing to do
with Qwen3. It emits a 440 Hz tone of deliberately *wrong* duration, and the
assertions are that the generic machinery still did its job: one call per
segment, the engine's own language name passed through untouched, per-speaker
references honoured, and every output segment corrected to its target duration.

Nothing here loads a model, touches CUDA, or needs a HuggingFace token.
"""

from __future__ import annotations

import math
import shutil
import struct
import wave

import pytest

import tts

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

SR = 24000
TONE_HZ = 440.0
# The fake always emits this, regardless of how long the segment asked for --
# so any output matching the target proves the duration lock ran.
FAKE_SECONDS = 1.0


def _tone_bytes(seconds: float, hz: float = TONE_HZ, amplitude: float = 12000.0) -> bytes:
    """A loud tone as a complete WAV byte string, as an engine would return."""
    import io

    frames = bytearray()
    for n in range(int(SR * seconds)):
        frames += struct.pack("<h", int(amplitude * math.sin(2.0 * math.pi * hz * n / SR)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(bytes(frames))
    return buffer.getvalue()


def _tone_file(path, seconds: float):
    path.write_bytes(_tone_bytes(seconds))
    return path


class FakeEngine(tts._TTSEngine):
    """
    A backend with no Qwen3 in it at all.

    `language_name` returns a token no Qwen3 map would ever produce, so if the
    orchestration were still reaching for _qwen3_language_name the assertions
    below would catch it.
    """

    name = "fake"

    def __init__(self):
        self.calls: list[dict] = []
        self.loads: list[bool] = []
        self.unloads = 0

    def language_name(self, target_lang):
        return f"fake-lang:{target_lang}"

    def load(self, use_cpu: bool = False):
        self.loads.append(use_cpu)
        # "cpu" keeps the VRAM probes and GPU cleanup out of the way.
        return object(), "cpu"

    def generate(self, model, text, language, ref_audio, ref_text, device="cuda"):
        self.calls.append(
            {
                "text": text,
                "language": language,
                "ref_audio": str(ref_audio),
                "ref_text": ref_text,
                "device": device,
            }
        )
        return _tone_bytes(FAKE_SECONDS), SR

    def unload(self):
        self.unloads += 1


@pytest.fixture
def ref_wav(tmp_path):
    """Short enough that _prepare_reference_audio passes it through untouched."""
    return _tone_file(tmp_path / "ref_voice.wav", seconds=2.0)


def _segments():
    return [
        {"start": 0.0, "end": 2.0, "text": "first line of speech", "speaker_id": "spk_00"},
        {"start": 2.0, "end": 5.0, "text": "second line of speech", "speaker_id": "spk_01"},
    ]


# --- the seam holds --------------------------------------------------------


def test_timed_segments_runs_entirely_through_the_engine(tmp_path, ref_wav):
    engine = FakeEngine()

    out = tts._synthesize_timed_segments(
        engine,
        _segments(),
        tmp_path / "out",
        output_stem="dub",
        target_lang="gu",
        ref_audio=ref_wav,
        ref_text="",
    )

    assert out.exists() and out.stat().st_size > 0
    assert engine.loads == [False], "model should be loaded once, on GPU by default"
    assert len(engine.calls) == 2, "one generate call per segment"


def test_engine_language_name_is_passed_through_untouched(tmp_path, ref_wav):
    """
    The routing fix is worthless if the orchestration re-maps the language
    itself. Whatever the engine says the language is, is what it receives.
    """
    engine = FakeEngine()

    tts._synthesize_timed_segments(
        engine,
        _segments(),
        tmp_path / "out",
        output_stem="dub",
        target_lang="gu",
        ref_audio=ref_wav,
        ref_text="",
    )

    assert {c["language"] for c in engine.calls} == {"fake-lang:gu"}
    # In particular, not Qwen3's "auto" fallback for an unsupported language.
    assert "auto" not in {c["language"] for c in engine.calls}


def test_duration_lock_corrects_the_engine_output(tmp_path, ref_wav):
    """
    The fake always emits 1.0s. Segments ask for 2.0s and 3.0s, so matching
    output durations can only come from the generic duration lock.
    """
    engine = FakeEngine()
    out_dir = tmp_path / "out"

    tts._synthesize_timed_segments(
        engine,
        _segments(),
        out_dir,
        output_stem="dub",
        target_lang="en",
        ref_audio=ref_wav,
        ref_text="",
    )

    matched = sorted((out_dir / "tts_segments").glob("*_matched.wav"))
    assert len(matched) == 2, "both segments should be duration-matched"
    for wav, expected in zip(matched, (2.0, 3.0)):
        assert tts._wav_duration(wav) == pytest.approx(expected, abs=0.08)


def test_no_duration_match_leaves_engine_output_alone(tmp_path, ref_wav):
    engine = FakeEngine()
    out_dir = tmp_path / "out"

    tts._synthesize_timed_segments(
        engine,
        _segments(),
        out_dir,
        output_stem="dub",
        target_lang="en",
        ref_audio=ref_wav,
        ref_text="",
        no_duration_match=True,
    )

    assert not list((out_dir / "tts_segments").glob("*_matched.wav"))


def test_per_speaker_references_survive_the_seam(tmp_path, ref_wav):
    """--multi_voice must keep working: each speaker's own clip reaches generate."""
    from speakers import SpeakerRef

    spk0 = _tone_file(tmp_path / "ref_spk_00.wav", seconds=2.0)
    spk1 = _tone_file(tmp_path / "ref_spk_01.wav", seconds=2.0)
    engine = FakeEngine()

    tts._synthesize_timed_segments(
        engine,
        _segments(),
        tmp_path / "out",
        output_stem="dub",
        target_lang="gu",
        ref_audio=ref_wav,
        ref_text="",
        speaker_refs={
            "spk_00": SpeakerRef(audio=spk0, text="hello there"),
            "spk_01": SpeakerRef(audio=spk1, text="good evening"),
        },
    )

    used = [c["ref_audio"] for c in engine.calls]
    assert used[0].endswith("ref_spk_00.wav")
    assert used[1].endswith("ref_spk_01.wav")


def test_bare_path_speaker_refs_still_accepted(tmp_path, ref_wav):
    """Callers that build references themselves keep working."""
    spk0 = _tone_file(tmp_path / "ref_spk_00.wav", seconds=2.0)
    engine = FakeEngine()

    tts._synthesize_timed_segments(
        engine,
        _segments(),
        tmp_path / "out",
        output_stem="dub",
        target_lang="gu",
        ref_audio=ref_wav,
        ref_text="",
        speaker_refs={"spk_00": spk0},
    )

    assert engine.calls[0]["ref_audio"].endswith("ref_spk_00.wav")
    # spk_01 has no clip of its own and falls back to the global reference.
    assert engine.calls[1]["ref_audio"].endswith("ref_voice.wav")


def test_unknown_speaker_falls_back_to_the_global_reference(tmp_path, ref_wav):
    engine = FakeEngine()

    tts._synthesize_timed_segments(
        engine,
        [{"start": 0.0, "end": 2.0, "text": "a line of speech", "speaker_id": "nobody"}],
        tmp_path / "out",
        output_stem="dub",
        target_lang="gu",
        ref_audio=ref_wav,
        ref_text="",
    )

    assert engine.calls[0]["ref_audio"].endswith("ref_voice.wav")


def test_overlapping_segments_take_the_timeline_mix_path(tmp_path, ref_wav):
    """
    Two people talking at once cannot be sequential concat -- that pushes every
    later line late. Overlap must still be detected below the engine seam.
    """
    engine = FakeEngine()
    overlapping = [
        {"start": 0.0, "end": 3.0, "text": "first speaker talking", "speaker_id": "spk_00"},
        {"start": 1.5, "end": 4.0, "text": "second speaker interrupts", "speaker_id": "spk_01"},
    ]

    out = tts._synthesize_timed_segments(
        engine,
        overlapping,
        tmp_path / "out",
        output_stem="dub",
        target_lang="gu",
        ref_audio=ref_wav,
        ref_text="",
    )

    # Mixed on an absolute timeline: total length is the last end time, not the
    # sum of the two segments (which concat would give as 5.5s).
    assert tts._wav_duration(out) == pytest.approx(4.0, abs=0.2)


def test_chunked_path_runs_through_the_engine(tmp_path, ref_wav):
    """The non-timed path shares the same seam."""
    engine = FakeEngine()

    out = tts._synthesize_chunked(
        engine,
        "This is a longer passage of narration that will be split into several "
        "chunks by the generic chunker before it ever reaches the backend.",
        tmp_path / "out",
        output_stem="dub",
        target_lang="hi",
        ref_audio=ref_wav,
        ref_text="",
    )

    assert out.exists() and out.stat().st_size > 0
    assert len(engine.calls) > 1, "long text should be chunked"
    assert {c["language"] for c in engine.calls} == {"fake-lang:hi"}


# --- registry and unloading -----------------------------------------------


def test_unknown_backend_is_rejected_with_the_supported_list():
    with pytest.raises(ValueError, match="not supported"):
        tts._get_engine("does-not-exist")


def test_get_engine_caches_the_instance():
    assert tts._get_engine("qwen3") is tts._get_engine("qwen3")


def test_unload_reaches_every_registered_backend(monkeypatch):
    """
    Sequential load/unload is the VRAM strategy. A run that routed to two
    backends has to release both, so unload must not stop at the last one used.
    """
    engine = FakeEngine()
    monkeypatch.setitem(tts._ENGINE_FACTORIES, "fake", FakeEngine)
    monkeypatch.setitem(tts._ENGINE_CACHE, "fake", engine)

    tts.unload_tts_model()

    assert engine.unloads == 1


def test_unload_of_one_backend_does_not_stop_the_others(monkeypatch):
    class Exploding(FakeEngine):
        name = "exploding"

        def unload(self):
            raise RuntimeError("driver went away")

    good = FakeEngine()
    monkeypatch.setitem(tts._ENGINE_FACTORIES, "exploding", Exploding)
    monkeypatch.setitem(tts._ENGINE_CACHE, "exploding", Exploding())
    monkeypatch.setitem(tts._ENGINE_FACTORIES, "fake", FakeEngine)
    monkeypatch.setitem(tts._ENGINE_CACHE, "fake", good)

    tts.unload_tts_model()  # must not raise

    assert good.unloads == 1
