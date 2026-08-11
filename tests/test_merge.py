"""
Regression tests for merge.py.

The bug these exist for: in preserve-BGM mode the dubbed track is needed twice,
once as the sidechain key for ducking and once as an input to the mix. A
filter_complex label can only be consumed once, so naming `[dub]` in both places
left an unconnected pad. ffmpeg then auto-connected the first unused input --
the original audio -- and exited 0. The result was a video that looked correct,
reported "ok", and carried the original language.

Tone signals make the assertion exact: bed and original are 440 Hz, the dub is
660 Hz, so "did the dub survive the mix" is a question about spectral energy
rather than about how something sounds.
"""

from __future__ import annotations

import shutil
import subprocess
import wave

import numpy as np
import pytest

from merge import merge_audio

SR = 8000
BED_HZ = 440.0
DUB_HZ = 660.0

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _sine(path, freq, seconds=2.0, gain_db=0.0, rate=SR, channels=1):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={rate}:duration={seconds}",
         "-af", f"volume={gain_db}dB", "-ac", str(channels),
         "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def _silent_video(path, seconds=2.0):
    """A video with a 440 Hz audio track: the 'original language' stand-in."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}:r=10",
         "-f", "lavfi", "-i", f"sine=frequency={BED_HZ}:sample_rate={SR}:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True,
    )
    return path


def _tone_energy(media_path, freq):
    """Energy at *freq* in the middle second of the file's audio."""
    wav = media_path.with_suffix(".probe.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(media_path),
         "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    with wave.open(str(wav), "rb") as w:
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)

    mid = len(samples) // 2
    window = samples[max(0, mid - rate // 2): mid + rate // 2]
    if window.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(window.size, 1 / rate)
    return float(spectrum[int(np.argmin(np.abs(freqs - freq)))])


@pytest.fixture
def clips(tmp_path):
    return {
        "video": _silent_video(tmp_path / "video.mp4"),
        # The real separated bed is near-silent for a talking-head video.
        "bed": _sine(tmp_path / "bed.wav", BED_HZ, gain_db=-60.0, rate=44100, channels=2),
        # Dub is mono at 24 kHz, exactly like Qwen3-TTS output.
        "dub": _sine(tmp_path / "dub.wav", DUB_HZ, rate=24000, channels=1),
    }


def test_preserve_bgm_with_bed_keeps_dubbed_audio(tmp_path, clips):
    """The dubbed voice must survive ducking and reach the output."""
    out = tmp_path / "out.mp4"
    merge_audio(clips["video"], clips["dub"], out,
                background_audio_path=clips["bed"], preserve_bgm=True)

    dub_energy = _tone_energy(out, DUB_HZ)
    original_energy = _tone_energy(out, BED_HZ)
    assert dub_energy > original_energy * 10, (
        f"dubbed tone missing from mix: {DUB_HZ:.0f}Hz={dub_energy:.3g} "
        f"vs {BED_HZ:.0f}Hz={original_energy:.3g}"
    )


def test_preserve_bgm_without_bed_keeps_dubbed_audio(tmp_path, clips):
    """Same guarantee on the no-separation path, which ducks the video's own audio."""
    out = tmp_path / "out_nobed.mp4"
    merge_audio(clips["video"], clips["dub"], out, preserve_bgm=True)

    dub_energy = _tone_energy(out, DUB_HZ)
    assert dub_energy > _tone_energy(out, BED_HZ), "dubbed tone missing from mix"


def test_replace_mode_discards_original_audio(tmp_path, clips):
    """Default mode: only the dub, no trace of the original."""
    out = tmp_path / "out_replace.mp4"
    merge_audio(clips["video"], clips["dub"], out)

    assert _tone_energy(out, DUB_HZ) > _tone_energy(out, BED_HZ) * 10


def test_bed_is_ducked_under_the_dub(tmp_path, tmp_path_factory):
    """A loud bed must still be audible, but attenuated -- not simply dropped."""
    video = _silent_video(tmp_path / "v.mp4")
    bed = _sine(tmp_path / "loud_bed.wav", BED_HZ, gain_db=-6.0, rate=44100, channels=2)
    dub = _sine(tmp_path / "d.wav", DUB_HZ, rate=24000, channels=1)

    out = tmp_path / "ducked.mp4"
    merge_audio(video, dub, out, background_audio_path=bed, preserve_bgm=True)

    bed_energy = _tone_energy(out, BED_HZ)
    dub_energy = _tone_energy(out, DUB_HZ)
    assert dub_energy > bed_energy, "dub should dominate while speaking"
    assert bed_energy > 0, "bed should be ducked, not removed"
