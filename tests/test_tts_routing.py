"""
Tests for TTS backend routing.

What these exist for: Qwen3-TTS supports exactly ten languages and not one of them
is Indian. Handed 'gu' it does not raise -- `_qwen3_language_name` logs "not
explicitly supported", falls back to 'auto', and the model generates confident
nonsense. That is the whole reason routing exists, and it is also why routing
breaking again would be invisible: the pipeline would still finish, still write a
video, and still sound wrong. So the language -> backend map is pinned here.

Nothing here loads a model, touches CUDA, or needs a HuggingFace token.
"""

from __future__ import annotations

import pytest

import config
import tts

# The ten languages Qwen3-TTS actually supports, per _QWEN3_LANG_MAP.
QWEN3_NATIVE = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

SUPPORTED_REQUESTS = {"auto", "qwen3", "indicf5"}


# ---------------------------------------------------------------------------
# The language map itself
# ---------------------------------------------------------------------------

def test_indicf5_covers_the_eleven_languages_it_claims():
    # The model card says 11. Anything else here means a language was dropped or
    # an unsupported one was smuggled in, and both fail silently at runtime.
    assert tts._INDICF5_LANGS == {
        "as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te",
    }


def test_the_two_backends_claim_no_overlapping_languages():
    # Overlap would make 'auto' arbitrary. They are genuinely disjoint today;
    # if that ever changes the tie needs a documented rule, not a set-order coin flip.
    assert tts._INDICF5_LANGS.isdisjoint(QWEN3_NATIVE)


# ---------------------------------------------------------------------------
# auto routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", sorted(tts._INDICF5_LANGS))
def test_every_indic_language_routes_to_indicf5(code):
    assert tts.resolve_tts_backend(code, "auto") == "indicf5"


@pytest.mark.parametrize("code", QWEN3_NATIVE)
def test_every_qwen3_native_language_stays_on_qwen3(code):
    assert tts.resolve_tts_backend(code, "auto") == "qwen3"


@pytest.mark.parametrize("code", ["ur", "fa", "ar", "th", "vi", "sw", "", "zzz"])
def test_languages_neither_model_supports_fall_back_to_qwen3(code):
    # Urdu is the interesting one: it is an Indian language but not one of
    # IndicF5's eleven, so it must NOT be routed there.
    assert tts.resolve_tts_backend(code, "auto") == "qwen3"


def test_urdu_is_not_quietly_treated_as_indicf5_supported():
    assert "ur" not in tts._INDICF5_LANGS


@pytest.mark.parametrize(
    "code",
    ["hi-IN", "hi_IN", "HI", " hi ", "pa_Guru", "pa-Guru-IN", "BN-in", "Gu"],
)
def test_region_and_script_subtags_still_route(code):
    # Callers pass whatever the CLI or GUI handed them. Only the language subtag
    # decides, and case and stray whitespace must not change the answer.
    assert tts.resolve_tts_backend(code, "auto") == "indicf5"


@pytest.mark.parametrize("requested", [None, "", "   ", "auto", "AUTO", "  Auto  "])
def test_a_blank_or_auto_request_all_mean_route_by_language(requested):
    assert tts.resolve_tts_backend("gu", requested) == "indicf5"
    assert tts.resolve_tts_backend("en", requested) == "qwen3"


# ---------------------------------------------------------------------------
# explicit overrides
# ---------------------------------------------------------------------------

def test_an_explicit_backend_wins_even_when_it_is_the_wrong_fit():
    # Forcing a mismatched backend is how you A/B them, so this is deliberate,
    # not a bug to be helpfully corrected.
    assert tts.resolve_tts_backend("gu", "qwen3") == "qwen3"
    assert tts.resolve_tts_backend("en", "indicf5") == "indicf5"


@pytest.mark.parametrize("requested", ["IndicF5", "  indicf5", "QWEN3 "])
def test_an_explicit_backend_is_normalised_before_it_wins(requested):
    assert tts.resolve_tts_backend("gu", requested) == requested.strip().lower()


def test_resolve_does_not_validate_and_get_engine_does():
    # Keeping validation in one place matters: resolve() is called from main.py,
    # studio_gui's mirror, and both synthesis entry points, and duplicating the
    # supported-backend list across them is how the lists drift apart.
    assert tts.resolve_tts_backend("gu", "bogus") == "bogus"
    with pytest.raises(ValueError, match="not supported"):
        tts._get_engine("bogus")


def test_every_resolvable_backend_has_a_registered_engine():
    for code in sorted(tts._INDICF5_LANGS) + QWEN3_NATIVE:
        backend = tts.resolve_tts_backend(code, "auto")
        assert backend in tts._ENGINE_FACTORIES
        assert tts._get_engine(backend).name == backend


# ---------------------------------------------------------------------------
# The routing choice has to agree everywhere it is spelled out
# ---------------------------------------------------------------------------

def test_config_default_routes_rather_than_pinning_one_backend():
    # config.py used to force-clamp this to "qwen3":
    #     if TTS_BACKEND != "qwen3": TTS_BACKEND = "qwen3"
    # which made --tts_backend dead code and every Indic dub silently wrong. Both
    # cases below caught that clamp; asserting only the live value would not,
    # because "qwen3" is a legal value and .env is free to select it.
    import importlib
    import os

    previous = os.environ.get("TTS_BACKEND")

    def _reload_with(value: str) -> str:
        os.environ["TTS_BACKEND"] = value
        return str(importlib.reload(config).TTS_BACKEND)

    try:
        # An unrecognised value must fall back to routing, not to a fixed backend.
        assert _reload_with("not-a-backend") == "auto"
        # And a legal non-Qwen3 choice must survive the module import.
        assert _reload_with("indicf5") == "indicf5"
    finally:
        if previous is None:
            os.environ.pop("TTS_BACKEND", None)
        else:
            os.environ["TTS_BACKEND"] = previous
        importlib.reload(config)

    assert config.TTS_BACKEND in SUPPORTED_REQUESTS


def test_cli_accepts_exactly_the_supported_backends():
    import main

    choices = next(
        a.choices for a in main.build_parser()._actions if a.dest == "tts_backend"
    )
    assert set(choices) == SUPPORTED_REQUESTS
    # Default must stay None so config.TTS_BACKEND (and therefore the .env) is
    # what decides when the flag is omitted.
    assert next(
        a.default for a in main.build_parser()._actions if a.dest == "tts_backend"
    ) is None


def test_studio_gui_routing_hint_matches_the_real_router():
    pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")
    import studio_gui

    # studio_gui duplicates the language set instead of importing tts, to keep
    # torch out of GUI startup. That is a reasonable trade only if the copy is
    # policed, because a stale copy makes the on-screen hint lie about which
    # engine will run.
    assert studio_gui.INDICF5_LANGS == set(tts._INDICF5_LANGS)
    assert set(studio_gui.TTS_BACKEND_VALS) == SUPPORTED_REQUESTS
    assert len(studio_gui.TTS_BACKENDS) == len(studio_gui.TTS_BACKEND_VALS)
    assert studio_gui.state["tts_backend"] == "auto"


def test_studio_gui_hint_agrees_with_resolve_for_every_selectable_target():
    pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")
    import studio_gui

    # state is a module-level singleton, so restore it -- otherwise the default
    # asserted in the test above becomes order-dependent.
    saved = (studio_gui.state["tts_backend"], studio_gui.state["tgt_idx"])
    try:
        for requested in sorted(SUPPORTED_REQUESTS):
            studio_gui.state["tts_backend"] = requested
            for idx, code in enumerate(studio_gui.TARGET_CODES):
                studio_gui.state["tgt_idx"] = idx
                assert studio_gui._resolved_tts_backend() == tts.resolve_tts_backend(
                    code, requested
                ), f"GUI and router disagree on {code!r} with {requested!r}"
    finally:
        studio_gui.state["tts_backend"], studio_gui.state["tgt_idx"] = saved
