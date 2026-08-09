"""
studio_gui.py
-------------
ScriptMelon Studio - Dear PyGui edition.

Install: pip install dearpygui
Run:     python studio_gui.py
"""

from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import dearpygui.dearpygui as dpg

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE    = "ScriptMelon Studio"
WIN_W, WIN_H = 1420, 900
PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_PY      = PROJECT_ROOT / "main.py"
OUTPUT_DIR   = PROJECT_ROOT / "output"

# Colours (RGBA 0-255)
C = {
    "bg":           (3,   5,   8,   255),
    "card":         (12,  18,  32,  220),
    "card_border":  (34,  50,  78,  160),
    "field":        (8,   14,  28,  255),
    "accent":       (97,  214, 255, 255),
    "accent2":      (36,  184, 241, 255),
    "accent_dim":   (97,  214, 255, 30),
    "success":      (48,  212, 156, 255),
    "warn":         (242, 184, 93,  255),
    "error":        (251, 125, 148, 255),
    "text":         (244, 247, 251, 255),
    "muted":        (107, 127, 163, 255),
    "purple":       (196, 181, 253, 255),
    "separator":    (34,  50,  78,  120),
    "btn_primary":  (36,  184, 241, 255),
    "btn_stop":     (251, 125, 148, 40),
    "transparent":  (0,   0,   0,   0),
}

LANGUAGES = [
    ("Auto Detect",  "auto"),
    ("English",      "en"),
    ("Hindi",        "hi"),
    ("Marathi",      "mr"),
    ("Bengali",      "bn"),
    ("Telugu",       "te"),
    ("Tamil",        "ta"),
    ("Gujarati",     "gu"),
    ("Kannada",      "kn"),
    ("Malayalam",    "ml"),
    ("Punjabi",      "pa"),
    ("Odia",         "or"),
    ("Assamese",     "as"),
    ("Urdu",         "ur"),
    ("Chinese",      "zh"),
    ("Japanese",     "ja"),
    ("Korean",       "ko"),
    ("German",       "de"),
    ("French",       "fr"),
    ("Russian",      "ru"),
    ("Portuguese",   "pt"),
    ("Spanish",      "es"),
    ("Italian",      "it"),
]

LANG_LABELS  = [f"{lbl} ({code})" if code != "auto" else lbl for lbl, code in LANGUAGES]
LANG_CODES   = [code for _, code in LANGUAGES]
TARGET_LABELS = LANG_LABELS[1:]
TARGET_CODES  = LANG_CODES[1:]

BACKENDS     = ["Auto", "Google", "IndicTrans2", "Marian", "NLLB"]
BACKEND_VALS = ["",     "google", "indictrans2", "marian", "nllb"]

INDIC_LANGS  = {"hi","mr","bn","te","ta","gu","kn","ml","pa","or","as","ur"}
QWEN_LANGS   = {"en","zh","ja","ko","de","fr","ru","pt","es","it"}

PROGRESS_STEPS = {
    "[1/5]": (14,  "Preparing audio"),
    "[2/5]": (34,  "Transcribing speech"),
    "[3/5]": (54,  "Translating script"),
    "[4/5]": (82,  "Synthesizing dubbed audio"),
    "[5/5]": (96,  "Merging final output"),
}

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

state = {
    "input_path":        "",
    "output_path":       "",
    "src_idx":           0,
    "tgt_idx":           1,
    "backend_idx":       1,   # Google
    "tts_backend":       "qwen3",
    "keep_temp":         True,
    "verbose":           True,
    "interactive":       False,
    "no_duration_match": False,
    "preserve_bgm":      True,
    "separate_audio":    True,
    "force_cpu":         False,
    "process":           None,
    "log_queue":         queue.Queue(),
    "runner_thread":     None,
    "progress":          0.0,
    "current_step":      "Idle",
    "status_tone":       "idle",
}


def _src_code()    -> str: return LANG_CODES[state["src_idx"]]
def _target_code() -> str: return TARGET_CODES[state["tgt_idx"]]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_command(parts: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(p) for p in parts)


def _default_output_path() -> str:
    raw_input = state["input_path"].strip()
    if not raw_input:
        return ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(raw_input).stem
    return str(OUTPUT_DIR / f"{stem}_{_target_code()}.mp4")


def _maybe_autofill_output(force: bool = False) -> None:
    current = state["output_path"].strip()
    auto_default = _default_output_path()
    if not auto_default:
        return
    if force or not current or Path(current).parent == OUTPUT_DIR:
        state["output_path"] = auto_default
        if dpg.does_item_exist("output_field"):
            dpg.set_value("output_field", auto_default)


def _build_command(preview: bool = False) -> list[str]:
    py  = sys.executable
    cmd = [py, str(MAIN_PY)]

    inp = state["input_path"] or ("video.mp4" if preview else "")
    if not inp and not preview:
        raise ValueError("No input file selected.")
    cmd += ["--input", inp]

    out = state["output_path"]
    if out:
        cmd += ["--output", out]

    src = _src_code()
    if src and src != "auto":
        cmd += ["--src_lang", src]

    cmd += ["--target_lang", _target_code()]
    cmd += ["--tts_backend", state["tts_backend"]]

    bval = BACKEND_VALS[state["backend_idx"]]
    if bval:
        cmd += ["--backend", bval]

    if state["keep_temp"]:         cmd.append("--keep_temp")
    if state["verbose"]:           cmd.append("--verbose")
    if state["interactive"]:       cmd.append("--interactive")
    if state["no_duration_match"]: cmd.append("--no_duration_match")
    if state["preserve_bgm"]:      cmd.append("--preserve_bgm")
    if state["separate_audio"]:    cmd.append("--separate_audio")
    if state["force_cpu"]:         cmd.append("--force_cpu")

    return cmd


def _refresh_command():
    try:
        parts = _build_command(preview=True)
        text  = _format_command(parts)
    except Exception as e:
        text = f"# Error: {e}"
    if dpg.does_item_exist("cmd_text"):
        dpg.set_value("cmd_text", text)


def _route_hint() -> tuple[str, tuple]:
    tgt = _target_code()
    if tgt in INDIC_LANGS:
        return f"Indic routing | Target: {tgt.upper()} | Validate pronunciation on final output.", C["warn"]
    if tgt in QWEN_LANGS:
        return f"Qwen3 native support | Target: {tgt.upper()} | Highest-confidence lane.", C["accent"]
    return f"Qwen3 multilingual transfer | Target: {tgt.upper()} | Run a short sample first.", C["muted"]


def _update_route_hint():
    hint, color = _route_hint()
    if dpg.does_item_exist("route_hint"):
        dpg.set_value("route_hint", hint)
        dpg.configure_item("route_hint", color=color)


def _set_status(text: str, tone: str = "idle"):
    state["status_tone"] = tone
    colors = {
        "idle":    C["accent"],
        "running": C["accent"],
        "success": C["success"],
        "warn":    C["warn"],
        "error":   C["error"],
    }
    color = colors.get(tone, C["accent"])
    if dpg.does_item_exist("status_label"):
        dpg.set_value("status_label", text)
        dpg.configure_item("status_label", color=color)
    if dpg.does_item_exist("bottom_status"):
        dpg.set_value("bottom_status", text)


def _set_progress(pct: float, label: str):
    state["progress"]     = pct
    state["current_step"] = label
    if dpg.does_item_exist("progress_bar"):
        dpg.set_value("progress_bar", pct / 100.0)
    if dpg.does_item_exist("step_label"):
        dpg.set_value("step_label", label)
    if dpg.does_item_exist("pct_label"):
        dpg.set_value("pct_label", f"{int(pct)}%")

    step_ids = ["dot1","dot2","dot3","dot4","dot5"]
    thresholds = [14, 34, 54, 82, 96]
    for i, sid in enumerate(step_ids):
        if not dpg.does_item_exist(sid):
            continue
        if pct >= thresholds[i]:
            dpg.configure_item(sid, color=C["success"])
        elif pct >= (thresholds[i-1] if i > 0 else 0):
            dpg.configure_item(sid, color=C["accent"])
        else:
            dpg.configure_item(sid, color=C["muted"])


def _append_log(text: str, color: tuple = None):
    if not dpg.does_item_exist("log_area"):
        return
    col = color or C["muted"]
    dpg.add_text(text, color=col, parent="log_area", wrap=800)


def _classify_log(line: str) -> tuple[str, tuple]:
    lo = line.lower().strip()
    for marker, (pct, label) in PROGRESS_STEPS.items():
        if marker in line:
            _set_progress(pct, label)
            _set_status(line.strip(), "running")
            return line.strip(), C["purple"]
    if "finished with exit code 0" in lo or "pipeline complete" in lo or "output is ready" in lo:
        _set_progress(100, "Render complete")
        _set_status("Output ready - check the output folder.", "success")
        return line.strip(), C["success"]
    if "interrupted" in lo or "stop requested" in lo:
        _set_status("Stopping pipeline...", "warn")
        return line.strip(), C["warn"]
    if "pipeline failed" in lo or "error" in lo:
        _set_status(line.strip(), "error")
        return line.strip(), C["error"]
    if "warning" in lo or "warn" in lo:
        return line.strip(), C["warn"]
    if "info" in lo:
        return line.strip(), C["muted"]
    return line.strip(), C["text"]


def _drain_log():
    q = state["log_queue"]
    try:
        while True:
            item = q.get_nowait()
            if item == "__DONE__":
                state["process"] = None
                if dpg.does_item_exist("run_btn"):
                    dpg.configure_item("run_btn", enabled=True, label="Start Dubbing")
                if dpg.does_item_exist("stop_btn"):
                    dpg.configure_item("stop_btn", enabled=False)
                if state["progress"] < 100 and state["status_tone"] == "running":
                    _set_progress(0, "Idle")
                    _set_status("Ready for a fresh dub.", "idle")
            else:
                txt, col = _classify_log(item)
                if txt:
                    _append_log(txt, col)
    except queue.Empty:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────

def on_browse_input():
    def cb(sender, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            state["input_path"] = path
            if dpg.does_item_exist("input_field"):
                dpg.set_value("input_field", path)
            _maybe_autofill_output(force=True)
            _refresh_command()
    dpg.add_file_dialog(
        label="Select Video File",
        callback=cb,
        modal=True,
        width=700, height=500,
        file_count=1,
    )


def on_browse_output():
    def cb(sender, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            state["output_path"] = path
            if dpg.does_item_exist("output_field"):
                dpg.set_value("output_field", path)
            _refresh_command()
    dpg.add_file_dialog(
        label="Save Output As",
        callback=cb,
        modal=True,
        width=700, height=500,
    )


def on_src_change(sender, val):
    state["src_idx"] = LANG_LABELS.index(val) if val in LANG_LABELS else 0
    _update_route_hint()
    _refresh_command()


def on_tgt_change(sender, val):
    state["tgt_idx"] = TARGET_LABELS.index(val) if val in TARGET_LABELS else 0
    _maybe_autofill_output()
    _update_route_hint()
    _refresh_command()


def on_backend_change(sender, val):
    state["backend_idx"] = BACKENDS.index(val) if val in BACKENDS else 0
    _refresh_command()


def on_toggle(sender, val, key):
    state[key] = val
    _refresh_command()


def on_start():
    if state["process"]:
        return
    try:
        cmd = _build_command(preview=False)
    except ValueError as e:
        _append_log(f"ERROR: {e}", C["error"])
        _set_status(str(e), "error")
        return

    _append_log("")
    _append_log("$ " + _format_command(cmd), C["accent"])
    _append_log("")
    _set_progress(6, "Booting pipeline")
    _set_status("Pipeline starting - logs streaming below.", "running")

    if dpg.does_item_exist("run_btn"):
        dpg.configure_item("run_btn", enabled=False, label="Running...")
    if dpg.does_item_exist("stop_btn"):
        dpg.configure_item("stop_btn", enabled=True)

    def runner():
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            state["process"] = proc
            assert proc.stdout
            for line in proc.stdout:
                state["log_queue"].put(line)
            rc = proc.wait()
            state["log_queue"].put(f"\nProcess finished with exit code {rc}.\n")
            state["log_queue"].put("__DONE__")
        except Exception as e:
            state["log_queue"].put(f"\nFailed to start pipeline: {e}\n")
            state["log_queue"].put("__DONE__")

    t = threading.Thread(target=runner, daemon=True)
    state["runner_thread"] = t
    t.start()


def on_stop():
    proc = state.get("process")
    if proc:
        try:
            proc.terminate()
            _append_log("Stop requested - pipeline halting.", C["warn"])
            _set_status("Stopping pipeline...", "warn")
        except Exception:
            pass


def on_copy_cmd():
    try:
        text = _format_command(_build_command(preview=True))
        dpg.set_clipboard_text(text)
        _set_status("Command copied to clipboard.", "idle")
    except Exception as e:
        _set_status(str(e), "error")


def on_open_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(OUTPUT_DIR))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(OUTPUT_DIR)])
    else:
        subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])


def on_clear_log():
    if dpg.does_item_exist("log_area"):
        dpg.delete_item("log_area", children_only=True)
    _append_log("Console cleared.", C["accent"])


def apply_preset(name: str):
    presets = {
        "creator": dict(keep_temp=True, verbose=True, interactive=True,  no_duration_match=False, preserve_bgm=True,  separate_audio=True,  force_cpu=False, backend_idx=1),
        "bgm":     dict(keep_temp=True, verbose=True, interactive=False, no_duration_match=False, preserve_bgm=True,  separate_audio=True,  force_cpu=False, backend_idx=1),
        "debug":   dict(keep_temp=True, verbose=True, interactive=False, no_duration_match=True,  preserve_bgm=False, separate_audio=False, force_cpu=False, backend_idx=1),
    }
    labels = {"creator": "Creator Demo", "bgm": "BGM Heavy", "debug": "Fast Debug"}
    p = presets[name]
    for k, v in p.items():
        state[k] = v

    toggle_map = {
        "keep_temp": "tog_keep", "verbose": "tog_verbose", "interactive": "tog_interactive",
        "no_duration_match": "tog_nodur", "preserve_bgm": "tog_bgm",
        "separate_audio": "tog_sep", "force_cpu": "tog_cpu",
    }
    for k, tag in toggle_map.items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, state[k])

    if dpg.does_item_exist("backend_combo"):
        dpg.set_value("backend_combo", BACKENDS[state["backend_idx"]])

    _set_status(f"Preset loaded: {labels[name]}", "idle")
    _append_log(f"Preset loaded: {labels[name]}", C["success"])
    _refresh_command()


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────

def build_theme():
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        C["bg"])
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         C["card"])
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         (15, 22, 38, 240))
            dpg.add_theme_color(dpg.mvThemeCol_Border,          C["card_border"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         C["field"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (20, 30, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,   (25, 40, 70, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,          (20, 32, 58, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   (36, 184, 241, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    (36, 184, 241, 120))
            dpg.add_theme_color(dpg.mvThemeCol_Header,          (36, 184, 241, 50))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   (36, 184, 241, 80))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,    (36, 184, 241, 110))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,       C["accent"])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,      C["accent"])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,C["accent2"])
            dpg.add_theme_color(dpg.mvThemeCol_Text,            C["text"])
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,    C["muted"])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     C["field"])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   (40, 60, 95, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (60, 90, 140, 200))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,   C["accent"])
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, C["accent2"])
            dpg.add_theme_color(dpg.mvThemeCol_Separator,       C["separator"])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         C["card"])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   C["card"])
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   8)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,    8)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding,   8)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   18, 16)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing,   20)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize,   10)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize,     12)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize,  1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize,  1)
    return theme


def btn_primary_theme():
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (36, 184, 241, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (97, 214, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (20, 140, 190, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,          (2,  13,  22,  255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,  18, 10)
    return t


def btn_stop_theme():
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (251, 125, 148, 40))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (251, 125, 148, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (251, 125, 148, 120))
            dpg.add_theme_color(dpg.mvThemeCol_Text,          C["error"])
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,  18, 10)
    return t


def section_header(label: str, parent=None):
    kwargs = {"parent": parent} if parent else {}
    dpg.add_text(label, color=C["muted"], **kwargs)
    dpg.add_separator(**kwargs)
    dpg.add_spacer(height=4, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# UI Build
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    with dpg.window(label=APP_TITLE, tag="main_win", no_title_bar=True,
                    no_move=True, no_resize=True, no_scrollbar=True):

        # ── TOP BAR ──────────────────────────────────────────────────────────
        with dpg.child_window(height=70, border=True, tag="topbar"):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("ScriptMelon Studio",
                                 color=C["text"])
                    dpg.add_text("AI VIDEO DUBBING PIPELINE | Dear PyGui Edition",
                                 color=C["muted"])
                dpg.add_spacer(width=20)
                with dpg.group():
                    dpg.add_spacer(height=8)
                    dpg.add_text("Ready for a fresh dub",
                                 tag="status_label", color=C["accent"])

        dpg.add_spacer(height=10)

        # ── BODY ─────────────────────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ╔═════════════════════════════════╗
            # ║         LEFT PANEL (420px)      ║
            # ╚═════════════════════════════════╝
            with dpg.child_window(width=430, border=False, tag="left_col"):

                # ── Input/Output ─────────────────────────────────────────────
                with dpg.child_window(height=180, border=True, tag="io_card"):
                    section_header("INPUT / OUTPUT")
                    dpg.add_text("Video file", color=C["muted"])
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="input_field", width=300,
                                           hint="No file selected...",
                                           callback=lambda s,v: state.update(input_path=v) or _maybe_autofill_output() or _refresh_command())
                        dpg.add_button(label="Browse", width=80,
                                       callback=on_browse_input)

                    dpg.add_spacer(height=6)
                    dpg.add_text("Output path (optional)", color=C["muted"])
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="output_field", width=300,
                                           hint="Auto-generated if empty...",
                                           callback=lambda s,v: state.update(output_path=v) or _refresh_command())
                        dpg.add_button(label="Save As", width=80,
                                       callback=on_browse_output)

                dpg.add_spacer(height=8)

                # ── Language ──────────────────────────────────────────────────
                with dpg.child_window(height=200, border=True, tag="lang_card"):
                    section_header("LANGUAGE ROUTING")
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("Source", color=C["muted"])
                            dpg.add_combo(LANG_LABELS, tag="src_combo",
                                         default_value=LANG_LABELS[0], width=190,
                                         callback=on_src_change)
                        dpg.add_spacer(width=8)
                        with dpg.group():
                            dpg.add_text("Target", color=C["muted"])
                            dpg.add_combo(TARGET_LABELS, tag="tgt_combo",
                                         default_value=TARGET_LABELS[1], width=190,
                                         callback=on_tgt_change)

                    dpg.add_spacer(height=6)
                    dpg.add_text("Translation backend", color=C["muted"])
                    dpg.add_combo(BACKENDS, tag="backend_combo", default_value="Google",
                                 width=395, callback=on_backend_change)

                    dpg.add_spacer(height=8)
                    hint, col = _route_hint()
                    dpg.add_text(hint, tag="route_hint", color=col, wrap=400)

                dpg.add_spacer(height=8)

                # ── Options ───────────────────────────────────────────────────
                with dpg.child_window(height=215, border=True, tag="opts_card"):
                    section_header("PIPELINE OPTIONS")
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_checkbox(label="Keep temp files", tag="tog_keep",
                                            default_value=True,
                                            callback=lambda s,v: on_toggle(s,v,"keep_temp"))
                            dpg.add_checkbox(label="Verbose logging", tag="tog_verbose",
                                            default_value=True,
                                            callback=lambda s,v: on_toggle(s,v,"verbose"))
                            dpg.add_checkbox(label="Interactive review", tag="tog_interactive",
                                            default_value=False,
                                            callback=lambda s,v: on_toggle(s,v,"interactive"))
                            dpg.add_checkbox(label="No duration match", tag="tog_nodur",
                                            default_value=False,
                                            callback=lambda s,v: on_toggle(s,v,"no_duration_match"))
                        dpg.add_spacer(width=20)
                        with dpg.group():
                            dpg.add_checkbox(label="Preserve BGM", tag="tog_bgm",
                                            default_value=True,
                                            callback=lambda s,v: on_toggle(s,v,"preserve_bgm"))
                            dpg.add_checkbox(label="Separate audio", tag="tog_sep",
                                            default_value=True,
                                            callback=lambda s,v: on_toggle(s,v,"separate_audio"))
                            dpg.add_checkbox(label="Force CPU", tag="tog_cpu",
                                            default_value=False,
                                            callback=lambda s,v: on_toggle(s,v,"force_cpu"))

                dpg.add_spacer(height=8)

                # ── Presets ───────────────────────────────────────────────────
                with dpg.child_window(height=80, border=True, tag="preset_card"):
                    section_header("QUICK PRESETS")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Creator Demo", width=130,
                                       callback=lambda: apply_preset("creator"))
                        dpg.add_spacer(width=4)
                        dpg.add_button(label="BGM Heavy", width=120,
                                       callback=lambda: apply_preset("bgm"))
                        dpg.add_spacer(width=4)
                        dpg.add_button(label="Fast Debug", width=120,
                                       callback=lambda: apply_preset("debug"))

            dpg.add_spacer(width=10)

            # ╔═════════════════════════════════════════╗
            # ║           RIGHT PANEL (fill)            ║
            # ╚═════════════════════════════════════════╝
            with dpg.child_window(border=False, tag="right_col"):

                # ── Progress ──────────────────────────────────────────────────
                with dpg.child_window(height=130, border=True, tag="progress_card"):
                    section_header("PIPELINE PROGRESS")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Idle - Ready to dub", tag="step_label",
                                    color=C["text"])
                        dpg.add_spacer(width=-60)
                        dpg.add_text("0%", tag="pct_label", color=C["accent"])

                    dpg.add_progress_bar(tag="progress_bar", default_value=0.0,
                                        width=-1, height=8)

                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        step_names = ["Audio","Transcribe","Translate","Synthesize","Merge"]
                        dot_tags   = ["dot1","dot2","dot3","dot4","dot5"]
                        for name, tag in zip(step_names, dot_tags):
                            with dpg.group():
                                dpg.add_text(f"● {name}", tag=tag, color=C["muted"])

                dpg.add_spacer(height=8)

                # ── Log Console ───────────────────────────────────────────────
                with dpg.child_window(height=380, border=True, tag="log_card"):
                    with dpg.group(horizontal=True):
                        section_header("RUN CONSOLE")
                        dpg.add_spacer(width=-90)
                        dpg.add_button(label="Clear Log", width=85,
                                       callback=on_clear_log)

                    with dpg.child_window(tag="log_area", border=False,
                                         autosize_x=True, height=310):
                        dpg.add_text("ScriptMelon Studio - Dear PyGui v2.0",
                                    color=C["accent"])
                        dpg.add_separator()
                        dpg.add_text("GPU: NVIDIA GeForce RTX 5060 Laptop GPU",
                                    color=C["muted"])
                        dpg.add_text("VRAM: 8.0GB | Tier: LOW | bfloat16 mode",
                                    color=C["muted"])
                        dpg.add_text("TTS: Qwen3-TTS-12Hz-1.7B-Base",
                                    color=C["muted"])
                        dpg.add_text("Translation: Google (batched)",
                                    color=C["muted"])
                        dpg.add_separator()
                        dpg.add_text("Pipeline ready. Select a video and hit Start.",
                                    color=C["success"])

                dpg.add_spacer(height=8)

                # ── Command Preview ───────────────────────────────────────────
                with dpg.child_window(height=90, border=True, tag="cmd_card"):
                    with dpg.group(horizontal=True):
                        section_header("COMMAND PREVIEW")
                        dpg.add_spacer(width=-105)
                        dpg.add_button(label="Copy Command", width=100,
                                       callback=on_copy_cmd)
                    dpg.add_input_text(
                        tag="cmd_text",
                        default_value="python main.py --input video.mp4 ...",
                        width=-1, height=42,
                        multiline=True, readonly=True,
                    )

                dpg.add_spacer(height=8)

                # ── Action Buttons ────────────────────────────────────────────
                with dpg.group(horizontal=True):
                    run_btn = dpg.add_button(label="Start Dubbing", tag="run_btn",
                                            width=220, height=44,
                                            callback=on_start)
                    dpg.add_spacer(width=8)
                    stop_btn = dpg.add_button(label="Stop", tag="stop_btn",
                                             width=110, height=44,
                                             enabled=False,
                                             callback=on_stop)
                    dpg.add_spacer(width=8)
                    dpg.add_button(label="Open Output", width=150, height=44,
                                  callback=on_open_output)

                    # Apply themes to buttons
                    dpg.bind_item_theme(run_btn,  btn_primary_theme())
                    dpg.bind_item_theme(stop_btn, btn_stop_theme())

        dpg.add_spacer(height=8)

        # ── BOTTOM BAR ───────────────────────────────────────────────────────
        with dpg.child_window(height=38, border=True, tag="bottom_bar"):
            with dpg.group(horizontal=True):
                dpg.add_text("Ready for a fresh dub.", tag="bottom_status",
                            color=C["muted"])
                dpg.add_spacer(width=-200)
                dpg.add_text("Built by Soham | ScriptMelon",
                            color=C["muted"])


# ─────────────────────────────────────────────────────────────────────────────
# Render Loop
# ─────────────────────────────────────────────────────────────────────────────

def frame_callback():
    """Called every frame — drain log queue and sync window size."""
    _drain_log()
    vp_w = dpg.get_viewport_client_width()
    vp_h = dpg.get_viewport_client_height()
    dpg.set_item_width("main_win",  vp_w)
    dpg.set_item_height("main_win", vp_h)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dpg.create_context()

    theme = build_theme()
    dpg.bind_theme(theme)

    build_ui()

    dpg.create_viewport(
        title=APP_TITLE,
        width=WIN_W,
        height=WIN_H,
        min_width=1100,
        min_height=700,
        small_icon="",
        large_icon="",
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()

    dpg.set_primary_window("main_win", True)

    _refresh_command()

    while dpg.is_dearpygui_running():
        frame_callback()
        dpg.render_dearpygui_frame()

    # Cleanup
    proc = state.get("process")
    if proc:
        try: proc.terminate()
        except Exception: pass

    dpg.destroy_context()


if __name__ == "__main__":
    main()
