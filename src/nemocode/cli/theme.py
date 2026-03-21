# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Shared UI theme and keybinding helpers for REPL, renderer, and TUI."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ThemePalette:
    name: str
    accent_hex: str
    accent_rich: str
    app_bg: str
    panel_bg: str
    surface_bg: str
    input_bg: str
    status_bg: str
    status_fg: str
    text_hex: str
    muted_hex: str


_THEMES = {
    "nvidia-dark": ThemePalette(
        name="nvidia-dark",
        accent_hex="#76B900",
        accent_rich="bright_green",
        app_bg="#0b0f0c",
        panel_bg="#0d110d",
        surface_bg="#11170f",
        input_bg="#0c100c",
        status_bg="#11170d",
        status_fg="#d6dfc4",
        text_hex="#f1f5e9",
        muted_hex="#a8b09d",
    ),
    "nvidia-light": ThemePalette(
        name="nvidia-light",
        accent_hex="#2D7A00",
        accent_rich="green4",
        app_bg="#f5f8f1",
        panel_bg="#ffffff",
        surface_bg="#edf3e6",
        input_bg="#ffffff",
        status_bg="#dfe8d4",
        status_fg="#233018",
        text_hex="#1f2b18",
        muted_hex="#4f5c46",
    ),
    "minimal": ThemePalette(
        name="minimal",
        accent_hex="#2F6F8F",
        accent_rich="cyan",
        app_bg="#111315",
        panel_bg="#15191c",
        surface_bg="#1b2024",
        input_bg="#101417",
        status_bg="#15191d",
        status_fg="#d6dde3",
        text_hex="#edf2f5",
        muted_hex="#97a5ae",
    ),
    "high-contrast": ThemePalette(
        name="high-contrast",
        accent_hex="#FFFF00",
        accent_rich="bright_yellow",
        app_bg="#000000",
        panel_bg="#050505",
        surface_bg="#111111",
        input_bg="#000000",
        status_bg="#111111",
        status_fg="#FFFFFF",
        text_hex="#FFFFFF",
        muted_hex="#D0D0D0",
    ),
}


def available_themes() -> tuple[str, ...]:
    return tuple(_THEMES)


def detect_terminal_background() -> str:
    """Best-effort terminal background detection for auto theme selection."""
    env_hint = os.environ.get("NEMOCODE_BACKGROUND", "").strip().lower()
    if env_hint in {"light", "dark"}:
        return env_hint

    colorfgbg = os.environ.get("COLORFGBG", "").strip()
    if colorfgbg:
        try:
            bg = int(colorfgbg.split(";")[-1])
        except ValueError:
            bg = -1
        if bg >= 0:
            return "light" if bg >= 7 else "dark"

    return "dark"


def resolve_theme_name(name: str | None) -> str:
    pref = (name or "nvidia-dark").strip().lower()
    if pref == "auto":
        return "nvidia-light" if detect_terminal_background() == "light" else "nvidia-dark"
    return pref if pref in _THEMES else "nvidia-dark"


def get_theme(name: str | None) -> ThemePalette:
    return _THEMES[resolve_theme_name(name)]


def canonical_key_spec(spec: str) -> str:
    normalized = spec.strip().lower()
    normalized = normalized.replace("_", "+").replace("-", "+")
    normalized = normalized.replace("control+", "ctrl+").replace("ctl+", "ctrl+")
    normalized = normalized.replace("esc", "escape").replace("return", "enter")
    return normalized


def prompt_toolkit_binding_parts(spec: str) -> tuple[str, ...]:
    key = canonical_key_spec(spec)
    if key in {"tab", "escape", "enter"}:
        return (key,)
    if key.startswith("ctrl+") and len(key) == 6:
        return (f"c-{key[-1]}",)
    if key.startswith("alt+") and len(key) == 5:
        return ("escape", key[-1])
    raise ValueError(f"Unsupported prompt_toolkit keybinding: {spec}")


def format_key_label(spec: str) -> str:
    parts = [part.strip() for part in canonical_key_spec(spec).split("+") if part.strip()]
    formatted = []
    for part in parts:
        if part == "ctrl":
            formatted.append("Ctrl")
        elif part == "alt":
            formatted.append("Alt")
        elif part == "cmd":
            formatted.append("Cmd")
        elif part == "escape":
            formatted.append("Escape")
        elif len(part) == 1:
            formatted.append(part.upper())
        else:
            formatted.append(part.capitalize())
    return "+".join(formatted) if formatted else spec


def build_tui_stylesheet(theme: ThemePalette) -> str:
    return f"""\
/* ── Global ────────────────────────────────────────────────── */

Screen {{
    layout: vertical;
    background: {theme.app_bg};
    color: {theme.text_hex};
}}

/* ── Chat history ──────────────────────────────────────────── */

#chat-scroll {{
    height: 1fr;
    border: solid {theme.accent_hex};
    border-title-color: {theme.accent_hex};
    background: {theme.panel_bg};
    padding: 0 1;
}}

.chat-user {{
    margin: 1 0 0 0;
    padding: 0 1;
    color: {theme.text_hex};
    background: {theme.surface_bg};
}}

.chat-user .label {{
    color: {theme.accent_hex};
    text-style: bold;
}}

.chat-assistant {{
    margin: 0 0 0 0;
    padding: 0 1;
    color: {theme.text_hex};
}}

.chat-thinking {{
    color: {theme.muted_hex};
    text-style: italic;
    padding: 0 1;
    margin: 0 0 0 2;
}}

.chat-phase {{
    color: {theme.accent_hex};
    text-style: bold;
    padding: 0 1;
    margin: 1 0 0 0;
}}

.chat-error {{
    color: $error;
    text-style: bold;
    padding: 0 1;
    margin: 0 0 0 0;
}}

.chat-system {{
    color: {theme.muted_hex};
    text-style: italic;
    padding: 0 1;
    margin: 0 0 0 0;
}}

/* ── Tool panel ────────────────────────────────────────────── */

#tool-panel {{
    height: auto;
    max-height: 12;
    border: solid {theme.accent_hex};
    border-title-color: {theme.accent_hex};
    background: {theme.surface_bg};
    padding: 0 1;
    display: none;
}}

#tool-panel.visible {{
    display: block;
}}

.tool-call {{
    color: {theme.muted_hex};
    padding: 0 0 0 1;
}}

.tool-result-ok {{
    color: {theme.accent_hex};
    padding: 0 0 0 2;
}}

.tool-result-error {{
    color: $error;
    padding: 0 0 0 2;
}}

/* ── Input area ────────────────────────────────────────────── */

#input-row {{
    height: auto;
    max-height: 8;
    min-height: 3;
    dock: bottom;
}}

#mode-label {{
    width: 10;
    height: 3;
    content-align: center middle;
    text-style: bold;
    padding: 0 1;
}}

#mode-label.mode-code {{
    color: {theme.accent_hex};
    background: {theme.surface_bg};
}}

#mode-label.mode-plan {{
    color: $warning;
    background: {theme.surface_bg};
}}

#mode-label.mode-auto {{
    color: $error;
    background: {theme.surface_bg};
}}

#chat-input {{
    height: auto;
    min-height: 3;
    max-height: 8;
    border: tall {theme.accent_hex};
    background: {theme.input_bg};
    color: {theme.text_hex};
}}

#chat-input:focus {{
    border: tall {theme.accent_hex};
}}

/* ── Status bar ────────────────────────────────────────────── */

#status-bar {{
    dock: bottom;
    height: 1;
    background: {theme.status_bg};
    color: {theme.status_fg};
    padding: 0 1;
}}

#status-bar .status-mode {{
    text-style: bold;
}}

/* ── Streaming indicator ───────────────────────────────────── */

.streaming-indicator {{
    color: {theme.accent_hex};
    text-style: bold italic;
    padding: 0 1;
}}
"""
