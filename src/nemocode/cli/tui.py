# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Full-screen TUI for NeMoCode — Textual-based alternative to the REPL.

Provides a rich terminal interface with:
  - Scrollable chat history with Markdown rendering
  - Multi-line input area with mode indicator
  - Status bar (context %, mode, endpoint, git branch, token usage)
  - Collapsible tool execution panel
  - Streaming text output
  - Slash command support (/help, /think, /compact, /reset, /undo, /cost,
    /endpoint, /formation, /mode, /quit)
  - Mode cycling (code/plan/auto) via Tab
  - Ctrl+C cancels current turn

Launch: ``nemo code --tui`` or ``python -m nemocode.cli.tui``
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
from textual.reactive import reactive
from textual.widgets import Footer, Static, TextArea

from nemocode import __version__
from nemocode.cli.render import (
    format_tool_call,
    summarize_delegate_result,
    tool_result_has_embedded_error,
)
from nemocode.config.agents import resolve_agent_reference
from nemocode.config import load_config
from nemocode.config.schema import AgentMode, NeMoCodeConfig
from nemocode.core.context import ContextManager
from nemocode.core.metrics import MetricsCollector, RequestMetrics
from nemocode.core.scheduler import AgentEvent
from nemocode.workflows.code_agent import CodeAgent

logger = logging.getLogger(__name__)

# NVIDIA green (#76B900) used throughout the TUI
NV_GREEN = "#76B900"

# Modes the TUI cycles through
_MODES = ("code", "plan", "auto")

# ---------------------------------------------------------------------------
# Inline CSS — Textual's CSS DSL
# ---------------------------------------------------------------------------

_STYLESHEET = (
    """\
/* ── Global ────────────────────────────────────────────────── */

Screen {
    layout: vertical;
    background: #0b0f0c;
    color: $text;
}

/* ── Chat history ──────────────────────────────────────────── */

#chat-scroll {
    height: 1fr;
    border: solid """
    + NV_GREEN
    + """;
    border-title-color: """
    + NV_GREEN
    + """;
    background: #0d110d;
    padding: 0 1;
}

.chat-user {
    margin: 1 0 0 0;
    padding: 0 1;
    color: $text;
    background: $surface-darken-1;
}

.chat-user .label {
    color: """
    + NV_GREEN
    + """;
    text-style: bold;
}

.chat-assistant {
    margin: 0 0 0 0;
    padding: 0 1;
    color: $text;
}

.chat-thinking {
    color: $text-muted;
    text-style: italic;
    padding: 0 1;
    margin: 0 0 0 2;
}

.chat-phase {
    color: """
    + NV_GREEN
    + """;
    text-style: bold;
    padding: 0 1;
    margin: 1 0 0 0;
}

.chat-error {
    color: $error;
    text-style: bold;
    padding: 0 1;
    margin: 0 0 0 0;
}

.chat-system {
    color: $text-muted;
    text-style: italic;
    padding: 0 1;
    margin: 0 0 0 0;
}

/* ── Tool panel ────────────────────────────────────────────── */

#tool-panel {
    height: auto;
    max-height: 12;
    border: solid $accent-darken-2;
    border-title-color: """
    + NV_GREEN
    + """;
    background: #11170f;
    padding: 0 1;
    display: none;
}

#tool-panel.visible {
    display: block;
}

.tool-call {
    color: $text-muted;
    padding: 0 0 0 1;
}

.tool-result-ok {
    color: """
    + NV_GREEN
    + """;
    padding: 0 0 0 2;
}

.tool-result-error {
    color: $error;
    padding: 0 0 0 2;
}

/* ── Input area ────────────────────────────────────────────── */

#input-row {
    height: auto;
    max-height: 8;
    min-height: 3;
    dock: bottom;
}

#mode-label {
    width: 10;
    height: 3;
    content-align: center middle;
    text-style: bold;
    padding: 0 1;
}

#mode-label.mode-code {
    color: """
    + NV_GREEN
    + """;
    background: $surface-darken-2;
}

#mode-label.mode-plan {
    color: $warning;
    background: $surface-darken-2;
}

#mode-label.mode-auto {
    color: $error;
    background: $surface-darken-2;
}

#chat-input {
    height: auto;
    min-height: 3;
    max-height: 8;
    border: tall """
    + NV_GREEN
    + """;
    background: #0c100c;
}

#chat-input:focus {
    border: tall """
    + NV_GREEN
    + """;
}

/* ── Status bar ────────────────────────────────────────────── */

#status-bar {
    dock: bottom;
    height: 1;
    background: #11170d;
    color: #d6dfc4;
    padding: 0 1;
}

#status-bar .status-mode {
    text-style: bold;
}

/* ── Streaming indicator ───────────────────────────────────── */

.streaming-indicator {
    color: """
    + NV_GREEN
    + """;
    text-style: bold italic;
    padding: 0 1;
}
"""
)


# ---------------------------------------------------------------------------
# Custom messages
# ---------------------------------------------------------------------------


class AgentEventMessage(TextualMessage):
    """Carries an AgentEvent from the worker to the UI thread."""

    def __init__(self, event: AgentEvent) -> None:
        super().__init__()
        self.event = event


class TurnComplete(TextualMessage):
    """Signals that the current agent turn finished."""

    def __init__(self, error: str | None = None) -> None:
        super().__init__()
        self.error = error


class SlashCommandResult(TextualMessage):
    """Result of a slash command dispatch."""

    def __init__(self, text: str, should_quit: bool = False) -> None:
        super().__init__()
        self.text = text
        self.should_quit = should_quit


# ---------------------------------------------------------------------------
# Helper: git branch detection (cached)
# ---------------------------------------------------------------------------

_cached_git_branch: str | None = None
_git_branch_ts: float = 0.0


def _get_git_branch() -> str:
    """Return the current git branch, cached for 10 seconds."""
    global _cached_git_branch, _git_branch_ts
    now = time.monotonic()
    if _cached_git_branch is not None and now - _git_branch_ts < 10.0:
        return _cached_git_branch
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        _cached_git_branch = r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        _cached_git_branch = ""
    _git_branch_ts = now
    return _cached_git_branch


# ---------------------------------------------------------------------------
# Slash command dispatcher (TUI variant)
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
[bold]Session:[/bold]
  /help          Show this help message
  /think         Toggle thinking trace display
  /compact       Compact conversation (free context window)
  /reset         Reset conversation
  /undo          Revert the last file change
  /cost          Session cost and token usage
  /mode          Cycle mode: code -> plan -> auto
  /quit          Exit

[bold]Configuration:[/bold]
  /endpoint      List endpoints  |  /endpoint <name>  Switch
  /formation     List formations |  /formation <name>  Activate
  /agent         List primary agents | /agent <name> Switch

[bold]Keyboard:[/bold]
  Enter          Send message (Shift+Enter for newline)
  Tab            Cycle mode
  Ctrl+C         Cancel current response
  Ctrl+Q         Exit
"""


@dataclass
class _TUIState:
    """Mutable session state for the TUI, analogous to _ReplState."""

    config: NeMoCodeConfig
    show_thinking: bool = False
    auto_yes: bool = False
    turn_count: int = 0
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    context_mgr: ContextManager = field(default_factory=ContextManager)
    agent: CodeAgent | None = None
    agent_name: str | None = None
    mode_idx: int = 0
    is_streaming: bool = False
    cancelled: bool = False
    # Accumulators for the current turn
    text_buf: str = ""
    think_buf: str = ""
    last_usage: dict[str, int] = field(default_factory=dict)
    tool_call_count: int = 0
    tool_error_count: int = 0
    turn_start: float = 0.0
    first_token_time: float | None = None
    current_model_id: str = ""
    reasoning_hint_shown: bool = False

    @property
    def mode(self) -> str:
        return _MODES[self.mode_idx]

    def cycle_mode(self) -> str:
        self.mode_idx = (self.mode_idx + 1) % len(_MODES)
        return self.mode

    def cancel(self) -> None:
        self.cancelled = True

    def clear_cancel(self) -> None:
        self.cancelled = False

    def reset_turn(self) -> None:
        """Reset per-turn accumulators."""
        self.text_buf = ""
        self.think_buf = ""
        self.last_usage = {}
        self.tool_call_count = 0
        self.tool_error_count = 0
        self.first_token_time = None
        self.current_model_id = ""
        self.reasoning_hint_shown = False
        self.turn_start = time.time()

    def record_metrics(self) -> None:
        """Snapshot the current turn metrics into the collector."""
        prompt_tokens = self.last_usage.get("prompt_tokens", 0)
        completion_tokens = self.last_usage.get("completion_tokens", 0)
        total_time = (time.time() - self.turn_start) * 1000 if self.turn_start else 0
        ttft = 0.0
        if self.first_token_time is not None and self.turn_start:
            ttft = (self.first_token_time - self.turn_start) * 1000

        ep_name = self.config.default_endpoint
        m = RequestMetrics(
            model_id=self.current_model_id,
            endpoint_name=ep_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_to_first_token_ms=ttft,
            total_time_ms=total_time,
            tool_calls=self.tool_call_count,
            tool_errors=self.tool_error_count,
        )
        self.metrics.record(m)

    def build_agent(self) -> CodeAgent:
        """Build (or rebuild) the CodeAgent based on current mode/config."""
        if self.auto_yes or self.mode == "auto":
            confirm_fn = _auto_confirm
        else:
            confirm_fn = _tui_confirm
        return CodeAgent(
            config=self.config,
            confirm_fn=confirm_fn,
            read_only=(self.mode == "plan"),
            agent_name=self.current_primary_agent_name(),
        )

    def current_primary_agent_name(self) -> str | None:
        if self.mode == "plan":
            return "plan" if "plan" in self.config.agents else None
        return self.agent_name

    def current_primary_agent_display(self) -> str:
        current = self.current_primary_agent_name()
        if current is None:
            return "default"
        agent = self.config.agents.get(current)
        if agent is None:
            return current
        return agent.display_name or agent.name

    def resolve_context_window(self) -> int:
        ep_name = self.config.default_endpoint
        ep = self.config.endpoints.get(ep_name)
        if ep:
            manifest = self.config.manifests.get(ep.model_id)
            if manifest:
                return manifest.context_window
        return 128_000


# Confirmation stubs — TUI auto-approve for now (interactive confirm is complex
# in Textual; a future iteration can use a modal dialog).


async def _auto_confirm(tool_name: str, args: dict) -> bool:
    return True


# For code mode, we auto-approve in the TUI as well but log it.
# A production upgrade would show a Textual modal.
async def _tui_confirm(tool_name: str, args: dict) -> bool:
    return True


# ---------------------------------------------------------------------------
# Token formatting
# ---------------------------------------------------------------------------


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class ModeLabel(Static):
    """Displays the current mode with color coding."""

    mode: reactive[str] = reactive("code")

    def render(self) -> str:
        return f" {self.mode.upper()} "

    def watch_mode(self, value: str) -> None:
        self.remove_class("mode-code", "mode-plan", "mode-auto")
        self.add_class(f"mode-{value}")


class StatusBar(Static):
    """Bottom status bar showing context %, mode, model name, git branch, tokens, tok/s."""

    context_pct: reactive[float] = reactive(0.0)
    mode: reactive[str] = reactive("code")
    endpoint: reactive[str] = reactive("")
    model_name: reactive[str] = reactive("")
    formation: reactive[str] = reactive("")
    agent_name: reactive[str] = reactive("")
    git_branch: reactive[str] = reactive("")
    total_tokens: reactive[int] = reactive(0)
    total_cost: reactive[float] = reactive(0.0)
    last_tps: reactive[float] = reactive(0.0)
    is_streaming: reactive[bool] = reactive(False)

    def render(self) -> str:
        parts: list[str] = []

        parts.append(f"[bold {NV_GREEN}]NVIDIA[/bold {NV_GREEN}]")

        # Streaming indicator
        if self.is_streaming:
            parts.append(f"[bold {NV_GREEN}]LIVE[/bold {NV_GREEN}]")

        # Context %
        if self.context_pct > 0:
            pct = self.context_pct
            if pct > 80:
                parts.append(f"[red]ctx:{pct:.0f}%[/red]")
            elif pct > 50:
                parts.append(f"[yellow]ctx:{pct:.0f}%[/yellow]")
            else:
                parts.append(f"ctx:{pct:.0f}%")

        # Mode
        mode_colors = {"code": NV_GREEN, "plan": "yellow", "auto": "red"}
        c = mode_colors.get(self.mode, NV_GREEN)
        parts.append(f"[{c}]{self.mode}[/{c}]")

        if self.formation:
            parts.append(f"[{NV_GREEN}]{self.formation}[/{NV_GREEN}]")
        if self.agent_name:
            parts.append(f"[blue]{self.agent_name}[/blue]")

        # Model display name (preferred) or endpoint fallback
        if self.model_name:
            parts.append(self.model_name)
        elif self.endpoint:
            parts.append(self.endpoint)

        # Git branch
        if self.git_branch:
            parts.append(f"[dim]{self.git_branch}[/dim]")

        # Token usage
        if self.total_tokens > 0:
            tok_str = _fmt_tokens(self.total_tokens)
            cost_str = f" ${self.total_cost:.4f}" if self.total_cost > 0 else ""
            parts.append(f"[dim]{tok_str} tok{cost_str}[/dim]")

        # Last throughput
        if self.last_tps > 0:
            parts.append(f"[dim]{self.last_tps:.0f} tok/s[/dim]")

        # Version
        parts.append(f"[dim]v{__version__}[/dim]")

        return "  ".join(parts)


class ToolPanel(VerticalScroll):
    """Collapsible panel showing tool call/result activity."""

    def add_tool_call(self, name: str, args: dict) -> None:
        line = format_tool_call(name, args)
        w = Static(f"[{NV_GREEN}]> [/{NV_GREEN}][dim]{line}[/dim]", classes="tool-call")
        self.mount(w)
        self._show()
        w.scroll_visible()

    def add_tool_result(self, name: str, result: str, is_error: bool) -> None:
        # Truncate long results
        display = result.strip()
        lines = display.splitlines()
        if len(lines) > 5:
            display = "\n".join(lines[:5]) + f"\n... ({len(lines)} lines total)"
        if len(display) > 500:
            display = display[:500] + "..."

        parsed = None
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            pass

        embedded_error = tool_result_has_embedded_error(name, result)
        if embedded_error:
            is_error = True

        if is_error:
            cls = "tool-result-error"
            prefix = "x"
            if embedded_error and isinstance(parsed, dict):
                summary = summarize_delegate_result(parsed)
                if summary:
                    headline, preview = summary
                    display = headline if not preview else f"{headline}\n{preview}"
        else:
            cls = "tool-result-ok"
            prefix = "ok"
            # For structured OK results, show just a summary
            if isinstance(parsed, dict):
                summary = summarize_delegate_result(parsed)
                if summary:
                    headline, preview = summary
                    display = headline if not preview else f"{headline}\n{preview}"
                elif parsed.get("status") == "ok":
                    b = parsed.get("bytes", "")
                    display = "ok" + (f" ({b:,} bytes)" if b else "")
                elif "exit_code" in parsed:
                    code = parsed["exit_code"]
                    if code == 0:
                        display = "ok (exit 0)"
                    else:
                        stderr = parsed.get("stderr", "")
                        display = f"exit {code}: {stderr[:200]}" if stderr else f"exit {code}"

        w = Static(f"  [{cls}]{prefix}[/{cls}] [dim]{display}[/dim]", classes=cls)
        self.mount(w)
        w.scroll_visible()

    def clear_panel(self) -> None:
        self.remove_children()
        self._hide()

    def _show(self) -> None:
        self.add_class("visible")

    def _hide(self) -> None:
        self.remove_class("visible")


class ChatScroll(VerticalScroll):
    """Scrollable chat history."""

    def add_user_message(self, text: str) -> None:
        content = f"[{NV_GREEN} bold]You:[/{NV_GREEN} bold] {text}"
        w = Static(content, classes="chat-user")
        self.mount(w)
        w.scroll_visible()

    def add_assistant_text(self, text: str) -> None:
        """Add or update the assistant's streaming response."""
        try:
            existing = self.query_one("#assistant-stream", Static)
            existing.update(text)
            existing.scroll_visible()
        except NoMatches:
            w = Static(text, id="assistant-stream", classes="chat-assistant")
            self.mount(w)
            w.scroll_visible()

    def finalize_assistant(self) -> None:
        """Freeze the current streaming response so the next turn starts fresh."""
        try:
            existing = self.query_one("#assistant-stream", Static)
            existing.id = None  # detach the live id
        except NoMatches:
            pass

    def add_thinking(self, text: str) -> None:
        try:
            existing = self.query_one("#thinking-stream", Static)
            existing.update(f"[italic dim]{text}[/italic dim]")
        except NoMatches:
            w = Static(
                f"[italic dim]{text}[/italic dim]",
                id="thinking-stream",
                classes="chat-thinking",
            )
            self.mount(w)
            w.scroll_visible()

    def finalize_thinking(self) -> None:
        try:
            existing = self.query_one("#thinking-stream", Static)
            existing.id = None
        except NoMatches:
            pass

    def add_phase(self, role: str, text: str) -> None:
        content = f"[{NV_GREEN} bold]--- {role}: {text} ---[/{NV_GREEN} bold]"
        w = Static(content, classes="chat-phase")
        self.mount(w)
        w.scroll_visible()

    def add_status(self, text: str) -> None:
        w = Static(f"[dim]· {text}[/dim]", classes="chat-system")
        self.mount(w)
        w.scroll_visible()

    def add_error(self, text: str) -> None:
        w = Static(f"[bold red]Error: {text}[/bold red]", classes="chat-error")
        self.mount(w)
        w.scroll_visible()

    def add_system(self, text: str) -> None:
        w = Static(f"[dim italic]{text}[/dim italic]", classes="chat-system")
        self.mount(w)
        w.scroll_visible()

    def add_streaming_indicator(self) -> None:
        try:
            self.query_one("#streaming-dot")
        except NoMatches:
            w = Static(
                f"[{NV_GREEN} bold italic]Thinking...[/{NV_GREEN} bold italic]",
                id="streaming-dot",
                classes="streaming-indicator",
            )
            self.mount(w)
            w.scroll_visible()

    def remove_streaming_indicator(self) -> None:
        try:
            self.query_one("#streaming-dot").remove()
        except NoMatches:
            pass


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------


class NeMoCodeTUI(App):
    """Full-screen TUI for NeMoCode."""

    TITLE = "NeMoCode"
    SUB_TITLE = "Terminal-first agentic coding"
    CSS = _STYLESHEET

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", show=True, priority=True),
        Binding("tab", "cycle_mode", "Mode", show=True),
        Binding("ctrl+c", "cancel_turn", "Cancel", show=True, priority=True),
        Binding("enter", "submit", "Send", show=True),
        Binding("ctrl+t", "toggle_tools", "Tools", show=True),
    ]

    # Reactive properties
    mode: reactive[str] = reactive("code")
    streaming: reactive[bool] = reactive(False)

    def __init__(
        self,
        config: NeMoCodeConfig | None = None,
        endpoint: str | None = None,
        formation: str | None = None,
        agent_name: str | None = None,
        show_thinking: bool = False,
        auto_yes: bool = False,
    ) -> None:
        super().__init__()
        cfg = config or load_config()
        if endpoint:
            cfg.default_endpoint = endpoint
        if formation:
            cfg.active_formation = formation

        self._state = _TUIState(
            config=cfg,
            show_thinking=show_thinking,
            auto_yes=auto_yes,
        )
        if agent_name:
            self._state.agent_name = resolve_agent_reference(cfg.agents, agent_name) or (
                agent_name if agent_name in cfg.agents else None
            )
        elif "build" in cfg.agents:
            self._state.agent_name = "build"
        self._state.context_mgr = ContextManager(
            context_window=self._state.resolve_context_window(),
        )
        self._state.agent = self._state.build_agent()

        # Track the async task for cancellation
        self._current_task: asyncio.Task | None = None

    # ── Compose ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ChatScroll(id="chat-scroll")
            yield ToolPanel(id="tool-panel")
            with Horizontal(id="input-row"):
                yield ModeLabel(id="mode-label")
                yield TextArea(id="chat-input")
            yield StatusBar(id="status-bar")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────

    def on_mount(self) -> None:
        """Initialize the UI after mounting."""
        chat = self.query_one("#chat-scroll", ChatScroll)
        ep_name = self._state.config.default_endpoint
        ep = self._state.config.endpoints.get(ep_name)
        model_id = ep.model_id if ep else "unknown"
        manifest = self._state.config.manifests.get(model_id) if ep else None
        model_display = manifest.display_name if manifest else model_id
        formation = self._state.config.active_formation or ""
        formation_str = f" · {formation}" if formation else ""
        agent_label = self._state.current_primary_agent_display()

        chat.border_title = "NeMoCode // NVIDIA NIM"
        chat.add_system(
            f"NeMoCode // NVIDIA NIM\n"
            f"{model_display} · {ep_name}{formation_str} · {agent_label}\n"
            f"{Path.cwd()}\n"
            f"/help · Tab mode · Ctrl+T trace"
        )

        tool_panel = self.query_one("#tool-panel", ToolPanel)
        tool_panel.border_title = "Inference Trace"

        mode_label = self.query_one("#mode-label", ModeLabel)
        mode_label.mode = self._state.mode

        # Configure TextArea
        text_area = self.query_one("#chat-input", TextArea)
        text_area.show_line_numbers = False
        text_area.focus()

        self._update_status_bar()

    # ── Reactive watchers ────────────────────────────────────

    def watch_mode(self, value: str) -> None:
        try:
            label = self.query_one("#mode-label", ModeLabel)
            label.mode = value
        except NoMatches:
            pass
        self._update_status_bar()

    def watch_streaming(self, value: bool) -> None:
        self._update_status_bar()

    # ── Actions ──────────────────────────────────────────────

    def action_quit_app(self) -> None:
        """Exit the TUI."""
        self.exit()

    def action_cycle_mode(self) -> None:
        """Cycle through code -> plan -> auto."""
        if self.streaming:
            return  # Don't switch modes mid-stream
        new_mode = self._state.cycle_mode()
        self._state.agent = self._state.build_agent()
        self.mode = new_mode
        chat = self.query_one("#chat-scroll", ChatScroll)
        mode_desc = {"code": "ask before tools", "plan": "text only", "auto": "auto-approve"}
        chat.add_system(f"Mode: {new_mode} ({mode_desc.get(new_mode, '')})")
        self._update_status_bar()

    def action_cancel_turn(self) -> None:
        """Cancel the current streaming turn."""
        if self.streaming:
            self._state.cancel()
            chat = self.query_one("#chat-scroll", ChatScroll)
            chat.remove_streaming_indicator()
            chat.add_system("Turn cancelled.")
            self.streaming = False
            self._state.is_streaming = False

    def action_submit(self) -> None:
        """Send the current input to the agent."""
        text_area = self.query_one("#chat-input", TextArea)
        text = text_area.text.strip()
        if not text:
            return
        if self.streaming:
            return  # Ignore while streaming

        text_area.clear()
        text_area.focus()

        # Handle slash commands
        if text.startswith("/"):
            self._dispatch_slash(text)
            return

        # Normal message
        self._send_message(text)

    def action_toggle_tools(self) -> None:
        """Toggle the tool panel visibility."""
        panel = self.query_one("#tool-panel", ToolPanel)
        panel.toggle_class("visible")

    # ── Input handling ───────────────────────────────────────

    @on(TextArea.Changed, "#chat-input")
    def _on_input_changed(self, event: TextArea.Changed) -> None:
        """Handle enter key for submission (newline requires Shift+Enter)."""
        pass  # TextArea handles multiline naturally

    def on_key(self, event) -> None:
        """Intercept Enter in the input area to submit instead of newline."""
        if event.key == "enter":
            text_area = self.query_one("#chat-input", TextArea)
            if text_area.has_focus:
                # Check if shift is held — if so, allow newline
                # Textual sends "enter" without shift distinction by default,
                # so we use a simple heuristic: if the text ends with \n and
                # the user just pressed enter, submit the text before the newline.
                event.prevent_default()
                event.stop()
                self.action_submit()

    # ── Slash commands ───────────────────────────────────────

    def _dispatch_slash(self, line: str) -> None:
        """Process a slash command and display the result."""
        parts = line.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        chat = self.query_one("#chat-scroll", ChatScroll)

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
            return

        if cmd == "/help":
            chat.add_system(_HELP_TEXT)
            return

        if cmd == "/think":
            self._state.show_thinking = not self._state.show_thinking
            status = "on" if self._state.show_thinking else "off"
            chat.add_system(f"Thinking trace: {status}")
            return

        if cmd == "/compact":
            try:
                if self._state.agent:
                    self._state.agent.compact()
                chat.add_system("Conversation compacted. Older messages trimmed.")
            except Exception as exc:
                chat.add_error(f"Compact failed: {exc}")
            self._update_status_bar()
            return

        if cmd == "/reset":
            if self._state.agent:
                self._state.agent.reset()
            self._state.turn_count = 0
            chat.add_system("Conversation reset.")
            self._update_status_bar()
            return

        if cmd == "/undo":
            try:
                from nemocode.tools.fs import undo_last, undo_stack_depth

                depth = undo_stack_depth()
                if depth == 0:
                    chat.add_system("Nothing to undo.")
                    return
                result = undo_last()
                if "error" in result:
                    chat.add_error(f"Undo failed: {result['error']}")
                else:
                    action = result.get("action", "reverted")
                    path = result.get("path", "?")
                    remaining = undo_stack_depth()
                    msg = f"Undo: {action} {path}"
                    if remaining > 0:
                        msg += f"  ({remaining} more available)"
                    chat.add_system(msg)
            except Exception as exc:
                chat.add_error(f"Undo error: {exc}")
            return

        if cmd == "/cost":
            summary = self._state.metrics.summary()
            lines = [
                "Session Cost:",
                f"  API requests:      {summary['requests']}",
                f"  Prompt tokens:     {summary['prompt_tokens']:,}",
                f"  Completion tokens: {summary['completion_tokens']:,}",
                f"  Total tokens:      {summary['total_tokens']:,}",
                f"  Estimated cost:    ${summary['estimated_cost_usd']:.6f}",
                f"  Avg latency:       {summary['avg_latency_ms']:.0f}ms",
            ]
            if summary["avg_tokens_per_sec"] > 0:
                lines.append(f"  Avg tok/s:         {summary['avg_tokens_per_sec']:.1f}")
            if summary["avg_ttft_ms"] > 0:
                lines.append(f"  Avg TTFT:          {summary['avg_ttft_ms']:.0f}ms")
            lines.append(f"  Session duration:  {summary['session_duration_s']:.0f}s")
            chat.add_system("\n".join(lines))
            return

        if cmd == "/endpoint":
            if not arg:
                endpoints = list(self._state.config.endpoints.keys())
                current = self._state.config.default_endpoint
                lines = ["Available endpoints:"]
                for ep_name in endpoints:
                    marker = " (active)" if ep_name == current else ""
                    ep = self._state.config.endpoints[ep_name]
                    lines.append(f"  {ep_name}{marker}  {ep.model_id}")
                chat.add_system("\n".join(lines))
            elif arg in self._state.config.endpoints:
                self._state.config.default_endpoint = arg
                self._state.config.active_formation = None
                self._state.context_mgr = ContextManager(
                    context_window=self._state.resolve_context_window()
                )
                self._state.agent = self._state.build_agent()
                self._state.turn_count = 0
                ep = self._state.config.endpoints[arg]
                chat.add_system(f"Switched to endpoint: {arg} ({ep.model_id})")
                self._update_status_bar()
            else:
                available = ", ".join(self._state.config.endpoints.keys())
                chat.add_error(f"Unknown endpoint: {arg}  (available: {available})")
            return

        if cmd == "/formation":
            if not arg:
                formations = list(self._state.config.formations.keys())
                current = self._state.config.active_formation
                if not formations:
                    chat.add_system("No formations configured.")
                    return
                lines = ["Available formations:"]
                for f_name in formations:
                    marker = " (active)" if f_name == current else ""
                    f = self._state.config.formations[f_name]
                    roles = ", ".join(s.role.value for s in f.slots)
                    lines.append(f"  {f_name}{marker}  [{roles}]")
                chat.add_system("\n".join(lines))
            elif arg in ("off", "none"):
                self._state.config.active_formation = None
                self._state.agent = self._state.build_agent()
                chat.add_system("Formation deactivated. Using single endpoint mode.")
                self._update_status_bar()
            elif arg in self._state.config.formations:
                self._state.config.active_formation = arg
                self._state.agent = self._state.build_agent()
                f = self._state.config.formations[arg]
                roles = ", ".join(s.role.value for s in f.slots)
                chat.add_system(f"Activated formation: {arg} [{roles}]")
                self._update_status_bar()
            else:
                available = ", ".join(self._state.config.formations.keys())
                chat.add_error(f"Unknown formation: {arg}  (available: {available})")
            return

        if cmd == "/agent":
            primary_agents = {
                name: agent
                for name, agent in self._state.config.agents.items()
                if agent.mode != AgentMode.SUBAGENT
            }
            if not arg:
                if not primary_agents:
                    chat.add_system("No primary agents configured.")
                    return
                current = self._state.current_primary_agent_name()
                lines = ["Primary agents:"]
                for name, agent in sorted(primary_agents.items()):
                    marker = " (active)" if name == current else ""
                    display = f"  {agent.display_name}" if agent.display_name else ""
                    lines.append(f"  {name}{marker}{display}")
                chat.add_system("\n".join(lines))
                return

            resolved = resolve_agent_reference(primary_agents, arg) or arg
            agent = primary_agents.get(resolved)
            if agent is None:
                available = ", ".join(sorted(primary_agents.keys()))
                chat.add_error(f"Unknown primary agent: {arg}  (available: {available})")
                return

            self._state.agent_name = resolved
            self._state.agent = self._state.build_agent()
            chat.add_system(
                f"Switched primary agent: {resolved} ({agent.display_name or resolved})"
            )
            self._update_status_bar()
            return

        if cmd == "/mode":
            self.action_cycle_mode()
            return

        chat.add_system(f"Unknown command: {cmd}. Type /help for commands.")

    # ── Agent interaction ────────────────────────────────────

    def _send_message(self, text: str) -> None:
        """Start an agent turn for the given user input."""
        chat = self.query_one("#chat-scroll", ChatScroll)
        chat.add_user_message(text)
        chat.add_streaming_indicator()

        tool_panel = self.query_one("#tool-panel", ToolPanel)
        tool_panel.clear_panel()

        self.streaming = True
        self._state.is_streaming = True
        self._state.clear_cancel()
        self._state.reset_turn()
        self._state.turn_count += 1
        self._update_status_bar()

        self._run_agent_turn(text)

    @work(exclusive=True, thread=False)
    async def _run_agent_turn(self, user_input: str) -> None:
        """Run the agent loop in a background worker, posting messages to the UI."""
        agent = self._state.agent
        if agent is None:
            self.post_message(TurnComplete(error="No agent configured"))
            return

        try:
            async for event in agent.run(user_input):
                if self._state.cancelled:
                    continue
                self.post_message(AgentEventMessage(event))
        except asyncio.CancelledError:
            self.post_message(TurnComplete(error=None))
            return
        except Exception as exc:
            logger.exception("Agent turn failed")
            self.post_message(TurnComplete(error=str(exc)))
            return

        self.post_message(TurnComplete())

    # ── Event handlers ───────────────────────────────────────

    @on(AgentEventMessage)
    def _on_agent_event(self, msg: AgentEventMessage) -> None:
        """Route AgentEvents to the appropriate UI update."""
        event = msg.event
        chat = self.query_one("#chat-scroll", ChatScroll)
        tool_panel = self.query_one("#tool-panel", ToolPanel)

        if event.kind == "text":
            if self._state.first_token_time is None:
                self._state.first_token_time = time.time()
                chat.remove_streaming_indicator()
            self._state.text_buf += event.text
            chat.add_assistant_text(self._state.text_buf)

        elif event.kind == "thinking":
            self._state.think_buf += event.thinking
            if self._state.show_thinking:
                chat.add_thinking(self._state.think_buf)

        elif event.kind == "phase":
            role = event.role.value if event.role else "agent"
            chat.add_phase(role, event.text)

        elif event.kind == "status":
            if not self._state.show_thinking and not self._state.reasoning_hint_shown:
                chat.add_system("Reasoning trace hidden. Use /think to show it when available.")
                self._state.reasoning_hint_shown = True
            chat.add_status(event.text)

        elif event.kind == "tool_call":
            self._state.tool_call_count += 1
            tool_panel.add_tool_call(event.tool_name, event.tool_args)

        elif event.kind == "tool_result":
            is_error = event.is_error or tool_result_has_embedded_error(
                event.tool_name,
                event.tool_result,
            )
            if is_error:
                self._state.tool_error_count += 1
            tool_panel.add_tool_result(event.tool_name, event.tool_result, is_error)

        elif event.kind == "error":
            chat.remove_streaming_indicator()
            chat.add_error(event.text)

        elif event.kind == "usage":
            self._state.last_usage = event.usage
            ep_name = self._state.config.default_endpoint
            ep = self._state.config.endpoints.get(ep_name)
            if ep:
                self._state.current_model_id = ep.model_id
            self._update_status_bar()

    @on(TurnComplete)
    def _on_turn_complete(self, msg: TurnComplete) -> None:
        """Clean up after an agent turn finishes."""
        chat = self.query_one("#chat-scroll", ChatScroll)
        chat.remove_streaming_indicator()
        chat.finalize_assistant()
        chat.finalize_thinking()

        if msg.error:
            chat.add_error(f"Turn failed: {msg.error}")

        # Record metrics
        self._state.record_metrics()

        # Show turn summary in chat
        self._show_turn_summary(chat)

        self.streaming = False
        self._state.is_streaming = False
        self._update_status_bar()

        # Re-focus the input
        try:
            self.query_one("#chat-input", TextArea).focus()
        except NoMatches:
            pass

    def _show_turn_summary(self, chat: ChatScroll) -> None:
        """Show a compact performance summary after a turn completes."""
        s = self._state
        elapsed = time.time() - s.turn_start if s.turn_start else 0
        completion_tokens = s.last_usage.get("completion_tokens", 0)
        total_tokens = s.last_usage.get("prompt_tokens", 0) + completion_tokens
        if total_tokens == 0:
            return

        parts: list[str] = []
        parts.append(f"{elapsed:.1f}s")
        parts.append(f"{total_tokens:,} tok")

        if completion_tokens > 0 and elapsed > 0:
            tps = completion_tokens / elapsed
            parts.append(f"{tps:.0f} tok/s")

        if s.first_token_time is not None and s.turn_start:
            ttft = s.first_token_time - s.turn_start
            parts.append(f"TTFT {ttft:.1f}s")

        if s.tool_call_count > 0:
            parts.append(f"{s.tool_call_count} tool{'s' if s.tool_call_count != 1 else ''}")

        chat.add_system(f"▸ {' │ '.join(parts)}")

    # ── Status bar update ────────────────────────────────────

    def _update_status_bar(self) -> None:
        """Refresh the status bar with current state."""
        try:
            bar = self.query_one("#status-bar", StatusBar)
        except NoMatches:
            return

        bar.mode = self._state.mode
        bar.endpoint = self._state.config.default_endpoint
        bar.formation = self._state.config.active_formation or ""
        bar.agent_name = self._state.current_primary_agent_display()
        bar.git_branch = _get_git_branch()
        bar.is_streaming = self._state.is_streaming

        # Model display name
        ep_name = self._state.config.default_endpoint
        ep = self._state.config.endpoints.get(ep_name)
        if ep:
            manifest = self._state.config.manifests.get(ep.model_id)
            bar.model_name = manifest.display_name if manifest else ep.model_id
        else:
            bar.model_name = ""

        # Last throughput
        bar.last_tps = self._state.metrics.last_tokens_per_sec

        # Context usage
        total_tokens = 0
        if self._state.agent:
            try:
                for session in self._state.agent.sessions.values():
                    total_tokens += self._state.context_mgr.usage(session.messages)
            except Exception:
                pass
        total_tokens = max(total_tokens, self._state.metrics.total_tokens)
        ctx_window = self._state.context_mgr.context_window
        bar.context_pct = (total_tokens / ctx_window * 100) if ctx_window > 0 else 0
        bar.total_tokens = total_tokens
        bar.total_cost = self._state.metrics.total_cost


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_tui(
    config: NeMoCodeConfig | None = None,
    endpoint: str | None = None,
    formation: str | None = None,
    agent_name: str | None = None,
    show_thinking: bool = False,
    auto_yes: bool = False,
) -> None:
    """Synchronous entry point to launch the TUI.

    Called from ``code_cmd`` when ``--tui`` is passed, or directly.
    """
    app = NeMoCodeTUI(
        config=config,
        endpoint=endpoint,
        formation=formation,
        agent_name=agent_name,
        show_thinking=show_thinking,
        auto_yes=auto_yes,
    )
    app.run()


def start_tui(
    endpoint: str | None = None,
    formation: str | None = None,
    agent_name: str | None = None,
    think: bool = False,
    yes: bool = False,
) -> None:
    """Mirror of ``start_repl`` for the TUI path.

    Loads config and launches the TUI. Used by ``code_cmd --tui``.
    """
    config = load_config()
    run_tui(
        config=config,
        endpoint=endpoint,
        formation=formation,
        agent_name=agent_name,
        show_thinking=think,
        auto_yes=yes,
    )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_tui()
