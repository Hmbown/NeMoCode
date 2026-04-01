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

from textual import events, on, work
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
from nemocode.cli.theme import build_tui_stylesheet, canonical_key_spec, format_key_label, get_theme
from nemocode.config import load_config
from nemocode.config.agents import resolve_agent_reference
from nemocode.config.schema import AgentMode, NeMoCodeConfig
from nemocode.core.context import ContextManager
from nemocode.core.metrics import MetricsCollector, RequestMetrics
from nemocode.core.scheduler import AgentEvent
from nemocode.core.sessions import TurnBoundary
from nemocode.workflows.code_agent import CodeAgent

logger = logging.getLogger(__name__)

# Modes the TUI cycles through
_MODES = ("code", "plan", "auto")
_STYLESHEET = build_tui_stylesheet(get_theme("nvidia-dark"))


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


class UserQuestionRequest(TextualMessage):
    """Prompt the user for an answer while an agent turn is running."""

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__()
        self.question = question
        self.options = options or []


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


def _short_model_ref(model_id: str) -> str:
    """Compact long local model paths for display surfaces."""
    if not model_id:
        return "-"
    return Path(model_id).name if model_id.startswith("/") else model_id


def _endpoint_summary(endpoint: object) -> str:
    name = getattr(endpoint, "name", "") or ""
    model_id = _short_model_ref(getattr(endpoint, "model_id", "") or "")
    if name and model_id and name != model_id:
        return f"{name} · {model_id}"
    return name or model_id or "-"


def _tui_theme(config: NeMoCodeConfig) -> object:
    return get_theme(config.theme)


def _theme_hex_for_widget(widget: Static) -> str:
    app = getattr(widget, "app", None)
    theme = getattr(app, "_theme", None)
    return theme.accent_hex if theme is not None else get_theme("nvidia-dark").accent_hex


def _turn_preview(text: str, limit: int = 60) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}…"


# ---------------------------------------------------------------------------
# Slash command dispatcher (TUI variant)
# ---------------------------------------------------------------------------


def _help_text(state: "_TUIState") -> str:
    keys = state.config.keybindings.tui
    return f"""\
[bold]Session:[/bold]
  /help          Show this help message
  /think         Toggle thinking trace display
  /compact       Compact conversation (free context window)
  /reset         Reset conversation
  /undo          Undo the last file change
  /retry         Revert and replay the last user turn
  /cost          Session cost and token usage
  /mode          Cycle mode: code -> plan -> auto
  /quit          Exit

[bold]Snapshots:[/bold]
  /snapshot      Create a git safety checkpoint
  /snapshots     List available snapshots
  /revert        Revert to a snapshot or rewind the last turn

[bold]Configuration:[/bold]
  /endpoint      List endpoints  |  /endpoint <name>  Switch
  /formation     List formations |  /formation <name>  Activate
  /agent         List primary agents | /agent <name> Switch

[bold]Modes:[/bold]
  code           Ask before tools
  plan           Read-only planning + approval
  auto           Auto-approve everything

[bold]Keyboard:[/bold]
  {format_key_label(keys.submit):<18}Send message
  {format_key_label(keys.cycle_mode):<18}Cycle mode
  {format_key_label(keys.cancel_turn):<18}Cancel current response
  {format_key_label(keys.toggle_tools):<18}Toggle tool trace
  {format_key_label(keys.exit):<18}Exit
"""


@dataclass
class _TUIState:
    """Mutable session state for the TUI, analogous to _ReplState."""

    config: NeMoCodeConfig
    show_thinking: bool = True
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
    active_turn: TurnBoundary | None = None
    last_turn: TurnBoundary | None = None

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

    def clear_turn_history(self) -> None:
        self.active_turn = None
        self.last_turn = None

    def begin_turn(self, user_input: str) -> None:
        try:
            from nemocode.tools.fs import undo_stack_depth

            undo_depth = undo_stack_depth()
        except Exception:
            undo_depth = 0

        agent = self.agent
        self.active_turn = TurnBoundary(
            user_input=user_input,
            turn_count_before=self.turn_count,
            metrics_request_count=self.metrics.request_count,
            undo_depth_before=undo_depth,
            session_checkpoints=(
                {role: session.checkpoint() for role, session in agent.sessions.items()}
                if agent
                else {}
            ),
            pending_plan_text=agent.pending_plan_text if agent else None,
            pending_plan_user_input=getattr(agent, "_pending_plan_user_input", None),
        )

    def finish_turn(self) -> None:
        if self.active_turn is not None:
            self.last_turn = self.active_turn
            self.active_turn = None

    def _restore_turn_sessions(self, turn: TurnBoundary) -> None:
        if self.agent is None:
            return
        sessions = self.agent.sessions
        for role in list(sessions):
            if role not in turn.session_checkpoints:
                del sessions[role]
        for role, checkpoint in turn.session_checkpoints.items():
            session = sessions.get(role)
            if session is not None:
                session.restore(checkpoint)

    def revert_last_turn(self) -> tuple[bool, str]:
        turn = self.last_turn
        if turn is None:
            return False, "Nothing to revert. Run a turn first."
        if self.agent is None:
            self.clear_turn_history()
            return False, "No active agent session is available to rewind."

        try:
            from nemocode.core.persistence import revert_to_point
            from nemocode.tools.fs import undo_stack_depth

            current_depth = undo_stack_depth()
            if current_depth < turn.undo_depth_before:
                self.clear_turn_history()
                return (
                    False,
                    "Last turn can no longer be rewound cleanly because the undo stack changed.",
                )
            results = revert_to_point(turn.undo_depth_before)
        except Exception as exc:
            return False, f"Failed to rewind last turn: {exc}"

        self._restore_turn_sessions(turn)
        del self.metrics._requests[turn.metrics_request_count :]
        self.turn_count = turn.turn_count_before
        self.agent._pending_plan_text = turn.pending_plan_text
        self.agent._pending_plan_user_input = turn.pending_plan_user_input
        self.clear_cancel()
        self.last_turn = None

        reverted_files = len([result for result in results if "error" not in result])
        if reverted_files > 0:
            return (
                True,
                f"Reverted last turn and restored {reverted_files} file change"
                f"{'s' if reverted_files != 1 else ''}.",
            )
        return True, "Reverted last turn. No file changes needed to be restored."

    def prepare_retry(self) -> tuple[bool, str, str | None]:
        turn = self.last_turn
        if turn is None:
            return False, "Nothing to retry. Run a turn first.", None

        preview = _turn_preview(turn.user_input)
        ok, message = self.revert_last_turn()
        if not ok:
            return False, message, None
        return True, f"Retrying: {preview}", turn.user_input

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

    def resolve_model_id(self) -> str:
        ep = self.config.endpoints.get(self.config.default_endpoint)
        return ep.model_id if ep else ""

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

    theme_name: reactive[str] = reactive("nvidia-dark")
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
        theme = get_theme(self.theme_name)

        parts.append(f"[bold {theme.accent_hex}]NVIDIA[/bold {theme.accent_hex}]")

        # Streaming indicator
        if self.is_streaming:
            parts.append(f"[bold {theme.accent_hex}]LIVE[/bold {theme.accent_hex}]")

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
        mode_colors = {"code": theme.accent_hex, "plan": "yellow", "auto": "red"}
        c = mode_colors.get(self.mode, theme.accent_hex)
        parts.append(f"[{c}]{self.mode}[/{c}]")

        if self.formation:
            parts.append(f"[{theme.accent_hex}]{self.formation}[/{theme.accent_hex}]")
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
        accent = _theme_hex_for_widget(self)
        w = Static(f"[{accent}]> [/{accent}][dim]{line}[/dim]", classes="tool-call")
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
        accent = _theme_hex_for_widget(self)
        content = f"[{accent} bold]You:[/{accent} bold] {text}"
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
            frozen = Static(existing.content, classes=" ".join(existing.classes))
            existing.remove()
            self.mount(frozen)
            frozen.scroll_visible()
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
            frozen = Static(existing.content, classes=" ".join(existing.classes))
            existing.remove()
            self.mount(frozen)
            frozen.scroll_visible()
        except NoMatches:
            pass

    def add_phase(self, role: str, text: str) -> None:
        accent = _theme_hex_for_widget(self)
        content = f"[{accent} bold]--- {role}: {text} ---[/{accent} bold]"
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
                f"[{_theme_hex_for_widget(self)} bold italic]"
                f"Thinking...[/{_theme_hex_for_widget(self)} bold italic]",
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
        Binding("ctrl+q", "quit_app", "Quit", show=True, priority=True, id="tui.exit"),
        Binding("tab", "cycle_mode", "Mode", show=True, id="tui.cycle_mode"),
        Binding("ctrl+c", "cancel_turn", "Cancel", show=True, priority=True, id="tui.cancel_turn"),
        Binding("enter", "submit", "Send", show=True, id="tui.submit"),
        Binding("ctrl+t", "toggle_tools", "Tools", show=True, id="tui.toggle_tools"),
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
        show_thinking: bool = True,
        auto_yes: bool = False,
    ) -> None:
        cfg = config or load_config()
        self._theme = _tui_theme(cfg)
        self.CSS = build_tui_stylesheet(self._theme)
        super().__init__()
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
            model_id=self._state.resolve_model_id(),
        )
        self._state.agent = self._state.build_agent()

        # Track the async task for cancellation
        self._current_task: asyncio.Task | None = None
        self._pending_user_future: asyncio.Future[str] | None = None
        self._apply_keybindings()

    def _apply_keybindings(self) -> None:
        keymap = {
            "tui.exit": self._state.config.keybindings.tui.exit,
            "tui.cycle_mode": self._state.config.keybindings.tui.cycle_mode,
            "tui.cancel_turn": self._state.config.keybindings.tui.cancel_turn,
            "tui.submit": self._state.config.keybindings.tui.submit,
            "tui.toggle_tools": self._state.config.keybindings.tui.toggle_tools,
        }
        self._bindings.apply_keymap(keymap)
        self._chat_children_before_turn = 0
        self._tool_children_before_turn = 0

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
        from nemocode.tools.clarify import set_ask_fn

        async def _ask_user_interactive(question: str, options: list[str]) -> str:
            if self._pending_user_future is not None and not self._pending_user_future.done():
                self._pending_user_future.set_result("")
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._pending_user_future = future
            self.post_message(UserQuestionRequest(question, options))
            try:
                return (await future).strip()
            finally:
                if self._pending_user_future is future:
                    self._pending_user_future = None

        set_ask_fn(_ask_user_interactive)

        chat = self.query_one("#chat-scroll", ChatScroll)
        ep_name = self._state.config.default_endpoint
        ep = self._state.config.endpoints.get(ep_name)
        model_id = ep.model_id if ep else "unknown"
        manifest = self._state.config.manifests.get(model_id) if ep else None
        model_display = manifest.display_name if manifest else _short_model_ref(model_id)
        formation = self._state.config.active_formation or ""
        formation_str = f" · {formation}" if formation else ""
        agent_label = self._state.current_primary_agent_display()
        tui_keys = self._state.config.keybindings.tui

        chat.border_title = "NeMoCode // NVIDIA NIM"
        chat.add_system(
            f"NeMoCode // NVIDIA NIM\n"
            f"{model_display} · {ep_name}{formation_str} · {agent_label}\n"
            f"{Path.cwd()}\n"
            f"/help · {format_key_label(tui_keys.cycle_mode)} mode · "
            f"{format_key_label(tui_keys.toggle_tools)} trace"
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
        self.refresh_bindings()

    def on_unmount(self) -> None:
        """Tear down interactive callbacks on exit."""
        from nemocode.tools.clarify import set_ask_fn

        if self._pending_user_future is not None and not self._pending_user_future.done():
            self._pending_user_future.set_result("")
        self._pending_user_future = None
        set_ask_fn(None)

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
        self._state.clear_turn_history()
        self.mode = new_mode
        chat = self.query_one("#chat-scroll", ChatScroll)
        mode_desc = {
            "code": "ask before tools",
            "plan": "read-only planning + approval",
            "auto": "auto-approve",
        }
        chat.add_system(f"Mode: {new_mode} ({mode_desc.get(new_mode, '')})")
        self._update_status_bar()

    def action_cancel_turn(self) -> None:
        """Cancel the current streaming turn."""
        if self.streaming:
            self._state.cancel()
            if self._pending_user_future is not None and not self._pending_user_future.done():
                self._pending_user_future.set_result("")
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

        text_area.clear()
        text_area.focus()

        if self._pending_user_future is not None and not self._pending_user_future.done():
            chat = self.query_one("#chat-scroll", ChatScroll)
            chat.add_user_message(text)
            self._pending_user_future.set_result(text)
            return

        if self.streaming:
            return  # Ignore while streaming unless answering a pending question

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

    def _rewind_visible_turn(self) -> None:
        chat = self.query_one("#chat-scroll", ChatScroll)
        for child in list(chat.children)[self._chat_children_before_turn :]:
            child.remove()

        panel = self.query_one("#tool-panel", ToolPanel)
        for child in list(panel.children)[self._tool_children_before_turn :]:
            child.remove()
        if not panel.children:
            panel.remove_class("visible")
        self._chat_children_before_turn = len(chat.children)
        self._tool_children_before_turn = len(panel.children)

    def _schedule_snapshot(self, label: str) -> None:
        asyncio.create_task(self._run_snapshot_command(label))

    async def _run_snapshot_command(self, label: str) -> None:
        chat = self.query_one("#chat-scroll", ChatScroll)
        try:
            if not self._state.agent:
                chat.add_error("No agent configured.")
                return
            snap = await self._state.agent.snapshot_mgr.create_snapshot(label or "manual")
            if snap:
                chat.add_system(
                    f"Snapshot created: {snap.id} "
                    f"({snap.files_changed} file{'s' if snap.files_changed != 1 else ''})"
                )
            else:
                chat.add_system("No changes to snapshot.")
        except Exception as exc:
            chat.add_error(f"Snapshot failed: {exc}")

    def _schedule_snapshots(self) -> None:
        asyncio.create_task(self._run_snapshots_command())

    async def _run_snapshots_command(self) -> None:
        chat = self.query_one("#chat-scroll", ChatScroll)
        try:
            if not self._state.agent:
                chat.add_error("No agent configured.")
                return
            snaps = await self._state.agent.snapshot_mgr.list_snapshots()
            if not snaps:
                chat.add_system("No snapshots available.")
                return
            lines = ["Snapshots:"]
            for snap in snaps:
                ts = time.strftime("%H:%M:%S", time.localtime(snap["timestamp"]))
                lines.append(f"  {snap['id']}  {snap['kind']}  {snap['files_changed']} files  {ts}")
            lines.append("Use /revert <id> to restore.")
            chat.add_system("\n".join(lines))
        except Exception as exc:
            chat.add_error(f"Failed to list snapshots: {exc}")

    def _schedule_snapshot_revert(self, snapshot_id: str) -> None:
        asyncio.create_task(self._run_snapshot_revert(snapshot_id))

    async def _run_snapshot_revert(self, snapshot_id: str) -> None:
        chat = self.query_one("#chat-scroll", ChatScroll)
        try:
            if not self._state.agent:
                chat.add_error("No agent configured.")
                return
            result = await self._state.agent.snapshot_mgr.restore_snapshot(snapshot_id)
            if "error" in result:
                chat.add_error(result["error"])
            else:
                self._state.clear_turn_history()
                chat.add_system(f"Reverted to snapshot {result['restored']} ({result['kind']})")
                self._update_status_bar()
        except Exception as exc:
            chat.add_error(f"Revert failed: {exc}")

    # ── Input handling ───────────────────────────────────────

    @on(TextArea.Changed, "#chat-input")
    def _on_input_changed(self, event: TextArea.Changed) -> None:
        """Handle enter key for submission (newline requires Shift+Enter)."""
        pass  # TextArea handles multiline naturally

    def on_key(self, event: events.Key) -> None:
        """Intercept Enter in the input area to submit instead of newline."""
        submit_key = canonical_key_spec(self._state.config.keybindings.tui.submit)
        if canonical_key_spec(event.key) != submit_key:
            return
        text_area = self.query_one("#chat-input", TextArea)
        if text_area.has_focus:
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
            chat.add_system(_help_text(self._state))
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
                self._state.clear_turn_history()
                chat.add_system("Conversation compacted. Older messages trimmed.")
            except Exception as exc:
                chat.add_error(f"Compact failed: {exc}")
            self._update_status_bar()
            return

        if cmd == "/reset":
            if self._state.agent:
                self._state.agent.reset()
            self._state.turn_count = 0
            self._state.clear_turn_history()
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
                    self._state.clear_turn_history()
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

        if cmd == "/retry":
            ok, message, retry_input = self._state.prepare_retry()
            if ok and retry_input is not None:
                self._rewind_visible_turn()
                chat.add_system(message)
                self._update_status_bar()
                self._send_message(retry_input)
            else:
                chat.add_system(message) if ok else chat.add_error(message)
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

        if cmd == "/snapshot":
            self._schedule_snapshot(arg)
            return

        if cmd == "/snapshots":
            self._schedule_snapshots()
            return

        if cmd == "/revert":
            if not arg:
                chat.add_system("Usage: /revert <snapshot-id> or /revert last")
                return

            if arg == "last":
                ok, message = self._state.revert_last_turn()
                if ok:
                    self._rewind_visible_turn()
                    chat.add_system(message)
                else:
                    chat.add_error(message)
                self._update_status_bar()
                return

            self._schedule_snapshot_revert(arg)
            return

        if cmd == "/endpoint":
            if not arg:
                endpoints = list(self._state.config.endpoints.keys())
                current = self._state.config.default_endpoint
                lines = ["Available endpoints:"]
                for ep_name in endpoints:
                    marker = " (active)" if ep_name == current else ""
                    ep = self._state.config.endpoints[ep_name]
                    lines.append(f"  {_endpoint_summary(ep)}{marker}")
                chat.add_system("\n".join(lines))
            elif arg in self._state.config.endpoints:
                self._state.config.default_endpoint = arg
                self._state.config.active_formation = None
                self._state.context_mgr = ContextManager(
                    context_window=self._state.resolve_context_window(),
                    model_id=self._state.resolve_model_id(),
                )
                self._state.agent = self._state.build_agent()
                self._state.turn_count = 0
                self._state.clear_turn_history()
                ep = self._state.config.endpoints[arg]
                chat.add_system(f"Switched to endpoint: {_endpoint_summary(ep)}")
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
                self._state.clear_turn_history()
                chat.add_system("Formation deactivated. Using single endpoint mode.")
                self._update_status_bar()
            elif arg in self._state.config.formations:
                self._state.config.active_formation = arg
                self._state.agent = self._state.build_agent()
                self._state.clear_turn_history()
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
            self._state.clear_turn_history()
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
        self._chat_children_before_turn = len(chat.children)
        tool_panel = self.query_one("#tool-panel", ToolPanel)
        self._tool_children_before_turn = 0
        chat.add_user_message(text)
        chat.add_streaming_indicator()
        tool_panel.clear_panel()

        self.streaming = True
        self._state.is_streaming = True
        self._state.clear_cancel()
        self._state.begin_turn(text)
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

        # Check if we're resuming a pending plan approval
        if agent.has_pending_plan:
            result = await agent.try_handle_plan_response(user_input)
            if result is not None:
                try:
                    async for event in result:
                        if self._state.cancelled:
                            continue
                        self.post_message(AgentEventMessage(event))
                except asyncio.CancelledError:
                    self.post_message(TurnComplete(error=None))
                    return
                except Exception as exc:
                    logger.exception("Plan approval handling failed")
                    self.post_message(TurnComplete(error=str(exc)))
                    return
                self.post_message(TurnComplete())
                return
            # Not a plan decision — clear pending and proceed normally
            agent._pending_plan_text = None
            agent._pending_plan_user_input = None

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
                chat.add_system("Reasoning trace hidden. Use /think to re-enable.")
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

    @on(UserQuestionRequest)
    def _on_user_question_request(self, msg: UserQuestionRequest) -> None:
        """Render a blocking agent question inside the chat UI."""
        chat = self.query_one("#chat-scroll", ChatScroll)
        chat.add_system(f"Agent question:\n{msg.question}")
        if msg.options:
            chat.add_system(f"Options: {', '.join(msg.options)}")
        chat.add_status("Waiting for your answer in the input box.")
        try:
            self.query_one("#chat-input", TextArea).focus()
        except NoMatches:
            pass

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
        self._state.finish_turn()

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

        # Check if plan approval is pending
        if self._state.agent and self._state.agent.has_pending_plan:
            chat.add_system(
                "Plan awaiting your decision:\n"
                "  1. Start implementing\n"
                "  2. Edit the plan\n"
                "  3. Ask a question\n"
                "  4. Cancel"
            )

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
        bar.theme_name = self._state.config.theme
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
            bar.model_name = manifest.display_name if manifest else _short_model_ref(ep.model_id)
            bar.endpoint = _endpoint_summary(ep)
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
    show_thinking: bool = True,
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
    think: bool = True,
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
