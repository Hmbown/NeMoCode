# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Interactive REPL for `nemo code` — the primary interactive experience.

Provides a streaming, tool-aware conversation loop with:
  - Multi-line input (triple-quote delimiters)
  - Streaming output with Markdown rendering
  - Tool call/result display with Rich panels
  - Slash commands for session control
  - Context window tracking
  - Graceful cancellation (Ctrl+C cancels current turn, not the session)

Uses prompt_toolkit for input when available, falls back to builtin input().
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from nemocode import __version__
from nemocode.config.schema import NeMoCodeConfig
from nemocode.core.context import ContextManager
from nemocode.core.metrics import MetricsCollector, RequestMetrics
from nemocode.core.scheduler import AgentEvent
from nemocode.workflows.code_agent import CodeAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Console instance — shared by all rendering functions
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# prompt_toolkit: optional dependency
# ---------------------------------------------------------------------------
_HAS_PROMPT_TOOLKIT = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    pass


# ===================================================================
# Input handling
# ===================================================================


def _get_history_path() -> Path:
    """Return the path for REPL command history persistence."""
    history_dir = Path(
        os.environ.get(
            "NEMOCODE_DATA_DIR",
            "~/.local/share/nemocode",
        )
    ).expanduser()
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / "repl_history"


class _InputReader:
    """Abstraction over prompt_toolkit / builtin input().

    Handles:
      - Single-line input with prompt
      - Multi-line mode triggered by triple-quote delimiters
      - Command history (when prompt_toolkit is available)
      - Ctrl+C / Ctrl+D handling
    """

    def __init__(self) -> None:
        self._session = None
        if _HAS_PROMPT_TOOLKIT:
            try:
                history = FileHistory(str(_get_history_path()))
                self._session = PromptSession(history=history)
            except Exception:
                # If history file is unwritable or any other issue, run without history
                self._session = PromptSession()

    async def read(self) -> str | None:
        """Read user input. Returns None on EOF (Ctrl+D). Raises KeyboardInterrupt on Ctrl+C."""
        raw = await self._read_line()
        if raw is None:
            return None

        stripped = raw.strip()

        # Multi-line mode: user starts with triple-quote
        if stripped.startswith('"""'):
            return await self._read_multiline(stripped)

        return raw

    async def _read_line(self) -> str | None:
        """Read a single line from the user."""
        if self._session is not None:
            try:
                return await self._session.prompt_async(
                    HTML("<ansigreen><b>&gt; </b></ansigreen>"),
                )
            except EOFError:
                return None
            # KeyboardInterrupt propagates up intentionally
        else:
            try:
                return input("> ")
            except EOFError:
                return None

    async def _read_multiline(self, first_line: str) -> str:
        """Collect lines until a closing triple-quote is found."""
        after_open = first_line[3:]
        if '"""' in after_open:
            return after_open[: after_open.index('"""')]

        lines: list[str] = []
        if after_open.strip():
            lines.append(after_open)

        while True:
            try:
                if self._session is not None:
                    line = await self._session.prompt_async(
                        HTML("<ansicyan><b>... </b></ansicyan>"),
                    )
                else:
                    line = input("... ")
            except EOFError:
                # Treat EOF mid-multiline as closing the block
                break
            except KeyboardInterrupt:
                # Cancel the multiline input entirely
                console.print("[dim]Multi-line input cancelled.[/dim]")
                return ""

            stripped = line.strip()
            if stripped == '"""':
                break
            # Allow closing """ at end of a line with content before it
            if stripped.endswith('"""'):
                lines.append(line[: line.rindex('"""')])
                break
            lines.append(line)

        return "\n".join(lines)


# ===================================================================
# Slash command handling
# ===================================================================

_HELP_TEXT = """\
[bold]Session:[/bold]
  [cyan]/help[/cyan]               Show this help message
  [cyan]/think[/cyan]              Toggle thinking trace display
  [cyan]/compact[/cyan]            Compact conversation (free context window)
  [cyan]/reset[/cyan]              Reset conversation
  [cyan]/cost[/cyan]               Session cost and token usage
  [cyan]/quit[/cyan]               Exit

[bold]Configuration:[/bold]
  [cyan]/endpoint[/cyan]           List endpoints  |  [cyan]/endpoint <name>[/cyan]  Switch
  [cyan]/formation[/cyan]          List formations  |  [cyan]/formation <name>[/cyan] Activate
  [cyan]/model[/cyan]              Show current model details
  [cyan]/hardware[/cyan]           Show detected hardware

[bold]Input:[/bold]
  Enter              Send message
  \"\"\"...\"\"\"            Multi-line input
  Ctrl+C             Cancel current response
  Ctrl+D             Exit
"""


class _SlashDispatcher:
    """Handles slash commands that modify session state."""

    def __init__(self, state: _ReplState) -> None:
        self._state = state

    def dispatch(self, line: str) -> bool:
        """Process a slash command. Returns True if the REPL should continue, False to exit."""
        parts = line.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        handler = {
            "/help": self._cmd_help,
            "/think": self._cmd_think,
            "/compact": self._cmd_compact,
            "/reset": self._cmd_reset,
            "/cost": self._cmd_cost,
            "/endpoint": self._cmd_endpoint,
            "/formation": self._cmd_formation,
            "/model": self._cmd_model,
            "/hardware": self._cmd_hardware,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/q": self._cmd_quit,
        }.get(cmd)

        if handler is None:
            console.print(f"[yellow]Unknown command: {cmd}. Type /help for commands.[/yellow]")
            return True

        return handler(arg)

    def _cmd_help(self, _arg: str) -> bool:
        console.print(_HELP_TEXT)
        return True

    def _cmd_think(self, _arg: str) -> bool:
        self._state.show_thinking = not self._state.show_thinking
        status = "on" if self._state.show_thinking else "off"
        console.print(f"[dim]Thinking trace: {status}[/dim]")
        return True

    def _cmd_compact(self, _arg: str) -> bool:
        try:
            self._state.agent.compact()
            console.print("[dim]Conversation compacted. Older messages trimmed.[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Compact failed: {exc}[/yellow]")
        return True

    def _cmd_reset(self, _arg: str) -> bool:
        self._state.agent.reset()
        self._state.turn_count = 0
        console.print("[dim]Conversation reset.[/dim]")
        return True

    def _cmd_cost(self, _arg: str) -> bool:
        mc = self._state.metrics
        summary = mc.summary()

        table = Table(title="Session Cost", show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("API requests", str(summary["requests"]))
        table.add_row("Prompt tokens", f"{summary['prompt_tokens']:,}")
        table.add_row("Completion tokens", f"{summary['completion_tokens']:,}")
        table.add_row("Total tokens", f"{summary['total_tokens']:,}")
        table.add_row("Estimated cost", f"${summary['estimated_cost_usd']:.6f}")
        table.add_row("Session duration", f"{summary['session_duration_s']:.0f}s")

        console.print(table)
        return True

    def _cmd_endpoint(self, arg: str) -> bool:
        if not arg:
            # List available endpoints
            endpoints = list(self._state.config.endpoints.keys())
            current = self._state.config.default_endpoint
            console.print("[bold]Available endpoints:[/bold]")
            for ep_name in endpoints:
                marker = " [green](active)[/green]" if ep_name == current else ""
                ep = self._state.config.endpoints[ep_name]
                console.print(f"  [cyan]{ep_name}[/cyan]{marker}  {ep.model_id}")
            return True

        if arg not in self._state.config.endpoints:
            console.print(
                f"[red]Unknown endpoint: {arg}[/red]\n"
                f"[dim]Available: {', '.join(self._state.config.endpoints.keys())}[/dim]"
            )
            return True

        self._state.config.default_endpoint = arg
        self._state.config.active_formation = None
        # Rebuild the agent with the new endpoint — reset is necessary because
        # the scheduler caches providers per-session and we want a clean switch
        self._state.rebuild_agent()
        ep = self._state.config.endpoints[arg]
        console.print(f"[dim]Switched to endpoint: {arg} ({ep.model_id})[/dim]")
        return True

    def _cmd_formation(self, arg: str) -> bool:
        if not arg:
            formations = list(self._state.config.formations.keys())
            current = self._state.config.active_formation
            if not formations:
                console.print("[dim]No formations configured.[/dim]")
                return True
            console.print("[bold]Available formations:[/bold]")
            for f_name in formations:
                marker = " [green](active)[/green]" if f_name == current else ""
                f = self._state.config.formations[f_name]
                roles = ", ".join(s.role.value for s in f.slots)
                console.print(f"  [cyan]{f_name}[/cyan]{marker}  [{roles}]")
            return True

        if arg == "off" or arg == "none":
            self._state.config.active_formation = None
            self._state.rebuild_agent()
            console.print("[dim]Formation deactivated. Using single endpoint mode.[/dim]")
            return True

        if arg not in self._state.config.formations:
            console.print(
                f"[red]Unknown formation: {arg}[/red]\n"
                f"[dim]Available: {', '.join(self._state.config.formations.keys())}[/dim]"
            )
            return True

        self._state.config.active_formation = arg
        self._state.rebuild_agent()
        f = self._state.config.formations[arg]
        roles = ", ".join(s.role.value for s in f.slots)
        console.print(f"[dim]Activated formation: {arg} [{roles}][/dim]")
        return True

    def _cmd_model(self, _arg: str) -> bool:
        ep_name = self._state.config.default_endpoint
        ep = self._state.config.endpoints.get(ep_name)
        if not ep:
            console.print("[yellow]No endpoint configured.[/yellow]")
            return True
        m = self._state.config.manifests.get(ep.model_id)
        console.print(f"\n[bold]Model:[/bold] {ep.model_id}")
        if m:
            console.print(f"  Display name: {m.display_name}")
            console.print(f"  Architecture: {m.arch.value}")
            if m.moe.total_params_b:
                console.print(
                    f"  Parameters: {m.moe.active_params_b:.0f}B active"
                    f" / {m.moe.total_params_b:.0f}B total"
                )
                console.print(f"  Precision: {m.moe.precision}")
            console.print(f"  Context: {m.context_window:,} tokens")
            console.print(f"  Tools: {'yes' if m.supports_tools else 'no'}")
            if m.reasoning.supports_thinking:
                console.print(
                    f"  Reasoning: enabled (budget control: "
                    f"{'yes' if m.reasoning.supports_budget_control else 'no'})"
                )
        return True

    def _cmd_hardware(self, _arg: str) -> bool:
        try:
            from nemocode.core.hardware import detect_hardware

            profile = detect_hardware()
            console.print(f"\n{profile.summary()}")
            rec = profile.recommend_formation()
            console.print(f"\n[bold]Recommended formation:[/bold] [cyan]{rec}[/cyan]")
            local = profile.recommend_local_models()
            if local:
                console.print(f"[bold]Local models:[/bold] {', '.join(local)}")
        except Exception as e:
            console.print(f"[yellow]Hardware detection failed: {e}[/yellow]")
        return True

    def _cmd_quit(self, _arg: str) -> bool:
        return False


# ===================================================================
# Session state
# ===================================================================


class _ReplState:
    """Mutable state bag for the REPL session."""

    def __init__(
        self,
        config: NeMoCodeConfig,
        show_thinking: bool,
        auto_yes: bool,
    ) -> None:
        self.config = config
        self.show_thinking = show_thinking
        self.auto_yes = auto_yes
        self.turn_count: int = 0
        self.metrics = MetricsCollector()
        self.context_mgr = ContextManager(
            context_window=self._resolve_context_window(),
        )
        self.agent = self._build_agent()
        self._cancelled = False

    def _resolve_context_window(self) -> int:
        """Determine context window size from manifest or default."""
        ep_name = self.config.default_endpoint
        ep = self.config.endpoints.get(ep_name)
        if ep:
            manifest = self.config.manifests.get(ep.model_id)
            if manifest:
                return manifest.context_window
        return 128_000  # conservative default

    def _build_agent(self) -> CodeAgent:
        confirm_fn = _auto_confirm if self.auto_yes else _interactive_confirm
        return CodeAgent(config=self.config, confirm_fn=confirm_fn)

    def rebuild_agent(self) -> None:
        """Rebuild the agent after config changes (endpoint/formation switch)."""
        self.context_mgr = ContextManager(
            context_window=self._resolve_context_window(),
        )
        self.agent = self._build_agent()
        self.turn_count = 0

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def clear_cancel(self) -> None:
        self._cancelled = False


# ===================================================================
# Confirmation functions
# ===================================================================


async def _auto_confirm(tool_name: str, args: dict) -> bool:
    return True


async def _interactive_confirm(tool_name: str, args: dict) -> bool:
    """Ask user for confirmation before executing a tool."""
    args_preview = json.dumps(args, indent=2)[:500]
    console.print(f"\n[yellow]Allow [bold]{tool_name}[/bold]?[/yellow]")
    console.print(f"[dim]{args_preview}[/dim]")
    try:
        response = console.input("[yellow]  [y/N]: [/yellow]")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() in ("y", "yes")


# ===================================================================
# Event rendering
# ===================================================================


class _TurnRenderer:
    """Renders AgentEvents for a single conversation turn.

    Accumulates text and thinking tokens, then renders the final
    assistant message as Markdown when the turn completes.
    Streams tool calls/results immediately as they arrive.
    """

    def __init__(self, state: _ReplState) -> None:
        self._state = state
        self._text_buf: str = ""
        self._think_buf: str = ""
        self._is_streaming_text: bool = False
        self._last_usage: dict[str, int] = {}
        self._tool_call_count: int = 0
        self._tool_error_count: int = 0
        self._turn_start: float = time.time()
        self._first_token_time: float | None = None
        self._current_model_id: str = ""

    def render_event(self, event: AgentEvent) -> None:
        """Render a single event. Called for each event from the agent loop."""
        if event.kind == "text":
            self._on_text(event)
        elif event.kind == "thinking":
            self._on_thinking(event)
        elif event.kind == "phase":
            self._on_phase(event)
        elif event.kind == "tool_call":
            self._on_tool_call(event)
        elif event.kind == "tool_result":
            self._on_tool_result(event)
        elif event.kind == "usage":
            self._on_usage(event)
        elif event.kind == "error":
            self._on_error(event)

    def finalize(self) -> None:
        """Called when the turn is complete. Renders accumulated Markdown and usage summary."""
        if self._text_buf.strip():
            self._flush_streaming_text()
        self._record_metrics()

    def _on_text(self, event: AgentEvent) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.time()
        self._text_buf += event.text
        console.print(event.text, end="", highlight=False)
        self._is_streaming_text = True

    def _on_thinking(self, event: AgentEvent) -> None:
        self._think_buf += event.thinking
        if self._state.show_thinking:
            console.print(event.thinking, end="", style="dim")

    def _on_phase(self, event: AgentEvent) -> None:
        if self._is_streaming_text:
            console.print()  # newline after streamed text
            self._is_streaming_text = False
        role_label = event.role.value if event.role else "agent"
        console.print(f"\n[bold blue]--- {role_label}: {event.text} ---[/bold blue]")

    def _on_tool_call(self, event: AgentEvent) -> None:
        self._tool_call_count += 1
        if self._is_streaming_text:
            console.print()  # newline after any streamed text
            self._is_streaming_text = False

        # Build a readable args display
        args_str = json.dumps(event.tool_args, indent=2)
        # Truncate very long args to keep the terminal readable
        if len(args_str) > 1500:
            args_str = args_str[:1500] + "\n... (truncated)"

        console.print(
            Panel(
                Syntax(args_str, "json", theme="monokai", word_wrap=True)
                if len(args_str) < 1500
                else Text(args_str, style="dim"),
                title=f"[bold cyan]{event.tool_name}[/bold cyan]",
                border_style="cyan",
                expand=False,
                padding=(0, 1),
            )
        )

    def _on_tool_result(self, event: AgentEvent) -> None:
        result_text = event.tool_result
        is_error = event.is_error

        if is_error:
            self._tool_error_count += 1

        # Truncate very long results
        if len(result_text) > 3000:
            result_text = result_text[:3000] + "\n... (truncated)"

        style = "red" if is_error else "green"
        label = "Error" if is_error else "Result"

        # For short results, inline. For longer ones, use a panel.
        if len(result_text) < 200 and "\n" not in result_text:
            console.print(f"  [{style}]{label}: {result_text}[/{style}]")
        else:
            console.print(
                Panel(
                    Text(result_text, style=style, overflow="fold"),
                    title=f"[{style}]{label}[/{style}]",
                    border_style=style,
                    expand=False,
                    padding=(0, 1),
                )
            )

    def _on_usage(self, event: AgentEvent) -> None:
        self._last_usage = event.usage
        # Resolve model ID for cost tracking from the endpoint config
        ep_name = self._state.config.default_endpoint
        ep = self._state.config.endpoints.get(ep_name)
        if ep:
            self._current_model_id = ep.model_id

    def _on_error(self, event: AgentEvent) -> None:
        if self._is_streaming_text:
            console.print()
            self._is_streaming_text = False
        console.print(f"\n[bold red]Error: {event.text}[/bold red]")

    def _flush_streaming_text(self) -> None:
        """Ensure a newline after streamed text output."""
        if self._is_streaming_text:
            console.print()  # final newline
            self._is_streaming_text = False

    def _record_metrics(self) -> None:
        """Record metrics for this turn into the collector."""
        prompt_tokens = self._last_usage.get("prompt_tokens", 0)
        completion_tokens = self._last_usage.get("completion_tokens", 0)
        total_time = (time.time() - self._turn_start) * 1000
        ttft = 0.0
        if self._first_token_time is not None:
            ttft = (self._first_token_time - self._turn_start) * 1000

        metrics = RequestMetrics(
            model_id=self._current_model_id,
            endpoint_name=self._state.config.default_endpoint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_to_first_token_ms=ttft,
            total_time_ms=total_time,
            tool_calls=self._tool_call_count,
            tool_errors=self._tool_error_count,
        )
        self._state.metrics.record(metrics)


# ===================================================================
# Context usage display
# ===================================================================


def _format_context_usage(state: _ReplState) -> str:
    """Build a context usage string like: [Context: 45K / 128K tokens (35.2%)]"""
    # Estimate from session messages if accessible; fall back to cumulative usage
    try:
        sessions = state.agent.sessions
        total_tokens = 0
        for session in sessions.values():
            total_tokens += state.context_mgr.usage(session.messages)
    except Exception:
        # If we can't access internals, use the metrics collector
        total_tokens = state.metrics.total_tokens

    ctx_window = state.context_mgr.context_window
    fraction = total_tokens / ctx_window if ctx_window > 0 else 0.0
    pct = fraction * 100

    # Format token counts with K/M suffixes for readability
    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.0f}K"
        return str(n)

    used_str = _fmt_tokens(total_tokens)
    window_str = _fmt_tokens(ctx_window)

    # Color based on usage level
    if pct > 80:
        color = "red"
    elif pct > 50:
        color = "yellow"
    else:
        color = "dim"

    return f"[{color}][Context: {used_str} / {window_str} tokens ({pct:.1f}%)][/{color}]"


# ===================================================================
# Session auto-save
# ===================================================================


def _auto_save_session(state: _ReplState) -> None:
    """Persist the current REPL session to disk after each turn.

    Saves silently -- errors are logged but do not interrupt the user.
    Includes metadata with the current working directory, endpoint, and
    git branch (if available).
    """
    try:
        from nemocode.core.persistence import save_session
        from nemocode.workflows.code_agent import _detect_git_context

        sessions = state.agent.sessions
        for session in sessions.values():
            git_ctx = _detect_git_context()
            git_branch = ""
            for line in git_ctx.splitlines():
                if line.startswith("Git branch:"):
                    git_branch = line.split(":", 1)[1].strip()
                    break

            metadata = {
                "cwd": str(Path.cwd()),
                "endpoint": state.config.default_endpoint,
            }
            if git_branch:
                metadata["git_branch"] = git_branch
            if state.config.active_formation:
                metadata["formation"] = state.config.active_formation

            save_session(session, metadata=metadata)
    except Exception as e:
        logger.debug("Session auto-save failed: %s", e)


# ===================================================================
# Welcome banner
# ===================================================================


_NVIDIA_GREEN = "bright_green"

_ASCII_LOGO = r"""[bright_green]
 _   _       __  __        ____          _
| \ | | ___ |  \/  | ___  / ___|___   __| | ___
|  \| |/ _ \| |\/| |/ _ \| |   / _ \ / _` |/ _ \
| |\  |  __/| |  | | (_) | |__| (_) | (_| |  __/
|_| \_|\___||_|  |_|\___/ \____\___/ \__,_|\___|[/bright_green]"""


def _print_banner(state: _ReplState) -> None:
    """Print the NVIDIA-themed welcome banner."""
    ep_name = state.config.default_endpoint
    ep = state.config.endpoints.get(ep_name)
    model_id = ep.model_id if ep else "unknown"
    formation = state.config.active_formation

    # Resolve model display info
    manifest = state.config.manifests.get(model_id) if ep else None
    model_display = manifest.display_name if manifest else model_id
    ctx_display = f"{manifest.context_window:,}" if manifest else "128K"
    arch_info = ""
    if manifest and manifest.moe.total_params_b:
        arch_info = (
            f"{manifest.moe.active_params_b:.0f}B active / {manifest.moe.total_params_b:.0f}B total"
        )

    console.print(_ASCII_LOGO)
    console.print()

    # Status info
    console.print(
        f"  [bold bright_green]v{__version__}[/bold bright_green]"
        f"  [dim]Powered by NVIDIA Nemotron 3[/dim]"
    )
    console.print()
    console.print(
        f"  [bold]Model:[/bold]     {model_display}"
        + (f"  [dim]({arch_info})[/dim]" if arch_info else "")
    )
    console.print(f"  [bold]Endpoint:[/bold]  {ep_name}  [dim]context: {ctx_display} tokens[/dim]")

    if formation:
        f_config = state.config.formations.get(formation)
        roles = ", ".join(s.role.value for s in f_config.slots) if f_config else "?"
        console.print(f"  [bold]Formation:[/bold] {formation}  [dim][{roles}][/dim]")

    cwd = Path.cwd()
    console.print(f"  [bold]Directory:[/bold] {cwd}")
    console.print()
    console.print(
        '  [dim]Type /help for commands, """ for multi-line, '
        "Ctrl+C to cancel, Ctrl+D to exit.[/dim]"
    )
    console.print(
        "  [dim italic]NeMoCode is a community project, "
        "not an official NVIDIA product.[/dim italic]"
    )
    console.print()


# ===================================================================
# Main REPL loop
# ===================================================================


async def run_repl(
    config: NeMoCodeConfig,
    endpoint_name: str | None,
    formation_name: str | None,
    show_thinking: bool,
    auto_yes: bool,
) -> None:
    """Run the interactive REPL. This is the main entry point called by code_cmd.

    The REPL loops forever until the user types /quit, presses Ctrl+D,
    or presses Ctrl+C twice in quick succession at the prompt.
    """
    # Apply overrides from CLI flags
    if endpoint_name:
        config.default_endpoint = endpoint_name
    if formation_name:
        config.active_formation = formation_name

    state = _ReplState(
        config=config,
        show_thinking=show_thinking,
        auto_yes=auto_yes,
    )
    reader = _InputReader()
    commands = _SlashDispatcher(state)

    # Check first-run / missing API key
    from nemocode.core.first_run import check_and_run_first_run

    check_and_run_first_run()

    _print_banner(state)

    # Track Ctrl+C at prompt: two in quick succession means exit
    last_interrupt_time: float = 0.0

    while True:
        # ---- Read input ----
        try:
            raw_input = await reader.read()
        except KeyboardInterrupt:
            now = time.time()
            if now - last_interrupt_time < 1.5:
                # Two Ctrl+C within 1.5 seconds => exit
                console.print("\n[dim]Goodbye.[/dim]")
                return
            last_interrupt_time = now
            console.print("\n[dim]Press Ctrl+C again to exit, or type /quit.[/dim]")
            continue

        # Ctrl+D (EOF)
        if raw_input is None:
            console.print("\n[dim]Goodbye.[/dim]")
            return

        stripped = raw_input.strip()

        # Empty input: ignore
        if not stripped:
            continue

        # Slash commands
        if stripped.startswith("/"):
            should_continue = commands.dispatch(stripped)
            if not should_continue:
                console.print("[dim]Goodbye.[/dim]")
                return
            continue

        # ---- Execute turn ----
        state.turn_count += 1
        state.clear_cancel()
        renderer = _TurnRenderer(state)

        try:
            await _run_turn(state, stripped, renderer)
        except KeyboardInterrupt:
            # Ctrl+C during streaming — the turn is cancelled but the session continues
            console.print("\n[dim]Turn cancelled.[/dim]")
        except Exception as exc:
            logger.exception("Unexpected error during turn")
            console.print(f"\n[bold red]Unexpected error: {exc}[/bold red]")
        finally:
            renderer.finalize()

        # Auto-save session after each turn
        _auto_save_session(state)

        # Show context usage after each turn
        console.print(_format_context_usage(state))
        console.print()  # blank line before next prompt


async def _run_turn(state: _ReplState, user_input: str, renderer: _TurnRenderer) -> None:
    """Execute a single conversation turn, streaming events to the renderer.

    Handles Ctrl+C gracefully: sets the cancelled flag and drains remaining events
    rather than raising, so the session stays in a consistent state.
    """
    # Show a spinner until the first event arrives
    first_event_received = False

    # We wrap the async generator consumption in a task so we can
    # catch KeyboardInterrupt cleanly on the outer level.
    async for event in state.agent.run(user_input):
        if state.cancelled:
            # Drain remaining events without rendering — the user cancelled
            continue

        if not first_event_received:
            first_event_received = True

        try:
            renderer.render_event(event)
        except KeyboardInterrupt:
            # User pressed Ctrl+C during rendering — mark cancelled, keep draining
            state.cancel()
            console.print("\n[dim]Cancelling...[/dim]")
            continue


# ===================================================================
# Entry point called from code.py
# ===================================================================


def start_repl(
    endpoint: str | None,
    formation: str | None,
    think: bool,
    yes: bool,
) -> None:
    """Synchronous entry point: loads config and runs the async REPL loop.

    This is the function that code_cmd calls when invoked without a prompt argument.
    """
    from nemocode.config import load_config

    config = load_config()

    try:
        asyncio.run(
            run_repl(
                config=config,
                endpoint_name=endpoint,
                formation_name=formation,
                show_thinking=think,
                auto_yes=yes,
            )
        )
    except KeyboardInterrupt:
        # Final safety net — clean exit on Ctrl+C
        console.print("\n[dim]Goodbye.[/dim]")
