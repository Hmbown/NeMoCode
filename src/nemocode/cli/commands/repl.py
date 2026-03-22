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
import logging
import os
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from nemocode import __version__
from nemocode.cli.render import (
    EventRenderer,
    format_confirm_summary,
    render_confirm_detail,
    tool_result_has_embedded_error,
)
from nemocode.cli.theme import format_key_label, get_theme
from nemocode.config.agents import resolve_agent_reference
from nemocode.config.schema import AgentMode, NeMoCodeConfig
from nemocode.core.context import ContextManager
from nemocode.core.metrics import MetricsCollector, RequestMetrics
from nemocode.core.scheduler import AgentEvent
from nemocode.core.sessions import TurnBoundary
from nemocode.workflows.code_agent import CodeAgent

try:
    from nemocode.cli.render import set_render_theme
except ImportError:  # pragma: no cover - compatibility with in-flight theme work

    def set_render_theme(_theme_name: str) -> None:
        return None


logger = logging.getLogger(__name__)
_NVIDIA_GREEN = "bright_green"


def _repl_theme(config: NeMoCodeConfig) -> object:
    return get_theme(config.theme)


def _apply_repl_theme(config: NeMoCodeConfig) -> None:
    global _NVIDIA_GREEN
    theme = _repl_theme(config)
    _NVIDIA_GREEN = theme.accent_rich
    set_render_theme(config.theme)


def _ptk_binding_args(spec: str) -> tuple[str, ...]:
    normalized = spec.strip().lower().replace("control+", "ctrl+")
    if normalized in {"tab", "escape", "enter"}:
        return (normalized,)
    if normalized.startswith("ctrl+") and len(normalized) == 6:
        return (f"c-{normalized[-1]}",)
    if normalized.startswith("alt+") and len(normalized) == 5:
        return ("escape", normalized[-1])
    raise ValueError(f"Unsupported prompt_toolkit keybinding: {spec}")


# ---------------------------------------------------------------------------
# Console instance — shared by all rendering functions
# ---------------------------------------------------------------------------
console = Console()


def _save_project_default_endpoint(endpoint_name: str) -> bool:
    """Persist the default endpoint to the project's .nemocode.yaml.

    Updates only the ``default_endpoint`` key (and clears ``active_formation``).
    Returns True on success, False on error.
    """
    config_path = Path.cwd() / ".nemocode.yaml"
    try:
        try:
            raw = yaml.safe_load(config_path.read_text()) or {}
        except FileNotFoundError:
            raw = {}
        raw["default_endpoint"] = endpoint_name
        raw["active_formation"] = None
        config_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
        return True
    except Exception as e:
        logger.warning("Failed to save project config: %s", e)
        return False


def _short_model_ref(model_id: str) -> str:
    """Compact long local model paths for endpoint displays."""
    if not model_id:
        return "-"
    return Path(model_id).name if model_id.startswith("/") else model_id


def _endpoint_summary(endpoint: object) -> str:
    name = getattr(endpoint, "name", "") or ""
    model_id = _short_model_ref(getattr(endpoint, "model_id", "") or "")
    if name and model_id and name != model_id:
        return f"{name} · {model_id}"
    return name or model_id or "-"


def _turn_preview(text: str, limit: int = 60) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}…"


# ---------------------------------------------------------------------------
# prompt_toolkit: optional dependency
# ---------------------------------------------------------------------------
_HAS_PROMPT_TOOLKIT = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    pass


# ===================================================================
# Slash command autocomplete
# ===================================================================


class _SlashCompleter(Completer):
    """Tab-completion for slash commands, their arguments, and file paths."""

    def __init__(self, state: _ReplState | None = None) -> None:
        self._state = state
        self._path_completer: Completer | None = None
        try:
            from prompt_toolkit.completion import PathCompleter

            self._path_completer = PathCompleter(expanduser=True)
        except ImportError:
            pass

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Slash commands
        if text.startswith("/"):
            yield from self._complete_slash(text, document, complete_event)
            return

        # File path completion for bare text input
        if self._path_completer:
            yield from self._path_completer.get_completions(document, complete_event)

    def _complete_slash(self, text, document, complete_event):
        parts = text.split(None, 1)
        cmd = parts[0]

        if len(parts) == 1 and not text.endswith(" "):
            # Complete command names
            for name in _SLASH_COMMANDS:
                if name.startswith(cmd):
                    yield Completion(name, start_position=-len(cmd))
        elif len(parts) >= 1:
            # Complete arguments
            arg = parts[1] if len(parts) > 1 else ""
            cmd_name = parts[0].lower()
            if cmd_name == "/endpoint" and self._state:
                for ep in self._state.config.endpoints:
                    if ep.startswith(arg):
                        yield Completion(ep, start_position=-len(arg))
            elif cmd_name == "/formation" and self._state:
                for f in list(self._state.config.formations) + ["off"]:
                    if f.startswith(arg):
                        yield Completion(f, start_position=-len(arg))
            elif cmd_name == "/agent" and self._state:
                for agent_name, agent in self._state.config.agents.items():
                    if agent.mode == AgentMode.SUBAGENT:
                        continue
                    for candidate in [agent_name, *agent.aliases]:
                        if candidate.startswith(arg):
                            yield Completion(candidate, start_position=-len(arg))
            elif cmd_name == "/resume":
                try:
                    from nemocode.core.persistence import list_sessions

                    for s in list_sessions(10):
                        sid = s["id"]
                        if sid.startswith(arg):
                            yield Completion(sid, start_position=-len(arg))
                except Exception:
                    pass


# All known slash command names (for autocomplete)
_SLASH_COMMANDS = [
    "/agent",
    "/help",
    "/think",
    "/compact",
    "/reset",
    "/undo",
    "/retry",
    "/cost",
    "/endpoint",
    "/formation",
    "/mode",
    "/model",
    "/hardware",
    "/doctor",
    "/resume",
    "/sessions",
    "/commit",
    "/review",
    "/snapshot",
    "/snapshots",
    "/revert",
    "/context",
    "/diff",
    "/quit",
    "/exit",
]


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
      - Tab to cycle modes, Escape to clear input
      - Bottom toolbar with mode and keyboard hints
      - Ctrl+C / Ctrl+D handling
    """

    def __init__(self, state: _ReplState | None = None) -> None:
        self._state = state
        self._session = None
        if _HAS_PROMPT_TOOLKIT:
            kwargs: dict = {}

            # Only wire up interactive features (keybindings, toolbar) in a real
            # terminal — prompt_toolkit leaks toolbar text into pipes otherwise.
            try:
                _is_tty = os.isatty(0)
            except (AttributeError, ValueError, OSError):
                _is_tty = False

            if state is not None and _is_tty:
                kb = KeyBindings()
                repl_keys = state.config.keybindings.repl

                @kb.add(*_ptk_binding_args(repl_keys.cycle_mode))
                def _tab_cycle(event):
                    """Cycle mode: code → plan → auto (unless completing a slash cmd)."""
                    buf = event.app.current_buffer
                    if buf.text.startswith("/"):
                        # Let completer handle it
                        buf.start_completion()
                        return
                    state.cycle_mode()
                    state.agent = state._build_agent()
                    event.app.invalidate()

                @kb.add(*_ptk_binding_args(repl_keys.clear_input))
                def _esc_clear(event):
                    """Clear the current input line."""
                    event.app.current_buffer.text = ""
                    event.app.current_buffer.cursor_position = 0

                @kb.add(*_ptk_binding_args(repl_keys.exit))
                def _exit_prompt(event):
                    """Exit the prompt using the configured key."""
                    event.app.exit(result=None)

                kwargs["key_bindings"] = kb
                kwargs["bottom_toolbar"] = self._toolbar
                kwargs["completer"] = _SlashCompleter(state)
                theme = _repl_theme(state.config)
                kwargs["style"] = Style.from_dict(
                    {
                        "bottom-toolbar": f"bg:{theme.status_bg} {theme.status_fg}",
                    }
                )

            try:
                history = FileHistory(str(_get_history_path()))
                self._session = PromptSession(history=history, **kwargs)
            except Exception:
                # If history file is unwritable or any other issue, run without history
                try:
                    self._session = PromptSession(**kwargs)
                except Exception:
                    pass  # No terminal available (e.g. Windows CI) — fall back to input()

    def _toolbar(self):
        """Bottom toolbar — always shows mode, model name, git branch, context %, tok/s."""
        if self._state is None:
            return ""
        s = self._state

        parts: list[str] = []
        theme = _repl_theme(s.config)

        # Mode indicator (always visible)
        mode = s.mode
        mode_colors = {"code": theme.accent_rich, "plan": "ansiyellow", "auto": "ansired"}
        mc = mode_colors.get(mode, theme.accent_rich)
        parts.append(f" <{mc}><b>▸ {mode}</b></{mc}>")

        # Model display name (instead of raw endpoint key)
        ep_name = s.config.default_endpoint
        ep = s.config.endpoints.get(ep_name)
        if ep:
            manifest = s.config.manifests.get(ep.model_id)
            display = manifest.display_name if manifest else _short_model_ref(ep.model_id)
            parts.append(f" <ansicyan>{display}</ansicyan>")
        else:
            parts.append(f" <ansicyan>{ep_name}</ansicyan>")
        parts.append(f" <ansiblue>{s.current_primary_agent_display()}</ansiblue>")

        # Git branch
        branch = _get_git_branch()
        if branch:
            parts.append(f" <ansimagenta>⎇ {branch}</ansimagenta>")

        # Context usage (always visible)
        try:
            total_tokens = 0
            for session in s.agent.sessions.values():
                total_tokens += s.context_mgr.usage(session.messages)
            total_tokens = max(total_tokens, s.metrics.total_tokens)
            ctx_window = s.context_mgr.context_window
            pct = (total_tokens / ctx_window * 100) if ctx_window > 0 else 0
            pct_color = "ansired" if pct > 80 else "ansiyellow" if pct > 50 else ""
            tag = f"<{pct_color}>" if pct_color else ""
            end_tag = f"</{pct_color}>" if pct_color else ""
            parts.append(f" {tag}ctx:{pct:.0f}%{end_tag}")
        except Exception:
            pass

        # During an active turn, show live elapsed timer instead of token count
        if s._turn_active and s._turn_start_time:
            elapsed = time.time() - s._turn_start_time
            parts.append(f" <ansiyellow>{elapsed:.0f}s</ansiyellow>")
        elif s.metrics.total_tokens > 0:
            # Token count (after first turn, when not streaming)
            t = s.metrics.total_tokens
            tok_str = (
                f"{t / 1_000_000:.1f}M"
                if t >= 1_000_000
                else (f"{t / 1_000:.0f}K" if t >= 1_000 else str(t))
            )
            parts.append(f" {tok_str}tok")

        # Last throughput
        last_tps = s.metrics.last_tokens_per_sec
        if last_tps > 0:
            parts.append(f" {last_tps:.0f} tok/s")

        return HTML(" │".join(parts))

    async def read(self, mode: str = "code") -> str | None:
        """Read user input. Returns None on EOF (Ctrl+D). Raises KeyboardInterrupt on Ctrl+C."""
        raw = await self._read_line(mode)
        if raw is None:
            return None

        stripped = raw.strip()

        # Multi-line mode: user starts with triple-quote
        if stripped.startswith('"""'):
            return await self._read_multiline(stripped)

        return raw

    async def _read_line(self, mode: str = "code") -> str | None:
        """Read a single line from the user."""
        if self._session is not None:
            # When state is available, use a callable prompt so Tab updates the
            # mode label in real time without restarting the prompt.
            if self._state is not None:
                theme = _repl_theme(self._state.config)

                def _dynamic_prompt():
                    m = self._state.mode
                    colors = {"code": theme.accent_rich, "plan": "ansiyellow", "auto": "ansired"}
                    c = colors.get(m, theme.accent_rich)
                    return HTML(f"\n<{c}><b>[{m}] ▸ </b></{c}>")

                try:
                    return await self._session.prompt_async(_dynamic_prompt)
                except EOFError:
                    return None
                # KeyboardInterrupt propagates up intentionally
            else:
                # Static prompt (no state — tests or fallback)
                mode_colors = {"code": "ansigreen", "plan": "ansiyellow", "auto": "ansired"}
                color = mode_colors.get(mode, "ansigreen")
                prompt_text = f"<{color}><b>[{mode}] ▸ </b></{color}>"
                try:
                    return await self._session.prompt_async(HTML(prompt_text))
                except EOFError:
                    return None
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


def _get_git_branch() -> str:
    """Get current git branch name (cached for toolbar)."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _help_text(state: _ReplState) -> str:
    keys = state.config.keybindings.repl
    return f"""\
[bold]Session:[/bold]
  [cyan]/help[/cyan]               Show this help message
  [cyan]/think[/cyan]              Toggle thinking trace display
  [cyan]/compact[/cyan]            Compact conversation (free context window)
  [cyan]/reset[/cyan]              Reset conversation
  [cyan]/undo[/cyan]               Undo the last file change
  [cyan]/retry[/cyan]              Revert and replay the last user turn
  [cyan]/cost[/cyan]               Session cost and token usage
  [cyan]/context[/cyan]            Context window usage details
  [cyan]/diff[/cyan]               Show git diff --stat inline
  [cyan]/mode[/cyan]               Cycle mode: code -> plan -> auto
  [cyan]/doctor[/cyan]             Run diagnostics (API, tools, hardware)
  [cyan]/quit[/cyan]               Exit

[bold]Skills:[/bold]
  [cyan]/commit[/cyan] [msg]       Generate commit message and commit
  [cyan]/review[/cyan] [ref]       Review uncommitted changes

[bold]Sessions:[/bold]
  [cyan]/sessions[/cyan]           List recent sessions
  [cyan]/resume[/cyan] <id>        Resume a saved session

[bold]Snapshots:[/bold]
  [cyan]/snapshot[/cyan] [label]   Create a git safety checkpoint
  [cyan]/snapshots[/cyan]          List available snapshots
  [cyan]/revert[/cyan] <id|last>   Revert to a snapshot or rewind the last turn

[bold]Modes:[/bold]
  [green]code[/green]    Ask before tool calls (default)
  [yellow]plan[/yellow]    Read-only planning, research subagents, approval gate
  [red]auto[/red]    Auto-approve everything — fast but risky

[bold]Configuration:[/bold]
  [cyan]/endpoint[/cyan]           List endpoints  |  [cyan]/endpoint <name> [--save][/cyan]  Switch
  [cyan]/formation[/cyan]          List formations  |  [cyan]/formation <name>[/cyan] Activate
  [cyan]/agent[/cyan]              List primary agents  |  [cyan]/agent <name>[/cyan]  Switch
  [cyan]/model[/cyan]              Show current model details
  [cyan]/hardware[/cyan]           Show detected hardware

[bold]Input:[/bold]
  Enter              Send message
  {format_key_label(keys.cycle_mode):<18}Cycle mode (or complete /commands)
  {format_key_label(keys.clear_input):<18}Clear input line
  \"\"\"...\"\"\"            Multi-line input
  Ctrl+C             Cancel current response
  {format_key_label(keys.exit):<18}Exit
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
            "/undo": self._cmd_undo,
            "/retry": self._cmd_retry,
            "/cost": self._cmd_cost,
            "/endpoint": self._cmd_endpoint,
            "/formation": self._cmd_formation,
            "/agent": self._cmd_agent,
            "/mode": self._cmd_mode,
            "/tab": self._cmd_mode,
            "/model": self._cmd_model,
            "/hardware": self._cmd_hardware,
            "/doctor": self._cmd_doctor,
            "/sessions": self._cmd_sessions,
            "/resume": self._cmd_resume,
            "/commit": self._cmd_commit,
            "/review": self._cmd_review,
            "/snapshot": self._cmd_snapshot,
            "/revert": self._cmd_revert,
            "/snapshots": self._cmd_snapshots,
            "/context": self._cmd_context,
            "/diff": self._cmd_diff,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/q": self._cmd_quit,
        }.get(cmd)

        if handler is None:
            console.print(f"[yellow]Unknown command: {cmd}. Type /help for commands.[/yellow]")
            return True

        return handler(arg)

    def _cmd_help(self, _arg: str) -> bool:
        console.print(_help_text(self._state))
        return True

    def _cmd_think(self, _arg: str) -> bool:
        self._state.show_thinking = not self._state.show_thinking
        status = "on" if self._state.show_thinking else "off"
        console.print(f"[dim]Thinking trace: {status}[/dim]")
        return True

    def _cmd_compact(self, _arg: str) -> bool:
        try:
            self._state.agent.compact()
            self._state.clear_turn_history()
            console.print("[dim]Conversation compacted. Older messages trimmed.[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Compact failed: {exc}[/yellow]")
        return True

    def _cmd_reset(self, _arg: str) -> bool:
        self._state.agent.reset()
        self._state.turn_count = 0
        self._state.clear_turn_history()
        console.print("[dim]Conversation reset.[/dim]")
        return True

    def _cmd_undo(self, _arg: str) -> bool:
        from nemocode.tools.fs import undo_last, undo_stack_depth

        depth = undo_stack_depth()
        if depth == 0:
            console.print("[dim]Nothing to undo.[/dim]")
            return True

        result = undo_last()
        if "error" in result:
            console.print(f"[red]Undo failed: {result['error']}[/red]")
        else:
            self._state.clear_turn_history()
            action = result.get("action", "reverted")
            path = result.get("path", "?")
            remaining = undo_stack_depth()
            console.print(f"[green]Undo: {action} {path}[/green]")
            if remaining > 0:
                console.print(f"[dim]  {remaining} more undo(s) available[/dim]")
        return True

    def _cmd_retry(self, _arg: str) -> bool:
        ok, message, retry_input = self._state.prepare_retry()
        style = "dim" if ok else "yellow"
        console.print(f"[{style}]{message}[/{style}]")
        if ok and retry_input is not None:
            self._state._queued_input = retry_input
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
        table.add_row("Avg latency", f"{summary['avg_latency_ms']:.0f}ms")
        if summary["avg_tokens_per_sec"] > 0:
            table.add_row("Avg tok/s", f"{summary['avg_tokens_per_sec']:.1f}")
        if summary["avg_ttft_ms"] > 0:
            table.add_row("Avg TTFT", f"{summary['avg_ttft_ms']:.0f}ms")
        table.add_row("Session duration", f"{summary['session_duration_s']:.0f}s")

        console.print(table)

        # Per-model breakdown (when formations are used)
        by_model: dict[str, dict[str, float]] = {}
        for r in mc._requests:
            model = r.model_id or "unknown"
            if model not in by_model:
                by_model[model] = {
                    "prompt": 0,
                    "completion": 0,
                    "cost": 0.0,
                    "requests": 0,
                    "tps_sum": 0.0,
                    "tps_count": 0,
                }
            by_model[model]["prompt"] += r.prompt_tokens
            by_model[model]["completion"] += r.completion_tokens
            by_model[model]["cost"] += r.estimated_cost
            by_model[model]["requests"] += 1
            if r.tokens_per_sec > 0:
                by_model[model]["tps_sum"] += r.tokens_per_sec
                by_model[model]["tps_count"] += 1

        if len(by_model) > 1:
            console.print("\n[bold]By model:[/bold]")
            for model, stats in by_model.items():
                p = int(stats["prompt"])
                c = int(stats["completion"])
                tps_str = ""
                if stats["tps_count"] > 0:
                    avg_tps = stats["tps_sum"] / stats["tps_count"]
                    tps_str = f"  {avg_tps:.0f} tok/s"
                console.print(
                    f"  [dim]{model}[/dim]  {p:,}p + {c:,}c"
                    f"  ${stats['cost']:.6f}  ({int(stats['requests'])} reqs){tps_str}"
                )

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
                console.print(f"  [cyan]{_endpoint_summary(ep)}[/cyan]{marker}")
            return True

        # Parse --save flag
        parts = arg.split()
        save = "--save" in parts
        ep_key = next((p for p in parts if p != "--save"), "")

        if ep_key not in self._state.config.endpoints:
            console.print(
                f"[red]Unknown endpoint: {ep_key}[/red]\n"
                f"[dim]Available: {', '.join(self._state.config.endpoints.keys())}[/dim]"
            )
            return True

        self._state.config.default_endpoint = ep_key
        self._state.config.active_formation = None
        # Rebuild the agent with the new endpoint — reset is necessary because
        # the scheduler caches providers per-session and we want a clean switch
        self._state.rebuild_agent()
        ep = self._state.config.endpoints[ep_key]
        console.print(f"[dim]Switched to endpoint: {_endpoint_summary(ep)}[/dim]")

        if save:
            if _save_project_default_endpoint(ep_key):
                console.print("[dim]Saved as default in .nemocode.yaml[/dim]")
            else:
                console.print("[yellow]Could not save to .nemocode.yaml[/yellow]")
        else:
            hint = f"[dim]Use [bold]/endpoint {ep_key} --save[/bold] to make permanent.[/dim]"
            console.print(hint)
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

    def _cmd_agent(self, arg: str) -> bool:
        primary_agents = {
            name: agent
            for name, agent in self._state.config.agents.items()
            if agent.mode != AgentMode.SUBAGENT
        }

        if not arg:
            if not primary_agents:
                console.print("[dim]No primary agents configured.[/dim]")
                return True
            console.print("[bold]Primary agents:[/bold]")
            current = self._state.current_primary_agent_name()
            for name, agent in sorted(primary_agents.items()):
                marker = " [green](active)[/green]" if name == current else ""
                display = f"  [dim]{agent.display_name}[/dim]" if agent.display_name else ""
                console.print(f"  [cyan]{name}[/cyan]{marker}{display}")
            return True

        resolved = resolve_agent_reference(primary_agents, arg) or arg
        agent = primary_agents.get(resolved)
        if agent is None:
            available = ", ".join(sorted(primary_agents.keys()))
            console.print(
                f"[red]Unknown primary agent: {arg}[/red]\n[dim]Available: {available}[/dim]"
            )
            return True

        self._state.agent_name = resolved
        self._state.rebuild_agent()
        label = agent.display_name or resolved
        console.print(f"[dim]Switched primary agent: {resolved} ({label})[/dim]")
        return True

    def _cmd_mode(self, _arg: str) -> bool:
        new_mode = self._state.cycle_mode()
        old_agent = self._state.agent
        new_agent = self._state._build_agent()
        self._state._transfer_sessions(old_agent, new_agent)
        # Preserve pending plan across mode switch
        new_agent._pending_plan_text = old_agent._pending_plan_text
        new_agent._pending_plan_user_input = old_agent._pending_plan_user_input
        self._state.agent = new_agent
        self._state.clear_turn_history()
        mode_desc = {
            "code": "ask before tools",
            "plan": "read-only planning + approval",
            "auto": "auto-approve",
        }
        console.print(f"[dim]Mode: {new_mode} ({mode_desc.get(new_mode, '')})[/dim]")
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

    def _cmd_doctor(self, _arg: str) -> bool:
        from nemocode.core.doctor import run_diagnostics

        report = run_diagnostics(self._state.config)
        status_icons = {
            "ok": "[green]OK[/green]",
            "warn": "[yellow]WARN[/yellow]",
            "fail": "[red]FAIL[/red]",
        }
        console.print("\n[bold]Diagnostics:[/bold]")
        for check in report.checks:
            icon = status_icons.get(check.status, "?")
            detail = f"  [dim]{check.detail}[/dim]" if check.detail else ""
            console.print(f"  {icon}  {check.name}{detail}")
        if report.ok:
            console.print("\n[green]All checks passed.[/green]")
        else:
            console.print("\n[yellow]Some checks need attention.[/yellow]")
        return True

    def _cmd_sessions(self, _arg: str) -> bool:
        from nemocode.core.persistence import list_sessions

        sessions = list_sessions(limit=10)
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
            return True
        console.print("[bold]Recent sessions:[/bold]")
        for s in sessions:
            import datetime

            ts = datetime.datetime.fromtimestamp(s.get("updated_at", 0))
            msgs = s.get("message_count", 0)
            sid = s["id"]
            ep = s.get("endpoint_name", "")
            console.print(
                f"  [cyan]{sid}[/cyan]  {msgs} msgs  {ep}  [dim]{ts:%Y-%m-%d %H:%M}[/dim]"
            )
        console.print("[dim]Use /resume <id> to load a session.[/dim]")
        return True

    def _cmd_resume(self, arg: str) -> bool:
        if not arg:
            console.print("[yellow]Usage: /resume <session-id>[/yellow]")
            return True
        from nemocode.config.schema import FormationRole
        from nemocode.core.persistence import load_session

        session = load_session(arg)
        if session is None:
            console.print(f"[red]Session not found: {arg}[/red]")
            return True

        # Inject the loaded session into the agent's scheduler
        self._state.agent._scheduler._sessions[FormationRole.EXECUTOR] = session
        self._state.clear_turn_history()
        msg_count = session.message_count()
        console.print(f"[dim]Resumed session {arg} ({msg_count} messages)[/dim]")
        return True

    def _cmd_commit(self, arg: str) -> bool:
        """Run the /commit skill."""
        self._state._pending_skill = ("commit", arg)
        return True

    def _cmd_review(self, arg: str) -> bool:
        """Run the /review skill."""
        self._state._pending_skill = ("review", arg)
        return True

    def _cmd_snapshot(self, arg: str) -> bool:
        """Create a git snapshot for safe rollback."""
        import asyncio as _aio

        try:
            loop = _aio.get_event_loop()
            snap = loop.run_until_complete(
                self._state.agent.snapshot_mgr.create_snapshot(arg or "manual")
            )
            if snap:
                console.print(
                    f"[green]Snapshot created: {snap.id} "
                    f"({snap.files_changed} file{'s' if snap.files_changed != 1 else ''})[/green]"
                )
            else:
                console.print("[dim]No changes to snapshot.[/dim]")
        except Exception as e:
            console.print(f"[red]Snapshot failed: {e}[/red]")
        return True

    def _cmd_snapshots(self, _arg: str) -> bool:
        """List available snapshots."""
        import asyncio as _aio

        try:
            loop = _aio.get_event_loop()
            snaps = loop.run_until_complete(self._state.agent.snapshot_mgr.list_snapshots())
            if not snaps:
                console.print("[dim]No snapshots available.[/dim]")
                return True
            console.print("[bold]Snapshots:[/bold]")
            for s in snaps:
                import datetime

                ts = datetime.datetime.fromtimestamp(s["timestamp"])
                console.print(
                    f"  [cyan]{s['id']}[/cyan]  {s['kind']}  "
                    f"{s['files_changed']} files  [dim]{ts:%H:%M:%S}[/dim]"
                )
            console.print("[dim]Use /revert <id> to restore.[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to list snapshots: {e}[/red]")
        return True

    def _cmd_revert(self, arg: str) -> bool:
        """Revert to a snapshot or rewind the last completed user turn."""
        if not arg:
            console.print("[yellow]Usage: /revert <snapshot-id> or /revert last[/yellow]")
            return True

        if arg == "last":
            ok, message = self._state.revert_last_turn()
            style = "green" if ok else "yellow"
            console.print(f"[{style}]{message}[/{style}]")
            return True

        # Try as a snapshot ID
        import asyncio as _aio

        try:
            loop = _aio.get_event_loop()
            result = loop.run_until_complete(self._state.agent.snapshot_mgr.restore_snapshot(arg))
            if "error" in result:
                console.print(f"[red]{result['error']}[/red]")
            else:
                self._state.clear_turn_history()
                console.print(
                    f"[green]Reverted to snapshot {result['restored']} ({result['kind']})[/green]"
                )
        except Exception as e:
            console.print(f"[red]Revert failed: {e}[/red]")
        return True

    def _cmd_context(self, _arg: str) -> bool:
        """Show current context window usage details."""
        s = self._state
        total_tokens = 0
        msg_count = 0
        try:
            for session in s.agent.sessions.values():
                total_tokens += s.context_mgr.usage(session.messages)
                msg_count += len(session.messages)
        except Exception:
            pass
        total_tokens = max(total_tokens, s.metrics.total_tokens)
        ctx_window = s.context_mgr.context_window
        pct = (total_tokens / ctx_window * 100) if ctx_window > 0 else 0
        remaining = ctx_window - total_tokens
        # Estimate remaining turns (rough: ~2K tokens per turn)
        est_turns = max(0, remaining // 2000) if remaining > 0 else 0

        console.print("[bold]Context Usage[/bold]")
        console.print(f"  Messages:         {msg_count}")
        console.print(f"  Tokens used:      {_fmt_tokens(total_tokens)}")
        console.print(f"  Context window:   {_fmt_tokens(ctx_window)}")
        pct_color = "red" if pct > 80 else "yellow" if pct > 50 else "green"
        console.print(f"  Usage:            [{pct_color}]{pct:.1f}%[/{pct_color}]")
        console.print(f"  Remaining:        ~{_fmt_tokens(remaining)}")
        console.print(f"  Est. turns left:  ~{est_turns}")
        return True

    def _cmd_diff(self, _arg: str) -> bool:
        """Show git diff --stat inline."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    console.print(f"[dim]{output}[/dim]")
                else:
                    console.print("[dim]No unstaged changes.[/dim]")
                # Also show staged
                staged = subprocess.run(
                    ["git", "diff", "--cached", "--stat"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if staged.returncode == 0 and staged.stdout.strip():
                    console.print("\n[bold]Staged:[/bold]")
                    console.print(f"[dim]{staged.stdout.strip()}[/dim]")
            else:
                console.print("[dim]Not in a git repository.[/dim]")
        except Exception as e:
            console.print(f"[red]git diff failed: {e}[/red]")
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
        show_thinking: bool = True,
        auto_yes: bool = False,
        agent_name: str | None = None,
    ) -> None:
        self.config = config
        self.show_thinking = show_thinking
        self.auto_yes = auto_yes
        resolved_agent = resolve_agent_reference(config.agents, agent_name) if agent_name else None
        if resolved_agent:
            self.agent_name: str | None = resolved_agent
        elif agent_name and agent_name in config.agents:
            self.agent_name = agent_name
        elif "build" in config.agents:
            self.agent_name = "build"
        else:
            self.agent_name = None
        self.turn_count: int = 0
        self.metrics = MetricsCollector()
        self.context_mgr = ContextManager(
            context_window=self._resolve_context_window(),
            model_id=self._resolve_model_id(),
        )
        # Mode: "code" (default), "plan" (read-only planning), "auto" (auto-approve all)
        self._modes = ["code", "plan", "auto"]
        self._mode_idx = 0
        self.agent = self._build_agent()
        self._cancelled = False
        self._pending_skill: tuple[str, str] | None = None
        self._auto_approve_remaining = False
        self._turn_active = False
        self._turn_start_time: float = 0.0
        self._active_turn: TurnBoundary | None = None
        self._last_turn: TurnBoundary | None = None
        self._queued_input: str | None = None

    def _resolve_context_window(self) -> int:
        """Determine context window size from manifest or default."""
        ep_name = self.config.default_endpoint
        ep = self.config.endpoints.get(ep_name)
        if ep:
            manifest = self.config.manifests.get(ep.model_id)
            if manifest:
                return manifest.context_window
        return 128_000  # conservative default

    def _resolve_model_id(self) -> str:
        ep = self.config.endpoints.get(self.config.default_endpoint)
        return ep.model_id if ep else ""

    def _build_agent(self) -> CodeAgent:
        if self.auto_yes or self.mode in ("auto", "plan"):
            confirm_fn = _auto_confirm
        else:
            confirm_fn = _interactive_confirm
        return CodeAgent(
            config=self.config,
            confirm_fn=confirm_fn,
            read_only=(self.mode == "plan"),
            agent_name=self.current_primary_agent_name(),
        )

    def _transfer_sessions(self, old_agent: CodeAgent, new_agent: CodeAgent) -> None:
        """Copy conversation history from old agent to new agent.

        Preserves context across mode switches (plan → code → auto).
        Skips the system prompt (each mode has its own) but copies all
        user/assistant/tool messages.
        """
        from nemocode.core.streaming import Role as MsgRole

        old_sessions = old_agent._scheduler._sessions
        if not old_sessions:
            return
        # Get the single session from the old agent (keyed by its role)
        old_session = next(iter(old_sessions.values()), None)
        if not old_session or not old_session.messages:
            return
        # Extract non-system messages
        history = [m for m in old_session.messages if m.role != MsgRole.SYSTEM]
        if not history:
            return
        # Force the new agent's session to be created (with its own system prompt)
        new_role = new_agent._scheduler._single_role
        if new_role not in new_agent._scheduler._sessions:
            from nemocode.core.sessions import Session

            s = Session(
                id=new_agent._scheduler._single_session_id,
                endpoint_name=old_session.endpoint_name,
            )
            prompt = new_agent._scheduler._single_prompt or ""
            project_ctx = new_agent._scheduler._project_context
            if project_ctx:
                prompt = f"{prompt}\n\n## Project Context\n{project_ctx}"
            if prompt:
                s.add_system(prompt)
            new_agent._scheduler._sessions[new_role] = s
        new_session = new_agent._scheduler._sessions[new_role]
        new_session.messages.extend(history)
        new_session.usage = old_session.usage

    def rebuild_agent(self) -> None:
        """Rebuild the agent after config changes (endpoint/formation switch)."""
        self.context_mgr = ContextManager(
            context_window=self._resolve_context_window(),
            model_id=self._resolve_model_id(),
        )
        self.agent = self._build_agent()
        self.turn_count = 0
        self.clear_turn_history()

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

    @property
    def mode(self) -> str:
        return self._modes[self._mode_idx]

    def cycle_mode(self) -> str:
        """Cycle to next mode and return its name."""
        self._mode_idx = (self._mode_idx + 1) % len(self._modes)
        return self.mode

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def clear_cancel(self) -> None:
        self._cancelled = False

    def clear_turn_history(self) -> None:
        self._active_turn = None
        self._last_turn = None
        self._queued_input = None

    def begin_turn(self, user_input: str) -> None:
        try:
            from nemocode.tools.fs import undo_stack_depth

            undo_depth = undo_stack_depth()
        except Exception:
            undo_depth = 0

        self._active_turn = TurnBoundary(
            user_input=user_input,
            turn_count_before=self.turn_count,
            metrics_request_count=self.metrics.request_count,
            undo_depth_before=undo_depth,
            session_checkpoints={
                role: session.checkpoint() for role, session in self.agent.sessions.items()
            },
            pending_plan_text=self.agent.pending_plan_text,
            pending_plan_user_input=getattr(self.agent, "_pending_plan_user_input", None),
        )

    def finish_turn(self) -> None:
        if self._active_turn is not None:
            self._last_turn = self._active_turn
            self._active_turn = None

    def _restore_turn_sessions(self, turn: TurnBoundary) -> None:
        sessions = self.agent.sessions
        for role in list(sessions):
            if role not in turn.session_checkpoints:
                del sessions[role]
        for role, checkpoint in turn.session_checkpoints.items():
            session = sessions.get(role)
            if session is not None:
                session.restore(checkpoint)

    def revert_last_turn(self) -> tuple[bool, str]:
        turn = self._last_turn
        if turn is None:
            return False, "Nothing to revert. Run a turn first."

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
        self._last_turn = None

        reverted_files = len([result for result in results if "error" not in result])
        if reverted_files > 0:
            return (
                True,
                f"Reverted last turn and restored {reverted_files} file change"
                f"{'s' if reverted_files != 1 else ''}.",
            )
        return True, "Reverted last turn. No file changes needed to be restored."

    def prepare_retry(self) -> tuple[bool, str, str | None]:
        turn = self._last_turn
        if turn is None:
            return False, "Nothing to retry. Run a turn first.", None

        preview = _turn_preview(turn.user_input)
        ok, message = self.revert_last_turn()
        if not ok:
            return False, message, None
        return True, f"Retrying: {preview}", turn.user_input


# ===================================================================
# Confirmation functions
# ===================================================================


async def _auto_confirm(tool_name: str, args: dict) -> bool:
    return True


# Module-level flag for auto-approve-remaining (set by 'a' response)
_auto_approve_remaining = False


async def _interactive_confirm(tool_name: str, args: dict) -> bool:
    """Ask user for confirmation before executing a tool.

    Responses: y=yes, n=no, a=auto-approve all remaining this turn.
    """
    global _auto_approve_remaining
    if _auto_approve_remaining:
        return True

    summary = format_confirm_summary(tool_name, args)
    console.print(f"\n[yellow]Allow [bold]{tool_name}[/bold]?[/yellow]")
    console.print(f"[dim]  {summary}[/dim]")
    render_confirm_detail(console, tool_name, args)
    try:
        response = console.input("[yellow]  [y/N/a(ll)]: [/yellow]")
    except (EOFError, KeyboardInterrupt):
        return False
    choice = response.strip().lower()
    if choice in ("a", "all"):
        _auto_approve_remaining = True
        return True
    return choice in ("y", "yes")


# ===================================================================
# Event rendering
# ===================================================================


class _TurnRenderer:
    """Renders AgentEvents for a single conversation turn.

    Delegates display to the shared EventRenderer while tracking
    metrics (token counts, TTFT, tool call/error counts) locally.
    """

    def __init__(self, state: _ReplState) -> None:
        self._state = state
        self._renderer = EventRenderer(console, show_thinking=state.show_thinking)
        self._text_buf: str = ""
        self._think_buf: str = ""
        self._last_usage: dict[str, int] = {}
        self._tool_call_count: int = 0
        self._tool_error_count: int = 0
        self._turn_start: float = time.time()
        self._first_token_time: float | None = None
        self._current_model_id: str = ""

    def start_thinking(self, phrase: str) -> None:
        """Start the thinking spinner via the inner EventRenderer."""
        self._renderer.start_thinking(phrase)

    def render_event(self, event: AgentEvent) -> None:
        """Track metrics and delegate display to the shared renderer."""
        # Track metrics per event type
        if event.kind == "text":
            if self._first_token_time is None:
                self._first_token_time = time.time()
            self._text_buf += event.text
        elif event.kind == "thinking":
            self._think_buf += event.thinking
        elif event.kind == "tool_call":
            self._tool_call_count += 1
        elif event.kind == "tool_result":
            if event.is_error or tool_result_has_embedded_error(event.tool_name, event.tool_result):
                self._tool_error_count += 1
        elif event.kind == "usage":
            self._last_usage = event.usage
            ep_name = self._state.config.default_endpoint
            ep = self._state.config.endpoints.get(ep_name)
            if ep:
                self._current_model_id = ep.model_id

        # Delegate display
        self._renderer.render(event)

    def finalize(self) -> None:
        """Flush pending output, print turn summary, and record metrics."""
        self._renderer.flush()
        # If the model did tool work but produced no text, show a summary
        if not self._text_buf.strip() and self._tool_call_count > 0:
            elapsed = time.time() - self._turn_start
            n = self._tool_call_count
            parts = [f"{n} tool call{'s' if n != 1 else ''}"]
            if self._tool_error_count:
                e = self._tool_error_count
                parts.append(f"{e} error{'s' if e != 1 else ''}")
            parts.append(f"{elapsed:.1f}s")
            console.print(f"\n[dim]Done — {' · '.join(parts)}[/dim]")
        self._record_metrics()
        self._print_turn_summary()

    def _print_turn_summary(self) -> None:
        """Print a compact performance summary line after each turn."""
        elapsed = time.time() - self._turn_start
        completion_tokens = self._last_usage.get("completion_tokens", 0)
        total_tokens = self._last_usage.get("prompt_tokens", 0) + completion_tokens
        if total_tokens == 0:
            return

        parts: list[str] = []
        parts.append(f"{elapsed:.1f}s")
        parts.append(f"{total_tokens:,} tok")

        # Throughput
        if completion_tokens > 0 and elapsed > 0:
            tps = completion_tokens / elapsed
            parts.append(f"{tps:.0f} tok/s")

        # TTFT
        if self._first_token_time is not None:
            ttft = self._first_token_time - self._turn_start
            parts.append(f"TTFT {ttft:.1f}s")

        # Tool calls
        if self._tool_call_count > 0:
            parts.append(f"{self._tool_call_count} tool{'s' if self._tool_call_count != 1 else ''}")

        console.print(f"\n  [dim]▸ {' │ '.join(parts)}[/dim]")

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


def _fmt_tokens(n: int) -> str:
    """Format token count with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _render_status_bar(state: _ReplState) -> None:
    """Print a persistent status bar after each turn.

    Always visible — shows mode, model display name, git branch, context %,
    tokens, tok/s, cost.
    Kept to a single compact line so it feels like instrumentation, not clutter.
    """
    parts: list[str] = []
    theme = _repl_theme(state.config)

    parts.append(f"[bold {theme.accent_rich}]NVIDIA[/bold {theme.accent_rich}]")

    # Mode
    mode = state.mode
    mode_colors = {"code": theme.accent_rich, "plan": "yellow", "auto": "red"}
    mc = mode_colors.get(mode, theme.accent_rich)
    parts.append(f"[bold {mc}]{mode}[/bold {mc}]")

    # Model display name (instead of raw endpoint key)
    ep_name = state.config.default_endpoint
    ep = state.config.endpoints.get(ep_name)
    if ep:
        manifest = state.config.manifests.get(ep.model_id)
        display = manifest.display_name if manifest else _short_model_ref(ep.model_id)
        parts.append(f"[cyan]{display}[/cyan]")
    else:
        parts.append(f"[cyan]{ep_name}[/cyan]")
    parts.append(f"[blue]{state.current_primary_agent_display()}[/blue]")

    if state.config.active_formation:
        parts.append(f"[{theme.accent_rich}]{state.config.active_formation}[/{theme.accent_rich}]")

    # Git branch
    branch = _get_git_branch()
    if branch:
        parts.append(f"[magenta]⎇ {branch}[/magenta]")

    # Context usage
    total_tokens = 0
    try:
        for session in state.agent.sessions.values():
            total_tokens += state.context_mgr.usage(session.messages)
    except Exception:
        pass
    total_tokens = max(total_tokens, state.metrics.total_tokens)
    ctx_window = state.context_mgr.context_window
    pct = (total_tokens / ctx_window * 100) if ctx_window > 0 else 0

    pct_color = "red" if pct > 80 else "yellow" if pct > 50 else "dim"
    parts.append(f"[{pct_color}]ctx:{pct:.0f}%[/{pct_color}]")

    # Tokens used
    if total_tokens > 0:
        parts.append(f"[dim]{_fmt_tokens(total_tokens)}tok[/dim]")

    # Last tok/s
    last_tps = state.metrics.last_tokens_per_sec
    if last_tps > 0:
        parts.append(f"[dim]{last_tps:.0f} tok/s[/dim]")

    # Cost — only when interesting (> $0.01)
    total_cost = state.metrics.total_cost
    if total_cost > 0.01:
        parts.append(f"[dim]${total_cost:.4f}[/dim]")

    bar = " [dim]│[/dim] ".join(parts)
    console.print(f"  {bar}")


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


def _compact_gpu_name(name: str) -> str:
    """Shorten vendor-heavy GPU names for display."""
    cleaned = name.strip()
    for prefix in ("NVIDIA GeForce ", "NVIDIA RTX ", "NVIDIA ", "GeForce "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned


def _get_hardware_line() -> str:
    """Build a compact hardware description for the banner.

    DGX Spark: 'DGX Spark (GB10) │ 128GB unified │ spark-vllm'
    Other GPU: 'RTX 4090 (24GB)' etc.
    Returns empty string if detection fails or no GPUs.
    Uses cached detect_hardware() — no startup penalty.
    """
    try:
        from nemocode.core.hardware import detect_hardware

        hw = detect_hardware()
        if hw.is_dgx_spark:
            mem = f"{hw.unified_memory_gb:.0f}GB unified"
            return f"DGX Spark · {mem}"
        elif hw.gpus:
            gpu = hw.gpus[0]
            vram = f"{gpu.vram_gb:.0f}GB"
            gpu_name = _compact_gpu_name(gpu.name)
            if len(hw.gpus) > 1:
                return f"{len(hw.gpus)}× {gpu_name} · {vram} each"
            return f"{gpu_name} · {vram}"
    except Exception:
        pass
    return ""


def _print_banner(state: _ReplState) -> None:
    """Print a clean NVIDIA-themed welcome banner.

    Detects terminal height — if < 30 rows, uses a compact 2-line banner.
    Full banner for tall terminals. Shows hardware info when available.
    """
    ep_name = state.config.default_endpoint
    ep = state.config.endpoints.get(ep_name)
    model_id = ep.model_id if ep else "unknown"
    formation = state.config.active_formation

    manifest = state.config.manifests.get(model_id) if ep else None
    model_display = manifest.display_name if manifest else _short_model_ref(model_id)
    theme = _repl_theme(state.config)
    if manifest:
        cw = manifest.context_window
        ctx_k = f"{cw // 1_000_000}M" if cw >= 1_000_000 else f"{cw // 1000}K"
    else:
        ctx_k = "128K"

    mode_colors = {"code": theme.accent_rich, "plan": "yellow", "auto": "red"}
    mode_style = mode_colors.get(state.mode, theme.accent_rich)
    formation_str = f" · {formation}" if formation else ""
    hw_line = _get_hardware_line()
    path_line = str(Path.cwd())
    keys = state.config.keybindings.repl

    # Compact banner for small terminals (< 30 rows)
    try:
        term_height = os.get_terminal_size().lines
    except (OSError, ValueError):
        term_height = 40  # default: assume large enough

    if term_height < 40:
        console.print(
            f"[bold {theme.accent_rich}]NeMoCode // NVIDIA NIM[/bold {theme.accent_rich}] "
            f"[dim]v{__version__}[/dim]"
        )
        console.print(
            f"[bold]{model_display}[/bold] [dim]· {ep_name} · {ctx_k} ctx{formation_str} · [/dim]"
            f"[bold {mode_style}]{state.mode.upper()}[/bold {mode_style}]"
        )
        if hw_line:
            console.print(f"[dim]{hw_line}[/dim]")
        console.print(f"[dim]{path_line}[/dim]")
        console.print(
            f'[dim]/help · """ multi-line · {format_key_label(keys.cycle_mode)} mode · '
            f"{format_key_label(keys.exit)} exit[/dim]"
        )
        return

    # Full banner for tall terminals
    from rich.panel import Panel
    from rich.text import Text

    content = Text()
    content.append(model_display, style="bold")
    content.append("  ·  ", style="dim")
    content.append(ep_name, style="dim")
    content.append("  ·  ", style="dim")
    content.append(f"{ctx_k} ctx", style="dim")
    if formation:
        content.append("  ·  ", style="dim")
        content.append(formation, style=f"bold {theme.accent_rich}")
    content.append("  ·  ", style="dim")
    content.append(state.mode.upper(), style=f"bold {mode_style}")
    content.append("\n")
    if hw_line:
        content.append(f"{hw_line}\n", style="dim")
    content.append(f"{path_line}\n", style="dim")
    content.append(
        f'/help · """ multi-line · {format_key_label(keys.cycle_mode)} mode · '
        f"{format_key_label(keys.exit)} exit",
        style="dim",
    )

    panel = Panel(
        content,
        title=f"[bold {theme.accent_rich}]NeMoCode // NVIDIA NIM[/bold {theme.accent_rich}]",
        subtitle=f"[dim]v{__version__}[/dim]",
        border_style=theme.accent_rich,
        padding=(0, 1),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ===================================================================
# Main REPL loop
# ===================================================================


async def run_repl(
    config: NeMoCodeConfig,
    endpoint_name: str | None,
    formation_name: str | None,
    agent_name: str | None,
    show_thinking: bool = True,
    auto_yes: bool = False,
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
        agent_name=agent_name,
    )
    _apply_repl_theme(config)
    reader = _InputReader(state)
    commands = _SlashDispatcher(state)

    # Initialize skills
    from nemocode.skills import create_default_registry

    skills = create_default_registry()

    # Check first-run / missing API key
    from nemocode.core.first_run import check_and_run_first_run

    check_and_run_first_run()

    # Initialize MCP servers
    if config.mcp.servers:
        try:
            await state.agent.init_mcp()
        except Exception as e:
            logger.debug("MCP init failed: %s", e)

    # Register the clarify callback so ask_user works interactively
    from nemocode.tools.clarify import set_ask_fn

    async def _ask_user_interactive(question: str, options: list[str]) -> str:
        # Stop any active spinner before prompting — they fight over the terminal
        renderer_ref = getattr(state, "_active_renderer", None)
        if renderer_ref:
            renderer_ref._renderer._stop_thinking()

        console.print(f"\n[bold yellow]{question}[/bold yellow]")
        if options:
            for opt in options:
                console.print(f"  [dim]{opt}[/dim]")
        try:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(
                None, lambda: console.input("[yellow]  ▸ [/yellow]")
            )
            return answer.strip()
        except (EOFError, KeyboardInterrupt):
            return "(no answer)"

    set_ask_fn(_ask_user_interactive)

    # Start file watcher for external change detection
    from nemocode.core.watcher import FileWatcher

    watcher = FileWatcher(Path.cwd())
    await watcher.start()

    _print_banner(state)

    # Track Ctrl+C at prompt: two in quick succession means exit
    last_interrupt_time: float = 0.0

    try:
        while True:
            # ---- Read input ----
            try:
                raw_input = await reader.read(mode=state.mode)
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

                # Check for pending skill execution
                if state._pending_skill:
                    skill_name, skill_arg = state._pending_skill
                    state._pending_skill = None
                    skill = skills.get(skill_name)
                    if skill:
                        try:
                            result = await skill.run(
                                skill_arg,
                                {"config": config, "agent": state.agent, "console": console},
                            )
                            if result:
                                console.print(f"[dim]{result}[/dim]")
                        except Exception as e:
                            console.print(f"[red]Skill error: {e}[/red]")
                if state._queued_input is None:
                    continue
                stripped = state._queued_input
                state._queued_input = None

            # ---- Execute turn ----
            global _auto_approve_remaining
            _auto_approve_remaining = False  # Reset per-turn auto-approve
            state.begin_turn(stripped)
            state.turn_count += 1
            state.clear_cancel()

            # Check for external file changes and prepend to input
            changes = watcher.get_changes()
            turn_input = stripped
            if changes:
                watcher.clear()
                changed_files = [c.path for c in changes[:10]]
                kinds = {c.path: c.kind.value for c in changes}
                change_note = ", ".join(f"{Path(f).name} ({kinds[f]})" for f in changed_files)
                if len(changes) > 10:
                    change_note += f", ... and {len(changes) - 10} more"
                turn_input = (
                    f"[Note: these files changed externally since last turn: "
                    f"{change_note}]\n\n{stripped}"
                )
                console.print(f"[dim]  {len(changes)} file(s) changed externally[/dim]")

            renderer = _TurnRenderer(state)
            state._active_renderer = renderer

            state._turn_active = True
            state._turn_start_time = time.time()
            try:
                await _run_turn(state, turn_input, renderer)
            except KeyboardInterrupt:
                # Ctrl+C during streaming — the turn is cancelled but the session continues
                console.print("\n[dim]Turn cancelled.[/dim]")
            except Exception as exc:
                logger.exception("Unexpected error during turn")
                console.print(f"\n[bold red]Unexpected error: {exc}[/bold red]")
            finally:
                state._turn_active = False
                state.finish_turn()
                renderer.finalize()
                _render_status_bar(state)

            # Check if plan approval is pending after the turn
            if state.agent.has_pending_plan:
                console.print(
                    "\n[yellow]Plan awaiting your decision:[/yellow]"
                    "\n  [dim]1. Start implementing[/dim]"
                    "\n  [dim]2. Edit the plan[/dim]"
                    "\n  [dim]3. Ask a question[/dim]"
                    "\n  [dim]4. Cancel[/dim]"
                )

            # Auto-save session after each turn
            _auto_save_session(state)
    finally:
        # Cleanup watcher
        try:
            await watcher.stop()
        except Exception:
            pass
        # Cleanup MCP connections on exit
        if state.agent._mcp_clients:
            try:
                await state.agent.cleanup_mcp()
            except Exception:
                pass


async def _run_turn(state: _ReplState, user_input: str, renderer: _TurnRenderer) -> None:
    """Execute a single conversation turn, streaming events to the renderer.

    Handles Ctrl+C gracefully: sets the cancelled flag and drains remaining events
    rather than raising, so the session stays in a consistent state.
    Also handles resume of pending plan approvals.
    """
    # Check if we're resuming a pending plan approval
    if state.agent.has_pending_plan:
        from nemocode.workflows.code_agent import CodeAgent

        decision, feedback = CodeAgent.parse_plan_decision(user_input)

        if decision == "ask":
            # Frame the question in the context of the existing plan
            question = feedback.strip() or user_input
            plan_text = state.agent.pending_plan_text or ""
            user_input = (
                f"The user is reviewing this plan and has a question:\n\n"
                f"## Current Plan\n{plan_text}\n\n"
                f"## Question\n{question}\n\n"
                "Answer the question concisely. Do not generate a new plan."
            )
            # Keep plan pending — menu will show again after this turn

        elif decision != "pending":
            # Recognized decision (approve, cancel, revise)
            result = await state.agent.try_handle_plan_response(user_input)
            if result is not None:
                renderer.start_thinking("Processing plan decision")
                try:
                    async for event in result:
                        if state.cancelled:
                            continue
                        try:
                            renderer.render_event(event)
                        except KeyboardInterrupt:
                            state.cancel()
                            console.print("\n[dim]Cancelling...[/dim]")
                            continue
                finally:
                    renderer._renderer.flush()
                return

        else:
            # Not a recognized decision — clear pending, proceed as normal input
            state.agent._pending_plan_text = None
            state.agent._pending_plan_user_input = None

    import random

    _THINKING_PHRASES = [
        "Tensor cores warming up",
        "Inference in progress",
        "Activating neural pathways",
        "Running through the MoE layers",
        "Reasoning at GPU speed",
        "Processing with Nemotron",
        "Parallel threads converging",
        "Computing",
    ]
    renderer.start_thinking(random.choice(_THINKING_PHRASES))

    try:
        async for event in state.agent.run(user_input):
            if state.cancelled:
                continue

            try:
                renderer.render_event(event)
            except KeyboardInterrupt:
                state.cancel()
                console.print("\n[dim]Cancelling...[/dim]")
                continue
    finally:
        renderer._renderer.flush()


# ===================================================================
# Entry point called from code.py
# ===================================================================


def start_repl(
    endpoint: str | None,
    formation: str | None,
    agent_name: str | None,
    think: bool = True,
    yes: bool = False,
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
                agent_name=agent_name,
                show_thinking=think,
                auto_yes=yes,
            )
        )
    except KeyboardInterrupt:
        # Final safety net — clean exit on Ctrl+C
        console.print("\n[dim]Goodbye.[/dim]")
