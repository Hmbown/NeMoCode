# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Shared event rendering for terminal output.

Design principles:
  - Tool calls are "background work" — compact, dim, de-emphasized
  - The AI's response is the main event — clear visual start
  - Read-only tools are collapsed into a single summary line
  - Only mutations (write, edit, bash errors) get expanded output
  - Diffs are always shown (they're the point)
  - A unified "thinking spinner" persists from turn start through tool
    gathering and only stops when the text response begins
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.status import Status
from rich.text import Text

from nemocode.core.scheduler import AgentEvent

# NVIDIA green accent used throughout the UI
_NV_GREEN = "bright_green"

# Custom NVIDIA beam spinner — a green bar that bounces left-to-right.
# Each frame is one position of the bright segment in a dim track.
_BEAM_WIDTH = 12
_BEAM_FRAMES = []
_TRACK = "─"
_BRIGHT = "━"
for i in range(_BEAM_WIDTH):
    bar = list(_TRACK * _BEAM_WIDTH)
    # 3-char bright segment, wrapping
    for offset in range(3):
        bar[(i + offset) % _BEAM_WIDTH] = _BRIGHT
    _BEAM_FRAMES.append("".join(bar))
# Bounce back
_BEAM_FRAMES += _BEAM_FRAMES[-2:0:-1]

# Tools whose results are just "gathered info" — don't need expanded output
_READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "search_files",
        "glob_files",
        "git_status",
        "git_diff",
        "git_log",
        "list_memories_tool",
        "list_tasks",
        "web_search",
        "lsp_diagnostics",
        "lsp_hover",
        "lsp_references",
    }
)

# Max lines of output to show for tool results — keep it compact.
# Success output is brief, errors get more room.
_MAX_RESULT_LINES = 5
_ERROR_RESULT_LINES = 15

# During multi-tool sequences, cap further
_MULTI_TOOL_RESULT_LINES = 2


@dataclass
class _ToolCallBuffer:
    """Buffer for collapsing multiple read-only tool calls into a summary."""

    calls: list[tuple[str, dict]] = field(default_factory=list)
    results: list[tuple[str, str, bool]] = field(default_factory=list)  # (name, result, is_error)

    def add_call(self, name: str, args: dict) -> None:
        self.calls.append((name, args))

    def add_result(self, name: str, result: str, is_error: bool) -> None:
        self.results.append((name, result, is_error))

    def is_empty(self) -> bool:
        return not self.calls

    def clear(self) -> None:
        self.calls.clear()
        self.results.clear()

    def get_summary(self) -> str:
        """Generate a summary of buffered tool calls.

        Shows file names when few files, counts when many.
        """
        if not self.calls:
            return ""

        reads: list[str] = []
        searches: list[str] = []
        git_ops: list[str] = []
        others: list[str] = []

        for name, args in self.calls:
            if name == "read_file":
                reads.append(_short_path(args.get("path", "?")))
            elif name in ("search_files", "glob_files"):
                pattern = args.get("pattern", "?")
                searches.append(f"/{pattern}/")
            elif name in ("git_status", "git_diff", "git_log"):
                git_ops.append(name.replace("_", " "))
            elif name == "web_search":
                q = args.get("query", "?")
                if len(q) > 30:
                    q = q[:27] + "..."
                others.append(f"web: {q}")
            elif name == "list_dir":
                others.append(f"list {args.get('path', '.')}")
            elif name in ("lsp_diagnostics", "lsp_hover", "lsp_references"):
                others.append(f"lsp: {_short_path(args.get('path', '?'))}")
            else:
                others.append(format_tool_call(name, args))

        parts: list[str] = []
        if reads:
            if len(reads) <= 3:
                parts.append(f"Read {', '.join(reads)}")
            else:
                parts.append(f"Read {len(reads)} files")
        if searches:
            if len(searches) == 1:
                parts.append(f"searched {searches[0]}")
            else:
                parts.append(f"searched {len(searches)} patterns")
        if git_ops:
            parts.extend(git_ops)
        parts.extend(others)

        return ", ".join(parts) if parts else f"{len(self.calls)} tool calls"


class EventRenderer:
    """Renders AgentEvents to a Rich console.

    Optimized for readability: tool calls are compact dim lines,
    read-only tools are collapsed into summaries,
    the AI's text response gets a clear visual separator.

    A unified "thinking spinner" persists from turn start through
    read-only tool execution, updating with progress. It stops only
    when the text response begins or a mutation tool needs display.
    """

    # Max lines of thinking to show at once (rolling window)
    _THINK_VISIBLE_LINES = 8

    def __init__(self, console: Console, *, show_thinking: bool = False) -> None:
        self._console = console
        self.show_thinking = show_thinking
        self._streaming = False
        self._md_buffer = ""
        self._live: Live | None = None
        self._last_refresh = 0.0
        self._thinking_spinner: Status | None = None  # unified thinking/progress
        self._tool_start: float = 0.0
        self._turn_start: float = 0.0  # set by start_thinking()
        self._in_tools = False  # True while executing tools (before response)
        self._showed_response_sep = False  # Only show separator once
        self._current_tool: str = ""  # Name of tool being executed
        self._tool_buffer = _ToolCallBuffer()  # Buffer for collapsing read-only tools
        # Thinking trace: buffered + Live rendered
        self._think_buffer = ""
        self._think_live: Live | None = None
        self._think_start: float = 0.0
        self._last_think_refresh: float = 0.0
        # Auto-detect interactive terminal for rich rendering
        try:
            self._interactive = console.file.isatty()
        except (AttributeError, ValueError):
            self._interactive = False

    def render(self, event: AgentEvent) -> None:
        """Render a single event to the console."""
        handler = {
            "text": self._text,
            "thinking": self._thinking,
            "phase": self._phase,
            "tool_call": self._tool_call,
            "tool_result": self._tool_result,
            "error": self._error,
        }.get(event.kind)
        if handler:
            handler(event)

    def flush(self) -> None:
        """Finalize all pending output."""
        self._end_md_stream()
        self._end_think_stream()
        self._stop_thinking()
        self._flush_tool_buffer_if_needed()

    def start_thinking(self, phrase: str) -> None:
        """Start the unified thinking/progress spinner.

        Uses a custom NVIDIA-themed bouncing beam animation that stays
        visible throughout tool execution.  Stops only when the text
        response begins or a mutation tool needs immediate display.
        """
        if not self._interactive:
            return
        self._turn_start = time.monotonic()
        self._thinking_spinner = _beam_status(
            f"  [{_NV_GREEN}]{phrase}…[/{_NV_GREEN}]",
            self._console,
        )
        self._thinking_spinner.start()

    # -- Event handlers --

    def _text(self, event: AgentEvent) -> None:
        # Collapse thinking trace + stop spinner on first text event
        self._end_think_stream()
        self._stop_thinking()
        # Show response separator — always on first text chunk of a turn
        if not self._showed_response_sep:
            self._show_response_separator()

        if self._interactive:
            self._md_buffer += event.text
            if self._live is None:
                self._live = Live(
                    Markdown(self._md_buffer),
                    console=self._console,
                    auto_refresh=False,
                    vertical_overflow="visible",
                )
                self._live.start()
            now = time.monotonic()
            if now - self._last_refresh >= 0.05 or "\n" in event.text:
                self._live.update(Markdown(self._md_buffer))
                self._live.refresh()
                self._last_refresh = now
        else:
            self._console.print(event.text, end="", highlight=False, markup=False)
        self._streaming = True

    def _thinking(self, event: AgentEvent) -> None:
        if not self.show_thinking:
            return

        # First thinking chunk: stop the spinner, thinking text takes over
        if not self._think_buffer:
            self._think_start = time.monotonic()
            self._stop_thinking()

        self._think_buffer += event.thinking

        if not self._interactive:
            self._console.print(
                event.thinking,
                end="",
                style="dim italic",
                highlight=False,
                markup=False,
            )
            return

        # Use Live for smooth, in-place rendering with a rolling window
        if self._think_live is None:
            self._end_md_stream()
            self._think_live = Live(
                Text(""),
                console=self._console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            self._think_live.start()

        now = time.monotonic()
        if now - self._last_think_refresh >= 0.05:
            self._render_think_live(now)
            self._last_think_refresh = now

    def _render_think_live(self, now: float) -> None:
        """Update the thinking Live widget with a rolling window of lines."""
        if not self._think_live:
            return
        lines = self._think_buffer.rstrip().splitlines()
        elapsed = now - self._think_start if self._think_start else 0

        # Header with elapsed time
        header = Text(f"  Thinking ({elapsed:.1f}s)", style=f"dim {_NV_GREEN}")

        # Rolling window: show last N lines to keep output compact
        max_lines = self._THINK_VISIBLE_LINES
        if len(lines) > max_lines:
            visible = lines[-max_lines:]
            header = Text(
                f"  Thinking ({elapsed:.1f}s, {len(lines)} lines)",
                style=f"dim {_NV_GREEN}",
            )
        else:
            visible = lines

        body = Text(
            "\n".join(f"  {line}" for line in visible),
            style="dim italic",
        )
        self._think_live.update(Group(header, body))
        self._think_live.refresh()

    def _end_think_stream(self) -> None:
        """Collapse thinking into a compact summary line."""
        if self._think_live:
            # Final summary replaces the rolling window
            elapsed = time.monotonic() - self._think_start if self._think_start else 0
            lines = self._think_buffer.rstrip().splitlines()
            summary = Text(
                f"  Thought for {elapsed:.1f}s ({len(lines)} lines)",
                style="dim italic",
            )
            self._think_live.update(summary)
            self._think_live.refresh()
            self._think_live.stop()
            self._think_live = None
        self._think_buffer = ""

    def _phase(self, event: AgentEvent) -> None:
        self._end_md_stream()
        role = event.role.value if event.role else "agent"
        self._console.print(f"\n[bold {_NV_GREEN}]━━ {role} ▸ {event.text} ━━[/bold {_NV_GREEN}]")

    def _tool_call(self, event: AgentEvent) -> None:
        """Handle tool call events — buffer read-only tools, show mutations immediately."""
        self._end_md_stream()
        self._end_think_stream()
        self._in_tools = True
        self._tool_start = time.monotonic()
        self._current_tool = event.tool_name

        # Buffer read-only tools for later summary
        if event.tool_name in _READ_ONLY_TOOLS:
            self._tool_buffer.add_call(event.tool_name, event.tool_args)
            # Update thinking spinner with current tool progress
            self._update_thinking_status(event.tool_name, event.tool_args)
        else:
            # Mutation: stop thinking, flush read-only summary, show tool line
            self._stop_thinking()
            self._flush_tool_buffer_if_needed()
            line = format_tool_call(event.tool_name, event.tool_args)
            self._console.print(f"  [{_NV_GREEN}]▸[/{_NV_GREEN}] [dim]{line}[/dim]", end="")

    def _tool_result(self, event: AgentEvent) -> None:
        """Handle tool result events — buffer read-only results, show mutation results."""
        tool_name = event.tool_name or self._current_tool

        if tool_name in _READ_ONLY_TOOLS:
            # Buffer the result — thinking spinner keeps running
            self._tool_buffer.add_result(tool_name, event.tool_result, event.is_error)
        else:
            # Show mutation result immediately
            elapsed = time.monotonic() - self._tool_start if self._tool_start else 0
            _render_tool_result_inline(
                self._console,
                tool_name,
                event.tool_result,
                event.is_error,
                elapsed,
                in_multi_tool=False,
            )
            # Restart spinner for the next API round so the user isn't
            # staring at a frozen terminal during the next LLM call
            self._restart_spinner()

    def _flush_tool_buffer_if_needed(self) -> None:
        """Flush buffered read-only tool calls as a summary line."""
        if self._tool_buffer.is_empty():
            return

        summary = self._tool_buffer.get_summary()
        # Count results to see if any errors
        error_count = sum(1 for _, _, is_error in self._tool_buffer.results if is_error)

        # Elapsed time since turn start
        elapsed_str = ""
        if self._turn_start:
            elapsed = time.monotonic() - self._turn_start
            if elapsed > 1.0:
                elapsed_str = f" [dim]({elapsed:.1f}s)[/dim]"

        if error_count > 0:
            err_s = "s" if error_count > 1 else ""
            self._console.print(
                f"  [{_NV_GREEN}]▸[/{_NV_GREEN}] [dim]{summary}[/dim]"
                f"{elapsed_str} [red]({error_count} error{err_s})[/red]"
            )
        else:
            self._console.print(f"  [{_NV_GREEN}]▸[/{_NV_GREEN}] [dim]{summary}[/dim]{elapsed_str}")

        self._tool_buffer.clear()
        # Restart spinner for the next API round
        self._restart_spinner()

    def _restart_spinner(self) -> None:
        """Re-show the spinner between tool rounds so the user sees activity."""
        if not self._interactive:
            return
        if self._thinking_spinner:
            return  # already running
        elapsed = time.monotonic() - self._turn_start if self._turn_start else 0
        self._thinking_spinner = _beam_status(
            f"  [{_NV_GREEN}]Working ({elapsed:.0f}s)…[/{_NV_GREEN}]",
            self._console,
        )
        self._thinking_spinner.start()

    def _error(self, event: AgentEvent) -> None:
        self._end_md_stream()
        self._stop_thinking()
        self._console.print(f"\n[bold red]Error: {event.text}[/bold red]")
        hint = _error_hint(event.text)
        if hint:
            self._console.print(f"  [dim]{hint}[/dim]")

    # -- Internal helpers --

    def _show_response_separator(self) -> None:
        """Print a visual separator before the AI's text response."""
        self._showed_response_sep = True
        self._in_tools = False
        # Flush any buffered tool calls before showing response
        self._flush_tool_buffer_if_needed()
        self._console.print()
        self._console.print(f"[{_NV_GREEN}]nemo ▸[/{_NV_GREEN}]")

    def _end_md_stream(self) -> None:
        """Stop markdown streaming and finalize output."""
        if self._live:
            self._live.update(Markdown(self._md_buffer))
            self._live.stop()
            self._live = None
            self._md_buffer = ""
        elif self._streaming and not self._interactive:
            self._console.print()  # trailing newline for raw mode
        self._md_buffer = ""
        self._streaming = False

    def _stop_thinking(self) -> None:
        """Stop the unified thinking/progress spinner."""
        if self._thinking_spinner:
            self._thinking_spinner.stop()
            self._thinking_spinner = None

    def _update_thinking_status(self, tool_name: str, tool_args: dict) -> None:
        """Update the thinking spinner text to show current tool progress."""
        if not self._interactive or not self._thinking_spinner:
            return

        # Elapsed since turn start
        elapsed = time.monotonic() - self._turn_start if self._turn_start else 0
        elapsed_str = f" ({elapsed:.1f}s)" if elapsed > 1.0 else ""

        if tool_name == "read_file":
            path = _short_path(tool_args.get("path", "?"))
            status = f"Reading {path}"
        elif tool_name == "search_files":
            pattern = tool_args.get("pattern", "?")
            status = f"Searching /{pattern}/"
        elif tool_name == "list_dir":
            status = f"Listing {tool_args.get('path', '.')}"
        elif tool_name in ("git_status", "git_diff", "git_log"):
            status = tool_name.replace("_", " ").title()
        elif tool_name == "web_search":
            q = tool_args.get("query", "?")
            if len(q) > 40:
                q = q[:37] + "..."
            status = f"Searching: {q}"
        else:
            status = format_tool_call(tool_name, tool_args)

        self._thinking_spinner.update(f"  [{_NV_GREEN}]{status}{elapsed_str}…[/{_NV_GREEN}]")


# ---------------------------------------------------------------------------
# Error recovery hints
# ---------------------------------------------------------------------------


def _error_hint(text: str) -> str:
    """Suggest recovery actions based on error content."""
    lower = text.lower()
    if "rate limit" in lower or "429" in lower:
        return "Rate limited. Wait a few seconds, or switch endpoints with /endpoint."
    if "auth" in lower or "401" in lower or "api key" in lower:
        return "Check your API key: run 'nemo auth setup' or set NVIDIA_API_KEY env var."
    if "forbidden" in lower or "403" in lower:
        return "Access denied. Check API permissions or billing."
    if "not found" in lower or "404" in lower:
        return "Resource not found. Check model name with /model."
    if "timeout" in lower or "timed out" in lower:
        return "Request timed out. Try again or /compact to reduce context."
    if "context" in lower and ("length" in lower or "window" in lower or "exceed" in lower):
        return "Context window full. Use /compact or /reset."
    if "connection" in lower or "network" in lower:
        return "Network error. Check your connection and endpoint URL."
    if "ssl" in lower or "certificate" in lower:
        return "SSL error. Check endpoint URL or proxy settings."
    if "dns" in lower or "resolve" in lower:
        return "DNS error. Check endpoint URL or network settings."
    if "refused" in lower or "econnrefused" in lower:
        return "Connection refused. Endpoint may be down or URL incorrect."
    if "invalid" in lower and "request" in lower:
        return "Invalid request. Check tool arguments or model parameters."
    if "internal server error" in lower or "500" in lower:
        return "Server error. Try again in a few moments."
    if "bad gateway" in lower or "502" in lower:
        return "Bad gateway. Endpoint may be overloaded. Try again."
    if "service unavailable" in lower or "503" in lower:
        return "Service unavailable. Try again or switch endpoints."
    if "gateway timeout" in lower or "504" in lower:
        return "Gateway timeout. Try again or /compact."
    return ""


# ---------------------------------------------------------------------------
# Tool call formatting
# ---------------------------------------------------------------------------


def format_tool_call(name: str, args: dict) -> str:
    """Format a tool call as a compact one-line string."""
    if name == "read_file":
        path = _short_path(args.get("path", "?"))
        extras = []
        if args.get("offset"):
            extras.append(f"from line {args['offset']}")
        if args.get("limit"):
            extras.append(f"limit {args['limit']}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"Read {path}{suffix}"

    if name == "write_file":
        path = _short_path(args.get("path", "?"))
        size = len(args.get("content", ""))
        return f"Write {path} ({size:,} chars)"

    if name == "edit_file":
        return f"Edit {_short_path(args.get('path', '?'))}"

    if name == "list_dir":
        path = args.get("path", ".")
        depth = args.get("max_depth", 1)
        return f"List {path}" + (f" (depth {depth})" if depth > 1 else "")

    if name == "bash_exec":
        cmd = args.get("command", "?")
        # Show first line only, truncated
        first_line = cmd.split("\n")[0]
        if len(first_line) > 72:
            first_line = first_line[:69] + "..."
        multi = " ..." if "\n" in cmd else ""
        return f"$ {first_line}{multi}"

    if name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        glob_filter = args.get("glob", "")
        suffix = f" {glob_filter}" if glob_filter else ""
        return f"Search /{pattern}/ in {path}{suffix}"

    if name in ("git_status", "git_diff", "git_log"):
        return name.replace("_", " ")

    if name == "git_commit":
        msg = args.get("message", "?")
        if len(msg) > 50:
            msg = msg[:47] + "..."
        return f'git commit "{msg}"'

    if name == "http_fetch":
        method = args.get("method", "GET")
        url = args.get("url", "?")
        if len(url) > 60:
            url = url[:57] + "..."
        return f"{method} {url}"

    if name == "run_tests":
        path = args.get("test_path", ".")
        return f"pytest {path}"

    if name == "web_search":
        q = args.get("query", "?")
        if len(q) > 50:
            q = q[:47] + "..."
        return f"Search web: {q}"

    if name == "parse_document":
        return f"Parse {_short_path(args.get('path', '?'))}"

    if name == "delegate":
        agent_type = args.get("agent_type", "?")
        task = args.get("task", "")
        if len(task) > 40:
            task = task[:37] + "..."
        return f"delegate ({agent_type}): {task}"

    if name in ("save_memory_tool", "forget_memory_tool", "list_memories_tool"):
        key = args.get("key", "")
        return f"{name.replace('_tool', '')} {key}"

    if name in ("create_task", "update_task", "list_tasks"):
        title = args.get("title", args.get("status", ""))
        return f"{name} {title}"

    if name == "glob_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        return f"Glob {pattern} in {path}"

    if name == "ask_user":
        question = args.get("question", "?")
        if len(question) > 50:
            question = question[:47] + "..."
        return f"Ask: {question}"

    if name == "multi_edit":
        path = _short_path(args.get("path", "?"))
        return f"Multi-edit {path}"

    if name == "apply_patch":
        return f"Patch {_short_path(args.get('path', '?'))}"

    if name == "lsp_diagnostics":
        return f"Diagnostics {_short_path(args.get('path', '?'))}"

    if name == "lsp_hover":
        path = _short_path(args.get("path", "?"))
        line = args.get("line", "?")
        return f"Hover {path}:{line}"

    if name == "lsp_references":
        path = _short_path(args.get("path", "?"))
        line = args.get("line", "?")
        return f"References {path}:{line}"

    # Generic fallback
    args_str = json.dumps(args)
    if len(args_str) > 60:
        args_str = args_str[:57] + "..."
    return f"{name} {args_str}"


def _short_path(path: str) -> str:
    """Shorten a file path for display — keep last 2 components."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) <= 2:
        return path
    return ".../" + "/".join(parts[-2:])


# ---------------------------------------------------------------------------
# Tool result formatting (inline — appended to tool call line)
# ---------------------------------------------------------------------------


def _render_tool_result_inline(
    con: Console,
    tool_name: str,
    result: str,
    is_error: bool,
    elapsed: float = 0.0,
    in_multi_tool: bool = False,
) -> None:
    """Render tool result inline after the tool call, on the same line.

    For read-only tools: just a brief status suffix.
    For mutations: status line + optional diff/output below.

    Args:
        in_multi_tool: If True, we're in a multi-tool sequence — reduce output verbosity.
    """
    time_suffix = f" {elapsed:.1f}s" if elapsed > 0.5 else ""

    # Parse structured results
    parsed = None
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Error handling (always show) ---
    if is_error:
        err_msg = ""
        if parsed and isinstance(parsed, dict) and "error" in parsed:
            err_msg = str(parsed["error"])[:80]
        else:
            err_msg = result.strip().splitlines()[0][:80] if result.strip() else "failed"
        con.print(f" [red]✗ {err_msg}[/red]")
        return

    # --- Read-only tools: compact inline suffix (legacy/direct calls) ---
    if tool_name in _READ_ONLY_TOOLS:
        suffix = _read_only_suffix(tool_name, result, parsed)
        con.print(f" [dim]{suffix}{time_suffix}[/dim]")
        return

    # --- Structured results ---
    if parsed and isinstance(parsed, dict):
        if parsed.get("status") == "ok":
            extra = ""
            if "bytes" in parsed:
                extra = f" ({parsed['bytes']:,} bytes)"
            con.print(f" [{_NV_GREEN}]✓{extra}{time_suffix}[/{_NV_GREEN}]")
            # Show diff for mutations
            if "diff" in parsed and parsed["diff"]:
                _render_diff(con, parsed["diff"])
            return

        if "exit_code" in parsed:
            _render_bash_inline(con, parsed, time_suffix, in_multi_tool=in_multi_tool)
            return

        # HTTP response
        if "status" in parsed and isinstance(parsed["status"], int):
            code = parsed["status"]
            style = _NV_GREEN if code < 400 else "red"
            con.print(f" [{style}]{code}{time_suffix}[/{style}]")
            return

    # --- Raw text fallback ---
    lines = result.strip().splitlines()
    n = len(lines)
    max_lines = _MULTI_TOOL_RESULT_LINES if in_multi_tool else _MAX_RESULT_LINES

    if n == 0:
        con.print(f" [dim]✓{time_suffix}[/dim]")
    elif n <= max_lines:
        con.print(f" [dim]({n} lines){time_suffix}[/dim]")
        for line in lines:
            con.print(f"    {line}", style="dim", highlight=False, markup=False)
    else:
        con.print(f" [dim]({n} lines){time_suffix}[/dim]")
        for line in lines[:max_lines]:
            con.print(f"    {line}", style="dim", highlight=False, markup=False)
        if max_lines == _MULTI_TOOL_RESULT_LINES:
            con.print(f"    [dim]... ({n} total, full output after response)[/dim]")
        else:
            con.print(f"    [dim]... ({n} lines total)[/dim]")


def _read_only_suffix(tool_name: str, result: str, parsed) -> str:
    """Generate a brief inline suffix for read-only tool results."""
    lines = result.strip().splitlines()
    n = len(lines)

    if tool_name == "read_file":
        return f"({n} lines)"

    if tool_name == "list_dir":
        if result.strip() == "(empty directory)":
            return "(empty)"
        return f"({n} entries)"

    if tool_name == "search_files":
        if result.strip() == "(no matches)":
            return "(no matches)"
        return f"({n} results)"

    if tool_name in ("git_status", "git_diff", "git_log"):
        if n == 0:
            return "(clean)"
        return f"({n} lines)"

    if tool_name == "web_search" and parsed and isinstance(parsed, dict):
        count = parsed.get("count", len(parsed.get("results", [])))
        return f"({count} results)"

    if tool_name in ("list_memories_tool", "list_tasks"):
        if parsed and isinstance(parsed, dict):
            count = parsed.get("count", 0)
            return f"({count})"

    return f"({n} lines)"


def _render_bash_inline(
    con: Console,
    parsed: dict,
    time_suffix: str = "",
    in_multi_tool: bool = False,
) -> None:
    """Render bash result inline — compact for success, expanded for errors.

    Args:
        in_multi_tool: If True, reduce output to 1-2 lines during multi-tool sequences.
    """
    code = parsed.get("exit_code", -1)
    stdout = parsed.get("stdout", "").strip()
    stderr = parsed.get("stderr", "").strip()

    max_lines = _MULTI_TOOL_RESULT_LINES if in_multi_tool else _MAX_RESULT_LINES

    if code == 0:
        if stdout:
            lines = stdout.splitlines()
            n = len(lines)
            if n <= max_lines:
                con.print(f" [dim]✓{time_suffix}[/dim]")
                for line in lines:
                    con.print(
                        f"    {line}",
                        style="dim",
                        highlight=False,
                        markup=False,
                    )
            else:
                con.print(f" [dim]✓ ({n} lines){time_suffix}[/dim]")
                for line in lines[:max_lines]:
                    con.print(
                        f"    {line}",
                        style="dim",
                        highlight=False,
                        markup=False,
                    )
                con.print(f"    [dim]... ({n} total)[/dim]")
        else:
            con.print(f" [{_NV_GREEN}]✓{time_suffix}[/{_NV_GREEN}]")
    else:
        con.print(f" [red]✗ exit {code}{time_suffix}[/red]")
        output = stderr or stdout
        if output:
            lines = output.splitlines()
            for line in lines[:_ERROR_RESULT_LINES]:
                con.print(
                    f"    {line}",
                    style="red",
                    highlight=False,
                    markup=False,
                )
            if len(lines) > _ERROR_RESULT_LINES:
                con.print(f"    [dim]... ({len(lines)} total)[/dim]")


def _render_diff(con: Console, diff_text: str) -> None:
    """Render a unified diff with colored +/- lines."""
    lines = diff_text.splitlines()
    for line in lines[:20]:
        if line.startswith("+++") or line.startswith("---"):
            con.print(
                f"    {line}",
                style="bold dim",
                highlight=False,
                markup=False,
            )
        elif line.startswith("+"):
            con.print(f"    {line}", style=_NV_GREEN, highlight=False, markup=False)
        elif line.startswith("-"):
            con.print(f"    {line}", style="red", highlight=False, markup=False)
        elif line.startswith("@@"):
            con.print(f"    {line}", style="cyan", highlight=False, markup=False)
        else:
            con.print(f"    {line}", style="dim", highlight=False, markup=False)
    if len(lines) > 20:
        con.print(f"    [dim]... ({len(lines)} lines total)[/dim]")


# ---------------------------------------------------------------------------
# Confirmation dialog helpers
# ---------------------------------------------------------------------------


def format_confirm_summary(tool_name: str, args: dict) -> str:
    """Format a clean confirmation summary for a tool call."""
    if tool_name == "write_file":
        path = args.get("path", "?")
        size = len(args.get("content", ""))
        return f"{path} ({size:,} chars)"
    if tool_name == "edit_file":
        return args.get("path", "?")
    if tool_name == "bash_exec":
        cmd = args.get("command", "?")
        if len(cmd) > 120:
            cmd = cmd[:117] + "..."
        return cmd
    if tool_name == "git_commit":
        msg = args.get("message", "?")
        files = args.get("files", ".")
        return f'"{msg}" (files: {files})'
    if tool_name == "run_tests":
        return args.get("test_path", ".")
    # Generic
    preview = json.dumps(args)
    if len(preview) > 200:
        preview = preview[:197] + "..."
    return preview


def render_confirm_detail(con: Console, tool_name: str, args: dict) -> None:
    """Render additional context for a confirmation dialog."""
    if tool_name == "write_file":
        _render_write_preview(con, args)
    elif tool_name == "edit_file":
        _render_edit_preview(con, args)
    elif tool_name == "bash_exec":
        cmd = args.get("command", "")
        if "\n" in cmd or len(cmd) > 120:
            con.print("[dim]  Command:[/dim]")
            for line in cmd.splitlines()[:10]:
                con.print(
                    f"    {line}",
                    style="dim",
                    highlight=False,
                    markup=False,
                )
            if cmd.count("\n") > 9:
                con.print(f"    [dim]... ({cmd.count(chr(10)) + 1} lines)[/dim]")
    elif tool_name == "git_commit":
        files = args.get("files", "")
        if files and files != ".":
            con.print(f"[dim]  Files: {files}[/dim]")


def _render_edit_preview(con: Console, args: dict) -> None:
    """Show context around the edit_file old_string."""
    from pathlib import Path

    path = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not old_string:
        return

    p = Path(path).resolve()
    if not p.exists():
        return

    try:
        content = p.read_text()
        idx = content.find(old_string)
        if idx == -1:
            con.print("[dim]  (old_string not found in file)[/dim]")
            return

        import difflib

        old_lines = old_string.splitlines(keepends=True)
        new_lines = new_string.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
        if diff:
            con.print("[dim]  Change:[/dim]")
            for line in diff[:15]:
                if line.startswith("+") and not line.startswith("+++"):
                    con.print(f"    [{_NV_GREEN}]{line}[/{_NV_GREEN}]")
                elif line.startswith("-") and not line.startswith("---"):
                    con.print(f"    [red]{line}[/red]")
                elif line.startswith("@@"):
                    con.print(f"    [cyan]{line}[/cyan]")
            if len(diff) > 15:
                con.print(f"    [dim]... ({len(diff)} total)[/dim]")
    except Exception:
        pass


def _render_write_preview(con: Console, args: dict) -> None:
    """Show a diff preview for write_file confirmation."""
    import difflib
    from pathlib import Path

    path = args.get("path", "")
    content = args.get("content", "")
    p = Path(path).resolve()

    if not p.exists():
        con.print("[dim]  (new file)[/dim]")
        lines = content.splitlines()
        for line in lines[:8]:
            con.print(f"    [{_NV_GREEN}]+{line}[/{_NV_GREEN}]")
        if len(lines) > 8:
            con.print(f"    [dim]... ({len(lines)} lines total)[/dim]")
        return

    try:
        old = p.read_text()
        diff_lines = list(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=str(p),
                tofile=str(p),
            )
        )
        if diff_lines:
            _render_diff(con, "".join(diff_lines))
        else:
            con.print("[dim]  (no changes)[/dim]")
    except Exception:
        pass


def _beam_status(text: str, console: Console) -> Status:
    """Create a Status widget with the custom NVIDIA beam spinner."""
    status = Status(text, console=console, spinner="dots", spinner_style=_NV_GREEN)
    # Override the internal spinner frames with our beam animation
    status._spinner.frames = _BEAM_FRAMES
    status._spinner.interval = 80  # ms per frame — smooth, visible
    return status


def render_tool_result(
    con: Console,
    tool_name: str,
    result: str,
    is_error: bool,
    elapsed: float = 0.0,
    in_multi_tool: bool = False,
) -> None:
    """Render a tool result to the console."""
    _render_tool_result_inline(con, tool_name, result, is_error, elapsed, in_multi_tool)
