# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NeMoCode TUI — interactive terminal interface built with Textual.

Features:
- Streaming chat with syntax-highlighted code blocks
- Tool call visualization
- Thinking trace display
- Status bar with endpoint, tokens, context usage
- Slash commands
"""

from __future__ import annotations

import json

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, RichLog, Static

from nemocode.config import load_config
from nemocode.core.context import ContextManager
from nemocode.core.scheduler import AgentEvent
from nemocode.workflows.code_agent import CodeAgent


class StatusBar(Static):
    """Bottom status bar showing endpoint, tokens, context usage."""

    def __init__(self) -> None:
        super().__init__("")
        self._endpoint = ""
        self._formation = ""
        self._tokens = 0
        self._cost = 0.0
        self._context_pct = 0.0
        self._current_tool: str | None = None

    def update_status(
        self,
        endpoint: str | None = None,
        formation: str | None = None,
        tokens: int | None = None,
        cost: float | None = None,
        context_pct: float | None = None,
        current_tool: str | None = None,
    ) -> None:
        if endpoint is not None:
            self._endpoint = endpoint
        if formation is not None:
            self._formation = formation
        if tokens is not None:
            self._tokens = tokens
        if cost is not None:
            self._cost = cost
        if context_pct is not None:
            self._context_pct = context_pct
        if current_tool is not None:
            self._current_tool = current_tool
        elif current_tool is None:
            # Explicitly clear if None is passed
            self._current_tool = None

        parts = []
        if self._endpoint:
            parts.append(f"EP: {self._endpoint}")
        if self._formation:
            parts.append(f"FM: {self._formation}")
        parts.append(f"Tokens: {self._tokens:,}")
        if self._cost > 0:
            parts.append(f"Cost: ${self._cost:.4f}")
        if self._current_tool:
            parts.append(f"Tool: {self._current_tool}")

        # Context bar
        filled = int(self._context_pct * 20)
        bar = "#" * filled + "-" * (20 - filled)
        parts.append(f"Ctx: [{bar}] {self._context_pct * 100:.0f}%")

        self.update(" | ".join(parts))


class NeMoCodeApp(App):
    """Interactive TUI for NeMoCode."""

    CSS = """
    Screen {
        background: #000000;
        color: #E5E5E5;
    }
    #chat-log {
        height: 1fr;
        border: solid #76B900;
        padding: 0 1;
    }
    #input-area {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    #input-area Input {
        background: #000000;
        color: #E5E5E5;
        width: 100%;
        height: 100%;
    }
    #input-area Input::placeholder {
        color: #8A8A8A;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: #000000;
        color: #8A8A8A;
        padding: 0 1;
        text-style: bold;
    }
    .tool-call {
        color: #76B900;
    }
    .thinking {
        color: #8A8A8A;
    }
    .error {
        color: #FF0000;
    }
    .phase {
        color: #76B900;
        text-style: bold;
    }
    Header {
        background: #000000;
        color: #E5E5E5;
        text-style: bold;
    }
    Footer {
        background: #000000;
        color: #E5E5E5;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        self._agent: CodeAgent | None = None
        self._context_mgr = ContextManager()
        self._total_tokens = 0
        self._show_thinking = True
        self._last_tool_call = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(RichLog(highlight=True, markup=True, id="chat-log"))
        yield StatusBar()
        yield Input(placeholder="Type a message or /help for commands...", id="input-area")
        yield Footer()

    def on_mount(self) -> None:
        self._log = self.query_one("#chat-log", RichLog)
        self._status = self.query_one(StatusBar)
        self._input = self.query_one("#input-area", Input)

        ep = self._config.default_endpoint
        fm = self._config.active_formation or ""
        self._status.update_status(endpoint=ep, formation=fm)

        self._log.write("[bold]NeMoCode[/bold] — Nemotron 3 agentic coding")
        self._log.write(f"[dim]Endpoint: {ep} | Formation: {fm or 'none'}[/dim]")
        self._log.write("[dim]Type /help for commands.[/dim]\n")

        self._input.focus()

    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self._input.clear()

        if text.startswith("/"):
            await self._handle_slash(text)
            return

        self._log.write(f"\n[bold cyan]You:[/bold cyan] {text}")
        self._run_agent(text)

    @work(thread=False)
    async def _run_agent(self, text: str) -> None:
        if self._agent is None:
            self._agent = CodeAgent(config=self._config, confirm_fn=self._tui_confirm)

        self._log.write("")

        async for event in self._agent.run(text):
            self._render_event(event)

    def _render_event(self, event: AgentEvent) -> None:
        if event.kind == "text":
            # Escape model output to prevent Rich markup errors
            from rich.text import Text

            self._log.write(Text(event.text), shrink=False)
        elif event.kind == "thinking" and self._show_thinking:
            self._log.write(f"[dim]{event.thinking}[/dim]", shrink=False)
        elif event.kind == "phase":
            role = event.role.value if event.role else "agent"
            self._log.write(f"\n[bold blue]--- {role}: {event.text} ---[/bold blue]")
        elif event.kind == "tool_call":
            args_str = json.dumps(event.tool_args, indent=2)[:500]
            self._log.write(f"\n[cyan]Tool: {event.tool_name}[/cyan]")
            self._log.write(f"[dim]{args_str}[/dim]")
        elif event.kind == "tool_result":
            result = event.tool_result[:1000]
            style = "red" if event.is_error else "green"
            self._log.write(f"[{style}]{result}[/{style}]")
        elif event.kind == "usage":
            tokens = event.usage.get("total_tokens", 0)
            self._total_tokens += tokens
            self._status.update_status(tokens=self._total_tokens)
        elif event.kind == "error":
            self._log.write(f"\n[bold red]Error: {event.text}[/bold red]")

    async def _tui_confirm(self, tool_name: str, args: dict) -> bool:
        # TUI auto-approves; modal confirmation requires Textual Screen/Dialog
        self._log.write(f"[yellow]Auto-approved: {tool_name}[/yellow]")
        return True

    async def _handle_slash(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/help":
            self._log.write("\n[bold]Commands:[/bold]")
            self._log.write("  /help      — Show this help")
            self._log.write("  /think     — Toggle thinking trace")
            self._log.write("  /cost      — Show session cost")
            self._log.write("  /hardware  — Show hardware info")
            self._log.write("  /endpoint  — Show current endpoint")
            self._log.write("  /formation — Show current formation")
            self._log.write("  /clear     — Clear chat")
            self._log.write("  /reset     — Reset conversation")
            self._log.write("  /quit      — Exit")
        elif cmd == "/think":
            self._show_thinking = not self._show_thinking
            state = "on" if self._show_thinking else "off"
            self._log.write(f"[dim]Thinking trace: {state}[/dim]")
        elif cmd == "/cost":
            self._log.write("\n[bold]Session Cost[/bold]")
            self._log.write(f"  Total tokens: {self._total_tokens:,}")
            from nemocode.core.metrics import estimate_cost

            ep = self._config.endpoints.get(self._config.default_endpoint)
            if ep:
                cost = estimate_cost(ep.model_id, self._total_tokens // 2, self._total_tokens // 2)
                self._log.write(f"  Estimated cost: ${cost:.4f}")
        elif cmd == "/hardware":
            from nemocode.core.hardware import detect_hardware

            profile = detect_hardware()
            self._log.write(f"\n{profile.summary()}")
        elif cmd == "/endpoint":
            self._log.write(f"\n[cyan]Current endpoint: {self._config.default_endpoint}[/cyan]")
        elif cmd == "/formation":
            fm = self._config.active_formation or "none"
            self._log.write(f"\n[cyan]Current formation: {fm}[/cyan]")
        elif cmd == "/clear":
            self._log.clear()
        elif cmd == "/reset":
            if self._agent:
                self._agent.reset()
            self._total_tokens = 0
            self._status.update_status(tokens=0, cost=0.0)
            self._log.write("[dim]Conversation reset.[/dim]")
        elif cmd == "/quit":
            self.exit()
        else:
            self._log.write(f"[yellow]Unknown command: {cmd}[/yellow]")

    def action_clear_chat(self) -> None:
        self._log.clear()

    def action_quit(self) -> None:
        self.exit()
