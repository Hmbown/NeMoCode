# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo code — agentic coding with tools."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console

from nemocode.cli.render import EventRenderer, format_confirm_summary, render_confirm_detail
from nemocode.workflows.code_agent import CodeAgent

console = Console()


def code_cmd(
    prompt: str = typer.Argument(None, help="Coding task (omit to start interactive REPL)"),
    endpoint: str = typer.Option(None, "-e", "--endpoint", help="Endpoint override"),
    formation: str = typer.Option(None, "-f", "--formation", help="Formation to use"),
    agent: str = typer.Option(None, "-a", "--agent", help="Primary agent profile to use"),
    think: bool = typer.Option(True, "--think/--no-think", help="Show/hide thinking trace"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Auto-approve all tool calls"),
    tui: bool = typer.Option(False, "--tui", help="Launch full-screen TUI instead of REPL"),
) -> None:
    """Agentic coding — one-shot, piped, or interactive REPL.

    Without a prompt argument (and not piped), starts an interactive REPL session.
    With --tui, launches a full-screen Textual interface instead.
    With a prompt, runs a single one-shot turn and exits.
    """
    # Read from stdin if piped
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()

    # No prompt available: launch interactive REPL or TUI
    if not prompt:
        if tui:
            from nemocode.cli.tui import start_tui

            start_tui(
                endpoint=endpoint,
                formation=formation,
                agent_name=agent,
                think=think,
                yes=yes,
            )
        else:
            from nemocode.cli.commands.repl import start_repl

            start_repl(
                endpoint=endpoint,
                formation=formation,
                agent_name=agent,
                think=think,
                yes=yes,
            )
        return

    # One-shot mode with a prompt
    asyncio.run(_code(prompt, endpoint, formation, agent, think, yes))


async def _confirm(tool_name: str, args: dict) -> bool:
    """Ask user for confirmation before executing a tool."""
    summary = format_confirm_summary(tool_name, args)
    console.print(f"\n[yellow]Allow [bold]{tool_name}[/bold]?[/yellow]")
    console.print(f"[dim]  {summary}[/dim]")
    render_confirm_detail(console, tool_name, args)
    try:
        response = console.input("[yellow]  [y/N]: [/yellow]")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.lower() in ("y", "yes")


async def _auto_confirm(tool_name: str, args: dict) -> bool:
    return True


async def _code(
    prompt: str,
    endpoint_name: str | None,
    formation_name: str | None,
    agent_name: str | None,
    show_thinking: bool,
    auto_yes: bool,
) -> None:
    from nemocode.config import load_config

    cfg = load_config()

    if endpoint_name:
        cfg.default_endpoint = endpoint_name
    if formation_name:
        cfg.active_formation = formation_name

    confirm_fn = _auto_confirm if auto_yes else _confirm
    agent = CodeAgent(config=cfg, confirm_fn=confirm_fn, agent_name=agent_name)

    renderer = EventRenderer(console, show_thinking=show_thinking)
    renderer.start_thinking("Working")

    # Track totals for final summary
    total_prompt = 0
    total_completion = 0
    tool_count = 0

    async for event in agent.run(prompt):
        renderer.render(event)
        if event.kind == "usage":
            total_prompt += event.usage.get("prompt_tokens", 0)
            total_completion += event.usage.get("completion_tokens", 0)
        elif event.kind == "tool_call":
            tool_count += 1

    renderer.flush()

    # Compact summary at the end
    total = total_prompt + total_completion
    if total > 0:
        parts = []
        if tool_count > 0:
            parts.append(f"{tool_count} tool call{'s' if tool_count != 1 else ''}")
        parts.append(f"{total:,} tokens")
        console.print(f"\n[dim]{' · '.join(parts)}[/dim]")
