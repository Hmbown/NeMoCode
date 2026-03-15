# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo code — agentic coding with tools."""

from __future__ import annotations

import asyncio
import json
import sys

import typer
from rich.console import Console
from rich.text import Text

from nemocode.core.scheduler import AgentEvent
from nemocode.workflows.code_agent import CodeAgent

console = Console()


def code_cmd(
    prompt: str = typer.Argument(None, help="Coding task (omit to start interactive REPL)"),
    endpoint: str = typer.Option(None, "-e", "--endpoint", help="Endpoint override"),
    formation: str = typer.Option(None, "-f", "--formation", help="Formation to use"),
    think: bool = typer.Option(False, "--think", help="Show thinking trace"),
    voice: bool = typer.Option(False, "--voice", help="Enable voice input (requires microphone)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Auto-approve all tool calls"),
) -> None:
    """Agentic coding — one-shot, piped, or interactive REPL.

    Without a prompt argument (and not piped), starts an interactive REPL session.
    With a prompt, runs a single one-shot turn and exits.
    """
    if voice:
        from nemocode.voice.detector import detect_voice_capabilities

        caps = detect_voice_capabilities()
        if not caps.available:
            console.print(f"[yellow]Voice mode unavailable: {caps.reason}[/yellow]")
            raise typer.Exit(1)
        console.print(f"[dim]Voice: {caps.reason}[/dim]")

    # Read from stdin if piped
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()

    # No prompt available: launch interactive REPL
    if not prompt:
        from nemocode.cli.commands.repl import start_repl

        start_repl(
            endpoint=endpoint,
            formation=formation,
            think=think,
            yes=yes,
        )
        return

    # One-shot mode with a prompt
    asyncio.run(_code(prompt, endpoint, formation, think, yes))


async def _confirm(tool_name: str, args: dict) -> bool:
    """Ask user for confirmation before executing a tool."""
    console.print(f"\n[yellow]Tool: {tool_name}[/yellow]")
    args_preview = json.dumps(args, indent=2)[:500]
    console.print(f"[dim]{args_preview}[/dim]")
    response = console.input("[yellow]Allow? [y/N]: [/yellow]")
    return response.lower() in ("y", "yes")


async def _auto_confirm(tool_name: str, args: dict) -> bool:
    return True


async def _code(
    prompt: str,
    endpoint_name: str | None,
    formation_name: str | None,
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
    agent = CodeAgent(config=cfg, confirm_fn=confirm_fn)

    console.print(f"[dim]Endpoint: {cfg.default_endpoint}[/dim]")
    if cfg.active_formation:
        console.print(f"[dim]Formation: {cfg.active_formation}[/dim]")
    console.print()

    async for event in agent.run(prompt):
        _render_event(event, show_thinking)


def _render_event(event: AgentEvent, show_thinking: bool) -> None:
    """Render an agent event to the terminal."""
    if event.kind == "text":
        console.print(event.text, end="", highlight=False, markup=False)
    elif event.kind == "thinking" and show_thinking:
        console.print(event.thinking, end="", style="dim", highlight=False, markup=False)
    elif event.kind == "phase":
        role_label = event.role.value if event.role else "agent"
        console.print(f"\n[bold blue]--- {role_label}: {event.text} ---[/bold blue]")
    elif event.kind == "tool_call":
        args = event.tool_args
        tool = event.tool_name
        if tool == "read_file":
            console.print(f"  [cyan]> read_file[/cyan] {args.get('path', '?')}")
        elif tool == "write_file":
            console.print(f"  [cyan]> write_file[/cyan] {args.get('path', '?')}")
        elif tool == "edit_file":
            console.print(f"  [cyan]> edit_file[/cyan] {args.get('path', '?')}")
        elif tool == "bash_exec":
            cmd = args.get("command", "?")
            if len(cmd) > 80:
                cmd = cmd[:77] + "..."
            console.print(f"  [cyan]> bash_exec[/cyan] {cmd}")
        elif tool == "list_dir":
            console.print(f"  [cyan]> list_dir[/cyan] {args.get('path', '.')}")
        else:
            console.print(f"  [cyan]> {tool}[/cyan]")
    elif event.kind == "tool_result":
        result = event.tool_result[:2000]
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                if "error" in parsed:
                    console.print(Text(f"    error: {parsed['error']}", style="red"))
                    return
                if parsed.get("status") == "ok":
                    console.print(Text("    ok", style="green"))
                    return
                if "exit_code" in parsed:
                    out = parsed.get("stdout", "").strip()
                    if parsed["exit_code"] == 0 and out:
                        for line in out.splitlines()[:10]:
                            console.print(Text(f"    {line}", style="dim"))
                        return
                    elif parsed["exit_code"] == 0:
                        console.print(Text("    ok", style="green"))
                        return
        except (json.JSONDecodeError, ValueError):
            pass
        for line in result.splitlines()[:15]:
            console.print(Text(f"    {line}", style="dim"))
    elif event.kind == "usage":
        u = event.usage
        console.print(
            f"\n[dim]Tokens: {u.get('prompt_tokens', 0)}p + {u.get('completion_tokens', 0)}c[/dim]"
        )
    elif event.kind == "error":
        console.print(f"\n[bold red]Error: {event.text}[/bold red]")
