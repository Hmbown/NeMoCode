# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo agent — inspect configured agent profiles."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from nemocode.config import load_config
from nemocode.config.agents import resolve_agent_reference
from nemocode.config.schema import AgentMode
from nemocode.core.subagents import list_runs
from nemocode.tools.delegate import _pick_endpoint

console = Console()
agent_app = typer.Typer(help="Inspect primary agents and sub-agents.")


@agent_app.command("ls")
def agent_ls() -> None:
    """List configured agent profiles."""
    cfg = load_config()
    table = Table(title="Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Display")
    table.add_column("Mode")
    table.add_column("Endpoint")
    table.add_column("Tools")

    for name, agent in sorted(cfg.agents.items()):
        endpoint = "-"
        if agent.mode != AgentMode.PRIMARY:
            endpoint = _pick_endpoint(cfg, agent.prefer_tiers, explicit_endpoint=agent.endpoint)
        tools = ", ".join(agent.tools) if agent.tools else "-"
        table.add_row(name, agent.display_name or "-", agent.mode.value, endpoint, tools)

    console.print(table)


@agent_app.command("show")
def agent_show(
    name: str = typer.Argument(..., help="Agent profile name"),
) -> None:
    """Show one configured agent profile."""
    cfg = load_config()
    resolved = resolve_agent_reference(cfg.agents, name) or name
    agent = cfg.agents.get(resolved)
    if not agent:
        console.print(f"[red]Unknown agent: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{agent.name}[/bold]")
    if agent.display_name:
        console.print(f"Display name: {agent.display_name}")
    console.print(f"Mode: {agent.mode.value}")
    console.print(f"Role: {agent.role.value}")
    if agent.description:
        console.print(f"Description: {agent.description}")
    if agent.aliases:
        console.print(f"Aliases: {', '.join(agent.aliases)}")
    if agent.hidden:
        console.print("Hidden: true")
    if agent.endpoint:
        console.print(f"Endpoint override: {agent.endpoint}")
    elif agent.prefer_tiers:
        resolved = _pick_endpoint(cfg, agent.prefer_tiers)
        console.print(f"Preferred tiers: {', '.join(agent.prefer_tiers)}")
        console.print(f"Resolved endpoint: {resolved}")
    if agent.tools:
        console.print(f"Tools: {', '.join(agent.tools)}")
    if agent.prompt:
        console.print("\n[bold]Prompt:[/bold]")
        console.print(agent.prompt)


@agent_app.command("runs")
def agent_runs(
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Max runs to show"),
) -> None:
    """Show recent delegated sub-agent runs."""
    runs = list_runs(limit=limit)
    if not runs:
        console.print("[dim]No delegated sub-agent runs yet.[/dim]")
        return

    table = Table(title="Recent Sub-Agent Runs")
    table.add_column("Run", style="cyan")
    table.add_column("Nickname")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Endpoint")
    table.add_column("Tools")

    for run in runs:
        tools = f"{run.tool_calls}"
        if run.errors:
            tools += f" / {run.errors} err"
        status = run.status + (" (closed)" if run.closed else "")
        table.add_row(
            run.id,
            run.nickname,
            f"{run.agent_name} ({run.display_name})",
            status,
            run.endpoint,
            tools,
        )

    console.print(table)
