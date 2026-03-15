# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo formation — manage and run formations."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from nemocode.config import load_config

console = Console()
formation_app = typer.Typer(help="Manage multi-model formations.")


@formation_app.command("ls")
def formation_ls() -> None:
    """List all configured formations."""
    cfg = load_config()
    table = Table(title="Formations")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Roles")
    table.add_column("Verification")

    for name, f in cfg.formations.items():
        roles = ", ".join(s.role.value for s in f.slots)
        marker = " *" if name == cfg.active_formation else ""
        table.add_row(
            f"{name}{marker}",
            f.description,
            roles,
            str(f.verification_rounds),
        )

    console.print(table)
    if cfg.active_formation:
        console.print("\n[dim]* = active formation[/dim]")


@formation_app.command("show")
def formation_show(
    name: str = typer.Argument(..., help="Formation name"),
) -> None:
    """Show formation details."""
    cfg = load_config()
    f = cfg.formations.get(name)
    if not f:
        console.print(f"[red]Unknown formation: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{f.name}[/bold]")
    console.print(f"[dim]{f.description}[/dim]\n")
    console.print(f"Tools: {', '.join(f.tools)}")
    console.print(f"Verification rounds: {f.verification_rounds}\n")
    console.print("[bold]Slots:[/bold]")
    for s in f.slots:
        extra = ""
        if s.reasoning_mode:
            extra = f" [dim](reasoning: {s.reasoning_mode})[/dim]"
        if s.lora_adapter:
            extra += f" [dim](LoRA: {s.lora_adapter})[/dim]"
        console.print(f"  {s.role.value:>10} -> {s.endpoint}{extra}")


@formation_app.command("run")
def formation_run(
    name: str = typer.Argument(..., help="Formation name"),
    prompt: str = typer.Argument(..., help="Task to execute"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Auto-approve all tool calls"),
) -> None:
    """Run a formation on a task."""
    asyncio.run(_run_formation(name, prompt, yes))


async def _run_formation(name: str, prompt: str, auto_yes: bool) -> None:
    from nemocode.cli.commands.code import _auto_confirm, _confirm, _render_event
    from nemocode.workflows.code_agent import CodeAgent

    cfg = load_config()
    cfg.active_formation = name

    confirm_fn = _auto_confirm if auto_yes else _confirm
    agent = CodeAgent(config=cfg, confirm_fn=confirm_fn)

    console.print(f"[bold]Formation: {name}[/bold]\n")

    async for event in agent.run(prompt):
        _render_event(event, show_thinking=False)
