# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo doctor — run diagnostics to check setup."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.config import load_config
from nemocode.core.doctor import run_diagnostics

console = Console()
doctor_app = typer.Typer(help="Run diagnostics to check setup.")


@doctor_app.callback(invoke_without_command=True)
def doctor(ctx: typer.Context) -> None:
    """Run all diagnostic checks and display a report."""
    if ctx.invoked_subcommand is None:
        doctor_show()


@doctor_app.command("show")
def doctor_show() -> None:
    """Show diagnostic report."""
    config = load_config()
    report = run_diagnostics(config)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="dim", width=20)
    table.add_column("Status")
    table.add_column("Detail")

    for check in report.checks:
        status_style = {
            "ok": "green",
            "warn": "yellow",
            "fail": "red",
        }.get(check.status, "white")
        table.add_row(
            check.name,
            f"[{status_style}]{check.status}[/{status_style}]",
            check.detail or "-",
        )

    console.print(Panel(table, title="Diagnostic Report", border_style="blue"))

    if report.ok:
        console.print("\n[bold green]All checks passed![/bold green]")
    else:
        console.print("\n[bold red]Some checks failed or warned.[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    doctor_app()
