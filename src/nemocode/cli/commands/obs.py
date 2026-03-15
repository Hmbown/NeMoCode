# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo obs — observability and cost tracking."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.metrics import PRICING

console = Console()
obs_app = typer.Typer(help="Observability and cost tracking.")


@obs_app.command("tail")
def obs_tail() -> None:
    """Show live metrics (placeholder — will stream from metrics collector)."""
    console.print("[dim]Live metrics streaming not yet implemented.[/dim]")
    console.print("[dim]Use /cost in the TUI for session cost tracking.[/dim]")


@obs_app.command("pricing")
def obs_pricing() -> None:
    """Show model pricing estimates."""
    table = Table(title="NIM API Pricing (per 1M tokens)")
    table.add_column("Model", style="cyan")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")

    for model_id, prices in PRICING.items():
        table.add_row(
            model_id,
            f"${prices['input']:.2f}",
            f"${prices['output']:.2f}",
        )

    console.print(table)
    console.print(
        "\n[dim]Prices are estimates for NIM API Catalog. Self-hosted costs vary by hardware.[/dim]"
    )
