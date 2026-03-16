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
def obs_tail(
    count: int = typer.Option(20, "-n", "--count", help="Number of recent metrics to show"),
) -> None:
    """Show recent request metrics from the metrics store."""
    # Load persisted metrics if available
    try:
        from nemocode.core.persistence import load_latest_session_metrics

        metrics_list = load_latest_session_metrics(count)
    except (ImportError, Exception):
        metrics_list = []

    if not metrics_list:
        console.print("[dim]No metrics recorded yet. Run a session first.[/dim]")
        console.print("[dim]Use /cost in the REPL for live session cost tracking.[/dim]")
        return

    table = Table(title=f"Recent Requests (last {len(metrics_list)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right", style="green")
    table.add_column("Tools", justify="right")

    for i, m in enumerate(metrics_list, 1):
        model = m.get("model_id", "?")
        if len(model) > 30:
            model = "..." + model[-27:]
        prompt_tok = m.get("prompt_tokens", 0)
        comp_tok = m.get("completion_tokens", 0)
        latency = m.get("total_time_ms", 0)
        cost = m.get("estimated_cost", 0)
        tools = m.get("tool_calls", 0)
        table.add_row(
            str(i),
            model,
            f"{prompt_tok:,}",
            f"{comp_tok:,}",
            f"{latency:.0f}ms",
            f"${cost:.6f}",
            str(tools),
        )

    console.print(table)


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
