# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo obs — observability and cost tracking."""

from __future__ import annotations

import re
import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.metrics import PRICING


def _console() -> Console:
    """Return a Console that writes to the current sys.stdout.

    This is important for testability — typer's CliRunner replaces
    sys.stdout at invocation time, so a module-level Console would
    miss the redirection.
    """
    return Console(file=sys.stdout)


# Kept for backward compatibility with obs_tail / obs_pricing.
console = _console()
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


# ---------------------------------------------------------------------------
# Duration parsing helper
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)\s*(h|d|m|s|hr|hrs|min|mins|sec|secs|hour|hours|day|days)$", re.I)

_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}


def _parse_duration(text: str) -> float | None:
    """Parse a human duration like '24h', '7d', '30m' into seconds. Returns None on failure."""
    match = _DURATION_RE.match(text.strip())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    multiplier = _UNIT_SECONDS.get(unit)
    if multiplier is None:
        return None
    return float(value * multiplier)


# ---------------------------------------------------------------------------
# nemo obs usage
# ---------------------------------------------------------------------------


@obs_app.command("usage")
def obs_usage(
    since: str = typer.Option(None, "--since", help="Time range, e.g. '1h', '24h', '7d'"),
) -> None:
    """Show token usage summary by model."""
    from nemocode.core.sqlite_store import load_usage_summary

    since_ts: float | None = None
    label = ""
    c = _console()

    if since:
        duration_s = _parse_duration(since)
        if duration_s is None:
            c.print(f"[red]Could not parse duration: '{since}'. Use e.g. 1h, 24h, 7d.[/red]")
            raise typer.Exit(code=1)
        since_ts = time.time() - duration_s
        label = f" (last {since})"

    summary = load_usage_summary(since=since_ts)

    if not summary:
        c.print(f"[dim]No usage data recorded yet{label}.[/dim]")
        return

    table = Table(title=f"Token Usage by Model{label}", min_width=100)
    table.add_column("Model", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Est. Cost", justify="right", style="green")

    total_requests = 0
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_cost = 0.0

    for row in summary:
        model = row["model_id"]
        if len(model) > 40:
            model = "..." + model[-37:]
        reqs = row["requests"]
        pt = row["prompt_tokens"]
        ct = row["completion_tokens"]
        tt = row["total_tokens"]
        cost = row["estimated_cost"]

        total_requests += reqs
        total_prompt += pt
        total_completion += ct
        total_tokens += tt
        total_cost += cost

        table.add_row(
            model,
            f"{reqs:,}",
            f"{pt:,}",
            f"{ct:,}",
            f"{tt:,}",
            f"${cost:.6f}",
        )

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_requests:,}[/bold]",
        f"[bold]{total_prompt:,}[/bold]",
        f"[bold]{total_completion:,}[/bold]",
        f"[bold]{total_tokens:,}[/bold]",
        f"[bold green]${total_cost:.6f}[/bold green]",
    )

    c.print(table)
