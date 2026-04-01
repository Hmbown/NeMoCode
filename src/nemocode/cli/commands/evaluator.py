# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo evaluator — manage NeMo Evaluator jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.core.nvidia_client import EvaluatorClient

console = Console()
evaluator_app = typer.Typer(help="Manage NeMo Evaluator jobs.")


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        console.print(f"[red]Spec file not found:[/red] {path}")
        raise typer.Exit(1) from exc
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {path}:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not isinstance(data, dict):
        console.print("[red]Evaluator spec must be a JSON object.[/red]")
        raise typer.Exit(1)
    return data


def _iter_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "jobs", "items", "results"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


@evaluator_app.command("create")
def evaluator_create(
    spec: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to evaluation spec JSON."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_EVAL_BASE_URL",
        help="Evaluator base URL.",
    ),
) -> None:
    """Create a NeMo Evaluator job from a JSON spec."""
    client = EvaluatorClient(base_url=base_url)
    job_spec = _load_json_file(spec)

    try:
        job = client.create_job(job_spec)
    except Exception as exc:
        console.print(f"[red]Job creation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"[bold]Job ID[/bold] {job.get('id', 'unknown')}\n"
            f"[bold]Status[/bold] {job.get('status', 'unknown')}\n"
            f"[bold]Spec[/bold] {spec}",
            title="[bold]Evaluator Job Created[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )


@evaluator_app.command("ls")
def evaluator_ls(
    page: int = typer.Option(1, "--page", min=1, help="Result page."),
    page_size: int = typer.Option(20, "--page-size", min=1, help="Rows per page."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_EVAL_BASE_URL",
        help="Evaluator base URL.",
    ),
) -> None:
    """List evaluator jobs."""
    client = EvaluatorClient(base_url=base_url)
    try:
        payload = client.list_jobs(page=page, page_size=page_size)
    except Exception as exc:
        console.print(f"[red]Failed to list jobs:[/red] {exc}")
        raise typer.Exit(1) from exc

    rows = _iter_rows(payload)
    if not rows:
        console.print("[dim]No evaluator jobs found.[/dim]")
        return

    table = Table(title="Evaluator Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Target")

    for row in rows:
        target = row.get("target")
        if isinstance(target, dict):
            target = target.get("name") or target.get("id") or json.dumps(target)
        table.add_row(
            str(row.get("id", "")),
            str(row.get("status", "")),
            str(row.get("created_at", "")),
            str(target or ""),
        )

    console.print(table)


@evaluator_app.command("status")
def evaluator_status(
    job_id: str = typer.Argument(..., help="Job ID to inspect."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_EVAL_BASE_URL",
        help="Evaluator base URL.",
    ),
) -> None:
    """Show evaluator job status."""
    client = EvaluatorClient(base_url=base_url)
    try:
        status = client.get_job_status(job_id)
    except Exception as exc:
        console.print(f"[red]Failed to get status:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(json.dumps(status, indent=2))


@evaluator_app.command("logs")
def evaluator_logs(
    job_id: str = typer.Argument(..., help="Job ID."),
    limit: int = typer.Option(100, "--limit", min=1, help="Maximum log entries."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_EVAL_BASE_URL",
        help="Evaluator base URL.",
    ),
) -> None:
    """Fetch evaluator job logs."""
    client = EvaluatorClient(base_url=base_url)
    try:
        logs = client.get_job_logs(job_id, limit=limit)
    except Exception as exc:
        console.print(f"[red]Failed to get logs:[/red] {exc}")
        raise typer.Exit(1) from exc

    entries = logs.get("data", [logs] if isinstance(logs, dict) else [])
    if not entries:
        console.print("[dim]No logs available yet.[/dim]")
        return

    for entry in entries:
        if isinstance(entry, dict):
            console.print(entry.get("message", json.dumps(entry)))
        else:
            console.print(str(entry))


@evaluator_app.command("results")
def evaluator_results(
    job_id: str = typer.Argument(..., help="Job ID."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write results JSON to a file instead of stdout.",
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Use the evaluator download endpoint when available.",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_EVAL_BASE_URL",
        help="Evaluator base URL.",
    ),
) -> None:
    """Show or save evaluator job results."""
    client = EvaluatorClient(base_url=base_url)
    try:
        payload = client.download_results(job_id) if download else client.get_job_results(job_id)
    except Exception as exc:
        console.print(f"[red]Failed to get results:[/red] {exc}")
        raise typer.Exit(1) from exc

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        console.print(f"[green]Wrote results to {output}[/green]")
        return

    console.print(json.dumps(payload, indent=2))
