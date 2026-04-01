# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo safe-synth — manage NeMo Safe Synthesizer jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.core.nvidia_client import SafeSynthesizerClient

console = Console()
safe_synth_app = typer.Typer(help="Manage NeMo Safe Synthesizer jobs.")


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
        console.print("[red]Safe Synthesizer spec must be a JSON object.[/red]")
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


@safe_synth_app.command("create")
def safe_synth_create(
    spec: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to safe-synth spec JSON."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_SAFE_SYNTH_BASE_URL",
        help="Safe Synthesizer base URL.",
    ),
) -> None:
    """Create a Safe Synthesizer job from a JSON spec."""
    client = SafeSynthesizerClient(base_url=base_url)
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
            title="[bold]Safe Synthesizer Job Created[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )


@safe_synth_app.command("ls")
def safe_synth_ls(
    page: int = typer.Option(1, "--page", min=1, help="Result page."),
    page_size: int = typer.Option(20, "--page-size", min=1, help="Rows per page."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_SAFE_SYNTH_BASE_URL",
        help="Safe Synthesizer base URL.",
    ),
) -> None:
    """List Safe Synthesizer jobs."""
    client = SafeSynthesizerClient(base_url=base_url)
    try:
        payload = client.list_jobs(page=page, page_size=page_size)
    except Exception as exc:
        console.print(f"[red]Failed to list jobs:[/red] {exc}")
        raise typer.Exit(1) from exc

    rows = _iter_rows(payload)
    if not rows:
        console.print("[dim]No Safe Synthesizer jobs found.[/dim]")
        return

    table = Table(title="Safe Synthesizer Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Name")

    for row in rows:
        table.add_row(
            str(row.get("id", "")),
            str(row.get("status", "")),
            str(row.get("created_at", "")),
            str(row.get("name", "")),
        )

    console.print(table)


@safe_synth_app.command("status")
def safe_synth_status(
    job_id: str = typer.Argument(..., help="Job ID to inspect."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_SAFE_SYNTH_BASE_URL",
        help="Safe Synthesizer base URL.",
    ),
) -> None:
    """Show Safe Synthesizer job status."""
    client = SafeSynthesizerClient(base_url=base_url)
    try:
        status = client.get_job_status(job_id)
    except Exception as exc:
        console.print(f"[red]Failed to get status:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(json.dumps(status, indent=2))


@safe_synth_app.command("logs")
def safe_synth_logs(
    job_id: str = typer.Argument(..., help="Job ID."),
    limit: int = typer.Option(100, "--limit", min=1, help="Maximum log entries."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_SAFE_SYNTH_BASE_URL",
        help="Safe Synthesizer base URL.",
    ),
) -> None:
    """Fetch Safe Synthesizer job logs."""
    client = SafeSynthesizerClient(base_url=base_url)
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


@safe_synth_app.command("results")
def safe_synth_results(
    job_id: str = typer.Argument(..., help="Job ID."),
    download: bool = typer.Option(
        False,
        "--download",
        help="Download the synthetic dataset instead of printing metadata.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Download destination path.",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_SAFE_SYNTH_BASE_URL",
        help="Safe Synthesizer base URL.",
    ),
) -> None:
    """Show or download Safe Synthesizer job results."""
    client = SafeSynthesizerClient(base_url=base_url)

    if download:
        try:
            data = client.download_synthetic_data(job_id)
        except Exception as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        dest = output or Path(f".nemocode/data/job-{job_id}-safe-synth.jsonl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        console.print(f"[green]Downloaded synthetic data to {dest}[/green]")
        return

    try:
        payload = client.get_job_results(job_id)
    except Exception as exc:
        console.print(f"[red]Failed to get results:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(json.dumps(payload, indent=2))
