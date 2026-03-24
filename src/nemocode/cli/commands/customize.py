# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo customize — LoRA fine-tuning for Nemotron models."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.providers.nvidia_client import (
    CustomizerAPIError,
    CustomizerClient,
    DEFAULT_LORA_RANK,
    DEFAULT_MODEL,
)

console = Console()
customize_app = typer.Typer(help="LoRA fine-tuning for Nemotron models.")


def _get_client() -> CustomizerClient:
    """Build a CustomizerClient from environment."""
    return CustomizerClient()


def _handle_api_error(err: CustomizerAPIError) -> None:
    """Print a friendly error message for API failures."""
    if err.status_code == 404:
        console.print(f"[red]Not found:[/red] {err.detail}")
    elif err.status_code == 401:
        console.print(
            "[red]Authentication failed.[/red] Set NVIDIA_API_KEY or check your credentials."
        )
    elif err.status_code == 422:
        console.print(f"[red]Validation error:[/red] {err.detail}")
    else:
        console.print(f"[red]API error (HTTP {err.status_code}):[/red] {err.detail}")
    raise typer.Exit(1)


def _handle_connection_error() -> None:
    """Print a friendly error for connection failures."""
    console.print(
        "[red]Could not connect to the Customizer service.[/red]\n"
        "Make sure it is running and NEMOCODE_CUSTOMIZER_BASE_URL is set correctly."
    )
    raise typer.Exit(1)


# -----------------------------------------------------------------------
# create
# -----------------------------------------------------------------------


@customize_app.command("create")
def customize_create(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to SFT dataset (JSONL).",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Base model to fine-tune.",
    ),
    lora_rank: int = typer.Option(
        DEFAULT_LORA_RANK,
        "--lora-rank",
        "-r",
        help="LoRA rank.",
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs."),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size."),
    learning_rate: float = typer.Option(1e-4, "--lr", help="Learning rate."),
    output_model: Optional[str] = typer.Option(
        None,
        "--output-model",
        help="Name for the output fine-tuned model.",
    ),
) -> None:
    """Submit a LoRA fine-tuning job."""
    client = _get_client()
    try:
        result = client.create_job(
            dataset_path=dataset,
            model=model,
            lora_rank=lora_rank,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_model=output_model,
        )
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    job_id = result.get("id", "unknown")
    status = result.get("status", "unknown")

    console.print(
        Panel(
            f"[bold green]Job created[/bold green]\n\n"
            f"  Job ID:  {job_id}\n"
            f"  Model:   {model}\n"
            f"  Dataset: {dataset}\n"
            f"  Status:  {status}\n"
            f"  LoRA rank: {lora_rank}\n"
            f"  Epochs:  {epochs}",
            title="Customization Job",
            border_style="green",
        )
    )


# -----------------------------------------------------------------------
# status
# -----------------------------------------------------------------------


@customize_app.command("status")
def customize_status(
    job_id: str = typer.Argument(..., help="Job ID to check."),
) -> None:
    """Show the status of a customization job."""
    client = _get_client()
    try:
        result = client.get_status(job_id)
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    status = result.get("status", "unknown")
    model = result.get("model", "")
    created = result.get("created_at", "")
    progress = result.get("progress")

    lines = [
        f"[bold]Job:[/bold] {job_id}",
        f"[bold]Status:[/bold] {status}",
        f"[bold]Model:[/bold] {model}",
        f"[bold]Created:[/bold] {created}",
    ]
    if progress is not None:
        lines.append(f"[bold]Progress:[/bold] {progress}%")

    # Show hyperparameters if present
    hp = result.get("hyperparameters", {})
    if hp:
        lines.append("")
        lines.append("[bold]Hyperparameters[/bold]")
        for key, val in hp.items():
            lines.append(f"  {key}: {val}")

    console.print(Panel("\n".join(lines), title="Job Status", border_style="blue"))


# -----------------------------------------------------------------------
# logs
# -----------------------------------------------------------------------


@customize_app.command("logs")
def customize_logs(
    job_id: str = typer.Argument(..., help="Job ID to get logs for."),
) -> None:
    """Show training logs for a customization job."""
    client = _get_client()
    try:
        result = client.get_logs(job_id)
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    log_lines = result.get("logs", [])
    if isinstance(log_lines, str):
        log_lines = log_lines.splitlines()

    if not log_lines:
        console.print("[dim]No logs available yet.[/dim]")
        return

    console.print(Panel("\n".join(log_lines), title=f"Logs — {job_id}", border_style="cyan"))


# -----------------------------------------------------------------------
# results
# -----------------------------------------------------------------------


@customize_app.command("results")
def customize_results(
    job_id: str = typer.Argument(..., help="Job ID to get results for."),
) -> None:
    """Show or download results for a completed customization job."""
    client = _get_client()
    try:
        result = client.get_results(job_id)
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    output_model = result.get("output_model", "")
    metrics = result.get("metrics", {})
    artifacts = result.get("artifacts", [])

    lines = [
        f"[bold]Job:[/bold] {job_id}",
        f"[bold]Output model:[/bold] {output_model}",
    ]

    if metrics:
        lines.append("")
        lines.append("[bold]Metrics[/bold]")
        for key, val in metrics.items():
            lines.append(f"  {key}: {val}")

    if artifacts:
        lines.append("")
        lines.append("[bold]Artifacts[/bold]")
        for art in artifacts:
            if isinstance(art, str):
                lines.append(f"  - {art}")
            elif isinstance(art, dict):
                lines.append(f"  - {art.get('name', art.get('path', str(art)))}")

    console.print(Panel("\n".join(lines), title="Results", border_style="green"))


# -----------------------------------------------------------------------
# ls
# -----------------------------------------------------------------------


@customize_app.command("ls")
def customize_ls() -> None:
    """List all customization jobs."""
    client = _get_client()
    try:
        jobs = client.list_jobs()
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    if not jobs:
        console.print("[dim]No customization jobs found.[/dim]")
        return

    table = Table(title="Customization Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Progress", justify="right")

    for job in jobs:
        progress = job.get("progress")
        progress_str = f"{progress}%" if progress is not None else "-"
        table.add_row(
            job.get("id", ""),
            job.get("model", ""),
            job.get("status", ""),
            job.get("created_at", ""),
            progress_str,
        )

    console.print(table)
