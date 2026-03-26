# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo customize — customization workflows for Nemotron models."""

from __future__ import annotations

import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.providers.nvidia_client import (
    CustomizerAPIError,
    CustomizerClient,
    DEFAULT_FINETUNING_TYPE,
    DEFAULT_LORA_RANK,
    DEFAULT_MODEL,
    DEFAULT_TRAINING_TYPE,
)

console = Console()
customize_app = typer.Typer(help="Customizer workflows for Nemotron models (LoRA and full SFT).")


def _normalize_finetuning_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"full-sft", "full_sft", "all-weights", "all_weights"}:
        return "all_weights"
    if normalized == "lora":
        return "lora"
    raise typer.BadParameter("finetuning type must be one of: lora, full-sft")


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
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Path to SFT dataset (JSONL) for the legacy local-file workflow.",
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
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Customizer config URN (preferred platform workflow).",
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--dataset-name",
        help="Dataset name in NeMo Data Store for platform workflow.",
    ),
    dataset_namespace: str = typer.Option(
        "default",
        "--dataset-namespace",
        help="Dataset namespace for platform workflow.",
    ),
    training_type: str = typer.Option(
        DEFAULT_TRAINING_TYPE,
        "--training-type",
        help="Training type for platform workflow.",
    ),
    finetuning_type: str = typer.Option(
        DEFAULT_FINETUNING_TYPE,
        "--finetuning-type",
        help="Finetuning type: lora or full-sft.",
    ),
    max_seq_length: Optional[int] = typer.Option(
        None,
        "--max-seq-length",
        help="Override max sequence length for platform workflow.",
    ),
) -> None:
    """Submit a customization job.

    Preferred path is config-driven NeMo Customizer:
      nemo customize create --config <config-urn> --dataset-name <dataset>

    Legacy local-dev fallback still accepts --dataset with a JSONL path.
    """
    client = _get_client()
    normalized_ft_type = _normalize_finetuning_type(finetuning_type)

    if config and not dataset_name:
        raise typer.BadParameter("--dataset-name is required when --config is used.")
    if dataset_name and not config:
        raise typer.BadParameter("--config is required when --dataset-name is used.")
    if normalized_ft_type == "all_weights" and not config:
        raise typer.BadParameter("full-sft requires --config and --dataset-name platform inputs.")

    create_kwargs = {
        "dataset_path": dataset,
        "model": model,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "output_model": output_model,
    }
    if config:
        create_kwargs.update(
            {
                "config": config,
                "dataset_name": dataset_name,
                "dataset_namespace": dataset_namespace,
                "training_type": training_type,
                "finetuning_type": normalized_ft_type,
                "max_seq_length": max_seq_length,
                "wandb_api_key": os.environ.get("WANDB_API_KEY"),
            }
        )

    try:
        result = client.create_job(**create_kwargs)
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    job_id = result.get("id", "unknown")
    status = result.get("status", "unknown")
    dataset_ref = (
        f"{dataset_namespace}/{dataset_name}" if config and dataset_name else dataset
    )
    workflow = "platform config" if config else "local file"

    console.print(
        Panel(
            f"[bold green]Job created[/bold green]\n\n"
            f"  Job ID:  {job_id}\n"
            f"  Mode:    {workflow}\n"
            f"  Model:   {model}\n"
            f"  Config:  {config or '-'}\n"
            f"  Dataset: {dataset_ref}\n"
            f"  Status:  {status}\n"
            f"  Finetune: {normalized_ft_type}\n"
            f"  LoRA rank: {lora_rank if normalized_ft_type == 'lora' else '-'}\n"
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


@customize_app.command("configs")
def customize_configs(
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help="Filter configs by base model.",
    ),
    training_type: str = typer.Option(
        DEFAULT_TRAINING_TYPE,
        "--training-type",
        help="Filter configs by training type.",
    ),
    finetuning_type: str = typer.Option(
        DEFAULT_FINETUNING_TYPE,
        "--finetuning-type",
        help="Filter configs by finetuning type: lora or full-sft.",
    ),
    show_disabled: bool = typer.Option(
        False,
        "--show-disabled",
        help="Include disabled configs.",
    ),
) -> None:
    """List available Customizer configs from the NeMo platform."""
    client = _get_client()
    normalized_ft_type = _normalize_finetuning_type(finetuning_type)
    try:
        configs = client.list_configs(
            base_model=base_model,
            training_type=training_type,
            finetuning_type=normalized_ft_type,
            enabled=not show_disabled,
        )
    except CustomizerAPIError as err:
        _handle_api_error(err)
    except Exception:
        _handle_connection_error()

    if not configs:
        console.print("[dim]No customization configs found.[/dim]")
        return

    table = Table(title="Customization Configs")
    table.add_column("Config", style="cyan")
    table.add_column("Base Model")
    table.add_column("Precision")
    table.add_column("Seq Len", justify="right")
    table.add_column("Training")
    table.add_column("FT Type")

    for cfg in configs:
        training_options = cfg.get("training_options", [])
        option = training_options[0] if training_options else {}
        config_name = cfg.get("name", "unknown")
        namespace = cfg.get("namespace")
        config_urn = f"{namespace}/{config_name}" if namespace else config_name
        table.add_row(
            config_urn,
            cfg.get("target", cfg.get("base_model", "-")),
            str(cfg.get("training_precision", cfg.get("precision", "-"))),
            str(cfg.get("max_seq_length", "-")),
            str(option.get("training_type", training_type)),
            str(option.get("finetuning_type", normalized_ft_type)),
        )

    console.print(table)
