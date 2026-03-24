# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo data — dataset preparation and fine-tuning pipeline."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
data_app = typer.Typer(help="Dataset preparation and fine-tuning pipeline.")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PipelineStep:
    name: str
    command: list[str]
    status: StepStatus = StepStatus.PENDING
    message: str = ""
    required: bool = True


@dataclass
class PipelineResult:
    steps: list[PipelineStep] = field(default_factory=list)
    dry_run: bool = False

    def add(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def print_summary(self) -> None:
        table = Table(title="Pipeline Summary")
        table.add_column("#", style="dim", width=3)
        table.add_column("Step", style="bold")
        table.add_column("Status")
        table.add_column("Message", max_width=50)

        for i, step in enumerate(self.steps, 1):
            status_style = {
                StepStatus.DONE: "[green]✓ done[/green]",
                StepStatus.SKIPPED: "[dim]⊘ skipped[/dim]",
                StepStatus.FAILED: "[red]✗ failed[/red]",
                StepStatus.RUNNING: "[yellow]● running[/yellow]",
                StepStatus.PENDING: "[dim]○ pending[/dim]",
            }[step.status]
            table.add_row(str(i), step.name, status_style, step.message)

        console.print(table)

        failed = [s for s in self.steps if s.status == StepStatus.FAILED]
        completed = [s for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED)]

        if failed:
            console.print(
                f"\n[bold red]Pipeline finished with {len(failed)} failed step(s).[/bold red]"
            )
        else:
            console.print(
                f"\n[bold green]Pipeline finished successfully — {len(completed)}/{len(self.steps)} steps completed.[/bold green]"
            )


def _run_step(step: PipelineStep) -> None:
    step.status = StepStatus.RUNNING
    try:
        result = subprocess.run(
            step.command,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            step.status = StepStatus.DONE
            step.message = "Completed successfully"
        else:
            step.status = StepStatus.FAILED
            step.message = (
                result.stderr.strip() or result.stdout.strip() or f"Exit code {result.returncode}"
            )
    except FileNotFoundError:
        step.status = StepStatus.FAILED
        step.message = f"Command not found: {step.command[0]}"
    except subprocess.TimeoutExpired:
        step.status = StepStatus.FAILED
        step.message = "Timed out after 600s"
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.message = str(exc)


def _build_steps(
    repo_path: str,
    max_records: int,
    skip_finetune: bool,
    output_dir: str,
    model: str,
) -> list[PipelineStep]:
    steps: list[PipelineStep] = [
        PipelineStep(
            name="Analyze repository",
            command=["nemo", "data", "analyze", "--repo", repo_path, "--output-dir", output_dir],
            required=True,
        ),
        PipelineStep(
            name="Export seed artifacts",
            command=[
                "nemo",
                "data",
                "export-seeds",
                "--repo",
                repo_path,
                "--output-dir",
                output_dir,
            ],
            required=True,
        ),
        PipelineStep(
            name="Preview dataset",
            command=["nemo", "data", "preview", "--repo", repo_path, "--output-dir", output_dir],
            required=False,
        ),
        PipelineStep(
            name="Export SFT dataset",
            command=[
                "nemo",
                "data",
                "export-sft",
                "--repo",
                repo_path,
                "--output-dir",
                output_dir,
                "--max-records",
                str(max_records),
            ],
            required=True,
        ),
    ]

    if not skip_finetune:
        dataset_path = f"{output_dir}/sft_dataset.jsonl"
        steps.append(
            PipelineStep(
                name="Submit fine-tuning job",
                command=[
                    "nemo",
                    "customize",
                    "create",
                    "--dataset",
                    dataset_path,
                    "--model",
                    model,
                ],
                required=True,
            ),
        )

    return steps


@data_app.command("pipeline")
def data_pipeline(
    repo_path: str = typer.Option(
        ".",
        "--repo",
        "-r",
        help="Path to the repository to analyze.",
    ),
    output_dir: str = typer.Option(
        ".nemo/data",
        "--output-dir",
        "-o",
        help="Directory for pipeline artifacts.",
    ),
    max_records: int = typer.Option(
        500,
        "--max-records",
        "-n",
        help="Maximum records for the SFT dataset.",
    ),
    model: str = typer.Option(
        "nvidia/nemotron-3-nano-4b-v1.1",
        "--model",
        "-m",
        help="Base model for fine-tuning.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the pipeline plan without executing.",
    ),
    skip_finetune: bool = typer.Option(
        False,
        "--skip-finetune",
        help="Skip the fine-tuning step.",
    ),
) -> None:
    """Run the full repo-to-fine-tuned-model pipeline."""
    steps = _build_steps(repo_path, max_records, skip_finetune, output_dir, model)
    result = PipelineResult(dry_run=dry_run)

    for step in steps:
        result.add(step)

    console.print(
        Panel(
            f"[bold]nemo data pipeline[/bold]\n"
            f"  Repo:        {repo_path}\n"
            f"  Output:      {output_dir}\n"
            f"  Max records: {max_records}\n"
            f"  Model:       {model}\n"
            f"  Dry run:     {dry_run}\n"
            f"  Skip ftune:  {skip_finetune}",
            title="Pipeline Configuration",
            border_style="blue",
        )
    )

    if dry_run:
        for step in steps:
            step.message = " ".join(step.command)
        result.print_summary()
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for step in steps:
            progress.start_task(progress.add_task(f"[bold]{step.name}[/bold]", total=None))
            _run_step(step)

            if step.status == StepStatus.FAILED and step.required:
                console.print(f"[yellow]Warning:[/yellow] {step.name} failed: {step.message}")
                console.print("[yellow]Continuing with remaining steps...[/yellow]")

            progress.stop()

    result.print_summary()
