# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo data — repo-aware planning and execution for NVIDIA data workflows."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from nemocode.core.data_workflows import (
    analyze_repo,
    build_preview_config,
    build_repo_data_plan,
    export_seeds,
    export_sft,
    generate_sft_via_nim,
    render_plan,
)

console = Console()
data_app = typer.Typer(help="Plan repo-aware NVIDIA synthetic data workflows.")
job_app = typer.Typer(help="Manage Data Designer async jobs.")
data_app.add_typer(job_app, name="job")


# ---------------------------------------------------------------------------
# Pipeline orchestrator (run the full repo-to-model flow in one command)
# ---------------------------------------------------------------------------


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
    continue_on_error: bool = False,
    skip_generate: bool = False,
) -> list[PipelineStep]:
    steps: list[PipelineStep] = [
        PipelineStep(
            name="Analyze repository",
            command=[
                "nemo",
                "data",
                "analyze",
                repo_path,
                "--output",
                f"{output_dir}/repo-data-plan.yaml",
            ],
            required=True,
        ),
        PipelineStep(
            name="Export seed artifacts",
            command=[
                "nemo",
                "data",
                "export-seeds",
                repo_path,
                "--output-dir",
                output_dir,
            ],
            required=True,
        ),
    ]

    if not skip_generate:
        # NIM API path: use Nemotron Super to generate high-quality synthetic data
        steps.append(
            PipelineStep(
                name="Generate SFT data (NIM API)",
                command=[
                    "nemo",
                    "data",
                    "generate",
                    "--seed-dir",
                    output_dir,
                    "--output",
                    f"{output_dir}/sft_generated.jsonl",
                    "--num-records",
                    str(max_records),
                ],
                required=True,
            ),
        )

    # Template-based fallback (always available, optional when generate is used)
    steps.append(
        PipelineStep(
            name="Export SFT dataset (template)",
            command=[
                "nemo",
                "data",
                "export-sft",
                repo_path,
                "--output",
                f"{output_dir}/sft_dataset.jsonl",
                "--max-records",
                str(max_records),
            ],
            required=skip_generate,
        ),
    )

    if not skip_finetune:
        # Use generated data if available, fall back to template-based
        dataset_path = (
            f"{output_dir}/sft_generated.jsonl"
            if not skip_generate
            else f"{output_dir}/sft_dataset.jsonl"
        )
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
        ".nemocode/data",
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
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
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
    skip_generate: bool = typer.Option(
        False,
        "--skip-generate",
        help="Skip NIM API generation; use template-based export-sft only.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="Continue past required step failures instead of stopping the pipeline.",
    ),
) -> None:
    """Run the full repo-to-fine-tuned-model pipeline."""
    steps = _build_steps(
        repo_path,
        max_records,
        skip_finetune,
        output_dir,
        model,
        continue_on_error=continue_on_error,
        skip_generate=skip_generate,
    )
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
            f"  Skip ftune:  {skip_finetune}\n"
            f"  Continue:    {continue_on_error}",
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
                if continue_on_error:
                    console.print("[yellow]Continuing with remaining steps...[/yellow]")
                else:
                    console.print("[red]Stopping after required step failure.[/red]")
                    progress.stop()
                    break

            progress.stop()

    result.print_summary()


# ---------------------------------------------------------------------------
# Default callback
# ---------------------------------------------------------------------------


@data_app.callback(invoke_without_command=True)
def data_default(ctx: typer.Context) -> None:
    """Analyze the current repository when no subcommand is specified."""
    if ctx.invoked_subcommand is None:
        data_analyze()


# ---------------------------------------------------------------------------
# nemo data analyze
# ---------------------------------------------------------------------------


@data_app.command("analyze")
def data_analyze(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the generated repo-to-data plan to this path.",
    ),
    output_format: str = typer.Option(
        "yaml",
        "--format",
        help="Plan serialization format: yaml or json.",
    ),
    show_plan: bool = typer.Option(
        False,
        "--show-plan",
        help="Print the generated plan after the summary.",
    ),
) -> None:
    """Analyze a repo and scaffold a repo-to-data plan for NVIDIA services."""
    output_format = output_format.lower().strip()
    if output_format not in {"yaml", "json"}:
        raise typer.BadParameter("format must be one of: yaml, json")

    profile = analyze_repo(path)
    plan = build_repo_data_plan(profile)

    console.print(
        Panel(
            f"[bold]Repo Root[/bold] {profile.root}\n"
            f"[bold]Files Scanned[/bold] {profile.total_files}\n"
            f"[bold]Code Files[/bold] {profile.code_files}\n"
            f"[bold]Tests[/bold] {'yes' if profile.has_tests else 'no'}\n"
            f"[bold]Docs[/bold] {'yes' if profile.has_docs else 'no'}",
            title="[bold]Repo Analysis[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Signal", style="cyan")
    table.add_column("Value")
    table.add_row(
        "Primary languages",
        ", ".join(profile.primary_languages) if profile.primary_languages else "none detected",
    )
    table.add_row(
        "Frameworks",
        ", ".join(profile.frameworks) if profile.frameworks else "none detected",
    )
    table.add_row(
        "Top directories",
        ", ".join(profile.top_directories) if profile.top_directories else "none detected",
    )
    table.add_row(
        "Dataset files",
        str(len(profile.dataset_files)) if profile.dataset_files else "0",
    )
    console.print(table)

    console.print("\n[bold]Recommended NVIDIA Stack[/bold]")
    stack = Table(show_header=True, box=None, padding=(0, 2))
    stack.add_column("Service", style="cyan")
    stack.add_column("Use It?")
    stack.add_column("Why", style="dim")
    for name, spec in plan["recommended_stack"].items():
        stack.add_row(name, spec["recommendation"], spec["why"])
    console.print(stack)

    console.print("\n[bold]Repo-to-Data MVP[/bold]")
    for step in plan["repo_to_data_mvp"]["recommended_next_steps"]:
        console.print(f"  - {step}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_plan(plan, output_format))
        console.print(f"\n[green]Wrote plan:[/green] {output}")
    else:
        console.print(
            "\n[dim]Write the plan with "
            "[bold]nemo data analyze --output .nemocode/data/repo-data-plan.yaml[/bold].[/dim]"
        )

    if show_plan:
        console.print()
        console.print(render_plan(plan, output_format), end="")


# ---------------------------------------------------------------------------
# nemo data export-seeds
# ---------------------------------------------------------------------------


@data_app.command("export-seeds")
def data_export_seeds(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False, dir_okay=True),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for seed artifacts (default: <repo>/.nemocode/data/).",
    ),
) -> None:
    """Export grounded repo-aware seed artifacts for Data Designer.

    Scans source, tests, and docs to produce:
    - repo_profile.yaml
    - file_manifest.jsonl
    - task_taxonomy.yaml
    - context_packs.jsonl
    """
    result = export_seeds(path, output_dir)

    console.print(
        Panel(
            f"[bold]Output Directory[/bold] {result.output_dir}\n"
            f"[bold]Files in Manifest[/bold] {result.file_count}\n"
            f"[bold]Profile[/bold] {result.profile_path}\n"
            f"[bold]Manifest[/bold] {result.manifest_path}\n"
            f"[bold]Taxonomy[/bold] {result.taxonomy_path}\n"
            f"[bold]Context Packs[/bold] {result.context_packs_path}",
            title="[bold]Seed Export Complete[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print("\n[dim]Next: preview synthetic data with [bold]nemo data preview[/bold].[/dim]")


# ---------------------------------------------------------------------------
# nemo data preview
# ---------------------------------------------------------------------------


@data_app.command("preview")
def data_preview(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False, dir_okay=True),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_DATA_BASE_URL",
        help="Data Designer base URL (default: http://localhost:8080).",
    ),
    num_records: int = typer.Option(
        5,
        "--num-records",
        "-n",
        help="Number of preview records to generate.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write preview results to a JSONL file.",
    ),
) -> None:
    """Preview synthetic data via a local Data Designer instance.

    Sends the repo-derived seed context to POST /v1/data-designer/preview
    and prints the generated records.
    """
    from nemocode.core.nvidia_client import DataDesignerClient

    profile = analyze_repo(path)
    config = build_preview_config(profile)

    client = DataDesignerClient(base_url=base_url)

    if not client.health():
        console.print(
            "[red]Cannot reach Data Designer[/red] at "
            f"[bold]{client.base_url}[/bold].\n"
            "Start the service or set [bold]--base-url[/bold] / "
            "[bold]NEMOCODE_DATA_BASE_URL[/bold]."
        )
        raise typer.Exit(1)

    console.print(f"[dim]Requesting {num_records} preview records from {client.base_url}...[/dim]")

    try:
        records = client.preview(config, num_records=num_records)
    except Exception as exc:
        console.print(f"[red]Preview failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not records:
        console.print("[yellow]No records returned.[/yellow]")
        raise typer.Exit(0)

    for i, rec in enumerate(records, 1):
        console.print(f"\n[bold cyan]Record {i}[/bold cyan]")
        console.print(json.dumps(rec, indent=2))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        console.print(f"\n[green]Wrote {len(records)} records to {output}[/green]")


# ---------------------------------------------------------------------------
# nemo data export-sft
# ---------------------------------------------------------------------------


@data_app.command("export-sft")
def data_export_sft(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False, dir_okay=True),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL path (default: <repo>/.nemocode/data/sft_dataset.jsonl).",
    ),
    max_records: int = typer.Option(
        0,
        "--max-records",
        help="Maximum records to generate (0 = all combinations).",
    ),
    include_tests: bool = typer.Option(
        False,
        "--include-tests/--no-include-tests",
        help="Include definitions from tests/. Disabled by default for higher-signal SFT data.",
    ),
    task_type: list[str] | None = typer.Option(
        None,
        "--task-type",
        help="Restrict output to one or more task types. Repeat the flag to include multiple types.",
    ),
) -> None:
    """Export a JSONL dataset suitable for SFT / instruction tuning.

    Each record has system/user/assistant messages grounded in the repo's
    implementation. The default export prefers implementation-heavy records
    and excludes tests for a higher-signal repo-specific tuning set.
    """
    seed_dir = path / ".nemocode" / "data" if (path / ".nemocode" / "data").exists() else None
    task_types = set(task_type) if task_type else None
    result = export_sft(
        path,
        output_path=output,
        seed_dir=seed_dir,
        max_records=max_records,
        include_tests=include_tests,
        task_types=task_types,
    )

    console.print(
        Panel(
            f"[bold]Output[/bold] {result.output_path}\n"
            f"[bold]Records[/bold] {result.record_count}\n"
            f"[bold]Include tests[/bold] {'yes' if include_tests else 'no'}\n"
            f"[bold]Task types[/bold] {', '.join(sorted(task_types)) if task_types else 'default high-signal set'}",
            title="[bold]SFT Dataset Export[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print("\n[dim]Use this dataset for repo-specific coding assistant fine-tuning.[/dim]")


# ---------------------------------------------------------------------------
# nemo data generate  (NIM API — no Data Designer Docker needed)
# ---------------------------------------------------------------------------


@data_app.command("generate")
def data_generate(
    seed_dir: Path = typer.Option(
        Path(".nemocode/data"),
        "--seed-dir",
        "-s",
        help="Directory containing seed artifacts from `nemo data export-seeds`.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL path (default: .nemocode/data/sft_generated.jsonl).",
    ),
    num_records: int = typer.Option(
        100,
        "--num-records",
        "-n",
        help="Number of SFT records to generate.",
    ),
    endpoint: str = typer.Option(
        "https://integrate.api.nvidia.com/v1",
        "--endpoint",
        "-e",
        envvar="NEMOCODE_NIM_ENDPOINT",
        help="NIM API endpoint (build.nvidia.com or local vLLM/NIM).",
    ),
    model: str = typer.Option(
        "nvidia/nemotron-3-super-120b-a12b",
        "--model",
        "-m",
        help="Model to use for generation (default: Nemotron 3 Super).",
    ),
) -> None:
    """Generate synthetic SFT training data via the NIM API.

    Uses Nemotron 3 Super (hosted on build.nvidia.com or running locally) to
    produce high-quality instruction-tuning data from your repo's seed artifacts.
    No Docker or Data Designer service required — just NVIDIA_API_KEY.

    \b
    Quick start:
      1. nemo data export-seeds          # scan repo, write seed artifacts
      2. nemo data generate              # call NIM API → sft_generated.jsonl
      3. nemo customize create --dataset .nemocode/data/sft_generated.jsonl

    For the full Data Designer experience with evaluation and quality scoring,
    use `nemo data preview` / `nemo data job create` with Docker instead.
    """
    seed_path = Path(seed_dir).resolve()
    if not seed_path.exists():
        console.print(
            f"[red]Seed directory not found: {seed_path}[/red]\n"
            "Run [bold]nemo data export-seeds[/bold] first."
        )
        raise typer.Exit(1)

    if not (seed_path / "context_packs.jsonl").exists():
        console.print(
            "[red]context_packs.jsonl not found in seed directory.[/red]\n"
            "Run [bold]nemo data export-seeds[/bold] first."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Seed dir:[/bold]   {seed_path}\n"
            f"[bold]Records:[/bold]    {num_records}\n"
            f"[bold]Endpoint:[/bold]   {endpoint}\n"
            f"[bold]Model:[/bold]      {model}",
            title="[bold]nemo data generate[/bold]",
            border_style="blue",
            expand=False,
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[bold]Generating {num_records} records via NIM...[/bold]", total=num_records
        )

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done)

        try:
            result = generate_sft_via_nim(
                seed_dir=seed_path,
                output_path=output,
                num_records=num_records,
                endpoint=endpoint,
                model=model,
                progress_callback=on_progress,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]Generation failed:[/red] {exc}")
            raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Output[/bold]    {result.output_path}\n"
            f"[bold]Records[/bold]   {result.record_count}\n"
            f"[bold]Skipped[/bold]   {result.skipped}\n"
            f"[bold]Model[/bold]     {result.model}",
            title="[bold green]Generation Complete[/bold green]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print(
        f"\n[dim]Next: fine-tune with "
        f"[bold]nemo customize create --dataset {result.output_path}[/bold][/dim]"
    )


# ---------------------------------------------------------------------------
# nemo data job create
# ---------------------------------------------------------------------------


@job_app.command("create")
def job_create(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False, dir_okay=True),
    name: str = typer.Option(
        "nemocode-data-job",
        "--name",
        help="Job name.",
    ),
    num_records: int = typer.Option(
        100,
        "--num-records",
        "-n",
        help="Number of records to generate.",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_DATA_BASE_URL",
        help="Data Designer base URL.",
    ),
    description: str = typer.Option(
        "",
        "--description",
        help="Job description.",
    ),
) -> None:
    """Create an async Data Designer generation job."""
    from nemocode.core.nvidia_client import DataDesignerClient

    profile = analyze_repo(path)
    config = build_preview_config(profile)
    client = DataDesignerClient(base_url=base_url)

    if not client.health():
        console.print(f"[red]Cannot reach Data Designer at {client.base_url}[/red]")
        raise typer.Exit(1)

    try:
        job = client.create_job(name, config, num_records=num_records, description=description)
    except Exception as exc:
        console.print(f"[red]Job creation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"[bold]Job ID[/bold] {job.get('id', 'unknown')}\n"
            f"[bold]Name[/bold] {job.get('name', name)}\n"
            f"[bold]Status[/bold] {job.get('status', 'unknown')}\n"
            f"[bold]Records[/bold] {num_records}",
            title="[bold]Job Created[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print(
        f"\n[dim]Check status: [bold]nemo data job status {job.get('id', '')}[/bold][/dim]"
    )


# ---------------------------------------------------------------------------
# nemo data job status
# ---------------------------------------------------------------------------


@job_app.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Job ID to check."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_DATA_BASE_URL",
        help="Data Designer base URL.",
    ),
) -> None:
    """Check the status of a Data Designer job."""
    from nemocode.core.nvidia_client import DataDesignerClient

    client = DataDesignerClient(base_url=base_url)
    try:
        status = client.get_job_status(job_id)
    except Exception as exc:
        console.print(f"[red]Failed to get status:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# nemo data job logs
# ---------------------------------------------------------------------------


@job_app.command("logs")
def job_logs(
    job_id: str = typer.Argument(..., help="Job ID."),
    limit: int = typer.Option(100, "--limit", help="Max log entries."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_DATA_BASE_URL",
        help="Data Designer base URL.",
    ),
) -> None:
    """Fetch logs for a Data Designer job."""
    from nemocode.core.nvidia_client import DataDesignerClient

    client = DataDesignerClient(base_url=base_url)
    try:
        logs = client.get_job_logs(job_id, limit=limit)
    except Exception as exc:
        console.print(f"[red]Failed to get logs:[/red] {exc}")
        raise typer.Exit(1) from exc

    for entry in logs.get("data", [logs] if "message" in logs else []):
        if isinstance(entry, dict):
            console.print(entry.get("message", json.dumps(entry)))
        else:
            console.print(str(entry))


# ---------------------------------------------------------------------------
# nemo data job results
# ---------------------------------------------------------------------------


@job_app.command("results")
def job_results(
    job_id: str = typer.Argument(..., help="Job ID."),
    download: bool = typer.Option(
        False, "--download", help="Download the dataset to .nemocode/data/."
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Download destination path."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="NEMOCODE_DATA_BASE_URL",
        help="Data Designer base URL.",
    ),
) -> None:
    """List or download results for a Data Designer job."""
    from nemocode.core.nvidia_client import DataDesignerClient

    client = DataDesignerClient(base_url=base_url)

    if download:
        try:
            data = client.download_dataset(job_id)
        except Exception as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        dest = output or Path(f".nemocode/data/job-{job_id}-dataset.jsonl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        console.print(f"[green]Downloaded dataset to {dest}[/green]")
        return

    try:
        results = client.get_job_results(job_id)
    except Exception as exc:
        console.print(f"[red]Failed to get results:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(json.dumps(results, indent=2))
