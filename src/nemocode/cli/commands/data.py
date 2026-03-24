# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo data — repo-aware planning and execution for NVIDIA data workflows."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.core.data_workflows import (
    analyze_repo,
    build_preview_config,
    build_repo_data_plan,
    export_seeds,
    export_sft,
    render_plan,
)

console = Console()
data_app = typer.Typer(help="Plan repo-aware NVIDIA synthetic data workflows.")
job_app = typer.Typer(help="Manage Data Designer async jobs.")
data_app.add_typer(job_app, name="job")


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
) -> None:
    """Export a JSONL dataset suitable for SFT / instruction tuning.

    Each record has system/user/assistant messages grounded in the repo's
    structure, languages, and task taxonomy. This is for repo-specific
    assistant tuning, not raw code dumping.
    """
    seed_dir = path / ".nemocode" / "data" if (path / ".nemocode" / "data").exists() else None
    result = export_sft(path, output_path=output, seed_dir=seed_dir, max_records=max_records)

    console.print(
        Panel(
            f"[bold]Output[/bold] {result.output_path}\n[bold]Records[/bold] {result.record_count}",
            title="[bold]SFT Dataset Export[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print("\n[dim]Use this dataset for repo-specific coding assistant fine-tuning.[/dim]")


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
