# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo hardware — hardware detection and recommendations."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.config import load_config
from nemocode.core.hardware import detect_hardware, refresh_hardware

console = Console()
hardware_app = typer.Typer(help="Hardware detection and recommendations.")


@hardware_app.callback(invoke_without_command=True)
def hardware_default(ctx: typer.Context) -> None:
    """Show detected hardware."""
    if ctx.invoked_subcommand is None:
        hardware_show()


@hardware_app.command("show")
def hardware_show() -> None:
    """Show detected hardware profile."""
    profile = detect_hardware()
    border = "bright_green" if profile.is_dgx_spark else "blue"
    title = "DGX Spark Detected" if profile.is_dgx_spark else "Hardware Profile"
    console.print(Panel(profile.summary(), title=title, border_style=border))


@hardware_app.command("recommend")
def hardware_recommend() -> None:
    """Recommend formations and local models based on hardware."""
    profile = detect_hardware()
    cfg = load_config()

    border = "bright_green" if profile.is_dgx_spark else "blue"
    title = "DGX Spark Detected" if profile.is_dgx_spark else "Hardware Profile"
    console.print(Panel(profile.summary(), title=title, border_style=border))
    console.print()

    # DGX Spark concurrent deployment info
    if profile.is_dgx_spark:
        console.print(
            "[bold bright_green]DGX Spark — Run Multiple Models"
            " Simultaneously[/bold bright_green]\n"
        )

        concurrent = profile.spark_concurrent_configs()
        if concurrent:
            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("Role", style="cyan")
            table.add_column("Model")
            table.add_column("Memory", justify="right")
            table.add_column("Port", justify="right")

            total_mem = 0.0
            for cfg_item in concurrent:
                table.add_row(
                    cfg_item["role"],
                    cfg_item["model_id"].split("/")[-1],
                    cfg_item["memory"],
                    cfg_item["port"],
                )
                # Parse "~80GB" → 80
                mem_str = cfg_item["memory"].replace("~", "").replace("GB", "").strip()
                try:
                    total_mem += float(mem_str)
                except ValueError:
                    pass

            remaining = profile.unified_memory_gb - total_mem
            table.add_row("", "", "─" * 8, "")
            table.add_row(
                "[dim]Total[/dim]",
                "",
                f"[bold]~{total_mem:.0f} GB[/bold]",
                "",
            )
            table.add_row(
                "[dim]Remaining[/dim]",
                "[dim]OS + apps[/dim]",
                f"[green]~{remaining:.0f} GB[/green]",
                "",
            )
            console.print(table)
            console.print()

        console.print(
            "[dim]Run [bold]nemo setup spark[/bold] for step-by-step"
            " deployment instructions.[/dim]\n"
        )

    # Formation recommendation
    rec_formation = profile.recommend_formation()
    console.print(f"[bold]Recommended formation:[/bold] [cyan]{rec_formation}[/cyan]")
    f = cfg.formations.get(rec_formation)
    if f:
        console.print(f"  [dim]{f.description}[/dim]")

    # Other good formations for this hardware
    if profile.is_dgx_spark:
        console.print("\n[bold]Also available on your Spark:[/bold]")
        spark_formations = [
            "spark",
            "spark-sglang",
            "spark-vllm",
            "spark-swarm",
            "spark-vision",
            "spark-retrieval",
            "spark-full",
        ]
        for fname in spark_formations:
            form = cfg.formations.get(fname)
            if form and fname != rec_formation:
                console.print(f"  [cyan]{fname}[/cyan] — {form.description}")

    # Local model recommendations
    local_models = profile.recommend_local_models()
    if local_models:
        console.print("\n[bold]Models that can run locally:[/bold]")
        for model_id in local_models:
            m = cfg.manifests.get(model_id)
            if m:
                console.print(f"  [green]{model_id}[/green] — {m.display_name}")
                if m.moe.total_params_b > 0:
                    active = f"{m.moe.active_params_b:.0f}B"
                    total = f"{m.moe.total_params_b:.0f}B"
                    vram = m.min_gpu_memory_gb
                    console.print(f"    [dim]{active} active / {total} total, needs {vram}GB[/dim]")
    else:
        console.print("\n[yellow]No Nemotron 3 models can run locally on this hardware.[/yellow]")
        console.print("[dim]Use hosted endpoints (build.nvidia.com) instead.[/dim]")

    # Model compatibility check
    console.print("\n[bold]Full compatibility check:[/bold]")
    table = Table()
    table.add_column("Model")
    table.add_column("Local?")
    table.add_column("Notes")

    for model_id, m in cfg.manifests.items():
        if m.min_gpu_memory_gb:
            can_run, reason = profile.can_run_model(m)
            status = "[green]Yes[/green]" if can_run else "[red]No[/red]"
            table.add_row(m.display_name, status, reason)

    console.print(table)

    # Voice mode availability
    if profile.has_microphone:
        console.print("\n[green]Microphone detected — voice mode available.[/green]")
    else:
        console.print("\n[dim]No microphone detected — voice mode unavailable.[/dim]")


@hardware_app.command("refresh")
def hardware_refresh_cmd() -> None:
    """Force re-detection of hardware (clear cache)."""
    console.print("Refreshing hardware detection...")
    profile = refresh_hardware()
    border = "bright_green" if profile.is_dgx_spark else "green"
    title = "DGX Spark (refreshed)" if profile.is_dgx_spark else "Hardware Profile (refreshed)"
    console.print(Panel(profile.summary(), title=title, border_style=border))
