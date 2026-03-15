# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo model — inspect model manifests."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.config import load_config

console = Console()
model_app = typer.Typer(help="Inspect Nemotron model manifests.")


@model_app.command("ls")
def model_ls() -> None:
    """List all known model manifests."""
    cfg = load_config()
    table = Table(title="Model Manifests")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name")
    table.add_column("Arch")
    table.add_column("Tier")
    table.add_column("Params (active/total)")
    table.add_column("Context", justify="right")
    table.add_column("Tools")

    for model_id, m in cfg.manifests.items():
        params = ""
        if m.moe.total_params_b:
            params = f"{m.moe.active_params_b:.0f}B / {m.moe.total_params_b:.0f}B"
        ctx = f"{m.context_window:,}"
        table.add_row(
            model_id,
            m.display_name,
            m.arch.value,
            m.tier_in_family or "-",
            params,
            ctx,
            "yes" if m.supports_tools else "no",
        )

    console.print(table)


@model_app.command("show")
def model_show(
    model_id: str = typer.Argument(..., help="Model ID to inspect"),
) -> None:
    """Show detailed manifest for a model."""
    cfg = load_config()
    m = cfg.manifests.get(model_id)
    if not m:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        console.print(f"[dim]Available: {', '.join(cfg.manifests.keys())}[/dim]")
        raise typer.Exit(1)

    lines = [
        f"[bold]{m.display_name}[/bold] ({model_id})",
        f"Architecture: {m.arch.value}",
        f"Family tier: {m.tier_in_family or 'N/A'}",
        f"Context window: {m.context_window:,} tokens",
        f"Max output: {m.max_output_tokens:,} tokens",
        f"Capabilities: {', '.join(c.value for c in m.capabilities)}",
    ]

    if m.moe.total_params_b:
        lines.extend(
            [
                "",
                "[bold]MoE Architecture[/bold]",
                f"  Total params: {m.moe.total_params_b:.0f}B",
                f"  Active params: {m.moe.active_params_b:.0f}B",
                f"  Latent MoE: {m.moe.uses_latent_moe}",
                f"  Multi-token prediction: {m.moe.uses_mtp}",
                f"  Mamba layers: {m.moe.uses_mamba}",
                f"  Precision: {m.moe.precision}",
            ]
        )

    if m.reasoning.supports_thinking:
        lines.extend(
            [
                "",
                "[bold]Reasoning[/bold]",
                f"  Thinking param: {m.reasoning.thinking_param}",
                f"  Budget control: {m.reasoning.supports_budget_control}",
                f"  Budget param: {m.reasoning.thinking_budget_param}",
                f"  No-think tag: {m.reasoning.no_think_tag}",
            ]
        )

    if m.min_gpu_memory_gb:
        lines.extend(
            [
                "",
                "[bold]Hardware Requirements[/bold]",
                f"  Min VRAM: {m.min_gpu_memory_gb}GB",
                f"  Min GPUs: {m.min_gpus or 1}",
                f"  Recommended: {m.recommended_gpu}",
            ]
        )

    console.print(Panel("\n".join(lines), title="Model Manifest", border_style="blue"))
