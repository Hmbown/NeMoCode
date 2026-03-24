# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo model — inspect model manifests, pull NIM containers."""

from __future__ import annotations

import os
import shutil
import subprocess

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.config import load_config

console = Console()
model_app = typer.Typer(help="Inspect Nemotron model manifests and manage NIM containers.")

# ---------------------------------------------------------------------------
# NGC NIM container URI mapping
# ---------------------------------------------------------------------------
NGC_CONTAINER_MAP: dict[str, str] = {
    "nemotron-3-nano-4b": "nvcr.io/nim/nvidia/nemotron-3-nano-4b-instruct:latest",
    "nemotron-3-super-120b": "nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b-instruct:latest",
    "nemotron-nano-9b": "nvcr.io/nim/nvidia/nemotron-nano-9b-v2:latest",
    "nemotron-nano-12b-vl": "nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest",
}


def _resolve_ngc_key() -> str | None:
    """Resolve NGC API key from credentials store or environment.

    Checks NGC_CLI_API_KEY first, then falls back to NGC_API_KEY.
    """
    # Try credentials store
    try:
        from nemocode.core.credentials import get_credential

        for key_name in ("NGC_CLI_API_KEY", "NGC_API_KEY"):
            key = get_credential(key_name)
            if key:
                return key
    except ImportError:
        pass

    # Fall back to environment variables
    for env_var in ("NGC_CLI_API_KEY", "NGC_API_KEY"):
        key = os.environ.get(env_var)
        if key:
            return key

    return None


def _docker_login(api_key: str) -> None:
    """Authenticate with nvcr.io using the NGC API key."""
    result = subprocess.run(
        ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
        input=api_key,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Docker login failed:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)


def _docker_pull(uri: str) -> None:
    """Pull a container image, streaming output to the console."""
    result = subprocess.run(
        ["docker", "pull", uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Docker pull failed:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@model_app.command("pull")
def model_pull(
    model_name: str = typer.Argument(..., help="Model name to pull (e.g. nemotron-nano-9b)"),
    tag: str = typer.Option("latest", "--tag", "-t", help="Container image tag"),
) -> None:
    """Pull a NIM container image from NGC registry."""
    # 1. Check Docker is available
    if not shutil.which("docker"):
        console.print("[red]Error:[/red] Docker is not installed or not in PATH.")
        console.print("[dim]Install Docker: https://docs.docker.com/get-docker/[/dim]")
        raise typer.Exit(1)

    # 2. Resolve model name to container URI
    uri = NGC_CONTAINER_MAP.get(model_name)
    if not uri:
        console.print(f"[red]Unknown model:[/red] {model_name}")
        console.print("[dim]Available models:[/dim]")
        for name in sorted(NGC_CONTAINER_MAP):
            console.print(f"  [cyan]{name}[/cyan]")
        raise typer.Exit(1)

    # Apply custom tag if specified
    if tag != "latest":
        uri = uri.rsplit(":", 1)[0] + f":{tag}"

    # 3. Resolve NGC API key
    api_key = _resolve_ngc_key()
    if not api_key:
        console.print("[red]Error:[/red] NGC API key not found.")
        console.print(
            "[dim]Set it with:[/dim]\n"
            "  nemo auth set NGC_CLI_API_KEY\n"
            "  [dim]or[/dim] export NGC_CLI_API_KEY='your-key'"
        )
        raise typer.Exit(1)

    # 4. Docker login
    with console.status("[bold green]Authenticating with nvcr.io..."):
        _docker_login(api_key)
    console.print("[green]\u2713[/green] Authenticated with nvcr.io")

    # 5. Pull the image
    with console.status(f"[bold green]Pulling {uri}..."):
        _docker_pull(uri)
    console.print(f"[green]\u2713[/green] Successfully pulled [cyan]{uri}[/cyan]")


@model_app.command("ls")
def model_ls(
    local: bool = typer.Option(False, "--local", help="List locally pulled NIM container images"),
) -> None:
    """List model manifests, or locally pulled NIM containers with --local."""
    if local:
        _list_local_containers()
        return

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


def _list_local_containers() -> None:
    """List locally available NIM container images."""
    if not shutil.which("docker"):
        console.print("[red]Error:[/red] Docker is not installed or not in PATH.")
        raise typer.Exit(1)

    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Failed to list images:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)

    table = Table(title="Local NIM Container Images")
    table.add_column("Image", style="cyan")
    table.add_column("Size", justify="right")

    found = False
    for line in result.stdout.strip().splitlines():
        if not line.startswith("nvcr.io/nim/"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            image, size = parts
        else:
            image, size = line, "?"
        table.add_row(image, size)
        found = True

    if not found:
        console.print("[dim]No NIM containers found locally.[/dim]")
        console.print("[dim]Pull one with: nemo model pull <model-name>[/dim]")
        return

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
