# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo setup — guided setup for local inference backends (NIM, vLLM, Ollama)."""

from __future__ import annotations

import shutil

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
setup_app = typer.Typer(help="Set up local inference backends.")


@setup_app.callback(invoke_without_command=True)
def setup_default(ctx: typer.Context) -> None:
    """Show available setup options."""
    if ctx.invoked_subcommand is None:
        console.print(Panel(
            "[bold]NeMoCode Local Inference Setup[/bold]\n\n"
            "Set up a local model server for offline/private coding.\n"
            "NeMoCode works with any OpenAI-compatible endpoint.",
            border_style="bright_green",
            expand=False,
        ))
        console.print()

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Backend", style="cyan bold")
        table.add_column("Command", style="dim")
        table.add_column("Best For")

        table.add_row(
            "NIM Container",
            "nemo setup nim",
            "Production GPU inference (H100/A100/B200)",
        )
        table.add_row(
            "vLLM",
            "nemo setup vllm",
            "Flexible GPU serving with quantization",
        )
        table.add_row(
            "Ollama",
            "nemo setup ollama",
            "Easy setup, CPU/Apple Silicon friendly",
        )

        console.print(table)
        console.print()
        console.print("[dim]All backends serve an OpenAI-compatible API that "
                      "NeMoCode connects to automatically.[/dim]")


@setup_app.command("nim")
def setup_nim() -> None:
    """Set up NVIDIA NIM container for local inference."""
    console.print("[bold]NVIDIA NIM Container Setup[/bold]\n")

    _check_docker()

    console.print("NIM containers provide optimized inference for Nemotron 3.\n")
    console.print("[bold]Prerequisites:[/bold]")
    console.print("  1. NVIDIA GPU with sufficient VRAM")
    console.print("  2. Docker with NVIDIA Container Toolkit")
    console.print("  3. NGC API key (for pulling containers)\n")

    console.print("[bold]Quick start (Nemotron 3 Nano, needs 24GB+ VRAM):[/bold]")
    console.print()
    console.print(Panel(
        "# Pull and run Nemotron 3 Nano via NIM\n"
        "export NGC_API_KEY='your-ngc-key'\n"
        "\n"
        "docker run -d --name nemotron-nano \\\n"
        "  --gpus all \\\n"
        "  -p 8000:8000 \\\n"
        "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
        "  nvcr.io/nim/nvidia/nemotron-3-nano-30b-a3b:latest\n"
        "\n"
        "# Then configure NeMoCode:\n"
        "nemo endpoint add local-nano \\\n"
        "  --base-url http://localhost:8000/v1 \\\n"
        "  --model-id nvidia/nemotron-3-nano-30b-a3b \\\n"
        "  --tier local-nim\n"
        "\n"
        "# Test it:\n"
        "nemo endpoint test local-nano\n"
        "nemo code -e local-nano",
        title="[bold]Shell Commands[/bold]",
        border_style="green",
    ))

    console.print("\n[bold]Nemotron 3 Super (needs 80GB+ VRAM):[/bold]")
    console.print(Panel(
        "docker run -d --name nemotron-super \\\n"
        "  --gpus all \\\n"
        "  -p 8000:8000 \\\n"
        "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
        "  nvcr.io/nim/nvidia/"
        "nemotron-3-super-120b-a12b:latest",
        title="[bold]Shell Commands[/bold]",
        border_style="green",
    ))

    console.print("\n[dim]Docs: https://docs.nvidia.com/nim/large-language-models/"
                  "latest/getting-started.html[/dim]")


@setup_app.command("vllm")
def setup_vllm() -> None:
    """Set up vLLM for local model serving."""
    console.print("[bold]vLLM Setup for Nemotron 3[/bold]\n")

    has_pip = shutil.which("pip") or shutil.which("pip3")
    if has_pip:
        console.print("[green]pip found[/green]\n")
    else:
        console.print("[yellow]pip not found — install Python first[/yellow]\n")

    console.print("[bold]Quick start:[/bold]")
    console.print()
    console.print(Panel(
        "# Install vLLM\n"
        "pip install vllm\n"
        "\n"
        "# Serve Nemotron 3 Nano (FP8, ~24GB VRAM)\n"
        "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\\n"
        "  --port 8000 \\\n"
        "  --max-model-len 32768 \\\n"
        "  --enable-auto-tool-choice \\\n"
        '  --tool-call-parser hermes\n'
        "\n"
        "# Configure NeMoCode:\n"
        "nemo endpoint add local-vllm \\\n"
        "  --base-url http://localhost:8000/v1 \\\n"
        "  --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\\n"
        "  --tier local-vllm\n"
        "\n"
        "# Test it:\n"
        "nemo code -e local-vllm",
        title="[bold]Shell Commands[/bold]",
        border_style="green",
    ))

    console.print("\n[bold]For Nemotron 3 Super (NVFP4, ~80GB VRAM):[/bold]")
    console.print(Panel(
        "vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \\\n"
        "  --port 8000 \\\n"
        "  --tensor-parallel-size 2 \\\n"
        "  --max-model-len 131072 \\\n"
        "  --enable-auto-tool-choice \\\n"
        '  --tool-call-parser hermes',
        title="[bold]Shell Commands[/bold]",
        border_style="green",
    ))

    console.print("\n[dim]vLLM supports tool calling via --enable-auto-tool-choice. "
                  "Use --tool-call-parser hermes for Nemotron models.[/dim]")
    console.print("[dim]Docs: https://docs.vllm.ai/en/latest/[/dim]")


@setup_app.command("ollama")
def setup_ollama() -> None:
    """Set up Ollama for local model serving."""
    console.print("[bold]Ollama Setup[/bold]\n")

    has_ollama = shutil.which("ollama")
    if has_ollama:
        console.print("[green]Ollama found[/green]\n")
    else:
        console.print("[yellow]Ollama not found.[/yellow]")
        console.print("Install: [cyan]curl -fsSL "
                      "https://ollama.com/install.sh | sh[/cyan]\n")

    console.print("[bold]Quick start:[/bold]")
    console.print()
    console.print(Panel(
        "# Pull and run a model\n"
        "ollama pull nemotron-nano\n"
        "ollama serve  # starts on port 11434\n"
        "\n"
        "# Configure NeMoCode:\n"
        "nemo endpoint add local-ollama \\\n"
        "  --base-url http://localhost:11434/v1 \\\n"
        "  --model-id nemotron-nano \\\n"
        "  --tier local-ollama\n"
        "\n"
        "# Test it:\n"
        "nemo code -e local-ollama",
        title="[bold]Shell Commands[/bold]",
        border_style="green",
    ))

    console.print("\n[dim]Ollama serves an OpenAI-compatible API at /v1. "
                  "Tool calling support varies by model.[/dim]")
    console.print("[dim]Docs: https://ollama.com[/dim]")


def _check_docker() -> None:
    """Check if Docker is available."""
    if shutil.which("docker"):
        console.print("[green]Docker found[/green]")
        if shutil.which("nvidia-smi"):
            console.print("[green]NVIDIA GPU detected[/green]\n")
        else:
            console.print("[yellow]No NVIDIA GPU detected — "
                          "NIM requires NVIDIA GPUs[/yellow]\n")
    else:
        console.print("[yellow]Docker not found. Install Docker first:[/yellow]")
        console.print("[dim]https://docs.docker.com/get-docker/[/dim]\n")
