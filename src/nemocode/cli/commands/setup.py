# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo setup — guided setup for local inference backends."""

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
        console.print(
            Panel(
                "[bold]NeMoCode Local Inference Setup[/bold]\n\n"
                "Set up a local model server for offline/private coding.\n"
                "NeMoCode works with any OpenAI-compatible endpoint.",
                border_style="bright_green",
                expand=False,
            )
        )
        console.print()

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Backend", style="cyan bold")
        table.add_column("Command", style="dim")
        table.add_column("Best For")

        table.add_row(
            "DGX Spark",
            "nemo setup spark",
            "Full-stack local AI (Super + Nano + VLM + RAG)",
        )
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
            "SGLang",
            "nemo setup sglang",
            "Best local path for Nemotron 3 Super long context on Spark",
        )
        table.add_row(
            "Brev (Cloud GPU)",
            "nemo setup brev",
            "Rent an NVIDIA GPU (L40S/A100/H100) in minutes",
        )

        console.print(table)
        console.print()
        console.print(
            "[dim]All backends serve an OpenAI-compatible API that "
            "NeMoCode connects to automatically.[/dim]"
        )


@setup_app.command("spark")
def setup_spark() -> None:
    """Set up NeMoCode on DGX Spark — the complete local AI workstation."""
    console.print("[bold]DGX Spark Setup — Personal AI Supercomputer[/bold]\n")

    # Check for Docker and GPU
    _check_docker()

    console.print(
        Panel(
            "[bold]DGX Spark Specs[/bold]\n"
            "  GPU: GB10 Grace Blackwell Superchip\n"
            "  Memory: 128GB unified LPDDR5x (shared CPU/GPU)\n"
            "  Compute: 1 PFLOP FP4 / 1000 TOPS\n"
            "  CUDA: SM 12.1 (Blackwell desktop)\n\n"
            "  Can run Super (120B) + Nano (9B) + Embed + Rerank concurrently!",
            border_style="bright_green",
            expand=False,
        )
    )

    # Step 1: NGC API key
    console.print("\n[bold]Step 1: NGC API Key[/bold]")
    console.print(
        Panel(
            "# Get your NGC API key from:\n"
            "# https://org.ngc.nvidia.com/setup/api-key\n"
            "\n"
            "export NGC_API_KEY='your-ngc-key'\n"
            "\n"
            "# Also set it for NeMoCode API catalog access:\n"
            "export NVIDIA_API_KEY='your-nim-api-key'\n"
            "\n"
            "# To persist, add both to ~/.bashrc or ~/.zshrc",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    # Step 2: Start Super (main brain)
    console.print("\n[bold]Step 2: Start Nemotron 3 Super (main brain, port 8000)[/bold]")
    console.print(
        Panel(
            "docker run -d --name nemo-super \\\n"
            "  --runtime=nvidia \\\n"
            "  --gpus all \\\n"
            "  --shm-size=16GB \\\n"
            "  -p 8000:8000 \\\n"
            "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
            "  -v ~/.cache/nim:/opt/nim/llm/models \\\n"
            "  nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:latest\n"
            "\n"
            "# Wait for it to be ready (~2-5 min first time):\n"
            "watch -n2 'curl -s localhost:8000/v1/models | head -5'",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    # Step 3: Start Nano 9B (mini-agent worker)
    console.print(
        "\n[bold]Step 3: Start Nemotron Nano 9B (mini-agent worker, port 8002)[/bold]"
    )
    console.print(
        Panel(
            "docker run -d --name nemo-nano9b \\\n"
            "  --runtime=nvidia \\\n"
            "  --gpus all \\\n"
            "  --shm-size=4GB \\\n"
            "  -p 8002:8000 \\\n"
            "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
            "  -v ~/.cache/nim:/opt/nim/llm/models \\\n"
            "  nvcr.io/nim/nvidia/nemotron-nano-9b-v2:latest",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    # Step 4: Start retrieval models (optional)
    console.print(
        "\n[bold]Step 4 (optional): Start Embed + Rerank for RAG[/bold]"
    )
    console.print(
        Panel(
            "# Embedding model (port 8004)\n"
            "docker run -d --name nemo-embed \\\n"
            "  --runtime=nvidia \\\n"
            "  --gpus all \\\n"
            "  -p 8004:8000 \\\n"
            "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
            "  nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest\n"
            "\n"
            "# Reranking model (port 8005)\n"
            "docker run -d --name nemo-rerank \\\n"
            "  --runtime=nvidia \\\n"
            "  --gpus all \\\n"
            "  -p 8005:8000 \\\n"
            "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
            "  nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:latest",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    # Step 5: Configure NeMoCode
    console.print("\n[bold]Step 5: Use the Spark formation[/bold]")
    console.print(
        Panel(
            "# Daily driver — Super brain + Nano mini-agents, all local:\n"
            "nemo code -f spark\n"
            "\n"
            "# Full stack — VLM + RAG + coding:\n"
            "nemo code -f spark-full\n"
            "\n"
            "# Vision — read screenshots/mockups:\n"
            "nemo code -f spark-vision\n"
            "\n"
            "# Verify endpoints:\n"
            "nemo endpoint test spark-nim-super\n"
            "nemo endpoint test spark-nim-nano9b",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    # Memory usage table
    console.print("\n[bold]Memory Budget (128GB unified):[/bold]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Container", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Memory", justify="right")
    table.add_column("Port", justify="right")

    table.add_row("nemo-super", "Nemotron 3 Super 120B", "~80 GB", "8000")
    table.add_row("nemo-nano9b", "Nemotron Nano 9B v2", "~10 GB", "8002")
    table.add_row("nemo-embed", "Nemotron Embed 1B v2", "~2 GB", "8004")
    table.add_row("nemo-rerank", "Nemotron Rerank 1B v2", "~2 GB", "8005")
    table.add_row("[dim]Total[/dim]", "", "[bold]~94 GB[/bold]", "")
    table.add_row("[dim]Remaining[/dim]", "[dim]OS + apps[/dim]", "[green]~34 GB[/green]", "")

    console.print(table)

    console.print(
        "\n[dim]Prefer Docker-free local serving? Run"
        " [bold]nemo setup sglang[/bold] or [bold]nemo setup vllm[/bold].[/dim]"
    )
    console.print(
        "\n[dim]Docs: https://docs.nvidia.com/dgx/dgx-spark/[/dim]"
    )
    console.print(
        "[dim]NIM on Spark: https://build.nvidia.com/spark/nim-llm[/dim]"
    )
    console.print(
        "[dim]Playbooks: https://github.com/NVIDIA/dgx-spark-playbooks[/dim]"
    )


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
    console.print(
        Panel(
            "# Pull and run Nemotron 3 Nano via NIM\n"
            "export NGC_API_KEY='your-ngc-key'\n"
            "\n"
            "docker run -d --name nemotron-nano \\\n"
            "  --runtime=nvidia \\\n"
            "  --gpus all \\\n"
            "  -p 8000:8000 \\\n"
            "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
            "  -v ~/.cache/nim:/opt/nim/llm/models \\\n"
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
        )
    )

    console.print("\n[bold]Nemotron 3 Super (needs 80GB+ VRAM):[/bold]")
    console.print(
        Panel(
            "docker run -d --name nemotron-super \\\n"
            "  --runtime=nvidia \\\n"
            "  --gpus all \\\n"
            "  --shm-size=16GB \\\n"
            "  -p 8000:8000 \\\n"
            "  -e NGC_API_KEY=$NGC_API_KEY \\\n"
            "  -v ~/.cache/nim:/opt/nim/llm/models \\\n"
            "  nvcr.io/nim/nvidia/"
            "nemotron-3-super-120b-a12b:latest",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    # NIM catalog overview
    console.print("\n[bold]Full NIM Catalog (all available via API):[/bold]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Category", style="cyan")
    table.add_column("Models")
    table.add_column("Use Case", style="dim")

    table.add_row(
        "Coding",
        "Super 120B, Nano 30B, Nano 9B",
        "Primary agentic coding",
    )
    table.add_row(
        "Vision",
        "Nano 12B VL, Nano VL 8B",
        "Screenshot/diagram understanding",
    )
    table.add_row(
        "Reasoning",
        "Llama Ultra 253B, Reasoning 49B/8B",
        "Hard architectural problems",
    )
    table.add_row(
        "Retrieval",
        "Embed 1B, Rerank 1B",
        "RAG-augmented coding",
    )
    table.add_row(
        "Speech",
        "Parakeet ASR, FastPitch TTS",
        "Voice mode",
    )
    table.add_row(
        "Safety",
        "NeMo Guardrails",
        "Content moderation",
    )
    console.print(table)

    console.print(
        "\n[dim]Docs: https://docs.nvidia.com/nim/large-language-models/"
        "latest/getting-started.html[/dim]"
    )


@setup_app.command("vllm")
def setup_vllm() -> None:
    """Set up vLLM for local model serving."""
    console.print("[bold]vLLM Setup for Nemotron 3[/bold]\n")

    has_pip = shutil.which("pip") or shutil.which("pip3")
    if has_pip:
        console.print("[green]pip found[/green]\n")
    else:
        console.print("[yellow]pip not found — install Python first[/yellow]\n")

    # DGX Spark section first — it's the best experience
    console.print("[bold bright_green]DGX Spark — vLLM (Recommended)[/bold bright_green]\n")
    console.print(
        "DGX Spark's 128GB unified memory runs Super + Nano concurrently\n"
        "with [bold]no tensor parallel[/bold] and [bold]no Docker required[/bold].\n"
    )
    console.print(
        Panel(
            "# Install vLLM\n"
            "pip install vllm\n"
            "\n"
            "# ── Terminal 1: Nemotron 3 Super NVFP4 (main brain, port 8000) ──\n"
            "# Use local path if downloaded, or HF hub ID to pull on first run.\n"
            "vllm serve /home/hmbown/HF_Models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\\n"
            "  --port 8000 \\\n"
            "  --max-model-len 131072 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser hermes \\\n"
            "  --trust-remote-code\n"
            "\n"
            "# ── Terminal 2: Nemotron Nano 9B (mini-agent, port 8001) ──\n"
            "vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\\n"
            "  --port 8001 \\\n"
            "  --max-model-len 32768 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser hermes \\\n"
            "  --trust-remote-code",
            title="[bold]DGX Spark — Shell Commands[/bold]",
            border_style="bright_green",
        )
    )

    # Memory budget for Spark + vLLM
    console.print("[bold]Memory Budget (128GB unified):[/bold]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Process", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Memory", justify="right")
    table.add_column("Port", justify="right")

    table.add_row("vLLM (Super)", "Nemotron 3 Super 120B", "~80 GB", "8000")
    table.add_row("vLLM (Nano)", "Nemotron Nano 9B v2", "~10 GB", "8001")
    table.add_row("[dim]Total[/dim]", "", "[bold]~90 GB[/bold]", "")
    table.add_row("[dim]Remaining[/dim]", "[dim]OS + apps[/dim]", "[green]~38 GB[/green]", "")

    console.print(table)

    console.print(
        Panel(
            "# Use the spark-vllm formation:\n"
            "nemo code -f spark-vllm\n"
            "\n"
            "# Or use Super directly:\n"
            "nemo code -e spark-vllm-super\n"
            "\n"
            "# Verify endpoints:\n"
            "curl -s localhost:8000/v1/models | python3 -m json.tool\n"
            "curl -s localhost:8001/v1/models | python3 -m json.tool",
            title="[bold]Use with NeMoCode[/bold]",
            border_style="bright_green",
        )
    )

    # Generic workstation section
    console.print("\n[bold]Other Workstations[/bold]\n")
    console.print(
        Panel(
            "# Install vLLM\n"
            "pip install vllm\n"
            "\n"
            "# Serve Nemotron 3 Nano (FP8, ~24GB VRAM)\n"
            "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\\n"
            "  --port 8000 \\\n"
            "  --max-model-len 32768 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser hermes\n"
            "\n"
            "# Test it:\n"
            "nemo code -e local-vllm-nano",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print("\n[bold]For Nemotron 3 Super (FP8, ~80GB VRAM, multi-GPU):[/bold]")
    console.print(
        Panel(
            "vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \\\n"
            "  --port 8000 \\\n"
            "  --tensor-parallel-size 2 \\\n"
            "  --max-model-len 131072 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser hermes",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print(
        "\n[dim]vLLM supports tool calling via --enable-auto-tool-choice. "
        "Use --tool-call-parser hermes for Nemotron models.[/dim]"
    )
    console.print("[dim]Docs: https://docs.vllm.ai/en/latest/[/dim]")


@setup_app.command("sglang")
def setup_sglang() -> None:
    """Set up SGLang for local model serving."""
    console.print("[bold]SGLang Setup for Nemotron 3[/bold]\n")

    has_pip = shutil.which("pip") or shutil.which("pip3")
    if has_pip:
        console.print("[green]pip found[/green]\n")
    else:
        console.print("[yellow]pip not found — install Python first[/yellow]\n")

    console.print("[bold bright_green]DGX Spark — SGLang (Recommended for Super)[/bold bright_green]\n")
    console.print(
        "Use a dedicated server venv so NeMoCode's own environment stays stable.\n"
        "SGLang is the cleanest local path here for Nemotron 3 Super with long-context override.\n"
    )
    console.print(
        Panel(
            "# Create an isolated runtime venv for the SGLang server\n"
            "python3 -m venv .venv-sglang\n"
            ". .venv-sglang/bin/activate\n"
            "\n"
            "# Install the Spark-tested SGLang runtime stack\n"
            "pip install -U pip setuptools wheel\n"
            "pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'\n"
            "pip install --upgrade --force-reinstall \\\n"
            "  --index-url https://download.pytorch.org/whl/cu130 \\\n"
            "  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1\n"
            "pip install --force-reinstall \\\n"
            "  'https://github.com/sgl-project/whl/releases/download/v0.4.0/sglang_kernel-0.4.0%2Bcu130-cp310-abi3-manylinux2014_aarch64.whl'\n"
            "\n"
            "# Launch Nemotron 3 Super on port 8000 with the Spark helper\n"
            "# The helper enables the 1M override, routes Triton to CUDA 13 ptxas,\n"
            "# and constrains flashinfer JIT to avoid OOM during first compile.\n"
            "./scripts/spark_sglang_super.sh",
            title="[bold]DGX Spark — Shell Commands[/bold]",
            border_style="bright_green",
        )
    )

    console.print("[bold]Use with NeMoCode:[/bold]")
    console.print(
        Panel(
            "# Use the SGLang-backed endpoint directly:\n"
            "nemo chat -e spark-sglang-super\n"
            "\n"
            "# Or use the Spark SGLang formation:\n"
            "nemo code -f spark-sglang\n"
            "\n"
            "# Verify the server:\n"
            "curl -s localhost:8000/v1/models | python3 -m json.tool\n"
            "nemo endpoint test spark-sglang-super",
            title="[bold]Use with NeMoCode[/bold]",
            border_style="bright_green",
        )
    )

    console.print(
        "\n[dim]If 1M context is unstable on first launch, lower"
        " --mem-fraction-static slightly before lowering context length.[/dim]"
    )
    console.print("[dim]Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4[/dim]")
    console.print("[dim]Docs: https://docs.sglang.io/[/dim]")


@setup_app.command("brev")
def setup_brev() -> None:
    """Rent an NVIDIA GPU via Brev for cloud inference."""
    console.print("[bold]NVIDIA Brev — Cloud GPU Instances[/bold]\n")

    console.print(
        "Brev provides preconfigured GPU instances with NVIDIA drivers,\n"
        "CUDA, Python, and Docker — ready in minutes.\n"
    )

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("GPU", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("From", justify="right")
    table.add_column("Good For")

    table.add_row("L40S", "48 GB", "$1.03/hr", "Nano 9B, Nano 30B")
    table.add_row("A100", "80 GB", "$2.21/hr", "Super 120B")
    table.add_row("H100", "80 GB", "$3.19/hr", "Super 120B (fastest)")
    table.add_row("2x L40S", "96 GB", "$2.06/hr", "Super with tensor parallel")

    console.print(table)

    console.print("\n[bold]Quick start:[/bold]")
    console.print()
    console.print(
        Panel(
            "# 1. Sign up at console.brev.dev\n"
            "# 2. Install the CLI\n"
            "pip install brev-cli\n"
            "\n"
            "# 3. Launch a GPU instance\n"
            "brev create my-nemotron --gpu L40S\n"
            "\n"
            "# 4. SSH in and start vLLM\n"
            "brev shell my-nemotron\n"
            "pip install vllm nemocode\n"
            "vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\\n"
            "  --trust-remote-code \\\n"
            "  --mamba_ssm_cache_dtype float32 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-parser-plugin nemotron_toolcall_parser.py \\\n"
            "  --tool-call-parser nemotron_json &\n"
            "\n"
            "# 5. Use NeMoCode with your local endpoint\n"
            "nemo code -e local-vllm-nano9b",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print("\n[dim]Brev: https://console.brev.dev[/dim]")
    console.print("[dim]Docs: https://docs.brev.dev[/dim]")


def _check_docker() -> None:
    """Check if Docker is available."""
    if shutil.which("docker"):
        console.print("[green]Docker found[/green]")
        if shutil.which("nvidia-smi"):
            console.print("[green]NVIDIA GPU detected[/green]\n")
        else:
            console.print("[yellow]No NVIDIA GPU detected — NIM requires NVIDIA GPUs[/yellow]\n")
    else:
        console.print("[yellow]Docker not found. Install Docker first:[/yellow]")
        console.print("[dim]https://docs.docker.com/get-docker/[/dim]\n")
