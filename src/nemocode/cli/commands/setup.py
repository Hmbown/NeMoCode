# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo setup — guided setup for local inference backends and data services."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.core.setup_wizard import run_setup_wizard

console = Console()
setup_app = typer.Typer(help="Set up local inference backends and NVIDIA data services.")


@setup_app.callback(invoke_without_command=True)
def setup_default(
    ctx: typer.Context,
    guided: bool = typer.Option(
        True,
        "--guided/--list",
        help="Run the guided setup wizard instead of only listing setup topics.",
    ),
) -> None:
    """Run guided setup or show available setup options."""
    if ctx.invoked_subcommand is None:
        if guided and sys.stdin.isatty():
            run_setup_wizard()
            return

        console.print(
            Panel(
                "[bold]NeMoCode Local Inference Setup[/bold]\n\n"
                "Set up local model servers and NVIDIA data workflow services.\n"
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
            "TensorRT-LLM",
            "nemo setup trt-llm",
            "Spark-tuned TensorRT serving with OpenAI-compatible APIs",
        )
        table.add_row(
            "llama.cpp",
            "nemo setup llama-cpp",
            "Official Nano 4B GGUF Q4_K_M local serving",
        )
        table.add_row(
            "Brev (Cloud GPU)",
            "nemo setup brev",
            "Rent an NVIDIA GPU (L40S/A100/H100) in minutes",
        )
        table.add_row(
            "Data Workflows",
            "nemo setup data",
            "Data Designer, Evaluator, Safe Synthesizer, Curator",
        )

        console.print(table)
        console.print()
        console.print(
            "[dim]All backends serve an OpenAI-compatible API that "
            "NeMoCode connects to automatically.[/dim]"
        )
        if guided and not sys.stdin.isatty():
            console.print(
                "[dim]Interactive wizard needs a TTY. Run [bold]nemo setup wizard[/bold]"
                " in a terminal, or use one of the subcommands below.[/dim]"
            )


@setup_app.command("wizard")
def setup_wizard() -> None:
    """Run the guided runtime setup wizard."""
    run_setup_wizard()


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
            "  Can run Super (120B) + Nano + Embed + Rerank concurrently!",
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
    console.print("\n[bold]Step 3: Start Nemotron Nano 9B (mini-agent worker, port 8002)[/bold]")
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
    console.print("\n[bold]Step 4 (optional): Start Embed + Rerank for RAG[/bold]")
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
        Panel(
            "[bold]Alternative Spark Fast Workers[/bold]\n"
            "  vLLM NVFP4/FP8: [cyan]nemo setup vllm[/cyan] → Super NVFP4, Nano 4B FP8\n"
            "  TRT-LLM NVFP4/FP8: [cyan]nemo setup trt-llm[/cyan] → native NVIDIA optimized serving\n"
            "  llama.cpp GGUF: [cyan]nemo setup llama-cpp[/cyan] → official Q4_K_M 4-bit server",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print(
        Panel(
            "[bold]Spark-specific Notes[/bold]\n"
            "  Treat Spark as UMA ARM64 hardware, not a discrete-GPU workstation.\n"
            "  `cudaMemGetInfo` can under-report allocatable memory on Spark.\n"
            "  GPUDirect RDMA is not supported on DGX Spark.\n"
            "  Prefer Spark/aarch64-tested wheels and containers for local backends.",
            border_style="yellow",
            expand=False,
        )
    )

    console.print(
        "\n[dim]Prefer an alternate local backend? Run"
        " [bold]nemo setup sglang[/bold], [bold]nemo setup trt-llm[/bold],"
        " [bold]nemo setup vllm[/bold], or [bold]nemo setup llama-cpp[/bold].[/dim]"
    )
    console.print("\n[dim]Docs: https://docs.nvidia.com/dgx/dgx-spark/[/dim]")
    console.print("[dim]NIM on Spark: https://build.nvidia.com/spark/nim-llm[/dim]")
    console.print("[dim]Playbooks: https://github.com/NVIDIA/dgx-spark-playbooks[/dim]")


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
        "with [bold]no tensor parallel[/bold]. For Nano 4B on Spark, follow the\n"
        "official FP8 vLLM path rather than generic workstation flags.\n"
    )
    console.print(
        Panel(
            "# Install the Spark-tested vLLM runtime (official Nano 4B card requires vLLM >= 0.15.1)\n"
            "pip install 'vllm>=0.15.1'\n"
            "\n"
            "# If NVIDIA publishes nano_v3_reasoning_parser.py separately, place it in this directory.\n"
            "# The Nano 4B FP8 Spark launch below will pick it up automatically when present.\n"
            "\n"
            "# ── Terminal 1: Nemotron 3 Super NVFP4 (main brain, port 8000) ──\n"
            "# NVIDIA's current Spark recipe uses the vllm/vllm-openai:v0.17.1-cu130 image.\n"
            "wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py\n"
            "export MODEL_CKPT=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4\n"
            "VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\\n"
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \\\n"
            "vllm serve $MODEL_CKPT \\\n"
            "  --served-model-name nemotron-3-super \\\n"
            "  --host 0.0.0.0 \\\n"
            "  --port 8000 \\\n"
            "  --async-scheduling \\\n"
            "  --dtype auto \\\n"
            "  --kv-cache-dtype fp8 \\\n"
            "  --tensor-parallel-size 1 \\\n"
            "  --pipeline-parallel-size 1 \\\n"
            "  --data-parallel-size 1 \\\n"
            "  --trust-remote-code \\\n"
            "  --gpu-memory-utilization 0.90 \\\n"
            "  --enable-chunked-prefill \\\n"
            "  --max-num-seqs 4 \\\n"
            "  --max-model-len 394000 \\\n"
            "  --attention-backend TRITON_ATTN \\\n"
            "  --mamba_ssm_cache_dtype float32 \\\n"
            "  --moe-backend marlin\n"
            "\n"
            "# ── Terminal 2: Nemotron 3 Nano 4B FP8 (fast worker, port 8001) ──\n"
            "# If nano_v3_reasoning_parser.py is unavailable, omit the two reasoning-parser flags below.\n"
            "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \\\n"
            "  --port 8001 \\\n"
            "  --max-model-len 262144 \\\n"
            "  --mamba_ssm_cache_dtype float32 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser qwen3_coder \\\n"
            "  --reasoning-parser nano_v3 \\\n"
            "  --reasoning-parser-plugin ./nano_v3_reasoning_parser.py \\\n"
            "  --kv-cache-dtype fp8 \\\n"
            "  --trust-remote-code",
            title="[bold]DGX Spark — Shell Commands[/bold]",
            border_style="bright_green",
        )
    )

    # Memory budget for Spark + vLLM
    console.print("[bold]Deployment Shape (128GB unified):[/bold]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Process", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Notes")
    table.add_column("Port", justify="right")

    table.add_row("vLLM (Super)", "Nemotron 3 Super 120B NVFP4", "Planner / executor", "8000")
    table.add_row("vLLM (Nano)", "Nemotron 3 Nano 4B FP8", "Fast worker, 262K ctx", "8001")

    console.print(table)

    console.print(
        Panel(
            "# Use the spark-vllm formation:\n"
            "nemo code -f spark-vllm\n"
            "\n"
            "# Or target the Spark Nano 4B worker directly:\n"
            "nemo code -e spark-vllm-nano4b\n"
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
            "# Serve Nemotron 3 Nano 30B NVFP4\n"
            "VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=throughput \\\n"
            "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \\\n"
            "  --port 8000 \\\n"
            "  --max-model-len 262144 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser qwen3_coder \\\n"
            "  --kv-cache-dtype fp8\n"
            "\n"
            "# Test it:\n"
            "nemo code -e local-vllm-nano",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print("\n[bold]For Nemotron 3 Super NVFP4 (~80GB VRAM, multi-GPU):[/bold]")
    console.print(
        Panel(
            "wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py\n"
            "export MODEL_CKPT=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4\n"
            "vllm serve $MODEL_CKPT \\\n"
            "  --served-model-name nvidia/nemotron-3-super \\\n"
            "  --port 8000 \\\n"
            "  --async-scheduling \\\n"
            "  --dtype auto \\\n"
            "  --max-model-len 262144 \\\n"
            "  --swap-space 0 \\\n"
            "  --trust-remote-code \\\n"
            "  --gpu-memory-utilization 0.9 \\\n"
            "  --max-cudagraph-capture-size 128 \\\n"
            "  --enable-chunked-prefill \\\n"
            "  --mamba-ssm-cache-dtype float16 \\\n"
            "  --reasoning-parser nemotron_v3 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser qwen3_coder",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print(
        "\n[dim]vLLM supports tool calling via --enable-auto-tool-choice. "
        "On Spark Nano 4B, use the model-card settings: 262144 context, "
        "float32 Mamba cache, qwen3_coder tool parser, and FP8 KV cache.[/dim]"
    )
    console.print(
        "[dim]If a Nemotron checkpoint does not have an official NVFP4 artifact yet, "
        "quantize the BF16/FP8 checkpoint to a unified Hugging Face export with TensorRT "
        "Model Optimizer or LLM Compressor, then deploy that export on vLLM or TRT-LLM.[/dim]"
    )
    console.print(
        "[dim]Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8[/dim]"
    )
    console.print(
        "[dim]Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4[/dim]"
    )
    console.print(
        "[dim]Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4[/dim]"
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

    console.print(
        "[bold bright_green]DGX Spark — SGLang (Recommended for Super)[/bold bright_green]\n"
    )
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
    console.print(
        "[dim]Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4[/dim]"
    )
    console.print("[dim]Docs: https://docs.sglang.io/[/dim]")


@setup_app.command("trt-llm")
def setup_trt_llm() -> None:
    """Set up TensorRT-LLM for local model serving."""
    console.print("[bold]TensorRT-LLM Setup for NeMoCode[/bold]\n")

    _check_docker()

    console.print("[bold bright_green]DGX Spark — TensorRT-LLM[/bold bright_green]\n")
    console.print(
        "TensorRT-LLM is a strong Spark fit when you want NVIDIA's native optimized serving\n"
        "stack with an OpenAI-compatible API. The current Spark playbook uses Docker + HF cache\n"
        "mounts. NeMoCode now defaults this backend to Nemotron 3 Super 120B + Nano 4B.\n"
    )
    console.print(
        Panel(
            "# Prereqs\n"
            "export HF_TOKEN=<your-huggingface-token>\n"
            "export DOCKER_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6\n"
            "mkdir -p $HOME/.cache/huggingface/\n"
            "\n"
            "# Terminal 1: Nemotron 3 Super 120B (planner/reviewer, port 8000)\n"
            "docker run --rm -it --gpus all --ipc host --network host \\\n"
            "  -e HF_TOKEN=$HF_TOKEN \\\n"
            "  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \\\n"
            "  $DOCKER_IMAGE \\\n"
            "  trtllm-serve nvidia/nemotron-3-super-120b-a12b \\\n"
            "  --trust_remote_code --port 8000\n"
            "\n"
            "# Terminal 2: Nemotron 3 Nano 4B FP8 (fast worker, port 8001)\n"
            "docker run --rm -it --gpus all --ipc host --network host \\\n"
            "  -e HF_TOKEN=$HF_TOKEN \\\n"
            "  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \\\n"
            "  $DOCKER_IMAGE \\\n"
            "  trtllm-serve nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \\\n"
            "  --trust_remote_code --port 8001\n"
            "\n"
            "# If Super 120B OOMs or stalls during weight load on Spark:\n"
            "export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1",
            title="[bold]DGX Spark — Shell Commands[/bold]",
            border_style="bright_green",
        )
    )

    console.print("[bold]Use with NeMoCode:[/bold]")
    console.print(
        Panel(
            "# Use the TensorRT-LLM-backed Spark formation:\n"
            "nemo code -f spark-trt-llm\n"
            "\n"
            "# Or target endpoints directly:\n"
            "nemo chat -e spark-trt-llm-super\n"
            "nemo code -e spark-trt-llm-nano4b\n"
            "\n"
            "# Verify the servers:\n"
            "curl -s localhost:8000/v1/models | python3 -m json.tool\n"
            "curl -s localhost:8001/v1/models | python3 -m json.tool\n"
            "nemo endpoint test spark-trt-llm-super\n"
            "nemo endpoint test spark-trt-llm-nano4b",
            title="[bold]Use with NeMoCode[/bold]",
            border_style="bright_green",
        )
    )

    console.print("\n[bold]Other Workstations[/bold]\n")
    console.print(
        Panel(
            "# Single-model Nemotron 3 Nano 4B FP8 on port 8000\n"
            "export HF_TOKEN=<your-huggingface-token>\n"
            "export DOCKER_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6\n"
            "mkdir -p $HOME/.cache/huggingface/\n"
            "\n"
            "docker run --rm -it --gpus all --ipc host --network host \\\n"
            "  -e HF_TOKEN=$HF_TOKEN \\\n"
            "  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \\\n"
            "  $DOCKER_IMAGE \\\n"
            "  trtllm-serve nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \\\n"
            "  --trust_remote_code --port 8000\n"
            "\n"
            "nemo code -e local-trt-llm-nano4b",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print(
        "\n[dim]TensorRT-LLM serves OpenAI-compatible chat/completions endpoints at"
        " /v1 via trtllm-serve.[/dim]"
    )
    console.print(
        "[dim]Exact TensorRT-LLM launch flags may vary by container release and whether"
        " the model needs a prebuilt engine. NVIDIA's official March 16, 2026"
        " Nemotron model cards now include NVFP4-ready Super / Nano 30B artifacts and"
        " FP8 Nano 4B examples; these repo defaults now prefer NVFP4 whenever NVIDIA"
        " ships an official checkpoint.[/dim]"
    )
    console.print("[dim]Spark playbook: https://build.nvidia.com/spark/trt-llm/instructions[/dim]")
    console.print(
        "[dim]CLI docs: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html[/dim]"
    )


@setup_app.command("llama-cpp")
def setup_llama_cpp() -> None:
    """Set up llama.cpp for official Nano 4B GGUF serving."""
    console.print("[bold]llama.cpp Setup for Nemotron 3 Nano 4B[/bold]\n")

    llama_server = shutil.which("llama-server")
    if llama_server:
        console.print(f"[green]llama-server found:[/green] {llama_server}\n")
    else:
        console.print(
            "[yellow]llama-server not found.[/yellow] Install a current llama.cpp build "
            "with CUDA support, then re-run this command.\n"
        )

    console.print("[bold bright_green]DGX Spark — Official GGUF Q4_K_M[/bold bright_green]\n")
    console.print(
        "Use the official Nano 4B GGUF Q4_K_M artifact for the lightest Spark fast-worker path. "
        "Keep BF16 or FP8 for training/customization; use GGUF for local serving.\n"
    )
    console.print(
        Panel(
            "# Terminal 1: Super planner / executor (reuse the official Spark vLLM path)\n"
            "nemo setup vllm\n"
            "\n"
            "# Terminal 2: Nano 4B GGUF Q4_K_M OpenAI-compatible server on port 8001\n"
            "llama-server -hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M \\\n"
            "  -c 0 --alias nemotron3-nano-4b-q4km --ngl 999 \\\n"
            "  --host 0.0.0.0 --port 8001",
            title="[bold]DGX Spark — Shell Commands[/bold]",
            border_style="bright_green",
        )
    )

    console.print("[bold]Use with NeMoCode:[/bold]")
    console.print(
        Panel(
            "# Hybrid Spark formation: Super on vLLM + Nano 4B Q4_K_M on llama.cpp\n"
            "nemo code -f spark-llama-cpp\n"
            "\n"
            "# Or target the Nano worker directly:\n"
            "nemo code -e spark-llama-cpp-nano4b\n"
            "\n"
            "# Verify the server:\n"
            "curl -s localhost:8001/v1/models | python3 -m json.tool",
            title="[bold]Use with NeMoCode[/bold]",
            border_style="bright_green",
        )
    )

    console.print("\n[bold]Other Workstations[/bold]\n")
    console.print(
        Panel(
            "llama-server -hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M \\\n"
            "  -c 0 --alias nemotron3-nano-4b-q4km --ngl 999 \\\n"
            "  --host 0.0.0.0 --port 8000\n"
            "\n"
            "nemo code -e local-llama-cpp-nano4b",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print(
        "\n[dim]Quantized serving artifacts are deployment targets, not customization bases. "
        "For post-training optimization beyond GGUF, NVIDIA's quantization stack covers FP8, "
        "INT4, and NVFP4 / FP4 paths on Blackwell-class systems.[/dim]"
    )
    console.print(
        "[dim]Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF[/dim]"
    )
    console.print("[dim]llama.cpp: https://github.com/ggml-org/llama.cpp[/dim]")

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

    table.add_row("L40S", "48 GB", "$1.03/hr", "Nano 30B NVFP4")
    table.add_row("A100", "80 GB", "$2.21/hr", "Super 120B NVFP4")
    table.add_row("H100", "80 GB", "$3.19/hr", "Super 120B NVFP4 (fastest)")
    table.add_row("2x L40S", "96 GB", "$2.06/hr", "Super NVFP4 with tensor parallel")

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
            "VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=throughput \\\n"
            "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \\\n"
            "  --trust-remote-code \\\n"
            "  --max-model-len 262144 \\\n"
            "  --enable-auto-tool-choice \\\n"
            "  --tool-call-parser qwen3_coder \\\n"
            "  --kv-cache-dtype fp8 &\n"
            "\n"
            "# 5. Use NeMoCode with your local endpoint\n"
            "nemo code -e local-vllm-nano",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print("\n[dim]Brev: https://console.brev.dev[/dim]")
    console.print("[dim]Docs: https://docs.brev.dev[/dim]")


@setup_app.command("data")
def setup_data() -> None:
    """Set up NVIDIA synthetic-data and evaluation services."""
    console.print("[bold]NVIDIA Data Workflow Setup[/bold]\n")

    docker_ok = shutil.which("docker") is not None
    compose_ok = _docker_compose_available()
    ngc_ok = shutil.which("ngc") is not None
    microservices_sdk = importlib.util.find_spec("nemo_microservices") is not None
    curator_sdk = importlib.util.find_spec("nemo_curator") is not None
    gpu_name = _detect_gpu_name()
    ngc_key = bool(os.environ.get("NGC_CLI_API_KEY") or os.environ.get("NGC_API_KEY"))
    nim_key = bool(os.environ.get("NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY"))

    status = Table(show_header=True, box=None, padding=(0, 2))
    status.add_column("Check", style="cyan")
    status.add_column("Status")
    status.add_column("Details", style="dim")
    status.add_row("Docker", _status_label(docker_ok), "Required for all Docker Compose paths")
    status.add_row(
        "Docker Compose",
        _status_label(compose_ok),
        "Required for Data Designer, Evaluator, Safe Synthesizer",
    )
    status.add_row("NGC CLI", _status_label(ngc_ok), "Required by NVIDIA quickstart downloads")
    status.add_row(
        "NGC key",
        _status_label(ngc_key),
        "Needed for nvcr.io and NGC quickstart access",
    )
    status.add_row("build.nvidia.com key", _status_label(nim_key), "Needed for model-backed flows")
    status.add_row("GPU", gpu_name or "not detected", "Safe Synthesizer is the strictest path")
    status.add_row(
        "nemo-microservices SDK",
        _status_label(microservices_sdk),
        "Optional, but useful for programmatic Data Designer and Evaluator access",
    )
    status.add_row(
        "nemo-curator",
        _status_label(curator_sdk),
        "Optional for large-scale curation pipelines",
    )
    console.print(status)

    console.print()
    console.print(
        Panel(
            "Recommended NeMoCode path:\n"
            "1. Analyze the repo with [bold]nemo data analyze[/bold]\n"
            "2. Launch NeMo Data Designer for preview and job APIs\n"
            "3. Use NeMo Evaluator to score generated tasks\n"
            "4. Add Safe Synthesizer only when private tabular data is involved\n"
            "5. Add Curator when you scale beyond repo-local artifacts into large corpora",
            border_style="bright_green",
            title="[bold]Suggested Order[/bold]",
            expand=False,
        )
    )

    console.print("\n[bold]NeMo Data Designer[/bold]")
    console.print(
        Panel(
            "# NVIDIA docs: https://docs.nvidia.com/nemo/microservices/latest/"
            "design-synthetic-data-from-scratch-or-seeds/docker-compose.html\n"
            "export NGC_CLI_API_KEY=<your-ngc-api-key>\n"
            "echo $NGC_CLI_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin\n"
            "ngc registry resource download-version "
            "\"nvidia/nemo-microservices/nemo-microservices-quickstart:25.12\"\n"
            "cd nemo-microservices-quickstart_v25.12\n"
            "export NEMO_MICROSERVICES_IMAGE_REGISTRY=nvcr.io/nvidia/nemo-microservices\n"
            "export NEMO_MICROSERVICES_IMAGE_TAG=25.12\n"
            "export NIM_API_KEY=<build.nvidia.com-api-key>\n"
            "docker compose --profile data-designer up --detach --quiet-pull --wait\n"
            "\n"
            "# Verify:\n"
            "curl -X POST -H 'Content-type: application/json' "
            "localhost:8080/v1/data-designer/preview -d @preview.json\n"
            "\n"
            "# SDK:\n"
            "pip install \"nemo-microservices[data-designer]\"",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print("\n[bold]NeMo Evaluator[/bold]")
    console.print(
        Panel(
            "# NVIDIA docs: https://docs.nvidia.com/nemo/microservices/latest/evaluate/docker-compose.html\n"
            "export NGC_CLI_API_KEY=<your-ngc-api-key>\n"
            "docker login nvcr.io -u '$oauthtoken' -p $NGC_CLI_API_KEY\n"
            "ngc registry resource download-version "
            "\"nvidia/nemo-microservices/nemo-microservices-quickstart:25.10\"\n"
            "cd nemo-microservices-quickstart_v25.10\n"
            "export NEMO_MICROSERVICES_IMAGE_REGISTRY=nvcr.io/nvidia/nemo-microservices\n"
            "export NEMO_MICROSERVICES_IMAGE_TAG=25.10\n"
            "docker compose --profile evaluator up --detach --quiet-pull --wait\n"
            "\n"
            "# Verify:\n"
            "curl -fv http://localhost:8080/v2/evaluation/jobs",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print("\n[bold]NeMo Safe Synthesizer[/bold]")
    console.print(
        Panel(
            "# NVIDIA docs: https://docs.nvidia.com/nemo/microservices/latest/"
            "generate-private-synthetic-data/docker-compose.html\n"
            "# Use this only when you have private tabular data to protect.\n"
            "export NGC_CLI_API_KEY=<your-ngc-api-key>\n"
            "echo $NGC_CLI_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin\n"
            "ngc registry resource download-version "
            "\"nvidia/nemo-microservices/nemo-microservices-quickstart:25.12\"\n"
            "cd nemo-microservices-quickstart_v25.12\n"
            "export NEMO_MICROSERVICES_IMAGE_REGISTRY=nvcr.io/nvidia/nemo-microservices\n"
            "export NEMO_MICROSERVICES_IMAGE_TAG=25.12\n"
            "export NIM_API_KEY=<build.nvidia.com-api-key>\n"
            "docker compose -f docker-compose.yaml -f gpu/config.yaml "
            "--profile safe-synthesizer up --detach --quiet-pull --wait\n"
            "\n"
            "# SDK:\n"
            "pip install \"nemo-microservices[safe-synthesizer]\"",
            title="[bold]Shell Commands[/bold]",
            border_style="yellow",
        )
    )

    console.print(
        "[dim]NVIDIA's Docker guide for Safe Synthesizer calls out 16GB RAM and an NVIDIA "
        "GPU with 80GB. Treat smaller or unlisted GPUs as experimental.[/dim]"
    )

    console.print("\n[bold]NeMo Curator[/bold]")
    console.print(
        Panel(
            "# NVIDIA docs: https://docs.nvidia.com/nemo/curator/latest/index.html\n"
            "pip install nemo-curator\n"
            "# or\n"
            "pip install 'nemo-curator[cuda12x]'",
            title="[bold]Shell Commands[/bold]",
            border_style="green",
        )
    )

    console.print(
        "\n[dim]Best first milestone: get Data Designer running, then use "
        "[bold]nemo data analyze[/bold] to produce a repo-aware plan and preview request.[/dim]"
    )


def _status_label(ok: bool) -> str:
    return "[green]ready[/green]" if ok else "[yellow]missing[/yellow]"


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _detect_gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


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
