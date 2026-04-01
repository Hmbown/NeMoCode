# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo serve — serve fine-tuned LoRA adapters locally."""

from __future__ import annotations

import json
import os  # noqa: F401 — used by test mocks (serve.os.killpg)
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from nemocode.config import _USER_CONFIG_DIR, _USER_CONFIG_PATH, ensure_config_dir, load_config
from nemocode.core.hardware import detect_hardware
from nemocode.core.validators import validate_file_path

console = Console()
serve_app = typer.Typer(help="Serve fine-tuned LoRA adapters locally.")

_SERVE_PID_FILE = _USER_CONFIG_DIR / ".serve.pid"
_SERVE_STATE_FILE = _USER_CONFIG_DIR / ".serve.json"
_DEFAULT_PORT = 8010
_HEALTH_CHECK_INTERVAL = 2
_HEALTH_CHECK_TIMEOUT = 300

_MODEL_BACKEND_MAP = {
    "nemotron-3-nano-4b": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nemotron-3-nano-4b-bf16": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nemotron-3-nano-4b-fp8": "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    "nemotron-3-nano-9b": "nvidia/nemotron-nano-9b-v2",
    "nemotron-3-nano-30b": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-3-super-nvfp4": "/home/hmbown/HF_Models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
}


def _resolve_model_id(model_alias: str) -> str:
    return _MODEL_BACKEND_MAP.get(model_alias, model_alias)


def _pick_backend(is_spark: bool) -> str:
    del is_spark  # auto-pick is based on adapter-serving support, not host family
    if _find_executable("vllm"):
        return "vllm"
    return "unknown"


def _find_executable(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _default_max_model_len(model_id: str) -> int:
    try:
        cfg = load_config()
        manifest = cfg.manifests.get(model_id)
        if manifest and manifest.context_window:
            return manifest.context_window
    except Exception:
        pass
    return 32768


def _default_reasoning_parser_plugin(model_id: str) -> Path | None:
    if "NVIDIA-Nemotron-3-Nano-4B" not in model_id:
        return None
    local_plugin = Path.cwd() / "nano_v3_reasoning_parser.py"
    if local_plugin.exists():
        return local_plugin
    return None


def _build_vllm_command(
    model_id: str,
    adapter_path: Path,
    port: int,
    max_model_len: int | None = None,
    reasoning_parser_plugin: Path | None = None,
) -> list[str]:
    resolved_max_model_len = max_model_len or _default_max_model_len(model_id)
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--port",
        str(port),
        "--max-model-len",
        str(resolved_max_model_len),
        "--enable-lora",
        "--lora-modules",
        f"my-lora={adapter_path}",
        "--trust-remote-code",
    ]
    if "NVIDIA-Nemotron-3-Nano-4B" in model_id:
        cmd.extend(
            [
                "--mamba_ssm_cache_dtype",
                "float32",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
            ]
        )
        if reasoning_parser_plugin is not None:
            cmd.extend(
                [
                    "--reasoning-parser-plugin",
                    str(reasoning_parser_plugin),
                    "--reasoning-parser",
                    "nano_v3",
                ]
            )
    if model_id.endswith("-FP8"):
        cmd.extend(["--kv-cache-dtype", "fp8"])
    return cmd


def _build_sglang_command(
    model_id: str,
    adapter_path: Path,
    port: int,
    max_model_len: int = 32768,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_id,
        "--port",
        str(port),
        "--max-running-requests",
        "128",
        "--host",
        "0.0.0.0",
    ]
    return cmd


def _health_check(base_url: str, timeout: int = _HEALTH_CHECK_TIMEOUT) -> bool:
    url = f"{base_url.rstrip('/')}/models"
    elapsed = 0
    console.print("[dim]Waiting for endpoint to become ready...[/dim]")
    with console.status("[bold green]Starting server...", spinner="dots"):
        while elapsed < timeout:
            try:
                resp = httpx.get(url, timeout=3)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(_HEALTH_CHECK_INTERVAL)
            elapsed += _HEALTH_CHECK_INTERVAL
    return False


def _write_state(state: dict) -> None:
    ensure_config_dir()
    _SERVE_STATE_FILE.write_text(json.dumps(state, indent=2))


def _read_state() -> dict | None:
    if not _SERVE_STATE_FILE.exists():
        return None
    try:
        return json.loads(_SERVE_STATE_FILE.read_text())
    except Exception:
        return None


def _clear_state() -> None:
    for p in (_SERVE_PID_FILE, _SERVE_STATE_FILE):
        if p.exists():
            p.unlink()


def _register_endpoint(
    name: str,
    base_url: str,
    model_id: str,
    tier: str,
) -> None:
    import yaml

    ensure_config_dir()
    existing = {}
    if _USER_CONFIG_PATH.exists():
        with open(_USER_CONFIG_PATH) as f:
            existing = yaml.safe_load(f) or {}

    if "endpoints" not in existing:
        existing["endpoints"] = {}

    existing["endpoints"][name] = {
        "name": f"LoRA Serve ({name})",
        "tier": tier,
        "base_url": base_url,
        "model_id": model_id,
        "capabilities": ["chat", "code"],
    }

    with open(_USER_CONFIG_PATH, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)

    console.print(f"[green]Registered endpoint '{name}' in config.[/green]")


@serve_app.command()
def start(
    model: str = typer.Option(
        "nemotron-3-nano-4b",
        "--model",
        "-m",
        help="Base model to serve.",
    ),
    adapter: str = typer.Option(
        ".nemocode/adapters/my-lora/",
        "--adapter",
        "-a",
        help="Path to LoRA adapter directory.",
    ),
    port: int = typer.Option(
        _DEFAULT_PORT,
        "--port",
        "-p",
        help="Port to serve on.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Endpoint name for NeMoCode config.",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Backend to use (vllm, sglang). Auto-detects if omitted.",
    ),
    max_model_len: Optional[int] = typer.Option(
        None,
        "--max-model-len",
        help="Maximum model context length (defaults to the model manifest context window).",
    ),
    reasoning_parser_plugin: Optional[str] = typer.Option(
        None,
        "--reasoning-parser-plugin",
        help="Path to nano_v3_reasoning_parser.py for Nemotron Nano 4B.",
    ),
) -> None:
    """Launch a local inference backend with a LoRA adapter."""
    try:
        validate_file_path(adapter)
    except ValueError as exc:
        console.print(f"[red]Invalid adapter path: {exc}[/red]")
        raise typer.Exit(1) from exc

    adapter_path = Path(adapter).resolve()
    if not adapter_path.exists():
        console.print(f"[red]Adapter path not found: {adapter_path}[/red]")
        raise typer.Exit(1)

    model_id = _resolve_model_id(model)
    profile = detect_hardware()

    if backend is None:
        backend = _pick_backend(profile.is_dgx_spark)

    if backend == "unknown":
        console.print(
            "[red]No supported backend found.[/red]\nInstall vLLM: [bold]pip install vllm[/bold]"
        )
        raise typer.Exit(1)

    if backend == "sglang":
        console.print(
            "[red]SGLang adapter serving is not wired up coherently here yet.[/red]\n"
            "Use [bold]--backend vllm[/bold] or let auto-detect choose vLLM so the LoRA adapter is actually loaded."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Model:[/bold]    {model_id}\n"
            f"[bold]Adapter:[/bold]  {adapter_path}\n"
            f"[bold]Backend:[/bold]  {backend}\n"
            f"[bold]Port:[/bold]     {port}\n"
            f"[bold]Ctx len:[/bold]  {max_model_len or _default_max_model_len(model_id)}",
            title="[bold green]nemo serve[/bold green]",
            border_style="green",
        )
    )

    parser_plugin_path = (
        Path(reasoning_parser_plugin).resolve()
        if reasoning_parser_plugin
        else _default_reasoning_parser_plugin(model_id)
    )
    if "NVIDIA-Nemotron-3-Nano-4B" in model_id and parser_plugin_path is None:
        console.print(
            "[dim]Nemotron Nano 4B reasoning parser not found locally. "
            "Tool calling will still work, but the dedicated nano_v3 parser is recommended when available.[/dim]"
        )

    if backend == "vllm":
        cmd = _build_vllm_command(
            model_id,
            adapter_path,
            port,
            max_model_len,
            parser_plugin_path,
        )
    elif backend == "sglang":
        cmd = _build_sglang_command(model_id, adapter_path, port, max_model_len)
    else:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Launching: {' '.join(cmd)}[/dim]\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except FileNotFoundError:
        console.print(f"[red]Backend executable not found: {cmd[0]}[/red]")
        raise typer.Exit(1)

    ensure_config_dir()
    _SERVE_PID_FILE.write_text(str(proc.pid))

    base_url = f"http://localhost:{port}/v1"
    endpoint_name = name or f"serve-lora-{port}"

    _write_state(
        {
            "pid": proc.pid,
            "model_id": model_id,
            "adapter_path": str(adapter_path),
            "backend": backend,
            "port": port,
            "base_url": base_url,
            "endpoint_name": endpoint_name,
        }
    )

    if not _health_check(base_url):
        console.print("[red]Server did not become ready in time.[/red]")
        proc.terminate()
        _clear_state()
        raise typer.Exit(1)

    tier = "local-sglang" if backend == "sglang" else "local-vllm"
    _register_endpoint(endpoint_name, base_url, model_id, tier)

    console.print()
    console.print(
        Panel(
            f"[bold green]Server is live![/bold green]\n\n"
            f"  Endpoint:  [cyan]{endpoint_name}[/cyan]\n"
            f"  URL:       {base_url}\n"
            f"  Model:     {model_id}\n"
            f"  Adapter:   {adapter_path}\n"
            f"  PID:       {proc.pid}\n\n"
            f"  [bold]Usage:[/bold]\n"
            f"    nemo chat -e {endpoint_name}\n"
            f"    nemo code -e {endpoint_name}\n"
            f"    nemo serve stop",
            title="[bold]Ready[/bold]",
            border_style="bright_green",
        )
    )

    console.print(
        "[dim]Server running in background. Use [bold]nemo serve stop[/bold] to shut down.[/dim]"
    )

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        _clear_state()
        console.print("\n[dim]Server stopped.[/dim]")


@serve_app.command("stop")
def serve_stop() -> None:
    """Stop the running serve process."""
    state = _read_state()
    if not state:
        console.print("[yellow]No running serve process found.[/yellow]")
        return

    pid = state.get("pid")
    endpoint_name = state.get("endpoint_name", "")
    base_url = state.get("base_url", "")

    if pid:
        try:
            import os

            os.killpg(os.getpgid(pid), signal.SIGTERM)
            console.print(f"[green]Sent SIGTERM to process group {pid}[/green]")
        except ProcessLookupError:
            console.print(f"[yellow]Process {pid} not found (already stopped).[/yellow]")
        except PermissionError:
            console.print(f"[red]No permission to stop process {pid}.[/red]")
            raise typer.Exit(1)

    _clear_state()

    if endpoint_name and base_url:
        console.print(f"[dim]Endpoint '{endpoint_name}' ({base_url}) removed from state.[/dim]")
        console.print(f"[dim]To remove from config: nemo endpoint remove {endpoint_name}[/dim]")

    console.print("[green]Serve process stopped.[/green]")


@serve_app.command("status")
def serve_status() -> None:
    """Show the status of the running serve process."""
    state = _read_state()
    if not state:
        console.print("[yellow]No serve process running.[/yellow]")
        return

    pid = state.get("pid")
    backend = state.get("backend", "unknown")
    port = state.get("port")
    base_url = state.get("base_url", "")
    model_id = state.get("model_id", "")
    endpoint_name = state.get("endpoint_name", "")

    alive = False
    if pid:
        try:
            import os

            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass

    status = "[green]Running[/green]" if alive else "[red]Stopped[/red]"

    console.print(
        Panel(
            f"[bold]Status:[/bold]    {status}\n"
            f"[bold]PID:[/bold]       {pid}\n"
            f"[bold]Backend:[/bold]   {backend}\n"
            f"[bold]Port:[/bold]      {port}\n"
            f"[bold]Endpoint:[/bold]  {endpoint_name}\n"
            f"[bold]Model:[/bold]     {model_id}\n"
            f"[bold]URL:[/bold]       {base_url}",
            title="Serve Status",
            border_style="green" if alive else "red",
        )
    )

    if alive and base_url:
        try:
            resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                console.print(
                    f"[green]Health check passed. {len(models)} model(s) available.[/green]"
                )
            else:
                console.print(f"[yellow]Health check returned HTTP {resp.status_code}[/yellow]")
        except Exception:
            console.print("[yellow]Health check failed (server may still be starting).[/yellow]")
