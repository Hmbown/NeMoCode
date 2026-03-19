# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Guided runtime setup for hosted NIM and local OpenAI-compatible backends."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from nemocode.config import _USER_CONFIG_PATH, ensure_config_dir
from nemocode.core.credentials import get_credential, set_credential, storage_backend

console = Console()


@dataclass(frozen=True)
class PresetChoice:
    """A runtime preset that can become the user's default endpoint."""

    endpoint_name: str
    label: str
    model_id: str | None = None
    tier: str | None = None
    base_url: str | None = None
    startup_command: str = ""

    @property
    def is_custom(self) -> bool:
        return self.tier is not None and self.base_url is not None and self.model_id is not None


_BACKEND_OPTIONS = [
    ("nim", "Hosted NVIDIA NIM (recommended)"),
    ("vllm", "Local vLLM server"),
    ("sglang", "Local SGLang server"),
]

_HOSTED_PRESETS = [
    PresetChoice("nim-super", "Nemotron 3 Super"),
    PresetChoice("nim-nano", "Nemotron 3 Nano 30B"),
    PresetChoice("nim-nano-9b", "Nemotron Nano 9B v2"),
    PresetChoice("nim-nano-4b", "Nemotron Nano 4B v1.1"),
]

_LOCAL_PRESETS: dict[str, list[PresetChoice]] = {
    "vllm": [
        PresetChoice(
            "local-vllm-super",
            "Nemotron 3 Super",
            startup_command=(
                "vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \\\n"
                "  --host 0.0.0.0 --port 8000"
            ),
        ),
        PresetChoice(
            "local-vllm-nano",
            "Nemotron 3 Nano 30B",
            startup_command=(
                "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\\n"
                "  --host 0.0.0.0 --port 8000"
            ),
        ),
        PresetChoice(
            "local-vllm-nano9b",
            "Nemotron Nano 9B v2",
            startup_command=(
                "vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\\n"
                "  --host 0.0.0.0 --port 8000"
            ),
        ),
        PresetChoice("__custom__", "Custom OpenAI-compatible vLLM model"),
    ],
    "sglang": [
        PresetChoice(
            "local-sglang-super",
            "Nemotron 3 Super",
            startup_command=(
                "python -m sglang.launch_server \\\n"
                "  --model nvidia/nemotron-3-super-120b-a12b \\\n"
                "  --host 0.0.0.0 --port 8000"
            ),
        ),
        PresetChoice(
            "local-sglang-nano9b",
            "Nemotron Nano 9B v2",
            startup_command=(
                "python -m sglang.launch_server \\\n"
                "  --model nvidia/nemotron-nano-9b-v2 \\\n"
                "  --host 0.0.0.0 --port 8000"
            ),
        ),
        PresetChoice(
            "local-sglang-nano4b",
            "Nemotron Nano 4B v1.1",
            startup_command=(
                "python -m sglang.launch_server \\\n"
                "  --model nvidia/llama-3.1-nemotron-nano-4b-v1.1 \\\n"
                "  --host 0.0.0.0 --port 8000"
            ),
        ),
        PresetChoice("__custom__", "Custom OpenAI-compatible SGLang model"),
    ],
}


def run_setup_wizard() -> dict[str, str]:
    """Prompt for a runtime backend and write user-level defaults."""
    console.print(
        Panel(
            "[bold]NeMoCode Guided Setup[/bold]\n\n"
            "Pick a default runtime once. New projects will inherit it unless they "
            "set an explicit project-level override.",
            border_style="bright_green",
            expand=False,
        )
    )
    console.print(f"[dim]Credential storage: {storage_backend()}[/dim]\n")

    backend = _choose_option("Choose your default runtime", _BACKEND_OPTIONS, default=1)
    if backend == "nim":
        _maybe_store_nvidia_key()
        preset = _choose_preset("Choose a hosted NIM default", _HOSTED_PRESETS, default=1)
        _write_user_runtime_config(preset)
        _print_hosted_summary(preset)
        return {"backend": backend, "endpoint": preset.endpoint_name}

    preset = _choose_preset(
        f"Choose your default {backend} model",
        _LOCAL_PRESETS[backend],
        default=1,
    )
    if preset.endpoint_name == "__custom__":
        preset = _prompt_custom_local_preset(backend)

    _write_user_runtime_config(preset)
    _print_local_summary(backend, preset)
    return {"backend": backend, "endpoint": preset.endpoint_name}


def _choose_preset(prompt: str, options: list[PresetChoice], default: int = 1) -> PresetChoice:
    keyed_options = [(option.endpoint_name, option.label) for option in options]
    choice = _choose_option(prompt, keyed_options, default=default)
    for option in options:
        if option.endpoint_name == choice:
            return option
    raise RuntimeError(f"Unknown setup preset: {choice}")


def _choose_option(prompt: str, options: list[tuple[str, str]], default: int = 1) -> str:
    for index, (_, label) in enumerate(options, start=1):
        console.print(f"  [cyan]{index}.[/cyan] {label}")
    console.print()

    while True:
        answer = typer.prompt(prompt, default=str(default)).strip()
        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
        console.print(f"[yellow]Enter a number between 1 and {len(options)}.[/yellow]")


def _maybe_store_nvidia_key() -> None:
    existing = get_credential("NVIDIA_API_KEY")
    if existing:
        console.print("[green]NVIDIA_API_KEY already configured.[/green]\n")
        return

    console.print(
        "[bold]NVIDIA API Key[/bold]\n"
        "[dim]Get a free key at https://build.nvidia.com and paste it here."
        " Press Enter to skip.[/dim]"
    )
    value = typer.prompt(
        "NVIDIA_API_KEY",
        default="",
        show_default=False,
        hide_input=True,
    ).strip()
    console.print()
    if not value:
        console.print(
            "[yellow]Skipping key storage. Hosted NIM will still require NVIDIA_API_KEY later."
            "[/yellow]\n"
        )
        return

    success, method = set_credential("NVIDIA_API_KEY", value)
    if success:
        console.print(f"[green]Stored NVIDIA_API_KEY in {method}.[/green]\n")
    else:
        console.print("[yellow]Keyring unavailable.[/yellow]")
        console.print('[dim]Export it manually: export NVIDIA_API_KEY="nvapi-..."[/dim]\n')


def _prompt_custom_local_preset(backend: str) -> PresetChoice:
    base_url = _prompt_nonempty("Base URL", default="http://localhost:8000/v1")
    model_id = _prompt_nonempty("Model ID")
    slug = _slugify(model_id) or "custom"
    endpoint_name = _prompt_nonempty(
        "Endpoint name",
        default=f"local-{backend}-{slug}"[:48],
    )
    tier = "local-vllm" if backend == "vllm" else "local-sglang"
    startup = (
        "# Start your server separately, then verify it with:\n"
        f"nemo endpoint test {endpoint_name}"
    )
    return PresetChoice(
        endpoint_name=endpoint_name,
        label=f"Custom {backend} model",
        model_id=model_id,
        tier=tier,
        base_url=base_url,
        startup_command=startup,
    )


def _prompt_nonempty(text: str, default: str | None = None) -> str:
    while True:
        value = typer.prompt(text, default=default or "").strip()
        if value:
            return value
        console.print("[yellow]A value is required.[/yellow]")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _load_user_config() -> dict[str, Any]:
    if not _USER_CONFIG_PATH.exists():
        return {}
    with open(_USER_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _write_user_runtime_config(preset: PresetChoice) -> None:
    ensure_config_dir()
    data = _load_user_config()
    data["default_endpoint"] = preset.endpoint_name
    data.pop("active_formation", None)

    if preset.is_custom:
        endpoints = data.setdefault("endpoints", {})
        endpoints[preset.endpoint_name] = {
            "name": preset.endpoint_name,
            "tier": preset.tier,
            "base_url": preset.base_url,
            "model_id": preset.model_id,
            "capabilities": ["chat", "code"],
        }

    with open(_USER_CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _print_hosted_summary(preset: PresetChoice) -> None:
    console.print(f"[green]Default endpoint set to {preset.endpoint_name}.[/green]")
    console.print(
        "[dim]Hosted Nemotron calls will use NVIDIA NIM via NVIDIA_API_KEY by default.[/dim]"
    )
    console.print(
        f"[dim]Next: nemo endpoint test {preset.endpoint_name}  or  nemo code[/dim]"
    )


def _print_local_summary(backend: str, preset: PresetChoice) -> None:
    console.print(f"[green]Default endpoint set to {preset.endpoint_name}.[/green]")
    console.print(
        f"[dim]NeMoCode will now prefer your local {backend} endpoint unless a project"
        " overrides it.[/dim]"
    )
    if preset.startup_command:
        console.print(
            Panel(
                preset.startup_command,
                title=f"[bold]{backend} launch command[/bold]",
                border_style="green",
            )
        )
    console.print(
        f"[dim]Next: start the server, run nemo endpoint test {preset.endpoint_name},"
        " then nemo code[/dim]"
    )
