# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""First-run experience — guides new users through setup.

Detects whether NeMoCode has been used before, checks for API keys,
detects hardware, and provides actionable setup guidance.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode import __version__
from nemocode.config import ensure_config_dir

console = Console()

_USER_CONFIG_DIR = Path(os.environ.get("NEMOCODE_CONFIG_DIR", "~/.config/nemocode")).expanduser()
_FIRST_RUN_MARKER = _USER_CONFIG_DIR / ".first_run_done"


def is_first_run() -> bool:
    """Check if this is the first time NeMoCode is being run."""
    return not _FIRST_RUN_MARKER.exists()


def mark_first_run_done() -> None:
    """Record that the first-run wizard has been shown."""
    ensure_config_dir()
    _FIRST_RUN_MARKER.touch()


def has_any_api_key() -> bool:
    """Check if any API key is configured (keyring or env)."""
    key_vars = [
        "NVIDIA_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "DEEPINFRA_API_KEY",
    ]
    # Check environment first (fast)
    for var in key_vars:
        if os.environ.get(var):
            return True

    # Check keyring
    try:
        from nemocode.core.credentials import get_credential

        for var in key_vars:
            if get_credential(var):
                return True
    except Exception:
        pass

    return False


def run_first_run_wizard() -> None:
    """Run the first-run setup wizard. Shows guidance, detects hardware."""
    console.print()
    console.print(
        Panel(
            f"[bold green]Welcome to NeMoCode v{__version__}[/bold green]\n\n"
            "[dim]Terminal-first agentic coding CLI for NVIDIA Nemotron 3[/dim]",
            border_style="green",
            expand=False,
            padding=(1, 4),
        )
    )
    console.print()

    # ── API Key Check ──
    if not has_any_api_key():
        console.print(
            "[bold yellow]No API key found.[/bold yellow] You need an API key to get started.\n"
        )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Step", style="bold cyan", justify="right", width=4)
        table.add_column("Action")

        table.add_row("1.", "Go to [link=https://build.nvidia.com]https://build.nvidia.com[/link]")
        table.add_row("2.", "Sign up (free) and generate an API key")
        table.add_row("3.", "Run: [cyan]nemo auth setup[/cyan]")
        table.add_row("", "")
        table.add_row("", "Or set the environment variable:")
        table.add_row("", '[dim]export NVIDIA_API_KEY="nvapi-..."[/dim]')

        console.print(table)
        console.print()
    else:
        console.print("[green]API key found.[/green] Ready to use.\n")

    # ── Hardware Detection ──
    console.print("[bold]Detected hardware:[/bold]")
    try:
        from nemocode.core.hardware import detect_hardware

        profile = detect_hardware()

        hw_lines = []
        hw_lines.append(f"  CPU: {profile.cpu_cores} cores")
        hw_lines.append(f"  RAM: {profile.ram_gb:.1f} GB")

        if profile.gpus:
            for gpu in profile.gpus:
                hw_lines.append(f"  GPU: {gpu.name} ({gpu.vram_gb:.0f} GB VRAM)")
        else:
            hw_lines.append("  GPU: [dim]None (using hosted endpoints)[/dim]")

        mic_status = "[green]Yes[/green]" if profile.has_microphone else "[dim]No[/dim]"
        spk_status = "[green]Yes[/green]" if profile.has_speakers else "[dim]No[/dim]"
        hw_lines.append(f"  Mic: {mic_status} | Speakers: {spk_status}")

        console.print("\n".join(hw_lines))

        # Formation recommendation
        rec = profile.recommend_formation()
        console.print(f"\n[bold]Recommended:[/bold] Use the [cyan]{rec}[/cyan] formation", end="")
        if rec == "solo":
            console.print(" with hosted Nemotron 3 Super.")
        elif rec == "super-nano":
            console.print(" (Super plans, Nano executes).")
        elif rec == "local":
            console.print(" for local-only operation.")
        else:
            console.print(".")
    except Exception:
        console.print("  [dim]Hardware detection unavailable.[/dim]")

    console.print()

    # ── Quick Start ──
    console.print("[bold]Quick start:[/bold]")
    console.print("  [cyan]nemo code[/cyan]                  Start interactive coding session")
    console.print("  [cyan]nemo code 'fix the bug'[/cyan]    One-shot task")
    console.print("  [cyan]nemo chat 'hello'[/cyan]          Simple chat (no tools)")
    console.print("  [cyan]nemo init[/cyan]                  Create project config")
    console.print("  [cyan]nemo hardware recommend[/cyan]    Hardware-based recommendations")
    console.print()

    mark_first_run_done()


def check_and_run_first_run() -> None:
    """Check if first run, and if so, run the wizard.

    Also checks for missing API keys even on subsequent runs,
    showing a brief warning.
    """
    if is_first_run():
        run_first_run_wizard()
    elif not has_any_api_key():
        console.print(
            "[yellow]No API key configured.[/yellow] "
            "Run [cyan]nemo auth setup[/cyan] or set NVIDIA_API_KEY.\n"
        )
