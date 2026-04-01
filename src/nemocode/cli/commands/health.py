# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo health — quick system health check."""

from __future__ import annotations

import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.config import load_config
from nemocode.core.credentials import KNOWN_KEYS, get_credential

console = Console()
health_app = typer.Typer(help="Quick system health check.")

COMMON_PORTS = [8000, 8001, 8002, 8004, 8005, 8080, 3000]

CACHE_DIRS = [
    Path("~/.cache/nemocode").expanduser(),
    Path("~/.cache/huggingface").expanduser(),
    Path("/tmp"),
]


@dataclass
class HealthCheck:
    name: str
    passed: bool
    detail: str


def _check_api_keys() -> list[HealthCheck]:
    """Check if API keys are configured and look structurally valid."""
    results: list[HealthCheck] = []

    nvidia_key = get_credential("NVIDIA_API_KEY")
    if nvidia_key:
        # NVIDIA API keys are typically nvapi-<hex> or similar long tokens
        looks_valid = len(nvidia_key) >= 16 and not nvidia_key.isspace()
        if looks_valid:
            masked = nvidia_key[:6] + "..." + nvidia_key[-4:]
            results.append(HealthCheck("API Key", True, f"NVIDIA_API_KEY set ({masked})"))
        else:
            results.append(
                HealthCheck("API Key", False, "NVIDIA_API_KEY set but looks invalid (too short)")
            )
    else:
        # Check if any known key is configured
        configured = [k for k in KNOWN_KEYS if get_credential(k)]
        if configured:
            results.append(
                HealthCheck(
                    "API Key",
                    True,
                    f"{configured[0]} configured (NVIDIA_API_KEY not set)",
                )
            )
        else:
            results.append(HealthCheck("API Key", False, "No API keys configured"))

    return results


def _check_local_ports() -> list[HealthCheck]:
    """Check if common local backend ports are listening."""
    listening: list[str] = []
    for port in COMMON_PORTS:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                listening.append(str(port))
        except OSError:
            pass

    if listening:
        return [
            HealthCheck(
                "Local Ports",
                True,
                f"Listening: {', '.join(listening)}",
            )
        ]
    return [
        HealthCheck(
            "Local Ports",
            False,
            "No local backports detected (common ports closed)",
        )
    ]


def _check_hardware() -> list[HealthCheck]:
    """Check GPU availability."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return [HealthCheck("Hardware", False, "nvidia-smi not found")]

    try:
        import subprocess

        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = [g.strip() for g in result.stdout.strip().splitlines()]
            return [
                HealthCheck(
                    "Hardware",
                    True,
                    f"{len(gpus)} GPU(s): {gpus[0]}",
                )
            ]
        return [HealthCheck("Hardware", False, "No NVIDIA GPU detected")]
    except (subprocess.TimeoutExpired, Exception) as e:
        return [HealthCheck("Hardware", False, f"GPU check failed: {e}")]


def _check_config() -> list[HealthCheck]:
    """Load and validate configuration."""
    try:
        cfg = load_config()
        n_ep = len(cfg.endpoints)
        n_fm = len(cfg.formations)
        n_manifests = len(cfg.manifests)
        return [
            HealthCheck(
                "Config",
                True,
                f"{n_ep} endpoints, {n_fm} formations, {n_manifests} models",
            )
        ]
    except Exception as e:
        return [HealthCheck("Config", False, f"Config load failed: {e}")]


def _check_disk_space() -> list[HealthCheck]:
    """Check disk space for cache/scratch directories."""
    issues: list[str] = []
    ok_dirs: list[str] = []

    for cache_dir in CACHE_DIRS:
        if not cache_dir.exists():
            continue
        try:
            usage = shutil.disk_usage(cache_dir)
            free_gb = usage.free / (1024**3)
            if free_gb < 1.0:
                issues.append(f"{cache_dir}: {free_gb:.1f}GB free")
            else:
                ok_dirs.append(f"{cache_dir.name}: {free_gb:.1f}GB free")
        except OSError:
            pass

    if issues:
        detail = "; ".join(issues)
        if ok_dirs:
            detail += " | OK: " + "; ".join(ok_dirs)
        return [HealthCheck("Disk Space", False, detail)]

    if ok_dirs:
        return [HealthCheck("Disk Space", True, "; ".join(ok_dirs))]

    return [HealthCheck("Disk Space", False, "No cache directories found")]


@health_app.callback(invoke_without_command=True)
def health(ctx: typer.Context) -> None:
    """Run all health checks and display a summary."""
    if ctx.invoked_subcommand is not None:
        return

    checks: list[HealthCheck] = []
    checks.extend(_check_api_keys())
    checks.extend(_check_local_ports())
    checks.extend(_check_hardware())
    checks.extend(_check_config())
    checks.extend(_check_disk_space())

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="dim", width=14)
    table.add_column("Status", width=8)
    table.add_column("Details")

    for check in checks:
        icon = "[green]✅[/green]" if check.passed else "[red]❌[/red]"
        table.add_row(check.name, icon, check.detail)

    console.print(Panel(table, title="Health Check", border_style="blue"))

    all_passed = all(c.passed for c in checks)
    if all_passed:
        console.print("\n[bold green]All checks passed![/bold green]")
    else:
        failed = [c.name for c in checks if not c.passed]
        console.print(f"\n[bold red]{len(failed)} check(s) failed: {', '.join(failed)}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    health_app()
