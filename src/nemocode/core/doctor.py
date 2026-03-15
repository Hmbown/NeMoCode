# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Diagnostics for /doctor command — checks API keys, connectivity, tools, hardware."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

from nemocode.config.schema import NeMoCodeConfig


@dataclass
class Check:
    name: str
    status: str  # "ok", "warn", "fail"
    detail: str = ""


@dataclass
class DiagnosticReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    @property
    def ok(self) -> bool:
        return all(c.status == "ok" for c in self.checks)


def run_diagnostics(config: NeMoCodeConfig) -> DiagnosticReport:
    """Run all diagnostic checks and return a report."""
    report = DiagnosticReport()

    # 1. API key checks
    _check_api_keys(config, report)

    # 2. Tools in PATH
    _check_tools_in_path(report)

    # 3. Endpoint connectivity
    _check_endpoint_connectivity(config, report)

    # 4. Hardware detection
    _check_hardware(report)

    # 5. Config validity
    _check_config(config, report)

    return report


def _check_api_keys(config: NeMoCodeConfig, report: DiagnosticReport) -> None:
    """Check that required API keys are set."""
    ep = config.endpoints.get(config.default_endpoint)
    if not ep:
        report.add("api_key", "fail", f"Default endpoint '{config.default_endpoint}' not found")
        return

    if ep.api_key_env:
        val = os.environ.get(ep.api_key_env, "")
        if val:
            masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
            report.add("api_key", "ok", f"{ep.api_key_env} set ({masked})")
        else:
            report.add("api_key", "fail", f"{ep.api_key_env} not set")
    else:
        report.add("api_key", "ok", "No API key required (local endpoint)")


def _check_tools_in_path(report: DiagnosticReport) -> None:
    """Check that required CLI tools are available."""
    tools = {
        "git": "Version control",
        "rg": "Fast file search (ripgrep)",
        "pytest": "Test runner",
    }
    for tool_name, desc in tools.items():
        path = shutil.which(tool_name)
        if path:
            report.add(tool_name, "ok", f"Found at {path}")
        else:
            status = "warn" if tool_name in ("rg", "pytest") else "fail"
            report.add(tool_name, status, f"{desc} not found in PATH")


def _check_endpoint_connectivity(config: NeMoCodeConfig, report: DiagnosticReport) -> None:
    """Check if the default endpoint is reachable."""
    ep = config.endpoints.get(config.default_endpoint)
    if not ep:
        return

    # For local endpoints, check if the port is open
    if "localhost" in ep.base_url or "127.0.0.1" in ep.base_url:
        try:
            import urllib.parse

            parsed = urllib.parse.urlparse(ep.base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8000
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                report.add("endpoint", "ok", f"{ep.base_url} reachable")
            else:
                report.add(
                    "endpoint",
                    "warn",
                    f"{ep.base_url} not reachable (local NIM not running?)",
                )
        except Exception as e:
            report.add("endpoint", "warn", f"Connectivity check failed: {e}")
    else:
        # For remote endpoints, just check the URL looks valid
        if ep.base_url.startswith("http"):
            report.add("endpoint", "ok", f"{ep.base_url} configured")
        else:
            report.add("endpoint", "fail", f"Invalid base URL: {ep.base_url}")


def _check_hardware(report: DiagnosticReport) -> None:
    """Check GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().splitlines()
            report.add("gpu", "ok", f"{len(gpus)} GPU(s): {gpus[0].strip()}")
        else:
            report.add("gpu", "warn", "No NVIDIA GPU detected")
    except FileNotFoundError:
        report.add("gpu", "warn", "nvidia-smi not found (no NVIDIA GPU or drivers)")
    except Exception as e:
        report.add("gpu", "warn", f"GPU detection failed: {e}")


def _check_config(config: NeMoCodeConfig, report: DiagnosticReport) -> None:
    """Validate configuration."""
    # Check default endpoint exists
    if config.default_endpoint not in config.endpoints:
        report.add(
            "config",
            "fail",
            f"Default endpoint '{config.default_endpoint}' not in endpoints",
        )
        return

    # Check active formation exists
    if config.active_formation and config.active_formation not in config.formations:
        report.add(
            "config", "warn", f"Active formation '{config.active_formation}' not in formations"
        )
        return

    # Check formation endpoint references
    for name, formation in config.formations.items():
        for slot in formation.slots:
            if slot.endpoint not in config.endpoints:
                report.add(
                    "config",
                    "warn",
                    f"Formation '{name}' references unknown endpoint '{slot.endpoint}'",
                )
                return

    n_ep = len(config.endpoints)
    n_fm = len(config.formations)
    report.add("config", "ok", f"{n_ep} endpoints, {n_fm} formations")
