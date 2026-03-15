# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test tools: run pytest tests."""

from __future__ import annotations

import json
from pathlib import Path

from nemocode.tools import tool

# Project root is set at import time; tools should not write outside it
_PROJECT_ROOT: Path = Path.cwd().resolve()


def set_project_root(root: Path) -> None:
    """Override the project root for path sandboxing."""
    global _PROJECT_ROOT
    _PROJECT_ROOT = root.resolve()


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to cwd, with safety checks."""
    p = Path(path).resolve()
    return p


def _is_within_project(p: Path) -> bool:
    """Check if a resolved path is within the project root."""
    try:
        p.relative_to(_PROJECT_ROOT)
        return True
    except ValueError:
        return False


@tool(name="run_tests", description="Run pytest tests and return the result.", category="test")
async def run_tests(test_path: str = ".", extra_args: list[str] | None = None) -> str:
    """Run pytest tests.
    test_path: Path to the test directory or file (default: current directory).
    extra_args: Additional arguments to pass to pytest (as a list of strings).
    """
    if extra_args is None:
        extra_args = []
    p = _resolve_path(test_path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot run tests outside project directory: {p}"})
    if not p.exists():
        return json.dumps({"error": f"Test path not found: {test_path}"})
    # Build pytest command
    cmd = ["pytest"] + extra_args + [str(p)]
    # Use subprocess to run the command
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_PROJECT_ROOT),
        )
        stdout, stderr = await proc.communicate()
        result = {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }
        return json.dumps(result)
    except FileNotFoundError:
        return json.dumps({"error": "pytest not found. Ensure pytest is installed and in PATH."})
    except Exception as e:
        return json.dumps({"error": str(e)})
