# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Shell execution tool."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys

from nemocode.tools import tool

_MAX_OUTPUT = 100_000  # chars

# Specific env vars to strip from child processes
_SENSITIVE_VARS = {
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPINFRA_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "DATABASE_URL",
    "DB_PASSWORD",
}
# Suffix patterns for dynamic key names (e.g., CUSTOM_API_KEY)
_SENSITIVE_SUFFIXES = ("_API_KEY", "_SECRET_KEY", "_SECRET", "_PASSWORD", "_TOKEN")


def _sanitized_env() -> dict[str, str]:
    """Create environment dict with sensitive variables removed."""
    env = {}
    for k, v in os.environ.items():
        if k in _SENSITIVE_VARS:
            continue
        upper = k.upper()
        if any(upper.endswith(suf) for suf in _SENSITIVE_SUFFIXES):
            continue
        env[k] = v
    return env


@tool(
    name="bash_exec",
    description="Execute a shell command and return its output.",
    requires_confirmation=True,
    category="bash",
)
async def bash_exec(command: str, timeout: int = 120, cwd: str = "") -> str:
    """Run a shell command.
    command: The shell command to execute.
    timeout: Timeout in seconds (default 120).
    cwd: Working directory (default: current directory).
    """
    work_dir = cwd or os.getcwd()
    try:
        # Use start_new_session so we can kill the entire process group on timeout
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=_sanitized_env(),
            start_new_session=(sys.platform != "win32"),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout_str = stdout.decode(errors="replace")
        stderr_str = stderr.decode(errors="replace")

        # Truncate if too large
        if len(stdout_str) > _MAX_OUTPUT:
            stdout_str = (
                stdout_str[:_MAX_OUTPUT] + f"\n... (truncated, {len(stdout_str)} total chars)"
            )
        if len(stderr_str) > _MAX_OUTPUT:
            stderr_str = (
                stderr_str[:_MAX_OUTPUT] + f"\n... (truncated, {len(stderr_str)} total chars)"
            )

        result: dict = {"exit_code": proc.returncode}
        if stdout_str.strip():
            result["stdout"] = stdout_str
        if stderr_str.strip():
            result["stderr"] = stderr_str
        return json.dumps(result)
    except asyncio.TimeoutError:
        # Kill the entire process group (shell + all children)
        try:
            if sys.platform != "win32" and proc.pid:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            await proc.wait()
        except Exception:
            pass
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})
