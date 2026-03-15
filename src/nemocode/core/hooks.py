# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Hooks system — execute shell commands before/after tool calls.

Hooks are configured in .nemocode.yaml:

    hooks:
      pre_write_file:
        - "echo 'Writing {{path}}'"
      post_git_commit:
        - "git push"
      pre_bash_exec:
        - "echo 'Running: {{command}}'"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HookRunner:
    """Execute configured hooks around tool calls."""

    def __init__(self, hooks_config: dict[str, list[str]] | None = None) -> None:
        self._hooks = hooks_config or {}

    @property
    def enabled(self) -> bool:
        return bool(self._hooks)

    async def run_pre(self, tool_name: str, args: dict[str, Any]) -> list[str]:
        """Run pre-tool hooks. Returns list of outputs."""
        key = f"pre_{tool_name}"
        return await self._run_hooks(key, args)

    async def run_post(self, tool_name: str, args: dict[str, Any], result: str = "") -> list[str]:
        """Run post-tool hooks. Returns list of outputs."""
        key = f"post_{tool_name}"
        template_vars = {**args, "result": result[:1000]}
        return await self._run_hooks(key, template_vars)

    async def _run_hooks(self, key: str, template_vars: dict[str, Any]) -> list[str]:
        """Run all hooks for a given key."""
        commands = self._hooks.get(key, [])
        if not commands:
            return []

        outputs = []
        for cmd_template in commands:
            cmd = _safe_template(cmd_template, template_vars)
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                output = stdout.decode().strip()
                if proc.returncode != 0:
                    err = stderr.decode().strip()
                    logger.warning("Hook failed (%s): %s", key, err)
                    outputs.append(f"[hook error] {err}")
                elif output:
                    outputs.append(output)
            except asyncio.TimeoutError:
                logger.warning("Hook timed out: %s", cmd)
                outputs.append("[hook timeout]")
            except Exception as e:
                logger.warning("Hook error (%s): %s", key, e)
                outputs.append(f"[hook error] {e}")

        return outputs


def _safe_template(template: str, variables: dict[str, Any]) -> str:
    """Substitute {{var}} placeholders safely.

    Only substitutes known variables; unknown placeholders are left as-is.
    Values are NOT shell-escaped — hook authors are responsible for safe commands.
    """
    result = template
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    return result
