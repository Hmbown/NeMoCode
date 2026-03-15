# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""High-level CodeAgent -- wraps Scheduler for easy use from CLI/TUI."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from nemocode.config import load_config
from nemocode.config.schema import NeMoCodeConfig
from nemocode.core.registry import Registry
from nemocode.core.scheduler import AgentEvent, Scheduler
from nemocode.tools.loader import load_tools

logger = logging.getLogger(__name__)


def _detect_git_context() -> str:
    """Detect git branch, staged changes, and merge conflict state.

    Returns a human-readable summary for injection into the system prompt.
    Returns empty string if not in a git repository.
    """
    parts: list[str] = []

    try:
        # Detect current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""  # Not a git repo
        branch = result.stdout.strip()
        parts.append(f"Git branch: {branch}")

        # Check for staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            staged_lines = result.stdout.strip().splitlines()
            parts.append(f"Staged changes: {len(staged_lines)} file(s)")

        # Check for merge conflicts
        result = subprocess.run(
            ["git", "ls-files", "--unmerged"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("WARNING: Merge conflict detected -- unmerged files present")

    except Exception as e:
        logger.debug("Git context detection failed: %s", e)

    return "\n".join(parts)


class CodeAgent:
    """Main entry point for agentic coding sessions."""

    def __init__(
        self,
        config: NeMoCodeConfig | None = None,
        confirm_fn: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._config = config or load_config(project_dir)
        self._registry = Registry(self._config)
        self._tool_registry = load_tools()
        self._scheduler = Scheduler(
            registry=self._registry,
            tool_registry=self._tool_registry,
            confirm_fn=confirm_fn,
        )
        self._inject_project_context()

    def _inject_project_context(self) -> None:
        """Inject project context and git state into the system prompt."""
        parts: list[str] = []

        # Project context from .nemocode.yaml
        project = self._config.project
        if project.name:
            parts.append(f"Project: {project.name}")
            if project.description:
                parts.append(f"Description: {project.description}")
            if project.tech_stack:
                parts.append(f"Tech stack: {', '.join(project.tech_stack)}")
            if project.conventions:
                parts.append("Conventions:")
                for conv in project.conventions:
                    parts.append(f"  - {conv}")

        # Git-aware context
        git_context = _detect_git_context()
        if git_context:
            parts.append("")
            parts.append(git_context)

        if parts:
            context = "\n".join(parts)
            self._scheduler.inject_system_context(context)

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run the agent with user input, yielding events."""
        formation = self._config.active_formation
        if formation:
            async for event in self._scheduler.run_formation(formation, user_input):
                yield event
        else:
            endpoint = self._config.default_endpoint
            async for event in self._scheduler.run_single(endpoint, user_input):
                yield event

    async def run_with_endpoint(
        self, endpoint_name: str, user_input: str
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._scheduler.run_single(endpoint_name, user_input):
            yield event

    async def run_with_formation(
        self, formation_name: str, user_input: str
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._scheduler.run_formation(formation_name, user_input):
            yield event

    def reset(self) -> None:
        self._scheduler.reset()

    def compact(self, keep: int = 20) -> None:
        """Compact all sessions, dropping older messages."""
        self._scheduler.compact_all(keep)

    @property
    def sessions(self) -> dict:
        """Access to scheduler sessions for context tracking."""
        return self._scheduler._sessions

    @property
    def config(self) -> NeMoCodeConfig:
        return self._config

    @property
    def registry(self) -> Registry:
        return self._registry
