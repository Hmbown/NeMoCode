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
from nemocode.core.hooks import HookRunner
from nemocode.core.memory import load_all_memories
from nemocode.core.permissions import PermissionEngine
from nemocode.core.registry import Registry
from nemocode.core.router import get_auto_endpoint, route_to_formation
from nemocode.core.scheduler import AgentEvent, Scheduler
from nemocode.core.snapshot import SnapshotManager
from nemocode.tools.delegate import create_delegate_tool
from nemocode.tools.loader import load_tools

logger = logging.getLogger(__name__)

_INSTRUCTION_FILENAMES = (
    "agents.md",
    "AGENTS.md",
    "NEMOCODE.md",
)


def _detect_git_context() -> str:
    """Detect git branch, status, recent commits, and merge conflict state.

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

        # Working tree status (short format)
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            status_lines = result.stdout.strip().splitlines()
            parts.append(f"Working tree: {len(status_lines)} changed file(s)")
            for line in status_lines[:15]:
                parts.append(f"  {line}")
            if len(status_lines) > 15:
                parts.append(f"  ... and {len(status_lines) - 15} more")

        # Recent commits
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("Recent commits:")
            for line in result.stdout.strip().splitlines():
                parts.append(f"  {line}")

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


def _append_instruction_files(
    base_dir: Path,
    heading: str,
    contents: list[str],
    seen: set[Path],
) -> None:
    """Read supported instruction files from one directory in priority order."""
    for filename in _INSTRUCTION_FILENAMES:
        path = base_dir / filename
        if not path.exists():
            continue
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        try:
            text = path.read_text().strip()
        except Exception:
            continue
        if not text:
            continue
        seen.add(resolved)
        contents.append(f"# {heading} ({path.name})\n{text}")


def _discover_instruction_files(start_dir: Path | None = None) -> str:
    """Search for supported instruction files and return combined content.

    Search order: ~/.config/nemocode/* → project root → cwd
    Supported filenames are, in priority order per directory:
      agents.md → AGENTS.md → NEMOCODE.md
    All found files are concatenated with source labels.
    """
    contents: list[str] = []
    seen: set[Path] = set()

    # Global instructions
    global_dir = Path("~/.config/nemocode").expanduser()
    _append_instruction_files(global_dir, "Global Instructions", contents, seen)

    # Walk up from cwd to find project root (has .nemocode.yaml or .git)
    cwd = (start_dir or Path.cwd()).resolve()
    project_root = cwd
    for parent in [cwd, *cwd.parents]:
        if (parent / ".nemocode.yaml").exists() or (parent / ".git").exists():
            project_root = parent
            break
        if parent == Path.home():
            break

    # Project root instructions
    if project_root != cwd:
        _append_instruction_files(
            project_root,
            f"Project Instructions ({project_root.name})",
            contents,
            seen,
        )

    # CWD instructions (if different from project root)
    _append_instruction_files(cwd, f"Directory Instructions ({cwd.name})", contents, seen)

    return "\n\n".join(contents)


class CodeAgent:
    """Main entry point for agentic coding sessions."""

    def __init__(
        self,
        config: NeMoCodeConfig | None = None,
        confirm_fn: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
        project_dir: Path | None = None,
        *,
        auto_route: bool = False,
        read_only: bool = False,
    ) -> None:
        self._config = config or load_config(project_dir)
        self._project_dir = project_dir
        self._registry = Registry(self._config)
        self._read_only = read_only
        if read_only:
            # Plan mode: read-only tools only — no writes, edits, bash, commits
            self._tool_registry = load_tools(["fs_read", "git_read", "rg", "glob", "clarify"])
        else:
            self._tool_registry = load_tools()
        self._auto_route = auto_route
        self._mcp_clients: list = []

        # Set up hooks
        hooks_dict = self._config.hooks.to_dict() if self._config.hooks else {}
        hook_runner = HookRunner(hooks_dict) if hooks_dict else None

        # Set up permission engine from config rules
        permission_engine = None
        if self._config.permissions.permission_rules:
            rules_config = [r.model_dump() for r in self._config.permissions.permission_rules]
            permission_engine = PermissionEngine.from_config(rules_config)

        # Git snapshot manager
        self._snapshot_mgr = SnapshotManager()

        self._scheduler = Scheduler(
            registry=self._registry,
            tool_registry=self._tool_registry,
            confirm_fn=confirm_fn,
            hook_runner=hook_runner,
            read_only=read_only,
            permission_engine=permission_engine,
            max_tool_rounds=self._config.max_tool_rounds,
        )
        self._inject_project_context()

        # Register delegate tool bound to our registry/config
        if not read_only:
            delegate_def = create_delegate_tool(self._registry, self._config)
            self._tool_registry.register(delegate_def)

    def _inject_project_context(self) -> None:
        """Inject project context, instruction files, memory, and git state."""
        parts: list[str] = []

        # Project instruction files (highest priority)
        instructions = _discover_instruction_files(self._project_dir)
        if instructions:
            parts.append(instructions)

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

        # Cross-session memories
        memories = load_all_memories()
        if memories:
            parts.append("")
            parts.append(memories)

        # Git-aware context
        git_context = _detect_git_context()
        if git_context:
            parts.append("")
            parts.append(git_context)

        if parts:
            context = "\n".join(parts)
            self._scheduler.inject_system_context(context)

    async def init_mcp(self) -> None:
        """Initialize MCP server connections (call once at REPL startup)."""
        servers = self._config.mcp.servers
        if not servers:
            return
        try:
            from nemocode.tools.mcp import register_mcp_tools

            self._mcp_clients = await register_mcp_tools(servers, self._tool_registry)
            logger.info("MCP: %d server(s) connected", len(self._mcp_clients))
        except Exception as e:
            logger.warning("MCP initialization failed: %s", e)

    async def cleanup_mcp(self) -> None:
        """Disconnect all MCP servers."""
        for client in self._mcp_clients:
            try:
                await client.disconnect()
            except Exception:
                pass
        self._mcp_clients.clear()

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run the agent with user input, yielding events."""
        formation = self._config.active_formation

        # Auto-routing when enabled and no explicit formation
        if self._auto_route and not formation:
            routed_formation = route_to_formation(user_input, self._config)
            if routed_formation:
                formation = routed_formation
            else:
                auto_ep = get_auto_endpoint(user_input, self._config)
                if auto_ep:
                    async for event in self._scheduler.run_single(auto_ep, user_input):
                        yield event
                    return

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

    @property
    def snapshot_mgr(self) -> SnapshotManager:
        return self._snapshot_mgr

    async def create_snapshot(self, label: str = "") -> dict | None:
        """Create a git snapshot for safe rollback."""
        snap = await self._snapshot_mgr.create_snapshot(label)
        return {"id": snap.id, "kind": snap.kind, "files": snap.files_changed} if snap else None
