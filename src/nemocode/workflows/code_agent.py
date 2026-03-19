# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""High-level CodeAgent -- wraps Scheduler for easy use from CLI/TUI."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from nemocode.config.agents import resolve_agent_reference
from nemocode.config import load_config
from nemocode.config.schema import AgentConfig, AgentMode, FormationRole, NeMoCodeConfig
from nemocode.core.hooks import HookRunner
from nemocode.core.memory import load_all_memories
from nemocode.core.permissions import PermissionEngine
from nemocode.core.registry import Registry
from nemocode.core.router import get_auto_endpoint, route_to_formation
from nemocode.core.scheduler import ROLE_PROMPTS, AgentEvent, Scheduler
from nemocode.core.snapshot import SnapshotManager
from nemocode.tools.clarify import request_user_response
from nemocode.tools.delegate import create_delegate_tools
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
        agent_name: str | None = None,
    ) -> None:
        self._config = config or load_config(project_dir)
        self._project_dir = project_dir
        self._registry = Registry(self._config)
        self._read_only = read_only
        self._confirm_fn = confirm_fn
        self._agent_profile = self._resolve_primary_agent(agent_name)
        if self._agent_profile:
            self._tool_registry = load_tools(self._agent_profile.tools)
        elif read_only:
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
            single_role=self._agent_profile.role if self._agent_profile else FormationRole.EXECUTOR,
            single_prompt=self._single_prompt_for_agent(),
        )
        self._inject_project_context()

        # Register sub-agent orchestration tools if the agent profile includes any of them
        agent_control_tool_names = {"delegate", "spawn_agent", "wait_agent", "close_agent", "resume_agent"}
        if self._agent_profile and any(tool in self._agent_profile.tools for tool in agent_control_tool_names):
            agent_control_defs = create_delegate_tools(
                self._registry,
                self._config,
                project_context=self._scheduler._project_context,
                hook_runner=hook_runner,
                permission_engine=permission_engine,
                parent_agent_name=self._agent_profile.name if self._agent_profile else "main",
                parent_read_only=self._read_only,
            )
            for tool_def in agent_control_defs:
                self._tool_registry.register(tool_def)

    def _resolve_primary_agent(self, agent_name: str | None) -> AgentConfig | None:
        resolved_name = agent_name
        if resolved_name is None:
            resolved_name = "plan" if self._read_only else "build"
        else:
            resolved_name = resolve_agent_reference(self._config.agents, resolved_name) or resolved_name

        agent = self._config.agents.get(resolved_name)
        if agent is None:
            if agent_name is None:
                return None
            raise ValueError(f"Unknown agent profile: {resolved_name}")

        if agent.mode == AgentMode.SUBAGENT:
            raise ValueError(f"Agent profile {resolved_name} is subagent-only")

        return agent

    def _single_prompt_for_agent(self) -> str | None:
        if self._agent_profile is None:
            return None
        if self._agent_profile.prompt:
            return self._agent_profile.prompt
        return ROLE_PROMPTS.get(self._agent_profile.role)

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
        if self._agent_profile and self._agent_profile.role == FormationRole.PLANNER:
            endpoint = self._config.default_endpoint
            async for event in self._run_plan_mode(endpoint, user_input):
                yield event
            return

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

    async def _run_plan_mode(
        self,
        endpoint: str,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Generate a plan, iterate on feedback, then hand off to execution."""
        replan_input = user_input
        max_iterations = 5

        for _ in range(max_iterations):
            plan_text = ""
            async for event in self._scheduler.run_single(endpoint, replan_input):
                if event.kind == "text":
                    plan_text += event.text
                yield event

            plan_text = plan_text.strip()
            if not plan_text:
                yield AgentEvent(
                    kind="error",
                    text="Planner produced no plan to approve.",
                    is_error=True,
                    role=FormationRole.PLANNER,
                )
                return

            decision, feedback = await self._request_plan_decision(plan_text)
            if decision == "approve":
                async for event in self._run_approved_plan(user_input, plan_text):
                    yield event
                return
            if decision == "cancel":
                yield AgentEvent(
                    kind="text",
                    text="\nPlan cancelled.\n",
                    role=FormationRole.PLANNER,
                )
                return
            if decision == "pending":
                yield AgentEvent(
                    kind="text",
                    text=(
                        "\nPlan is waiting for your approval. Reply with approve, revise, or"
                        " cancel in your next message.\n"
                    ),
                    role=FormationRole.PLANNER,
                )
                return

            feedback_text = feedback.strip() or "Make the plan more concrete and actionable."
            yield AgentEvent(
                kind="status",
                text="Revising plan from user feedback.",
                role=FormationRole.PLANNER,
                phase="planning",
            )
            replan_input = (
                f"## Original Request\n{user_input}\n\n"
                f"## Previous Plan\n{plan_text}\n\n"
                f"## User Feedback\n{feedback_text}\n\n"
                "Revise the plan to address the feedback. Present only the updated plan."
            )

        yield AgentEvent(
            kind="error",
            text="Max plan revisions reached without approval.",
            is_error=True,
            role=FormationRole.PLANNER,
        )

    async def _request_plan_decision(self, plan_text: str) -> tuple[str, str]:
        """Ask the user to approve, revise, or cancel the current plan."""
        answer, pending = await request_user_response(
            (
                "Review this plan:\n\n"
                f"{plan_text}\n\n"
                "Reply with approve, revise, or cancel. If you want changes, include the"
                " feedback in your reply."
            ),
            ["approve", "revise", "cancel"],
        )
        if pending:
            return "pending", ""

        normalized = answer.strip()
        lowered = normalized.lower()
        if lowered in {"approve", "approved", "yes", "y"}:
            return "approve", ""
        if lowered in {"cancel", "cancelled", "decline", "declined", "no", "n"}:
            return "cancel", ""
        if lowered.startswith("revise"):
            _, _, remainder = normalized.partition(":")
            feedback = remainder.strip() or normalized[len("revise") :].strip()
            return "revise", feedback
        if not normalized:
            return "cancel", ""
        return "revise", normalized

    async def _run_approved_plan(
        self,
        user_input: str,
        plan_text: str,
    ) -> AsyncIterator[AgentEvent]:
        """Execute an approved plan with the normal build agent in auto-routing mode."""
        yield AgentEvent(
            kind="phase",
            phase="executing",
            role=FormationRole.EXECUTOR,
            text="Executing approved plan...",
        )
        executor_agent = CodeAgent(
            config=self._config,
            confirm_fn=self._confirm_fn,
            project_dir=self._project_dir,
            auto_route=True,
            read_only=False,
            agent_name="build",
        )
        executor_input = (
            f"{user_input}\n\n"
            f"## Approved Plan\n{plan_text}\n\n"
            "Execute the approved plan. Do not ask for plan approval again."
        )
        async for event in executor_agent.run(executor_input):
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
