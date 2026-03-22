# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""High-level CodeAgent -- wraps Scheduler for easy use from CLI/TUI."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from nemocode.config import load_config
from nemocode.config.agents import resolve_agent_reference
from nemocode.config.schema import AgentConfig, AgentMode, FormationRole, NeMoCodeConfig
from nemocode.core.hooks import HookRunner
from nemocode.core.memory import load_all_memories
from nemocode.core.permissions import PermissionEngine
from nemocode.core.registry import Registry
from nemocode.core.router import get_auto_endpoint, route_to_formation
from nemocode.core.scheduler import ROLE_PROMPTS, AgentEvent, Scheduler
from nemocode.core.snapshot import SnapshotManager
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
        self._project_dir = (project_dir or Path.cwd()).resolve()
        self._registry = Registry(self._config)
        self._read_only = read_only
        self._confirm_fn = confirm_fn
        self._pending_plan_text: str | None = None
        self._pending_plan_user_input: str | None = None
        self._agent_profile = self._resolve_primary_agent(agent_name)
        if self._agent_profile:
            self._tool_registry = load_tools(
                self._agent_profile.tools,
                project_dir=self._project_dir,
            )
        elif read_only:
            # Plan mode: no file writes or git commits, but bash is allowed
            self._tool_registry = load_tools(
                ["fs_read", "git_read", "bash", "rg", "glob", "clarify"],
                project_dir=self._project_dir,
            )
        else:
            self._tool_registry = load_tools(project_dir=self._project_dir)
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
        agent_control_tool_names = {
            "delegate",
            "spawn_agent",
            "wait_agent",
            "close_agent",
            "resume_agent",
        }
        if self._agent_profile and any(
            tool in self._agent_profile.tools for tool in agent_control_tool_names
        ):
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
            resolved_name = (
                resolve_agent_reference(self._config.agents, resolved_name) or resolved_name
            )

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
        """Run the agent with user input, yielding events.

        In plan mode, the response streams normally (with read-only tools),
        then a decision menu appears so the user can execute, revise, ask
        questions, or just continue chatting.
        """
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
        """Stream a plan-mode response and store it as pending for decision.

        The response streams normally (read-only tools allowed). Afterward,
        the plan text is saved as pending so the REPL can show the decision
        menu non-blockingly. The user then types 1-4 or continues chatting.
        """
        plan_text = ""
        had_activity = False
        async for event in self._scheduler.run_single(endpoint, user_input):
            if event.kind == "text":
                plan_text += event.text
            if event.kind in ("text", "thinking", "tool_call"):
                had_activity = True
            yield event

        plan_text = plan_text.strip()
        if plan_text:
            self._pending_plan_text = plan_text
            self._pending_plan_user_input = user_input
        elif had_activity:
            # Model did work (thinking/tools) but produced no text — nudge it
            yield AgentEvent(
                kind="text",
                text="\n[No text response was generated. Try rephrasing your request.]\n",
                role=FormationRole.PLANNER,
            )

    async def _run_approved_plan(
        self,
        user_input: str,
        plan_text: str,
    ) -> AsyncIterator[AgentEvent]:
        """Execute an approved plan using the current endpoint.

        Transfers the planner's session history to the executor so it has
        all the research context (file reads, git checks, tool results).
        """
        yield AgentEvent(
            kind="phase",
            phase="executing",
            role=FormationRole.EXECUTOR,
            text="Executing approved plan...",
        )
        executor_agent = CodeAgent(
            config=self._config,
            confirm_fn=None,  # Plan was approved — auto-approve all tools
            project_dir=self._project_dir,
            auto_route=False,
            read_only=False,
            agent_name="build",
        )
        # Transfer planner's conversation history to executor
        old_sessions = self._scheduler._sessions
        if old_sessions:
            old_session = next(iter(old_sessions.values()), None)
            if old_session and old_session.messages:
                from nemocode.core.sessions import Session
                from nemocode.core.streaming import Role as MsgRole

                exec_role = executor_agent._scheduler._single_role
                s = Session(
                    id=executor_agent._scheduler._single_session_id,
                    endpoint_name=old_session.endpoint_name,
                )
                # Set executor system prompt
                exec_prompt = executor_agent._scheduler._single_prompt or ROLE_PROMPTS.get(
                    FormationRole.EXECUTOR, ""
                )
                if executor_agent._scheduler._project_context:
                    ctx = executor_agent._scheduler._project_context
                    exec_prompt += f"\n\n## Project Context\n{ctx}"
                if exec_prompt:
                    s.add_system(exec_prompt)
                # Copy non-system messages from planner session
                s.messages.extend(m for m in old_session.messages if m.role != MsgRole.SYSTEM)
                executor_agent._scheduler._sessions[exec_role] = s

        executor_input = (
            f"## Approved Plan\n{plan_text}\n\n"
            "Execute the approved plan above. You have the full conversation "
            "context from the planning phase. Do not re-plan or ask for approval."
        )
        async for event in executor_agent.run(executor_input):
            yield event

        # Transfer executor's session back so the parent has full context
        exec_sessions = executor_agent._scheduler._sessions
        if exec_sessions:
            exec_session = next(iter(exec_sessions.values()), None)
            if exec_session and exec_session.messages:
                from nemocode.core.streaming import Role as MsgRole

                parent_role = self._scheduler._single_role
                parent_session = self._scheduler._sessions.get(parent_role)
                if parent_session:
                    # Append executor's non-system messages to parent session
                    parent_session.messages.extend(
                        m for m in exec_session.messages if m.role != MsgRole.SYSTEM
                    )

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

    @property
    def has_pending_plan(self) -> bool:
        """Whether a plan approval is waiting for user response."""
        return self._pending_plan_text is not None

    @property
    def pending_plan_text(self) -> str | None:
        """The plan text awaiting approval, or None."""
        return self._pending_plan_text

    @staticmethod
    def parse_plan_decision(user_input: str) -> tuple[str, str]:
        """Parse user input as a plan decision without prompting.

        Returns (decision, feedback) where decision is one of:
        approve, cancel, revise, ask, or pending (not a recognized decision).
        """
        normalized = user_input.strip()
        lowered = normalized.lower()
        # Numbered shortcuts
        if lowered in {"1", "1."}:
            return "approve", ""
        if lowered in {"4", "4."}:
            return "cancel", ""
        if lowered.startswith("2"):
            feedback = normalized[1:].strip().lstrip(".").strip()
            return "revise", feedback
        if lowered.startswith("3"):
            question = normalized[1:].strip().lstrip(".").strip()
            return "ask", question
        # Legacy text inputs
        if lowered in {"approve", "approved", "yes", "y"}:
            return "approve", ""
        if lowered in {"cancel", "cancelled", "decline", "declined", "no", "n"}:
            return "cancel", ""
        if lowered.startswith("revise"):
            _, _, remainder = normalized.partition(":")
            feedback = remainder.strip() or normalized[len("revise") :].strip()
            return "revise", feedback
        return "pending", ""

    async def try_handle_plan_response(self, user_input: str) -> AsyncIterator[AgentEvent] | None:
        """If a plan approval is pending, interpret user_input as a decision.

        Returns an async iterator of events if the input was a decision,
        or None if the input was not a recognized plan decision.
        """
        if not self._pending_plan_text:
            return None

        decision, feedback = self.parse_plan_decision(user_input)
        if decision == "pending":
            return None  # Not a recognized decision — caller should treat as new input

        plan_text = self._pending_plan_text
        original_input = self._pending_plan_user_input or ""

        # For "ask", keep the plan pending — _execute_plan_decision will
        # answer the question and re-prompt, potentially re-setting pending state.
        if decision != "ask":
            self._pending_plan_text = None
            self._pending_plan_user_input = None

        return self._execute_plan_decision(decision, feedback, original_input, plan_text)

    async def _execute_plan_decision(
        self,
        decision: str,
        feedback: str,
        original_input: str,
        plan_text: str,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a plan decision (approve, cancel, or revise).

        Never blocks for user input — revise streams the new plan and
        stores it as pending for the REPL to handle.
        """
        endpoint = self._config.default_endpoint

        if decision == "approve":
            async for event in self._run_approved_plan(original_input, plan_text):
                yield event
        elif decision == "cancel":
            yield AgentEvent(kind="text", text="\nPlan cancelled.\n", role=FormationRole.PLANNER)
        elif decision == "revise":
            feedback_text = feedback.strip() or "Make the plan more concrete and actionable."
            yield AgentEvent(
                kind="status",
                text="Revising plan from user feedback.",
                role=FormationRole.PLANNER,
                phase="planning",
            )
            replan_input = (
                f"## Original Request\n{original_input}\n\n"
                f"## Previous Plan\n{plan_text}\n\n"
                f"## User Feedback\n{feedback_text}\n\n"
                "Revise the plan to address the feedback. Present only the updated plan."
            )
            plan_text = ""
            async for event in self._scheduler.run_single(endpoint, replan_input):
                if event.kind == "text":
                    plan_text += event.text
                yield event
            plan_text = plan_text.strip()
            if not plan_text:
                self._pending_plan_text = None
                self._pending_plan_user_input = None
                yield AgentEvent(
                    kind="error",
                    text="Revision produced no output. Plan cleared.",
                    is_error=True,
                    role=FormationRole.PLANNER,
                )
                return
            # Store revised plan as pending — REPL shows menu again
            self._pending_plan_text = plan_text
            self._pending_plan_user_input = original_input
        else:
            self._pending_plan_text = plan_text
            self._pending_plan_user_input = original_input
            yield AgentEvent(
                kind="text",
                text="\nPlan waiting for your decision. Choose 1-4 in your next message.\n",
                role=FormationRole.PLANNER,
            )
