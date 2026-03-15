# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Scheduler — drives formations through their pipeline.

Handles planner -> executor -> reviewer loop, tool execution,
confirmation gates, and the Super + Nano dispatch pattern.
Integrates context window management for automatic compaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from nemocode.config.schema import FormationRole, FormationSlot
from nemocode.core.context import ContextManager
from nemocode.core.registry import Registry
from nemocode.core.sessions import Session
from nemocode.core.streaming import Message, Role, ToolCall
from nemocode.providers.nim_chat import NIMChatProvider
from nemocode.tools import ToolRegistry

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 30
ConfirmFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass
class AgentEvent:
    kind: str  # text, thinking, tool_call, tool_result, error, usage, phase
    text: str = ""
    thinking: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    is_error: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    role: FormationRole | None = None
    phase: str = ""


ROLE_PROMPTS: dict[FormationRole, str] = {
    FormationRole.PLANNER: (
        "You are the PLANNER in a Nemotron 3 formation. Analyze the request and produce "
        "a concrete, ordered execution plan.\n\n"
        "Produce numbered steps, each including:\n"
        "- The file path(s) to read or modify\n"
        "- The specific change or action\n"
        "- Risks or edge cases to watch for\n"
        "- Success criteria for that step\n\n"
        "Be specific -- the EXECUTOR follows your plan verbatim."
    ),
    FormationRole.EXECUTOR: (
        "You are NeMoCode, an expert coding assistant powered by NVIDIA Nemotron 3 Super.\n\n"
        "## Available Tools\n"
        "- read_file: Read file contents (with optional offset/limit)\n"
        "- write_file: Create or overwrite a file (requires confirmation)\n"
        "- edit_file: Search-and-replace edit on a file (old_string must be unique)\n"
        "- list_dir: List directory contents with optional depth\n"
        "- bash_exec: Execute a shell command (requires confirmation)\n"
        "- git_status: Show working tree status\n"
        "- git_diff: Show staged/unstaged changes\n"
        "- git_log: Show recent commit history\n"
        "- git_commit: Stage files and commit (requires confirmation)\n"
        "- search_files: Search file contents with regex (ripgrep)\n"
        "- http_fetch: Fetch content from a URL\n\n"
        "## Workflow\n"
        "1. UNDERSTAND -- Read relevant files before changing them.\n"
        "2. PLAN -- Think through your approach.\n"
        "3. IMPLEMENT -- Make precise, surgical edits.\n"
        "4. VERIFY -- Run tests/linters to validate.\n"
        "5. ITERATE -- If something fails, diagnose and fix.\n\n"
        "## Rules\n"
        "- YOU MUST use tool calls to interact with the codebase. "
        "Do NOT just describe what to do.\n"
        "- Always read a file before editing it.\n"
        "- Prefer edit_file over write_file for existing files.\n"
        "- Run tests after making changes.\n"
        "- If a command fails, read the error output carefully before retrying.\n"
    ),
    FormationRole.REVIEWER: (
        "You are the REVIEWER in a Nemotron 3 formation.\n\n"
        "Review the changes made by the executor. Check for:\n"
        "- Correctness: Does the code do what was requested?\n"
        "- Edge cases: Are boundary conditions handled?\n"
        "- Error handling: Are failures handled gracefully?\n"
        "- Test coverage: Are there tests for the new/changed code?\n"
        "- Style: Does the code follow project conventions?\n\n"
        "Output exactly one of:\n"
        "- APPROVE: If all checks pass, with a brief summary of what was done well.\n"
        "- REQUEST_CHANGES: If issues were found, with specific items to fix."
    ),
    FormationRole.DEBUGGER: (
        "You are the DEBUGGER. Analyze the error, read relevant source, identify root cause, "
        "suggest a specific fix."
    ),
    FormationRole.FAST: (
        "You are a fast execution agent powered by Nemotron 3 Nano. Handle targeted subtasks "
        "quickly and precisely. Focus on the specific task given -- do not over-think."
    ),
}


class Scheduler:
    def __init__(
        self,
        registry: Registry,
        tool_registry: ToolRegistry,
        confirm_fn: ConfirmFn | None = None,
        context_window: int = 1_048_576,
    ) -> None:
        self._reg = registry
        self._tools = tool_registry
        self._confirm_fn = confirm_fn
        self._sessions: dict[FormationRole, Session] = {}
        self._context_mgr = ContextManager(context_window=context_window)
        self._project_context: str = ""

    # ------------------------------------------------------------------
    # Single-model mode
    # ------------------------------------------------------------------

    async def run_single(self, endpoint_name: str, user_input: str) -> AsyncIterator[AgentEvent]:
        provider = self._reg.get_chat_provider(endpoint_name)
        if FormationRole.EXECUTOR not in self._sessions:
            s = Session(id="main", endpoint_name=endpoint_name)
            prompt = ROLE_PROMPTS[FormationRole.EXECUTOR]
            if self._project_context:
                prompt = f"{prompt}\n\n## Project Context\n{self._project_context}"
            s.add_system(prompt)
            self._sessions[FormationRole.EXECUTOR] = s
        session = self._sessions[FormationRole.EXECUTOR]
        session.add_user(user_input)
        async for ev in self._agent_loop(provider, session, FormationRole.EXECUTOR):
            yield ev

    # ------------------------------------------------------------------
    # Formation mode
    # ------------------------------------------------------------------

    async def run_formation(
        self, formation_name: str, user_input: str
    ) -> AsyncIterator[AgentEvent]:
        form = self._reg.get_formation(formation_name)
        by_role: dict[FormationRole, FormationSlot] = {s.role: s for s in form.slots}

        # Phase 1: Plan
        plan = ""
        if FormationRole.PLANNER in by_role:
            slot = by_role[FormationRole.PLANNER]
            yield AgentEvent(
                kind="phase", phase="planning", role=FormationRole.PLANNER, text="Planning..."
            )
            plan = await self._run_text_only(slot, user_input)
            yield AgentEvent(kind="text", text=plan, role=FormationRole.PLANNER, phase="planning")

        # Phase 2: Execute
        exec_role = (
            FormationRole.EXECUTOR if FormationRole.EXECUTOR in by_role else FormationRole.FAST
        )
        if exec_role in by_role:
            slot = by_role[exec_role]
            yield AgentEvent(kind="phase", phase="executing", role=exec_role, text="Executing...")
            provider = self._reg.get_chat_provider(slot.endpoint)
            session = self._get_session(exec_role, slot)
            inp = user_input
            if plan:
                inp = f"## User Request\n{user_input}\n\n## Plan\n{plan}\n\nFollow the plan."
            session.add_user(inp)
            async for ev in self._agent_loop(provider, session, exec_role):
                yield ev

        # Phase 3: Review
        if FormationRole.REVIEWER in by_role:
            for rnd in range(form.verification_rounds):
                slot = by_role[FormationRole.REVIEWER]
                yield AgentEvent(
                    kind="phase",
                    phase="reviewing",
                    role=FormationRole.REVIEWER,
                    text=f"Review round {rnd + 1}...",
                )
                provider = self._reg.get_chat_provider(slot.endpoint)

                exec_summary = ""
                for r in (FormationRole.EXECUTOR, FormationRole.FAST):
                    if r in self._sessions:
                        exec_summary = self._sessions[r].last_assistant_text()
                        break

                review_input = (
                    f"## Original Request\n{user_input}\n\n## Plan\n{plan or '(none)'}\n\n"
                    f"## Executor Output\n{exec_summary}\n\nReview the changes."
                )
                rev_session = Session(id=f"review-{rnd}", endpoint_name=slot.endpoint)
                rev_session.add_system(slot.system_prompt or ROLE_PROMPTS[FormationRole.REVIEWER])
                rev_session.add_user(review_input)

                review_text = ""
                async for ev in self._agent_loop(provider, rev_session, FormationRole.REVIEWER):
                    if ev.kind == "text":
                        review_text += ev.text
                    yield ev

                if "APPROVE" in review_text.upper():
                    yield AgentEvent(
                        kind="text",
                        text="\nChanges approved.\n",
                        role=FormationRole.REVIEWER,
                        phase="reviewing",
                    )
                    break
                elif exec_role in by_role:
                    yield AgentEvent(
                        kind="phase",
                        phase="executing",
                        role=exec_role,
                        text="Addressing review feedback...",
                    )
                    session = self._sessions[exec_role]
                    session.add_user(f"## Review Feedback\n{review_text}\n\nAddress these issues.")
                    exec_provider = self._reg.get_chat_provider(by_role[exec_role].endpoint)
                    async for ev in self._agent_loop(exec_provider, session, exec_role):
                        yield ev

    # ------------------------------------------------------------------
    # Agent loop — ReAct tool cycle
    # ------------------------------------------------------------------

    async def _agent_loop(
        self, provider: NIMChatProvider, session: Session, role: FormationRole
    ) -> AsyncIterator[AgentEvent]:
        schemas = self._tools.get_schemas()
        for _ in range(MAX_TOOL_ROUNDS):
            # Auto-compact if context usage exceeds 80%
            if self._context_mgr.should_compact(session.messages):
                logger.info("Auto-compacting session %s (context usage > 80%%)", session.id)
                session.compact()

            text_buf, think_buf = "", ""
            tool_calls: list[ToolCall] = []

            async for chunk in provider.stream(session.messages, tools=schemas or None):
                if chunk.text:
                    text_buf += chunk.text
                    yield AgentEvent(kind="text", text=chunk.text, role=role)
                if chunk.thinking:
                    think_buf += chunk.thinking
                    yield AgentEvent(kind="thinking", thinking=chunk.thinking, role=role)
                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
                if chunk.usage:
                    session.usage.add(
                        chunk.usage.get("prompt_tokens", 0), chunk.usage.get("completion_tokens", 0)
                    )
                    yield AgentEvent(kind="usage", usage=chunk.usage, role=role)

            if not tool_calls:
                session.add_assistant(
                    Message(role=Role.ASSISTANT, content=text_buf, thinking=think_buf)
                )
                return

            session.add_assistant(
                Message(
                    role=Role.ASSISTANT, content=text_buf, tool_calls=tool_calls, thinking=think_buf
                )
            )

            for tc in tool_calls:
                yield AgentEvent(
                    kind="tool_call", tool_name=tc.name, tool_args=tc.arguments, role=role
                )

                td = self._tools.get(tc.name)
                if td and td.requires_confirmation and self._confirm_fn:
                    if not await self._confirm_fn(tc.name, tc.arguments):
                        r = json.dumps({"error": "User denied"})
                        session.add_tool_result(tc.id, r, is_error=True)
                        yield AgentEvent(
                            kind="tool_result",
                            tool_name=tc.name,
                            tool_result=r,
                            is_error=True,
                            role=role,
                        )
                        continue

                # Execute with timeout to prevent hanging tools
                tool_errored = False
                try:
                    result = await asyncio.wait_for(
                        self._tools.execute(tc.name, tc.arguments),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    result = json.dumps({"error": f"Tool {tc.name} timed out"})
                    tool_errored = True
                except Exception as e:
                    result = json.dumps({"error": f"Tool {tc.name} failed: {e}"})
                    tool_errored = True

                if not tool_errored:
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and "error" in parsed:
                            tool_errored = True
                    except (json.JSONDecodeError, ValueError):
                        pass
                session.add_tool_result(tc.id, result, is_error=tool_errored)
                yield AgentEvent(
                    kind="tool_result",
                    tool_name=tc.name,
                    tool_result=result,
                    is_error=tool_errored,
                    role=role,
                )

        yield AgentEvent(
            kind="error", text=f"Hit max tool rounds ({MAX_TOOL_ROUNDS})", is_error=True, role=role
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run_text_only(self, slot: FormationSlot, user_input: str) -> str:
        provider = self._reg.get_chat_provider(slot.endpoint)
        msgs = [
            Message(
                role=Role.SYSTEM, content=slot.system_prompt or ROLE_PROMPTS.get(slot.role, "")
            ),
            Message(role=Role.USER, content=user_input),
        ]
        extra = {}
        if slot.reasoning_mode == "low_effort":
            manifest = self._reg.get_manifest_for_endpoint(slot.endpoint)
            if manifest and manifest.reasoning.supports_budget_control:
                budget_param = manifest.reasoning.thinking_budget_param or "reasoning_budget"
                extra = {"chat_template_kwargs": {budget_param: 1024}}
        r = await provider.complete(msgs, extra_body=extra or None)
        return r.content

    def _get_session(self, role: FormationRole, slot: FormationSlot) -> Session:
        if role not in self._sessions:
            s = Session(id=role.value, endpoint_name=slot.endpoint)
            prompt = slot.system_prompt or ROLE_PROMPTS.get(role, "")
            if self._project_context:
                prompt = f"{prompt}\n\n## Project Context\n{self._project_context}"
            s.add_system(prompt)
            self._sessions[role] = s
        return self._sessions[role]

    def reset(self) -> None:
        self._sessions.clear()

    def compact_all(self, keep: int = 20) -> None:
        for s in self._sessions.values():
            s.compact(keep)

    def inject_system_context(self, context: str) -> None:
        """Store project context to be injected when sessions are created."""
        self._project_context = context
        # Also inject into any already-existing sessions
        for role, session in self._sessions.items():
            if session.messages and session.messages[0].role == Role.SYSTEM:
                current = session.messages[0].content
                if "## Project Context" not in current:
                    session.add_system(f"{current}\n\n## Project Context\n{context}")
