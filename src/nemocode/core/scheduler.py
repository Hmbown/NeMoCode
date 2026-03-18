# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Scheduler — drives formations through their pipeline.

Handles planner -> executor -> reviewer loop, tool execution,
confirmation gates, and the Super + Nano dispatch pattern.
Integrates context window management for automatic compaction.
Supports parallel tool execution when multiple tool calls arrive
and none require user confirmation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from nemocode.config.schema import FormationRole, FormationSlot
from nemocode.core.context import ContextManager
from nemocode.core.registry import Registry
from nemocode.core.sessions import Session
from nemocode.core.streaming import Message, Role, ToolCall
from nemocode.providers.nim_chat import NIMChatProvider
from nemocode.tools import ToolRegistry

logger = logging.getLogger(__name__)
DEFAULT_MAX_TOOL_ROUNDS = 100
ConfirmFn = Callable[[str, dict[str, Any]], Awaitable[bool]]

# Tool output truncation threshold (characters)
_TRUNCATION_THRESHOLD = 20_000
_SCRATCH_DIR = Path(os.environ.get("NEMOCODE_SCRATCH_DIR", "/tmp/nemocode-scratch"))


# ---------------------------------------------------------------------------
# Stagnation detection
# ---------------------------------------------------------------------------


@dataclass
class _StagnationTracker:
    """Detects doom loops: repeated identical tool calls or errors.

    Deliberately permissive — LLMs routinely take 20+ tool rounds on
    complex tasks. Only fire on truly repetitive patterns.
    """

    _call_hashes: list[str] = field(default_factory=list)
    _error_hashes: list[str] = field(default_factory=list)
    _idle_count: int = 0  # rounds with ZERO activity (no text, no tools)
    _last_tool_name: str = ""  # name of last recorded tool (for messages)
    _last_error_preview: str = ""  # truncated preview of last error

    # Thresholds — generous to avoid false positives
    repeat_threshold: int = 5  # same exact tool call 5 times → stagnation
    error_threshold: int = 5  # same exact error 5 times → stagnation
    idle_threshold: int = 15  # 15 rounds with no text AND no tools → stagnation

    def record_tool_call(self, name: str, args: dict) -> None:
        h = hashlib.md5(f"{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()[:12]
        self._call_hashes.append(h)
        self._last_tool_name = name
        if len(self._call_hashes) > 50:
            self._call_hashes = self._call_hashes[-50:]
        # Tool activity = progress
        self._idle_count = 0

    def record_error(self, error_text: str) -> None:
        h = hashlib.md5(error_text.encode()).hexdigest()[:12]
        self._error_hashes.append(h)
        self._last_error_preview = error_text[:80].replace("\n", " ")
        if len(self._error_hashes) > 50:
            self._error_hashes = self._error_hashes[-50:]

    def record_turn(self, text: str, had_tool_calls: bool = False) -> None:
        """Record a turn's output. Only idle rounds (no text AND no tools) count."""
        if text.strip() or had_tool_calls:
            self._idle_count = 0
        else:
            self._idle_count += 1

    def is_stagnant(self) -> str | None:
        """Check for stagnation. Returns a descriptive reason string or None."""
        # Check for repeated identical tool calls
        if len(self._call_hashes) >= self.repeat_threshold:
            recent = self._call_hashes[-self.repeat_threshold :]
            if len(set(recent)) == 1:
                last_name = self._last_tool_name or "unknown tool"
                return (
                    f"Stuck calling {last_name} on the same arguments "
                    f"{self.repeat_threshold} times. "
                    f"Try rephrasing or use /reset."
                )

        # Check for repeated identical errors
        if len(self._error_hashes) >= self.error_threshold:
            recent = self._error_hashes[-self.error_threshold :]
            if len(set(recent)) == 1:
                preview = self._last_error_preview or "same error"
                return (
                    f"Same error repeated {self.error_threshold} times: "
                    f"{preview}. "
                    f"Fix the underlying issue or try a different approach."
                )

        # Check for total idleness (no text, no tools)
        if self._idle_count >= self.idle_threshold:
            return (
                f"No activity in {self.idle_threshold} rounds — the model "
                f"is not producing text or calling tools. "
                f"Try rephrasing or use /reset."
            )

        return None

    def reset(self) -> None:
        self._call_hashes.clear()
        self._error_hashes.clear()
        self._idle_count = 0
        self._last_tool_name = ""
        self._last_error_preview = ""


def _truncate_output(result: str, tool_name: str) -> str:
    """Truncate large tool output, saving full content to a scratch file.

    For structured JSON results (bash_exec etc.), truncates the inner
    stdout/stderr fields rather than the outer JSON, preserving parseability.
    """
    if len(result) <= _TRUNCATION_THRESHOLD:
        return result

    _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    scratch_file = _SCRATCH_DIR / f"{tool_name}_{ts}.txt"
    scratch_file.write_text(result)

    # Try to truncate structured JSON content (bash_exec, etc.)
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            changed = False
            for key in ("stdout", "stderr", "output"):
                val = parsed.get(key)
                if isinstance(val, str) and len(val) > _TRUNCATION_THRESHOLD:
                    original_len = len(parsed[key])
                    parsed[key] = (
                        parsed[key][:_TRUNCATION_THRESHOLD]
                        + f"\n\n... (truncated at {_TRUNCATION_THRESHOLD:,} chars, "
                        f"full output: {original_len:,} chars saved to {scratch_file})"
                    )
                    changed = True
            if changed:
                return json.dumps(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: raw truncation
    truncated = result[:_TRUNCATION_THRESHOLD]
    return (
        f"{truncated}\n\n"
        f"... (output truncated at {_TRUNCATION_THRESHOLD:,} chars, "
        f"full output: {len(result):,} chars saved to {scratch_file})"
    )


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
        "You are NeMoCode, an autonomous coding agent powered by NVIDIA Nemotron 3.\n\n"
        "You COMPLETE tasks — you don't just describe them. When the user asks you "
        "to do something, you DO it: read files, write code, run tests, fix errors, "
        "and iterate until the task is done.\n\n"
        "## Workflow\n"
        "1. FIND relevant files with glob_files or search_files\n"
        "2. READ what you need with read_file\n"
        "3. IMPLEMENT changes with edit_file (preferred) or write_file (new files)\n"
        "4. For multiple edits in one file, use multi_edit instead of repeated edit_file calls\n"
        "5. VERIFY by running tests: bash_exec with pytest, ruff, etc.\n"
        "6. FIX any failures — read the error, edit the code, re-run\n"
        "7. REPORT what you did (brief summary, not a lecture)\n\n"
        "## Tools\n"
        "- glob_files: Find files by pattern (e.g. '**/*.py'). Use this FIRST to locate files.\n"
        "- search_files: Search file contents with regex. Great for finding usages.\n"
        "- read_file: Read file contents. Read before editing.\n"
        "- edit_file: Search-and-replace edit. Preferred for existing files.\n"
        "- multi_edit: Apply multiple edits to one file atomically. Use when making 2+ changes.\n"
        "- write_file: Create new files or full rewrites.\n"
        "- bash_exec: Run shell commands (tests, builds, git, etc.).\n"
        "- ask_user: Ask the user a question when you need clarification.\n\n"
        "## Rules\n"
        "- ALWAYS use tools — never just describe changes.\n"
        "- Read a file before editing it.\n"
        "- Keep text responses SHORT. The code speaks for itself.\n"
        "- Do not ask for permission — just do the work.\n"
        "- If something is unclear, use ask_user to get clarification.\n"
        "- Batch operations efficiently — use multi_edit for multiple changes in one file.\n"
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


_PLAN_MODE_PROMPT = (
    "You are NeMoCode, an expert coding assistant powered by NVIDIA Nemotron 3 Super.\n\n"
    "You are in PLAN MODE (read-only). You can read files and explore the "
    "codebase, but you CANNOT modify anything.\n\n"
    "## Available Tools (read-only)\n"
    "- read_file: Read file contents\n"
    "- list_dir: List directory contents\n"
    "- git_status: Show working tree status\n"
    "- git_diff: Show staged/unstaged changes\n"
    "- git_log: Show recent commit history\n"
    "- search_files: Search file contents with regex\n\n"
    "## Your Role\n"
    "Discuss, analyze, and plan. Read code to understand it. "
    "Propose changes but do NOT attempt to write, edit, or execute anything. "
    "Produce clear, actionable plans that the user can implement or "
    "switch to code/auto mode to execute.\n"
)


class Scheduler:
    def __init__(
        self,
        registry: Registry,
        tool_registry: ToolRegistry,
        confirm_fn: ConfirmFn | None = None,
        context_window: int = 1_048_576,
        hook_runner: Any = None,
        read_only: bool = False,
        permission_engine: Any = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> None:
        self._reg = registry
        self._tools = tool_registry
        self._confirm_fn = confirm_fn
        self._sessions: dict[FormationRole, Session] = {}
        self._context_mgr = ContextManager(context_window=context_window)
        self._project_context: str = ""
        self._hook_runner = hook_runner
        self._read_only = read_only
        self._permission_engine = permission_engine
        self._max_tool_rounds = max_tool_rounds
        self._stagnation = _StagnationTracker()

    # ------------------------------------------------------------------
    # Single-model mode
    # ------------------------------------------------------------------

    async def run_single(self, endpoint_name: str, user_input: str) -> AsyncIterator[AgentEvent]:
        provider = self._reg.get_chat_provider(endpoint_name)
        if FormationRole.EXECUTOR not in self._sessions:
            s = Session(id="main", endpoint_name=endpoint_name)
            prompt = _PLAN_MODE_PROMPT if self._read_only else ROLE_PROMPTS[FormationRole.EXECUTOR]
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
                async for ev in self._agent_loop(
                    provider, rev_session, FormationRole.REVIEWER, use_tools=False
                ):
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
        self, provider: NIMChatProvider, session: Session, role: FormationRole,
        *, use_tools: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        schemas = self._tools.get_schemas() if use_tools else []
        self._stagnation.reset()

        for _ in range(self._max_tool_rounds):
            # Check stagnation before each round
            reason = self._stagnation.is_stagnant()
            if reason:
                yield AgentEvent(
                    kind="error",
                    text=f"Stagnation detected: {reason}. Stopping to avoid a doom loop. "
                    f"Try rephrasing your request or use /reset.",
                    is_error=True,
                    role=role,
                )
                return

            # Auto-compact if context usage exceeds 80% — use smart compaction
            if self._context_mgr.should_compact(session.messages):
                logger.info("Auto-compacting session %s (context usage > 80%%)", session.id)
                session.messages = self._context_mgr.smart_compact(session.messages)

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

            # Track progress — tool-using rounds count as active
            self._stagnation.record_turn(text_buf, had_tool_calls=bool(tool_calls))

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

            # Track tool calls for stagnation detection
            for tc in tool_calls:
                self._stagnation.record_tool_call(tc.name, tc.arguments)

            # Decide: parallel or sequential execution
            needs_confirm = any(self._needs_confirmation(tc) for tc in tool_calls)

            if len(tool_calls) > 1 and not needs_confirm:
                # Parallel execution — yield all calls, execute concurrently, yield all results
                async for ev in self._execute_parallel(tool_calls, session, role):
                    yield ev
            else:
                # Sequential execution
                async for ev in self._execute_sequential(tool_calls, session, role):
                    yield ev

        yield AgentEvent(
            kind="error",
            text=f"Hit max tool rounds ({self._max_tool_rounds})",
            is_error=True,
            role=role,
        )

    def _needs_confirmation(self, tc: ToolCall) -> bool:
        """Check if a tool call needs confirmation, respecting permission rules."""
        td = self._tools.get(tc.name)
        if not td or not td.requires_confirmation:
            return False

        # Check permission engine for auto-approve rules
        if self._permission_engine:
            decision = self._permission_engine.should_auto_approve(tc.name, tc.arguments)
            if decision is True:
                return False
            if decision is False:
                return True  # explicit deny — always confirm

        return True

    # ------------------------------------------------------------------
    # Tool execution strategies
    # ------------------------------------------------------------------

    async def _execute_sequential(
        self, tool_calls: list[ToolCall], session: Session, role: FormationRole
    ) -> AsyncIterator[AgentEvent]:
        """Execute tool calls one at a time (used when confirmation is needed)."""
        for tc in tool_calls:
            yield AgentEvent(kind="tool_call", tool_name=tc.name, tool_args=tc.arguments, role=role)

            if self._needs_confirmation(tc) and self._confirm_fn:
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

            result, is_error = await self._run_tool(tc)

            # Track errors for stagnation detection
            if is_error:
                self._stagnation.record_error(result[:200])

            # Truncate large outputs
            result = _truncate_output(result, tc.name)

            session.add_tool_result(tc.id, result, is_error=is_error)
            yield AgentEvent(
                kind="tool_result",
                tool_name=tc.name,
                tool_result=result,
                is_error=is_error,
                role=role,
            )

    async def _execute_parallel(
        self, tool_calls: list[ToolCall], session: Session, role: FormationRole
    ) -> AsyncIterator[AgentEvent]:
        """Execute tool calls concurrently (used when no confirmation needed)."""
        # Yield all tool_call events up front
        for tc in tool_calls:
            yield AgentEvent(kind="tool_call", tool_name=tc.name, tool_args=tc.arguments, role=role)

        # Execute all tools concurrently
        results = await asyncio.gather(
            *(self._run_tool(tc) for tc in tool_calls),
            return_exceptions=True,
        )

        # Yield all results
        for tc, result_or_exc in zip(tool_calls, results):
            if isinstance(result_or_exc, BaseException):
                result = json.dumps({"error": f"Tool {tc.name} failed: {result_or_exc}"})
                is_error = True
            else:
                result, is_error = result_or_exc

            # Track errors for stagnation detection
            if is_error:
                self._stagnation.record_error(result[:200])

            # Truncate large outputs
            result = _truncate_output(result, tc.name)

            session.add_tool_result(tc.id, result, is_error=is_error)
            yield AgentEvent(
                kind="tool_result",
                tool_name=tc.name,
                tool_result=result,
                is_error=is_error,
                role=role,
            )

    async def _run_tool(self, tc: ToolCall) -> tuple[str, bool]:
        """Execute a single tool call with timeout and hooks. Returns (result, is_error)."""
        # Pre-hook
        if self._hook_runner and self._hook_runner.enabled:
            try:
                await self._hook_runner.run_pre(tc.name, tc.arguments)
            except Exception as e:
                logger.debug("Pre-hook failed for %s: %s", tc.name, e)

        try:
            result = await asyncio.wait_for(
                self._tools.execute(tc.name, tc.arguments),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            return json.dumps({"error": f"Tool {tc.name} timed out"}), True
        except Exception as e:
            return json.dumps({"error": f"Tool {tc.name} failed: {e}"}), True

        is_error = False
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                is_error = True
        except (json.JSONDecodeError, ValueError):
            pass

        # Post-hook
        if self._hook_runner and self._hook_runner.enabled:
            try:
                await self._hook_runner.run_post(tc.name, tc.arguments, result)
            except Exception as e:
                logger.debug("Post-hook failed for %s: %s", tc.name, e)

        return result, is_error

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
