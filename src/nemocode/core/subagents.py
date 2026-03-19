# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""In-memory lifecycle manager for delegated sub-agent runs."""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import asdict, dataclass
from typing import Any

_FINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class SubagentRun:
    id: str
    agent_name: str
    display_name: str
    nickname: str
    parent_agent: str
    task: str
    endpoint: str
    status: str
    tool_calls: int
    errors: int
    started_at: float
    context: str = ""
    session_id: str = ""
    finished_at: float | None = None
    output_preview: str = ""
    output: str = ""
    error: str = ""
    closed: bool = False
    previous_status: str = ""
    last_event_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _SubagentRuntime:
    run: SubagentRun
    task: asyncio.Task[Any] | None = None


_runs: list[SubagentRun] = []
_runtimes: dict[str, _SubagentRuntime] = {}
_run_counter = itertools.count(1)


def is_final_status(status: str) -> bool:
    return status in _FINAL_STATUSES


def reset_runs() -> None:
    """Reset the in-memory run ledger (for tests)."""
    for runtime in _runtimes.values():
        if runtime.task and not runtime.task.done():
            runtime.task.cancel()
    _runs.clear()
    _runtimes.clear()
    global _run_counter
    _run_counter = itertools.count(1)


def start_run(
    *,
    agent_name: str,
    display_name: str,
    nickname: str,
    parent_agent: str,
    task: str,
    endpoint: str,
    context: str = "",
    session_id: str = "",
) -> SubagentRun:
    run = SubagentRun(
        id=f"subagent-{next(_run_counter):04d}",
        agent_name=agent_name,
        display_name=display_name,
        nickname=nickname,
        parent_agent=parent_agent,
        task=task,
        endpoint=endpoint,
        status="running",
        tool_calls=0,
        errors=0,
        started_at=time.time(),
        context=context,
        session_id=session_id,
    )
    _runs.append(run)
    _runtimes[run.id] = _SubagentRuntime(run=run)
    return run


def attach_task(run_id: str, task: asyncio.Task[Any]) -> None:
    runtime = _runtimes.get(run_id)
    if runtime is None:
        return
    runtime.task = task


def append_output(run_id: str, text: str) -> None:
    run = get_run(run_id)
    if run is None or not text:
        return
    run.output += text
    run.output_preview = run.output[:500]
    run.last_event_at = time.time()


def record_tool_result(run_id: str, *, is_error: bool = False) -> None:
    run = get_run(run_id)
    if run is None:
        return
    run.tool_calls += 1
    if is_error:
        run.errors += 1
    run.last_event_at = time.time()


def complete_run(
    run_id: str,
    *,
    output_preview: str,
    tool_calls: int,
    errors: int,
    output: str | None = None,
) -> None:
    run = get_run(run_id)
    if run is None:
        return
    run.status = "completed"
    run.finished_at = time.time()
    run.output_preview = output_preview
    if output is not None:
        run.output = output
    run.tool_calls = tool_calls
    run.errors = errors


def fail_run(run_id: str, error: str) -> None:
    run = get_run(run_id)
    if run is None:
        return
    run.status = "failed"
    run.finished_at = time.time()
    run.error = error
    if not run.output_preview and run.output:
        run.output_preview = run.output[:500]


def cancel_run(run_id: str, error: str = "Cancelled") -> None:
    run = get_run(run_id)
    if run is None:
        return
    run.status = "cancelled"
    run.finished_at = time.time()
    run.error = error
    if not run.output_preview and run.output:
        run.output_preview = run.output[:500]


async def wait_for_run(
    run_id: str,
    timeout_s: float | None = None,
) -> tuple[SubagentRun | None, bool]:
    runtime = _runtimes.get(run_id)
    if runtime is None:
        return None, False

    task = runtime.task
    if task is None or task.done() or is_final_status(runtime.run.status):
        return runtime.run, False

    try:
        if timeout_s is None:
            await asyncio.shield(task)
            return runtime.run, False
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        return runtime.run, False
    except asyncio.TimeoutError:
        return runtime.run, True
    except asyncio.CancelledError:
        return runtime.run, False


async def close_run(
    run_id: str,
    *,
    cancel: bool = False,
) -> tuple[SubagentRun | None, str]:
    runtime = _runtimes.get(run_id)
    if runtime is None:
        return None, ""

    run = runtime.run
    previous_status = run.status
    run.previous_status = previous_status
    run.closed = True

    if cancel and runtime.task and not runtime.task.done():
        runtime.task.cancel()
        try:
            await runtime.task
        except asyncio.CancelledError:
            pass
        if run.status == "running":
            cancel_run(run.id, "Cancelled by close_agent")

    return run, previous_status


def resume_run(run_id: str) -> SubagentRun | None:
    run = get_run(run_id)
    if run is None:
        return None
    run.closed = False
    return run


def get_run(run_id: str) -> SubagentRun | None:
    runtime = _runtimes.get(run_id)
    if runtime is not None:
        return runtime.run
    for run in _runs:
        if run.id == run_id:
            return run
    return None


def list_runs(limit: int = 20) -> list[SubagentRun]:
    if limit <= 0:
        return []
    return list(reversed(_runs[-limit:]))


def snapshot_run(
    run: SubagentRun,
    *,
    include_output: bool = False,
    timed_out: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": run.id,
        "agent_type": run.agent_name,
        "display_name": run.display_name,
        "nickname": run.nickname,
        "status": run.status,
        "closed": run.closed,
        "parent_agent": run.parent_agent,
        "endpoint": run.endpoint,
        "task": run.task,
        "tool_calls": run.tool_calls,
        "errors": run.errors,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "output_preview": run.output_preview,
    }
    if run.context:
        data["context"] = run.context
    if run.session_id:
        data["session_id"] = run.session_id
    if run.previous_status:
        data["previous_status"] = run.previous_status
    if run.error:
        data["error"] = run.error
    if include_output:
        data["output"] = run.output
    if timed_out:
        data["timed_out"] = True
    return data
