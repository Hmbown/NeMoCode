# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the in-memory delegated sub-agent lifecycle manager."""

from __future__ import annotations

import asyncio

import pytest

from nemocode.core.subagents import (
    append_output,
    attach_task,
    close_run,
    complete_run,
    fail_run,
    get_run,
    list_runs,
    reset_runs,
    resume_run,
    start_run,
    wait_for_run,
)


class TestSubagentLedger:
    def setup_method(self):
        reset_runs()

    def teardown_method(self):
        reset_runs()

    def test_completed_and_failed_runs_are_tracked(self):
        run1 = start_run(
            agent_name="general",
            display_name="Joe Nemotron",
            nickname="Orbit Joe",
            parent_agent="build",
            task="Inspect the repo",
            endpoint="nim-nano",
        )
        complete_run(run1.id, output_preview="done", tool_calls=2, errors=0)

        run2 = start_run(
            agent_name="debug",
            display_name="Joebug Nemotron",
            nickname="Crash Carson",
            parent_agent="build",
            task="Debug the failure",
            endpoint="nim-super",
        )
        fail_run(run2.id, "boom")

        latest = list_runs()
        assert [run.id for run in latest] == [run2.id, run1.id]
        assert get_run(run1.id).status == "completed"
        assert get_run(run1.id).tool_calls == 2
        assert get_run(run2.id).status == "failed"
        assert get_run(run2.id).error == "boom"

    @pytest.mark.asyncio
    async def test_wait_for_run_observes_completion(self):
        run = start_run(
            agent_name="general",
            display_name="Joe Nemotron",
            nickname="Orbit Joe",
            parent_agent="build",
            task="Inspect the repo",
            endpoint="nim-nano",
        )

        async def _finish():
            await asyncio.sleep(0.01)
            append_output(run.id, "done")
            complete_run(run.id, output_preview="done", output="done", tool_calls=1, errors=0)

        task = asyncio.create_task(_finish())
        attach_task(run.id, task)

        waited, timed_out = await wait_for_run(run.id, timeout_s=1.0)
        assert timed_out is False
        assert waited is not None
        assert waited.status == "completed"
        assert waited.output == "done"

    @pytest.mark.asyncio
    async def test_close_run_can_cancel_active_task(self):
        run = start_run(
            agent_name="debug",
            display_name="Joebug Nemotron",
            nickname="Crash Carson",
            parent_agent="build",
            task="Debug the failure",
            endpoint="nim-super",
        )

        async def _hang():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(_hang())
        attach_task(run.id, task)

        closed, previous_status = await close_run(run.id, cancel=True)
        assert closed is not None
        assert previous_status == "running"
        assert closed.closed is True
        assert closed.status == "cancelled"

    def test_resume_run_reopens_closed_handle(self):
        run = start_run(
            agent_name="review",
            display_name="Giovannemotron",
            nickname="Sheriff Diff",
            parent_agent="build",
            task="Review the diff",
            endpoint="nim-super",
        )
        run.closed = True

        resumed = resume_run(run.id)
        assert resumed is not None
        assert resumed.closed is False
