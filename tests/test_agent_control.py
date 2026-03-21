# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for spawned sub-agent lifecycle tools."""

from __future__ import annotations

import asyncio
import json

import pytest

import nemocode.tools.delegate as delegate_mod
from nemocode.config import _parse_config
from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    NeMoCodeConfig,
    ToolPermissions,
)
from nemocode.core.registry import Registry
from nemocode.core.subagents import append_output, complete_run, reset_runs
from nemocode.tools.delegate import create_delegate_tools


@pytest.fixture
def orchestration_config() -> NeMoCodeConfig:
    return NeMoCodeConfig(
        default_endpoint="nim-super",
        endpoints={
            "nim-super": Endpoint(
                name="nim-super",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                capabilities=[Capability.CHAT],
            ),
        },
        agents=_parse_config({}).agents,
        permissions=ToolPermissions(),
    )


def _tool_fns(config: NeMoCodeConfig) -> dict[str, object]:
    registry = Registry(config)
    return {
        td.name: td.fn for td in create_delegate_tools(registry, config, parent_agent_name="build")
    }


class TestAgentControlTools:
    def setup_method(self):
        reset_runs()

    def teardown_method(self):
        reset_runs()

    @pytest.mark.asyncio
    async def test_spawn_wait_close_and_resume(self, orchestration_config, monkeypatch):
        async def _fake_drive(run_id, scheduler, endpoint, user_msg):
            append_output(run_id, f"finished: {user_msg}")
            complete_run(
                run_id,
                output_preview=f"finished: {user_msg}",
                output=f"finished: {user_msg}",
                tool_calls=2,
                errors=0,
            )

        monkeypatch.setattr("nemocode.tools.delegate._drive_subagent", _fake_drive)
        tools = _tool_fns(orchestration_config)

        spawned = json.loads(await tools["spawn_agent"](task="Inspect src/", agent_type="general"))
        assert spawned["status"] == "running"
        assert spawned["parent_agent"] == "build"

        waited = json.loads(await tools["wait_agent"](agent_id=spawned["run_id"], timeout_s=1.0))
        assert waited["status"] == "completed"
        assert waited["output"].startswith("finished:")
        assert waited["tool_calls"] == 2

        closed = json.loads(await tools["close_agent"](agent_id=spawned["run_id"]))
        assert closed["closed"] is True
        assert closed["previous_status"] == "completed"

        resumed = json.loads(await tools["resume_agent"](agent_id=spawned["run_id"]))
        assert resumed["closed"] is False
        assert resumed["status"] == "completed"

    @pytest.mark.asyncio
    async def test_delegate_waits_for_final_result(self, orchestration_config, monkeypatch):
        async def _fake_drive(run_id, scheduler, endpoint, user_msg):
            append_output(run_id, "delegated answer")
            complete_run(
                run_id,
                output_preview="delegated answer",
                output="delegated answer",
                tool_calls=1,
                errors=0,
            )

        monkeypatch.setattr("nemocode.tools.delegate._drive_subagent", _fake_drive)
        tools = _tool_fns(orchestration_config)

        result = json.loads(await tools["delegate"](task="Check the repo", agent_type="explore"))
        assert result["status"] == "completed"
        assert result["output"] == "delegated answer"

    @pytest.mark.asyncio
    async def test_wait_agent_times_out_without_cancelling(self, orchestration_config, monkeypatch):
        async def _slow_drive(run_id, scheduler, endpoint, user_msg):
            await asyncio.sleep(0.05)
            append_output(run_id, "done later")
            complete_run(
                run_id,
                output_preview="done later",
                output="done later",
                tool_calls=0,
                errors=0,
            )

        monkeypatch.setattr("nemocode.tools.delegate._drive_subagent", _slow_drive)
        tools = _tool_fns(orchestration_config)

        spawned = json.loads(await tools["spawn_agent"](task="Wait a bit", agent_type="test"))
        waited = json.loads(await tools["wait_agent"](agent_id=spawned["run_id"], timeout_s=0.01))
        assert waited["status"] == "running"
        assert waited["timed_out"] is True

        final = json.loads(await tools["wait_agent"](agent_id=spawned["run_id"], timeout_s=1.0))
        assert final["status"] == "completed"
        assert final["output"] == "done later"

    @pytest.mark.asyncio
    async def test_read_only_parent_restricts_spawned_subagent_tools(
        self,
        orchestration_config,
        monkeypatch,
    ):
        captured: dict[str, object] = {}
        real_load_tools = delegate_mod.load_tools

        def _capture_load_tools(categories=None):
            captured["categories"] = list(categories) if categories is not None else None
            return real_load_tools(categories)

        async def _fake_drive(run_id, scheduler, endpoint, user_msg):
            captured["scheduler_read_only"] = scheduler._read_only
            append_output(run_id, "done")
            complete_run(
                run_id,
                output_preview="done",
                output="done",
                tool_calls=0,
                errors=0,
            )

        monkeypatch.setattr("nemocode.tools.delegate.load_tools", _capture_load_tools)
        monkeypatch.setattr("nemocode.tools.delegate._drive_subagent", _fake_drive)

        registry = Registry(orchestration_config)
        tools = {
            td.name: td.fn
            for td in create_delegate_tools(
                registry,
                orchestration_config,
                parent_agent_name="plan",
                parent_read_only=True,
            )
        }

        spawned = json.loads(
            await tools["spawn_agent"](task="Run the test suite", agent_type="test")
        )
        final = json.loads(await tools["wait_agent"](agent_id=spawned["run_id"], timeout_s=1.0))

        assert final["status"] == "completed"
        # Subagents get their own tools regardless of parent read-only state
        assert captured["categories"] == ["bash", "fs_read"]
        assert captured["scheduler_read_only"] is False
