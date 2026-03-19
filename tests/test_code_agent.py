# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for planner-mode orchestration in CodeAgent."""

from __future__ import annotations

import pytest

from nemocode.config import _parse_config
from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    NeMoCodeConfig,
    ToolPermissions,
)
from nemocode.core.scheduler import AgentEvent
from nemocode.workflows.code_agent import CodeAgent


@pytest.fixture
def plan_config() -> NeMoCodeConfig:
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
        formations={
            "plan_execute": Formation(
                name="Plan Execute",
                description="Planner and executor formation",
                slots=[FormationSlot(endpoint="nim-super", role=FormationRole.EXECUTOR)],
            ),
        },
        agents=_parse_config({}).agents,
        permissions=ToolPermissions(),
    )


class TestCodeAgentPlanMode:
    @pytest.mark.asyncio
    async def test_plan_mode_revises_then_executes(self, plan_config):
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        prompts: list[tuple[str, str]] = []
        plan_outputs = iter(["Plan v1", "Plan v2"])
        decisions = iter([("revise", "Add tests"), ("approve", "")])
        executions: list[tuple[str, str]] = []

        async def _fake_run_single(endpoint: str, user_input: str):
            prompts.append((endpoint, user_input))
            yield AgentEvent(kind="text", text=next(plan_outputs), role=FormationRole.PLANNER)

        async def _fake_request_plan_decision(plan_text: str) -> tuple[str, str]:
            return next(decisions)

        async def _fake_run_approved_plan(user_input: str, plan_text: str):
            executions.append((user_input, plan_text))
            yield AgentEvent(kind="text", text="executed", role=FormationRole.EXECUTOR)

        agent._scheduler.run_single = _fake_run_single
        agent._request_plan_decision = _fake_request_plan_decision
        agent._run_approved_plan = _fake_run_approved_plan

        events = [event async for event in agent.run("Ship the release")]

        assert prompts[0] == ("nim-super", "Ship the release")
        assert prompts[1][0] == "nim-super"
        assert "## Previous Plan\nPlan v1" in prompts[1][1]
        assert "## User Feedback\nAdd tests" in prompts[1][1]
        assert executions == [("Ship the release", "Plan v2")]
        assert any(event.kind == "status" and "Revising plan" in event.text for event in events)
        assert any(event.kind == "text" and event.text == "executed" for event in events)

    @pytest.mark.asyncio
    async def test_plan_mode_runs_before_active_formation(self, plan_config):
        plan_config.active_formation = "plan_execute"
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        calls: list[tuple[str, str]] = []

        async def _fake_run_plan_mode(endpoint: str, user_input: str):
            calls.append((endpoint, user_input))
            yield AgentEvent(kind="text", text="planned", role=FormationRole.PLANNER)

        async def _unexpected_run_formation(*args, **kwargs):
            raise AssertionError("run_formation should not be used before plan approval")
            yield AgentEvent(kind="text", text="")

        agent._run_plan_mode = _fake_run_plan_mode
        agent._scheduler.run_formation = _unexpected_run_formation

        events = [event async for event in agent.run("Prepare release notes")]

        assert calls == [("nim-super", "Prepare release notes")]
        assert [event.text for event in events if event.kind == "text"] == ["planned"]
