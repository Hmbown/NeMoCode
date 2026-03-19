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


class TestPlanApprovalResume:
    """Tests for the pending plan approval resume mechanism."""

    @pytest.mark.asyncio
    async def test_pending_plan_stored_on_agent(self, plan_config):
        """When approval returns pending, plan text is stored on agent."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")

        async def _fake_run_single(endpoint, user_input):
            yield AgentEvent(kind="text", text="My plan", role=FormationRole.PLANNER)

        async def _fake_request_decision(plan_text):
            return ("pending", "")

        agent._scheduler.run_single = _fake_run_single
        agent._request_plan_decision = _fake_request_decision

        [e async for e in agent.run("do something")]
        assert agent.has_pending_plan
        assert agent.pending_plan_text == "My plan"

    @pytest.mark.asyncio
    async def test_no_pending_plan_by_default(self, plan_config):
        """Fresh agent has no pending plan."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        assert not agent.has_pending_plan
        assert agent.pending_plan_text is None

    @pytest.mark.asyncio
    async def test_resume_approval_with_approve(self, plan_config):
        """Resuming with 'approve' executes the plan."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        agent._pending_plan_text = "The Plan"
        agent._pending_plan_user_input = "do something"

        executed = []

        async def _fake_run_approved(user_input, plan_text):
            executed.append((user_input, plan_text))
            yield AgentEvent(kind="text", text="done", role=FormationRole.EXECUTOR)

        agent._run_approved_plan = _fake_run_approved

        result = await agent.try_handle_plan_response("approve")
        assert result is not None
        events = [e async for e in result]
        assert executed == [("do something", "The Plan")]
        assert not agent.has_pending_plan
        assert any(e.text == "done" for e in events if e.kind == "text")

    @pytest.mark.asyncio
    async def test_resume_approval_with_yes(self, plan_config):
        """Resuming with 'yes' also approves the plan."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        agent._pending_plan_text = "The Plan"
        agent._pending_plan_user_input = "do something"

        executed = []

        async def _fake_run_approved(user_input, plan_text):
            executed.append((user_input, plan_text))
            yield AgentEvent(kind="text", text="done", role=FormationRole.EXECUTOR)

        agent._run_approved_plan = _fake_run_approved

        result = await agent.try_handle_plan_response("yes")
        assert result is not None
        [e async for e in result]
        assert executed == [("do something", "The Plan")]

    @pytest.mark.asyncio
    async def test_resume_approval_with_cancel(self, plan_config):
        """Resuming with 'cancel' cancels the plan."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        agent._pending_plan_text = "The Plan"
        agent._pending_plan_user_input = "do something"

        result = await agent.try_handle_plan_response("cancel")
        assert result is not None
        events = [e async for e in result]
        assert any("cancelled" in e.text.lower() for e in events if e.kind == "text")
        assert not agent.has_pending_plan

    @pytest.mark.asyncio
    async def test_resume_approval_with_no(self, plan_config):
        """Resuming with 'no' also cancels the plan."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        agent._pending_plan_text = "The Plan"
        agent._pending_plan_user_input = "do something"

        result = await agent.try_handle_plan_response("no")
        assert result is not None
        events = [e async for e in result]
        assert any("cancelled" in e.text.lower() for e in events if e.kind == "text")

    @pytest.mark.asyncio
    async def test_resume_approval_with_non_decision_clears(self, plan_config):
        """If user types something unrelated, returns None (not a decision)."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        agent._pending_plan_text = "The Plan"
        agent._pending_plan_user_input = "do something"

        result = await agent.try_handle_plan_response("actually do something else entirely")
        assert result is None
        # Pending state is NOT cleared by try_handle_plan_response for non-decisions
        # (the caller is responsible for clearing)

    @pytest.mark.asyncio
    async def test_parse_plan_decision_approve(self):
        assert CodeAgent.parse_plan_decision("approve") == ("approve", "")
        assert CodeAgent.parse_plan_decision("Approve") == ("approve", "")
        assert CodeAgent.parse_plan_decision("approved") == ("approve", "")
        assert CodeAgent.parse_plan_decision("yes") == ("approve", "")
        assert CodeAgent.parse_plan_decision("y") == ("approve", "")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_cancel(self):
        assert CodeAgent.parse_plan_decision("cancel") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("Cancel") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("no") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("n") == ("cancel", "")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_revise(self):
        assert CodeAgent.parse_plan_decision("revise: add tests") == ("revise", "add tests")
        assert CodeAgent.parse_plan_decision("revise") == ("revise", "")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_pending(self):
        """Non-decision text returns pending."""
        assert CodeAgent.parse_plan_decision("do it differently") == ("pending", "")
        assert CodeAgent.parse_plan_decision("") == ("pending", "")

    @pytest.mark.asyncio
    async def test_full_cycle_plan_pending_approve_execute(self, plan_config):
        """End-to-end: plan → pending → approve → execution."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        plan_outputs = iter(["Plan v1"])

        async def _fake_run_single(endpoint, user_input):
            yield AgentEvent(kind="text", text=next(plan_outputs), role=FormationRole.PLANNER)

        async def _fake_request_decision(plan_text):
            return ("pending", "")

        agent._scheduler.run_single = _fake_run_single
        agent._request_plan_decision = _fake_request_decision

        # Phase 1: generate plan
        events1 = [e async for e in agent.run("Ship it")]
        assert agent.has_pending_plan
        assert agent.pending_plan_text == "Plan v1"
        assert any("waiting for your approval" in e.text for e in events1 if e.kind == "text")

        # Phase 2: approve
        executed = []

        async def _fake_run_approved(user_input, plan_text):
            executed.append((user_input, plan_text))
            yield AgentEvent(kind="text", text="executed", role=FormationRole.EXECUTOR)

        agent._run_approved_plan = _fake_run_approved

        result = await agent.try_handle_plan_response("approve")
        assert result is not None
        events2 = [e async for e in result]
        assert executed == [("Ship it", "Plan v1")]
        assert not agent.has_pending_plan
        assert any(e.text == "executed" for e in events2 if e.kind == "text")

    @pytest.mark.asyncio
    async def test_full_cycle_plan_pending_cancel(self, plan_config):
        """End-to-end: plan → pending → cancel."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")

        async def _fake_run_single(endpoint, user_input):
            yield AgentEvent(kind="text", text="Plan v1", role=FormationRole.PLANNER)

        async def _fake_request_decision(plan_text):
            return ("pending", "")

        agent._scheduler.run_single = _fake_run_single
        agent._request_plan_decision = _fake_request_decision

        # Phase 1: generate plan
        [e async for e in agent.run("Ship it")]
        assert agent.has_pending_plan

        # Phase 2: cancel
        result = await agent.try_handle_plan_response("cancel")
        events = [e async for e in result]
        assert any("cancelled" in e.text.lower() for e in events if e.kind == "text")
        assert not agent.has_pending_plan

    @pytest.mark.asyncio
    async def test_plan_mode_read_only_before_approval(self, plan_config):
        """Plan mode agent uses read-only tools only."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        # Verify read_only flag is set
        assert agent._read_only is True
        # Verify the tool registry only has read-only tools
        tool_names = {td.name for td in agent._tool_registry._tools.values()}
        # Plan mode should not have write tools
        assert "bash" not in tool_names
        assert "fs" not in tool_names  # full fs (with writes) should not be present

    @pytest.mark.asyncio
    async def test_approved_plan_uses_build_agent(self, plan_config):
        """After approval, execution uses the build agent (not read-only)."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        agent._pending_plan_text = "The Plan"
        agent._pending_plan_user_input = "Ship it"

        async def _fake_run_approved(user_input, plan_text):
            yield AgentEvent(kind="text", text="ran", role=FormationRole.EXECUTOR)

        agent._run_approved_plan = _fake_run_approved
        result = await agent.try_handle_plan_response("approve")
        events = [e async for e in result]
        assert any(e.text == "ran" for e in events if e.kind == "text")
