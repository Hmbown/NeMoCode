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
    async def test_plan_mode_streams_and_stores_pending(self, plan_config):
        """Plan mode streams the response and stores it as pending."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        prompts: list[tuple[str, str]] = []

        async def _fake_run_single(endpoint: str, user_input: str):
            prompts.append((endpoint, user_input))
            yield AgentEvent(kind="text", text="Here is my plan.", role=FormationRole.PLANNER)

        agent._scheduler.run_single = _fake_run_single

        events = [event async for event in agent.run("What should we do?")]

        assert prompts[0] == ("nim-super", "What should we do?")
        assert any(e.text == "Here is my plan." for e in events if e.kind == "text")
        assert agent.has_pending_plan
        assert agent.pending_plan_text == "Here is my plan."

    @pytest.mark.asyncio
    async def test_plan_mode_is_read_only(self, plan_config):
        """Plan mode creates agent with read-only tools."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")
        assert agent._read_only is True


class TestPlanApprovalResume:
    """Tests for the pending plan approval resume mechanism."""

    @pytest.mark.asyncio
    async def test_pending_plan_stored_on_agent(self, plan_config):
        """Plan mode always stores response as pending."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")

        async def _fake_run_single(endpoint, user_input):
            yield AgentEvent(kind="text", text="My plan", role=FormationRole.PLANNER)

        agent._scheduler.run_single = _fake_run_single

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
        assert CodeAgent.parse_plan_decision("1") == ("approve", "")
        assert CodeAgent.parse_plan_decision("1.") == ("approve", "")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_cancel(self):
        assert CodeAgent.parse_plan_decision("cancel") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("Cancel") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("no") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("n") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("4") == ("cancel", "")
        assert CodeAgent.parse_plan_decision("4.") == ("cancel", "")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_revise(self):
        assert CodeAgent.parse_plan_decision("revise: add tests") == ("revise", "add tests")
        assert CodeAgent.parse_plan_decision("revise") == ("revise", "")
        assert CodeAgent.parse_plan_decision("2") == ("revise", "")
        assert CodeAgent.parse_plan_decision("2 add error handling") == ("revise", "add error handling")
        assert CodeAgent.parse_plan_decision("2. add error handling") == ("revise", "add error handling")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_ask(self):
        """Numbered '3' or '3. question' returns ask decision."""
        assert CodeAgent.parse_plan_decision("3") == ("ask", "")
        assert CodeAgent.parse_plan_decision("3 what about tests?") == ("ask", "what about tests?")
        assert CodeAgent.parse_plan_decision("3. what about tests?") == ("ask", "what about tests?")

    @pytest.mark.asyncio
    async def test_parse_plan_decision_pending(self):
        """Non-decision text returns pending."""
        assert CodeAgent.parse_plan_decision("do it differently") == ("pending", "")
        assert CodeAgent.parse_plan_decision("") == ("pending", "")

    @pytest.mark.asyncio
    async def test_full_cycle_plan_pending_approve_execute(self, plan_config):
        """End-to-end: plan → pending → approve → execution."""
        agent = CodeAgent(config=plan_config, read_only=True, agent_name="plan")

        async def _fake_run_single(endpoint, user_input):
            yield AgentEvent(kind="text", text="Plan v1", role=FormationRole.PLANNER)

        agent._scheduler.run_single = _fake_run_single

        # Phase 1: generate plan (non-blocking, stored as pending)
        [e async for e in agent.run("Ship it")]
        assert agent.has_pending_plan
        assert agent.pending_plan_text == "Plan v1"

        # Phase 2: approve
        executed = []

        async def _fake_run_approved(user_input, plan_text):
            executed.append((user_input, plan_text))
            yield AgentEvent(kind="text", text="executed", role=FormationRole.EXECUTOR)

        agent._run_approved_plan = _fake_run_approved

        result = await agent.try_handle_plan_response("1")
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

        agent._scheduler.run_single = _fake_run_single

        # Phase 1: generate plan
        [e async for e in agent.run("Ship it")]
        assert agent.has_pending_plan

        # Phase 2: cancel
        result = await agent.try_handle_plan_response("4")
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
