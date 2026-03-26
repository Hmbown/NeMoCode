# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for formation scheduling edge cases."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nemocode.config.schema import (
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    NeMoCodeConfig,
)
from nemocode.core.registry import Registry
from nemocode.core.scheduler import Scheduler
from nemocode.core.streaming import CompletionResult, StreamChunk
from nemocode.tools.loader import load_tools


@pytest.fixture
def formation_config() -> NeMoCodeConfig:
    ep = Endpoint(
        name="test",
        tier=EndpointTier.DEV_HOSTED,
        base_url="https://test.com/v1",
        model_id="test-model",
    )
    return NeMoCodeConfig(
        default_endpoint="test-ep",
        endpoints={"test-ep": ep},
        formations={
            "solo": Formation(
                name="Solo",
                slots=[FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR)],
                verification_rounds=0,
            ),
            "full": Formation(
                name="Full",
                slots=[
                    FormationSlot(endpoint="test-ep", role=FormationRole.PLANNER),
                    FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR),
                    FormationSlot(endpoint="test-ep", role=FormationRole.REVIEWER),
                ],
                verification_rounds=2,
            ),
            "plan_execute": Formation(
                name="PlanExecute",
                slots=[
                    FormationSlot(endpoint="test-ep", role=FormationRole.PLANNER),
                    FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR),
                ],
                verification_rounds=0,
            ),
        },
    )


def _make_text_provider(text: str) -> AsyncMock:
    """Create a mock provider that streams a simple text response."""
    provider = AsyncMock()

    async def mock_stream(messages, tools=None, extra_body=None, response_format=None):
        yield StreamChunk(text=text)
        yield StreamChunk(
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        )

    provider.stream = mock_stream
    provider.complete = AsyncMock(return_value=CompletionResult(content=text, finish_reason="stop"))
    return provider


def _make_reject_provider(reject_count: int) -> AsyncMock:
    """Create a mock provider that rejects N times, then approves."""
    provider = AsyncMock()
    call_count = 0

    async def mock_stream(messages, tools=None, extra_body=None, response_format=None):
        nonlocal call_count
        call_count += 1
        # Check if any message mentions "Review" to distinguish reviewer from executor
        is_review = any("Review" in (m.content or "") for m in messages if hasattr(m, "content"))
        if is_review and call_count <= reject_count:
            yield StreamChunk(text="REQUEST_CHANGES: Fix the spacing.")
        elif is_review:
            yield StreamChunk(text="APPROVE: Looks good now.")
        else:
            yield StreamChunk(text="Done implementing.")
        yield StreamChunk(
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        )

    provider.stream = mock_stream
    provider.complete = AsyncMock(return_value=CompletionResult(content="", finish_reason="stop"))
    return provider


class TestEmptyPlan:
    @pytest.mark.asyncio
    async def test_planner_produces_empty_plan_executor_still_runs(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])

        # Planner returns empty string, executor returns real text
        empty_planner = AsyncMock()
        empty_planner.complete = AsyncMock(
            return_value=CompletionResult(content="", finish_reason="stop")
        )

        executor = _make_text_provider("Implemented the change.")

        def get_provider(name):
            return executor

        with patch.object(registry, "get_chat_provider", side_effect=get_provider):
            # Override _run_text_only to return empty plan
            scheduler = Scheduler(registry, tools)

            async def empty_plan(slot, user_input):
                return ""

            with patch.object(scheduler, "_run_text_only", side_effect=empty_plan):
                events = []
                async for ev in scheduler.run_formation("plan_execute", "Add a feature"):
                    events.append(ev)

        # Should have phase events and text from executor
        phase_events = [e for e in events if e.kind == "phase"]
        text_events = [e for e in events if e.kind == "text"]
        assert len(phase_events) >= 2  # planning + executing
        assert any("Implemented" in e.text for e in text_events)


class TestReviewerRejection:
    @pytest.mark.asyncio
    async def test_reviewer_rejects_up_to_verification_rounds_limit(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])
        provider = _make_reject_provider(reject_count=10)  # Always reject

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)

            async def mock_plan(slot, user_input):
                return "Step 1: Do the thing"

            with patch.object(scheduler, "_run_text_only", side_effect=mock_plan):
                events = []
                async for ev in scheduler.run_formation("full", "Fix the bug"):
                    events.append(ev)

        # "full" formation has verification_rounds=2, so we should see at most 2 review phases
        review_phases = [e for e in events if e.kind == "phase" and e.phase == "reviewing"]
        assert len(review_phases) <= 2


class TestExecutorStagnation:
    @pytest.mark.asyncio
    async def test_executor_stagnation_triggers_error_event(self, formation_config):
        """Repeated identical tool calls trigger stagnation detection."""
        from nemocode.core.streaming import ToolCall

        registry = Registry(formation_config)
        tools = load_tools(["fs"])

        # Provider that always emits the same tool call — triggers repeat detection
        stagnant_provider = AsyncMock()

        async def stagnant_stream(messages, tools=None, extra_body=None, response_format=None):
            yield StreamChunk(text="Let me check.")
            yield StreamChunk(
                tool_calls=[ToolCall(id="tc_loop", name="read_file", arguments={"path": "same.py"})]
            )
            yield StreamChunk(
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            )

        stagnant_provider.stream = stagnant_stream

        with patch.object(registry, "get_chat_provider", return_value=stagnant_provider):
            scheduler = Scheduler(registry, tools, max_tool_rounds=20)
            # Lower threshold for faster test
            scheduler._stagnation.repeat_threshold = 3

            events = []
            async for ev in scheduler.run_single("test-ep", "Do something"):
                events.append(ev)

        error_events = [e for e in events if e.kind == "error"]
        assert len(error_events) >= 1
        err_text = error_events[0].text.lower()
        assert "stagnation" in err_text or "stuck" in err_text


class TestExecutorOnlyFormation:
    @pytest.mark.asyncio
    async def test_run_formation_with_only_executor(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])
        provider = _make_text_provider("Just executing directly.")

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_formation("solo", "Hello"):
                events.append(ev)

        # Should have executing phase and text output
        phase_events = [e for e in events if e.kind == "phase"]
        text_events = [e for e in events if e.kind == "text"]
        assert any(e.phase == "executing" for e in phase_events)
        assert any("executing" in e.text.lower() for e in text_events) or len(text_events) > 0


class TestFormationPhaseEvents:
    @pytest.mark.asyncio
    async def test_run_formation_yields_correct_phase_events(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])

        provider = AsyncMock()

        async def approve_stream(messages, tools=None, extra_body=None, response_format=None):
            yield StreamChunk(text="APPROVE: All good.")
            yield StreamChunk(
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            )

        provider.stream = approve_stream
        provider.complete = AsyncMock(
            return_value=CompletionResult(content="My plan", finish_reason="stop")
        )

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_formation("full", "Add tests"):
                events.append(ev)

        phases = [e.phase for e in events if e.kind == "phase"]
        # Full formation should have planning, executing, and reviewing phases
        assert "planning" in phases
        assert "executing" in phases
        assert "reviewing" in phases

    @pytest.mark.asyncio
    async def test_phase_events_carry_correct_roles(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])

        provider = AsyncMock()

        async def approve_stream(messages, tools=None, extra_body=None, response_format=None):
            yield StreamChunk(text="APPROVE: Fine.")
            yield StreamChunk(
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            )

        provider.stream = approve_stream
        provider.complete = AsyncMock(
            return_value=CompletionResult(content="Plan here", finish_reason="stop")
        )

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_formation("full", "Fix it"):
                events.append(ev)

        phase_events = [e for e in events if e.kind == "phase"]
        planning = [e for e in phase_events if e.phase == "planning"]
        executing = [e for e in phase_events if e.phase == "executing"]
        reviewing = [e for e in phase_events if e.phase == "reviewing"]

        assert all(e.role == FormationRole.PLANNER for e in planning)
        assert all(e.role == FormationRole.EXECUTOR for e in executing)
        assert all(e.role == FormationRole.REVIEWER for e in reviewing)


class TestFormationBackendFailures:
    @pytest.mark.asyncio
    async def test_planner_backend_error_aborts_later_phases(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])

        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=CompletionResult(
                content="SGLang endpoint spark-sglang-super is not reachable.",
                finish_reason="error",
            )
        )
        provider.stream = AsyncMock()

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_formation("full", "Fix it"):
                events.append(ev)

        phases = [e.phase for e in events if e.kind == "phase"]
        error_events = [e for e in events if e.kind == "error"]
        assert phases == ["planning"]
        assert error_events
        assert "not reachable" in error_events[0].text
        provider.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_executor_backend_error_skips_review(self, formation_config):
        registry = Registry(formation_config)
        tools = load_tools(["fs"])

        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=CompletionResult(content="Plan here", finish_reason="stop")
        )

        async def failing_stream(messages, tools=None, extra_body=None, response_format=None):
            yield StreamChunk(
                text="Local SGLang endpoint test-ep is not reachable.",
                finish_reason="error",
            )

        provider.stream = failing_stream

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_formation("full", "Fix it"):
                events.append(ev)

        phases = [e.phase for e in events if e.kind == "phase"]
        error_events = [e for e in events if e.kind == "error"]
        assert phases == ["planning", "executing"]
        assert error_events
