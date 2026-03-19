# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test the scheduler and agent loop."""

from __future__ import annotations

import asyncio
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
from nemocode.core.scheduler import ROLE_PROMPTS, Scheduler
from nemocode.core.sessions import Session
from nemocode.core.streaming import CompletionResult, StreamChunk
from nemocode.tools.loader import load_tools


@pytest.fixture
def mock_provider():
    """Create a mock chat provider that returns a simple text response."""
    provider = AsyncMock()

    async def mock_stream(messages, tools=None, extra_body=None):
        yield StreamChunk(text="Hello, I can help with that.")
        yield StreamChunk(
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    provider.stream = mock_stream
    provider.complete = AsyncMock(
        return_value=CompletionResult(
            content="Here's my plan...",
            finish_reason="stop",
        )
    )
    return provider


@pytest.fixture
def scheduler_config() -> NeMoCodeConfig:
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
            "planning": Formation(
                name="Planning",
                slots=[
                    FormationSlot(endpoint="test-ep", role=FormationRole.PLANNER),
                    FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR),
                    FormationSlot(endpoint="test-ep", role=FormationRole.REVIEWER),
                ],
                verification_rounds=1,
            ),
        },
    )


class TestScheduler:
    @pytest.mark.asyncio
    async def test_run_single_collects_text(self, scheduler_config, mock_provider):
        registry = Registry(scheduler_config)
        tools = load_tools(["fs"])

        with patch.object(registry, "get_chat_provider", return_value=mock_provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_single("test-ep", "Hello"):
                events.append(ev)

        text_events = [e for e in events if e.kind == "text"]
        assert len(text_events) > 0
        full_text = "".join(e.text for e in text_events)
        assert "Hello" in full_text

    @pytest.mark.asyncio
    async def test_run_single_tracks_usage(self, scheduler_config, mock_provider):
        registry = Registry(scheduler_config)
        tools = load_tools(["fs"])

        with patch.object(registry, "get_chat_provider", return_value=mock_provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_single("test-ep", "Hello"):
                events.append(ev)

        usage_events = [e for e in events if e.kind == "usage"]
        assert len(usage_events) > 0

    @pytest.mark.asyncio
    async def test_run_single_emits_status_during_silent_wait(self, scheduler_config):
        registry = Registry(scheduler_config)
        tools = load_tools(["fs"])

        async def slow_stream(messages, tools=None, extra_body=None):
            await asyncio.sleep(0.1)
            yield StreamChunk(text="Done.")
            yield StreamChunk(
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )

        provider = AsyncMock()
        provider.stream = slow_stream

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools, status_interval_s=0.02)
            events = []
            async for ev in scheduler.run_single("test-ep", "Hello"):
                events.append(ev)

        status_events = [e for e in events if e.kind == "status"]
        assert status_events
        assert "Still working after" in status_events[0].text
        assert "analyzing the request" in status_events[0].text

    def test_reset_clears_sessions(self, scheduler_config):
        registry = Registry(scheduler_config)
        tools = load_tools(["fs"])
        scheduler = Scheduler(registry, tools)
        scheduler._sessions[FormationRole.EXECUTOR] = Session(id="test")
        scheduler.reset()
        assert len(scheduler._sessions) == 0


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, scheduler_config):
        """Multiple tool calls without confirmation should execute in parallel."""
        from nemocode.core.streaming import ToolCall

        registry = Registry(scheduler_config)
        tools = load_tools(["fs"])

        # Mock provider that returns two tool calls at once
        async def mock_stream_with_tools(messages, tools=None, extra_body=None):
            yield StreamChunk(text="Let me read both files.")
            yield StreamChunk(
                tool_calls=[
                    ToolCall(id="tc1", name="read_file", arguments={"path": "/dev/null"}),
                    ToolCall(id="tc2", name="list_dir", arguments={"path": "."}),
                ]
            )
            yield StreamChunk(
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 10},
            )

        provider = AsyncMock()
        provider.stream = mock_stream_with_tools

        # Second call: provider returns text only (no more tools)
        call_count = 0

        async def counting_stream(messages, tools=None, extra_body=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async for chunk in mock_stream_with_tools(messages, tools, extra_body):
                    yield chunk
            else:
                yield StreamChunk(text="Done.")
                yield StreamChunk(
                    finish_reason="stop",
                    usage={"prompt_tokens": 50, "completion_tokens": 10},
                )

        provider.stream = counting_stream

        with patch.object(registry, "get_chat_provider", return_value=provider):
            scheduler = Scheduler(registry, tools)
            events = []
            async for ev in scheduler.run_single("test-ep", "Read both"):
                events.append(ev)

        # Should have 2 tool_call events and 2 tool_result events
        tool_calls = [e for e in events if e.kind == "tool_call"]
        tool_results = [e for e in events if e.kind == "tool_result"]
        assert len(tool_calls) == 2
        assert len(tool_results) == 2


class TestRolePrompts:
    def test_all_roles_have_prompts(self):
        expected = {
            FormationRole.PLANNER,
            FormationRole.EXECUTOR,
            FormationRole.REVIEWER,
            FormationRole.DEBUGGER,
            FormationRole.FAST,
        }
        assert expected.issubset(set(ROLE_PROMPTS.keys()))

    def test_executor_prompt_mentions_tools(self):
        prompt = ROLE_PROMPTS[FormationRole.EXECUTOR]
        assert "tools" in prompt.lower()
        assert "read" in prompt.lower()
