# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test the scheduler and agent loop."""

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

    def test_reset_clears_sessions(self, scheduler_config):
        registry = Registry(scheduler_config)
        tools = load_tools(["fs"])
        scheduler = Scheduler(registry, tools)
        scheduler._sessions[FormationRole.EXECUTOR] = Session(id="test")
        scheduler.reset()
        assert len(scheduler._sessions) == 0


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
