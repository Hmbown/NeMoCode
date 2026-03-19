# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the interactive REPL."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemocode.cli.commands.repl import (
    _endpoint_summary,
    _fmt_tokens,
    _InputReader,
    _render_status_bar,
    _ReplState,
    _short_model_ref,
    _SlashDispatcher,
    _TurnRenderer,
)
from nemocode.config import _parse_config
from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    Manifest,
    NeMoCodeConfig,
    ToolPermissions,
)
from nemocode.core.scheduler import AgentEvent


@pytest.fixture
def sample_config():
    ep = Endpoint(
        name="test-ep",
        tier=EndpointTier.DEV_HOSTED,
        base_url="https://test.api.com/v1",
        model_id="nvidia/nemotron-3-super-120b-a12b",
        capabilities=[Capability.CHAT],
    )
    manifest = Manifest(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        display_name="Test Model",
        context_window=1_048_576,
    )
    return NeMoCodeConfig(
        default_endpoint="test-ep",
        endpoints={
            "test-ep": ep,
            "other-ep": Endpoint(
                name="other-ep",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://other.api.com/v1",
                model_id="other-model",
                capabilities=[Capability.CHAT],
            ),
        },
        manifests={"nvidia/nemotron-3-super-120b-a12b": manifest},
        formations={
            "solo": Formation(
                name="Solo",
                description="Single model",
                slots=[FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR)],
            ),
        },
        permissions=ToolPermissions(),
    )


@pytest.fixture
def repl_state(sample_config):
    return _ReplState(
        config=sample_config,
        show_thinking=False,
        auto_yes=True,
    )


class TestInputReader:
    def _make_reader_with_mock(self, return_values):
        """Create an InputReader and mock its _session.prompt to return values in sequence."""
        reader = _InputReader()
        if isinstance(return_values, list):
            if reader._session is not None:
                reader._session.prompt = MagicMock(side_effect=return_values)
            else:
                # Fallback path: mock input()
                pass
        return reader

    @pytest.mark.asyncio
    async def test_single_line(self):
        from unittest.mock import AsyncMock

        reader = _InputReader()
        if reader._session is not None:
            reader._session.prompt_async = AsyncMock(return_value="hello")
        else:
            pytest.skip("prompt_toolkit not available")
        result = await reader.read()
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_eof_returns_none(self):
        from unittest.mock import AsyncMock

        reader = _InputReader()
        if reader._session is not None:
            reader._session.prompt_async = AsyncMock(side_effect=EOFError)
        else:
            pytest.skip("prompt_toolkit not available")
        result = await reader.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_multiline_detection(self):
        from unittest.mock import AsyncMock

        reader = _InputReader()
        if reader._session is not None:
            reader._session.prompt_async = AsyncMock(side_effect=['"""first', "second", '"""'])
        else:
            pytest.skip("prompt_toolkit not available")
        result = await reader.read()
        assert "first" in result
        assert "second" in result

    @pytest.mark.asyncio
    async def test_multiline_same_line(self):
        from unittest.mock import AsyncMock

        reader = _InputReader()
        if reader._session is not None:
            reader._session.prompt_async = AsyncMock(return_value='"""short text"""')
        else:
            pytest.skip("prompt_toolkit not available")
        result = await reader.read()
        assert result == "short text"

    def test_short_model_ref_compacts_local_path(self):
        assert _short_model_ref("/models/NVIDIA-Nemotron") == "NVIDIA-Nemotron"

    def test_endpoint_summary_compacts_local_model_path(self):
        endpoint = Endpoint(
            name="local-sglang-super",
            tier=EndpointTier.LOCAL_SGLANG,
            base_url="http://localhost:8000/v1",
            model_id="/models/NVIDIA-Nemotron-3-Super-120B",
            capabilities=[Capability.CHAT],
        )
        assert _endpoint_summary(endpoint) == "local-sglang-super · NVIDIA-Nemotron-3-Super-120B"


class TestSlashDispatcher:
    def test_help_returns_true(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        assert dispatcher.dispatch("/help") is True

    def test_quit_returns_false(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        assert dispatcher.dispatch("/quit") is False

    def test_exit_returns_false(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        assert dispatcher.dispatch("/exit") is False

    def test_think_toggles(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        assert repl_state.show_thinking is False
        dispatcher.dispatch("/think")
        assert repl_state.show_thinking is True
        dispatcher.dispatch("/think")
        assert repl_state.show_thinking is False

    def test_reset(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        repl_state.turn_count = 5
        dispatcher.dispatch("/reset")
        assert repl_state.turn_count == 0

    def test_unknown_command(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        result = dispatcher.dispatch("/unknown")
        assert result is True  # should continue, not crash

    def test_endpoint_list(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        result = dispatcher.dispatch("/endpoint")
        assert result is True

    def test_endpoint_switch(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        dispatcher.dispatch("/endpoint other-ep")
        assert repl_state.config.default_endpoint == "other-ep"

    def test_endpoint_unknown(self, repl_state):
        dispatcher = _SlashDispatcher(repl_state)
        dispatcher.dispatch("/endpoint nonexistent")
        # Should not crash, endpoint should remain unchanged
        assert repl_state.config.default_endpoint == "test-ep"

    def test_undo_empty_stack(self, repl_state):
        """Undo with nothing to undo should print message and continue."""
        dispatcher = _SlashDispatcher(repl_state)
        result = dispatcher.dispatch("/undo")
        assert result is True  # should continue

    def test_undo_restores_file(self, repl_state, tmp_path):
        """Undo should restore the previous file content."""
        from nemocode.tools.fs import _UNDO_STACK

        # Set up: push a known state onto the undo stack
        test_file = tmp_path / "test.txt"
        test_file.write_text("modified content")
        _UNDO_STACK.append((str(test_file), "original content"))

        dispatcher = _SlashDispatcher(repl_state)
        result = dispatcher.dispatch("/undo")
        assert result is True

        # Verify file was restored
        assert test_file.read_text() == "original content"

        # Clean up
        _UNDO_STACK.clear()

    def test_agent_switch_by_alias(self, repl_state):
        repl_state.config.agents = _parse_config({}).agents
        dispatcher = _SlashDispatcher(repl_state)
        result = dispatcher.dispatch("/agent builder")
        assert result is True
        assert repl_state.agent_name == "build"


class TestReplState:
    def test_initial_state(self, sample_config):
        state = _ReplState(config=sample_config, show_thinking=True, auto_yes=False)
        assert state.show_thinking is True
        assert state.auto_yes is False
        assert state.turn_count == 0
        assert state.agent is not None

    def test_cancel_and_clear(self, repl_state):
        assert not repl_state.cancelled
        repl_state.cancel()
        assert repl_state.cancelled
        repl_state.clear_cancel()
        assert not repl_state.cancelled

    def test_rebuild_resets_turn_count(self, repl_state):
        repl_state.turn_count = 10
        repl_state.rebuild_agent()
        assert repl_state.turn_count == 0

    def test_primary_agent_display_defaults_when_no_profiles(self, sample_config):
        state = _ReplState(config=sample_config, show_thinking=False, auto_yes=True)
        assert state.current_primary_agent_name() is None
        assert state.current_primary_agent_display() == "default"


class TestTurnRenderer:
    def test_text_event(self, repl_state):
        renderer = _TurnRenderer(repl_state)
        event = AgentEvent(kind="text", text="hello")
        renderer.render_event(event)
        assert renderer._text_buf == "hello"

    def test_thinking_hidden_by_default(self, repl_state):
        repl_state.show_thinking = False
        renderer = _TurnRenderer(repl_state)
        event = AgentEvent(kind="thinking", thinking="hmm")
        renderer.render_event(event)
        assert renderer._think_buf == "hmm"

    def test_tool_call_tracking(self, repl_state):
        renderer = _TurnRenderer(repl_state)
        event = AgentEvent(kind="tool_call", tool_name="read_file", tool_args={"path": "test.py"})
        renderer.render_event(event)
        assert renderer._tool_call_count == 1

    def test_tool_error_tracking(self, repl_state):
        renderer = _TurnRenderer(repl_state)
        event = AgentEvent(
            kind="tool_result",
            tool_name="bash_exec",
            tool_result='{"error": "failed"}',
            is_error=True,
        )
        renderer.render_event(event)
        assert renderer._tool_error_count == 1

    def test_finalize_records_metrics(self, repl_state):
        renderer = _TurnRenderer(repl_state)
        renderer._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        renderer._current_model_id = "test-model"
        renderer.finalize()
        assert repl_state.metrics.request_count == 1


class TestContextUsageFormat:
    def test_format_includes_tokens(self, repl_state):
        # Record enough tokens so usage exceeds the 10% display threshold
        # Context window is 1M, so need >100K tokens
        from nemocode.core.metrics import RequestMetrics

        repl_state.metrics.record(
            RequestMetrics(
                model_id="test",
                endpoint_name="test-ep",
                prompt_tokens=100_000,
                completion_tokens=60_000,
            )
        )
        # _render_status_bar prints to console, just verify no crash
        _render_status_bar(repl_state)

    def test_fmt_tokens_small(self, repl_state):
        assert _fmt_tokens(500) == "500"

    def test_fmt_tokens_thousands(self, repl_state):
        assert _fmt_tokens(5_000) == "5K"

    def test_fmt_tokens_millions(self, repl_state):
        assert _fmt_tokens(1_500_000) == "1.5M"

    def test_status_bar_no_crash(self, repl_state):
        """Status bar should render without error even with no data."""
        _render_status_bar(repl_state)
