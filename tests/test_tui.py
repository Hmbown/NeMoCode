# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the full-screen TUI."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemocode.config import _parse_config
from nemocode.cli.tui import (
    NeMoCodeTUI,
    StatusBar,
    ToolPanel,
    _fmt_tokens,
    _get_git_branch,
    _TUIState,
)
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        display_name="Nemotron 3 Super",
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
def tui_state(sample_config):
    state = _TUIState(config=sample_config)
    state.agent = MagicMock()
    state.agent.sessions = {}
    return state


# ---------------------------------------------------------------------------
# App construction tests
# ---------------------------------------------------------------------------


class TestAppConstruction:
    def test_app_creates_with_defaults(self, sample_config):
        """App should construct without errors using config."""
        app = NeMoCodeTUI(config=sample_config)
        assert app._state.config is sample_config
        assert app._state.mode == "code"
        assert app._state.show_thinking is False

    def test_app_applies_endpoint_override(self, sample_config):
        app = NeMoCodeTUI(config=sample_config, endpoint="other-ep")
        assert app._state.config.default_endpoint == "other-ep"

    def test_app_applies_formation_override(self, sample_config):
        app = NeMoCodeTUI(config=sample_config, formation="solo")
        assert app._state.config.active_formation == "solo"

    def test_app_show_thinking_flag(self, sample_config):
        app = NeMoCodeTUI(config=sample_config, show_thinking=True)
        assert app._state.show_thinking is True

    def test_app_auto_yes_flag(self, sample_config):
        app = NeMoCodeTUI(config=sample_config, auto_yes=True)
        assert app._state.auto_yes is True

    def test_app_has_correct_bindings(self, sample_config):
        app = NeMoCodeTUI(config=sample_config)
        binding_keys = {b.key for b in app.BINDINGS}
        assert "ctrl+q" in binding_keys
        assert "tab" in binding_keys
        assert "ctrl+c" in binding_keys
        assert "enter" in binding_keys
        assert "ctrl+t" in binding_keys


# ---------------------------------------------------------------------------
# TUI State tests
# ---------------------------------------------------------------------------


class TestTUIState:
    def test_initial_state(self, sample_config):
        state = _TUIState(config=sample_config)
        assert state.mode == "code"
        assert state.show_thinking is False
        assert state.auto_yes is False
        assert state.turn_count == 0
        assert state.is_streaming is False
        assert state.cancelled is False

    def test_mode_cycling(self, sample_config):
        state = _TUIState(config=sample_config)
        assert state.mode == "code"
        assert state.cycle_mode() == "plan"
        assert state.cycle_mode() == "auto"
        assert state.cycle_mode() == "code"

    def test_cancel_and_clear(self, sample_config):
        state = _TUIState(config=sample_config)
        assert not state.cancelled
        state.cancel()
        assert state.cancelled
        state.clear_cancel()
        assert not state.cancelled

    def test_reset_turn(self, sample_config):
        state = _TUIState(config=sample_config)
        state.text_buf = "some text"
        state.think_buf = "some thinking"
        state.tool_call_count = 5
        state.tool_error_count = 2
        state.reasoning_hint_shown = True
        state.reset_turn()
        assert state.text_buf == ""
        assert state.think_buf == ""
        assert state.tool_call_count == 0
        assert state.tool_error_count == 0
        assert state.reasoning_hint_shown is False
        assert state.first_token_time is None
        assert state.turn_start > 0

    def test_record_metrics(self, sample_config):
        state = _TUIState(config=sample_config)
        state.reset_turn()
        state.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        state.current_model_id = "test-model"
        state.tool_call_count = 3
        state.tool_error_count = 1
        state.record_metrics()
        assert state.metrics.request_count == 1
        assert state.metrics.total_tokens == 150

    def test_resolve_context_window_from_manifest(self, sample_config):
        state = _TUIState(config=sample_config)
        assert state.resolve_context_window() == 1_048_576

    def test_resolve_context_window_default(self):
        config = NeMoCodeConfig()
        state = _TUIState(config=config)
        assert state.resolve_context_window() == 128_000

    def test_build_agent_code_mode(self, sample_config):
        state = _TUIState(config=sample_config)
        agent = state.build_agent()
        assert agent is not None

    def test_build_agent_plan_mode(self, sample_config):
        state = _TUIState(config=sample_config)
        state.mode_idx = 1  # plan
        agent = state.build_agent()
        assert agent is not None
        # Plan mode agent should be read_only
        assert agent._read_only is True

    def test_build_agent_auto_mode(self, sample_config):
        state = _TUIState(config=sample_config)
        state.mode_idx = 2  # auto
        agent = state.build_agent()
        assert agent is not None

    def test_primary_agent_display_defaults_when_no_profiles(self):
        state = _TUIState(config=NeMoCodeConfig())
        assert state.current_primary_agent_name() is None
        assert state.current_primary_agent_display() == "default"
        assert state.build_agent() is not None


# ---------------------------------------------------------------------------
# Slash command dispatch tests
# ---------------------------------------------------------------------------


class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_help_displays(self, sample_config):
        """The /help command should add text to the chat."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/help")
            # Verify no crash; the system message is added to chat scroll

    @pytest.mark.asyncio
    async def test_think_toggle(self, sample_config):
        """The /think command should toggle thinking display."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            assert app._state.show_thinking is False
            app._dispatch_slash("/think")
            assert app._state.show_thinking is True
            app._dispatch_slash("/think")
            assert app._state.show_thinking is False

    @pytest.mark.asyncio
    async def test_mode_cycle(self, sample_config):
        """The /mode command should cycle modes."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            assert app._state.mode == "code"
            app._dispatch_slash("/mode")
            assert app._state.mode == "plan"
            app._dispatch_slash("/mode")
            assert app._state.mode == "auto"
            app._dispatch_slash("/mode")
            assert app._state.mode == "code"

    @pytest.mark.asyncio
    async def test_reset(self, sample_config):
        """The /reset command should reset conversation state."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._state.turn_count = 5
            app._dispatch_slash("/reset")
            assert app._state.turn_count == 0

    @pytest.mark.asyncio
    async def test_cost_no_crash(self, sample_config):
        """The /cost command should display without errors."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/cost")  # Should not raise

    @pytest.mark.asyncio
    async def test_endpoint_list(self, sample_config):
        """The /endpoint command without args should list endpoints."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/endpoint")  # Should not raise

    @pytest.mark.asyncio
    async def test_endpoint_switch(self, sample_config):
        """The /endpoint <name> command should switch endpoints."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/endpoint other-ep")
            assert app._state.config.default_endpoint == "other-ep"

    @pytest.mark.asyncio
    async def test_endpoint_unknown(self, sample_config):
        """The /endpoint with unknown name should show error."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/endpoint nonexistent")
            assert app._state.config.default_endpoint == "test-ep"  # unchanged

    @pytest.mark.asyncio
    async def test_formation_list(self, sample_config):
        """The /formation command without args should list formations."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/formation")  # Should not raise

    @pytest.mark.asyncio
    async def test_formation_activate(self, sample_config):
        """The /formation <name> command should activate a formation."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/formation solo")
            assert app._state.config.active_formation == "solo"

    @pytest.mark.asyncio
    async def test_formation_off(self, sample_config):
        """The /formation off command should deactivate formation."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._state.config.active_formation = "solo"
            app._dispatch_slash("/formation off")
            assert app._state.config.active_formation is None

    @pytest.mark.asyncio
    async def test_agent_switch_by_alias(self, sample_config):
        """The /agent <alias> command should resolve built-in aliases."""
        sample_config.agents = _parse_config({}).agents
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/agent builder")
            assert app._state.agent_name == "build"

    @pytest.mark.asyncio
    async def test_unknown_command(self, sample_config):
        """An unknown slash command should display an error message."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app._dispatch_slash("/foobar")  # Should not crash


# ---------------------------------------------------------------------------
# Key binding tests
# ---------------------------------------------------------------------------


class TestKeyBindings:
    @pytest.mark.asyncio
    async def test_tab_cycles_mode(self, sample_config):
        """Tab key should cycle through modes."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            assert app._state.mode == "code"
            app.action_cycle_mode()
            assert app._state.mode == "plan"
            assert app.mode == "plan"

    @pytest.mark.asyncio
    async def test_ctrl_c_cancels(self, sample_config):
        """Ctrl+C should cancel a streaming turn."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app.streaming = True
            app._state.is_streaming = True
            app.action_cancel_turn()
            assert app._state.cancelled is True
            assert app.streaming is False

    @pytest.mark.asyncio
    async def test_cancel_noop_when_not_streaming(self, sample_config):
        """Ctrl+C when not streaming should do nothing."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            app.action_cancel_turn()
            assert app._state.cancelled is False

    @pytest.mark.asyncio
    async def test_toggle_tools(self, sample_config):
        """Ctrl+T should toggle the tool panel visibility."""
        async with NeMoCodeTUI(config=sample_config).run_test() as pilot:
            app = pilot.app
            panel = app.query_one("#tool-panel", ToolPanel)
            assert not panel.has_class("visible")
            app.action_toggle_tools()
            assert panel.has_class("visible")
            app.action_toggle_tools()
            assert not panel.has_class("visible")


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestTokenFormatting:
    def test_small_number(self):
        assert _fmt_tokens(42) == "42"

    def test_thousands(self):
        assert _fmt_tokens(1500) == "2K"

    def test_millions(self):
        assert _fmt_tokens(1_500_000) == "1.5M"

    def test_exact_thousand(self):
        assert _fmt_tokens(1000) == "1K"

    def test_exact_million(self):
        assert _fmt_tokens(1_000_000) == "1.0M"

    def test_zero(self):
        assert _fmt_tokens(0) == "0"


class TestGitBranch:
    def test_git_branch_returns_string(self):
        """_get_git_branch should return a string (possibly empty)."""
        branch = _get_git_branch()
        assert isinstance(branch, str)


# ---------------------------------------------------------------------------
# AgentEvent handling (unit-level, without full app)
# ---------------------------------------------------------------------------


class TestAgentEventState:
    def test_text_event_accumulates(self, tui_state):
        """Text events should accumulate in the state buffer."""
        tui_state.reset_turn()
        tui_state.text_buf += "hello "
        tui_state.text_buf += "world"
        assert tui_state.text_buf == "hello world"

    def test_thinking_event_accumulates(self, tui_state):
        tui_state.reset_turn()
        tui_state.think_buf += "step 1... "
        tui_state.think_buf += "step 2"
        assert tui_state.think_buf == "step 1... step 2"

    def test_tool_call_count(self, tui_state):
        tui_state.reset_turn()
        tui_state.tool_call_count += 1
        tui_state.tool_call_count += 1
        assert tui_state.tool_call_count == 2

    def test_tool_error_count(self, tui_state):
        tui_state.reset_turn()
        tui_state.tool_error_count += 1
        assert tui_state.tool_error_count == 1

    def test_usage_tracking(self, tui_state):
        tui_state.reset_turn()
        tui_state.last_usage = {"prompt_tokens": 200, "completion_tokens": 100}
        tui_state.current_model_id = "nvidia/nemotron-3-super-120b-a12b"
        tui_state.record_metrics()
        assert tui_state.metrics.total_tokens == 300


# ---------------------------------------------------------------------------
# StatusBar rendering tests
# ---------------------------------------------------------------------------


class TestStatusBar:
    def test_render_includes_mode(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.endpoint = "test-ep"
        rendered = bar.render()
        assert "code" in rendered

    def test_render_includes_endpoint(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.endpoint = "my-endpoint"
        rendered = bar.render()
        assert "my-endpoint" in rendered

    def test_render_includes_streaming(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.is_streaming = True
        rendered = bar.render()
        assert "LIVE" in rendered

    def test_render_includes_context_pct(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.context_pct = 55.0
        rendered = bar.render()
        assert "55%" in rendered

    def test_render_includes_tokens(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.total_tokens = 5000
        rendered = bar.render()
        assert "5K" in rendered

    def test_render_includes_git_branch(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.git_branch = "main"
        rendered = bar.render()
        assert "main" in rendered

    def test_render_includes_version(self):
        bar = StatusBar()
        bar.mode = "code"
        rendered = bar.render()
        assert "v" in rendered
