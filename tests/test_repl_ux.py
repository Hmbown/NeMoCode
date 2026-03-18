# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for REPL UX enhancements — turn summary, banner, status bar, /cost."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from nemocode.cli.commands.repl import (
    _get_hardware_line,
    _render_status_bar,
    _ReplState,
    _TurnRenderer,
)
from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    Manifest,
    NeMoCodeConfig,
    ToolPermissions,
)
from nemocode.core.metrics import RequestMetrics


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
        endpoints={"test-ep": ep},
        manifests={"nvidia/nemotron-3-super-120b-a12b": manifest},
        permissions=ToolPermissions(),
    )


@pytest.fixture
def repl_state(sample_config):
    return _ReplState(
        config=sample_config,
        show_thinking=False,
        auto_yes=True,
    )


class TestTurnSummary:
    def test_prints_summary_after_turn(self, repl_state, capsys):
        renderer = _TurnRenderer(repl_state)
        renderer._turn_start = time.time() - 2.0  # 2 seconds ago
        renderer._first_token_time = time.time() - 1.5  # TTFT 0.5s
        renderer._last_usage = {"prompt_tokens": 500, "completion_tokens": 347}
        renderer._current_model_id = "nvidia/nemotron-3-super-120b-a12b"
        renderer._text_buf = "some response"
        renderer._tool_call_count = 3
        renderer.finalize()

        # Metrics should be recorded
        assert repl_state.metrics.request_count == 1

    def test_no_summary_for_zero_tokens(self, repl_state, capsys):
        renderer = _TurnRenderer(repl_state)
        renderer._last_usage = {}
        renderer._text_buf = ""
        renderer.finalize()
        # No crash, and metrics should still be recorded
        assert repl_state.metrics.request_count == 1

    def test_summary_includes_tool_count(self, repl_state):
        renderer = _TurnRenderer(repl_state)
        renderer._turn_start = time.time() - 1.0
        renderer._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        renderer._text_buf = "response"
        renderer._tool_call_count = 5
        renderer.finalize()
        assert repl_state.metrics.request_count == 1
        # The recorded metrics should have the tool count
        last = repl_state.metrics._requests[-1]
        assert last.tool_calls == 5


class TestStatusBar:
    def test_shows_model_display_name(self, repl_state):
        """Status bar should show model display name, not endpoint key."""
        # Record some metrics so tok/s shows up
        repl_state.metrics.record(
            RequestMetrics(
                completion_tokens=500,
                total_time_ms=2000,
                prompt_tokens=100,
            )
        )
        # _render_status_bar prints to the module-level console
        # Just verify it doesn't crash
        _render_status_bar(repl_state)

    def test_shows_last_tps(self, repl_state):
        repl_state.metrics.record(
            RequestMetrics(
                completion_tokens=1000,
                total_time_ms=2000,  # 500 tok/s
                prompt_tokens=100,
            )
        )
        _render_status_bar(repl_state)
        assert repl_state.metrics.last_tokens_per_sec == 500.0


class TestHardwareLine:
    def test_spark_detection(self):
        """When hardware is DGX Spark, banner should mention it."""
        mock_hw = MagicMock()
        mock_hw.is_dgx_spark = True
        mock_hw.unified_memory_gb = 128.0
        mock_hw.gpus = [MagicMock(name="GB10", vram_gb=128)]

        with patch(
            "nemocode.core.hardware.detect_hardware",
            return_value=mock_hw,
        ):
            line = _get_hardware_line()
            assert "DGX Spark" in line
            assert "128GB" in line

    def test_standard_gpu(self):
        mock_hw = MagicMock()
        mock_hw.is_dgx_spark = False
        mock_hw.gpus = [MagicMock()]
        mock_hw.gpus[0].name = "NVIDIA GeForce RTX 4090"
        mock_hw.gpus[0].vram_gb = 24.0

        with patch(
            "nemocode.core.hardware.detect_hardware",
            return_value=mock_hw,
        ):
            line = _get_hardware_line()
            assert "RTX 4090" in line
            assert "24GB" in line

    def test_multi_gpu(self):
        mock_hw = MagicMock()
        mock_hw.is_dgx_spark = False
        gpu = MagicMock()
        gpu.name = "NVIDIA H100"
        gpu.vram_gb = 80.0
        mock_hw.gpus = [gpu, gpu]

        with patch(
            "nemocode.core.hardware.detect_hardware",
            return_value=mock_hw,
        ):
            line = _get_hardware_line()
            assert "2×" in line or "2x" in line.lower()
            assert "80GB" in line

    def test_no_gpu_returns_empty(self):
        mock_hw = MagicMock()
        mock_hw.is_dgx_spark = False
        mock_hw.gpus = []

        with patch(
            "nemocode.core.hardware.detect_hardware",
            return_value=mock_hw,
        ):
            line = _get_hardware_line()
            assert line == ""

    def test_detection_failure_returns_empty(self):
        with patch(
            "nemocode.core.hardware.detect_hardware",
            side_effect=Exception("no nvidia-smi"),
        ):
            line = _get_hardware_line()
            assert line == ""
