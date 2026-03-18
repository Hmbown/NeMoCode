# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for TUI UX enhancements — status bar model_name/last_tps, turn summary."""

from __future__ import annotations

import pytest

from nemocode.cli.tui import StatusBar, _TUIState
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


class TestStatusBarModelName:
    def test_render_includes_model_name(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.model_name = "Nemotron 3 Super"
        rendered = bar.render()
        assert "Nemotron 3 Super" in rendered

    def test_render_falls_back_to_endpoint(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.model_name = ""
        bar.endpoint = "nim-super"
        rendered = bar.render()
        assert "nim-super" in rendered

    def test_model_name_preferred_over_endpoint(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.model_name = "Nemotron 3 Super"
        bar.endpoint = "nim-super"
        rendered = bar.render()
        assert "Nemotron 3 Super" in rendered
        # Endpoint should NOT appear when model_name is set
        assert "nim-super" not in rendered

    def test_render_includes_formation(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.formation = "spark"
        rendered = bar.render()
        assert "spark" in rendered


class TestStatusBarThroughput:
    def test_render_includes_tps(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.last_tps = 350.0
        rendered = bar.render()
        assert "350" in rendered
        assert "tok/s" in rendered

    def test_no_tps_when_zero(self):
        bar = StatusBar()
        bar.mode = "code"
        bar.last_tps = 0.0
        rendered = bar.render()
        assert "tok/s" not in rendered


class TestTUIStateThroughput:
    def test_record_metrics_updates_tps(self, sample_config):
        state = _TUIState(config=sample_config)
        state.reset_turn()
        state.last_usage = {"prompt_tokens": 100, "completion_tokens": 500}
        state.current_model_id = "nvidia/nemotron-3-super-120b-a12b"

        # Simulate 2 seconds of turn time
        import time

        state.turn_start = time.time() - 2.0

        state.record_metrics()

        assert state.metrics.request_count == 1
        assert state.metrics.last_tokens_per_sec > 0

    def test_metrics_throughput_accessible(self, sample_config):
        state = _TUIState(config=sample_config)
        state.metrics.record(RequestMetrics(completion_tokens=1000, total_time_ms=2000))
        assert state.metrics.last_tokens_per_sec == 500.0
        assert state.metrics.avg_tokens_per_sec == 500.0
