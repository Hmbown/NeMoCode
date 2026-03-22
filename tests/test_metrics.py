# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from nemocode.core.metrics import MetricsCollector, RequestMetrics


class TestMetricsCollector:
    def test_metrics_collector(self):
        collector = MetricsCollector()

        assert collector.total_tokens == 0
        assert collector.total_cost == 0.0
        assert collector.request_count == 0

        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                endpoint_name="nim-super",
                prompt_tokens=1000,
                completion_tokens=500,
                total_time_ms=2000.0,
            )
        )

        assert collector.request_count == 1
        assert collector.total_prompt_tokens == 1000
        assert collector.total_completion_tokens == 500
        assert collector.total_tokens == 1500
        assert collector.total_cost > 0
        assert collector.avg_latency_ms == 2000.0

    def test_metrics_collector_per_endpoint(self):
        collector = MetricsCollector()

        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                endpoint_name="nim-super",
                prompt_tokens=500,
                completion_tokens=200,
                total_time_ms=1000.0,
            )
        )

        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                endpoint_name="nim-nano",
                prompt_tokens=300,
                completion_tokens=100,
                total_time_ms=500.0,
            )
        )

        assert collector.request_count == 2
        assert collector.total_tokens == 1100
        assert collector.avg_latency_ms == 750.0

        summary = collector.summary()
        assert summary["requests"] == 2
        assert summary["total_tokens"] == 1100
        assert "estimated_cost_usd" in summary

    def test_metrics_collector_reset(self):
        collector = MetricsCollector()
        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                endpoint_name="nim-super",
                prompt_tokens=100,
                completion_tokens=50,
            )
        )

        assert collector.total_tokens > 0
        collector.reset()
        assert collector.total_tokens == 0
        assert collector.request_count == 0

    def test_metrics_tokens_per_sec(self):
        collector = MetricsCollector()

        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                endpoint_name="nim-super",
                prompt_tokens=100,
                completion_tokens=200,
                total_time_ms=1000.0,
            )
        )

        assert collector.avg_tokens_per_sec == 200.0
        assert collector.last_tokens_per_sec == 200.0
