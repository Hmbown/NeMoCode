# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for throughput metrics (tokens_per_sec, avg/last/ttft)."""

from __future__ import annotations

from nemocode.core.metrics import MetricsCollector, RequestMetrics


class TestRequestMetricsThroughput:
    def test_tokens_per_sec_basic(self):
        r = RequestMetrics(completion_tokens=500, total_time_ms=2000)
        assert r.tokens_per_sec == 250.0

    def test_tokens_per_sec_zero_time(self):
        r = RequestMetrics(completion_tokens=500, total_time_ms=0)
        assert r.tokens_per_sec == 0.0

    def test_tokens_per_sec_zero_tokens(self):
        r = RequestMetrics(completion_tokens=0, total_time_ms=1000)
        assert r.tokens_per_sec == 0.0

    def test_tokens_per_sec_both_zero(self):
        r = RequestMetrics()
        assert r.tokens_per_sec == 0.0

    def test_tokens_per_sec_high_throughput(self):
        # 1000 tokens in 500ms = 2000 tok/s
        r = RequestMetrics(completion_tokens=1000, total_time_ms=500)
        assert r.tokens_per_sec == 2000.0

    def test_tokens_per_sec_slow(self):
        # 10 tokens in 10s = 1 tok/s
        r = RequestMetrics(completion_tokens=10, total_time_ms=10_000)
        assert r.tokens_per_sec == 1.0


class TestCollectorThroughput:
    def test_avg_tokens_per_sec_empty(self):
        mc = MetricsCollector()
        assert mc.avg_tokens_per_sec == 0.0

    def test_avg_tokens_per_sec_single(self):
        mc = MetricsCollector()
        mc.record(RequestMetrics(completion_tokens=500, total_time_ms=2000))
        assert mc.avg_tokens_per_sec == 250.0

    def test_avg_tokens_per_sec_multiple(self):
        mc = MetricsCollector()
        mc.record(RequestMetrics(completion_tokens=500, total_time_ms=2000))  # 250
        mc.record(RequestMetrics(completion_tokens=1000, total_time_ms=2000))  # 500
        assert mc.avg_tokens_per_sec == 375.0

    def test_avg_tokens_per_sec_skips_zero(self):
        """Requests with 0 completion tokens should not count in the average."""
        mc = MetricsCollector()
        mc.record(RequestMetrics(completion_tokens=500, total_time_ms=2000))  # 250
        mc.record(RequestMetrics(completion_tokens=0, total_time_ms=1000))  # 0 — skipped
        assert mc.avg_tokens_per_sec == 250.0

    def test_last_tokens_per_sec_empty(self):
        mc = MetricsCollector()
        assert mc.last_tokens_per_sec == 0.0

    def test_last_tokens_per_sec(self):
        mc = MetricsCollector()
        mc.record(RequestMetrics(completion_tokens=500, total_time_ms=2000))
        mc.record(RequestMetrics(completion_tokens=1000, total_time_ms=2000))
        assert mc.last_tokens_per_sec == 500.0

    def test_avg_ttft_ms_empty(self):
        mc = MetricsCollector()
        assert mc.avg_ttft_ms == 0.0

    def test_avg_ttft_ms_single(self):
        mc = MetricsCollector()
        mc.record(RequestMetrics(time_to_first_token_ms=300))
        assert mc.avg_ttft_ms == 300.0

    def test_avg_ttft_ms_skips_zero(self):
        mc = MetricsCollector()
        mc.record(RequestMetrics(time_to_first_token_ms=300))
        mc.record(RequestMetrics(time_to_first_token_ms=0))  # skipped
        mc.record(RequestMetrics(time_to_first_token_ms=500))
        assert mc.avg_ttft_ms == 400.0

    def test_summary_includes_throughput_fields(self):
        mc = MetricsCollector()
        mc.record(
            RequestMetrics(
                completion_tokens=500,
                total_time_ms=2000,
                time_to_first_token_ms=300,
            )
        )
        s = mc.summary()
        assert "avg_tokens_per_sec" in s
        assert "last_tokens_per_sec" in s
        assert "avg_ttft_ms" in s
        assert s["avg_tokens_per_sec"] == 250.0
        assert s["last_tokens_per_sec"] == 250.0
        assert s["avg_ttft_ms"] == 300.0

    def test_summary_throughput_zero_when_empty(self):
        mc = MetricsCollector()
        s = mc.summary()
        assert s["avg_tokens_per_sec"] == 0.0
        assert s["last_tokens_per_sec"] == 0.0
        assert s["avg_ttft_ms"] == 0.0

    def test_reset_clears_throughput(self):
        mc = MetricsCollector()
        mc.record(RequestMetrics(completion_tokens=500, total_time_ms=2000))
        mc.reset()
        assert mc.avg_tokens_per_sec == 0.0
        assert mc.last_tokens_per_sec == 0.0
