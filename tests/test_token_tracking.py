# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for token tracking — metrics grouping, SQLite persistence, and CLI."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nemocode.core.metrics import MetricsCollector, RequestMetrics


# ---------------------------------------------------------------------------
# MetricsCollector.usage_by_model / usage_by_endpoint / to_records
# ---------------------------------------------------------------------------


class TestUsageGrouping:
    def _make_collector(self) -> MetricsCollector:
        collector = MetricsCollector()
        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                endpoint_name="nim-super",
                prompt_tokens=1000,
                completion_tokens=500,
                total_time_ms=2000.0,
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
        collector.record(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                endpoint_name="nim-super",
                prompt_tokens=200,
                completion_tokens=100,
                total_time_ms=800.0,
            )
        )
        return collector

    def test_usage_by_model_keys(self):
        collector = self._make_collector()
        by_model = collector.usage_by_model
        assert set(by_model.keys()) == {
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-nano-30b-a3b",
        }

    def test_usage_by_model_counts(self):
        collector = self._make_collector()
        by_model = collector.usage_by_model
        super_group = by_model["nvidia/nemotron-3-super-120b-a12b"]
        assert super_group["requests"] == 2
        assert super_group["prompt_tokens"] == 1200
        assert super_group["completion_tokens"] == 600
        assert super_group["total_tokens"] == 1800
        assert super_group["estimated_cost_usd"] > 0

    def test_usage_by_model_nano(self):
        collector = self._make_collector()
        by_model = collector.usage_by_model
        nano_group = by_model["nvidia/nemotron-3-nano-30b-a3b"]
        assert nano_group["requests"] == 1
        assert nano_group["prompt_tokens"] == 300
        assert nano_group["completion_tokens"] == 100
        assert nano_group["total_tokens"] == 400

    def test_usage_by_endpoint_keys(self):
        collector = self._make_collector()
        by_endpoint = collector.usage_by_endpoint
        assert set(by_endpoint.keys()) == {"nim-super", "nim-nano"}

    def test_usage_by_endpoint_counts(self):
        collector = self._make_collector()
        by_endpoint = collector.usage_by_endpoint
        assert by_endpoint["nim-super"]["requests"] == 2
        assert by_endpoint["nim-super"]["total_tokens"] == 1800
        assert by_endpoint["nim-nano"]["requests"] == 1
        assert by_endpoint["nim-nano"]["total_tokens"] == 400

    def test_usage_by_model_empty(self):
        collector = MetricsCollector()
        assert collector.usage_by_model == {}

    def test_usage_by_endpoint_empty(self):
        collector = MetricsCollector()
        assert collector.usage_by_endpoint == {}

    def test_to_records_length(self):
        collector = self._make_collector()
        records = collector.to_records()
        assert len(records) == 3

    def test_to_records_fields(self):
        collector = self._make_collector()
        records = collector.to_records()
        r = records[0]
        assert r["model_id"] == "nvidia/nemotron-3-super-120b-a12b"
        assert r["endpoint_name"] == "nim-super"
        assert r["prompt_tokens"] == 1000
        assert r["completion_tokens"] == 500
        assert r["total_tokens"] == 1500
        assert r["total_time_ms"] == 2000.0
        assert r["estimated_cost_usd"] > 0
        assert "timestamp" in r


# ---------------------------------------------------------------------------
# SQLite round-trip tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Point the sqlite_store at a temporary database."""
    import nemocode.core.sqlite_store as store

    db_path = tmp_path / "test_sessions.db"
    store._override_db_path(db_path)
    yield db_path
    # Restore default (not critical for tests, but tidy)
    store._override_db_path(Path("~/.local/share/nemocode/sessions.db").expanduser())


class TestSQLiteMetrics:
    def test_save_and_load(self, tmp_db: Path):
        from nemocode.core.sqlite_store import (
            load_request_metrics,
            save_request_metrics,
        )

        m = RequestMetrics(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            endpoint_name="nim-super",
            prompt_tokens=500,
            completion_tokens=200,
            total_time_ms=1000.0,
            timestamp=time.time(),
        )
        save_request_metrics(m)

        rows = load_request_metrics(limit=10)
        assert len(rows) == 1
        assert rows[0]["model_id"] == "nvidia/nemotron-3-super-120b-a12b"
        assert rows[0]["prompt_tokens"] == 500
        assert rows[0]["completion_tokens"] == 200
        assert rows[0]["total_tokens"] == 700

    def test_load_filtered_by_model(self, tmp_db: Path):
        from nemocode.core.sqlite_store import (
            load_request_metrics,
            save_request_metrics,
        )

        now = time.time()
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                prompt_tokens=100,
                completion_tokens=50,
                timestamp=now,
            )
        )
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                prompt_tokens=200,
                completion_tokens=80,
                timestamp=now + 1,
            )
        )

        all_rows = load_request_metrics(limit=10)
        assert len(all_rows) == 2

        super_rows = load_request_metrics(limit=10, model_id="nvidia/nemotron-3-super-120b-a12b")
        assert len(super_rows) == 1
        assert super_rows[0]["prompt_tokens"] == 100

    def test_usage_summary(self, tmp_db: Path):
        from nemocode.core.sqlite_store import (
            load_usage_summary,
            save_request_metrics,
        )

        now = time.time()
        for i in range(3):
            save_request_metrics(
                RequestMetrics(
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    prompt_tokens=100,
                    completion_tokens=50,
                    timestamp=now + i,
                )
            )
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                prompt_tokens=200,
                completion_tokens=80,
                timestamp=now + 3,
            )
        )

        summary = load_usage_summary()
        assert len(summary) == 2
        # Sorted by total_tokens DESC — super has 450 total, nano has 280
        assert summary[0]["model_id"] == "nvidia/nemotron-3-super-120b-a12b"
        assert summary[0]["requests"] == 3
        assert summary[0]["prompt_tokens"] == 300
        assert summary[0]["completion_tokens"] == 150
        assert summary[0]["total_tokens"] == 450

    def test_usage_summary_since(self, tmp_db: Path):
        from nemocode.core.sqlite_store import (
            load_usage_summary,
            save_request_metrics,
        )

        now = time.time()
        # Old record
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                prompt_tokens=100,
                completion_tokens=50,
                timestamp=now - 3600,  # 1 hour ago
            )
        )
        # Recent record
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                prompt_tokens=200,
                completion_tokens=80,
                timestamp=now,
            )
        )

        # Only get records from the last 30 minutes
        summary = load_usage_summary(since=now - 1800)
        assert len(summary) == 1
        assert summary[0]["model_id"] == "nvidia/nemotron-3-nano-30b-a3b"


# ---------------------------------------------------------------------------
# CLI obs usage command
# ---------------------------------------------------------------------------


class TestObsUsageCLI:
    def test_usage_no_data(self, tmp_db: Path):
        from nemocode.cli.commands.obs import obs_app

        runner = CliRunner()
        result = runner.invoke(obs_app, ["usage"])
        assert result.exit_code == 0
        assert "No usage data" in result.output

    def test_usage_with_data(self, tmp_db: Path):
        from nemocode.core.sqlite_store import save_request_metrics

        from nemocode.cli.commands.obs import obs_app

        now = time.time()
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                prompt_tokens=1000,
                completion_tokens=500,
                timestamp=now,
            )
        )
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                prompt_tokens=300,
                completion_tokens=100,
                timestamp=now + 1,
            )
        )

        runner = CliRunner()
        result = runner.invoke(obs_app, ["usage"])
        assert result.exit_code == 0
        assert "Total" in result.output
        assert "1,000" in result.output

    def test_usage_with_since(self, tmp_db: Path):
        from nemocode.core.sqlite_store import save_request_metrics

        from nemocode.cli.commands.obs import obs_app

        now = time.time()
        save_request_metrics(
            RequestMetrics(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                prompt_tokens=500,
                completion_tokens=200,
                timestamp=now,
            )
        )

        runner = CliRunner()
        result = runner.invoke(obs_app, ["usage", "--since", "24h"])
        assert result.exit_code == 0
        assert "last 24h" in result.output
