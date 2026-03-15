# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for /doctor diagnostics."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from nemocode.config.schema import (
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    NeMoCodeConfig,
)
from nemocode.core.doctor import DiagnosticReport, run_diagnostics


@pytest.fixture
def config():
    return NeMoCodeConfig(
        default_endpoint="nim-super",
        endpoints={
            "nim-super": Endpoint(
                name="test",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key_env="NVIDIA_API_KEY",
                model_id="nvidia/nemotron-3-super",
            ),
        },
        formations={
            "solo": Formation(
                name="Solo",
                slots=[
                    FormationSlot(endpoint="nim-super", role=FormationRole.EXECUTOR),
                ],
            ),
        },
    )


class TestDiagnosticReport:
    def test_add_check(self):
        report = DiagnosticReport()
        report.add("test", "ok", "details")
        assert len(report.checks) == 1
        assert report.checks[0].name == "test"
        assert report.checks[0].status == "ok"

    def test_ok_all_pass(self):
        report = DiagnosticReport()
        report.add("a", "ok")
        report.add("b", "ok")
        assert report.ok is True

    def test_ok_fails_on_fail(self):
        report = DiagnosticReport()
        report.add("a", "ok")
        report.add("b", "fail")
        assert report.ok is False


class TestRunDiagnostics:
    def test_basic_run(self, config):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test123456789012"}):
            report = run_diagnostics(config)
            assert len(report.checks) > 0
            # API key should be found
            api_check = next(c for c in report.checks if c.name == "api_key")
            assert api_check.status == "ok"

    def test_missing_api_key(self, config):
        env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            report = run_diagnostics(config)
            api_check = next(c for c in report.checks if c.name == "api_key")
            assert api_check.status == "fail"

    def test_config_validation(self, config):
        report = run_diagnostics(config)
        cfg_check = next(c for c in report.checks if c.name == "config")
        assert cfg_check.status == "ok"

    def test_config_bad_formation_ref(self):
        config = NeMoCodeConfig(
            default_endpoint="nim-super",
            endpoints={
                "nim-super": Endpoint(
                    name="test",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://test.com/v1",
                    model_id="test",
                ),
            },
            formations={
                "bad": Formation(
                    name="Bad",
                    slots=[
                        FormationSlot(endpoint="nonexistent", role=FormationRole.EXECUTOR),
                    ],
                ),
            },
        )
        report = run_diagnostics(config)
        cfg_check = next(c for c in report.checks if c.name == "config")
        assert cfg_check.status == "warn"
