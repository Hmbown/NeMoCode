# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""CLI tests for NeMo microservice command surfaces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from nemocode.cli.main import app
from nemocode.config.schema import Endpoint, EndpointTier, GuardrailsConfig
from nemocode.providers.nim_guardrails import SafetyResult

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestEvaluatorCLI:
    def test_help(self):
        result = runner.invoke(app, ["evaluator", "--help"])
        assert result.exit_code == 0
        assert "Evaluator" in _strip(result.stdout)

    def test_create(self, tmp_path: Path):
        spec = tmp_path / "eval.json"
        spec.write_text(json.dumps({"target": {"name": "demo"}, "config": {"metrics": ["acc"]}}))

        with patch("nemocode.cli.commands.evaluator.EvaluatorClient") as mock_cls:
            mock_cls.return_value.create_job.return_value = {"id": "eval-1", "status": "pending"}
            result = runner.invoke(app, ["evaluator", "create", str(spec)])

        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "Evaluator Job Created" in out
        assert "eval-1" in out

    def test_results_to_file(self, tmp_path: Path):
        dest = tmp_path / "results.json"

        with patch("nemocode.cli.commands.evaluator.EvaluatorClient") as mock_cls:
            mock_cls.return_value.get_job_results.return_value = {"summary": {"score": 0.91}}
            result = runner.invoke(app, ["evaluator", "results", "eval-1", "--output", str(dest)])

        assert result.exit_code == 0
        assert dest.exists()
        assert json.loads(dest.read_text())["summary"]["score"] == 0.91


class TestSafeSynthCLI:
    def test_help(self):
        result = runner.invoke(app, ["safe-synth", "--help"])
        assert result.exit_code == 0
        assert "Safe Synthesizer" in _strip(result.stdout)

    def test_create(self, tmp_path: Path):
        spec = tmp_path / "safe.json"
        spec.write_text(json.dumps({"spec": {"input": {"type": "csv"}}}))

        with patch("nemocode.cli.commands.safe_synth.SafeSynthesizerClient") as mock_cls:
            mock_cls.return_value.create_job.return_value = {"id": "ss-1", "status": "pending"}
            result = runner.invoke(app, ["safe-synth", "create", str(spec)])

        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "Safe Synthesizer Job Created" in out
        assert "ss-1" in out

    def test_download_results(self, tmp_path: Path):
        dest = tmp_path / "synthetic.jsonl"

        with patch("nemocode.cli.commands.safe_synth.SafeSynthesizerClient") as mock_cls:
            mock_cls.return_value.download_synthetic_data.return_value = b'{"row":1}\n'
            result = runner.invoke(
                app,
                [
                    "safe-synth",
                    "results",
                    "ss-1",
                    "--download",
                    "--output",
                    str(dest),
                ],
            )

        assert result.exit_code == 0
        assert dest.exists()
        assert dest.read_text() == '{"row":1}\n'


class TestEntityCLI:
    def test_help(self):
        result = runner.invoke(app, ["entity", "--help"])
        assert result.exit_code == 0
        assert "Entity Store" in _strip(result.stdout)

    def test_namespace_create(self):
        with patch("nemocode.cli.commands.entity.EntityStoreClient") as mock_cls:
            mock_cls.return_value.create_resource.return_value = {"id": "ns-1", "name": "default"}
            result = runner.invoke(app, ["entity", "namespace", "create", "default"])

        assert result.exit_code == 0
        assert '"name": "default"' in _strip(result.stdout)

    def test_dataset_create(self):
        with patch("nemocode.cli.commands.entity.EntityStoreClient") as mock_cls:
            mock_cls.return_value.create_resource.return_value = {"id": "ds-1", "name": "train"}
            result = runner.invoke(
                app,
                [
                    "entity",
                    "dataset",
                    "create",
                    "train",
                    "--namespace",
                    "default",
                    "--project",
                    "nemocode",
                ],
            )

        assert result.exit_code == 0
        assert '"id": "ds-1"' in _strip(result.stdout)


class TestAuditorCLI:
    def test_help(self):
        result = runner.invoke(app, ["auditor", "--help"])
        assert result.exit_code == 0
        assert "Auditor" in _strip(result.stdout)

    def test_target_create(self, tmp_path: Path):
        spec = tmp_path / "target.json"
        spec.write_text(json.dumps({"name": "demo-target", "type": "model"}))

        with patch("nemocode.cli.commands.auditor.AuditorClient") as mock_cls:
            mock_cls.return_value.create_target.return_value = {"id": "t-1", "name": "demo-target"}
            result = runner.invoke(app, ["auditor", "target", "create", str(spec)])

        assert result.exit_code == 0
        assert '"id": "t-1"' in _strip(result.stdout)

    def test_job_results_download(self, tmp_path: Path):
        dest = tmp_path / "audit-report.bin"

        with patch("nemocode.cli.commands.auditor.AuditorClient") as mock_cls:
            mock_cls.return_value.download_report.return_value = b"audit-report"
            result = runner.invoke(
                app,
                ["auditor", "job", "results", "job-1", "--download", "--output", str(dest)],
            )

        assert result.exit_code == 0
        assert dest.exists()
        assert dest.read_bytes() == b"audit-report"


class TestGuardrailsCLI:
    def test_help(self):
        result = runner.invoke(app, ["guardrails", "--help"])
        assert result.exit_code == 0
        assert "Guardrails" in _strip(result.stdout)

    def test_config(self):
        endpoint = Endpoint(
            name="nim-content-safety",
            tier=EndpointTier.DEV_HOSTED,
            base_url="https://example.com/v1",
            api_key_env="NVIDIA_API_KEY",
            model_id="nvidia/nemo-guardrails",
        )

        class DummyConfig:
            guardrails = GuardrailsConfig(enabled=True, endpoint="nim-content-safety")
            endpoints = {"nim-content-safety": endpoint}

        with patch("nemocode.cli.commands.guardrails.load_config", return_value=DummyConfig()):
            result = runner.invoke(app, ["guardrails", "config"])

        assert result.exit_code == 0
        assert "nim-content-safety" in _strip(result.stdout)

    def test_check_json(self):
        endpoint = Endpoint(
            name="nim-content-safety",
            tier=EndpointTier.DEV_HOSTED,
            base_url="https://example.com/v1",
            api_key_env="NVIDIA_API_KEY",
            model_id="nvidia/nemo-guardrails",
        )

        class DummyConfig:
            guardrails = GuardrailsConfig(enabled=True, endpoint="nim-content-safety")
            endpoints = {"nim-content-safety": endpoint}

        with patch("nemocode.cli.commands.guardrails.load_config", return_value=DummyConfig()):
            with patch("nemocode.cli.commands.guardrails.get_api_key", return_value="key"):
                with patch("nemocode.cli.commands.guardrails.NIMGuardrailsProvider") as mock_cls:
                    mock_cls.return_value.check = AsyncMock(
                        return_value=SafetyResult(
                            safe=False,
                            blocked_categories=["violence"],
                            categories={"violence": 0.91},
                        )
                    )
                    result = runner.invoke(app, ["guardrails", "check", "bad text", "--json"])

        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert '"safe": false' in out
        assert '"violence"' in out
