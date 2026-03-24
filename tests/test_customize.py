# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for ``nemo customize`` — LoRA fine-tuning CLI and CustomizerClient."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from nemocode.cli.main import app
from nemocode.providers.nvidia_client import (
    CustomizerAPIError,
    CustomizerClient,
)

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


# -----------------------------------------------------------------------
# CustomizerClient unit tests
# -----------------------------------------------------------------------


class TestCustomizerClient:
    """Unit tests for the REST client itself (mocked httpx)."""

    def _mock_response(
        self, status_code: int = 200, json_data: dict | list | None = None, text: str = ""
    ) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.text = text or ""
        resp.json.return_value = json_data if json_data is not None else {}
        return resp

    # -- create_job --

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_create_job_success(self, mock_client_cls):
        resp_data = {"id": "job-001", "status": "pending", "model": "nvidia/nemotron-3-nano-4b-v1.1"}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_response(200, resp_data)
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000", api_key="test-key")
        result = client.create_job("/data/sft.jsonl")

        assert result["id"] == "job-001"
        assert result["status"] == "pending"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/v2/nemo/customization/jobs" in call_args[0][0]

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_create_job_api_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        error_resp = self._mock_response(422, {"detail": "Invalid dataset"}, "Invalid dataset")
        mock_client.post.return_value = error_resp
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        with pytest.raises(CustomizerAPIError) as exc_info:
            client.create_job("/bad/path.jsonl")
        assert exc_info.value.status_code == 422

    # -- list_jobs --

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_list_jobs_success(self, mock_client_cls):
        resp_data = {
            "jobs": [
                {"id": "job-001", "model": "m1", "status": "completed", "created_at": "2026-01-01"},
                {"id": "job-002", "model": "m2", "status": "running", "created_at": "2026-01-02"},
            ]
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, resp_data)
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        jobs = client.list_jobs()
        assert len(jobs) == 2
        assert jobs[0]["id"] == "job-001"

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_list_jobs_bare_list(self, mock_client_cls):
        """API may return a bare list instead of {jobs: [...]}."""
        resp_data = [{"id": "job-001", "status": "pending"}]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, resp_data)
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        jobs = client.list_jobs()
        assert len(jobs) == 1

    # -- get_status --

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_get_status_success(self, mock_client_cls):
        resp_data = {
            "id": "job-001",
            "status": "running",
            "model": "nvidia/nemotron-3-nano-4b-v1.1",
            "progress": 42,
            "created_at": "2026-01-01T00:00:00Z",
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, resp_data)
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        result = client.get_status("job-001")
        assert result["status"] == "running"
        assert result["progress"] == 42

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_get_status_not_found(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(
            404, {"detail": "Job not found"}, "Job not found"
        )
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        with pytest.raises(CustomizerAPIError) as exc_info:
            client.get_status("nonexistent")
        assert exc_info.value.status_code == 404

    # -- get_logs --

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_get_logs_success(self, mock_client_cls):
        resp_data = {"logs": ["epoch 1: loss=2.3", "epoch 2: loss=1.8"]}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, resp_data)
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        result = client.get_logs("job-001")
        assert len(result["logs"]) == 2

    # -- get_results --

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_get_results_success(self, mock_client_cls):
        resp_data = {
            "output_model": "my-lora-model",
            "metrics": {"final_loss": 0.5, "eval_accuracy": 0.92},
            "artifacts": ["/models/my-lora-model/adapter.safetensors"],
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, resp_data)
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000")
        result = client.get_results("job-001")
        assert result["output_model"] == "my-lora-model"
        assert result["metrics"]["eval_accuracy"] == 0.92

    # -- auth header --

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_auth_header_set(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, {"jobs": []})
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000", api_key="my-secret")
        client.list_jobs()

        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-secret"

    @patch("nemocode.providers.nvidia_client.httpx.Client")
    def test_no_auth_header_when_no_key(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = self._mock_response(200, {"jobs": []})
        mock_client_cls.return_value = mock_client

        client = CustomizerClient(base_url="http://test:8000", api_key=None)
        # Clear env to be safe
        with patch.dict("os.environ", {}, clear=True):
            client._api_key = None
            client.list_jobs()

        call_kwargs = mock_client.get.call_args[1]
        assert "Authorization" not in call_kwargs["headers"]


# -----------------------------------------------------------------------
# CLI command tests
# -----------------------------------------------------------------------


class TestCustomizeCLIHelp:
    """Smoke tests for CLI help output."""

    def test_customize_help(self):
        result = runner.invoke(app, ["customize", "--help"])
        assert result.exit_code == 0
        assert "LoRA" in _strip_ansi(result.stdout)

    def test_customize_create_help(self):
        result = runner.invoke(app, ["customize", "create", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in _strip_ansi(result.stdout)

    def test_customize_status_help(self):
        result = runner.invoke(app, ["customize", "status", "--help"])
        assert result.exit_code == 0

    def test_customize_logs_help(self):
        result = runner.invoke(app, ["customize", "logs", "--help"])
        assert result.exit_code == 0

    def test_customize_results_help(self):
        result = runner.invoke(app, ["customize", "results", "--help"])
        assert result.exit_code == 0

    def test_customize_ls_help(self):
        result = runner.invoke(app, ["customize", "ls", "--help"])
        assert result.exit_code == 0


class TestCustomizeCreate:
    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_create_success(self, mock_cls):
        mock_client = MagicMock()
        mock_client.create_job.return_value = {
            "id": "job-abc",
            "status": "pending",
        }
        mock_cls.return_value = mock_client

        result = runner.invoke(
            app,
            ["customize", "create", "--dataset", "/data/sft.jsonl"],
        )
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "job-abc" in out
        assert "pending" in out

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_create_custom_params(self, mock_cls):
        mock_client = MagicMock()
        mock_client.create_job.return_value = {"id": "job-xyz", "status": "pending"}
        mock_cls.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "customize",
                "create",
                "--dataset",
                "/data/sft.jsonl",
                "--lora-rank",
                "32",
                "--epochs",
                "5",
                "--batch-size",
                "8",
                "--lr",
                "5e-5",
                "--output-model",
                "my-model",
            ],
        )
        assert result.exit_code == 0
        mock_client.create_job.assert_called_once_with(
            dataset_path="/data/sft.jsonl",
            model="nvidia/nemotron-3-nano-4b-v1.1",
            lora_rank=32,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            output_model="my-model",
        )

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_create_api_error(self, mock_cls):
        mock_client = MagicMock()
        mock_client.create_job.side_effect = CustomizerAPIError(422, "Bad dataset format")
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "create", "--dataset", "/data/bad.jsonl"])
        assert result.exit_code == 1
        assert "422" in _strip_ansi(result.stdout) or "Validation" in _strip_ansi(result.stdout)

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_create_connection_error(self, mock_cls):
        mock_client = MagicMock()
        mock_client.create_job.side_effect = httpx.ConnectError("Connection refused")
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "create", "--dataset", "/data/sft.jsonl"])
        assert result.exit_code == 1
        assert "Could not connect" in _strip_ansi(result.stdout)


class TestCustomizeStatus:
    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_status_success(self, mock_cls):
        mock_client = MagicMock()
        mock_client.get_status.return_value = {
            "id": "job-001",
            "status": "running",
            "model": "nvidia/nemotron-3-nano-4b-v1.1",
            "created_at": "2026-01-01T00:00:00Z",
            "progress": 55,
            "hyperparameters": {"lora_rank": 16, "epochs": 3},
        }
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "status", "job-001"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "running" in out
        assert "55%" in out
        assert "lora_rank" in out

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_status_not_found(self, mock_cls):
        mock_client = MagicMock()
        mock_client.get_status.side_effect = CustomizerAPIError(404, "Job not found")
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "status", "nonexistent"])
        assert result.exit_code == 1
        assert "Not found" in _strip_ansi(result.stdout)


class TestCustomizeLogs:
    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_logs_success(self, mock_cls):
        mock_client = MagicMock()
        mock_client.get_logs.return_value = {
            "logs": ["epoch 1: loss=2.3", "epoch 2: loss=1.8", "epoch 3: loss=1.2"]
        }
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "logs", "job-001"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "epoch 1" in out
        assert "loss=1.2" in out

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_logs_empty(self, mock_cls):
        mock_client = MagicMock()
        mock_client.get_logs.return_value = {"logs": []}
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "logs", "job-001"])
        assert result.exit_code == 0
        assert "No logs" in _strip_ansi(result.stdout)

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_logs_string_format(self, mock_cls):
        """Some backends return logs as a single string."""
        mock_client = MagicMock()
        mock_client.get_logs.return_value = {"logs": "line1\nline2\nline3"}
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "logs", "job-001"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "line1" in out
        assert "line3" in out


class TestCustomizeResults:
    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_results_success(self, mock_cls):
        mock_client = MagicMock()
        mock_client.get_results.return_value = {
            "output_model": "my-lora-nano",
            "metrics": {"final_loss": 0.5, "eval_accuracy": 0.92},
            "artifacts": ["/models/my-lora-nano/adapter.safetensors"],
        }
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "results", "job-001"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "my-lora-nano" in out
        assert "final_loss" in out
        assert "0.92" in out

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_results_not_found(self, mock_cls):
        mock_client = MagicMock()
        mock_client.get_results.side_effect = CustomizerAPIError(404, "Job not found")
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "results", "nonexistent"])
        assert result.exit_code == 1


class TestCustomizeLs:
    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_ls_success(self, mock_cls):
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            {
                "id": "job-001",
                "model": "nvidia/nemotron-3-nano-4b-v1.1",
                "status": "completed",
                "created_at": "2026-01-01",
                "progress": 100,
            },
            {
                "id": "job-002",
                "model": "nvidia/nemotron-3-nano-4b-v1.1",
                "status": "running",
                "created_at": "2026-01-02",
                "progress": 42,
            },
        ]
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "ls"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "job-001" in out
        assert "job-002" in out
        assert "completed" in out
        assert "42%" in out

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_ls_empty(self, mock_cls):
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = []
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "ls"])
        assert result.exit_code == 0
        assert "No customization jobs" in _strip_ansi(result.stdout)

    @patch("nemocode.cli.commands.customize.CustomizerClient")
    def test_ls_connection_error(self, mock_cls):
        mock_client = MagicMock()
        mock_client.list_jobs.side_effect = httpx.ConnectError("Connection refused")
        mock_cls.return_value = mock_client

        result = runner.invoke(app, ["customize", "ls"])
        assert result.exit_code == 1
        assert "Could not connect" in _strip_ansi(result.stdout)


class TestCustomizerAPIError:
    def test_error_message(self):
        err = CustomizerAPIError(500, "Internal server error")
        assert "500" in str(err)
        assert "Internal server error" in str(err)
        assert err.status_code == 500
        assert err.detail == "Internal server error"

    def test_auth_error_display(self):
        """CLI should show auth-specific message for 401."""
        with patch("nemocode.cli.commands.customize.CustomizerClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_jobs.side_effect = CustomizerAPIError(401, "Unauthorized")
            mock_cls.return_value = mock_client

            result = runner.invoke(app, ["customize", "ls"])
            assert result.exit_code == 1
            assert "Authentication" in _strip_ansi(result.stdout) or "NVIDIA_API_KEY" in _strip_ansi(
                result.stdout
            )
