# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for ``nemo serve`` — local LoRA adapter serving."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemocode.cli.main import app
from nemocode.cli.commands.serve import (
    _build_sglang_command,
    _build_vllm_command,
    _health_check,
    _pick_backend,
    _read_state,
    _register_endpoint,
    _resolve_model_id,
    _write_state,
)

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


# -----------------------------------------------------------------------
# Unit tests for helper functions
# -----------------------------------------------------------------------


class TestResolveModelId:
    def test_nano_4b_alias(self):
        assert _resolve_model_id("nemotron-3-nano-4b") == "nvidia/llama-3.1-nemotron-nano-4b-v1.1"

    def test_nano_9b_alias(self):
        assert _resolve_model_id("nemotron-3-nano-9b") == "nvidia/nemotron-nano-9b-v2"

    def test_nano_30b_alias(self):
        assert _resolve_model_id("nemotron-3-nano-30b") == "nvidia/nemotron-3-nano-30b-a3b"

    def test_super_alias(self):
        assert _resolve_model_id("nemotron-3-super") == "nvidia/nemotron-3-super-120b-a12b"

    def test_passthrough_unknown(self):
        assert _resolve_model_id("my-custom-model") == "my-custom-model"

    def test_passthrough_hf_id(self):
        assert _resolve_model_id("nvidia/some-model") == "nvidia/some-model"


class TestPickBackend:
    @patch("nemocode.cli.commands.serve._find_executable")
    def test_spark_prefers_sglang(self, mock_find):
        mock_find.side_effect = lambda n: "/usr/bin/sglang" if n == "sglang" else None
        assert _pick_backend(True) == "sglang"

    @patch("nemocode.cli.commands.serve._find_executable")
    def test_spark_falls_back_to_vllm(self, mock_find):
        mock_find.side_effect = lambda n: "/usr/bin/vllm" if n == "vllm" else None
        assert _pick_backend(True) == "vllm"

    @patch("nemocode.cli.commands.serve._find_executable")
    def test_non_spark_uses_vllm(self, mock_find):
        mock_find.side_effect = lambda n: "/usr/bin/vllm" if n == "vllm" else None
        assert _pick_backend(False) == "vllm"

    @patch("nemocode.cli.commands.serve._find_executable")
    def test_no_backend(self, mock_find):
        mock_find.return_value = None
        assert _pick_backend(True) == "unknown"
        assert _pick_backend(False) == "unknown"


class TestBuildVllmCommand:
    def test_basic_command(self):
        cmd = _build_vllm_command(
            "nvidia/llama-3.1-nemotron-nano-4b-v1.1", Path("/tmp/adapter"), 8010
        )
        assert cmd[0] == "vllm"
        assert cmd[1] == "serve"
        assert "nvidia/llama-3.1-nemotron-nano-4b-v1.1" in cmd
        assert "--port" in cmd
        assert "8010" in cmd
        assert "--enable-lora" in cmd
        assert "--lora-modules" in cmd

    def test_custom_max_model_len(self):
        cmd = _build_vllm_command("model", Path("/a"), 8000, max_model_len=131072)
        assert "131072" in cmd


class TestBuildSglangCommand:
    def test_basic_command(self):
        cmd = _build_sglang_command(
            "nvidia/llama-3.1-nemotron-nano-4b-v1.1", Path("/tmp/adapter"), 8010
        )
        assert cmd[0] == "python"
        assert "sglang.launch_server" in cmd
        assert "--port" in cmd
        assert "8010" in cmd


# -----------------------------------------------------------------------
# Health check tests
# -----------------------------------------------------------------------


class TestHealthCheck:
    @patch("nemocode.cli.commands.serve.httpx")
    def test_health_check_success_immediate(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx.get.return_value = mock_resp
        with patch("nemocode.cli.commands.serve.time"):
            assert _health_check("http://localhost:8010/v1", timeout=10) is True

    @patch("nemocode.cli.commands.serve.httpx")
    def test_health_check_eventual_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx.get.side_effect = [
            Exception("Connection refused"),
            Exception("Connection refused"),
            mock_resp,
        ]
        with patch("nemocode.cli.commands.serve.time"):
            assert _health_check("http://localhost:8010/v1", timeout=10) is True

    @patch("nemocode.cli.commands.serve.httpx")
    def test_health_check_timeout(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("Connection refused")
        with patch("nemocode.cli.commands.serve.time") as mock_time:
            assert _health_check("http://localhost:8010/v1", timeout=2) is False

    @patch("nemocode.cli.commands.serve.httpx")
    def test_health_check_non_200(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_httpx.get.return_value = mock_resp
        with patch("nemocode.cli.commands.serve.time") as mock_time:
            assert _health_check("http://localhost:8010/v1", timeout=2) is False


# -----------------------------------------------------------------------
# State management tests
# -----------------------------------------------------------------------


class TestServeState:
    def test_write_state(self):
        with patch("nemocode.cli.commands.serve.ensure_config_dir"):
            with patch("nemocode.cli.commands.serve._SERVE_STATE_FILE") as mock_file:
                _write_state({"pid": 12345, "port": 8010})
                mock_file.write_text.assert_called_once()

    def test_read_state_no_file(self):
        with patch("nemocode.cli.commands.serve._SERVE_STATE_FILE") as mock_file:
            mock_file.exists.return_value = False
            assert _read_state() is None

    def test_read_state_valid(self):
        state = {"pid": 12345, "port": 8010}
        with patch("nemocode.cli.commands.serve._SERVE_STATE_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = json.dumps(state)
            assert _read_state() == state

    def test_read_state_invalid_json(self):
        with patch("nemocode.cli.commands.serve._SERVE_STATE_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = "not json"
            assert _read_state() is None


# -----------------------------------------------------------------------
# Endpoint registration tests
# -----------------------------------------------------------------------


class TestRegisterEndpoint:
    def test_registers_endpoint_in_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        with patch("nemocode.cli.commands.serve.ensure_config_dir"):
            with patch("nemocode.cli.commands.serve._USER_CONFIG_PATH", config_path):
                _register_endpoint(
                    "my-serve", "http://localhost:8010/v1", "nvidia/model", "local-vllm"
                )

        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert "endpoints" in cfg
        assert "my-serve" in cfg["endpoints"]
        ep = cfg["endpoints"]["my-serve"]
        assert ep["base_url"] == "http://localhost:8010/v1"
        assert ep["model_id"] == "nvidia/model"
        assert ep["tier"] == "local-vllm"
        assert "chat" in ep["capabilities"]

    def test_preserves_existing_endpoints(self, tmp_path):
        import yaml

        config_path = tmp_path / "config.yaml"
        existing = {
            "endpoints": {
                "existing-ep": {
                    "name": "Existing",
                    "tier": "dev-hosted",
                    "base_url": "https://api.example.com/v1",
                    "model_id": "model",
                    "capabilities": ["chat"],
                }
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(existing, f)

        with patch("nemocode.cli.commands.serve.ensure_config_dir"):
            with patch("nemocode.cli.commands.serve._USER_CONFIG_PATH", config_path):
                _register_endpoint(
                    "new-serve", "http://localhost:8010/v1", "nvidia/model", "local-vllm"
                )

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert "existing-ep" in cfg["endpoints"]
        assert "new-serve" in cfg["endpoints"]


# -----------------------------------------------------------------------
# CLI command tests
# -----------------------------------------------------------------------


class TestServeCLIHelp:
    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "LoRA" in out or "adapter" in out

    def test_serve_start_help(self):
        result = runner.invoke(app, ["serve", "start", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--model" in out
        assert "--adapter" in out

    def test_serve_stop_help(self):
        result = runner.invoke(app, ["serve", "stop", "--help"])
        assert result.exit_code == 0

    def test_serve_status_help(self):
        result = runner.invoke(app, ["serve", "status", "--help"])
        assert result.exit_code == 0


class TestServeStart:
    @patch("nemocode.cli.commands.serve._write_state")
    @patch("nemocode.cli.commands.serve._SERVE_PID_FILE")
    @patch("nemocode.cli.commands.serve._health_check", return_value=True)
    @patch("nemocode.cli.commands.serve._pick_backend", return_value="vllm")
    @patch("nemocode.cli.commands.serve.detect_hardware")
    @patch("nemocode.cli.commands.serve.subprocess.Popen")
    @patch("nemocode.cli.commands.serve.ensure_config_dir")
    @patch("nemocode.cli.commands.serve._register_endpoint")
    def test_start_success(
        self,
        mock_reg,
        mock_ensure,
        mock_popen,
        mock_hw,
        mock_backend,
        mock_health,
        mock_pid,
        mock_write,
    ):
        mock_hw.return_value = MagicMock(is_dgx_spark=False)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        adapter_dir = Path("/tmp/test-adapter")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "resolve", return_value=adapter_dir):
                result = runner.invoke(
                    app,
                    ["serve", "start", "--adapter", str(adapter_dir)],
                )

        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "Server is live" in out or "Ready" in out
        mock_popen.assert_called_once()
        mock_reg.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "vllm"

    @patch("nemocode.cli.commands.serve._pick_backend", return_value="unknown")
    @patch("nemocode.cli.commands.serve.detect_hardware")
    def test_start_no_backend(self, mock_hw, mock_backend):
        mock_hw.return_value = MagicMock(is_dgx_spark=False)
        adapter_dir = Path("/tmp/test-adapter")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "resolve", return_value=adapter_dir):
                result = runner.invoke(
                    app,
                    ["serve", "start", "--adapter", str(adapter_dir)],
                )
        assert result.exit_code == 1
        out = _strip_ansi(result.stdout)
        assert "No supported backend" in out

    @patch("nemocode.cli.commands.serve._pick_backend", return_value="vllm")
    @patch("nemocode.cli.commands.serve.detect_hardware")
    def test_start_adapter_not_found(self, mock_hw, mock_backend):
        mock_hw.return_value = MagicMock(is_dgx_spark=False)
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "resolve", return_value=Path("/nonexistent/adapter")):
                result = runner.invoke(
                    app,
                    ["serve", "start", "--adapter", "/nonexistent/adapter"],
                )
        assert result.exit_code == 1
        out = _strip_ansi(result.stdout)
        assert "not found" in out

    @patch("nemocode.cli.commands.serve._clear_state")
    @patch("nemocode.cli.commands.serve._write_state")
    @patch("nemocode.cli.commands.serve._SERVE_PID_FILE")
    @patch("nemocode.cli.commands.serve._health_check", return_value=False)
    @patch("nemocode.cli.commands.serve._pick_backend", return_value="vllm")
    @patch("nemocode.cli.commands.serve.detect_hardware")
    @patch("nemocode.cli.commands.serve.subprocess.Popen")
    @patch("nemocode.cli.commands.serve.ensure_config_dir")
    def test_start_health_check_timeout(
        self,
        mock_ensure,
        mock_popen,
        mock_hw,
        mock_backend,
        mock_health,
        mock_pid,
        mock_write,
        mock_clear,
    ):
        mock_hw.return_value = MagicMock(is_dgx_spark=False)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        adapter_dir = Path("/tmp/test-adapter")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "resolve", return_value=adapter_dir):
                result = runner.invoke(
                    app,
                    ["serve", "start", "--adapter", str(adapter_dir)],
                )

        assert result.exit_code == 1
        out = _strip_ansi(result.stdout)
        assert "ready" in out
        mock_clear.assert_called_once()

    @patch("nemocode.cli.commands.serve._write_state")
    @patch("nemocode.cli.commands.serve._SERVE_PID_FILE")
    @patch("nemocode.cli.commands.serve._health_check", return_value=True)
    @patch("nemocode.cli.commands.serve._pick_backend", return_value="sglang")
    @patch("nemocode.cli.commands.serve.detect_hardware")
    @patch("nemocode.cli.commands.serve.subprocess.Popen")
    @patch("nemocode.cli.commands.serve.ensure_config_dir")
    @patch("nemocode.cli.commands.serve._register_endpoint")
    def test_start_sglang_backend(
        self,
        mock_reg,
        mock_ensure,
        mock_popen,
        mock_hw,
        mock_backend,
        mock_health,
        mock_pid,
        mock_write,
    ):
        mock_hw.return_value = MagicMock(is_dgx_spark=True)
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_popen.return_value = mock_proc

        adapter_dir = Path("/tmp/test-adapter")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "resolve", return_value=adapter_dir):
                result = runner.invoke(
                    app,
                    ["serve", "start", "--adapter", str(adapter_dir)],
                )

        assert result.exit_code == 0
        mock_reg.assert_called_once()
        tier_arg = mock_reg.call_args[0][3]
        assert tier_arg == "local-sglang"


class TestServeStop:
    def test_stop_no_state(self):
        with patch("nemocode.cli.commands.serve._read_state", return_value=None):
            result = runner.invoke(app, ["serve", "stop"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "No running" in out

    @patch("nemocode.cli.commands.serve._clear_state")
    @patch("nemocode.cli.commands.serve.os.killpg")
    def test_stop_running(self, mock_kill, mock_clear):
        state = {
            "pid": 12345,
            "endpoint_name": "my-lora",
            "base_url": "http://localhost:8010/v1",
        }
        with patch("nemocode.cli.commands.serve._read_state", return_value=state):
            with patch("nemocode.cli.commands.serve.os.getpgid", return_value=12345):
                result = runner.invoke(app, ["serve", "stop"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "stopped" in out.lower()
        mock_kill.assert_called_once()
        mock_clear.assert_called_once()

    @patch("nemocode.cli.commands.serve._clear_state")
    @patch("nemocode.cli.commands.serve.os.killpg", side_effect=ProcessLookupError)
    def test_stop_already_dead(self, mock_kill, mock_clear):
        state = {"pid": 99999, "endpoint_name": "dead-lora", "base_url": "http://localhost:8010/v1"}
        with patch("nemocode.cli.commands.serve._read_state", return_value=state):
            with patch("nemocode.cli.commands.serve.os.getpgid", return_value=99999):
                result = runner.invoke(app, ["serve", "stop"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "not found" in out


class TestServeStatus:
    def test_status_no_state(self):
        with patch("nemocode.cli.commands.serve._read_state", return_value=None):
            result = runner.invoke(app, ["serve", "status"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "No serve" in out

    @patch("nemocode.cli.commands.serve.os.kill")
    @patch("nemocode.cli.commands.serve.httpx")
    def test_status_running_healthy(self, mock_httpx, mock_kill):
        state = {
            "pid": 12345,
            "backend": "vllm",
            "port": 8010,
            "base_url": "http://localhost:8010/v1",
            "model_id": "nvidia/llama-3.1-nemotron-nano-4b-v1.1",
            "endpoint_name": "my-lora",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "model"}]}
        mock_httpx.get.return_value = mock_resp

        with patch("nemocode.cli.commands.serve._read_state", return_value=state):
            result = runner.invoke(app, ["serve", "status"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "Running" in out
        assert "Health check passed" in out

    @patch("nemocode.cli.commands.serve.os.kill", side_effect=ProcessLookupError)
    def test_status_stopped(self, mock_kill):
        state = {
            "pid": 12345,
            "backend": "vllm",
            "port": 8010,
            "base_url": "http://localhost:8010/v1",
            "model_id": "nvidia/model",
            "endpoint_name": "my-lora",
        }
        with patch("nemocode.cli.commands.serve._read_state", return_value=state):
            result = runner.invoke(app, ["serve", "status"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "Stopped" in out


class TestServeCustomPort:
    @patch("nemocode.cli.commands.serve._write_state")
    @patch("nemocode.cli.commands.serve._SERVE_PID_FILE")
    @patch("nemocode.cli.commands.serve._health_check", return_value=True)
    @patch("nemocode.cli.commands.serve._pick_backend", return_value="vllm")
    @patch("nemocode.cli.commands.serve.detect_hardware")
    @patch("nemocode.cli.commands.serve.subprocess.Popen")
    @patch("nemocode.cli.commands.serve.ensure_config_dir")
    @patch("nemocode.cli.commands.serve._register_endpoint")
    def test_custom_port(
        self,
        mock_reg,
        mock_ensure,
        mock_popen,
        mock_hw,
        mock_backend,
        mock_health,
        mock_pid,
        mock_write,
    ):
        mock_hw.return_value = MagicMock(is_dgx_spark=False)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        adapter_dir = Path("/tmp/test-adapter")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "resolve", return_value=adapter_dir):
                result = runner.invoke(
                    app,
                    ["serve", "start", "--adapter", str(adapter_dir), "--port", "9999"],
                )

        assert result.exit_code == 0
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "9999" in call_args
        mock_reg.assert_called_once()
        assert "9999" in mock_reg.call_args[0][1]
