# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for nemo model pull / nemo model ls --local."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from nemocode.cli.commands.model import NGC_CONTAINER_MAP, _resolve_ngc_key
from nemocode.cli.main import app

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


class TestNameResolution:
    def test_known_models_resolve(self):
        """All documented model names map to nvcr.io URIs."""
        assert "nemotron-3-nano-4b" in NGC_CONTAINER_MAP
        assert "nemotron-3-super-120b" in NGC_CONTAINER_MAP
        assert "nemotron-nano-9b" in NGC_CONTAINER_MAP
        assert "nemotron-nano-12b-vl" in NGC_CONTAINER_MAP

    def test_uris_start_with_nvcr(self):
        for name, uri in NGC_CONTAINER_MAP.items():
            assert uri.startswith("nvcr.io/nim/nvidia/"), f"{name} URI missing prefix: {uri}"

    def test_uris_have_tag(self):
        for name, uri in NGC_CONTAINER_MAP.items():
            assert ":" in uri, f"{name} URI missing tag: {uri}"

    def test_unknown_model_rejected(self):
        assert "nonexistent-model" not in NGC_CONTAINER_MAP


# ---------------------------------------------------------------------------
# NGC key resolution
# ---------------------------------------------------------------------------


class TestNGCKeyResolution:
    @patch.dict("os.environ", {"NGC_CLI_API_KEY": "test-key-123"}, clear=False)
    @patch("nemocode.core.credentials.get_credential", return_value=None)
    def test_env_var_ngc_cli_api_key(self, _mock_cred):
        key = _resolve_ngc_key()
        assert key == "test-key-123"

    @patch.dict(
        "os.environ",
        {"NGC_API_KEY": "fallback-key"},
        clear=False,
    )
    @patch("nemocode.core.credentials.get_credential", return_value=None)
    def test_env_var_ngc_api_key_fallback(self, _mock_cred):
        import os

        os.environ.pop("NGC_CLI_API_KEY", None)
        key = _resolve_ngc_key()
        assert key == "fallback-key"

    @patch.dict("os.environ", {}, clear=True)
    @patch("nemocode.core.credentials.get_credential", return_value=None)
    def test_no_key_returns_none(self, _mock_cred):
        key = _resolve_ngc_key()
        assert key is None


# ---------------------------------------------------------------------------
# nemo model pull
# ---------------------------------------------------------------------------


class TestModelPull:
    @patch("shutil.which", return_value=None)
    def test_missing_docker(self, _mock_which):
        result = runner.invoke(app, ["model", "pull", "nemotron-nano-9b"])
        assert result.exit_code != 0
        assert "Docker" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_unknown_model_name(self, _mock_which):
        result = runner.invoke(app, ["model", "pull", "nonexistent-model"])
        assert result.exit_code != 0
        assert "Unknown model" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("nemocode.cli.commands.model._resolve_ngc_key", return_value=None)
    def test_missing_ngc_key(self, _mock_key, _mock_which):
        result = runner.invoke(app, ["model", "pull", "nemotron-nano-9b"])
        assert result.exit_code != 0
        assert "NGC API key" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("nemocode.cli.commands.model._resolve_ngc_key", return_value="test-key")
    @patch("subprocess.run")
    def test_successful_pull(self, mock_run, _mock_key, _mock_which):
        # Mock both docker login and docker pull as successful
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.invoke(app, ["model", "pull", "nemotron-nano-9b"])
        assert result.exit_code == 0
        assert "Successfully pulled" in _strip_ansi(result.stdout)

        # Verify docker login was called with nvcr.io
        login_call = mock_run.call_args_list[0]
        assert "nvcr.io" in login_call.args[0]
        assert login_call.kwargs["input"] == "test-key"

        # Verify docker pull was called with correct URI
        pull_call = mock_run.call_args_list[1]
        assert pull_call.args[0] == [
            "docker",
            "pull",
            "nvcr.io/nim/nvidia/nemotron-nano-9b-v2:latest",
        ]

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("nemocode.cli.commands.model._resolve_ngc_key", return_value="test-key")
    @patch("subprocess.run")
    def test_docker_login_failure(self, mock_run, _mock_key, _mock_which):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="unauthorized: authentication required"
        )
        result = runner.invoke(app, ["model", "pull", "nemotron-nano-9b"])
        assert result.exit_code != 0
        assert "Docker login failed" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("nemocode.cli.commands.model._resolve_ngc_key", return_value="test-key")
    @patch("subprocess.run")
    def test_docker_pull_failure(self, mock_run, _mock_key, _mock_which):
        # Login succeeds, pull fails
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # login ok
            MagicMock(returncode=1, stdout="", stderr="network timeout"),  # pull fails
        ]
        result = runner.invoke(app, ["model", "pull", "nemotron-nano-9b"])
        assert result.exit_code != 0
        assert "Docker pull failed" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("nemocode.cli.commands.model._resolve_ngc_key", return_value="test-key")
    @patch("subprocess.run")
    def test_custom_tag(self, mock_run, _mock_key, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.invoke(app, ["model", "pull", "nemotron-nano-9b", "--tag", "1.0.0"])
        assert result.exit_code == 0
        pull_call = mock_run.call_args_list[1]
        assert "1.0.0" in pull_call.args[0][2]


# ---------------------------------------------------------------------------
# nemo model ls --local
# ---------------------------------------------------------------------------


class TestModelLsLocal:
    @patch("shutil.which", return_value=None)
    def test_missing_docker(self, _mock_which):
        result = runner.invoke(app, ["model", "ls", "--local"])
        assert result.exit_code != 0
        assert "Docker" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_no_local_images(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.invoke(app, ["model", "ls", "--local"])
        assert result.exit_code == 0
        assert "No NIM containers" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_local_images_displayed(self, mock_run, _mock_which):
        docker_output = (
            "nvcr.io/nim/nvidia/nemotron-nano-9b-v2:latest 8.5GB\n"
            "nvcr.io/nim/nvidia/nemotron-3-nano-4b-instruct:latest 4.2GB\n"
            "ubuntu:22.04 77.8MB\n"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=docker_output, stderr="")
        result = runner.invoke(app, ["model", "ls", "--local"])
        assert result.exit_code == 0
        output = _strip_ansi(result.stdout)
        # NIM images should be shown
        assert "nemotron-nano-9b-v2" in output
        assert "nemotron-3-nano-4b-instruct" in output
        # Non-NIM images should be filtered out
        assert "ubuntu" not in output

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_docker_images_failure(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Cannot connect to Docker daemon"
        )
        result = runner.invoke(app, ["model", "ls", "--local"])
        assert result.exit_code != 0
        assert "Failed to list images" in _strip_ansi(result.stdout)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_only_nim_images_shown(self, mock_run, _mock_which):
        docker_output = (
            "python:3.11 900MB\n"
            "nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b-instruct:latest 15GB\n"
            "nginx:latest 150MB\n"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=docker_output, stderr="")
        result = runner.invoke(app, ["model", "ls", "--local"])
        assert result.exit_code == 0
        output = _strip_ansi(result.stdout)
        assert "nemotron-3-super-120b" in output
        assert "python" not in output
        assert "nginx" not in output
