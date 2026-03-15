# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""CLI command smoke tests."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from nemocode.cli.main import app

runner = CliRunner()


class TestCLIHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "NeMoCode" in result.stdout

    def test_chat_help(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0

    def test_code_help(self):
        result = runner.invoke(app, ["code", "--help"])
        assert result.exit_code == 0

    def test_endpoint_help(self):
        result = runner.invoke(app, ["endpoint", "--help"])
        assert result.exit_code == 0

    def test_model_help(self):
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0

    def test_formation_help(self):
        result = runner.invoke(app, ["formation", "--help"])
        assert result.exit_code == 0

    def test_auth_help(self):
        result = runner.invoke(app, ["auth", "--help"])
        assert result.exit_code == 0

    def test_hardware_help(self):
        result = runner.invoke(app, ["hardware", "--help"])
        assert result.exit_code == 0

    def test_session_help(self):
        result = runner.invoke(app, ["session", "--help"])
        assert result.exit_code == 0

    def test_obs_help(self):
        result = runner.invoke(app, ["obs", "--help"])
        assert result.exit_code == 0


class TestEndpointCommand:
    def test_endpoint_ls(self):
        result = runner.invoke(app, ["endpoint", "ls"])
        # Should at least not crash — may warn about missing config
        assert result.exit_code == 0


class TestModelCommand:
    def test_model_ls(self):
        result = runner.invoke(app, ["model", "ls"])
        assert result.exit_code == 0


class TestFormationCommand:
    def test_formation_ls(self):
        result = runner.invoke(app, ["formation", "ls"])
        assert result.exit_code == 0


class TestAuthCommand:
    def test_auth_show(self):
        with patch("nemocode.core.credentials._keyring_available", return_value=False):
            result = runner.invoke(app, ["auth", "show"])
            assert result.exit_code == 0
            assert "Credentials" in result.stdout


class TestInitCommand:
    def test_init_creates_config(self, tmp_path):
        import os

        os.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--name", "TestProject"])
        assert result.exit_code == 0
        assert (tmp_path / ".nemocode.yaml").exists()

    def test_init_refuses_overwrite(self, tmp_path):
        import os

        os.chdir(tmp_path)
        (tmp_path / ".nemocode.yaml").write_text("existing: true")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1


class TestObsCommand:
    def test_obs_pricing(self):
        result = runner.invoke(app, ["obs", "pricing"])
        assert result.exit_code == 0
        assert "Pricing" in result.stdout
