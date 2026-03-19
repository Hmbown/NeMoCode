# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""CLI command smoke tests."""

from __future__ import annotations

import yaml
from unittest.mock import patch

from typer.testing import CliRunner

from nemocode.cli.main import app
from nemocode.core.subagents import complete_run, reset_runs, start_run

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
        assert "--agent" in result.stdout

    def test_agent_help(self):
        result = runner.invoke(app, ["agent", "--help"])
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


class TestAgentCommand:
    def test_agent_ls(self):
        result = runner.invoke(app, ["agent", "ls"])
        assert result.exit_code == 0

    def test_agent_runs(self):
        reset_runs()
        run = start_run(
            agent_name="general",
            display_name="Joe Nemotron",
            nickname="Orbit Joe",
            parent_agent="build",
            task="Inspect the repo",
            endpoint="nim-nano",
        )
        complete_run(run.id, output_preview="done", tool_calls=2, errors=0)
        try:
            result = runner.invoke(app, ["agent", "runs"])
            assert result.exit_code == 0
            assert run.id in result.stdout
            assert "Orbit Joe" in result.stdout
        finally:
            reset_runs()


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
        config_path = tmp_path / ".nemocode.yaml"
        assert config_path.exists()
        raw = yaml.safe_load(config_path.read_text())
        assert raw["project"]["name"] == "TestProject"
        assert "default_endpoint" not in raw
        assert "active_formation" not in raw

    def test_init_can_set_project_endpoint_override(self, tmp_path):
        import os

        os.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--endpoint", "nim-nano-4b"])
        assert result.exit_code == 0
        raw = yaml.safe_load((tmp_path / ".nemocode.yaml").read_text())
        assert raw["default_endpoint"] == "nim-nano-4b"

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


class TestDoctorCommand:
    def test_doctor_show(self):
        from nemocode.config.schema import (
            Endpoint,
            EndpointTier,
            Formation,
            FormationRole,
            FormationSlot,
            NeMoCodeConfig,
        )
        from nemocode.core.doctor import DiagnosticReport

        config = NeMoCodeConfig(
            default_endpoint="test",
            endpoints={
                "test": Endpoint(
                    name="test",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://test.com/v1",
                    api_key_env="TEST_API_KEY",
                    model_id="test",
                ),
            },
            formations={
                "solo": Formation(
                    name="Solo",
                    slots=[
                        FormationSlot(endpoint="test", role=FormationRole.EXECUTOR),
                    ],
                ),
            },
        )
        report = DiagnosticReport()
        report.add("test_check", "ok", "details")

        with (
            patch("nemocode.cli.commands.doctor.load_config", return_value=config),
            patch("nemocode.cli.commands.doctor.run_diagnostics", return_value=report),
        ):
            result = runner.invoke(app, ["doctor", "show"])
            assert result.exit_code == 0
            assert "Diagnostic Report" in result.stdout
            assert "test_check" in result.stdout
            assert "ok" in result.stdout
