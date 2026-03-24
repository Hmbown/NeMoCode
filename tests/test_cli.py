# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""CLI command smoke tests."""

from __future__ import annotations

import re
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from nemocode.cli.main import app
from nemocode.core.subagents import complete_run, reset_runs, start_run

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for reliable string matching."""
    return _ANSI_ESCAPE_RE.sub("", text)


class TestCLIHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "NeMoCode" in _strip_ansi(result.stdout)

    def test_chat_help(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0

    def test_code_help(self):
        result = runner.invoke(app, ["code", "--help"])
        assert result.exit_code == 0
        assert "--agent" in _strip_ansi(result.stdout)

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

    def test_data_help(self):
        result = runner.invoke(app, ["data", "--help"])
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
            assert run.id in _strip_ansi(result.stdout)
            assert "Orbit Joe" in _strip_ansi(result.stdout)
        finally:
            reset_runs()


class TestAuthCommand:
    def test_auth_show(self):
        with patch("nemocode.core.credentials._keyring_available", return_value=False):
            result = runner.invoke(app, ["auth", "show"])
            assert result.exit_code == 0
            assert "Credentials" in _strip_ansi(result.stdout)


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


class TestDataCommand:
    def test_data_analyze(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            "[project]\nname='demo'\ndependencies=['typer>=0.12', 'rich>=13.0', 'pytest>=8.0']\n"
        )
        (repo / "README.md").write_text("# Demo\n")
        src_dir = repo / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("print('hello')\n")
        tests_dir = repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_app.py").write_text("def test_ok():\n    assert True\n")

        result = runner.invoke(app, ["data", "analyze", str(repo)])

        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "Recommended NVIDIA Stack" in out
        assert "data_designer" in out

    def test_data_analyze_writes_plan(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "package.json").write_text(
            '{"dependencies":{"react":"19.0.0"},"devDependencies":{"vitest":"3.0.0"}}'
        )
        (repo / "src").mkdir()
        (repo / "src" / "app.tsx").write_text("export const x = 1;\n")
        out_path = tmp_path / "plan.yaml"

        result = runner.invoke(
            app,
            ["data", "analyze", str(repo), "--output", str(out_path)],
        )

        assert result.exit_code == 0
        assert out_path.exists()
        raw = yaml.safe_load(out_path.read_text())
        assert raw["repo_profile"]["root"] == str(repo.resolve())
        assert "data_designer_starter" in raw

    def test_export_seeds_help(self):
        result = runner.invoke(app, ["data", "export-seeds", "--help"])
        assert result.exit_code == 0

    def test_export_sft_help(self):
        result = runner.invoke(app, ["data", "export-sft", "--help"])
        assert result.exit_code == 0

    def test_preview_help(self):
        result = runner.invoke(app, ["data", "preview", "--help"])
        assert result.exit_code == 0

    def test_job_help(self):
        result = runner.invoke(app, ["data", "job", "--help"])
        assert result.exit_code == 0

    def test_job_create_help(self):
        result = runner.invoke(app, ["data", "job", "create", "--help"])
        assert result.exit_code == 0

    def test_job_status_help(self):
        result = runner.invoke(app, ["data", "job", "status", "--help"])
        assert result.exit_code == 0

    def test_job_logs_help(self):
        result = runner.invoke(app, ["data", "job", "logs", "--help"])
        assert result.exit_code == 0

    def test_job_results_help(self):
        result = runner.invoke(app, ["data", "job", "results", "--help"])
        assert result.exit_code == 0

    def test_setup_data(self):
        result = runner.invoke(app, ["setup", "data"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "NVIDIA Data Workflow Setup" in out
        assert "nemo data analyze" in out


class TestObsCommand:
    def test_obs_pricing(self):
        result = runner.invoke(app, ["obs", "pricing"])
        assert result.exit_code == 0
        assert "Pricing" in _strip_ansi(result.stdout)


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
            assert "Diagnostic Report" in _strip_ansi(result.stdout)
            assert "test_check" in _strip_ansi(result.stdout)
            assert "ok" in _strip_ansi(result.stdout)
