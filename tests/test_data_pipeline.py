# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for ``nemo data pipeline`` — end-to-end repo-to-model workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemocode.cli.commands.data import (
    PipelineResult,
    PipelineStep,
    StepStatus,
    _build_steps,
    _run_step,
)

runner = CliRunner()
_AnsiRe = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _AnsiRe.sub("", text)


def _make_completed_process(returncode=0, stdout="", stderr=""):
    cp = MagicMock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


# -----------------------------------------------------------------------
# Unit tests for helper functions
# -----------------------------------------------------------------------


class TestBuildSteps:
    def test_all_steps_present(self):
        steps = _build_steps(
            repo_path="/tmp/repo",
            max_records=100,
            skip_finetune=False,
            output_dir="/tmp/out",
            model="test-model",
        )
        names = [s.name for s in steps]
        assert "Analyze repository" in names
        assert "Export seed artifacts" in names
        assert "Generate SFT data (NIM API)" in names
        assert "Export SFT dataset (template)" in names
        assert "Submit fine-tuning job" in names
        assert len(steps) == 5

    def test_skip_finetune_removes_last_step(self):
        steps = _build_steps(
            repo_path=".",
            max_records=500,
            skip_finetune=True,
            output_dir=".nemo/data",
            model="m",
        )
        names = [s.name for s in steps]
        assert "Submit fine-tuning job" not in names
        assert len(steps) == 4

    def test_skip_generate_falls_back_to_template(self):
        steps = _build_steps(".", 100, False, ".nemo/data", "m", skip_generate=True)
        names = [s.name for s in steps]
        assert "Generate SFT data (NIM API)" not in names
        assert "Export SFT dataset (template)" in names
        # Template step should be required when generate is skipped
        sft = [s for s in steps if s.name == "Export SFT dataset (template)"][0]
        assert sft.required is True

    def test_generate_present_makes_template_optional(self):
        steps = _build_steps(".", 100, False, ".nemo/data", "m")
        sft = [s for s in steps if s.name == "Export SFT dataset (template)"][0]
        assert sft.required is False

    def test_finetune_uses_generated_data(self):
        steps = _build_steps(".", 100, False, ".nemo/data", "m")
        ft = [s for s in steps if s.name == "Submit fine-tuning job"][0]
        assert "sft_generated.jsonl" in " ".join(ft.command)

    def test_finetune_uses_template_data_when_generate_skipped(self):
        steps = _build_steps(".", 100, False, ".nemo/data", "m", skip_generate=True)
        ft = [s for s in steps if s.name == "Submit fine-tuning job"][0]
        assert "sft_dataset.jsonl" in " ".join(ft.command)

    def test_max_records_in_generate_command(self):
        steps = _build_steps(".", 200, False, ".nemo/data", "m")
        gen = [s for s in steps if s.name == "Generate SFT data (NIM API)"][0]
        assert "--num-records" in gen.command
        assert "200" in gen.command


class TestRunStep:
    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = _make_completed_process()
        step = PipelineStep(name="test", command=["echo", "hi"])
        _run_step(step)
        assert step.status == StepStatus.DONE

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = _make_completed_process(returncode=1, stderr="oops")
        step = PipelineStep(name="test", command=["false"])
        _run_step(step)
        assert step.status == StepStatus.FAILED
        assert "oops" in step.message

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=600)
        step = PipelineStep(name="test", command=["sleep", "999"])
        _run_step(step)
        assert step.status == StepStatus.FAILED
        assert "Timed out" in step.message

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_command_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("nope")
        step = PipelineStep(name="test", command=["nonexistent_cmd"])
        _run_step(step)
        assert step.status == StepStatus.FAILED
        assert "Command not found" in step.message


class TestPipelineResult:
    def test_print_summary_all_done(self, capsys):
        result = PipelineResult()
        result.add(PipelineStep(name="A", command=["a"], status=StepStatus.DONE, message="ok"))
        result.add(PipelineStep(name="B", command=["b"], status=StepStatus.DONE, message="ok"))
        result.print_summary()
        output = capsys.readouterr().out
        assert "Pipeline finished successfully" in output

    def test_print_summary_with_failure(self, capsys):
        result = PipelineResult()
        result.add(PipelineStep(name="A", command=["a"], status=StepStatus.DONE))
        result.add(PipelineStep(name="B", command=["b"], status=StepStatus.FAILED, message="err"))
        result.print_summary()
        output = capsys.readouterr().out
        assert "failed" in output.lower()


# -----------------------------------------------------------------------
# CLI integration tests
# -----------------------------------------------------------------------


class TestDataPipelineHelp:
    def test_data_help(self):
        from nemocode.cli.main import app

        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "dataset" in out.lower() or "pipeline" in out.lower()

    def test_pipeline_help(self):
        from nemocode.cli.main import app

        result = runner.invoke(app, ["data", "pipeline", "--help"])
        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "--dry-run" in out
        assert "--skip-finetune" in out
        assert "--max-records" in out


class TestDataPipelineDryRun:
    @patch("nemocode.cli.commands.data._run_step")
    def test_dry_run_no_execution(self, mock_run):
        from nemocode.cli.main import app

        result = runner.invoke(app, ["data", "pipeline", "--dry-run"])
        assert result.exit_code == 0
        mock_run.assert_not_called()
        out = _strip(result.stdout)
        assert "Pipeline Summary" in out

    @patch("nemocode.cli.commands.data._run_step")
    def test_dry_run_shows_commands(self, mock_run):
        from nemocode.cli.main import app

        result = runner.invoke(app, ["data", "pipeline", "--dry-run"])
        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "nemo data analyze" in out
        assert "nemo data export-seeds" in out


class TestDataPipelineExecution:
    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_all_steps_succeed(self, mock_run):
        from nemocode.cli.main import app

        mock_run.return_value = _make_completed_process()
        result = runner.invoke(
            app,
            ["data", "pipeline", "--repo", "/tmp/test", "--skip-finetune"],
        )
        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "Pipeline finished successfully" in out
        assert mock_run.call_count == 4

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_graceful_degradation(self, mock_run):
        from nemocode.cli.main import app

        def side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "export-seeds" in cmd:
                return _make_completed_process(returncode=1, stderr="service unavailable")
            return _make_completed_process()

        mock_run.side_effect = side_effect
        result = runner.invoke(
            app,
            ["data", "pipeline", "--repo", "/tmp/test", "--skip-finetune"],
        )
        out = _strip(result.stdout)
        assert "Warning" in out or "failed" in out.lower()
        assert "Stopping after required step failure" in out

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_continue_on_error_keeps_running(self, mock_run):
        from nemocode.cli.main import app

        def side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "export-seeds" in cmd:
                return _make_completed_process(returncode=1, stderr="service unavailable")
            return _make_completed_process()

        mock_run.side_effect = side_effect
        result = runner.invoke(
            app,
            ["data", "pipeline", "--repo", "/tmp/test", "--skip-finetune", "--continue-on-error"],
        )
        out = _strip(result.stdout)
        assert "Continuing" in out

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_finetune_step_included(self, mock_run):
        from nemocode.cli.main import app

        mock_run.return_value = _make_completed_process()
        result = runner.invoke(
            app,
            ["data", "pipeline", "--repo", "/tmp/test"],
        )
        assert result.exit_code == 0
        assert mock_run.call_count == 5
        last_call = mock_run.call_args_list[-1]
        cmd = last_call[0][0]
        assert "customize" in cmd
        assert "create" in cmd

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_custom_model_passed(self, mock_run):
        from nemocode.cli.main import app

        mock_run.return_value = _make_completed_process()
        result = runner.invoke(
            app,
            ["data", "pipeline", "--repo", ".", "--model", "custom/model", "--skip-finetune"],
        )
        assert result.exit_code == 0
        assert mock_run.call_count == 4

    @patch("nemocode.cli.commands.data.subprocess.run")
    def test_custom_max_records(self, mock_run):
        from nemocode.cli.commands.data import _build_steps

        steps = _build_steps(".", 1000, True, ".nemo/data", "m")
        sft = [s for s in steps if s.name == "Export SFT dataset (template)"][0]
        assert "1000" in sft.command
