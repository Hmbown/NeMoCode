# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for repo-aware data workflows: profiling, seeds, SFT export, clients, CLI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from nemocode.cli.main import app
from nemocode.core.data_workflows import (
    RepoProfile,
    SeedExportResult,
    SFTExportResult,
    analyze_repo,
    build_preview_config,
    build_repo_data_plan,
    export_seeds,
    export_sft,
)

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    """Create a small sample repository for testing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\nname='demo'\ndependencies=['typer>=0.12', 'rich>=13.0', 'pytest>=8.0']\n"
    )
    (repo / "README.md").write_text("# Demo\nA sample project.\n")
    src = repo / "src"
    src.mkdir()
    (src / "app.py").write_text("def main():\n    print('hello')\n")
    (src / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_ok():\n    assert True\n")
    docs = repo / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\nHow to use this.\n")
    return repo


@pytest.fixture()
def sample_profile(sample_repo: Path) -> RepoProfile:
    return analyze_repo(sample_repo)


# ---------------------------------------------------------------------------
# Repo profiling tests
# ---------------------------------------------------------------------------


class TestRepoAnalysis:
    def test_basic_profile(self, sample_repo: Path):
        profile = analyze_repo(sample_repo)
        assert profile.total_files > 0
        assert profile.code_files > 0
        assert profile.has_tests
        assert profile.has_docs
        assert "python" in profile.primary_languages

    def test_detects_frameworks(self, sample_repo: Path):
        profile = analyze_repo(sample_repo)
        assert "Pytest" in profile.frameworks
        assert "Typer" in profile.frameworks
        assert "Rich" in profile.frameworks

    def test_ignores_hidden_dirs(self, sample_repo: Path):
        cache = sample_repo / "__pycache__"
        cache.mkdir()
        (cache / "cached.pyc").write_bytes(b"\x00")
        profile = analyze_repo(sample_repo)
        # __pycache__ files should not appear
        assert all("__pycache__" not in f for f in profile.dataset_files)

    def test_empty_repo(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        profile = analyze_repo(empty)
        assert profile.total_files == 0
        assert not profile.has_tests
        assert not profile.has_docs

    def test_data_plan_structure(self, sample_profile: RepoProfile):
        plan = build_repo_data_plan(sample_profile)
        assert "repo_profile" in plan
        assert "recommended_stack" in plan
        assert "data_designer" in plan["recommended_stack"]
        assert "evaluator" in plan["recommended_stack"]
        assert "safe_synthesizer" in plan["recommended_stack"]
        assert "data_designer_starter" in plan
        assert "repo_to_data_mvp" in plan

    def test_preview_config(self, sample_profile: RepoProfile):
        config = build_preview_config(sample_profile)
        assert "model_configs" in config
        assert "columns" in config
        aliases = {m["alias"] for m in config["model_configs"]}
        assert "code" in aliases
        assert "judge" in aliases


# ---------------------------------------------------------------------------
# Seed export tests
# ---------------------------------------------------------------------------


class TestSeedExport:
    def test_export_creates_all_artifacts(self, sample_repo: Path):
        result = export_seeds(sample_repo)
        assert isinstance(result, SeedExportResult)
        assert Path(result.profile_path).exists()
        assert Path(result.manifest_path).exists()
        assert Path(result.taxonomy_path).exists()
        assert Path(result.context_packs_path).exists()
        assert result.file_count > 0

    def test_profile_yaml_is_valid(self, sample_repo: Path):
        result = export_seeds(sample_repo)
        data = yaml.safe_load(Path(result.profile_path).read_text())
        assert data["has_tests"] is True
        assert "python" in data["primary_languages"]

    def test_manifest_is_valid_jsonl(self, sample_repo: Path):
        result = export_seeds(sample_repo)
        entries = []
        for line in Path(result.manifest_path).read_text().splitlines():
            entry = json.loads(line)
            assert "path" in entry
            assert "category" in entry
            entries.append(entry)
        assert len(entries) > 0
        categories = {e["category"] for e in entries}
        assert "source" in categories

    def test_taxonomy_has_task_families(self, sample_repo: Path):
        result = export_seeds(sample_repo)
        data = yaml.safe_load(Path(result.taxonomy_path).read_text())
        assert "task_families" in data
        assert "bugfix_request" in data["task_families"]
        assert "test_generation" in data["task_families"]

    def test_context_packs_have_snippets(self, sample_repo: Path):
        result = export_seeds(sample_repo)
        packs = []
        for line in Path(result.context_packs_path).read_text().splitlines():
            pack = json.loads(line)
            assert "snippet" in pack
            assert "path" in pack
            packs.append(pack)
        assert len(packs) > 0

    def test_excludes_secrets(self, sample_repo: Path):
        (sample_repo / ".env").write_text("SECRET=abc123\n")
        (sample_repo / "credentials.json").write_text('{"key": "secret"}\n')
        result = export_seeds(sample_repo)
        manifest_text = Path(result.manifest_path).read_text()
        assert ".env" not in manifest_text
        assert "credentials.json" not in manifest_text

    def test_excludes_lockfiles(self, sample_repo: Path):
        (sample_repo / "package-lock.json").write_text("{}\n")
        result = export_seeds(sample_repo)
        manifest_text = Path(result.manifest_path).read_text()
        assert "package-lock.json" not in manifest_text

    def test_custom_output_dir(self, sample_repo: Path, tmp_path: Path):
        out = tmp_path / "custom_seeds"
        result = export_seeds(sample_repo, output_dir=out)
        assert result.output_dir == str(out)
        assert Path(result.profile_path).exists()


# ---------------------------------------------------------------------------
# SFT export tests
# ---------------------------------------------------------------------------


class TestSFTExport:
    def test_export_creates_jsonl(self, sample_repo: Path):
        result = export_sft(sample_repo)
        assert isinstance(result, SFTExportResult)
        assert Path(result.output_path).exists()
        assert result.record_count > 0

    def test_records_have_message_structure(self, sample_repo: Path):
        result = export_sft(sample_repo)
        for line in Path(result.output_path).read_text().splitlines():
            record = json.loads(line)
            assert "messages" in record
            roles = [m["role"] for m in record["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert "metadata" in record
            assert "task_type" in record["metadata"]

    def test_max_records(self, sample_repo: Path):
        result = export_sft(sample_repo, max_records=3)
        assert result.record_count == 3

    def test_uses_context_packs(self, sample_repo: Path):
        # Export seeds first to create context packs
        export_seeds(sample_repo)
        result = export_sft(sample_repo)
        text = Path(result.output_path).read_text()
        # At least some records should include file snippets
        assert "Relevant file:" in text

    def test_custom_output_path(self, sample_repo: Path, tmp_path: Path):
        out = tmp_path / "custom.jsonl"
        result = export_sft(sample_repo, output_path=out)
        assert result.output_path == str(out)
        assert out.exists()

    def test_default_export_prefers_implementation_files(self, sample_repo: Path):
        result = export_sft(sample_repo)
        rows = [
            json.loads(line)
            for line in Path(result.output_path).read_text().splitlines()
            if line.strip()
        ]
        assert rows
        assert all(not row["metadata"]["filepath"].startswith("tests/") for row in rows)

    def test_include_tests_can_restore_test_definitions(self, sample_repo: Path):
        result = export_sft(sample_repo, include_tests=True)
        rows = [
            json.loads(line)
            for line in Path(result.output_path).read_text().splitlines()
            if line.strip()
        ]
        assert rows
        assert any(row["metadata"]["filepath"].startswith("tests/") for row in rows)


# ---------------------------------------------------------------------------
# NVIDIA client tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestDataDesignerClient:
    def test_health_check(self):
        from nemocode.core.nvidia_client import DataDesignerClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get.return_value = MagicMock(status_code=200)

            client = DataDesignerClient(base_url="http://test:8080")
            assert client.health()

    def test_health_check_fails(self):
        from nemocode.core.nvidia_client import DataDesignerClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            import httpx

            mock_client.get.side_effect = httpx.ConnectError("refused")

            client = DataDesignerClient(base_url="http://test:8080")
            assert not client.health()

    def test_create_job(self):
        from nemocode.core.nvidia_client import DataDesignerClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"id": "job-123", "status": "pending"}
            mock_client.post.return_value = mock_resp

            client = DataDesignerClient(base_url="http://test:8080")
            result = client.create_job("test-job", {"columns": []})
            assert result["id"] == "job-123"

    def test_get_job_status(self):
        from nemocode.core.nvidia_client import DataDesignerClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"status": "completed"}
            mock_client.get.return_value = mock_resp

            client = DataDesignerClient(base_url="http://test:8080")
            result = client.get_job_status("job-123")
            assert result["status"] == "completed"


class TestEvaluatorClient:
    def test_create_job(self):
        from nemocode.core.nvidia_client import EvaluatorClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"id": "eval-456", "status": "pending"}
            mock_client.post.return_value = mock_resp

            client = EvaluatorClient(base_url="http://test:8080")
            result = client.create_job({"target": {}, "config": {}})
            assert result["id"] == "eval-456"

    def test_list_jobs(self):
        from nemocode.core.nvidia_client import EvaluatorClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": [], "pagination": {}}
            mock_client.get.return_value = mock_resp

            client = EvaluatorClient(base_url="http://test:8080")
            result = client.list_jobs()
            assert "data" in result


class TestSafeSynthesizerClient:
    def test_create_job(self):
        from nemocode.core.nvidia_client import SafeSynthesizerClient

        with patch("nemocode.core.nvidia_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"id": "ss-789", "status": "pending"}
            mock_client.post.return_value = mock_resp

            client = SafeSynthesizerClient(base_url="http://test:8080")
            result = client.create_job({"spec": {}})
            assert result["id"] == "ss-789"

    def test_env_var_override(self):
        from nemocode.core.nvidia_client import SafeSynthesizerClient

        with patch.dict(os.environ, {"NEMOCODE_SAFE_SYNTH_BASE_URL": "http://custom:9090"}):
            with patch("nemocode.core.nvidia_client.httpx.Client"):
                client = SafeSynthesizerClient()
                assert client.base_url == "http://custom:9090"


# ---------------------------------------------------------------------------
# CLI tests for new subcommands
# ---------------------------------------------------------------------------


class TestDataCLI:
    def test_data_help(self):
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0

    def test_export_seeds_help(self):
        result = runner.invoke(app, ["data", "export-seeds", "--help"])
        assert result.exit_code == 0
        assert "seed" in _strip(result.stdout).lower()

    def test_preview_help(self):
        result = runner.invoke(app, ["data", "preview", "--help"])
        assert result.exit_code == 0
        assert "preview" in _strip(result.stdout).lower()

    def test_export_sft_help(self):
        result = runner.invoke(app, ["data", "export-sft", "--help"])
        assert result.exit_code == 0
        assert "sft" in _strip(result.stdout).lower()

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

    def test_export_seeds_runs(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            "[project]\nname='demo'\ndependencies=['pytest>=8.0']\n"
        )
        src = repo / "src"
        src.mkdir()
        (src / "app.py").write_text("print('hi')\n")
        tests = repo / "tests"
        tests.mkdir()
        (tests / "test_app.py").write_text("def test_ok(): assert True\n")

        result = runner.invoke(app, ["data", "export-seeds", str(repo)])
        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "Seed Export Complete" in out
        assert (repo / ".nemocode" / "data" / "repo_profile.yaml").exists()
        assert (repo / ".nemocode" / "data" / "file_manifest.jsonl").exists()
        assert (repo / ".nemocode" / "data" / "task_taxonomy.yaml").exists()

    def test_export_sft_runs(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project]\nname='demo'\ndependencies=[]\n")
        (repo / "main.py").write_text("print('hello')\n")

        result = runner.invoke(app, ["data", "export-sft", str(repo), "--max-records", "5"])
        assert result.exit_code == 0
        out = _strip(result.stdout)
        assert "SFT Dataset Export" in out


# ---------------------------------------------------------------------------
# Live smoke tests (opt-in, gated behind env vars)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("NEMOCODE_DATA_BASE_URL"),
    reason="Live Data Designer not configured (set NEMOCODE_DATA_BASE_URL)",
)
class TestLiveDataDesigner:
    def test_health(self):
        from nemocode.core.nvidia_client import DataDesignerClient

        client = DataDesignerClient()
        assert client.health()

    def test_preview(self, sample_repo: Path):
        from nemocode.core.nvidia_client import DataDesignerClient

        profile = analyze_repo(sample_repo)
        config = build_preview_config(profile)
        client = DataDesignerClient()
        records = client.preview(config, num_records=2)
        assert len(records) > 0


@pytest.mark.skipif(
    not os.environ.get("NEMOCODE_EVAL_BASE_URL"),
    reason="Live Evaluator not configured (set NEMOCODE_EVAL_BASE_URL)",
)
class TestLiveEvaluator:
    def test_health(self):
        from nemocode.core.nvidia_client import EvaluatorClient

        client = EvaluatorClient()
        assert client.health()

    def test_list_jobs(self):
        from nemocode.core.nvidia_client import EvaluatorClient

        client = EvaluatorClient()
        result = client.list_jobs()
        assert "data" in result
