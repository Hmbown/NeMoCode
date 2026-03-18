# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for project instruction file discovery."""

from __future__ import annotations

from nemocode.workflows.code_agent import _discover_instruction_files


class TestInstructionDiscovery:
    def test_prefers_agents_md_before_legacy_nemocode_md(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "NEMOCODE.md").write_text("legacy instructions")
        (repo / "agents.md").write_text("preferred instructions")

        text = _discover_instruction_files(repo)

        assert "preferred instructions" in text
        assert "legacy instructions" in text
        assert text.index("preferred instructions") < text.index("legacy instructions")

    def test_loads_global_project_and_directory_instruction_files(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        config_dir = home / ".config" / "nemocode"
        config_dir.mkdir(parents=True)
        (config_dir / "agents.md").write_text("global instructions")

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "AGENTS.md").write_text("project instructions")

        cwd = repo / "pkg"
        cwd.mkdir()
        (cwd / "agents.md").write_text("directory instructions")

        monkeypatch.setenv("HOME", str(home))

        text = _discover_instruction_files(cwd)

        assert "global instructions" in text
        assert "project instructions" in text
        assert "directory instructions" in text
        assert text.index("global instructions") < text.index("project instructions")
        assert text.index("project instructions") < text.index("directory instructions")
