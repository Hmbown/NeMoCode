# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the glob_files tool."""

from __future__ import annotations

import json

import pytest

from nemocode.tools.glob import glob_files


@pytest.fixture
def temp_tree(tmp_path):
    """Create a directory tree for glob testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "src" / "utils.py").write_text("# utils")
    (tmp_path / "src" / "sub").mkdir()
    (tmp_path / "src" / "sub" / "deep.py").write_text("# deep")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("# test")
    (tmp_path / "README.md").write_text("# README")
    (tmp_path / "setup.py").write_text("# setup")
    return tmp_path


@pytest.mark.asyncio
async def test_glob_py_files(temp_tree):
    result = json.loads(await glob_files("**/*.py", path=str(temp_tree)))
    assert result["count"] >= 4
    assert any("main.py" in f for f in result["files"])


@pytest.mark.asyncio
async def test_glob_md_files(temp_tree):
    result = json.loads(await glob_files("*.md", path=str(temp_tree)))
    assert result["count"] == 1
    assert "README.md" in result["files"]


@pytest.mark.asyncio
async def test_glob_no_matches(temp_tree):
    result = json.loads(await glob_files("*.rs", path=str(temp_tree)))
    assert result["count"] == 0
    assert result["files"] == []


@pytest.mark.asyncio
async def test_glob_max_results(temp_tree):
    result = json.loads(await glob_files("**/*.py", path=str(temp_tree), max_results=2))
    assert result["count"] == 2
    assert result.get("truncated") is True


@pytest.mark.asyncio
async def test_glob_nonexistent_path():
    result = json.loads(await glob_files("*.py", path="/nonexistent/path"))
    assert "error" in result


@pytest.mark.asyncio
async def test_glob_skips_hidden_dirs(temp_tree):
    """Hidden directories should be skipped."""
    hidden = temp_tree / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("# secret")

    result = json.loads(await glob_files("**/*.py", path=str(temp_tree)))
    assert not any(".hidden" in f for f in result["files"])
