# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for filesystem path sandboxing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemocode.tools.fs import (
    _is_within_project,
    _resolve_path,
    edit_file,
    set_project_root,
    write_file,
)


class TestPathSandbox:
    def test_path_within_project(self, tmp_path):
        set_project_root(tmp_path)
        p = _resolve_path(str(tmp_path / "test.txt"))
        assert _is_within_project(p)

    def test_path_outside_project(self, tmp_path):
        set_project_root(tmp_path)
        p = _resolve_path("/etc/passwd")
        assert not _is_within_project(p)

    def test_path_traversal_blocked(self, tmp_path):
        set_project_root(tmp_path)
        p = _resolve_path(str(tmp_path / ".." / ".." / "etc" / "passwd"))
        assert not _is_within_project(p)

    @pytest.mark.asyncio
    async def test_write_outside_project_rejected(self, tmp_path):
        set_project_root(tmp_path)
        result = await write_file("/tmp/nemocode_test_escape.txt", "pwned")
        data = json.loads(result)
        assert "error" in data
        assert "outside project" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_write_inside_project_allowed(self, tmp_path):
        set_project_root(tmp_path)
        target = tmp_path / "allowed.txt"
        result = await write_file(str(target), "safe content")
        data = json.loads(result)
        assert data["status"] == "ok"
        assert target.read_text() == "safe content"

    @pytest.mark.asyncio
    async def test_edit_outside_project_rejected(self, tmp_path):
        set_project_root(tmp_path)
        # Create a file outside the project
        import tempfile

        outside = Path(tempfile.gettempdir()) / "nemocode_test_outside.txt"
        outside.write_text("original")
        try:
            result = await edit_file(str(outside), "original", "modified")
            data = json.loads(result)
            assert "error" in data
            assert "outside project" in data["error"].lower()
            # File should not have been modified
            assert outside.read_text() == "original"
        finally:
            outside.unlink(missing_ok=True)
