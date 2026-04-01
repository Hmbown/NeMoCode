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
        with pytest.raises(PermissionError, match="outside project"):
            _resolve_path("/etc/passwd")

    def test_path_traversal_blocked(self, tmp_path):
        set_project_root(tmp_path)
        with pytest.raises(PermissionError, match="outside project"):
            _resolve_path(str(tmp_path / ".." / ".." / "etc" / "passwd"))

    @pytest.mark.asyncio
    async def test_write_outside_project_rejected(self, tmp_path):
        set_project_root(tmp_path)
        with pytest.raises(PermissionError, match="outside project"):
            await write_file("/tmp/nemocode_test_escape.txt", "pwned")

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
        with pytest.raises(PermissionError, match="outside project"):
            await edit_file("/etc/passwd", "old", "new")
