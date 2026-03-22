# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from nemocode.tools.fs import edit_file, set_project_root


@pytest.fixture(autouse=True)
def _set_project_root(project_dir):
    set_project_root(project_dir)


class TestEditFileEdgeCases:
    @pytest.mark.asyncio
    async def test_edit_file_multiple_matches(self, project_dir):
        test_file = project_dir / "multi.txt"
        content = "line one\nsame text here\nline two\nsame text here\nline three\nsame text here\n"
        test_file.write_text(content)

        result = json.loads(await edit_file(str(test_file), "same text here", "replaced text"))

        assert "error" in result
        assert "3 times" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_file_empty_old_string(self, project_dir):
        test_file = project_dir / "empty_test.txt"
        test_file.write_text("hello world\n")

        result = json.loads(await edit_file(str(test_file), "", "new content"))

        assert "error" in result
        assert "empty" in result["error"].lower()
