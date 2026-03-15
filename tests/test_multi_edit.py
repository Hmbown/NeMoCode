# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for multi_edit and apply_patch tools."""

from __future__ import annotations

import json

import pytest

from nemocode.tools.fs import set_project_root
from nemocode.tools.multi_edit import apply_patch, multi_edit


@pytest.fixture
def sample_file(tmp_path):
    set_project_root(tmp_path)
    f = tmp_path / "test.py"
    f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    return f


@pytest.mark.asyncio
async def test_multi_edit_basic(sample_file):
    edits = json.dumps([
        {"old_string": "return 1", "new_string": "return 42"},
        {"old_string": "return 2", "new_string": "return 99"},
    ])
    result = json.loads(await multi_edit(str(sample_file), edits))
    assert result["status"] == "ok"
    assert result["applied"] == 2
    content = sample_file.read_text()
    assert "return 42" in content
    assert "return 99" in content


@pytest.mark.asyncio
async def test_multi_edit_partial_failure(sample_file):
    edits = json.dumps([
        {"old_string": "return 1", "new_string": "return 42"},
        {"old_string": "nonexistent", "new_string": "x"},
    ])
    result = json.loads(await multi_edit(str(sample_file), edits))
    assert result["status"] == "ok"
    assert result["applied"] == 1
    assert len(result["errors"]) == 1


@pytest.mark.asyncio
async def test_multi_edit_all_fail(sample_file):
    edits = json.dumps([
        {"old_string": "nonexistent", "new_string": "x"},
    ])
    result = json.loads(await multi_edit(str(sample_file), edits))
    assert "error" in result


@pytest.mark.asyncio
async def test_multi_edit_invalid_json(sample_file):
    result = json.loads(await multi_edit(str(sample_file), "not json"))
    assert "error" in result


@pytest.mark.asyncio
async def test_apply_patch_basic(sample_file):
    patch = """\
--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def foo():
-    return 1
+    return 42
"""
    result = json.loads(await apply_patch(str(sample_file), patch))
    assert result["status"] == "ok"
    content = sample_file.read_text()
    assert "return 42" in content


@pytest.mark.asyncio
async def test_apply_patch_nonexistent(tmp_path):
    set_project_root(tmp_path)
    result = json.loads(await apply_patch(str(tmp_path / "nope.py"), "patch"))
    assert "error" in result
