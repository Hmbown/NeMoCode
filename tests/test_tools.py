# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test tool execution and schema generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemocode.tools import ToolRegistry, tool
from nemocode.tools.loader import load_tools


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()

        @tool(name="test_tool", description="A test tool", category="test")
        async def test_fn(arg1: str, arg2: int = 0) -> str:
            return json.dumps({"arg1": arg1, "arg2": arg2})

        registry.register_function(test_fn)
        td = registry.get("test_tool")
        assert td is not None
        assert td.name == "test_tool"
        assert td.category == "test"

    def test_schema_generation(self):
        @tool(name="schema_test", description="Test schema gen")
        async def test_fn(name: str, count: int, flag: bool = False) -> str:
            return ""

        td = test_fn._tool_def
        assert "name" in td.parameters["properties"]
        assert "count" in td.parameters["properties"]
        assert "flag" in td.parameters["properties"]
        assert "name" in td.parameters["required"]
        assert "count" in td.parameters["required"]
        assert "flag" not in td.parameters["required"]

    def test_openai_format_schemas(self):
        registry = ToolRegistry()

        @tool(name="test_tool", description="A test")
        async def test_fn(x: str) -> str:
            return x

        registry.register_function(test_fn)
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        registry = ToolRegistry()

        @tool(name="echo", description="Echo input")
        async def echo(text: str) -> str:
            return text

        registry.register_function(echo)
        result = await registry.execute("echo", {"text": "hello"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})
        data = json.loads(result)
        assert "error" in data

    def test_load_tools_by_category(self):
        registry = load_tools(["fs", "git"])
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "git_status" in names
        assert "bash_exec" not in names

    def test_load_all_tools(self):
        registry = load_tools()
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "bash_exec" in names
        assert "git_status" in names
        assert "search_files" in names
        assert "http_fetch" in names


class TestFSTools:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        from nemocode.tools.fs import read_file, set_project_root

        set_project_root(tmp_path)

        result = await read_file(str(test_file))
        assert "line1" in result
        assert "line2" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tmp_path: Path):
        from nemocode.tools.fs import read_file

        result = await read_file(str(tmp_path / "nonexistent.txt"))
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path: Path):
        from nemocode.tools.fs import set_project_root, write_file

        set_project_root(tmp_path)
        target = tmp_path / "output.txt"
        result = await write_file(str(target), "hello world")
        data = json.loads(result)
        assert data["status"] == "ok"
        assert target.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_path: Path):
        test_file = tmp_path / "edit.txt"
        test_file.write_text("hello world")

        from nemocode.tools.fs import edit_file, set_project_root

        set_project_root(tmp_path)
        result = await edit_file(str(test_file), "world", "universe")
        data = json.loads(result)
        assert data["status"] == "ok"
        assert test_file.read_text() == "hello universe"

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self, tmp_path: Path):
        from nemocode.tools.fs import edit_file

        result = await edit_file(str(tmp_path / "nope.txt"), "a", "b")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_list_dir(self, project_dir: Path):
        from nemocode.tools.fs import list_dir, set_project_root

        set_project_root(project_dir)

        result = await list_dir(str(project_dir))
        assert "README.md" in result
        assert "main.py" in result
