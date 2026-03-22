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

    def test_load_no_tools_with_empty_list(self):
        registry = load_tools([])
        assert registry.list_tools() == []

    def test_load_tools_discovers_project_plugin(self, monkeypatch, tmp_path: Path):
        plugin_dir = tmp_path / ".nemocode" / "tools"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo_plugin.py").write_text(
            "import json\n"
            "from nemocode.tools import tool\n\n"
            '@tool(name="demo_plugin", description="Demo plugin", category="plugin")\n'
            "async def demo_plugin(subject: str = 'world') -> str:\n"
            '    return json.dumps({"hello": subject})\n'
        )

        monkeypatch.chdir(tmp_path)
        registry = load_tools()
        names = {t.name for t in registry.list_tools()}
        assert "demo_plugin" in names

    def test_load_tools_filters_project_plugin_by_category(self, monkeypatch, tmp_path: Path):
        plugin_dir = tmp_path / ".nemocode" / "tools"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo_plugin.py").write_text(
            "from nemocode.tools import tool\n\n"
            '@tool(name="demo_plugin", description="Demo plugin", category="plugin")\n'
            "async def demo_plugin() -> str:\n"
            '    return "ok"\n'
        )

        monkeypatch.chdir(tmp_path)
        registry = load_tools(["plugin"])
        names = {t.name for t in registry.list_tools()}
        assert names == {"demo_plugin"}

    def test_load_tools_uses_explicit_project_dir_for_plugins(self, monkeypatch, tmp_path: Path):
        plugin_dir = tmp_path / ".nemocode" / "tools"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo_plugin.py").write_text(
            "from nemocode.tools import tool\n\n"
            '@tool(name="demo_plugin", description="Demo plugin", category="plugin")\n'
            "async def demo_plugin() -> str:\n"
            '    return "ok"\n'
        )
        nested_dir = tmp_path / "services" / "api"
        nested_dir.mkdir(parents=True)

        monkeypatch.chdir(nested_dir)
        registry = load_tools(["plugin"], project_dir=tmp_path)
        names = {t.name for t in registry.list_tools()}
        assert names == {"demo_plugin"}

    def test_load_no_tools_with_empty_list_ignores_plugins(self, monkeypatch, tmp_path: Path):
        plugin_dir = tmp_path / ".nemocode" / "tools"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo_plugin.py").write_text(
            "from nemocode.tools import tool\n\n"
            '@tool(name="demo_plugin", description="Demo plugin", category="plugin")\n'
            "async def demo_plugin() -> str:\n"
            '    return "ok"\n'
        )

        monkeypatch.chdir(tmp_path)
        registry = load_tools([])
        assert registry.list_tools() == []

    @pytest.mark.asyncio
    async def test_search_files_accepts_text_alias(self, tmp_path: Path):
        from nemocode.tools.rg import search_files

        target = tmp_path / "sample.py"
        target.write_text("needle = 1\n")

        result = await search_files(text="needle", path=str(tmp_path))
        assert "sample.py:1:needle = 1" in result

    @pytest.mark.asyncio
    async def test_execute_project_plugin_tool(self, monkeypatch, tmp_path: Path):
        plugin_dir = tmp_path / ".nemocode" / "tools"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo_plugin.py").write_text(
            "import json\n"
            "from nemocode.tools import tool\n\n"
            '@tool(name="demo_plugin", description="Demo plugin", category="plugin")\n'
            "async def demo_plugin(subject: str) -> str:\n"
            '    return json.dumps({"hello": subject})\n'
        )

        monkeypatch.chdir(tmp_path)
        registry = load_tools(["plugin"])
        result = await registry.execute("demo_plugin", {"subject": "mars"})
        assert json.loads(result) == {"hello": "mars"}


class TestWebTools:
    @pytest.mark.asyncio
    async def test_web_fetch_extracts_main_content(self, monkeypatch):
        from nemocode.tools.web import web_fetch

        html_doc = """
        <html>
          <head><title>Example Docs</title><style>.noise { display:none; }</style></head>
          <body>
            <nav>Navigation noise</nav>
            <main>
              <h1>Install Guide</h1>
              <p>Step one: install the package.</p>
              <p>Step two: run the command.</p>
            </main>
            <script>console.log("ignore me")</script>
          </body>
        </html>
        """

        class _FakeHeaders:
            def __init__(self, content_type: str) -> None:
                self._content_type = content_type

            def get(self, name: str, default=None):
                if name.lower() == "content-type":
                    return self._content_type
                return default

            def get_content_charset(self):
                return "utf-8"

        class _FakeResponse:
            def __init__(self, body: str) -> None:
                self.headers = _FakeHeaders("text/html; charset=utf-8")
                self.status = 200
                self._body = body.encode()

            def read(self):
                return self._body

            def geturl(self):
                return "https://example.com/docs"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(
            "nemocode.tools.web.urllib.request.urlopen",
            lambda req, timeout=20: _FakeResponse(html_doc),
        )

        result = json.loads(await web_fetch("https://example.com/docs"))
        assert result["status"] == 200
        assert result["title"] == "Example Docs"
        assert "Install Guide" in result["content"]
        assert "Step one: install the package." in result["content"]
        assert "Navigation noise" not in result["content"]
        assert "ignore me" not in result["content"]

    @pytest.mark.asyncio
    async def test_web_fetch_plain_text_passthrough(self, monkeypatch):
        from nemocode.tools.web import web_fetch

        class _FakeHeaders:
            def get(self, name: str, default=None):
                if name.lower() == "content-type":
                    return "text/plain; charset=utf-8"
                return default

            def get_content_charset(self):
                return "utf-8"

        class _FakeResponse:
            def __init__(self) -> None:
                self.headers = _FakeHeaders()
                self.status = 200

            def read(self):
                return b"line one\nline two\n"

            def geturl(self):
                return "https://example.com/plain.txt"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(
            "nemocode.tools.web.urllib.request.urlopen",
            lambda req, timeout=20: _FakeResponse(),
        )

        result = json.loads(await web_fetch("https://example.com/plain.txt"))
        assert result["content"] == "line one\nline two"
        assert result["title"] == ""


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

    @pytest.mark.asyncio
    async def test_edit_file_returns_diff(self, tmp_path: Path):
        test_file = tmp_path / "diff.txt"
        test_file.write_text("hello world\nfoo bar\n")

        from nemocode.tools.fs import edit_file, set_project_root

        set_project_root(tmp_path)
        result = await edit_file(str(test_file), "world", "universe")
        data = json.loads(result)
        assert data["status"] == "ok"
        assert "diff" in data
        assert "-hello world" in data["diff"]
        assert "+hello universe" in data["diff"]

    @pytest.mark.asyncio
    async def test_write_file_returns_diff(self, tmp_path: Path):
        from nemocode.tools.fs import set_project_root, write_file

        set_project_root(tmp_path)
        target = tmp_path / "new.txt"
        result = await write_file(str(target), "line1\nline2\n")
        data = json.loads(result)
        assert data["status"] == "ok"
        assert "diff" in data
        assert "+line1" in data["diff"]

    @pytest.mark.asyncio
    async def test_write_overwrite_shows_diff(self, tmp_path: Path):
        from nemocode.tools.fs import set_project_root, write_file

        set_project_root(tmp_path)
        target = tmp_path / "overwrite.txt"
        target.write_text("old content\n")
        result = await write_file(str(target), "new content\n")
        data = json.loads(result)
        assert "-old content" in data["diff"]
        assert "+new content" in data["diff"]


class TestUndoSystem:
    @pytest.mark.asyncio
    async def test_undo_edit(self, tmp_path: Path):
        from nemocode.tools.fs import _UNDO_STACK, edit_file, set_project_root, undo_last

        set_project_root(tmp_path)
        _UNDO_STACK.clear()

        test_file = tmp_path / "undo.txt"
        test_file.write_text("original")

        await edit_file(str(test_file), "original", "modified")
        assert test_file.read_text() == "modified"
        assert len(_UNDO_STACK) == 1

        result = undo_last()
        assert result["status"] == "ok"
        assert test_file.read_text() == "original"
        assert len(_UNDO_STACK) == 0

    @pytest.mark.asyncio
    async def test_undo_write_new_file(self, tmp_path: Path):
        from nemocode.tools.fs import _UNDO_STACK, set_project_root, undo_last, write_file

        set_project_root(tmp_path)
        _UNDO_STACK.clear()

        target = tmp_path / "created.txt"
        await write_file(str(target), "content")
        assert target.exists()

        result = undo_last()
        assert result["action"] == "deleted"
        assert not target.exists()

    def test_undo_empty_stack(self):
        from nemocode.tools.fs import _UNDO_STACK, undo_last

        _UNDO_STACK.clear()
        result = undo_last()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_undo_stack_limit(self, tmp_path: Path):
        from nemocode.tools.fs import _MAX_UNDO, _UNDO_STACK, set_project_root, write_file

        set_project_root(tmp_path)
        _UNDO_STACK.clear()

        # Write more files than the undo limit
        for i in range(_MAX_UNDO + 5):
            target = tmp_path / f"file_{i}.txt"
            await write_file(str(target), f"content {i}")

        assert len(_UNDO_STACK) == _MAX_UNDO
        _UNDO_STACK.clear()
