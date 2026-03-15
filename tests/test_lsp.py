# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the LSP client — construction, message formatting, language detection."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from nemocode.core.lsp import (
    Diagnostic,
    Location,
    LSPClient,
    _JsonRpcTransport,
    detect_language,
    find_server,
)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("foo.py") == "python"
        assert detect_language("/home/user/bar.pyi") == "python"

    def test_typescript(self):
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "typescript"

    def test_javascript(self):
        assert detect_language("index.js") == "javascript"
        assert detect_language("lib.mjs") == "javascript"

    def test_rust(self):
        assert detect_language("main.rs") == "rust"

    def test_go(self):
        assert detect_language("main.go") == "go"

    def test_c_cpp(self):
        assert detect_language("util.c") == "c"
        assert detect_language("util.h") == "c"
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("header.hpp") == "cpp"

    def test_unknown_extension(self):
        assert detect_language("file.xyz") is None
        assert detect_language("Makefile") is None

    def test_case_insensitive_extension(self):
        assert detect_language("FOO.PY") == "python"
        assert detect_language("bar.RS") == "rust"


# ---------------------------------------------------------------------------
# Server detection
# ---------------------------------------------------------------------------


class TestFindServer:
    def test_returns_none_when_not_on_path(self):
        with patch("shutil.which", return_value=None):
            assert find_server("python") is None

    def test_finds_pyright(self):
        with patch("shutil.which", side_effect=lambda b: b if b == "pyright-langserver" else None):
            result = find_server("python")
            assert result is not None
            binary, args = result
            assert binary == "pyright-langserver"
            assert "--stdio" in args

    def test_falls_back_to_pylsp(self):
        def which_side(b):
            return b if b == "pylsp" else None

        with patch("shutil.which", side_effect=which_side):
            result = find_server("python")
            assert result is not None
            assert result[0] == "pylsp"

    def test_finds_rust_analyzer(self):
        with patch("shutil.which", side_effect=lambda b: b if b == "rust-analyzer" else None):
            result = find_server("rust")
            assert result is not None
            assert result[0] == "rust-analyzer"

    def test_unknown_language(self):
        assert find_server("brainfuck") is None


# ---------------------------------------------------------------------------
# JSON-RPC message framing
# ---------------------------------------------------------------------------


class TestJsonRpcTransport:
    def test_write_formats_content_length(self):
        transport = _JsonRpcTransport()
        proc = MagicMock()
        stdin = MagicMock()
        proc.stdin = stdin
        proc.returncode = None
        transport._proc = proc

        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        transport._write(body)

        written = stdin.write.call_args[0][0]
        header, payload = written.split(b"\r\n\r\n", 1)
        assert header.startswith(b"Content-Length: ")
        length = int(header.split(b": ", 1)[1])
        assert length == len(payload)
        assert json.loads(payload) == body

    def test_dispatch_response(self):
        transport = _JsonRpcTransport()
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        transport._pending[42] = fut

        transport._dispatch({"id": 42, "result": {"capabilities": {}}})
        assert fut.done()
        assert fut.result() == {"capabilities": {}}
        loop.close()

    def test_dispatch_error_response(self):
        transport = _JsonRpcTransport()
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        transport._pending[7] = fut

        transport._dispatch(
            {"id": 7, "error": {"code": -32600, "message": "Invalid request"}}
        )
        assert fut.done()
        with pytest.raises(RuntimeError, match="Invalid request"):
            fut.result()
        loop.close()

    def test_dispatch_notification(self):
        transport = _JsonRpcTransport()
        transport._dispatch(
            {
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": "file:///foo.py", "diagnostics": []},
            }
        )
        notes = transport.pop_notifications("textDocument/publishDiagnostics")
        assert len(notes) == 1

    def test_pop_notifications_clears(self):
        transport = _JsonRpcTransport()
        transport._dispatch({"method": "window/logMessage", "params": {}})
        assert len(transport.pop_notifications("window/logMessage")) == 1
        assert len(transport.pop_notifications("window/logMessage")) == 0


# ---------------------------------------------------------------------------
# LSPClient construction and parse helpers
# ---------------------------------------------------------------------------


class TestLSPClient:
    def test_initial_state(self):
        client = LSPClient()
        assert not client.is_running
        assert client._language == ""
        assert client._open_files == {}

    def test_parse_diagnostic(self):
        raw = {
            "message": "Undefined variable 'x'",
            "severity": 1,
            "range": {
                "start": {"line": 10, "character": 5},
                "end": {"line": 10, "character": 6},
            },
            "source": "pyright",
            "code": "reportUndefinedVariable",
        }
        diag = LSPClient._parse_diagnostic(raw)
        assert isinstance(diag, Diagnostic)
        assert diag.message == "Undefined variable 'x'"
        assert diag.severity == 1
        assert diag.severity_label == "error"
        assert diag.line == 10
        assert diag.col == 5
        assert diag.source == "pyright"

    def test_parse_locations_single(self):
        raw = {
            "uri": "file:///src/foo.py",
            "range": {
                "start": {"line": 5, "character": 0},
                "end": {"line": 5, "character": 10},
            },
        }
        locs = LSPClient._parse_locations(raw)
        assert len(locs) == 1
        assert locs[0].uri == "file:///src/foo.py"
        assert locs[0].line == 5

    def test_parse_locations_list(self):
        raw = [
            {
                "uri": "file:///a.py",
                "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}},
            },
            {
                "uri": "file:///b.py",
                "range": {"start": {"line": 2, "character": 3}, "end": {"line": 2, "character": 8}},
            },
        ]
        locs = LSPClient._parse_locations(raw)
        assert len(locs) == 2
        assert locs[1].uri == "file:///b.py"
        assert locs[1].col == 3

    def test_parse_locations_link(self):
        raw = [
            {
                "targetUri": "file:///target.py",
                "targetRange": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 10, "character": 0},
                },
                "targetSelectionRange": {
                    "start": {"line": 3, "character": 4},
                    "end": {"line": 3, "character": 12},
                },
            }
        ]
        locs = LSPClient._parse_locations(raw)
        assert len(locs) == 1
        assert locs[0].uri == "file:///target.py"
        assert locs[0].line == 3
        assert locs[0].col == 4

    def test_parse_locations_none(self):
        assert LSPClient._parse_locations(None) == []

    def test_extract_hover_string(self):
        assert LSPClient._extract_hover_text("hello") == "hello"

    def test_extract_hover_markup_content(self):
        assert (
            LSPClient._extract_hover_text({"kind": "markdown", "value": "**int**"})
            == "**int**"
        )

    def test_extract_hover_list(self):
        result = LSPClient._extract_hover_text(
            [{"language": "python", "value": "def foo():"}, "A function"]
        )
        assert "def foo():" in result
        assert "A function" in result


# ---------------------------------------------------------------------------
# Location data class
# ---------------------------------------------------------------------------


class TestLocation:
    def test_path_from_file_uri(self):
        loc = Location(uri="file:///home/user/foo.py", line=0, col=0)
        assert loc.path == "/home/user/foo.py"

    def test_path_from_non_file_uri(self):
        loc = Location(uri="untitled:1", line=0, col=0)
        assert loc.path == "untitled:1"


# ---------------------------------------------------------------------------
# Diagnostic data class
# ---------------------------------------------------------------------------


class TestDiagnostic:
    def test_severity_labels(self):
        assert Diagnostic(message="", severity=1).severity_label == "error"
        assert Diagnostic(message="", severity=2).severity_label == "warning"
        assert Diagnostic(message="", severity=3).severity_label == "info"
        assert Diagnostic(message="", severity=4).severity_label == "hint"
        assert Diagnostic(message="", severity=99).severity_label == "unknown"
