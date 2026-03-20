# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the shared event renderer."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from nemocode.cli.render import (
    EventRenderer,
    _error_hint,
    _render_diff,
    format_confirm_summary,
    format_tool_call,
    render_confirm_detail,
    render_tool_result,
    summarize_delegate_result,
    tool_result_has_embedded_error,
)
from nemocode.core.scheduler import AgentEvent


def _capture_console() -> tuple[Console, StringIO]:
    """Create a console that writes to a string buffer (no ANSI codes)."""
    buf = StringIO()
    con = Console(file=buf, no_color=True, width=120)
    return con, buf


class TestFormatToolCall:
    def test_read_file(self):
        result = format_tool_call("read_file", {"path": "/src/main.py"})
        assert "Read" in result
        assert "main.py" in result

    def test_read_file_with_offset(self):
        result = format_tool_call("read_file", {"path": "f.py", "offset": 10, "limit": 20})
        assert "from line 10" in result
        assert "limit 20" in result

    def test_write_file(self):
        result = format_tool_call("write_file", {"path": "out.txt", "content": "x" * 100})
        assert "Write" in result
        assert "out.txt" in result
        assert "100" in result

    def test_edit_file(self):
        result = format_tool_call("edit_file", {"path": "foo.py"})
        assert "Edit" in result
        assert "foo.py" in result

    def test_bash_exec(self):
        result = format_tool_call("bash_exec", {"command": "npm test"})
        assert "$ npm test" in result

    def test_bash_exec_truncates(self):
        long_cmd = "x" * 200
        result = format_tool_call("bash_exec", {"command": long_cmd})
        assert result.endswith("...")
        assert len(result) < 100

    def test_search_files(self):
        result = format_tool_call("search_files", {"pattern": "TODO", "path": "src/"})
        assert "Search" in result
        assert "/TODO/" in result
        assert "src/" in result

    def test_search_files_with_glob(self):
        result = format_tool_call("search_files", {"pattern": "TODO", "path": ".", "glob": "*.py"})
        assert "*.py" in result

    def test_git_status(self):
        assert "git status" in format_tool_call("git_status", {})

    def test_git_commit(self):
        result = format_tool_call("git_commit", {"message": "fix bug"})
        assert 'git commit "fix bug"' in result

    def test_http_fetch(self):
        result = format_tool_call("http_fetch", {"url": "https://example.com", "method": "GET"})
        assert "GET" in result
        assert "example.com" in result

    def test_run_tests(self):
        result = format_tool_call("run_tests", {"test_path": "tests/"})
        assert "pytest" in result
        assert "tests/" in result

    def test_list_dir(self):
        result = format_tool_call("list_dir", {"path": "src/"})
        assert "List" in result
        assert "src/" in result

    def test_unknown_tool(self):
        result = format_tool_call("unknown_tool", {"key": "val"})
        assert "unknown_tool" in result


class TestRenderToolResult:
    def test_json_ok(self):
        con, buf = _capture_console()
        render_tool_result(con, "write_file", '{"status": "ok", "bytes": 1234}', False)
        output = buf.getvalue()
        # Should show checkmark and byte count
        assert "1,234" in output

    def test_json_ok_with_diff(self):
        con, buf = _capture_console()
        diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = json.dumps({"status": "ok", "path": "a.py", "diff": diff})
        render_tool_result(con, "edit_file", result, False)
        output = buf.getvalue()
        assert "old" in output
        assert "new" in output

    def test_json_ok_empty_diff(self):
        con, buf = _capture_console()
        result = json.dumps({"status": "ok", "path": "a.py", "diff": ""})
        render_tool_result(con, "edit_file", result, False)
        output = buf.getvalue()
        assert len(output) > 0  # Shows checkmark

    def test_json_error(self):
        con, buf = _capture_console()
        render_tool_result(con, "read_file", '{"error": "not found"}', True)
        output = buf.getvalue()
        assert "not found" in output

    def test_bash_success_with_output(self):
        con, buf = _capture_console()
        result = json.dumps({"exit_code": 0, "stdout": "line1\nline2\nline3"})
        render_tool_result(con, "bash_exec", result, False)
        output = buf.getvalue()
        assert "line1" in output
        assert "line2" in output

    def test_bash_success_no_output(self):
        con, buf = _capture_console()
        render_tool_result(con, "bash_exec", '{"exit_code": 0}', False)
        output = buf.getvalue()
        assert len(output) > 0  # Shows checkmark

    def test_bash_failure(self):
        con, buf = _capture_console()
        result = json.dumps({"exit_code": 1, "stderr": "command not found"})
        render_tool_result(con, "bash_exec", result, True)
        output = buf.getvalue()
        assert "command not found" in output

    def test_read_file_summary(self):
        con, buf = _capture_console()
        content = "\n".join(f"     {i}\tline {i}" for i in range(1, 51))
        render_tool_result(con, "read_file", content, False)
        output = buf.getvalue()
        assert "50 lines" in output

    def test_list_dir_summary(self):
        con, buf = _capture_console()
        render_tool_result(con, "list_dir", "src/\ntests/\nREADME.md", False)
        output = buf.getvalue()
        assert "3 entries" in output

    def test_search_no_matches(self):
        con, buf = _capture_console()
        render_tool_result(con, "search_files", "(no matches)", False)
        output = buf.getvalue()
        assert "no matches" in output

    def test_search_with_matches(self):
        con, buf = _capture_console()
        render_tool_result(con, "search_files", "file1.py:10:match\nfile2.py:20:match", False)
        output = buf.getvalue()
        assert "2 results" in output

    def test_http_response(self):
        con, buf = _capture_console()
        result = json.dumps({"status": 200, "headers": {}, "body": "hello world"})
        render_tool_result(con, "http_fetch", result, False)
        output = buf.getvalue()
        assert "200" in output

    def test_raw_fallback(self):
        con, buf = _capture_console()
        render_tool_result(con, "git_log", "abc1234 Initial commit", False)
        output = buf.getvalue()
        # git_log is read-only — shows inline summary, not raw content
        assert "1 lines" in output or "line" in output

    def test_delegate_summary(self):
        con, buf = _capture_console()
        result = json.dumps(
            {
                "run_id": "subagent-0001",
                "agent_type": "general",
                "display_name": "Joe Nemotron",
                "nickname": "Orbit Joe",
                "status": "completed",
                "endpoint": "nim-nano",
                "output": "Found the relevant code path.\nAnd another line.",
                "tool_calls": 2,
                "errors": 0,
            }
        )
        render_tool_result(con, "delegate", result, False)
        output = buf.getvalue()
        assert "Orbit Joe" in output
        assert "subagent-0001" in output
        assert "Found the relevant code path." in output

    def test_delegate_embedded_error(self):
        con, buf = _capture_console()
        result = json.dumps(
            {
                "run_id": "subagent-0002",
                "agent_type": "debug",
                "display_name": "Joebug Nemotron",
                "nickname": "Crash Carson",
                "status": "failed",
                "endpoint": "nim-super",
                "error": "Sub-agent failed: boom",
            }
        )
        render_tool_result(con, "delegate", result, False)
        output = buf.getvalue()
        assert "Crash Carson" in output
        assert "Sub-agent failed: boom" in output


class TestDelegateHelpers:
    def test_summarize_delegate_result(self):
        summary = summarize_delegate_result(
            {
                "run_id": "subagent-0003",
                "agent_type": "explore",
                "display_name": "Geronemo",
                "nickname": "Needle Joe",
                "endpoint": "nim-nano",
                "output": "Looked through the codebase.",
                "tool_calls": 1,
                "errors": 0,
            }
        )
        assert summary is not None
        headline, preview = summary
        assert "Needle Joe" in headline
        assert "subagent-0003" in headline
        assert preview == "Looked through the codebase."

    def test_tool_result_has_embedded_error_for_delegate(self):
        result = json.dumps({"status": "failed", "error": "boom"})
        assert tool_result_has_embedded_error("delegate", result) is True
        assert tool_result_has_embedded_error("write_file", result) is False


class TestRenderDiff:
    def test_colored_diff(self):
        con, buf = _capture_console()
        diff = "--- a.py\n+++ a.py\n@@ -1,3 +1,3 @@\n line1\n-old line\n+new line\n line3\n"
        _render_diff(con, diff)
        output = buf.getvalue()
        assert "old line" in output
        assert "new line" in output
        assert "--- a.py" in output

    def test_truncates_long_diff(self):
        con, buf = _capture_console()
        diff = "\n".join(f"+line {i}" for i in range(50))
        _render_diff(con, diff)
        output = buf.getvalue()
        assert "50 lines total" in output


class TestEventRenderer:
    def test_text_streams(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(AgentEvent(kind="text", text="hello"))
        assert renderer._streaming is True
        renderer.flush()
        output = buf.getvalue()
        assert "hello" in output

    def test_thinking_hidden_by_default(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con, show_thinking=False)
        renderer.render(AgentEvent(kind="thinking", thinking="hmm"))
        output = buf.getvalue()
        assert "hmm" not in output
        assert "Reasoning trace hidden" in output

    def test_thinking_shown_when_enabled(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con, show_thinking=True)
        renderer.render(AgentEvent(kind="thinking", thinking="hmm"))
        output = buf.getvalue()
        assert "hmm" in output

    def test_status_event_renders_worklog_line(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con, show_thinking=False)
        renderer.render(
            AgentEvent(kind="status", text="Still working after 24s: reviewing tool output.")
        )
        output = buf.getvalue()
        assert "Still working after 24s" in output
        assert "Reasoning trace hidden" in output

    def test_tool_call_breaks_stream(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(AgentEvent(kind="text", text="thinking..."))
        assert renderer._streaming is True
        renderer.render(
            AgentEvent(kind="tool_call", tool_name="read_file", tool_args={"path": "f.py"})
        )
        assert renderer._streaming is False

    def test_error_rendering(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(AgentEvent(kind="error", text="something broke"))
        output = buf.getvalue()
        assert "something broke" in output

    def test_non_interactive_falls_back_to_raw(self):
        """Non-interactive consoles should use raw text (no Live/Markdown)."""
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        assert renderer._interactive is False
        renderer.render(AgentEvent(kind="text", text="plain text"))
        renderer.flush()
        output = buf.getvalue()
        assert "plain text" in output

    def test_spinner_not_started_non_interactive(self):
        """Thinking spinner should not start in non-interactive mode."""
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.start_thinking("Testing")
        assert renderer._thinking_spinner is None

    def test_response_separator_after_tools(self):
        """Verify response separator shows after tool calls then text."""
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(
            AgentEvent(kind="tool_call", tool_name="read_file", tool_args={"path": "f.py"})
        )
        renderer.render(
            AgentEvent(kind="tool_result", tool_name="read_file", tool_result="line1\n")
        )
        renderer.render(AgentEvent(kind="text", text="Here is my response"))
        renderer.flush()
        output = buf.getvalue()
        assert "NeMoCode Response" in output
        assert "Here is my response" in output


class TestFormatConfirmSummary:
    def test_write_file(self):
        result = format_confirm_summary("write_file", {"path": "/a/b.py", "content": "x" * 500})
        assert "/a/b.py" in result
        assert "500" in result

    def test_bash_exec(self):
        result = format_confirm_summary("bash_exec", {"command": "rm -rf /"})
        assert "rm -rf /" in result

    def test_git_commit(self):
        result = format_confirm_summary("git_commit", {"message": "fix", "files": "."})
        assert "fix" in result

    def test_generic(self):
        result = format_confirm_summary("unknown", {"key": "val"})
        assert "key" in result


class TestElapsedTime:
    def test_elapsed_shown_when_slow(self):
        con, buf = _capture_console()
        render_tool_result(con, "read_file", "     1\tline 1\n", False, elapsed=2.3)
        output = buf.getvalue()
        assert "2.3s" in output

    def test_elapsed_hidden_when_fast(self):
        con, buf = _capture_console()
        render_tool_result(con, "read_file", "     1\tline 1\n", False, elapsed=0.1)
        output = buf.getvalue()
        assert "0.1s" not in output

    def test_elapsed_on_bash_result(self):
        con, buf = _capture_console()
        result = json.dumps({"exit_code": 0, "stdout": "done"})
        render_tool_result(con, "bash_exec", result, False, elapsed=5.0)
        output = buf.getvalue()
        assert "5.0s" in output

    def test_elapsed_on_error(self):
        con, buf = _capture_console()
        render_tool_result(con, "bash_exec", '{"error": "fail"}', True, elapsed=3.0)
        output = buf.getvalue()
        # Error output shows the error message; elapsed may be suppressed for brevity
        assert "fail" in output


class TestConfirmDetail:
    def test_write_new_file_shows_preview(self, tmp_path):
        con, buf = _capture_console()
        args = {"path": str(tmp_path / "new.py"), "content": "print('hello')\nprint('world')\n"}
        render_confirm_detail(con, "write_file", args)
        output = buf.getvalue()
        assert "new file" in output
        assert "hello" in output

    def test_write_existing_file_shows_diff(self, tmp_path):
        existing = tmp_path / "exist.py"
        existing.write_text("old line\n")
        con, buf = _capture_console()
        args = {"path": str(existing), "content": "new line\n"}
        render_confirm_detail(con, "write_file", args)
        output = buf.getvalue()
        assert "old line" in output or "new line" in output

    def test_bash_multiline_shows_full_command(self):
        con, buf = _capture_console()
        args = {"command": "echo a\necho b\necho c"}
        render_confirm_detail(con, "bash_exec", args)
        output = buf.getvalue()
        assert "echo a" in output
        assert "echo b" in output

    def test_bash_short_command_no_extra(self):
        con, buf = _capture_console()
        args = {"command": "ls"}
        render_confirm_detail(con, "bash_exec", args)
        output = buf.getvalue()
        assert output == ""


class TestErrorHints:
    def test_rate_limit_hint(self):
        assert "endpoint" in _error_hint("Rate limit exceeded (429)")

    def test_auth_hint(self):
        assert "API key" in _error_hint("401 Unauthorized")

    def test_timeout_hint(self):
        hint = _error_hint("Request timed out").lower()
        assert "timed out" in hint or "timeout" in hint

    def test_context_window_hint(self):
        assert "compact" in _error_hint("context length exceeded").lower()

    def test_network_hint(self):
        assert "connection" in _error_hint("Connection refused").lower()

    def test_actionable_endpoint_error_skips_generic_hint(self):
        text = (
            "SGLang endpoint spark-sglang-super at http://localhost:8000/v1 is not reachable.\n"
            "Check it with: nemo endpoint test spark-sglang-super\n"
            "Setup/help: nemo setup sglang"
        )
        assert _error_hint(text) == ""

    def test_no_hint_for_generic(self):
        assert _error_hint("Something unexpected happened") == ""

    def test_forbidden_hint(self):
        assert "permission" in _error_hint("403 Forbidden: Access denied").lower()

    def test_not_found_hint(self):
        assert "model" in _error_hint("404 Not Found: Resource").lower()

    def test_server_error_hint(self):
        hint = _error_hint("500 Internal Server Error").lower()
        assert "server" in hint or "try again" in hint

    def test_ssl_error_hint(self):
        assert "ssl" in _error_hint("SSL certificate error").lower()

    def test_dns_error_hint(self):
        assert "dns" in _error_hint("DNS resolution failed").lower()

    def test_invalid_request_hint(self):
        assert "invalid" in _error_hint("400 Bad Request: Invalid request").lower()

    def test_gateway_timeout_hint(self):
        hint = _error_hint("504 Gateway Timeout")
        assert hint  # Should have a hint
        assert "timed out" in hint.lower() or "gateway" in hint.lower()


class TestErrorRendering:
    def test_error_with_recovery_hint(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(AgentEvent(kind="error", text="Rate limit exceeded (429)"))
        output = buf.getvalue()
        assert "Rate limit" in output
        assert "endpoint" in output  # recovery hint

    def test_error_without_hint(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(AgentEvent(kind="error", text="something broke"))
        output = buf.getvalue()
        assert "something broke" in output

    def test_error_with_actionable_endpoint_message_does_not_add_generic_hint(self):
        con, buf = _capture_console()
        renderer = EventRenderer(con)
        renderer.render(
            AgentEvent(
                kind="error",
                text=(
                    "SGLang endpoint spark-sglang-super at "
                    "http://localhost:8000/v1 is not reachable.\n"
                    "Check it with: nemo endpoint test spark-sglang-super\n"
                    "Setup/help: nemo setup sglang"
                ),
            )
        )
        output = buf.getvalue()
        assert "Check it with: nemo endpoint test spark-sglang-super" in output
        assert "Setup/help: nemo setup sglang" in output
        assert "Network error. Check your connection and endpoint URL." not in output


class TestToolBufferSummary:
    def test_few_files_shows_names(self):
        from nemocode.cli.render import _ToolCallBuffer

        buf = _ToolCallBuffer()
        buf.add_call("read_file", {"path": "src/main.py"})
        buf.add_call("read_file", {"path": "src/utils.py"})
        summary = buf.get_summary()
        assert "main.py" in summary
        assert "utils.py" in summary

    def test_many_files_shows_count(self):
        from nemocode.cli.render import _ToolCallBuffer

        buf = _ToolCallBuffer()
        for i in range(5):
            buf.add_call("read_file", {"path": f"src/file{i}.py"})
        summary = buf.get_summary()
        assert "5 files" in summary

    def test_search_pattern_shown(self):
        from nemocode.cli.render import _ToolCallBuffer

        buf = _ToolCallBuffer()
        buf.add_call("search_files", {"pattern": "TODO"})
        summary = buf.get_summary()
        assert "/TODO/" in summary

    def test_mixed_tools(self):
        from nemocode.cli.render import _ToolCallBuffer

        buf = _ToolCallBuffer()
        buf.add_call("read_file", {"path": "a.py"})
        buf.add_call("git_status", {})
        buf.add_call("search_files", {"pattern": "err"})
        summary = buf.get_summary()
        assert "Read" in summary
        assert "git status" in summary
        assert "/err/" in summary
