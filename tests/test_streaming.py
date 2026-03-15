# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test streaming types."""

from __future__ import annotations

from nemocode.core.streaming import (
    CompletionResult,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)


class TestMessage:
    def test_basic_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_calls == []

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="tc1", name="read_file", arguments={"path": "test.py"})
        msg = Message(role=Role.ASSISTANT, content="Let me read that", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read_file"

    def test_tool_result_message(self):
        msg = Message(role=Role.TOOL, content="file contents", tool_call_id="tc1")
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "tc1"


class TestStreamChunk:
    def test_text_chunk(self):
        chunk = StreamChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.thinking == ""
        assert chunk.tool_calls == []

    def test_usage_chunk(self):
        chunk = StreamChunk(usage={"prompt_tokens": 100, "completion_tokens": 50})
        assert chunk.usage["prompt_tokens"] == 100


class TestCompletionResult:
    def test_basic_result(self):
        result = CompletionResult(content="Response", finish_reason="stop")
        assert result.content == "Response"
        assert result.tool_calls == []
