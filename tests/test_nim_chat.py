# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemocode.config.schema import (
    Endpoint,
    Manifest,
)
from nemocode.core.streaming import Message, Role
from nemocode.providers.nim_chat import NIMChatProvider


@pytest.fixture
def provider(sample_endpoint: Endpoint, sample_manifest: Manifest) -> NIMChatProvider:
    return NIMChatProvider(
        endpoint=sample_endpoint,
        manifest=sample_manifest,
        api_key="test-key",
    )


class TestNIMChatBuildBody:
    def test_nim_chat_build_body(self, provider: NIMChatProvider):
        messages = [
            Message(role=Role.SYSTEM, content="You are a coding assistant."),
            Message(role=Role.USER, content="Hello"),
        ]
        body = provider._build_body(messages)

        assert body["model"] == provider.endpoint.model_id
        assert len(body["messages"]) == 2
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == provider.endpoint.max_tokens
        assert body["top_p"] == 0.95

        assert "chat_template_kwargs" in body
        assert body["chat_template_kwargs"]["enable_thinking"] is True

    def test_nim_chat_build_body_without_manifest(self, sample_endpoint: Endpoint):
        provider = NIMChatProvider(endpoint=sample_endpoint, manifest=None, api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        body = provider._build_body(messages)

        assert body["model"] == sample_endpoint.model_id
        assert "chat_template_kwargs" not in body

    def test_nim_chat_build_body_with_tools(self, provider: NIMChatProvider):
        messages = [Message(role=Role.USER, content="Hello")]
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        body = provider._build_body(messages, tools=tools)

        assert "tools" in body
        assert len(body["tools"]) == 1

    def test_nim_chat_build_body_extra_body_override(self, provider: NIMChatProvider):
        messages = [Message(role=Role.USER, content="Hello")]
        body = provider._build_body(messages, extra_body={"temperature": 1.0})

        assert body["temperature"] == 1.0


class TestNIMChatStreamToolAssembly:
    @pytest.mark.asyncio
    async def test_nim_chat_stream_tool_assembly(self, provider: NIMChatProvider):
        tool_call_name_data = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc123",
                                    "function": {"name": "read_file", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        tool_call_args_data = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"path": "/tmp/test.py"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        finish_data = json.dumps(
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }
        )

        sse_lines = [
            f"data: {tool_call_name_data}",
            f"data: {tool_call_args_data}",
            f"data: {finish_data}",
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = mock_stream_ctx

        with patch("nemocode.providers.nim_chat.httpx.AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in provider.stream(
                [Message(role=Role.USER, content="Read the file")],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            ):
                chunks.append(chunk)

        tool_call_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_call_chunks) == 1
        tc = tool_call_chunks[0].tool_calls[0]
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/test.py"}
        assert tc.id == "call_abc123"

        finish_chunks = [c for c in chunks if c.finish_reason]
        assert len(finish_chunks) == 1
        assert finish_chunks[0].finish_reason == "tool_calls"
