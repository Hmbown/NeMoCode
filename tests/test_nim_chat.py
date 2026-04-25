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
from nemocode.providers.nim_chat import NIMChatProvider, _strictify_tools


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


class TestDeepSeekSentinelLeak:
    """NIM's hosted DeepSeek V3/V4 leaks raw tool-call special tokens
    (`\\n\\n`, `<`, `｜DSML｜`, `tool`, `_calls`) into the streamed content
    field before populating structured tool_calls. The provider must
    suppress those leaked sentinels."""

    @pytest.mark.asyncio
    async def test_deepseek_sentinel_suppressed_when_tool_calls_follow(
        self, provider: NIMChatProvider
    ):
        # Sequence observed against deepseek-ai/deepseek-v4-flash on NIM
        leaked_chunks = ["\n\n", "<", "｜DSML｜", "tool", "_c", "alls"]
        sse_lines = []
        for piece in leaked_chunks:
            sse_lines.append(
                "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
            )
        sse_lines.append(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_x",
                                        "function": {
                                            "name": "list_dir",
                                            "arguments": '{"path": "/tmp"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        )
        sse_lines.append(
            "data: "
            + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        )
        sse_lines.append("data: [DONE]")

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        chunks = []
        async for chunk in provider._process_stream(mock_response):
            chunks.append(chunk)

        text_emitted = "".join(c.text for c in chunks if c.text)
        # The whole sentinel leak should be suppressed.
        assert "DSML" not in text_emitted
        assert "｜" not in text_emitted
        assert "tool_calls" not in text_emitted
        # Tool call should still be parsed from the structured field.
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_calls[0].name == "list_dir"
        assert tool_chunks[0].tool_calls[0].arguments == {"path": "/tmp"}

    @pytest.mark.asyncio
    async def test_normal_content_still_streams(self, provider: NIMChatProvider):
        """Non-sentinel content must still stream through unchanged."""
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "world"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        chunks = []
        async for chunk in provider._process_stream(mock_response):
            chunks.append(chunk)

        text_emitted = "".join(c.text for c in chunks if c.text)
        assert text_emitted == "Hello world"

    @pytest.mark.asyncio
    async def test_lone_lt_followed_by_normal_text_flushes(
        self, provider: NIMChatProvider
    ):
        """A bare `<` followed by code (not a sentinel) must still be emitted."""
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "<"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "html>"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        chunks = []
        async for chunk in provider._process_stream(mock_response):
            chunks.append(chunk)

        text_emitted = "".join(c.text for c in chunks if c.text)
        assert text_emitted == "<html>"

    @pytest.mark.asyncio
    async def test_hallucinated_bash_exec_xml_is_dropped(
        self, provider: NIMChatProvider
    ):
        """Invented `<bash_exec>...</bash_exec>` blocks must be suppressed."""
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Sure, "}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "<bash_exec>"}}]}),
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "<command>ls /tmp</command>"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "</bash_exec>"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        chunks = []
        async for chunk in provider._process_stream(mock_response):
            chunks.append(chunk)

        text_emitted = "".join(c.text for c in chunks if c.text)
        # The intro text streams; the invented XML block is suppressed.
        assert "bash_exec" not in text_emitted
        assert "Sure, " in text_emitted

    @pytest.mark.asyncio
    async def test_real_tool_result_wrapper_is_not_dropped(
        self, provider: NIMChatProvider
    ):
        """`<tool_result>` is real V4 syntax — it must NOT be filtered."""
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "<tool_result>"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "</tool_result>"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = aiter_lines

        chunks = []
        async for chunk in provider._process_stream(mock_response):
            chunks.append(chunk)

        text_emitted = "".join(c.text for c in chunks if c.text)
        assert "<tool_result>" in text_emitted
        assert "ok" in text_emitted


class TestStrictifyTools:
    def test_adds_strict_flag_and_additional_properties(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        out = _strictify_tools(tools)
        assert out[0]["function"]["strict"] is True
        assert out[0]["function"]["parameters"]["additionalProperties"] is False

    def test_widens_optional_fields_to_allow_null(self):
        """Properties not in original `required` get anyOf-with-null treatment
        and end up in the new `required` list (per strict-mode rules)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        out = _strictify_tools(tools)
        params = out[0]["function"]["parameters"]
        assert sorted(params["required"]) == ["limit", "query"]
        assert params["properties"]["query"] == {"type": "string"}
        # `limit` was optional → widened
        limit = params["properties"]["limit"]
        assert "anyOf" in limit
        types = sorted(s["type"] for s in limit["anyOf"])
        assert types == ["integer", "null"]
        assert limit.get("default") is None

    def test_strips_unsupported_keywords(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 99,
                            },
                            "tags": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 5,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "tags"],
                    },
                },
            }
        ]
        out = _strictify_tools(tools)
        params = out[0]["function"]["parameters"]
        for prop in params["properties"].values():
            for k in ("minLength", "maxLength", "minItems", "maxItems"):
                assert k not in prop

    def test_does_not_mutate_input(self):
        original = {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        }
        original_copy = json.loads(json.dumps(original))
        _ = _strictify_tools([original])
        assert original == original_copy
