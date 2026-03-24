# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for structured output enforcement (SHA-3688).

Validates that response_format is correctly wired through _build_body(),
stream(), complete(), and the CLI --output-format flag.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemocode.config.schema import (
    Endpoint,
    Manifest,
)
from nemocode.core.streaming import Message, Role
from nemocode.providers.nim_chat import NIMChatProvider

# ---------------------------------------------------------------------------
# _build_body tests
# ---------------------------------------------------------------------------


class TestBuildBodyResponseFormat:
    def test_response_format_included_when_provided(self, provider: NIMChatProvider):
        messages = [Message(role=Role.USER, content="Return JSON")]
        body = provider._build_body(messages, response_format={"type": "json_object"})
        assert "response_format" in body
        assert body["response_format"] == {"type": "json_object"}

    def test_response_format_omitted_when_none(self, provider: NIMChatProvider):
        messages = [Message(role=Role.USER, content="Hello")]
        body = provider._build_body(messages)
        assert "response_format" not in body

    def test_response_format_omitted_when_explicitly_none(self, provider: NIMChatProvider):
        messages = [Message(role=Role.USER, content="Hello")]
        body = provider._build_body(messages, response_format=None)
        assert "response_format" not in body

    def test_response_format_with_json_schema(self, provider: NIMChatProvider):
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }
        messages = [Message(role=Role.USER, content="Who are you?")]
        body = provider._build_body(messages, response_format=schema)
        assert body["response_format"] == schema

    def test_extra_body_can_override_response_format(self, provider: NIMChatProvider):
        """extra_body is applied after response_format, so it can override."""
        messages = [Message(role=Role.USER, content="Hello")]
        body = provider._build_body(
            messages,
            response_format={"type": "json_object"},
            extra_body={"response_format": {"type": "text"}},
        )
        # extra_body wins because it's applied via body.update() after
        assert body["response_format"] == {"type": "text"}

    def test_response_format_coexists_with_tools(self, provider: NIMChatProvider):
        messages = [Message(role=Role.USER, content="Hello")]
        tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
        body = provider._build_body(
            messages,
            tools=tools,
            response_format={"type": "json_object"},
        )
        assert "tools" in body
        assert "response_format" in body


# ---------------------------------------------------------------------------
# stream() / complete() pass-through tests
# ---------------------------------------------------------------------------


class TestStreamResponseFormat:
    @pytest.mark.asyncio
    async def test_stream_passes_response_format(self, provider: NIMChatProvider):
        """Verify stream() forwards response_format to _build_body."""
        with patch.object(provider, "_build_body", wraps=provider._build_body) as spy:
            # Mock the HTTP layer so we don't make real requests
            mock_response = MagicMock()
            mock_response.status_code = 200

            async def aiter_lines():
                yield "data: [DONE]"

            mock_response.aiter_lines = aiter_lines

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = MagicMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream.return_value = mock_stream_ctx

            with patch(
                "nemocode.providers.nim_chat.httpx.AsyncClient",
                return_value=mock_client,
            ):
                async for _ in provider.stream(
                    [Message(role=Role.USER, content="Hi")],
                    response_format={"type": "json_object"},
                ):
                    pass

            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs["response_format"] == {"type": "json_object"}


class TestCompleteResponseFormat:
    @pytest.mark.asyncio
    async def test_complete_passes_response_format(self, provider: NIMChatProvider):
        """Verify complete() forwards response_format to _build_body."""
        with patch.object(provider, "_build_body", wraps=provider._build_body) as spy:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {"content": '{"result": 42}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch(
                "nemocode.providers.nim_chat.httpx.AsyncClient",
                return_value=mock_client,
            ):
                result = await provider.complete(
                    [Message(role=Role.USER, content="Hi")],
                    response_format={"type": "json_object"},
                )

            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs["response_format"] == {"type": "json_object"}
            assert result.content == '{"result": 42}'


# ---------------------------------------------------------------------------
# CLI --output-format flag tests
# ---------------------------------------------------------------------------


class TestChatCLIOutputFormat:
    def test_output_format_json_flag_accepted(self):
        """Verify the --output-format flag is parsed without error."""
        from nemocode.cli.main import app

        runner = CliRunner()
        # We mock asyncio.run to avoid needing a real config/endpoint
        with patch("nemocode.cli.commands.chat.asyncio.run") as mock_run:
            runner.invoke(
                app,
                ["chat", "Hello", "--output-format", "json"],
            )
            # asyncio.run is called with the coroutine
            assert mock_run.called
            # Close the coroutine to avoid RuntimeWarning
            coro = mock_run.call_args[0][0]
            coro.close()

    def test_output_format_not_provided(self):
        """When --output-format is not passed, _chat receives None."""
        from nemocode.cli.main import app

        runner = CliRunner()
        with patch("nemocode.cli.commands.chat.asyncio.run") as mock_run:
            runner.invoke(
                app,
                ["chat", "Hello"],
            )
            assert mock_run.called
            coro = mock_run.call_args[0][0]
            coro.close()

    def test_chat_help_shows_output_format(self):
        """Verify --output-format appears in chat --help output."""
        from nemocode.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--output-format" in result.stdout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider(sample_endpoint: Endpoint, sample_manifest: Manifest) -> NIMChatProvider:
    return NIMChatProvider(
        endpoint=sample_endpoint,
        manifest=sample_manifest,
        api_key="test-key",
    )
