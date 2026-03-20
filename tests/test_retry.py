# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for provider retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nemocode.config.schema import Capability, Endpoint, EndpointTier
from nemocode.core.streaming import Message, Role
from nemocode.providers.nim_chat import NIMChatProvider, _retry_delay


@pytest.fixture
def provider():
    ep = Endpoint(
        name="test",
        tier=EndpointTier.DEV_HOSTED,
        base_url="https://test.api.com/v1",
        model_id="test-model",
        capabilities=[Capability.CHAT],
    )
    return NIMChatProvider(endpoint=ep, api_key="test-key")


@pytest.fixture
def messages():
    return [Message(role=Role.USER, content="hello")]


class TestRetryDelay:
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        with patch("nemocode.providers.nim_chat.asyncio.sleep") as mock_sleep:
            delay = await _retry_delay(0)
            assert delay == 1.0
            mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.asyncio
    async def test_second_attempt_doubles(self):
        with patch("nemocode.providers.nim_chat.asyncio.sleep") as mock_sleep:
            delay = await _retry_delay(1)
            assert delay == 2.0
            mock_sleep.assert_awaited_once_with(2.0)

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self):
        with patch("nemocode.providers.nim_chat.asyncio.sleep") as mock_sleep:
            delay = await _retry_delay(0, status_code=429, retry_after="5")
            assert delay == 5.0
            mock_sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_caps_at_30_seconds(self):
        with patch("nemocode.providers.nim_chat.asyncio.sleep"):
            delay = await _retry_delay(10)  # 2^10 = 1024, should cap at 30
            assert delay == 30.0


class TestStreamRetry:
    @pytest.mark.asyncio
    async def test_stream_yields_error_on_non_retryable(self, provider, messages):
        """Non-retryable status codes should yield error immediately."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 401
        mock_resp.aread = AsyncMock(return_value=b"Unauthorized")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nemocode.providers.nim_chat.httpx.AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in provider.stream(messages):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "401" in chunks[0].text
        assert "Unauthorized" in chunks[0].text


class TestCompleteRetry:
    @pytest.mark.asyncio
    async def test_complete_returns_error_on_non_retryable(self, provider, messages):
        """Non-retryable errors should return error result immediately."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_resp.headers = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nemocode.providers.nim_chat.httpx.AsyncClient", return_value=mock_client):
            result = await provider.complete(messages)

        assert "400" in result.content
        assert result.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_complete_allows_null_tool_calls(self, provider, messages):
        """OpenAI-compatible responses may return tool_calls as null."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "NeMoCode OK",
                        "reasoning_content": "short trace",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nemocode.providers.nim_chat.httpx.AsyncClient", return_value=mock_client):
            result = await provider.complete(messages)

        assert result.content == "NeMoCode OK"
        assert result.thinking == "short trace"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_local_backend_connection_error_is_actionable(self, messages):
        ep = Endpoint(
            name="Spark SGLang Super",
            tier=EndpointTier.LOCAL_SGLANG,
            base_url="http://localhost:8000/v1",
            model_id="nvidia/nemotron-3-super-120b-a12b",
            capabilities=[Capability.CHAT],
        )
        provider = NIMChatProvider(endpoint=ep, endpoint_name="spark-sglang-super")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("All connection attempts failed")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("nemocode.providers.nim_chat.httpx.AsyncClient", return_value=mock_client),
            patch("nemocode.providers.nim_chat._retry_delay", new=AsyncMock(return_value=0.0)),
        ):
            result = await provider.complete(messages)

        assert result.finish_reason == "error"
        assert "spark-sglang-super" in result.content
        assert "nemo endpoint test spark-sglang-super" in result.content
        assert "nemo setup sglang" in result.content
