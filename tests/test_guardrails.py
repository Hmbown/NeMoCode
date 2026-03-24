# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    GuardrailsConfig,
)
from nemocode.providers.nim_guardrails import NIMGuardrailsProvider, SafetyResult


@pytest.fixture
def safety_endpoint() -> Endpoint:
    return Endpoint(
        name="Nemotron Content Safety 4B",
        tier=EndpointTier.DEV_HOSTED,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_id="nvidia/nemotron-content-safety-reasoning-4b",
        capabilities=[Capability.CHAT],
        max_tokens=4096,
    )


@pytest.fixture
def guardrails_config() -> GuardrailsConfig:
    return GuardrailsConfig(
        enabled=True,
        endpoint="nim-content-safety",
        timeout=5.0,
        reject_categories=["violence", "self_harm", "hate_speech"],
    )


@pytest.fixture
def provider(
    safety_endpoint: Endpoint, guardrails_config: GuardrailsConfig
) -> NIMGuardrailsProvider:
    return NIMGuardrailsProvider(
        endpoint=safety_endpoint,
        config=guardrails_config,
        api_key="test-key",
    )


class TestSafetyResult:
    def test_default_safe(self):
        result = SafetyResult()
        assert result.safe is True
        assert result.categories == {}
        assert result.blocked_categories == []
        assert result.error is None

    def test_unsafe(self):
        result = SafetyResult(
            safe=False,
            categories={"violence": 0.9},
            blocked_categories=["violence"],
        )
        assert result.safe is False
        assert len(result.blocked_categories) == 1


class TestNIMGuardrailsProvider:
    def test_init_defaults(self, safety_endpoint: Endpoint):
        provider = NIMGuardrailsProvider(endpoint=safety_endpoint)
        assert provider.config.enabled is False
        assert provider.config.endpoint == "nim-content-safety"

    def test_init_with_config(self, safety_endpoint: Endpoint, guardrails_config: GuardrailsConfig):
        provider = NIMGuardrailsProvider(
            endpoint=safety_endpoint,
            config=guardrails_config,
            api_key="test-key",
        )
        assert provider.config.enabled is True
        assert provider.config.timeout == 5.0

    @pytest.mark.asyncio
    async def test_check_empty_text(self, provider: NIMGuardrailsProvider):
        result = await provider.check("")
        assert result.safe is True

    @pytest.mark.asyncio
    async def test_check_whitespace_only(self, provider: NIMGuardrailsProvider):
        result = await provider.check("   \n  ")
        assert result.safe is True

    @pytest.mark.asyncio
    async def test_check_safe_response(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_scores": {
                                    "violence": 0.01,
                                    "self_harm": 0.0,
                                    "hate_speech": 0.02,
                                    "sexual_content": 0.05,
                                }
                            }
                        )
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("This is a harmless coding question.")

        assert result.safe is True
        assert result.error is None
        assert result.categories["violence"] == 0.01
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_blocked_response(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_scores": {
                                    "violence": 0.95,
                                    "self_harm": 0.01,
                                    "hate_speech": 0.03,
                                }
                            }
                        )
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Some violent content here.")

        assert result.safe is False
        assert "violence" in result.blocked_categories
        assert len(result.blocked_categories) == 1

    @pytest.mark.asyncio
    async def test_check_multiple_blocked(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_scores": {
                                    "violence": 0.9,
                                    "self_harm": 0.8,
                                    "hate_speech": 0.7,
                                }
                            }
                        )
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Very dangerous content.")

        assert result.safe is False
        assert len(result.blocked_categories) == 3

    @pytest.mark.asyncio
    async def test_check_endpoint_unreachable_connect_error(self, provider: NIMGuardrailsProvider):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Hello world")

        assert result.safe is True
        assert result.error == "Connection refused"

    @pytest.mark.asyncio
    async def test_check_endpoint_timeout(self, provider: NIMGuardrailsProvider):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Timed out"))

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Hello world")

        assert result.safe is True
        assert "Timed out" in result.error

    @pytest.mark.asyncio
    async def test_check_http_error(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Hello world")

        assert result.safe is True
        assert "HTTP 503" in result.error

    @pytest.mark.asyncio
    async def test_check_empty_choices(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Hello world")

        assert result.safe is True
        assert result.error == "empty response"

    @pytest.mark.asyncio
    async def test_check_invalid_json_response(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "not valid json"}}]}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Hello world")

        assert result.safe is True

    @pytest.mark.asyncio
    async def test_check_list_response(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "[1, 2, 3]"}}]}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Hello world")

        assert result.safe is True

    @pytest.mark.asyncio
    async def test_check_boundary_score_exactly_0_5(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_scores": {
                                    "violence": 0.5,
                                }
                            }
                        )
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Borderline content.")

        assert result.safe is True
        assert len(result.blocked_categories) == 0

    @pytest.mark.asyncio
    async def test_check_no_matching_reject_categories(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_scores": {
                                    "financial_advice": 0.95,
                                    "medical_advice": 0.8,
                                }
                            }
                        )
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            result = await provider.check("Some advice here.")

        assert result.safe is True

    def test_headers_include_api_key(self, provider: NIMGuardrailsProvider):
        headers = provider._headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_headers_no_api_key(self, safety_endpoint: Endpoint):
        provider = NIMGuardrailsProvider(endpoint=safety_endpoint, api_key=None)
        headers = provider._headers()
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_check_sends_correct_payload(self, provider: NIMGuardrailsProvider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"category_scores": {}})}}]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("nemocode.providers.nim_guardrails.httpx.AsyncClient", return_value=mock_client):
            await provider.check("Test input")

        call_args = mock_client.post.call_args
        body = call_args.kwargs["json"]
        assert body["model"] == "nvidia/nemotron-content-safety-reasoning-4b"
        assert body["temperature"] == 0.0
        assert body["messages"][0]["content"] == "Test input"
