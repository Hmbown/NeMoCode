# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM Guardrails provider — post-generation content safety checker.

Calls the nemotron-content-safety-4b endpoint to classify generated text
across 22 safety categories. Designed as a post-processor in the
chat/code streaming pipeline, not a full chat provider.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from nemocode.config.schema import Endpoint, GuardrailsConfig
from nemocode.providers import NIMProviderBase

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    safe: bool = True
    categories: dict[str, float] = field(default_factory=dict)
    blocked_categories: list[str] = field(default_factory=list)
    error: str | None = None


class NIMGuardrailsProvider(NIMProviderBase):
    def __init__(
        self,
        endpoint: Endpoint,
        config: GuardrailsConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(endpoint, api_key)
        self.config = config or GuardrailsConfig()

    async def check(self, text: str) -> SafetyResult:
        if not text.strip():
            return SafetyResult()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout)) as client:
                body = {
                    "model": self.endpoint.model_id,
                    "messages": [
                        {"role": "user", "content": text},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.0,
                }
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=self._headers(),
                )

                if resp.status_code != 200:
                    logger.warning(
                        "Guardrails endpoint returned HTTP %d: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return SafetyResult(
                        safe=True,
                        error=f"HTTP {resp.status_code}",
                    )

                data = resp.json()
                return self._parse_response(data, text)

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            logger.warning("Guardrails endpoint unreachable: %s", exc)
            return SafetyResult(safe=True, error=str(exc))

    def _parse_response(self, data: dict[str, Any], _text: str) -> SafetyResult:
        choices = data.get("choices", [])
        if not choices:
            return SafetyResult(safe=True, error="empty response")

        message = choices[0].get("message") or {}
        content = message.get("content", "")

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return SafetyResult(safe=True)

        if not isinstance(parsed, dict):
            return SafetyResult(safe=True)

        categories: dict[str, float] = {}
        if "category_scores" in parsed:
            categories = {
                str(k): float(v)
                for k, v in parsed["category_scores"].items()
                if isinstance(v, (int, float))
            }

        blocked: list[str] = []
        for cat_name, score in categories.items():
            cat_lower = cat_name.lower().replace(" ", "_").replace("-", "_")
            for reject in self.config.reject_categories:
                if reject.lower() in cat_lower and score > 0.5:
                    blocked.append(cat_name)

        return SafetyResult(
            safe=len(blocked) == 0,
            categories=categories,
            blocked_categories=blocked,
        )
