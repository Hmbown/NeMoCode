# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM Reranking provider — /v1/ranking."""

from __future__ import annotations

import logging

import httpx

from nemocode.providers import NIMProviderBase

logger = logging.getLogger(__name__)


class NIMRerankProvider(NIMProviderBase):
    """NVIDIA NIM reranking provider."""

    async def rerank(
        self, query: str, passages: list[str], top_n: int = 5
    ) -> list[tuple[int, float]]:
        """Rerank passages by relevance to query. Returns (index, score) pairs."""
        url = f"{self._base_url}/ranking"
        body = {
            "model": self.endpoint.model_id,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            "top_n": max(1, min(top_n, len(passages))),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        rankings = data.get("rankings", [])
        return [(r["index"], r.get("logit", r.get("score", 0.0))) for r in rankings]
