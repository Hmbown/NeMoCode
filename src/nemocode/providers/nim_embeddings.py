# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM Embeddings provider — /v1/embeddings."""

from __future__ import annotations

import logging

import httpx

from nemocode.providers import NIMProviderBase

logger = logging.getLogger(__name__)


class NIMEmbeddingProvider(NIMProviderBase):
    """OpenAI-compatible embedding provider for NVIDIA NIM endpoints."""

    async def embed(
        self, texts: list[str], input_type: str = "query"
    ) -> list[list[float]]:
        """Embed a list of texts, returning vectors.

        Args:
            texts: Strings to embed.
            input_type: "query" for search queries, "passage" for document indexing.
        """
        url = f"{self._base_url}/embeddings"
        body = {
            "model": self.endpoint.model_id,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        embeddings = []
        for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
            embeddings.append(item["embedding"])
        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        result = await self.embed([text])
        return result[0] if result else []
