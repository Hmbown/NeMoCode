# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM Document Parse provider — VLM-based document extraction."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from nemocode.providers import NIMProviderBase

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE_MB = 20


class NIMParseProvider(NIMProviderBase):
    """Document extraction via Nemotron Parse VLM."""

    async def parse_file(
        self, file_path: str, prompt: str = "Extract all text from this document."
    ) -> str:
        """Parse a document file and extract its contents."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > _MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large ({size_mb:.1f}MB). Maximum is {_MAX_FILE_SIZE_MB}MB.")

        content_b64 = base64.b64encode(p.read_bytes()).decode()
        suffix = p.suffix.lower()
        mime = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(suffix, "application/octet-stream")

        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self.endpoint.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{content_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        return choices[0].get("message", {}).get("content", "") if choices else ""
