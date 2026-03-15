# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Document parsing tool using Nemotron Parse VLM endpoint."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

from nemocode.tools import tool

logger = logging.getLogger(__name__)

_PARSE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf", ".tiff", ".tif"}


def _is_parseable(path: str) -> bool:
    """Check if a file extension is supported for parsing."""
    return Path(path).suffix.lower() in _PARSE_EXTENSIONS


@tool(
    description=(
        "Parse an image or PDF using Nemotron Parse VLM. "
        "Extracts text, tables, and structured content."
    ),
    category="parse",
)
async def parse_document(
    path: str,
    prompt: str = "Extract all text content from this document.",
) -> str:
    """Parse an image or PDF using Nemotron Parse.

    path: Path to the image or PDF file
    prompt: Instruction for the parser (default: extract all text)
    """
    p = Path(path).resolve()
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})

    if not _is_parseable(path):
        return json.dumps(
            {
                "error": (
                    f"Unsupported file type: {p.suffix}. "
                    f"Supported: {', '.join(sorted(_PARSE_EXTENSIONS))}"
                )
            }
        )

    # Read and encode the file
    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
    except Exception as e:
        return json.dumps({"error": f"Failed to read file: {e}"})

    # Determine MIME type
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    mime = mime_map.get(p.suffix.lower(), "application/octet-stream")

    # Call Nemotron Parse via OpenAI-compatible API
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "NVIDIA_API_KEY not set — required for Nemotron Parse"})

    try:
        import urllib.request

        payload = {
            "model": "nvidia/nemotron-parse",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        req = urllib.request.Request(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return json.dumps(
            {
                "status": "ok",
                "path": str(p),
                "content": content,
                "tokens": result.get("usage", {}),
            }
        )
    except Exception as e:
        logger.debug("Nemotron Parse failed: %s", e)
        return json.dumps({"error": f"Parse failed: {e}"})
