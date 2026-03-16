# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""HTTP fetch tool."""

from __future__ import annotations

import json

import httpx

from nemocode.tools import tool

_MAX_BODY = 50_000
# Default timeout for HTTP requests (seconds).  Kept shorter than bash_exec
# (120s) since network calls should fail fast rather than block the agent.
_HTTP_TIMEOUT = 30


@tool(name="http_fetch", description="Fetch content from a URL.", category="http")
async def http_fetch(
    url: str, method: str = "GET", headers: str = "", timeout: int = _HTTP_TIMEOUT
) -> str:
    """Fetch a URL.
    url: The URL to fetch.
    method: HTTP method (GET, POST, etc.).
    headers: JSON-encoded headers dict.
    timeout: Request timeout in seconds (default 30).
    """
    hdrs = {}
    if headers:
        try:
            hdrs = json.loads(headers)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid headers JSON"})

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.request(method, url, headers=hdrs)
            body = resp.text
            if len(body) > _MAX_BODY:
                body = body[:_MAX_BODY] + f"\n... (truncated, {len(body)} total chars)"
            return json.dumps(
                {
                    "status": resp.status_code,
                    "headers": dict(resp.headers),
                    "body": body,
                }
            )
    except Exception as e:
        return json.dumps({"error": str(e)})
