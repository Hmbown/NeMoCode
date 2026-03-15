# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Web search tool using DuckDuckGo Lite (no API key required)."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from nemocode.tools import tool

logger = logging.getLogger(__name__)

_DDG_URL = "https://lite.duckduckgo.com/lite/"
_USER_AGENT = "NeMoCode/0.2.0"


@tool(
    description=(
        "Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets."
    ),
    category="web",
)
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web.

    query: Search query string
    max_results: Maximum number of results to return (default 5)
    """
    try:
        # Use DuckDuckGo HTML API
        params = urllib.parse.urlencode({"q": query, "kl": ""})
        req = urllib.request.Request(
            f"{_DDG_URL}?{params}",
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        results = _parse_ddg_lite(html, max_results)
        if not results:
            return json.dumps({"results": [], "query": query, "message": "No results found"})

        return json.dumps({"results": results, "query": query, "count": len(results)})
    except Exception as e:
        logger.debug("Web search failed: %s", e)
        return json.dumps({"error": f"Search failed: {e}"})


def _parse_ddg_lite(html: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo Lite HTML response for result links and snippets."""
    results = []
    # DuckDuckGo Lite uses a simple table layout
    # Find result links: <a rel="nofollow" href="..." class="result-link">
    import re

    # Match result links
    link_pattern = re.compile(
        r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    # Fallback: match any result link pattern
    if not link_pattern.search(html):
        link_pattern = re.compile(
            r'<a[^>]*rel="nofollow"[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

    # Match snippets (td class="result-snippet")
    snippet_pattern = re.compile(
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (url, title) in enumerate(links[:max_results]):
        # Clean HTML tags from title and snippet
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        # Skip DuckDuckGo internal links
        if "duckduckgo.com" in url:
            continue

        results.append(
            {
                "title": clean_title,
                "url": url,
                "snippet": snippet,
            }
        )

    return results[:max_results]
