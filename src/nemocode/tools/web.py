# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Web search and readable page-fetch tools."""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from nemocode.tools import tool

logger = logging.getLogger(__name__)

_DDG_URL = "https://lite.duckduckgo.com/lite/"
_USER_AGENT = "NeMoCode/0.2.0"
_MAX_FETCH_CHARS = 12_000
_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_SKIP_TAGS = {"script", "style", "svg", "noscript", "template"}


class _ReadableHTMLParser(HTMLParser):
    """Best-effort readable text extraction from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._all_parts: list[str] = []
        self._main_parts: list[str] = []
        self._skip_depth = 0
        self._focus_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"main", "article"}:
            self._focus_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in {"main", "article"} and self._focus_depth:
            self._focus_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = html.unescape(data)
        if self._in_title:
            self.title += text
        self._append(text)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._append(html.unescape(f"&#{name};"))

    def extracted_text(self) -> str:
        focused = _normalize_extracted_text("".join(self._main_parts))
        if focused:
            return focused
        return _normalize_extracted_text("".join(self._all_parts))

    def _append(self, chunk: str) -> None:
        if not chunk:
            return
        self._all_parts.append(chunk)
        if self._focus_depth:
            self._main_parts.append(chunk)


def _normalize_extracted_text(text: str) -> str:
    """Collapse noisy whitespace while keeping paragraph breaks."""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized:
            cleaned_lines.append(normalized)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Cap tool output size without cutting in the middle of whitespace when possible."""
    if len(text) <= max_chars:
        return text, False
    head = text[:max_chars].rstrip()
    cut = head.rfind("\n")
    if cut >= max_chars // 2:
        head = head[:cut].rstrip()
    return head, True


def _decode_response_body(resp, body: bytes) -> str:
    charset = None
    try:
        charset = resp.headers.get_content_charset()
    except Exception:
        charset = None
    for encoding in (charset, "utf-8", "latin-1"):
        if not encoding:
            continue
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _extract_readable_text(raw_html: str) -> tuple[str, str]:
    parser = _ReadableHTMLParser()
    parser.feed(raw_html)
    parser.close()
    title = _normalize_extracted_text(parser.title)
    content = parser.extracted_text()
    return title, content


@tool(
    description=(
        "Fetch a webpage and extract readable content for documentation/reference pages."
    ),
    category="web",
)
async def web_fetch(url: str, max_chars: int = _MAX_FETCH_CHARS) -> str:
    """Fetch a URL and return readable extracted content.

    url: URL to fetch.
    max_chars: Maximum characters of extracted content to return (default 12000).
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            resolved_url = resp.geturl()
            status = getattr(resp, "status", 200)

        text = _decode_response_body(resp, body)

        if "html" in content_type.lower() or "<html" in text[:512].lower():
            title, content = _extract_readable_text(text)
        else:
            title = ""
            content = _normalize_extracted_text(text)

        if not content:
            return json.dumps({"error": f"No readable content extracted from {url}"})

        content, truncated = _truncate_text(content, max(500, max_chars))
        return json.dumps(
            {
                "status": status,
                "url": resolved_url,
                "title": title,
                "content_type": content_type,
                "content": content,
                "truncated": truncated,
            }
        )
    except Exception as e:
        logger.debug("Web fetch failed: %s", e)
        return json.dumps({"error": f"Fetch failed: {e}"})


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
