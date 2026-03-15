# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""LSP tools — expose language server diagnostics, hover, and references."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nemocode.core.lsp import detect_language, get_client
from nemocode.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> str:
    """Best-effort project root (cwd)."""
    return str(Path.cwd().resolve())


async def _client_for(path: str) -> "LSPClient":  # noqa: F821
    lang = detect_language(path)
    if lang is None:
        raise ValueError(f"Cannot determine language for {path}")

    return await get_client(_project_root(), lang)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="lsp_diagnostics",
    description=("Get diagnostics (errors, warnings) for a file from the language server."),
    category="lsp",
)
async def lsp_diagnostics(path: str) -> str:
    """Get language server diagnostics for a file.

    path: Path to the file to check.
    """
    try:
        client = await _client_for(path)
        diags = await client.get_diagnostics(path)
        items = [
            {
                "severity": d.severity_label,
                "message": d.message,
                "line": d.line + 1,  # 1-indexed for humans
                "col": d.col + 1,
                "source": d.source,
                "code": d.code,
            }
            for d in diags
        ]
        return json.dumps({"path": path, "diagnostics": items, "count": len(items)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="lsp_hover",
    description="Get hover information (type, docs) for a symbol at a position.",
    category="lsp",
)
async def lsp_hover(path: str, line: int, column: int) -> str:
    """Get hover information at a position.

    path: Path to the file.
    line: 1-indexed line number.
    column: 1-indexed column number.
    """
    try:
        client = await _client_for(path)
        text = await client.get_hover(path, line - 1, column - 1)
        if not text:
            return json.dumps({"path": path, "hover": None, "message": "No hover info"})
        return json.dumps({"path": path, "line": line, "column": column, "hover": text})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="lsp_references",
    description="Find all references to the symbol at a given position.",
    category="lsp",
)
async def lsp_references(path: str, line: int, column: int) -> str:
    """Find references to the symbol at a position.

    path: Path to the file.
    line: 1-indexed line number.
    column: 1-indexed column number.
    """
    try:
        client = await _client_for(path)
        refs = await client.get_references(path, line - 1, column - 1)
        items = [
            {
                "path": r.path,
                "line": r.line + 1,
                "col": r.col + 1,
            }
            for r in refs
        ]
        return json.dumps(
            {"path": path, "line": line, "column": column, "references": items, "count": len(items)}
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
