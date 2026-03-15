# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Filesystem tools: read, write, edit, list directory.

Includes undo support: file-mutating operations snapshot the original
content before writing, enabling /undo to revert the last change.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from nemocode.tools import tool

# Project root is set at import time; tools should not write outside it
_PROJECT_ROOT: Path = Path.cwd().resolve()

# ---------------------------------------------------------------------------
# Undo stack — stores (path, old_content_or_None) for each file mutation
# ---------------------------------------------------------------------------
_MAX_UNDO = 20
_UNDO_STACK: list[tuple[str, str | None]] = []


def _push_undo(path: str, old_content: str | None) -> None:
    """Record a file state for later undo."""
    _UNDO_STACK.append((path, old_content))
    # Trim to prevent unbounded growth
    while len(_UNDO_STACK) > _MAX_UNDO:
        _UNDO_STACK.pop(0)


def undo_last() -> dict[str, str]:
    """Revert the last file mutation. Returns status dict."""
    if not _UNDO_STACK:
        return {"error": "Nothing to undo"}
    path, old_content = _UNDO_STACK.pop()
    try:
        if old_content is None:
            # File was newly created — remove it
            Path(path).unlink(missing_ok=True)
            return {"status": "ok", "action": "deleted", "path": path}
        else:
            Path(path).write_text(old_content)
            return {"status": "ok", "action": "restored", "path": path}
    except Exception as e:
        return {"error": f"Undo failed: {e}"}


def undo_stack_depth() -> int:
    """Return the number of undoable operations."""
    return len(_UNDO_STACK)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def set_project_root(root: Path) -> None:
    """Override the project root for path sandboxing."""
    global _PROJECT_ROOT
    _PROJECT_ROOT = root.resolve()


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to cwd, with safety checks."""
    p = Path(path).resolve()
    return p


def _is_within_project(p: Path) -> bool:
    """Check if a resolved path is within the project root."""
    try:
        p.relative_to(_PROJECT_ROOT)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------


def _make_diff(old_content: str, new_content: str, path: str) -> str:
    """Generate a unified diff string."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path)
    return "".join(diff)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(name="read_file", description="Read the contents of a file.", category="fs")
async def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read file contents.
    path: Path to the file to read.
    offset: Line number to start reading from (0 = beginning).
    limit: Maximum number of lines to read (0 = all).
    """
    p = _resolve_path(path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot read outside project directory: {p}"})
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not p.is_file():
        return json.dumps({"error": f"Not a file: {path}"})
    try:
        text = p.read_text(errors="replace")
        lines = text.splitlines(keepends=True)
        # Only add line numbers when using offset/limit (partial reads)
        if offset > 0 or limit > 0:
            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]
            numbered = []
            start = offset + 1
            for i, line in enumerate(lines, start=start):
                numbered.append(f"{i:>6}\t{line}")
            return "".join(numbered)
        return text
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="write_file",
    description="Write content to a file (creates or overwrites).",
    requires_confirmation=True,
    category="fs",
)
async def write_file(path: str, content: str) -> str:
    """Write content to a file.
    path: Path to write to.
    content: The content to write.
    """
    p = _resolve_path(path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot write outside project directory: {p}"})
    try:
        # Snapshot for undo + diff
        old_content = p.read_text() if p.exists() else None
        _push_undo(str(p), old_content)

        diff = _make_diff(old_content or "", content, str(p))

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return json.dumps({"status": "ok", "path": str(p), "bytes": len(content), "diff": diff})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(name="edit_file", description="Apply a search-and-replace edit to a file.", category="fs")
async def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in file.
    path: Path to the file.
    old_string: Exact string to find.
    new_string: Replacement string.
    """
    p = _resolve_path(path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot edit outside project directory: {p}"})
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not old_string:
        return json.dumps(
            {"error": "old_string must not be empty. Use write_file to create new content."}
        )
    try:
        text = p.read_text()
        if old_string not in text:
            return json.dumps({"error": "old_string not found in file"})
        count = text.count(old_string)
        if count > 1:
            return json.dumps(
                {"error": f"old_string found {count} times — must be unique. Provide more context."}
            )

        # Snapshot for undo
        _push_undo(str(p), text)

        new_text = text.replace(old_string, new_string, 1)
        diff = _make_diff(text, new_text, str(p))

        p.write_text(new_text)
        return json.dumps({"status": "ok", "path": str(p), "diff": diff})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(name="list_dir", description="List files and directories at a path.", category="fs")
async def list_dir(path: str = ".", max_depth: int = 1) -> str:
    """List directory contents.
    path: Directory path to list.
    max_depth: How many levels deep to recurse (1 = immediate children only).
    """
    p = _resolve_path(path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot list outside project directory: {p}"})
    if not p.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    if not p.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})

    entries: list[str] = []

    def _walk(dir_path: Path, depth: int, prefix: str = "") -> None:
        if depth > max_depth:
            return
        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        for item in items:
            if item.name.startswith(".") and depth > 1:
                continue
            marker = "/" if item.is_dir() else ""
            entries.append(f"{prefix}{item.name}{marker}")
            if item.is_dir() and depth < max_depth:
                _walk(item, depth + 1, prefix + "  ")

    _walk(p, 1)
    return "\n".join(entries) if entries else "(empty directory)"
