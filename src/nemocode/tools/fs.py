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

from nemocode.core.sandbox import Sandbox
from nemocode.core.secrets import scanner
from nemocode.core.tool_cache import get_tool_cache
from nemocode.tools import tool

# ---------------------------------------------------------------------------
# Sandbox — configurable security layer
# ---------------------------------------------------------------------------
_sandbox = Sandbox.from_env()

# Backwards compatibility: legacy functions delegate to the sandbox
_PROJECT_ROOT: Path = _sandbox._project_root

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
    _sandbox._project_root = root.resolve()


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to cwd, enforcing project sandbox."""
    return _sandbox.validate_path(path)


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

    # Check cache for full reads (no offset/limit)
    cache = get_tool_cache()
    if offset == 0 and limit == 0:
        stat = p.stat()
        cache_key = f"read:{p}:{stat.st_mtime}:{stat.st_size}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        text = p.read_text(errors="replace")
        # Guard against reading extremely large files
        if len(text) > 10_000_000:  # 10MB hard limit
            return json.dumps(
                {
                    "error": (
                        f"File too large to read ({len(text):,} chars). "
                        f"Use offset/limit parameters."
                    )
                }
            )
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
            result = scanner.scan_and_redact("".join(numbered))
        else:
            result = scanner.scan_and_redact(text)
            # Cache full reads
            stat = p.stat()
            cache_key = f"read:{p}:{stat.st_mtime}:{stat.st_size}"
            cache.put(cache_key, result, ttl=30.0)
        return result
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
    if not _sandbox.can_write():
        return json.dumps({"error": f"Write blocked by sandbox policy: {path}"})
    p = _resolve_path(path)
    try:
        # Snapshot for undo + diff
        old_content = p.read_text() if p.exists() else None
        _push_undo(str(p), old_content)

        diff = _make_diff(old_content or "", content, str(p))

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        # Invalidate cache for this path
        cache = get_tool_cache()
        cache.invalidate_prefix(f"read:{p}:")
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
    if not _sandbox.can_write():
        return json.dumps({"error": f"Edit blocked by sandbox policy: {path}"})
    p = _resolve_path(path)
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
        # Invalidate cache for this path
        cache = get_tool_cache()
        cache.invalidate_prefix(f"read:{p}:")
        return json.dumps({"status": "ok", "path": str(p), "diff": diff})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(name="list_dir", description="List files and directories at a path.", category="fs")
async def list_dir(path: str = ".", max_depth: int = 1) -> str:
    """List directory contents.
    path: Directory path to list.
    max_depth: How many levels deep to recurse (1 = immediate children only, max 5).
    """
    max_depth = min(max_depth, 5)  # Hard cap to prevent expensive recursion
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
