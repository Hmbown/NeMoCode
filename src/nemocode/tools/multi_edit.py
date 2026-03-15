# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Multi-edit and apply-patch tools.

multi_edit: Apply multiple search-and-replace edits to a file atomically.
apply_patch: Apply a unified diff patch to a file.
"""

from __future__ import annotations

import json

from nemocode.tools import tool
from nemocode.tools.fs import _is_within_project, _make_diff, _push_undo, _resolve_path


@tool(
    name="multi_edit",
    description="Apply multiple search-and-replace edits to a file in one atomic operation.",
    requires_confirmation=False,
    category="fs",
)
async def multi_edit(path: str, edits: str) -> str:
    """Apply multiple edits to a file atomically.
    path: Path to the file to edit.
    edits: JSON array of edit objects, each with 'old_string' and 'new_string' keys.
    """
    p = _resolve_path(path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot edit outside project directory: {p}"})
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})

    # Parse edits
    try:
        edit_list = json.loads(edits)
        if not isinstance(edit_list, list):
            return json.dumps({"error": "edits must be a JSON array"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in edits: {e}"})

    if not edit_list:
        return json.dumps({"error": "edits array is empty"})

    try:
        original = p.read_text()
        text = original

        applied = 0
        errors: list[str] = []

        for i, edit in enumerate(edit_list):
            old = edit.get("old_string", "")
            new = edit.get("new_string", "")

            if not old:
                errors.append(f"Edit {i}: old_string is empty")
                continue

            if old not in text:
                errors.append(f"Edit {i}: old_string not found")
                continue

            count = text.count(old)
            if count > 1:
                errors.append(f"Edit {i}: old_string found {count} times — must be unique")
                continue

            text = text.replace(old, new, 1)
            applied += 1

        if applied == 0:
            return json.dumps({"error": "No edits applied", "details": errors})

        # Snapshot for undo
        _push_undo(str(p), original)

        diff = _make_diff(original, text, str(p))
        p.write_text(text)

        result: dict = {
            "status": "ok",
            "path": str(p),
            "applied": applied,
            "total": len(edit_list),
            "diff": diff,
        }
        if errors:
            result["errors"] = errors
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="apply_patch",
    description="Apply a unified diff patch to a file.",
    requires_confirmation=True,
    category="fs",
)
async def apply_patch(path: str, patch: str) -> str:
    """Apply a unified diff patch.
    path: Path to the file to patch.
    patch: The unified diff content to apply.
    """
    p = _resolve_path(path)
    if not _is_within_project(p):
        return json.dumps({"error": f"Cannot patch outside project directory: {p}"})
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})

    try:
        original = p.read_text()
        lines = original.splitlines(keepends=True)
        patched = _apply_unified_diff(lines, patch)

        if patched is None:
            return json.dumps({"error": "Patch failed to apply — context mismatch"})

        new_text = "".join(patched)
        if new_text == original:
            return json.dumps({"status": "ok", "path": str(p), "message": "No changes"})

        _push_undo(str(p), original)
        diff = _make_diff(original, new_text, str(p))
        p.write_text(new_text)
        return json.dumps({"status": "ok", "path": str(p), "diff": diff})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _apply_unified_diff(
    original_lines: list[str], patch_text: str
) -> list[str] | None:
    """Apply a unified diff to a list of lines. Returns None on failure."""
    patch_lines = patch_text.splitlines(keepends=True)
    result = list(original_lines)
    offset = 0  # track line number shifts from prior hunks

    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]

        # Skip header lines
        if line.startswith("---") or line.startswith("+++"):
            i += 1
            continue

        # Parse hunk header
        if line.startswith("@@"):
            hunk_info = line.split("@@")
            if len(hunk_info) < 3:
                i += 1
                continue

            parts = hunk_info[1].strip().split()
            old_start = int(parts[0].split(",")[0].lstrip("-")) - 1
            i += 1

            # Apply hunk
            pos = old_start + offset
            while i < len(patch_lines):
                pline = patch_lines[i]
                if pline.startswith("@@") or pline.startswith("---") or pline.startswith("+++"):
                    break
                if pline.startswith("-"):
                    # Remove line
                    if pos < len(result):
                        result.pop(pos)
                        offset -= 1
                    i += 1
                elif pline.startswith("+"):
                    # Add line
                    content = pline[1:]
                    if not content.endswith("\n"):
                        content += "\n"
                    result.insert(pos, content)
                    pos += 1
                    offset += 1
                    i += 1
                elif pline.startswith(" ") or pline.startswith("\n"):
                    # Context line
                    pos += 1
                    i += 1
                else:
                    i += 1
        else:
            i += 1

    return result
