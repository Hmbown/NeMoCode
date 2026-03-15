# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Auto-memory system — cross-session memory persisted to disk.

Stores memories at ~/.config/nemocode/memory/ (global) and
.nemocode/memory/ (per-project). Memories are loaded into the
system prompt at session start.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GLOBAL_MEMORY_DIR = Path(
    os.environ.get("NEMOCODE_MEMORY_DIR", "~/.config/nemocode/memory")
).expanduser()

_MAX_MEMORY_TOKENS = 4000  # rough limit for system prompt injection


def _project_memory_dir() -> Path | None:
    """Find project-level memory directory (.nemocode/memory/)."""
    cwd = Path.cwd()
    # Walk up to find .nemocode/ directory
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".nemocode" / "memory"
        if candidate.exists():
            return candidate
        # Stop at home directory
        if parent == Path.home():
            break
    return None


def _ensure_dirs() -> None:
    _GLOBAL_MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def save_memory(
    key: str,
    content: str,
    *,
    scope: str = "global",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Save a memory entry.

    key: Unique identifier (e.g. 'user_prefers_pytest')
    content: The memory content
    scope: 'global' or 'project'
    tags: Optional categorization tags
    """
    _ensure_dirs()
    if scope == "project":
        mem_dir = _project_memory_dir()
        if mem_dir is None:
            mem_dir = Path.cwd() / ".nemocode" / "memory"
            mem_dir.mkdir(parents=True, exist_ok=True)
    else:
        mem_dir = _GLOBAL_MEMORY_DIR

    # Sanitize key for filename
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    path = mem_dir / f"{safe_key}.json"

    entry = {
        "key": key,
        "content": content,
        "tags": tags or [],
        "created_at": time.time(),
        "updated_at": time.time(),
        "scope": scope,
    }

    # Update existing entry timestamp if it exists
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            entry["created_at"] = existing.get("created_at", entry["created_at"])
        except Exception:
            pass

    path.write_text(json.dumps(entry, indent=2))
    return {"status": "ok", "key": key, "path": str(path)}


def forget_memory(key: str, *, scope: str = "global") -> dict[str, Any]:
    """Delete a memory entry."""
    if scope == "project":
        mem_dir = _project_memory_dir() or (Path.cwd() / ".nemocode" / "memory")
    else:
        mem_dir = _GLOBAL_MEMORY_DIR

    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    path = mem_dir / f"{safe_key}.json"

    if path.exists():
        path.unlink()
        return {"status": "ok", "key": key, "action": "deleted"}
    return {"status": "ok", "key": key, "action": "not_found"}


def load_all_memories() -> str:
    """Load all memories and format for system prompt injection."""
    memories: list[dict[str, Any]] = []

    # Load global memories
    if _GLOBAL_MEMORY_DIR.exists():
        for path in sorted(_GLOBAL_MEMORY_DIR.glob("*.json")):
            try:
                entry = json.loads(path.read_text())
                memories.append(entry)
            except Exception:
                continue

    # Load project memories
    proj_dir = _project_memory_dir()
    if proj_dir and proj_dir.exists():
        for path in sorted(proj_dir.glob("*.json")):
            try:
                entry = json.loads(path.read_text())
                entry["scope"] = "project"
                memories.append(entry)
            except Exception:
                continue

    if not memories:
        return ""

    # Format for injection
    parts = ["## Memories"]
    total_len = 0
    for mem in memories:
        line = f"- [{mem.get('scope', 'global')}] {mem['key']}: {mem['content']}"
        if total_len + len(line) > _MAX_MEMORY_TOKENS * 4:  # rough char estimate
            parts.append("- ... (additional memories truncated)")
            break
        parts.append(line)
        total_len += len(line)

    return "\n".join(parts)


def list_memories(scope: str = "all") -> list[dict[str, Any]]:
    """List all memory entries."""
    memories: list[dict[str, Any]] = []

    if scope in ("all", "global") and _GLOBAL_MEMORY_DIR.exists():
        for path in sorted(_GLOBAL_MEMORY_DIR.glob("*.json")):
            try:
                memories.append(json.loads(path.read_text()))
            except Exception:
                continue

    if scope in ("all", "project"):
        proj_dir = _project_memory_dir()
        if proj_dir and proj_dir.exists():
            for path in sorted(proj_dir.glob("*.json")):
                try:
                    entry = json.loads(path.read_text())
                    entry["scope"] = "project"
                    memories.append(entry)
                except Exception:
                    continue

    return memories
