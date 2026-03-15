# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Memory tools — save and recall cross-session memories."""

from __future__ import annotations

import json

from nemocode.core.memory import forget_memory, list_memories, save_memory
from nemocode.tools import tool


@tool(
    description=(
        "Save a memory for future sessions. Use for user "
        "preferences, project patterns, or important context."
    ),
    category="memory",
)
async def save_memory_tool(
    key: str,
    content: str,
    scope: str = "global",
    tags: str = "",
) -> str:
    """Save a memory that persists across sessions.

    key: Unique identifier (e.g. 'user_prefers_pytest', 'project_uses_ruff')
    content: The memory content to save
    scope: 'global' (all projects) or 'project' (this project only)
    tags: Comma-separated tags for categorization
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    result = save_memory(key, content, scope=scope, tags=tag_list)
    return json.dumps(result)


@tool(
    description="Delete a previously saved memory.",
    category="memory",
)
async def forget_memory_tool(key: str, scope: str = "global") -> str:
    """Delete a memory entry.

    key: The memory key to delete
    scope: 'global' or 'project'
    """
    result = forget_memory(key, scope=scope)
    return json.dumps(result)


@tool(
    description="List all saved memories.",
    category="memory",
)
async def list_memories_tool(scope: str = "all") -> str:
    """List all saved memories.

    scope: 'all', 'global', or 'project'
    """
    memories = list_memories(scope=scope)
    if not memories:
        return json.dumps({"memories": [], "count": 0})
    return json.dumps({"memories": memories, "count": len(memories)})
