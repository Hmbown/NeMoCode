# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""In-session task tracking tools."""

from __future__ import annotations

import json
import time
from typing import Any

from nemocode.tools import tool

# Session-scoped task store (not persisted)
_tasks: list[dict[str, Any]] = []
_next_id: int = 1


def _reset_tasks() -> None:
    """Reset task store (for testing)."""
    global _next_id
    _tasks.clear()
    _next_id = 1


@tool(
    description="Create a task to track work in this session.",
    category="tasks",
)
async def create_task(
    title: str,
    description: str = "",
    priority: str = "medium",
) -> str:
    """Create a new task.

    title: Short task description
    description: Detailed description
    priority: 'high', 'medium', or 'low'
    """
    global _next_id
    task = {
        "id": _next_id,
        "title": title,
        "description": description,
        "priority": priority,
        "status": "pending",
        "created_at": time.time(),
    }
    _tasks.append(task)
    _next_id += 1
    return json.dumps({"status": "ok", "task": task})


@tool(
    description="Update a task's status or details.",
    category="tasks",
)
async def update_task(
    task_id: int,
    status: str = "",
    title: str = "",
) -> str:
    """Update an existing task.

    task_id: The task ID to update
    status: New status: 'pending', 'in_progress', 'done', 'blocked'
    title: New title (optional)
    """
    for task in _tasks:
        if task["id"] == task_id:
            if status:
                task["status"] = status
            if title:
                task["title"] = title
            return json.dumps({"status": "ok", "task": task})
    return json.dumps({"error": f"Task {task_id} not found"})


@tool(
    description="List all tasks in this session.",
    category="tasks",
)
async def list_tasks(status_filter: str = "") -> str:
    """List all session tasks.

    status_filter: Filter by status (empty for all)
    """
    filtered = _tasks
    if status_filter:
        filtered = [t for t in _tasks if t["status"] == status_filter]
    return json.dumps({"tasks": filtered, "count": len(filtered)})
