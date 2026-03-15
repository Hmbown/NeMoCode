# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Session persistence — save/load conversations."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from nemocode.core.sessions import Session
from nemocode.core.streaming import Message, Role, ToolCall

_SESSION_DIR = Path(
    os.environ.get("NEMOCODE_SESSION_DIR", "~/.local/share/nemocode/sessions")
).expanduser()


def _message_to_dict(msg: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
    if msg.thinking:
        d["thinking"] = msg.thinking
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls
        ]
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.is_error:
        d["is_error"] = True
    return d


def _dict_to_message(d: dict[str, Any]) -> Message:
    tool_calls = []
    for tc in d.get("tool_calls", []):
        tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
    return Message(
        role=Role(d["role"]),
        content=d.get("content", ""),
        thinking=d.get("thinking", ""),
        tool_calls=tool_calls,
        tool_call_id=d.get("tool_call_id"),
        is_error=d.get("is_error", False),
    )


def _safe_session_path(session_id: str) -> Path:
    """Resolve a session path safely, rejecting path traversal."""
    # Sanitize: only allow alphanumeric, hyphen, underscore
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe_id:
        raise ValueError(f"Invalid session ID: {session_id!r}")
    path = (_SESSION_DIR / f"{safe_id}.json").resolve()
    if not str(path).startswith(str(_SESSION_DIR.resolve())):
        raise ValueError(f"Invalid session ID: {session_id!r}")
    return path


def save_session(
    session: Session,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a session to disk.

    Args:
        session: The session to persist.
        metadata: Optional dict with extra fields (e.g. git_branch, cwd).
    """
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "id": session.id,
        "endpoint_name": session.endpoint_name,
        "messages": [_message_to_dict(m) for m in session.messages],
        "usage": session.usage.as_dict(),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": session.message_count(),
    }
    if metadata:
        data["metadata"] = metadata
    path = _safe_session_path(session.id)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_session(session_id: str) -> Session | None:
    """Load a session from disk."""
    path = _safe_session_path(session_id)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    session = Session(id=data["id"], endpoint_name=data.get("endpoint_name", ""))

    for msg_data in data.get("messages", []):
        msg = _dict_to_message(msg_data)
        session.messages.append(msg)

    usage = data.get("usage", {})
    session.usage.prompt_tokens = usage.get("prompt_tokens", 0)
    session.usage.completion_tokens = usage.get("completion_tokens", 0)
    session.usage.total_tokens = usage.get("total_tokens", 0)
    session.created_at = data.get("created_at", time.time())
    session.updated_at = data.get("updated_at", time.time())

    return session


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List saved sessions (most recent first)."""
    if not _SESSION_DIR.exists():
        return []

    sessions = []
    for path in sorted(_SESSION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[
        :limit
    ]:
        try:
            data = json.loads(path.read_text())
            sessions.append(
                {
                    "id": data.get("id", path.stem),
                    "endpoint_name": data.get("endpoint_name", ""),
                    "message_count": data.get("message_count", 0),
                    "usage": data.get("usage", {}),
                    "created_at": data.get("created_at", 0),
                    "updated_at": data.get("updated_at", 0),
                }
            )
        except Exception:
            continue

    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a saved session."""
    path = _safe_session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False
