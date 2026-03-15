# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Session persistence — save/load conversations.

Delegates to SQLite storage by default (core.sqlite_store).
Falls back to JSON files if SQLite fails for any reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from nemocode.core.sessions import Session
from nemocode.core.streaming import Message, Role, ToolCall

# ---------------------------------------------------------------------------
# Backend selection: SQLite preferred, JSON fallback
# ---------------------------------------------------------------------------

_USE_SQLITE = True
try:
    from nemocode.core import sqlite_store as _sqlite
except Exception:
    _USE_SQLITE = False

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
) -> Path | str:
    """Save a session. Uses SQLite when available, falls back to JSON."""
    if _USE_SQLITE:
        try:
            return _sqlite.save_session(session, metadata)
        except Exception as e:
            logging.getLogger(__name__).debug("SQLite save failed, using JSON: %s", e)
    return _save_session_json(session, metadata)


def load_session(session_id: str) -> Session | None:
    """Load a session. Tries SQLite first, then JSON."""
    if _USE_SQLITE:
        try:
            result = _sqlite.load_session(session_id)
            if result is not None:
                return result
        except Exception:
            pass
    return _load_session_json(session_id)


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List sessions. Uses SQLite when available, falls back to JSON."""
    if _USE_SQLITE:
        try:
            return _sqlite.list_sessions(limit)
        except Exception:
            pass
    return _list_sessions_json(limit)


def delete_session(session_id: str) -> bool:
    """Delete a session from both stores."""
    deleted = False
    if _USE_SQLITE:
        try:
            deleted = _sqlite.delete_session(session_id)
        except Exception:
            pass
    # Also try JSON
    path = _safe_session_path(session_id)
    if path.exists():
        path.unlink()
        deleted = True
    return deleted


# ---------------------------------------------------------------------------
# JSON fallback implementations
# ---------------------------------------------------------------------------


def _save_session_json(
    session: Session,
    metadata: dict[str, Any] | None = None,
) -> Path:
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


def _load_session_json(session_id: str) -> Session | None:
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


def _list_sessions_json(limit: int = 50) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Session retry — retry API calls with exponential backoff
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


async def retry_with_backoff(
    fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_errors: tuple = (ConnectionError, TimeoutError, OSError),
    **kwargs,
):
    """Retry an async function with exponential backoff.

    Args:
        fn: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        retryable_errors: Tuple of exception types to retry on
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable_errors as e:
            last_error = e
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            _logger.debug(
                "Retry %d/%d for %s after %.1fs: %s",
                attempt + 1,
                max_retries,
                fn.__name__,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Improved revert — revert to any point in the undo stack
# ---------------------------------------------------------------------------


def revert_to_point(target_depth: int) -> list[dict[str, str]]:
    """Revert multiple file changes back to a specific point in the undo stack.

    Args:
        target_depth: The depth to revert to (0 = revert everything)

    Returns:
        List of result dicts for each reverted operation.
    """
    from nemocode.tools.fs import undo_last, undo_stack_depth

    results = []
    current = undo_stack_depth()
    while current > target_depth:
        result = undo_last()
        results.append(result)
        if "error" in result:
            break  # Stop on error
        current = undo_stack_depth()
    return results
