# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""SQLite-backed session storage.

Replaces the JSON-file persistence layer with a single SQLite database
at ``~/.local/share/nemocode/sessions.db``.  Provides the same public API
as :mod:`nemocode.core.persistence` plus full-text search.

Thread-safe: each call acquires its own connection from a module-level
path; the ``sqlite3`` module handles its own internal locking.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from nemocode.core.sessions import Session, TokenUsage
from nemocode.core.streaming import Message, Role, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------

_DB_DIR = Path(os.environ.get("NEMOCODE_DATA_DIR", "~/.local/share/nemocode")).expanduser()
_DB_PATH = _DB_DIR / "sessions.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    endpoint_name    TEXT NOT NULL DEFAULT '',
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    message_count    INTEGER NOT NULL DEFAULT 0,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens     INTEGER NOT NULL DEFAULT 0,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    thinking        TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    tool_call_id    TEXT,
    is_error        INTEGER NOT NULL DEFAULT 0,
    position        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, position);
"""


def _override_db_path(path: Path) -> None:
    """Override the database path (used by tests)."""
    global _DB_PATH, _DB_DIR
    _DB_PATH = path
    _DB_DIR = path.parent


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    """Return a new connection with WAL mode and foreign keys enabled."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


def _get_conn() -> sqlite3.Connection:
    """Get a ready-to-use connection (schema guaranteed)."""
    conn = _connect()
    _ensure_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Message serialisation helpers
# ---------------------------------------------------------------------------


def _message_to_row(msg: Message, session_id: str, position: int) -> tuple:
    tc_json = json.dumps(
        [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
    )
    return (
        session_id,
        msg.role.value,
        msg.content,
        msg.thinking,
        tc_json,
        msg.tool_call_id,
        1 if msg.is_error else 0,
        position,
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    tool_calls: list[ToolCall] = []
    for tc in json.loads(row["tool_calls_json"]):
        tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
    return Message(
        role=Role(row["role"]),
        content=row["content"],
        thinking=row["thinking"],
        tool_calls=tool_calls,
        tool_call_id=row["tool_call_id"],
        is_error=bool(row["is_error"]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_session(session: Session, metadata: dict[str, Any] | None = None) -> str:
    """Persist a session (upsert) and return its id."""
    conn = _get_conn()
    try:
        meta_json = json.dumps(metadata or {}, default=str)
        conn.execute(
            """
            INSERT INTO sessions
                (id, endpoint_name, created_at, updated_at,
                 message_count, prompt_tokens, completion_tokens,
                 total_tokens, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                endpoint_name    = excluded.endpoint_name,
                updated_at       = excluded.updated_at,
                message_count    = excluded.message_count,
                prompt_tokens    = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                total_tokens     = excluded.total_tokens,
                metadata_json    = excluded.metadata_json
            """,
            (
                session.id,
                session.endpoint_name,
                session.created_at,
                session.updated_at,
                session.message_count(),
                session.usage.prompt_tokens,
                session.usage.completion_tokens,
                session.usage.total_tokens,
                meta_json,
            ),
        )
        # Replace all messages for this session
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))
        rows = [_message_to_row(msg, session.id, idx) for idx, msg in enumerate(session.messages)]
        conn.executemany(
            """
            INSERT INTO messages
                (session_id, role, content, thinking,
                 tool_calls_json, tool_call_id, is_error, position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return session.id


def load_session(session_id: str) -> Session | None:
    """Load a session by id, or return None if it does not exist."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None

        session = Session(id=row["id"], endpoint_name=row["endpoint_name"])
        session.created_at = row["created_at"]
        session.updated_at = row["updated_at"]
        session.usage = TokenUsage(
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
        )

        msg_rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        for mr in msg_rows:
            session.messages.append(_row_to_message(mr))

        return session
    finally:
        conn.close()


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List sessions ordered by most recently updated, returning summary dicts."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "endpoint_name": r["endpoint_name"],
                    "message_count": r["message_count"],
                    "usage": {
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "total_tokens": r["total_tokens"],
                    },
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "metadata": json.loads(r["metadata_json"]),
                }
            )
        return results
    finally:
        conn.close()


def delete_session(session_id: str) -> bool:
    """Delete a session and its messages. Returns True if a row was removed."""
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def search_sessions(query: str) -> list[dict[str, Any]]:
    """Search sessions by message content (case-insensitive LIKE).

    Returns session summaries whose messages contain *query*.
    """
    conn = _get_conn()
    try:
        like_pattern = f"%{query}%"
        rows = conn.execute(
            """
            SELECT DISTINCT s.*
            FROM sessions s
            JOIN messages m ON m.session_id = s.id
            WHERE m.content LIKE ?
            ORDER BY s.updated_at DESC
            """,
            (like_pattern,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "endpoint_name": r["endpoint_name"],
                    "message_count": r["message_count"],
                    "usage": {
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "total_tokens": r["total_tokens"],
                    },
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "metadata": json.loads(r["metadata_json"]),
                }
            )
        return results
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migration: import existing JSON sessions
# ---------------------------------------------------------------------------


def migrate_json_sessions(json_dir: Path | None = None) -> int:
    """Import JSON session files into SQLite.

    Returns the number of sessions migrated.  Skips sessions that already
    exist in the database.
    """
    from nemocode.core.persistence import _SESSION_DIR as _DEFAULT_JSON_DIR

    src = json_dir or _DEFAULT_JSON_DIR
    if not src.exists():
        return 0

    count = 0
    conn = _get_conn()
    try:
        for path in src.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sid = data.get("id", path.stem)

                # Skip if already migrated
                exists = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone()
                if exists:
                    continue

                # Reconstruct a Session object
                session = Session(id=sid, endpoint_name=data.get("endpoint_name", ""))
                session.created_at = data.get("created_at", time.time())
                session.updated_at = data.get("updated_at", time.time())
                usage = data.get("usage", {})
                session.usage = TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
                for md in data.get("messages", []):
                    tool_calls: list[ToolCall] = []
                    for tc in md.get("tool_calls", []):
                        tool_calls.append(
                            ToolCall(
                                id=tc["id"],
                                name=tc["name"],
                                arguments=tc["arguments"],
                            )
                        )
                    session.messages.append(
                        Message(
                            role=Role(md["role"]),
                            content=md.get("content", ""),
                            thinking=md.get("thinking", ""),
                            tool_calls=tool_calls,
                            tool_call_id=md.get("tool_call_id"),
                            is_error=md.get("is_error", False),
                        )
                    )

                save_session(session, metadata=data.get("metadata"))
                count += 1
            except Exception:
                logger.debug("Failed to migrate %s", path, exc_info=True)
                continue
    finally:
        conn.close()

    logger.info("Migrated %d JSON sessions to SQLite", count)
    return count
