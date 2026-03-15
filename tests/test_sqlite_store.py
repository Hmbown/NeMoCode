# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for SQLite session storage."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from nemocode.core.sessions import Session, TokenUsage
from nemocode.core.streaming import Message, Role, ToolCall

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path: Path):
    """Point the sqlite_store at a temp DB for every test."""
    from nemocode.core import sqlite_store

    sqlite_store._override_db_path(tmp_path / "test_sessions.db")
    yield


def _make_session(
    sid: str = "test-1",
    endpoint: str = "ep",
    n_messages: int = 3,
) -> Session:
    s = Session(id=sid, endpoint_name=endpoint)
    s.add_system("You are helpful.")
    s.add_user("Hello")
    s.add_assistant(Message(role=Role.ASSISTANT, content="Hi there!"))
    # Add extra messages if requested
    for i in range(n_messages - 3):
        s.add_user(f"Follow-up {i}")
    s.usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return s


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_returns_id(self):
        from nemocode.core.sqlite_store import save_session

        s = _make_session()
        result = save_session(s)
        assert result == "test-1"

    def test_load_round_trip(self):
        from nemocode.core.sqlite_store import load_session, save_session

        s = _make_session()
        save_session(s)
        loaded = load_session("test-1")

        assert loaded is not None
        assert loaded.id == "test-1"
        assert loaded.endpoint_name == "ep"
        assert loaded.message_count() == 3
        assert loaded.usage.prompt_tokens == 100
        assert loaded.usage.completion_tokens == 50
        assert loaded.usage.total_tokens == 150

    def test_load_preserves_message_order(self):
        from nemocode.core.sqlite_store import load_session, save_session

        s = _make_session()
        save_session(s)
        loaded = load_session("test-1")

        assert loaded is not None
        assert loaded.messages[0].role == Role.SYSTEM
        assert loaded.messages[1].role == Role.USER
        assert loaded.messages[2].role == Role.ASSISTANT

    def test_load_preserves_tool_calls(self):
        from nemocode.core.sqlite_store import load_session, save_session

        s = Session(id="tc-1", endpoint_name="ep")
        s.add_user("Do something")
        s.add_assistant(
            Message(
                role=Role.ASSISTANT,
                content="Calling tool...",
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "/foo.py"}),
                ],
            )
        )
        s.add_tool_result("call-1", '{"content": "..."}')
        save_session(s)

        loaded = load_session("tc-1")
        assert loaded is not None
        assistant_msg = loaded.messages[1]
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0].name == "read_file"
        assert assistant_msg.tool_calls[0].arguments == {"path": "/foo.py"}

        tool_msg = loaded.messages[2]
        assert tool_msg.role == Role.TOOL
        assert tool_msg.tool_call_id == "call-1"

    def test_load_preserves_thinking(self):
        from nemocode.core.sqlite_store import load_session, save_session

        s = Session(id="think-1")
        s.add_assistant(Message(role=Role.ASSISTANT, content="Answer", thinking="Let me think..."))
        save_session(s)

        loaded = load_session("think-1")
        assert loaded is not None
        assert loaded.messages[0].thinking == "Let me think..."

    def test_load_nonexistent_returns_none(self):
        from nemocode.core.sqlite_store import load_session

        assert load_session("does-not-exist") is None

    def test_save_upsert(self):
        from nemocode.core.sqlite_store import load_session, save_session

        s = _make_session()
        save_session(s)

        s.add_user("More input")
        save_session(s)

        loaded = load_session("test-1")
        assert loaded is not None
        assert loaded.message_count() == 4

    def test_save_with_metadata(self):
        from nemocode.core.sqlite_store import list_sessions, save_session

        s = _make_session()
        save_session(s, metadata={"branch": "main", "cwd": "/tmp"})

        sessions = list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["metadata"] == {"branch": "main", "cwd": "/tmp"}

    def test_save_preserves_is_error(self):
        from nemocode.core.sqlite_store import load_session, save_session

        s = Session(id="err-1")
        s.add_tool_result("call-1", '{"error": "fail"}', is_error=True)
        save_session(s)

        loaded = load_session("err-1")
        assert loaded is not None
        assert loaded.messages[0].is_error is True


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_empty(self):
        from nemocode.core.sqlite_store import list_sessions

        assert list_sessions() == []

    def test_list_returns_summaries(self):
        from nemocode.core.sqlite_store import list_sessions, save_session

        save_session(_make_session("a"))
        save_session(_make_session("b"))

        result = list_sessions()
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert ids == {"a", "b"}

    def test_list_respects_limit(self):
        from nemocode.core.sqlite_store import list_sessions, save_session

        for i in range(10):
            s = _make_session(f"s-{i}")
            s.updated_at = time.time() + i
            save_session(s)

        result = list_sessions(limit=3)
        assert len(result) == 3

    def test_list_ordered_by_updated(self):
        from nemocode.core.sqlite_store import list_sessions, save_session

        old = _make_session("old")
        old.updated_at = 1000.0
        new = _make_session("new")
        new.updated_at = 2000.0

        save_session(old)
        save_session(new)

        result = list_sessions()
        assert result[0]["id"] == "new"
        assert result[1]["id"] == "old"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_existing(self):
        from nemocode.core.sqlite_store import delete_session, load_session, save_session

        save_session(_make_session())
        assert delete_session("test-1") is True
        assert load_session("test-1") is None

    def test_delete_nonexistent(self):
        from nemocode.core.sqlite_store import delete_session

        assert delete_session("ghost") is False

    def test_delete_cascades_messages(self):
        from nemocode.core.sqlite_store import _get_conn, delete_session, save_session

        save_session(_make_session())
        delete_session("test-1")

        conn = _get_conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("test-1",)
            ).fetchone()[0]
            assert count == 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearchSessions:
    def test_search_by_content(self):
        from nemocode.core.sqlite_store import save_session, search_sessions

        s1 = Session(id="s1")
        s1.add_user("How do I configure nginx?")
        save_session(s1)

        s2 = Session(id="s2")
        s2.add_user("Help me with Docker")
        save_session(s2)

        results = search_sessions("nginx")
        assert len(results) == 1
        assert results[0]["id"] == "s1"

    def test_search_no_match(self):
        from nemocode.core.sqlite_store import save_session, search_sessions

        save_session(_make_session())
        results = search_sessions("zzzznotfound")
        assert results == []

    def test_search_case_insensitive(self):
        from nemocode.core.sqlite_store import save_session, search_sessions

        s = Session(id="ci")
        s.add_user("Configure NGINX proxy")
        save_session(s)

        assert len(search_sessions("nginx")) == 1
        assert len(search_sessions("NGINX")) == 1

    def test_search_returns_distinct(self):
        from nemocode.core.sqlite_store import save_session, search_sessions

        s = Session(id="dup")
        s.add_user("foo bar")
        s.add_assistant(Message(role=Role.ASSISTANT, content="foo baz"))
        save_session(s)

        results = search_sessions("foo")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Migration from JSON
# ---------------------------------------------------------------------------


class TestMigration:
    def test_migrate_json_files(self, tmp_path: Path):
        from nemocode.core.sqlite_store import load_session, migrate_json_sessions

        json_dir = tmp_path / "json_sessions"
        json_dir.mkdir()

        data = {
            "id": "migrated-1",
            "endpoint_name": "nim",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "created_at": 1000.0,
            "updated_at": 2000.0,
            "message_count": 3,
        }
        (json_dir / "migrated-1.json").write_text(json.dumps(data))

        count = migrate_json_sessions(json_dir)
        assert count == 1

        loaded = load_session("migrated-1")
        assert loaded is not None
        assert loaded.endpoint_name == "nim"
        assert loaded.message_count() == 3

    def test_migrate_skips_existing(self, tmp_path: Path):
        from nemocode.core.sqlite_store import migrate_json_sessions, save_session

        # Pre-save the session so it already exists
        save_session(_make_session("already"))

        json_dir = tmp_path / "json_sessions"
        json_dir.mkdir()
        data = {
            "id": "already",
            "messages": [{"role": "user", "content": "old"}],
            "usage": {},
            "created_at": 0,
            "updated_at": 0,
        }
        (json_dir / "already.json").write_text(json.dumps(data))

        count = migrate_json_sessions(json_dir)
        assert count == 0

    def test_migrate_empty_dir(self, tmp_path: Path):
        from nemocode.core.sqlite_store import migrate_json_sessions

        empty = tmp_path / "empty"
        empty.mkdir()
        assert migrate_json_sessions(empty) == 0

    def test_migrate_nonexistent_dir(self, tmp_path: Path):
        from nemocode.core.sqlite_store import migrate_json_sessions

        assert migrate_json_sessions(tmp_path / "nope") == 0
