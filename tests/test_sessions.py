# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test conversation sessions and compaction."""

from __future__ import annotations

from nemocode.core.persistence import (
    delete_session,
    list_sessions,
    load_session,
    revert_to_point,
    save_session,
)
from nemocode.core.sessions import Session, TokenUsage
from nemocode.core.streaming import Message, Role


class TestTokenUsage:
    def test_add(self):
        usage = TokenUsage()
        usage.add(100, 50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_add_multiple(self):
        usage = TokenUsage()
        usage.add(100, 50)
        usage.add(200, 100)
        assert usage.total_tokens == 450

    def test_as_dict(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = usage.as_dict()
        assert d["prompt_tokens"] == 100
        assert d["total_tokens"] == 150


class TestSession:
    def test_add_system(self):
        session = Session(id="test")
        session.add_system("You are helpful")
        assert len(session.messages) == 1
        assert session.messages[0].role == Role.SYSTEM

    def test_replace_system(self):
        session = Session(id="test")
        session.add_system("First")
        session.add_system("Second")
        assert len(session.messages) == 1
        assert session.messages[0].content == "Second"

    def test_add_user(self):
        session = Session(id="test")
        session.add_user("Hello")
        assert len(session.messages) == 1
        assert session.messages[0].role == Role.USER

    def test_add_assistant(self):
        session = Session(id="test")
        msg = Message(role=Role.ASSISTANT, content="Hi there")
        session.add_assistant(msg)
        assert session.last_assistant_text() == "Hi there"

    def test_last_assistant_text_empty(self):
        session = Session(id="test")
        assert session.last_assistant_text() == ""

    def test_compact(self):
        session = Session(id="test")
        session.add_system("System prompt")
        for i in range(30):
            session.add_user(f"Message {i}")
        assert session.message_count() == 31
        session.compact(keep=5)
        assert session.message_count() <= 6  # system + 5 recent
        assert session.messages[0].role == Role.SYSTEM

    def test_compact_noop_if_small(self):
        session = Session(id="test")
        session.add_system("System")
        session.add_user("Hello")
        session.compact(keep=10)
        assert session.message_count() == 2

    def test_to_dict(self):
        session = Session(id="test-123", endpoint_name="nim-super")
        session.usage.add(100, 50)
        d = session.to_dict()
        assert d["id"] == "test-123"
        assert d["endpoint_name"] == "nim-super"
        assert d["usage"]["total_tokens"] == 150

    def test_checkpoint_restore(self):
        session = Session(id="checkpoint-test")
        session.add_system("System prompt")
        session.add_user("Original request")
        checkpoint = session.checkpoint()

        session.add_assistant(Message(role=Role.ASSISTANT, content="Changed"))
        session.usage.add(120, 30)

        session.restore(checkpoint)

        assert session.message_count() == 2
        assert session.last_assistant_text() == ""
        assert session.last_user_text() == "Original request"
        assert session.usage.total_tokens == 0


class TestSessionPersistence:
    def test_save_and_load_session(self, tmp_path):
        session = Session(id="persist-test")
        session.add_system("System prompt")
        session.add_user("Hello")
        session.add_assistant(Message(role=Role.ASSISTANT, content="Hi there"))

        # Save
        save_session(session, {"cwd": str(tmp_path), "endpoint": "nim-super"})

        # Load
        loaded = load_session("persist-test")
        assert loaded is not None
        assert loaded.id == "persist-test"
        assert len(loaded.messages) == 3
        assert loaded.messages[0].role == Role.SYSTEM
        assert loaded.messages[0].content == "System prompt"
        assert loaded.messages[1].role == Role.USER
        assert loaded.messages[1].content == "Hello"
        assert loaded.messages[2].role == Role.ASSISTANT
        assert loaded.messages[2].content == "Hi there"

        # Cleanup
        delete_session("persist-test")

    def test_list_sessions(self, tmp_path):
        session1 = Session(id="list-test-unique-1")
        session1.add_user("First")
        save_session(session1)

        session2 = Session(id="list-test-unique-2")
        session2.add_user("Second")
        save_session(session2)

        # List
        sessions = list_sessions(limit=10)
        ids = [s["id"] for s in sessions]
        assert "list-test-unique-1" in ids
        assert "list-test-unique-2" in ids

        # Cleanup
        for sid in ["list-test-unique-1", "list-test-unique-2"]:
            delete_session(sid)

    def test_delete_session(self, tmp_path):
        session = Session(id="delete-test")
        session.add_user("Test")
        save_session(session)

        # Delete
        assert delete_session("delete-test")

        # Verify gone
        loaded = load_session("delete-test")
        assert loaded is None


class TestTurnRevert:
    def test_revert_to_point_restores_multiple_file_changes(self, tmp_path):
        from nemocode.tools.fs import _UNDO_STACK

        _UNDO_STACK.clear()
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("new first")
        second.write_text("new second")
        _UNDO_STACK.append((str(first), "old first"))
        _UNDO_STACK.append((str(second), "old second"))

        results = revert_to_point(0)

        assert len(results) == 2
        assert first.read_text() == "old first"
        assert second.read_text() == "old second"
        assert _UNDO_STACK == []
