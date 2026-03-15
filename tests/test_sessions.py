# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test conversation sessions and compaction."""

from __future__ import annotations

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
