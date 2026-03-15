# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test context window management."""

from __future__ import annotations

from pathlib import Path

from nemocode.core.context import ContextManager, estimate_tokens
from nemocode.core.streaming import Message, Role


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short_text(self):
        tokens = estimate_tokens("Hello world")
        assert tokens > 0

    def test_long_text(self):
        text = "x" * 4000
        tokens = estimate_tokens(text)
        assert tokens == 1000  # 4000 / 4


class TestContextManager:
    def test_usage(self):
        mgr = ContextManager()
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="Hello"),
        ]
        usage = mgr.usage(messages)
        assert usage > 0

    def test_usage_fraction(self):
        mgr = ContextManager(context_window=1000)
        messages = [
            Message(role=Role.USER, content="x" * 2000),  # ~500 tokens
        ]
        frac = mgr.usage_fraction(messages)
        assert 0 < frac < 1

    def test_should_compact(self):
        mgr = ContextManager(context_window=100)
        messages = [
            Message(role=Role.USER, content="x" * 4000),  # Way over threshold
        ]
        assert mgr.should_compact(messages, threshold=0.8) is True

    def test_usage_bar(self):
        mgr = ContextManager(context_window=10000)
        messages = [Message(role=Role.USER, content="x" * 20000)]
        bar = mgr.usage_bar(messages)
        assert "[" in bar
        assert "]" in bar
        assert "%" in bar

    def test_smart_compact(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"Message {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Response {i}"))
        assert len(messages) == 61

        compacted = mgr.smart_compact(messages, keep_recent=10)
        assert len(compacted) < len(messages)
        assert compacted[0].role == Role.SYSTEM
        # Last messages should be preserved
        assert compacted[-1].content == "Response 29"

    def test_smart_compact_noop_if_small(self):
        mgr = ContextManager()
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi"),
        ]
        result = mgr.smart_compact(messages, keep_recent=10)
        assert result == messages

    def test_load_project_context(self, tmp_path: Path):
        mgr = ContextManager()
        (tmp_path / "README.md").write_text("# My Project\nThis is a test.")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        context = mgr.load_project_context(["README.md", "pyproject.toml"], tmp_path)
        assert "My Project" in context
        assert "test" in context

    def test_load_missing_context_file(self, tmp_path: Path):
        mgr = ContextManager()
        context = mgr.load_project_context(["nonexistent.md"], tmp_path)
        assert context == ""
