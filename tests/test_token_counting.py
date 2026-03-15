# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for accurate token counting."""

from __future__ import annotations

from nemocode.core.context import estimate_message_tokens, estimate_tokens, is_accurate
from nemocode.core.streaming import Message, Role


class TestTokenCounting:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        tokens = estimate_tokens("Hello world")
        assert tokens >= 1

    def test_long_text(self):
        text = "word " * 1000
        tokens = estimate_tokens(text)
        assert tokens > 100

    def test_message_tokens(self):
        msg = Message(role=Role.USER, content="Hello, how are you?")
        tokens = estimate_message_tokens(msg)
        assert tokens >= 5

    def test_message_with_thinking(self):
        msg = Message(role=Role.ASSISTANT, content="Answer", thinking="Let me think...")
        tokens = estimate_message_tokens(msg)
        assert tokens > estimate_message_tokens(
            Message(role=Role.ASSISTANT, content="Answer")
        )

    def test_is_accurate_reports_status(self):
        # is_accurate should return a bool regardless of tiktoken availability
        result = is_accurate()
        assert isinstance(result, bool)

    def test_consistency(self):
        """Same text should always return same count."""
        text = "The quick brown fox jumps over the lazy dog"
        a = estimate_tokens(text)
        b = estimate_tokens(text)
        assert a == b

    def test_proportional(self):
        """Longer text should have more tokens."""
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short
