# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test context window management."""

from __future__ import annotations

from pathlib import Path

from nemocode.core import context as context_mod
from nemocode.core.context import (
    ContextManager,
    configure_token_counting,
    estimate_tokens,
    is_accurate,
    token_count_status,
)
from nemocode.core.streaming import Message, Role


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        tokens = estimate_tokens("Hello world")
        assert tokens > 0

    def test_long_text(self):
        text = "x" * 4000
        tokens = estimate_tokens(text)
        assert tokens > 0  # exact count depends on tiktoken availability

    def test_prefers_configured_model_tokenizer_when_cached_locally(self, monkeypatch):
        class FakeTokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False):
                return {"input_ids": text.split()}

        monkeypatch.setattr(context_mod, "_DEFAULT_MODEL_ID", None)
        monkeypatch.setattr(
            context_mod,
            "_load_cached_transformers_tokenizer",
            lambda model: FakeTokenizer(),
        )
        monkeypatch.setattr(context_mod, "_resolve_exact_tiktoken_encoding", lambda model: None)
        configure_token_counting("nvidia/test-model")

        status = token_count_status()
        assert status.exact is True
        assert status.method == "transformers-local"
        assert estimate_tokens("one two three") == 3
        assert is_accurate() is True

    def test_unsupported_model_reports_estimate_status(self, monkeypatch):
        class FakeEncoding:
            name = "o200k_base"

            def encode(self, text: str, disallowed_special=()):
                return [0, 1, 2, 3]

        monkeypatch.setattr(context_mod, "_DEFAULT_MODEL_ID", None)
        monkeypatch.setattr(context_mod, "_load_cached_transformers_tokenizer", lambda model: None)
        monkeypatch.setattr(context_mod, "_resolve_exact_tiktoken_encoding", lambda model: None)
        monkeypatch.setattr(context_mod, "_get_tiktoken_encoding", lambda name: FakeEncoding())
        configure_token_counting("nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8")

        status = token_count_status()
        assert status.exact is False
        assert status.method == "tiktoken:o200k_base"
        assert "estimate" in status.detail
        assert estimate_tokens("hello world") == 4

    def test_character_fallback_status_is_explicit(self, monkeypatch):
        monkeypatch.setattr(context_mod, "_DEFAULT_MODEL_ID", None)
        monkeypatch.setattr(context_mod, "_HAS_TIKTOKEN", False)
        monkeypatch.setattr(context_mod, "_load_cached_transformers_tokenizer", lambda model: None)
        monkeypatch.setattr(context_mod, "_resolve_exact_tiktoken_encoding", lambda model: None)
        monkeypatch.setattr(context_mod, "_get_tiktoken_encoding", lambda name: None)

        status = token_count_status("nvidia/test-model")
        assert status.exact is False
        assert status.method == "chars/4"
        assert "Install tiktoken" in status.detail


class TestContextManager:
    def test_usage(self):
        mgr = ContextManager()
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="Hello"),
        ]
        usage = mgr.usage(messages)
        assert usage > 0

    def test_usage_uses_manager_model_id(self, monkeypatch):
        seen_models: list[str | None] = []

        def fake_estimate_message_tokens(msg: Message, model_id: str | None = None) -> int:
            seen_models.append(model_id)
            return 10

        monkeypatch.setattr(context_mod, "estimate_message_tokens", fake_estimate_message_tokens)
        mgr = ContextManager(context_window=1000, model_id="nvidia/test-model")
        messages = [Message(role=Role.USER, content="Hello")]
        assert mgr.usage(messages) == 10
        assert seen_models == ["nvidia/test-model"]

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
