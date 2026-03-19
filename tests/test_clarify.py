# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the ask_user clarification tool."""

from __future__ import annotations

import json

import pytest

from nemocode.tools.ask_user import ask_user
from nemocode.tools.clarify import ask_clarify, request_user_response, set_ask_fn


@pytest.fixture(autouse=True)
def reset_ask_fn():
    """Ensure ask_fn is reset between tests."""
    set_ask_fn(None)
    yield
    set_ask_fn(None)


@pytest.mark.asyncio
async def test_ask_clarify_no_callback():
    """Without a callback, returns pending message."""
    result = json.loads(await ask_clarify("What language?"))
    assert result["question"] == "What language?"
    assert result["pending"] is True


@pytest.mark.asyncio
async def test_ask_clarify_with_callback():
    """With a callback, returns the user's answer."""

    async def mock_ask(question: str, options: list[str]) -> str:
        return "Python"

    set_ask_fn(mock_ask)
    result = json.loads(await ask_clarify("What language?"))
    assert result["answer"] == "Python"
    assert result["question"] == "What language?"


@pytest.mark.asyncio
async def test_ask_clarify_with_options():
    """Options are passed to the callback."""

    received_options = []

    async def mock_ask(question: str, options: list[str]) -> str:
        received_options.extend(options)
        return "Python"

    set_ask_fn(mock_ask)
    result = json.loads(await ask_clarify("Pick language", options="Python, Rust, Go"))
    assert result["answer"] == "Python"
    assert received_options == ["Python", "Rust", "Go"]


@pytest.mark.asyncio
async def test_ask_clarify_empty_question():
    result = json.loads(await ask_clarify(""))
    assert "error" in result


@pytest.mark.asyncio
async def test_ask_clarify_callback_error():
    """Callback errors are handled gracefully."""

    async def failing_ask(question: str, options: list[str]) -> str:
        raise RuntimeError("Input failed")

    set_ask_fn(failing_ask)
    result = json.loads(await ask_clarify("question"))
    assert "error" in result


@pytest.mark.asyncio
async def test_request_user_response_uses_callback():
    async def mock_ask(question: str, options: list[str]) -> str:
        assert question == "Approve?"
        assert options == ["yes", "no"]
        return "yes"

    set_ask_fn(mock_ask)
    answer, pending = await request_user_response("Approve?", ["yes", "no"])
    assert answer == "yes"
    assert pending is False


@pytest.mark.asyncio
async def test_ask_user_uses_callback():
    async def mock_ask(question: str, options: list[str]) -> str:
        assert question == "Need approval?"
        assert options == []
        return "approve"

    set_ask_fn(mock_ask)
    assert await ask_user("Need approval?") == "approve"
