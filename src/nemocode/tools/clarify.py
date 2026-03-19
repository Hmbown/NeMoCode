# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""User clarification tool — ask questions mid-turn.

The agent can use this tool to pause execution and ask the user
a question when it needs clarification before proceeding.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Awaitable, Callable

from nemocode.tools import tool

# Callback set by the REPL/TUI at startup. When None, the tool returns
# a message asking the user to answer inline.
_ask_fn: Callable[[str, list[str]], Awaitable[str]] | None = None


def set_ask_fn(fn: Callable[[str, list[str]], Awaitable[str]] | None) -> None:
    """Register the user-input callback (called by REPL/TUI at init)."""
    global _ask_fn
    _ask_fn = fn


async def request_user_response(
    question: str,
    options: list[str] | None = None,
) -> tuple[str, bool]:
    """Get a user response via the registered callback or a TTY fallback.

    Returns a tuple of ``(answer, pending)`` where ``pending`` indicates that no
    interactive input path was available and the caller should defer to a later turn.
    """
    option_list = [o.strip() for o in (options or []) if o.strip()]

    if _ask_fn is not None:
        answer = await _ask_fn(question, option_list)
        return answer, False

    if sys.stdin.isatty():
        prompt = question
        if option_list:
            prompt += f"\nOptions: {', '.join(option_list)}"
        prompt += "\n> "
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, lambda: input(prompt))
        return answer.strip(), False

    return "", True


@tool(
    name="ask_clarify",
    description="Ask the user a clarifying question and wait for their response.",
    category="clarify",
)
async def ask_clarify(
    question: str,
    options: str = "",
) -> str:
    """Ask the user a question and return their answer.
    question: The question to ask the user.
    options: Comma-separated list of suggested options (optional).
    """
    if not question.strip():
        return json.dumps({"error": "Question must not be empty"})

    option_list = [o.strip() for o in options.split(",") if o.strip()] if options else []

    try:
        answer, pending = await request_user_response(question, option_list)
        if pending:
            msg = f"[Clarification needed] {question}"
            if option_list:
                msg += f"\nOptions: {', '.join(option_list)}"
            msg += "\nPlease answer this question in your next message."
            return json.dumps({"question": question, "answer": "", "pending": True, "message": msg})
        return json.dumps({"question": question, "answer": answer})
    except asyncio.CancelledError:
        return json.dumps({"question": question, "answer": "(cancelled)"})
    except Exception as e:
        return json.dumps({"error": f"Failed to get user input: {e}"})
