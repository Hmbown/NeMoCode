# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""User clarification tool — ask questions mid-turn.

The agent can use this tool to pause execution and ask the user
a question when it needs clarification before proceeding.
"""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from nemocode.tools import tool

# Callback set by the REPL/TUI at startup. When None, the tool returns
# a message asking the user to answer inline.
_ask_fn: Callable[[str, list[str]], Awaitable[str]] | None = None


def set_ask_fn(fn: Callable[[str, list[str]], Awaitable[str]] | None) -> None:
    """Register the user-input callback (called by REPL/TUI at init)."""
    global _ask_fn
    _ask_fn = fn


@tool(
    name="ask_user",
    description="Ask the user a clarifying question and wait for their response.",
    category="clarify",
)
async def ask_user(
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

    if _ask_fn is not None:
        try:
            answer = await _ask_fn(question, option_list)
            return json.dumps({"question": question, "answer": answer})
        except asyncio.CancelledError:
            return json.dumps({"question": question, "answer": "(cancelled)"})
        except Exception as e:
            return json.dumps({"error": f"Failed to get user input: {e}"})
    else:
        # No callback registered — return instruction for non-interactive mode
        msg = f"[Clarification needed] {question}"
        if option_list:
            msg += f"\nOptions: {', '.join(option_list)}"
        msg += "\nPlease answer this question in your next message."
        return json.dumps({"question": question, "answer": "", "pending": True, "message": msg})
