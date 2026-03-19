# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tool to ask the user a question and get a response."""

from __future__ import annotations

import asyncio
import sys
from nemocode.tools import tool

@tool(
    name="ask_user",
    description="Ask the user a question and get a response.",
)
async def ask_user(question: str) -> str:
    """Ask the user a question and return their response.

    Args:
        question: The question to ask the user.

    Returns:
        The user's response as a string.
    """
    # If stdin is not a TTY, we cannot interact; return empty string or default?
    if not sys.stdin.isatty():
        # In non-interactive mode, we return an empty string or a default?
        # For now, return empty string to avoid blocking.
        return ""
    # Run the blocking input in a thread to avoid blocking the event loop
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: input(f"{question} ")
        )
        return response.strip()
    except (EOFError, KeyboardInterrupt):
        return ""