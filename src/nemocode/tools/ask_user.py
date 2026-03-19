# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tool to ask the user a question and get a response."""

from __future__ import annotations

from nemocode.tools import tool
from nemocode.tools.clarify import request_user_response


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
    try:
        response, pending = await request_user_response(question)
        if pending:
            return ""
        return response.strip()
    except (EOFError, KeyboardInterrupt):
        return ""
