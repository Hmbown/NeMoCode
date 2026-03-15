# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Conversation session state, token tracking, and compaction."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from nemocode.core.streaming import Message, Role


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class Session:
    """Manages conversation state for one role in a formation."""

    def __init__(self, id: str = "", endpoint_name: str = "") -> None:
        self.id = id or uuid.uuid4().hex[:8]
        self.endpoint_name = endpoint_name
        self.messages: list[Message] = []
        self.usage = TokenUsage()
        self.created_at = time.time()
        self.updated_at = time.time()

    def add_system(self, content: str) -> None:
        if self.messages and self.messages[0].role == Role.SYSTEM:
            self.messages[0] = Message(role=Role.SYSTEM, content=content)
        else:
            self.messages.insert(0, Message(role=Role.SYSTEM, content=content))
        self.updated_at = time.time()

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role=Role.USER, content=content))
        self.updated_at = time.time()

    def add_assistant(self, msg: Message) -> None:
        self.messages.append(msg)
        self.updated_at = time.time()

    def add_tool_result(self, tool_call_id: str, content: str, is_error: bool = False) -> None:
        self.messages.append(
            Message(
                role=Role.TOOL,
                content=content,
                tool_call_id=tool_call_id,
                is_error=is_error,
            )
        )
        self.updated_at = time.time()

    def last_assistant_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == Role.ASSISTANT and msg.content:
                return msg.content
        return ""

    def compact(self, keep: int = 20) -> None:
        """Keep system prompt + last N messages, drop the middle.

        Adjusts the cut point to avoid splitting tool_call/tool_result pairs.
        """
        if len(self.messages) <= keep + 1:
            return
        system = self.messages[0] if self.messages[0].role == Role.SYSTEM else None
        start_idx = 1 if system else 0

        # Find a safe cut point: don't split in the middle of a tool sequence.
        # Walk backwards from the intended cut to find a user message boundary.
        cut = len(self.messages) - keep
        if cut <= start_idx:
            return
        # Move cut earlier until we land on a user message or start_idx
        while cut > start_idx and self.messages[cut].role in (Role.TOOL, Role.ASSISTANT):
            # If this is a tool result, its assistant+tool_call is before it — keep going back
            if self.messages[cut].role == Role.TOOL:
                cut -= 1
            elif self.messages[cut].role == Role.ASSISTANT and self.messages[cut].tool_calls:
                cut -= 1
            else:
                break

        tail = self.messages[cut:]
        self.messages = ([system] if system else []) + tail
        self.updated_at = time.time()

    def message_count(self) -> int:
        return len(self.messages)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "endpoint_name": self.endpoint_name,
            "message_count": self.message_count(),
            "usage": self.usage.as_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
