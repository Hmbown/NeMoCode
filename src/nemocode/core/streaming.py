# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Core message and streaming types.

Defines the message protocol used by all providers and the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Protocol


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False
    name: str | None = None


@dataclass
class StreamChunk:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


@dataclass
class CompletionResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class ChatProvider(Protocol):
    """Protocol for chat completion providers."""

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RerankProvider(Protocol):
    async def rerank(
        self, query: str, passages: list[str], top_n: int = 5
    ) -> list[tuple[int, float]]: ...
