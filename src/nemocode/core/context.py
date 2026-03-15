# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Context window manager — tracks usage and manages smart compaction."""

from __future__ import annotations

import logging
from pathlib import Path

from nemocode.core.streaming import Message, Role

logger = logging.getLogger(__name__)

# Approximate tokens per character ratio for English text
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Message) -> int:
    """Estimate token count for a single message."""
    tokens = estimate_tokens(msg.content)
    if msg.thinking:
        tokens += estimate_tokens(msg.thinking)
    for tc in msg.tool_calls:
        tokens += estimate_tokens(str(tc.arguments)) + 20  # function overhead
    return tokens + 4  # role/message overhead


class ContextManager:
    """Manages context window usage and smart compaction."""

    def __init__(self, context_window: int = 1_048_576) -> None:
        self.context_window = context_window

    def usage(self, messages: list[Message]) -> int:
        """Estimate total token usage for a message list."""
        return sum(estimate_message_tokens(m) for m in messages)

    def usage_fraction(self, messages: list[Message]) -> float:
        """Return context usage as a fraction (0.0 to 1.0)."""
        return self.usage(messages) / self.context_window

    def usage_bar(self, messages: list[Message], width: int = 30) -> str:
        """Visual progress bar of context usage."""
        frac = self.usage_fraction(messages)
        filled = int(frac * width)
        bar = "#" * filled + "-" * (width - filled)
        pct = frac * 100
        return f"[{bar}] {pct:.1f}%"

    def should_compact(self, messages: list[Message], threshold: float = 0.8) -> bool:
        """Check if compaction is needed."""
        return self.usage_fraction(messages) > threshold

    def smart_compact(self, messages: list[Message], keep_recent: int = 20) -> list[Message]:
        """Compact messages while preserving important context.

        Avoids breaking tool_call/tool_result pairs and avoids producing
        multiple system messages (some APIs reject that).
        """
        if len(messages) <= keep_recent + 1:
            return messages

        system = messages[0] if messages[0].role == Role.SYSTEM else None
        start_idx = 1 if system else 0

        # Find a safe cut point that doesn't split tool_call/tool_result pairs
        cut = len(messages) - keep_recent
        if cut <= start_idx:
            return messages
        while cut > start_idx and messages[cut].role in (Role.TOOL, Role.ASSISTANT):
            if messages[cut].role == Role.TOOL:
                cut -= 1
            elif messages[cut].role == Role.ASSISTANT and messages[cut].tool_calls:
                cut -= 1
            else:
                break

        tail = messages[cut:]
        middle = messages[start_idx:cut]
        if not middle:
            return messages

        # Summarize the middle section
        summary_parts = []
        for msg in middle:
            if msg.role == Role.USER:
                preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                summary_parts.append(f"User: {preview}")
            elif msg.role == Role.ASSISTANT and msg.content:
                preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                summary_parts.append(f"Assistant: {preview}")

        summary_text = "\n".join(summary_parts[-10:])  # Keep last 10 summaries

        # Append summary to system message instead of creating a second one
        summary_block = (
            f"\n\n[Conversation summary - {len(middle)} messages compacted]\n{summary_text}"
        )
        if system:
            combined_system = Message(
                role=Role.SYSTEM,
                content=system.content + summary_block,
            )
            return [combined_system] + tail
        else:
            # No system message — inject summary as a user context note
            summary_msg = Message(role=Role.USER, content=summary_block)
            return [summary_msg] + tail

    def load_project_context(self, context_files: list[str], base_dir: Path | None = None) -> str:
        """Load and format project context files for injection into system prompt."""
        base = base_dir or Path.cwd()
        parts = []

        for fname in context_files:
            fpath = base / fname
            if not fpath.exists():
                continue
            try:
                content = fpath.read_text()
                if len(content) > 10_000:
                    content = content[:10_000] + "\n... (truncated)"
                parts.append(f"### {fname}\n```\n{content}\n```")
            except Exception as e:
                logger.debug("Failed to read context file %s: %s", fname, e)

        return "\n\n".join(parts)
