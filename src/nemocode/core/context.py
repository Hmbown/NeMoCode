# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Context window manager — tracks usage and manages smart compaction.

Token counting prefers an exact local tokenizer for the active model, then
falls back to tiktoken when it can resolve the model or a deliberate family
estimate, and only uses the old character heuristic as a last resort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from nemocode.core.streaming import Message, Role

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting — exact when possible, explicit estimate otherwise
# ---------------------------------------------------------------------------

_tiktoken = None
_HAS_TIKTOKEN = False
_DEFAULT_MODEL_ID: str | None = None

try:
    import tiktoken

    _tiktoken = tiktoken
    _HAS_TIKTOKEN = True
    logger.debug("tiktoken available for token counting")
except ImportError:
    logger.debug("tiktoken not installed — using character-based estimation")

# Approximate tokens per character ratio for English text (fallback)
_CHARS_PER_TOKEN = 4
_GENERIC_TIKTOKEN_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class TokenCountStatus:
    """Describes how token counts are being produced for a model."""

    exact: bool
    method: str
    detail: str


def configure_token_counting(model_id: str | None) -> None:
    """Set the default model used for token counting when callers omit one."""
    global _DEFAULT_MODEL_ID
    _DEFAULT_MODEL_ID = model_id.strip() if model_id and model_id.strip() else None


def _active_model_id(model_id: str | None = None) -> str | None:
    candidate = model_id.strip() if model_id and model_id.strip() else None
    return candidate or _DEFAULT_MODEL_ID


@lru_cache(maxsize=32)
def _load_cached_transformers_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as exc:
        logger.debug("No local tokenizer cache for %s: %s", model_id, exc)
        return None


@lru_cache(maxsize=8)
def _get_tiktoken_encoding(name: str):
    if not _HAS_TIKTOKEN or _tiktoken is None:
        return None
    try:
        return _tiktoken.get_encoding(name)
    except Exception as exc:
        logger.debug("Failed to load tiktoken encoding %s: %s", name, exc)
        return None


@lru_cache(maxsize=32)
def _resolve_exact_tiktoken_encoding(model_id: str):
    if not _HAS_TIKTOKEN or _tiktoken is None:
        return None
    try:
        return _tiktoken.encoding_for_model(model_id)
    except KeyError:
        return None
    except Exception as exc:
        logger.debug("Failed to resolve exact tiktoken encoding for %s: %s", model_id, exc)
        return None


def _family_tiktoken_encoding_name(model_id: str) -> str | None:
    lowered = model_id.lower()
    if any(
        marker in lowered
        for marker in (
            "nemotron-3",
            "nvidia-nemotron-3",
            "nemotron-nano",
            "nemotron-content-safety",
            "llama-3.1-nemotron",
        )
    ):
        return "o200k_base"
    return None


def _transformers_count(text: str, tokenizer) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return len(input_ids)


def token_count_status(model_id: str | None = None) -> TokenCountStatus:
    """Return the current token counting mode for the active or given model."""
    active_model = _active_model_id(model_id)

    if active_model:
        tokenizer = _load_cached_transformers_tokenizer(active_model)
        if tokenizer is not None:
            return TokenCountStatus(
                exact=True,
                method="transformers-local",
                detail=f"Using the locally cached tokenizer for {active_model}.",
            )

        encoding = _resolve_exact_tiktoken_encoding(active_model)
        if encoding is not None:
            return TokenCountStatus(
                exact=True,
                method=f"tiktoken:{encoding.name}",
                detail=f"Using tiktoken's exact model mapping for {active_model}.",
            )

        family_encoding_name = _family_tiktoken_encoding_name(active_model)
        if family_encoding_name:
            return TokenCountStatus(
                exact=False,
                method=f"tiktoken:{family_encoding_name}",
                detail=(
                    "Using a model-family tiktoken estimate because this model id does not "
                    "have an exact built-in tokenizer mapping."
                ),
            )

    if _HAS_TIKTOKEN:
        return TokenCountStatus(
            exact=False,
            method=f"tiktoken:{_GENERIC_TIKTOKEN_ENCODING}",
            detail=(
                "Using a generic tiktoken estimate. Configure a model or cache its tokenizer "
                "locally for exact counts."
            ),
        )

    return TokenCountStatus(
        exact=False,
        method="chars/4",
        detail="Using a character heuristic. Install tiktoken for better token estimates.",
    )


def estimate_tokens(text: str, model_id: str | None = None) -> int:
    """Count or estimate tokens in text.

    Prefers an exact local tokenizer when available, then exact tiktoken model
    mappings, then explicit estimate paths.
    """
    if not text:
        return 0

    active_model = _active_model_id(model_id)
    if active_model:
        tokenizer = _load_cached_transformers_tokenizer(active_model)
        if tokenizer is not None:
            try:
                return _transformers_count(text, tokenizer)
            except Exception as exc:
                logger.debug("Local tokenizer count failed for %s: %s", active_model, exc)

        encoding = _resolve_exact_tiktoken_encoding(active_model)
        if encoding is not None:
            try:
                return len(encoding.encode(text, disallowed_special=()))
            except Exception as exc:
                logger.debug("Exact tiktoken count failed for %s: %s", active_model, exc)

        family_encoding_name = _family_tiktoken_encoding_name(active_model)
        if family_encoding_name:
            encoding = _get_tiktoken_encoding(family_encoding_name)
            if encoding is not None:
                try:
                    return len(encoding.encode(text, disallowed_special=()))
                except Exception as exc:
                    logger.debug(
                        "Model-family tiktoken estimate failed for %s via %s: %s",
                        active_model,
                        family_encoding_name,
                        exc,
                    )

    generic_encoding = _get_tiktoken_encoding(_GENERIC_TIKTOKEN_ENCODING)
    if generic_encoding is not None:
        try:
            return len(generic_encoding.encode(text, disallowed_special=()))
        except Exception as exc:
            logger.debug("Generic tiktoken estimate failed: %s", exc)

    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Message, model_id: str | None = None) -> int:
    """Count or estimate token count for a single message."""
    tokens = estimate_tokens(msg.content, model_id=model_id)
    if msg.thinking:
        tokens += estimate_tokens(msg.thinking, model_id=model_id)
    for tc in msg.tool_calls:
        tokens += estimate_tokens(str(tc.arguments), model_id=model_id) + 20  # function overhead
    return tokens + 4  # role/message overhead


def is_accurate(model_id: str | None = None) -> bool:
    """Return True if current token counting is exact for the given model."""
    return token_count_status(model_id=model_id).exact


class ContextManager:
    """Manages context window usage and smart compaction."""

    def __init__(self, context_window: int = 1_048_576, model_id: str | None = None) -> None:
        self.context_window = context_window
        self.model_id = _active_model_id(model_id)

    def usage(self, messages: list[Message]) -> int:
        """Estimate total token usage for a message list."""
        return sum(estimate_message_tokens(m, model_id=self.model_id) for m in messages)

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
