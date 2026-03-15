# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Smart formation auto-routing — classify task complexity and pick optimal formation.

Active when mode is 'auto' and no formation explicitly set.
NVIDIA advantage: routes simple tasks to Nano (75% cheaper) automatically.
"""

from __future__ import annotations

import logging
import re

from nemocode.config.schema import NeMoCodeConfig

logger = logging.getLogger(__name__)


class TaskComplexity:
    SIMPLE = "simple"  # Short question, lookup → Nano
    MODERATE = "moderate"  # Single-file edit, explanation → Solo (Super)
    COMPLEX = "complex"  # Multi-file, architecture → Super+Nano formation
    REVIEW = "review"  # Code review request → Solo with reviewer prompt


# Heuristic patterns
_SIMPLE_PATTERNS = [
    r"^what\s+(is|are|does)\b",
    r"^how\s+(do|does|to)\b",
    r"^explain\b",
    r"^show\s+me\b",
    r"^list\b",
    r"^where\s+(is|are)\b",
    r"^which\b",
    r"^find\b",
]

_COMPLEX_PATTERNS = [
    r"\brefactor\b",
    r"\bmigrat(e|ion)\b",
    r"\barchitect(ure)?\b",
    r"\bredesign\b",
    r"\bimplement\b.*\band\b",
    r"\bmulti.?file\b",
    r"\bacross\s+(all|multiple|every)\b",
    r"\bproject.?wide\b",
]

_REVIEW_PATTERNS = [
    r"\breview\b",
    r"\baudit\b",
    r"\bcheck\s+(for|my|the|this)\b",
    r"\bfind\s+(bugs?|issues?|problems?)\b",
    r"\bcode\s+quality\b",
]


def classify_task(user_input: str) -> str:
    """Classify task complexity based on input heuristics.

    Returns one of TaskComplexity values.
    """
    lower = user_input.lower().strip()
    word_count = len(lower.split())

    # Check review patterns first (specific intent, regardless of length)
    for pattern in _REVIEW_PATTERNS:
        if re.search(pattern, lower):
            return TaskComplexity.REVIEW

    # Check complex patterns before short-input heuristic
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, lower):
            return TaskComplexity.COMPLEX

    # Very short inputs are usually simple
    if word_count <= 5:
        return TaskComplexity.SIMPLE

    # Check simple patterns
    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, lower):
            return TaskComplexity.SIMPLE

    # Medium-length requests without strong signals → moderate
    if word_count <= 20:
        return TaskComplexity.MODERATE

    # Long, detailed requests are likely complex
    return TaskComplexity.COMPLEX


def route_to_formation(user_input: str, config: NeMoCodeConfig) -> str | None:
    """Choose a formation based on task classification.

    Returns formation name to use, or None for default single-endpoint mode.
    """
    complexity = classify_task(user_input)
    logger.debug("Task classified as: %s", complexity)

    if complexity == TaskComplexity.SIMPLE:
        # Use Nano endpoint if available
        for name, ep in config.endpoints.items():
            if "nano" in ep.model_id.lower() and "local" not in name:
                # Switch to nano endpoint, no formation
                return None  # Will use nano endpoint directly

    if complexity == TaskComplexity.COMPLEX:
        # Use super-nano formation if available
        if "super-nano" in config.formations:
            return "super-nano"

    # For MODERATE and REVIEW, use default single endpoint (Super)
    return None


def get_auto_endpoint(user_input: str, config: NeMoCodeConfig) -> str | None:
    """Get the optimal endpoint for auto-routing.

    Returns endpoint name override, or None to keep default.
    """
    complexity = classify_task(user_input)

    if complexity == TaskComplexity.SIMPLE:
        # Find a nano endpoint
        for name, ep in config.endpoints.items():
            if "nano" in ep.model_id.lower() and "local" not in name:
                return name

    # For everything else, use default (Super)
    return None
