# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Metrics — token usage, latency, and cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Pricing per million tokens (approximate, NIM API Catalog)
PRICING: dict[str, dict[str, float]] = {
    "nvidia/nemotron-3-super-120b-a12b": {"input": 0.35, "output": 0.40},
    "nvidia/nemotron-3-nano-30b-a3b": {"input": 0.10, "output": 0.10},
    "nvidia/nemotron-3-ultra": {"input": 1.00, "output": 1.20},
    "nvidia/llama-nemotron-embed-1b-v2": {"input": 0.02, "output": 0.0},
    "nvidia/llama-nemotron-rerank-1b-v2": {"input": 0.02, "output": 0.0},
}


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts."""
    pricing = PRICING.get(model_id, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens / 1_000_000) * pricing["input"]
    cost += (completion_tokens / 1_000_000) * pricing["output"]
    return cost


@dataclass
class RequestMetrics:
    model_id: str = ""
    endpoint_name: str = ""
    formation_role: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_sec(self) -> float:
        """Completion tokens per second for this request."""
        if self.total_time_ms <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / (self.total_time_ms / 1000)

    @property
    def estimated_cost(self) -> float:
        return estimate_cost(self.model_id, self.prompt_tokens, self.completion_tokens)


class MetricsCollector:
    """Collects metrics across a session."""

    def __init__(self) -> None:
        self._requests: list[RequestMetrics] = []
        self._session_start = time.time()

    def record(self, metrics: RequestMetrics) -> None:
        self._requests.append(metrics)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self._requests)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self._requests)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost(self) -> float:
        return sum(r.estimated_cost for r in self._requests)

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def avg_latency_ms(self) -> float:
        if not self._requests:
            return 0.0
        return sum(r.total_time_ms for r in self._requests) / len(self._requests)

    @property
    def avg_tokens_per_sec(self) -> float:
        """Average completion tokens/sec across all requests with output."""
        reqs = [r for r in self._requests if r.tokens_per_sec > 0]
        if not reqs:
            return 0.0
        return sum(r.tokens_per_sec for r in reqs) / len(reqs)

    @property
    def last_tokens_per_sec(self) -> float:
        """Tokens/sec of the most recent request."""
        if not self._requests:
            return 0.0
        return self._requests[-1].tokens_per_sec

    @property
    def avg_ttft_ms(self) -> float:
        """Average time-to-first-token across requests that reported it."""
        reqs = [r for r in self._requests if r.time_to_first_token_ms > 0]
        if not reqs:
            return 0.0
        return sum(r.time_to_first_token_ms for r in reqs) / len(reqs)

    @property
    def session_duration_s(self) -> float:
        return time.time() - self._session_start

    def summary(self) -> dict[str, Any]:
        return {
            "requests": self.request_count,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens_per_sec": round(self.avg_tokens_per_sec, 1),
            "last_tokens_per_sec": round(self.last_tokens_per_sec, 1),
            "avg_ttft_ms": round(self.avg_ttft_ms, 1),
            "session_duration_s": round(self.session_duration_s, 1),
        }

    def reset(self) -> None:
        self._requests.clear()
        self._session_start = time.time()
