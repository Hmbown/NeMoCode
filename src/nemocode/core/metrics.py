# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Metrics — token usage, latency, and cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Pricing per million tokens (approximate, NIM API Catalog)
PRICING: dict[str, dict[str, float]] = {
    # Nemotron family
    "nvidia/nemotron-3-super-120b-a12b": {"input": 0.35, "output": 0.40},
    "nvidia/nemotron-3-nano-30b-a3b": {"input": 0.10, "output": 0.10},
    "nvidia/nemotron-3-ultra": {"input": 1.00, "output": 1.20},
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": {"input": 0.90, "output": 1.10},
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": {"input": 0.20, "output": 0.30},
    "nvidia/llama-nemotron-embed-1b-v2": {"input": 0.02, "output": 0.0},
    "nvidia/llama-nemotron-rerank-1b-v2": {"input": 0.02, "output": 0.0},
    # DeepSeek family (NIM hosted)
    "deepseek-ai/deepseek-v4-pro": {"input": 0.55, "output": 1.10},
    "deepseek-ai/deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-ai/deepseek-v3.2": {"input": 0.27, "output": 1.10},
    "deepseek-ai/deepseek-v3.1-terminus": {"input": 0.27, "output": 1.10},
    "deepseek-ai/deepseek-coder-6.7b-instruct": {"input": 0.05, "output": 0.10},
    # Other frontier NIM
    "qwen/qwen3-coder-480b-a35b-instruct": {"input": 0.40, "output": 0.80},
    "qwen/qwen3-next-80b-a3b-thinking": {"input": 0.20, "output": 0.40},
    "openai/gpt-oss-120b": {"input": 0.30, "output": 0.60},
    "moonshotai/kimi-k2-thinking": {"input": 0.60, "output": 1.20},
    "moonshotai/kimi-k2.5": {"input": 0.50, "output": 1.00},
    "mistralai/mistral-large-3-675b-instruct-2512": {"input": 0.80, "output": 2.40},
    "mistralai/devstral-2-123b-instruct-2512": {"input": 0.30, "output": 0.60},
    "z-ai/glm5": {"input": 0.30, "output": 0.60},
    "minimaxai/minimax-m2.7": {"input": 0.30, "output": 1.20},
    "meta/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.40},
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

    @property
    def usage_by_model(self) -> dict[str, dict[str, Any]]:
        """Return token usage grouped by model_id."""
        groups: dict[str, dict[str, Any]] = {}
        for r in self._requests:
            if r.model_id not in groups:
                groups[r.model_id] = {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            g = groups[r.model_id]
            g["requests"] += 1
            g["prompt_tokens"] += r.prompt_tokens
            g["completion_tokens"] += r.completion_tokens
            g["total_tokens"] += r.total_tokens
            g["estimated_cost_usd"] += r.estimated_cost
        # Round costs
        for g in groups.values():
            g["estimated_cost_usd"] = round(g["estimated_cost_usd"], 6)
        return groups

    @property
    def usage_by_endpoint(self) -> dict[str, dict[str, Any]]:
        """Return token usage grouped by endpoint_name."""
        groups: dict[str, dict[str, Any]] = {}
        for r in self._requests:
            if r.endpoint_name not in groups:
                groups[r.endpoint_name] = {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            g = groups[r.endpoint_name]
            g["requests"] += 1
            g["prompt_tokens"] += r.prompt_tokens
            g["completion_tokens"] += r.completion_tokens
            g["total_tokens"] += r.total_tokens
            g["estimated_cost_usd"] += r.estimated_cost
        # Round costs
        for g in groups.values():
            g["estimated_cost_usd"] = round(g["estimated_cost_usd"], 6)
        return groups

    def to_records(self) -> list[dict[str, Any]]:
        """Return list of dicts suitable for SQLite storage."""
        return [
            {
                "model_id": r.model_id,
                "endpoint_name": r.endpoint_name,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "thinking_tokens": r.thinking_tokens,
                "total_tokens": r.total_tokens,
                "total_time_ms": r.total_time_ms,
                "tool_calls": r.tool_calls,
                "estimated_cost_usd": round(r.estimated_cost, 6),
                "timestamp": r.timestamp,
            }
            for r in self._requests
        ]

    def reset(self) -> None:
        self._requests.clear()
        self._session_start = time.time()
