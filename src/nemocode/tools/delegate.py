# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Sub-agent delegation tool — spawn focused sub-agents for subtasks.

NVIDIA advantage: explore/fast/test use Nano mini-agents (75% cheaper),
review/plan use Super. On DGX Spark, all agents run locally.
"""

from __future__ import annotations

import json
import logging

from nemocode.config.schema import FormationRole, NeMoCodeConfig
from nemocode.core.registry import Registry
from nemocode.core.scheduler import ROLE_PROMPTS, Scheduler
from nemocode.tools import ToolDef, tool
from nemocode.tools.loader import load_tools

logger = logging.getLogger(__name__)

# Agent type → (preferred endpoint tier, tools, role, prompt)
_AGENT_TYPES = {
    "explore": {
        "prefer_tier": "nano-9b",
        "tools": ["fs", "rg"],
        "role": FormationRole.FAST,
        "prompt": (
            "You are an exploration agent. Search the codebase to answer the question. "
            "Read files and search for patterns. Report your findings concisely."
        ),
    },
    "plan": {
        "prefer_tier": "super",
        "tools": [],
        "role": FormationRole.PLANNER,
        "prompt": ROLE_PROMPTS[FormationRole.PLANNER],
    },
    "review": {
        "prefer_tier": "super",
        "tools": ["fs", "rg", "git"],
        "role": FormationRole.REVIEWER,
        "prompt": ROLE_PROMPTS[FormationRole.REVIEWER],
    },
    "fast": {
        "prefer_tier": "nano",
        "tools": ["fs", "git", "bash", "rg"],
        "role": FormationRole.FAST,
        "prompt": ROLE_PROMPTS[FormationRole.FAST],
    },
    # New mini-agent types for Nano
    "code-search": {
        "prefer_tier": "nano-9b",
        "tools": ["fs", "rg", "glob"],
        "role": FormationRole.FAST,
        "prompt": (
            "You are a code search agent. Find specific code patterns, definitions, "
            "usages, and references. Return file paths and line numbers."
        ),
    },
    "test": {
        "prefer_tier": "nano",
        "tools": ["bash", "fs"],
        "role": FormationRole.FAST,
        "prompt": (
            "You are a test execution agent. Run the specified tests and report "
            "results. If tests fail, include the relevant error output."
        ),
    },
    "doc": {
        "prefer_tier": "nano-9b",
        "tools": ["fs", "rg"],
        "role": FormationRole.FAST,
        "prompt": (
            "You are a documentation agent. Read the code and generate clear, "
            "concise documentation. Focus on what the code does and why."
        ),
    },
    "debug": {
        "prefer_tier": "nano",
        "tools": ["fs", "bash", "rg", "git"],
        "role": FormationRole.DEBUGGER,
        "prompt": ROLE_PROMPTS[FormationRole.DEBUGGER],
    },
}


def _pick_endpoint(config: NeMoCodeConfig, prefer_tier: str) -> str:
    """Pick an endpoint based on preferred tier.

    Prefers Spark-local endpoints when available, then hosted.
    Tier preference order:
      nano-9b: spark-* -nano9b → nim-nano-9b → nim-nano → default
      nano:    spark-* -nano → nim-nano → default
      super:   spark-* -super → nim-super → default
    """
    # Priority 1: Spark-local endpoints (zero latency, free)
    # Check NIM first, then SGLang, then vLLM — all are local on Spark.
    spark_nim_map = {
        "nano-9b": "spark-nim-nano9b",
        "nano": "spark-nim-nano",
        "super": "spark-nim-super",
    }
    spark_sglang_map = {
        "nano-9b": "spark-sglang-nano9b",
        "nano": "spark-sglang-nano9b",  # SGLang Spark defaults to Nano 9B for mini-agents
        "super": "spark-sglang-super",
    }
    spark_vllm_map = {
        "nano-9b": "spark-vllm-nano9b",
        "nano": "spark-vllm-nano9b",  # vLLM Spark doesn't have Nano 30B, fall back to 9B
        "super": "spark-vllm-super",
    }
    for spark_map in (spark_nim_map, spark_sglang_map, spark_vllm_map):
        spark_name = spark_map.get(prefer_tier)
        if spark_name and spark_name in config.endpoints:
            return spark_name

    # Priority 2: Match by tier keyword in model_id
    tier_keywords = {
        "nano-9b": ["nano-9b", "nano9b"],
        "nano": ["nano"],
        "super": ["super"],
    }
    keywords = tier_keywords.get(prefer_tier, [prefer_tier])

    for name, ep in config.endpoints.items():
        model_lower = ep.model_id.lower()
        # Skip local endpoints that aren't running (Spark handles this above)
        if "local" in name and "spark" not in name:
            continue
        for keyword in keywords:
            if keyword in model_lower:
                return name

    return config.default_endpoint


def create_delegate_tool(registry: Registry, config: NeMoCodeConfig) -> ToolDef:
    """Create a delegate tool bound to the given registry and config.

    Returns a ToolDef that can be registered on a ToolRegistry.
    """

    @tool(
        description="Delegate a subtask to a focused sub-agent.",
        category="delegate",
    )
    async def delegate(
        task: str,
        agent_type: str = "explore",
        context: str = "",
    ) -> str:
        """Spawn a sub-agent to handle a focused subtask.

        task: The specific task or question for the sub-agent
        agent_type: One of 'explore', 'plan', 'review', 'fast',
            'code-search', 'test', 'doc', 'debug'
        context: Additional context to provide to the sub-agent
        """
        spec = _AGENT_TYPES.get(agent_type)
        if spec is None:
            valid = ", ".join(_AGENT_TYPES.keys())
            return json.dumps({"error": f"Unknown agent type: {agent_type}. Use: {valid}"})

        endpoint = _pick_endpoint(config, spec["prefer_tier"])
        tool_cats = spec["tools"]
        tool_registry = load_tools(tool_cats) if tool_cats else load_tools([])

        scheduler = Scheduler(
            registry=registry,
            tool_registry=tool_registry,
            confirm_fn=None,  # sub-agents auto-approve
        )

        user_msg = task
        if context:
            user_msg = f"## Context\n{context}\n\n## Task\n{task}"

        output_parts: list[str] = []
        tool_results: list[dict] = []

        try:
            async for event in scheduler.run_single(endpoint, user_msg):
                if event.kind == "text":
                    output_parts.append(event.text)
                elif event.kind == "tool_result":
                    tool_results.append(
                        {
                            "tool": event.tool_name,
                            "is_error": event.is_error,
                        }
                    )
                elif event.kind == "error":
                    output_parts.append(f"[ERROR] {event.text}")
        except Exception as e:
            return json.dumps({"error": f"Sub-agent failed: {e}"})

        result = {
            "agent_type": agent_type,
            "endpoint": endpoint,
            "output": "".join(output_parts),
            "tool_calls": len(tool_results),
            "errors": sum(1 for t in tool_results if t["is_error"]),
        }
        return json.dumps(result)

    return delegate._tool_def
