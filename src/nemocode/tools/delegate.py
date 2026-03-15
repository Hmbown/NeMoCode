# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Sub-agent delegation tool — spawn focused sub-agents for subtasks.

NVIDIA advantage: explore/fast use Nano (75% cheaper), review uses Super.
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

# Agent type → (preferred endpoint tier, tools, role)
_AGENT_TYPES = {
    "explore": {
        "prefer_tier": "nano",
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
}


def _pick_endpoint(config: NeMoCodeConfig, prefer_tier: str) -> str:
    """Pick an endpoint based on preferred tier."""
    for name, ep in config.endpoints.items():
        model_lower = ep.model_id.lower()
        if prefer_tier == "nano" and "nano" in model_lower:
            return name
        if prefer_tier == "super" and "super" in model_lower:
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
        agent_type: One of 'explore', 'plan', 'review', 'fast'
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
