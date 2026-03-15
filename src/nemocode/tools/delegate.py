# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Sub-agent delegation tool — spawn focused sub-agents for subtasks.

NVIDIA advantage: explore/fast use Nano (75% cheaper), review uses Super.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from nemocode.config.schema import FormationRole
from nemocode.core.registry import Registry
from nemocode.core.scheduler import ROLE_PROMPTS, Scheduler
from nemocode.tools import tool
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

# Module-level reference set by CodeAgent
_registry: Registry | None = None
_config: Any = None


def configure_delegate(registry: Registry, config: Any) -> None:
    """Set the registry and config for delegation. Called by CodeAgent.__init__."""
    global _registry, _config
    _registry = registry
    _config = config


def _pick_endpoint(prefer_tier: str) -> str:
    """Pick an endpoint based on preferred tier."""
    if _config is None:
        return "nim-super"

    # Try to find an endpoint matching the preferred tier
    for name, ep in _config.endpoints.items():
        model_lower = ep.model_id.lower()
        if prefer_tier == "nano" and "nano" in model_lower:
            return name
        if prefer_tier == "super" and "super" in model_lower:
            return name

    # Fall back to default
    return _config.default_endpoint


@tool(
    description=("Delegate a subtask to a focused sub-agent. Types: explore, plan, review, fast."),
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
    if _registry is None or _config is None:
        return json.dumps({"error": "Delegate not configured — call configure_delegate first"})

    spec = _AGENT_TYPES.get(agent_type)
    if spec is None:
        return json.dumps(
            {"error": f"Unknown agent type: {agent_type}. Use: {', '.join(_AGENT_TYPES.keys())}"}
        )

    endpoint = _pick_endpoint(spec["prefer_tier"])
    tool_cats = spec["tools"]
    tool_registry = load_tools(tool_cats) if tool_cats else load_tools([])

    scheduler = Scheduler(
        registry=_registry,
        tool_registry=tool_registry,
        confirm_fn=None,  # sub-agents auto-approve
    )

    # Build the user message
    user_msg = task
    if context:
        user_msg = f"## Context\n{context}\n\n## Task\n{task}"

    # Run the sub-agent and collect output
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
