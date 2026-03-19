# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Sub-agent orchestration tools backed by named agent profiles."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging

from nemocode.config.agents import resolve_agent_reference
from nemocode.config.schema import (
    AgentConfig,
    AgentMode,
    Endpoint,
    EndpointTier,
    FormationRole,
    NeMoCodeConfig,
)
from nemocode.core.registry import Registry
from nemocode.core.scheduler import ROLE_PROMPTS, Scheduler
from nemocode.core.subagents import (
    append_output,
    attach_task,
    cancel_run,
    close_run,
    complete_run,
    fail_run,
    get_run,
    is_final_status,
    record_tool_result,
    resume_run,
    snapshot_run,
    start_run,
    wait_for_run,
)
from nemocode.tools import ToolDef, tool
from nemocode.tools.loader import load_tools

logger = logging.getLogger(__name__)
_nickname_counters: dict[str, itertools.count] = {}
_READ_ONLY_CATEGORY_ALIASES = {
    "fs": "fs_read",
    "git": "git_read",
}
_READ_ONLY_SAFE_CATEGORIES = {
    "fs_read",
    "git_read",
    "rg",
    "glob",
    "clarify",
    "web",
    "parse",
    "lsp",
}

_LOCAL_TIERS = {
    EndpointTier.LOCAL_NIM,
    EndpointTier.LOCAL_OLLAMA,
    EndpointTier.LOCAL_SGLANG,
    EndpointTier.LOCAL_VLLM,
}


def _agent_is_subagent(agent: AgentConfig) -> bool:
    return agent.mode in {AgentMode.SUBAGENT, AgentMode.ALL}


def _list_subagents(
    config: NeMoCodeConfig,
    *,
    include_hidden: bool = False,
) -> list[tuple[str, AgentConfig]]:
    items = []
    for name, agent in config.agents.items():
        if not _agent_is_subagent(agent):
            continue
        if agent.hidden and not include_hidden:
            continue
        items.append((name, agent))
    return sorted(items, key=lambda item: item[0])


def _resolve_subagent(config: NeMoCodeConfig, name: str) -> AgentConfig | None:
    resolved = resolve_agent_reference(config.agents, name) or name
    agent = config.agents.get(resolved)
    if agent and _agent_is_subagent(agent):
        return agent
    return None


def _resolve_subagent_name(config: NeMoCodeConfig, name: str) -> str | None:
    resolved = resolve_agent_reference(config.agents, name) or name
    agent = config.agents.get(resolved)
    if agent and _agent_is_subagent(agent):
        return resolved
    return None


def _matches_tier(endpoint_name: str, endpoint: Endpoint, prefer_tier: str) -> bool:
    prefer = prefer_tier.lower()
    blob = f"{endpoint_name} {endpoint.name} {endpoint.model_id}".lower()

    if prefer == "nano-4b":
        return any(token in blob for token in ("nano-4b", "nano4b", "nemotron nano 4b"))
    if prefer == "nano-9b":
        return any(token in blob for token in ("nano-9b", "nano9b", "nemotron nano 9b"))
    if prefer == "nano":
        return any(
            token in blob
            for token in (
                "nemotron-3-nano",
                "nemotron 3 nano",
                "nano 30b",
                " nano ",
                "-nano",
            )
        )
    if prefer == "super":
        return "super" in blob
    return prefer in blob


def _prefer_local_endpoints(config: NeMoCodeConfig) -> bool:
    default_endpoint = config.endpoints.get(config.default_endpoint)
    if default_endpoint and default_endpoint.tier in _LOCAL_TIERS:
        return True

    if config.active_formation:
        formation = config.formations.get(config.active_formation)
        if formation:
            for slot in formation.slots:
                endpoint = config.endpoints.get(slot.endpoint)
                if endpoint and endpoint.tier in _LOCAL_TIERS:
                    return True

    return False


def _endpoint_rank(
    name: str,
    endpoint: Endpoint,
    *,
    prefer_local: bool,
) -> tuple[int, int, int, int, str]:
    if prefer_local:
        locality_rank = 0 if name.startswith("spark-") else 1 if endpoint.tier in _LOCAL_TIERS else 2
    else:
        locality_rank = 0 if endpoint.tier not in _LOCAL_TIERS else 1 if name.startswith("spark-") else 2

    transport_rank = {
        EndpointTier.LOCAL_SGLANG: 0,
        EndpointTier.LOCAL_NIM: 1,
        EndpointTier.LOCAL_VLLM: 2,
        EndpointTier.LOCAL_OLLAMA: 3,
        EndpointTier.DEV_HOSTED: 4,
        EndpointTier.PROD_HOSTED: 4,
        EndpointTier.INFERENCE_PARTNER: 5,
        EndpointTier.OPENROUTER: 6,
    }.get(endpoint.tier, 99)

    nvidia_hosted_rank = (
        0
        if endpoint.base_url.startswith("https://integrate.api.nvidia.com")
        or name.startswith("nim-")
        else 1
    )

    return (locality_rank, transport_rank, nvidia_hosted_rank, len(endpoint.model_id), name)


def _pick_endpoint(
    config: NeMoCodeConfig,
    prefer_tier: str | list[str] | None,
    explicit_endpoint: str | None = None,
) -> str:
    """Pick an endpoint based on explicit override or preferred family tiers."""
    if explicit_endpoint:
        if explicit_endpoint in config.endpoints:
            return explicit_endpoint
        logger.warning("Agent requested unknown endpoint %s", explicit_endpoint)

    tiers: list[str] = []
    if isinstance(prefer_tier, str):
        if prefer_tier:
            tiers = [prefer_tier]
    elif prefer_tier:
        tiers = [tier for tier in prefer_tier if tier]

    for tier in tiers:
        matches = [
            (name, endpoint)
            for name, endpoint in config.endpoints.items()
            if _matches_tier(name, endpoint, tier)
        ]
        if matches:
            prefer_local = _prefer_local_endpoints(config)
            matches.sort(
                key=lambda item: _endpoint_rank(item[0], item[1], prefer_local=prefer_local)
            )
            return matches[0][0]

    return config.default_endpoint


def _agent_prompt(agent: AgentConfig) -> str:
    if agent.prompt:
        return agent.prompt
    return ROLE_PROMPTS.get(agent.role, ROLE_PROMPTS[FormationRole.FAST])


def _display_name(agent_name: str, agent: AgentConfig) -> str:
    return agent.display_name or agent_name


def _pick_nickname(agent_name: str, agent: AgentConfig) -> str:
    candidates = agent.nickname_candidates or [_display_name(agent_name, agent)]
    counter = _nickname_counters.setdefault(agent_name, itertools.count(0))
    idx = next(counter) % len(candidates)
    return candidates[idx]


def _available_subagents_text(
    config: NeMoCodeConfig,
    *,
    parent_read_only: bool = False,
) -> list[str]:
    visible = _list_subagents(config)
    if not visible:
        return ["No sub-agents are configured."]

    lines = ["Available sub-agents:"]
    if parent_read_only:
        lines.append(
            "All delegated sub-agents inherit read-only mode here: no file writes, shell commands, or commits."
        )
    for name, agent in visible:
        title = _display_name(name, agent)
        detail = agent.description or "Specialized sub-agent."
        if agent.prefer_tiers:
            detail += f" Preferred tiers: {', '.join(agent.prefer_tiers)}."
        lines.append(f"- {name} ({title}): {detail}")
    return lines


def _delegate_description(
    config: NeMoCodeConfig,
    *,
    parent_read_only: bool = False,
) -> str:
    lines = ["Delegate a focused subtask to a named sub-agent and wait for the final result."]
    lines.extend(_available_subagents_text(config, parent_read_only=parent_read_only))
    return "\n".join(lines)


def _spawn_description(
    config: NeMoCodeConfig,
    *,
    parent_read_only: bool = False,
) -> str:
    lines = [
        "Spawn a background sub-agent for a bounded parallel task.",
        "Use this only when the user has explicitly asked for delegation, sub-agents, or parallel agent work.",
        "Continue with non-overlapping local work after spawning. Only call wait_agent when blocked on the child result.",
    ]
    lines.extend(_available_subagents_text(config, parent_read_only=parent_read_only))
    return "\n".join(lines)


def _compose_user_message(task: str, context: str) -> str:
    if context:
        return f"## Context\n{context}\n\n## Task\n{task}"
    return task


def _restrict_to_read_only_categories(categories: list[str]) -> list[str]:
    """Map mixed tool categories to a read-only-safe subset."""
    restricted: list[str] = []
    seen: set[str] = set()
    for name in categories:
        mapped = _READ_ONLY_CATEGORY_ALIASES.get(name, name)
        if mapped in _READ_ONLY_SAFE_CATEGORIES and mapped not in seen:
            restricted.append(mapped)
            seen.add(mapped)
    return restricted


def _result_payload(
    run_id: str,
    *,
    include_output: bool = False,
    timed_out: bool = False,
) -> dict[str, object]:
    run = get_run(run_id)
    if run is None:
        return {"error": f"Unknown sub-agent run: {run_id}"}
    return snapshot_run(run, include_output=include_output, timed_out=timed_out)


async def _drive_subagent(
    run_id: str,
    scheduler: Scheduler,
    endpoint: str,
    user_msg: str,
) -> None:
    last_error = ""
    try:
        async for event in scheduler.run_single(endpoint, user_msg):
            if event.kind == "text":
                append_output(run_id, event.text)
            elif event.kind == "tool_result":
                record_tool_result(run_id, is_error=event.is_error)
            elif event.kind == "error":
                last_error = event.text or last_error
                append_output(run_id, f"[ERROR] {event.text}")
    except asyncio.CancelledError:
        cancel_run(run_id, "Cancelled by close_agent")
        raise
    except Exception as exc:
        fail_run(run_id, f"Sub-agent failed: {exc}")
        return

    run = get_run(run_id)
    if run is None:
        return

    if last_error:
        fail_run(run_id, last_error)
        return

    complete_run(
        run_id,
        output_preview=run.output[:500],
        output=run.output,
        tool_calls=run.tool_calls,
        errors=run.errors,
    )


def _spawn_subagent_run(
    registry: Registry,
    config: NeMoCodeConfig,
    *,
    task: str,
    agent_type: str,
    context: str,
    project_context: str,
    hook_runner: object | None,
    permission_engine: object | None,
    parent_agent_name: str,
    parent_read_only: bool,
) -> dict[str, object]:
    resolved_name = _resolve_subagent_name(config, agent_type)
    agent = _resolve_subagent(config, agent_type)
    if agent is None or resolved_name is None:
        valid = ", ".join(name for name, _ in _list_subagents(config, include_hidden=True))
        return {"error": f"Unknown sub-agent: {agent_type}. Available: {valid or '(none)'}"}

    endpoint = _pick_endpoint(
        config,
        agent.prefer_tiers,
        explicit_endpoint=agent.endpoint,
    )
    nickname = _pick_nickname(resolved_name, agent)
    session_id = f"{resolved_name}-{next(_nickname_counters.setdefault('__session__', itertools.count(1))):04d}"

    tool_categories = agent.tools
    if parent_read_only:
        tool_categories = _restrict_to_read_only_categories(tool_categories)
    tool_registry = load_tools(tool_categories) if tool_categories else load_tools([])
    scheduler = Scheduler(
        registry=registry,
        tool_registry=tool_registry,
        confirm_fn=None,
        hook_runner=hook_runner,
        read_only=parent_read_only,
        permission_engine=permission_engine,
        max_tool_rounds=config.max_tool_rounds,
        single_role=agent.role,
        single_prompt=_agent_prompt(agent),
        single_session_id=session_id,
    )
    if project_context:
        scheduler.inject_system_context(project_context)

    run = start_run(
        agent_name=resolved_name,
        display_name=_display_name(resolved_name, agent),
        nickname=nickname,
        parent_agent=parent_agent_name,
        task=task,
        context=context,
        endpoint=endpoint,
        session_id=session_id,
    )

    child_task = asyncio.create_task(
        _drive_subagent(
            run.id,
            scheduler,
            endpoint,
            _compose_user_message(task, context),
        )
    )
    attach_task(run.id, child_task)
    return snapshot_run(run, include_output=False)


def create_delegate_tools(
    registry: Registry,
    config: NeMoCodeConfig,
    *,
    project_context: str = "",
    hook_runner: object | None = None,
    permission_engine: object | None = None,
    parent_agent_name: str = "main",
    parent_read_only: bool = False,
) -> list[ToolDef]:
    """Create the sub-agent orchestration tool suite bound to the given runtime."""

    @tool(
        description=_delegate_description(config, parent_read_only=parent_read_only),
        category="delegate",
    )
    async def delegate(
        task: str,
        agent_type: str = "general",
        context: str = "",
    ) -> str:
        """Spawn a focused sub-agent and wait for the final result.

        task: The subtask for the delegated agent.
        agent_type: Name of the configured sub-agent profile to use.
        context: Extra context to prepend before the delegated task.
        """
        spawned = _spawn_subagent_run(
            registry,
            config,
            task=task,
            agent_type=agent_type,
            context=context,
            project_context=project_context,
            hook_runner=hook_runner,
            permission_engine=permission_engine,
            parent_agent_name=parent_agent_name,
            parent_read_only=parent_read_only,
        )
        if "error" in spawned:
            return json.dumps(spawned)

        run_id = str(spawned["run_id"])
        run, _ = await wait_for_run(run_id, timeout_s=None)
        if run is None:
            return json.dumps({"error": f"Unknown sub-agent run: {run_id}"})
        return json.dumps(snapshot_run(run, include_output=True))

    @tool(
        description=_spawn_description(config, parent_read_only=parent_read_only),
        category="delegate",
    )
    async def spawn_agent(
        task: str,
        agent_type: str = "general",
        context: str = "",
    ) -> str:
        """Spawn a background sub-agent and return its run metadata.

        task: The subtask for the spawned agent.
        agent_type: Name of the configured sub-agent profile to use.
        context: Extra context to prepend before the delegated task.
        """
        return json.dumps(
            _spawn_subagent_run(
                registry,
                config,
                task=task,
                agent_type=agent_type,
                context=context,
                project_context=project_context,
                hook_runner=hook_runner,
                permission_engine=permission_engine,
                parent_agent_name=parent_agent_name,
                parent_read_only=parent_read_only,
            )
        )

    @tool(
        description=(
            "Wait for a spawned sub-agent to finish. Use this sparingly and only when the child "
            "result is on the critical path."
        ),
        category="delegate",
    )
    async def wait_agent(
        agent_id: str,
        timeout_s: float = 30.0,
    ) -> str:
        """Wait for a spawned sub-agent to reach a final state.

        agent_id: Run id returned by spawn_agent.
        timeout_s: Maximum seconds to wait before returning the current status.
        """
        if timeout_s <= 0:
            return json.dumps({"error": "timeout_s must be greater than zero"})

        run, timed_out = await wait_for_run(agent_id, timeout_s=timeout_s)
        if run is None:
            return json.dumps({"error": f"Unknown sub-agent run: {agent_id}"})

        include_output = not timed_out and is_final_status(run.status)
        return json.dumps(_result_payload(agent_id, include_output=include_output, timed_out=timed_out))

    @tool(
        description=(
            "Close a spawned sub-agent handle. Set cancel=true to stop a running child; otherwise "
            "the child continues in the background and can be reopened with resume_agent."
        ),
        category="delegate",
    )
    async def close_agent(
        agent_id: str,
        cancel: bool = False,
    ) -> str:
        """Close a spawned sub-agent handle.

        agent_id: Run id returned by spawn_agent.
        cancel: When true, cancel a running child before closing it.
        """
        run, previous_status = await close_run(agent_id, cancel=cancel)
        if run is None:
            return json.dumps({"error": f"Unknown sub-agent run: {agent_id}"})

        payload = snapshot_run(run, include_output=is_final_status(run.status))
        payload["previous_status"] = previous_status
        payload["cancel_requested"] = cancel
        return json.dumps(payload)

    @tool(
        description="Resume a previously closed sub-agent handle so it can be waited on or inspected again.",
        category="delegate",
    )
    async def resume_agent(
        agent_id: str,
    ) -> str:
        """Reopen a previously closed sub-agent handle.

        agent_id: Run id returned by spawn_agent.
        """
        run = resume_run(agent_id)
        if run is None:
            return json.dumps({"error": f"Unknown sub-agent run: {agent_id}"})
        return json.dumps(snapshot_run(run, include_output=is_final_status(run.status)))

    return [
        delegate._tool_def,
        spawn_agent._tool_def,
        wait_agent._tool_def,
        close_agent._tool_def,
        resume_agent._tool_def,
    ]
