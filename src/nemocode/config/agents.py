# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Agent profile defaults and markdown-based discovery."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from nemocode.config.schema import AgentConfig

_BUILTIN_AGENTS_RAW: dict[str, dict[str, Any]] = {
    "build": {
        "description": "Default full-access coding agent.",
        "display_name": "NeMoCode",
        "aliases": ["builder", "build-mode"],
        "nickname_candidates": ["NeMoCode", "Build Mode"],
        "mode": "primary",
        "role": "executor",
        "tools": [
            "fs",
            "git",
            "bash",
            "rg",
            "glob",
            "http",
            "memory",
            "tasks",
            "web",
            "parse",
            "test",
            "clarify",
            "lsp",
        ],
    },
    "plan": {
        "description": "Read-only planning agent for analysis and code exploration.",
        "display_name": "Plan Mode",
        "aliases": ["planner", "plan-mode"],
        "nickname_candidates": ["Plan Mode", "Planner"],
        "mode": "primary",
        "role": "planner",
        "tools": [
            "fs_read",
            "git_read",
            "rg",
            "glob",
            "clarify",
            "delegate",
            "spawn_agent",
            "wait_agent",
            "close_agent",
            "resume_agent",
        ],
        "prompt": (
            "You are NeMoCode in plan mode. Read the codebase, analyze the task, "
            "and propose concrete next steps without modifying files or running "
            "destructive commands. Use ask_clarify only when blocked on requirements. "
            "You can also spawn read-only research subagents for exploration. "
            "When you have a plan, present it clearly for approval. "
            "The controller handles approval, revision, or cancellation after you "
            "respond. If revised, incorporate the feedback and return only the "
            "updated plan."
        ),
    },
    "general": {
        "description": (
            "General-purpose subagent for focused implementation, validation, and "
            "multi-step research tasks."
        ),
        "display_name": "Joe Nemotron",
        "aliases": ["scout", "generalist", "joe", "joey"],
        "nickname_candidates": [
            "Joseph Nemotron III",
            "Joey Nemo",
            "Geronemo",
            "Giovannemotron",
        ],
        "mode": "subagent",
        "role": "fast",
        "prefer_tiers": ["nano-9b", "nano", "super"],
        "tools": ["fs", "git", "bash", "rg", "glob"],
        "prompt": (
            "You are a general-purpose NeMoCode subagent. Complete the requested "
            "subtask directly, keep scope tight, and report the result concisely."
        ),
    },
    "explore": {
        "description": "Fast read-only subagent for codebase exploration and search.",
        "display_name": "Geronemo",
        "aliases": ["explorer", "searcher"],
        "nickname_candidates": ["Geronemo", "Joegle Scout", "Needle Joe"],
        "mode": "subagent",
        "role": "fast",
        "prefer_tiers": ["nano-4b", "nano-9b", "nano", "super"],
        "tools": ["fs_read", "rg", "glob"],
        "prompt": (
            "You are an exploration subagent. Search the codebase, read the minimum "
            "necessary files, and answer with concrete paths and findings."
        ),
    },
    "review": {
        "description": "Read-only review subagent for correctness and regression checks.",
        "display_name": "Giovannemotron",
        "aliases": ["reviewer", "critic"],
        "nickname_candidates": ["Giovannemotron", "Joe Verdict", "Joeverruled"],
        "mode": "subagent",
        "role": "reviewer",
        "prefer_tiers": ["super"],
        "tools": ["fs_read", "rg", "git_read"],
    },
    "debug": {
        "description": "Debugging subagent for isolating failures and root causes.",
        "display_name": "Joebug Nemotron",
        "aliases": ["debugger", "tracer"],
        "nickname_candidates": ["Joebug Nemotron", "Stack Joeverflow", "Breakpoint Joe"],
        "mode": "subagent",
        "role": "debugger",
        "prefer_tiers": ["nano-9b", "nano", "super"],
        "tools": ["fs_read", "bash", "rg", "git_read"],
    },
    "test": {
        "description": "Focused subagent for running tests and summarizing failures.",
        "display_name": "Joe Testotron",
        "aliases": ["tester", "pytester"],
        "nickname_candidates": ["Joe Testotron", "Joeverify", "Assertin Joe"],
        "mode": "subagent",
        "role": "fast",
        "prefer_tiers": ["nano-4b", "nano-9b", "nano"],
        "tools": ["bash", "fs_read"],
        "prompt": (
            "You are a test execution subagent. Run the requested checks, keep the "
            "logs relevant, and report failures with the most useful excerpts."
        ),
    },
    "doc": {
        "description": "Documentation subagent for concise code-aware writing tasks.",
        "display_name": "Joecumentron",
        "aliases": ["docs", "writer"],
        "nickname_candidates": ["Joecumentron", "Readme Joe", "Manual Nemotron"],
        "mode": "subagent",
        "role": "fast",
        "prefer_tiers": ["nano-4b", "nano-9b", "nano"],
        "tools": ["fs", "rg"],
        "prompt": (
            "You are a documentation subagent. Write clear, concise project-facing "
            "documentation based on the code and request."
        ),
    },
    "code-search": {
        "description": "Targeted code search subagent for symbols, references, and patterns.",
        "display_name": "Joegle Nemo",
        "aliases": ["needle", "codesearch"],
        "nickname_candidates": ["Joegle Nemo", "Needle Joe", "Ctrl+Joe"],
        "mode": "subagent",
        "role": "fast",
        "prefer_tiers": ["nano-4b", "nano-9b", "nano"],
        "tools": ["fs_read", "rg", "glob"],
        "prompt": (
            "You are a code-search subagent. Find definitions, callsites, patterns, "
            "and relevant files, then report exact locations."
        ),
    },
    "fast": {
        "description": "Legacy fast worker subagent for tightly scoped tasks.",
        "display_name": "Joetron Express",
        "aliases": ["speedy", "sprinter"],
        "nickname_candidates": ["Joetron Express", "Turbo Joe", "Joey Quickstep"],
        "mode": "subagent",
        "role": "fast",
        "prefer_tiers": ["nano", "nano-9b", "super"],
        "tools": ["fs", "git", "bash", "rg"],
    },
}


def builtin_agents_raw() -> dict[str, dict[str, Any]]:
    """Return a copy of the built-in agent profile definitions."""
    return deepcopy(_BUILTIN_AGENTS_RAW)


def resolve_agent_reference(agents: dict[str, AgentConfig], query: str) -> str | None:
    """Resolve a user-facing agent name, alias, or display name to its canonical id."""
    if query in agents:
        return query

    lowered = query.strip().lower()
    if not lowered:
        return None

    for name, agent in agents.items():
        if name.lower() == lowered:
            return name
        if agent.display_name and agent.display_name.lower() == lowered:
            return name
        if lowered in {alias.lower() for alias in agent.aliases}:
            return name
    return None


def discover_agent_markdown(project_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load markdown-defined agents from global and project directories."""
    project_dir = (project_dir or Path.cwd()).resolve()
    global_dir = Path(os.environ.get("NEMOCODE_CONFIG_DIR", "~/.config/nemocode")).expanduser()
    global_dir = global_dir / "agents"
    project_agents_dir = project_dir / ".nemocode" / "agents"

    discovered: dict[str, dict[str, Any]] = {}
    for directory in (global_dir, project_agents_dir):
        discovered.update(_load_agent_dir(directory))
    return discovered


def _load_agent_dir(directory: Path) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return agents

    for path in sorted(directory.rglob("*.md")):
        config = _parse_agent_markdown(path)
        name = path.relative_to(directory).with_suffix("").as_posix()
        agents[name] = config
    return agents


def _parse_agent_markdown(path: Path) -> dict[str, Any]:
    text = path.read_text()
    frontmatter, body = _split_frontmatter(text)
    data = yaml.safe_load(frontmatter) if frontmatter else {}
    if not isinstance(data, dict):
        data = {}
    prompt = body.strip()
    if prompt:
        data["prompt"] = prompt
    return data


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text

    end = text.find("\n---", 4)
    if end == -1:
        return "", text

    frontmatter = text[4:end]
    body_start = end + 4
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    return frontmatter, text[body_start:]
