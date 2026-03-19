# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Configuration loader.

Merge order: defaults.yaml → ~/.config/nemocode/config.yaml → .nemocode.yaml → env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from nemocode.config.agents import builtin_agents_raw, discover_agent_markdown
from nemocode.config.schema import (
    AgentConfig,
    Capability,
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    HooksConfig,
    Manifest,
    MCPConfig,
    MCPServerConfig,
    MoEConfig,
    NeMoCodeConfig,
    NemotronArch,
    ProjectContext,
    ReasoningConfig,
    StructuredOutputConfig,
    ToolPermissions,
)

# Locate the bundled defaults.yaml (inside the package, not repo root)
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULTS_PATH = _PACKAGE_DIR / "defaults.yaml"
_USER_CONFIG_DIR = Path(os.environ.get("NEMOCODE_CONFIG_DIR", "~/.config/nemocode")).expanduser()
_USER_CONFIG_PATH = _USER_CONFIG_DIR / "config.yaml"
_PROJECT_CONFIG_NAME = ".nemocode.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _parse_endpoint(name: str, data: dict[str, Any]) -> Endpoint:
    caps = [Capability(c) for c in data.get("capabilities", ["chat"])]
    tier = EndpointTier(data.get("tier", "dev-hosted"))
    return Endpoint(
        name=data.get("name", name),
        tier=tier,
        base_url=data.get("base_url", ""),
        api_key_env=data.get("api_key_env"),
        model_id=data.get("model_id", ""),
        capabilities=caps,
        max_tokens=data.get("max_tokens", 16_384),
        extra_headers=data.get("extra_headers", {}),
        transport=data.get("transport", "openai"),
    )


def _parse_manifest(model_id: str, data: dict[str, Any]) -> Manifest:
    caps = [Capability(c) for c in data.get("capabilities", ["chat"])]
    moe_data = data.get("moe", {})
    reasoning_data = data.get("reasoning", {})
    structured_data = data.get("structured", {})

    return Manifest(
        model_id=model_id,
        display_name=data.get("display_name", ""),
        arch=NemotronArch(data.get("arch", "other")),
        tier_in_family=data.get("tier_in_family", ""),
        capabilities=caps,
        context_window=data.get("context_window", 128_000),
        max_output_tokens=data.get("max_output_tokens", 16_384),
        moe=MoEConfig(**moe_data) if moe_data else MoEConfig(),
        reasoning=ReasoningConfig(**reasoning_data) if reasoning_data else ReasoningConfig(),
        structured=StructuredOutputConfig(**structured_data)
        if structured_data
        else StructuredOutputConfig(),
        supports_tools=data.get("supports_tools", True),
        supports_parallel_tools=data.get("supports_parallel_tools", False),
        force_nonempty_content=data.get("force_nonempty_content", False),
        supports_lora=data.get("supports_lora", False),
        lora_adapters=data.get("lora_adapters", []),
        default_extra_body=data.get("default_extra_body", {}),
        min_gpu_memory_gb=data.get("min_gpu_memory_gb"),
        min_gpus=data.get("min_gpus"),
        recommended_gpu=data.get("recommended_gpu", ""),
    )


def _parse_formation(name: str, data: dict[str, Any]) -> Formation:
    slots = []
    for slot_data in data.get("slots", []):
        slots.append(
            FormationSlot(
                endpoint=slot_data["endpoint"],
                role=FormationRole(slot_data["role"]),
                lora_adapter=slot_data.get("lora_adapter"),
                system_prompt=slot_data.get("system_prompt"),
                reasoning_mode=slot_data.get("reasoning_mode"),
            )
        )
    return Formation(
        name=data.get("name", name),
        description=data.get("description", ""),
        slots=slots,
        tools=data.get("tools", ["fs", "git", "bash", "rg"]),
        verification_rounds=data.get("verification_rounds", 1),
        output_style=data.get("output_style"),
    )


def _parse_agent(name: str, data: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        name=data.get("name", name),
        description=data.get("description", ""),
        display_name=data.get("display_name", ""),
        aliases=data.get("aliases", []),
        nickname_candidates=data.get("nickname_candidates", []),
        mode=data.get("mode", "subagent"),
        hidden=data.get("hidden", False),
        endpoint=data.get("endpoint"),
        prefer_tiers=data.get("prefer_tiers", []),
        tools=data.get("tools", []),
        role=data.get("role", FormationRole.FAST),
        prompt=data.get("prompt", ""),
    )


def _parse_config(raw: dict[str, Any]) -> NeMoCodeConfig:
    endpoints = {}
    for name, edata in raw.get("endpoints", {}).items():
        endpoints[name] = _parse_endpoint(name, edata)

    manifests = {}
    for model_id, mdata in raw.get("manifests", {}).items():
        manifests[model_id] = _parse_manifest(model_id, mdata)

    formations = {}
    for name, fdata in raw.get("formations", {}).items():
        formations[name] = _parse_formation(name, fdata)

    agents = {}
    agents_raw = builtin_agents_raw()
    agents_raw = _deep_merge(agents_raw, raw.get("agents", {}))
    for name, adata in agents_raw.items():
        agents[name] = _parse_agent(name, adata)

    perms_data = raw.get("permissions", {})
    perms = ToolPermissions(**perms_data) if perms_data else ToolPermissions()

    project_data = raw.get("project", {})
    project = ProjectContext(**project_data) if project_data else ProjectContext()

    mcp_data = raw.get("mcp", {})
    mcp_servers = []
    for srv in mcp_data.get("servers", []):
        mcp_servers.append(MCPServerConfig(**srv))
    mcp = MCPConfig(servers=mcp_servers)

    hooks_data = raw.get("hooks", {})
    hooks = HooksConfig(**hooks_data) if hooks_data else HooksConfig()

    return NeMoCodeConfig(
        default_endpoint=raw.get("default_endpoint", "nim-super"),
        active_formation=raw.get("active_formation"),
        max_tool_rounds=raw.get("max_tool_rounds", 100),
        endpoints=endpoints,
        manifests=manifests,
        formations=formations,
        agents=agents,
        permissions=perms,
        project=project,
        mcp=mcp,
        hooks=hooks,
    )


def _apply_env_overrides(cfg: NeMoCodeConfig) -> NeMoCodeConfig:
    """Apply environment variable overrides."""
    ep = os.environ.get("NEMOCODE_ENDPOINT")
    if ep:
        cfg.default_endpoint = ep
    fm = os.environ.get("NEMOCODE_FORMATION")
    if fm:
        cfg.active_formation = fm
    return cfg


def load_config(project_dir: Path | None = None) -> NeMoCodeConfig:
    """Load and merge configuration from all sources."""
    # Layer 1: Bundled defaults
    raw = _load_yaml(_DEFAULTS_PATH)

    # Layer 2: User config
    user_raw = _load_yaml(_USER_CONFIG_PATH)
    if user_raw:
        raw = _deep_merge(raw, user_raw)

    # Layer 3: Project config
    if project_dir is None:
        project_dir = Path.cwd()
    project_path = project_dir / _PROJECT_CONFIG_NAME
    project_raw = _load_yaml(project_path)
    if project_raw:
        raw = _deep_merge(raw, project_raw)

    # Layer 3b: Agent markdown profiles
    discovered_agents = discover_agent_markdown(project_dir)
    if discovered_agents:
        raw["agents"] = _deep_merge(raw.get("agents", {}), discovered_agents)

    # Parse into typed config
    cfg = _parse_config(raw)

    # Layer 4: Environment overrides
    cfg = _apply_env_overrides(cfg)

    return cfg


def get_api_key(endpoint: Endpoint) -> str | None:
    """Resolve API key for an endpoint. Checks credentials store first, then env."""
    if endpoint.api_key_env is None:
        return None

    # Try credentials store first
    try:
        from nemocode.core.credentials import get_credential

        key = get_credential(endpoint.api_key_env)
        if key:
            return key
    except ImportError:
        pass

    # Fall back to environment variable
    return os.environ.get(endpoint.api_key_env)


def ensure_config_dir() -> Path:
    """Create and return the user config directory."""
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _USER_CONFIG_DIR
