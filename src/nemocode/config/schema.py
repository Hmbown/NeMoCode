# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Configuration schema for NeMoCode.

Three primitives:
  Endpoint   — where a model lives (hosted, self-hosted, local)
  Manifest   — what a model can do and its architecture-specific quirks
  Formation  — named multi-model setup mapping roles to endpoints

Built around the Nemotron 3 family (Nano / Super / Ultra), not Llama fine-tunes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class Capability(str, Enum):
    CHAT = "chat"
    CODE = "code"
    EMBED = "embed"
    RERANK = "rerank"
    PARSE = "parse"
    SPEECH = "speech"
    IMAGE = "image"
    VLM = "vlm"


# ---------------------------------------------------------------------------
# Endpoint tiers
# ---------------------------------------------------------------------------


class EndpointTier(str, Enum):
    DEV_HOSTED = "dev-hosted"
    PROD_HOSTED = "prod-hosted"
    LOCAL_NIM = "local-nim"
    LOCAL_OLLAMA = "local-ollama"
    LOCAL_SGLANG = "local-sglang"
    LOCAL_VLLM = "local-vllm"
    OPENROUTER = "openrouter"
    INFERENCE_PARTNER = "inference-partner"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


class Endpoint(BaseModel):
    """Where a model lives and how to reach it."""

    name: str
    tier: EndpointTier
    base_url: str
    api_key_env: str | None = None
    model_id: str
    capabilities: list[Capability] = Field(default_factory=lambda: [Capability.CHAT])
    max_tokens: int = 16_384
    extra_headers: dict[str, str] = Field(default_factory=dict)
    transport: Literal["openai", "grpc", "http"] = "openai"


# ---------------------------------------------------------------------------
# Manifest — per-model architecture quirks
# ---------------------------------------------------------------------------


class NemotronArch(str, Enum):
    NEMOTRON_3 = "nemotron-3"
    NEMOTRON_EMBED = "nemotron-embed"
    NEMOTRON_RERANK = "nemotron-rerank"
    NEMOTRON_PARSE = "nemotron-parse"
    NEMOTRON_VLM = "nemotron-vlm"
    COSMOS = "cosmos"
    SPEECH_ASR = "speech-asr"
    SPEECH_TTS = "speech-tts"
    GUARDRAILS = "guardrails"
    OTHER = "other"


class ReasoningConfig(BaseModel):
    supports_thinking: bool = False
    thinking_param: str | None = None
    thinking_budget_param: str | None = None
    no_think_tag: str | None = None
    supports_budget_control: bool = False
    low_effort: bool = False


class StructuredOutputConfig(BaseModel):
    json_schema: bool = False
    regex: bool = False
    grammar: bool = False
    choices: bool = False


class MoEConfig(BaseModel):
    total_params_b: float = 0
    active_params_b: float = 0
    uses_latent_moe: bool = False
    uses_mtp: bool = False
    uses_mamba: bool = False
    precision: str = "fp8"


class Manifest(BaseModel):
    """What a model can do and its architecture-specific quirks."""

    model_id: str
    display_name: str = ""
    arch: NemotronArch = NemotronArch.OTHER
    tier_in_family: str = ""
    capabilities: list[Capability] = Field(default_factory=lambda: [Capability.CHAT])
    context_window: int = 128_000
    max_output_tokens: int = 16_384
    moe: MoEConfig = Field(default_factory=MoEConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    structured: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    supports_tools: bool = True
    supports_parallel_tools: bool = False
    force_nonempty_content: bool = False
    supports_lora: bool = False
    lora_adapters: list[str] = Field(default_factory=list)
    default_extra_body: dict[str, Any] = Field(default_factory=dict)
    min_gpu_memory_gb: int | None = None
    min_gpus: int | None = None
    recommended_gpu: str = ""


# ---------------------------------------------------------------------------
# Formation roles
# ---------------------------------------------------------------------------


class FormationRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"
    FAST = "fast"
    EMBED = "embed"
    RERANK = "rerank"
    PARSE = "parse"
    VISION = "vision"
    ASR = "asr"
    TTS = "tts"


class FormationSlot(BaseModel):
    endpoint: str
    role: FormationRole
    lora_adapter: str | None = None
    system_prompt: str | None = None
    reasoning_mode: str | None = None


class Formation(BaseModel):
    name: str
    description: str = ""
    slots: list[FormationSlot]
    tools: list[str] = Field(default_factory=lambda: ["fs", "git", "bash", "rg"])
    verification_rounds: int = 1
    output_style: str | None = None


# ---------------------------------------------------------------------------
# Agent profiles
# ---------------------------------------------------------------------------


class AgentMode(str, Enum):
    PRIMARY = "primary"
    SUBAGENT = "subagent"
    ALL = "all"


class AgentConfig(BaseModel):
    """A named agent profile for primary sessions or delegated sub-agents."""

    name: str
    description: str = ""
    display_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    nickname_candidates: list[str] = Field(default_factory=list)
    mode: AgentMode = AgentMode.SUBAGENT
    hidden: bool = False
    endpoint: str | None = None
    prefer_tiers: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    role: FormationRole = FormationRole.FAST
    prompt: str = ""


# ---------------------------------------------------------------------------
# Tool permissions
# ---------------------------------------------------------------------------


class PermissionRuleConfig(BaseModel):
    """A single permission rule for fine-grained auto-approve/deny."""

    tool: str = "*"
    action: str = "allow"  # "allow" or "deny"
    conditions: dict[str, str] = Field(default_factory=dict)


class ToolPermissions(BaseModel):
    allow_shell: bool = True
    allow_file_write: bool = True
    allow_file_delete: bool = False
    auto_approve_reads: bool = True
    auto_approve_shell: bool = False
    require_confirmation: list[str] = Field(
        default_factory=lambda: ["bash_exec", "file_delete", "git_commit"]
    )
    permission_rules: list[PermissionRuleConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Project config (from .nemocode.yaml)
# ---------------------------------------------------------------------------


class ProjectContext(BaseModel):
    name: str = ""
    description: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    conventions: list[str] = Field(default_factory=list)
    context_files: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class HooksConfig(BaseModel):
    """Pre/post tool execution hooks — shell commands with template variables."""

    pre_write_file: list[str] = Field(default_factory=list)
    post_write_file: list[str] = Field(default_factory=list)
    pre_bash_exec: list[str] = Field(default_factory=list)
    post_bash_exec: list[str] = Field(default_factory=list)
    pre_git_commit: list[str] = Field(default_factory=list)
    post_git_commit: list[str] = Field(default_factory=list)
    pre_edit_file: list[str] = Field(default_factory=list)
    post_edit_file: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to flat dict for HookRunner."""
        d: dict[str, list[str]] = {}
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if val:
                d[field_name] = val
        return d


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class NeMoCodeConfig(BaseModel):
    default_endpoint: str = "nim-super"
    active_formation: str | None = None
    max_tool_rounds: int = 100
    endpoints: dict[str, Endpoint] = Field(default_factory=dict)
    manifests: dict[str, Manifest] = Field(default_factory=dict)
    formations: dict[str, Formation] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    permissions: ToolPermissions = Field(default_factory=ToolPermissions)
    project: ProjectContext = Field(default_factory=ProjectContext)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
