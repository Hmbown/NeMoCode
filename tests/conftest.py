# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    Manifest,
    MoEConfig,
    NeMoCodeConfig,
    NemotronArch,
    ReasoningConfig,
    ToolPermissions,
)
from nemocode.core.registry import Registry
from nemocode.tools.loader import load_tools


@pytest.fixture
def sample_endpoint() -> Endpoint:
    return Endpoint(
        name="test-endpoint",
        tier=EndpointTier.DEV_HOSTED,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_id="nvidia/nemotron-3-super-120b-a12b",
        capabilities=[Capability.CHAT, Capability.CODE],
    )


@pytest.fixture
def sample_manifest() -> Manifest:
    return Manifest(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        display_name="Nemotron 3 Super",
        arch=NemotronArch.NEMOTRON_3,
        tier_in_family="super",
        capabilities=[Capability.CHAT, Capability.CODE],
        context_window=1_048_576,
        moe=MoEConfig(
            total_params_b=120,
            active_params_b=12,
            uses_latent_moe=True,
            uses_mtp=True,
            uses_mamba=True,
            precision="nvfp4",
        ),
        reasoning=ReasoningConfig(
            supports_thinking=True,
            thinking_param="enable_thinking",
            thinking_budget_param="thinking_budget",
            no_think_tag="/no_think",
            supports_budget_control=True,
        ),
        supports_tools=True,
        min_gpu_memory_gb=80,
        min_gpus=1,
    )


@pytest.fixture
def sample_config(sample_endpoint: Endpoint, sample_manifest: Manifest) -> NeMoCodeConfig:
    return NeMoCodeConfig(
        default_endpoint="test-ep",
        endpoints={"test-ep": sample_endpoint},
        manifests={sample_manifest.model_id: sample_manifest},
        formations={
            "solo": Formation(
                name="Solo",
                description="Single model",
                slots=[FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR)],
                verification_rounds=0,
            ),
        },
        permissions=ToolPermissions(),
    )


@pytest.fixture
def sample_registry(sample_config: NeMoCodeConfig) -> Registry:
    return Registry(sample_config)


@pytest.fixture
def tool_registry():
    return load_tools(["fs", "git", "rg"])


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with sample files."""
    (tmp_path / "README.md").write_text("# Test Project")
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    return tmp_path
