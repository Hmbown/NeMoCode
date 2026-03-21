# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test configuration loading, merging, and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from nemocode.config import _deep_merge, _parse_config, load_config
from nemocode.config.agents import resolve_agent_reference
from nemocode.config.schema import AgentMode, Capability, EndpointTier, FormationRole


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_override_non_dict(self):
        base = {"a": {"x": 1}}
        override = {"a": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"a": "replaced"}


class TestParseConfig:
    def test_minimal_config(self):
        raw = {"default_endpoint": "nim-super"}
        cfg = _parse_config(raw)
        assert cfg.default_endpoint == "nim-super"
        assert cfg.endpoints == {}

    def test_endpoint_parsing(self):
        raw = {
            "endpoints": {
                "test": {
                    "name": "Test",
                    "tier": "dev-hosted",
                    "base_url": "https://example.com/v1",
                    "model_id": "test-model",
                    "capabilities": ["chat", "code"],
                }
            }
        }
        cfg = _parse_config(raw)
        assert "test" in cfg.endpoints
        ep = cfg.endpoints["test"]
        assert ep.tier == EndpointTier.DEV_HOSTED
        assert Capability.CODE in ep.capabilities

    def test_local_trt_llm_endpoint_parsing(self):
        raw = {
            "endpoints": {
                "trt": {
                    "name": "TensorRT-LLM",
                    "tier": "local-trt-llm",
                    "base_url": "http://localhost:8000/v1",
                    "model_id": "nvidia/nemotron-3-super-120b-a12b",
                }
            }
        }
        cfg = _parse_config(raw)
        assert cfg.endpoints["trt"].tier == EndpointTier.LOCAL_TRT_LLM

    def test_manifest_parsing(self):
        raw = {
            "manifests": {
                "nvidia/test-model": {
                    "display_name": "Test Model",
                    "arch": "nemotron-3",
                    "context_window": 1048576,
                    "moe": {"total_params_b": 120, "active_params_b": 12},
                    "reasoning": {"supports_thinking": True, "thinking_param": "enable_thinking"},
                }
            }
        }
        cfg = _parse_config(raw)
        m = cfg.manifests["nvidia/test-model"]
        assert m.context_window == 1_048_576
        assert m.moe.total_params_b == 120
        assert m.reasoning.supports_thinking is True

    def test_formation_parsing(self):
        raw = {
            "formations": {
                "solo": {
                    "name": "Solo",
                    "slots": [{"role": "executor", "endpoint": "nim-super"}],
                    "tools": ["fs", "bash"],
                    "verification_rounds": 0,
                }
            }
        }
        cfg = _parse_config(raw)
        f = cfg.formations["solo"]
        assert f.name == "Solo"
        assert len(f.slots) == 1
        assert f.slots[0].role.value == "executor"

    def test_agent_parsing(self):
        raw = {
            "agents": {
                "reviewer": {
                    "description": "Review changes",
                    "display_name": "Lint Eastwood",
                    "aliases": ["critic"],
                    "nickname_candidates": ["Sheriff Diff"],
                    "mode": "subagent",
                    "role": "reviewer",
                    "prefer_tiers": ["super"],
                    "tools": ["fs_read", "git_read"],
                }
            }
        }
        cfg = _parse_config(raw)
        agent = cfg.agents["reviewer"]
        assert agent.mode == AgentMode.SUBAGENT
        assert agent.role == FormationRole.REVIEWER
        assert agent.display_name == "Lint Eastwood"
        assert agent.aliases == ["critic"]
        assert agent.nickname_candidates == ["Sheriff Diff"]
        assert agent.prefer_tiers == ["super"]
        assert agent.tools == ["fs_read", "git_read"]

    def test_agent_reference_resolution(self):
        cfg = _parse_config({})
        assert resolve_agent_reference(cfg.agents, "builder") == "build"
        assert resolve_agent_reference(cfg.agents, "NeMoCode") == "build"
        assert resolve_agent_reference(cfg.agents, "planner") == "plan"
        assert resolve_agent_reference(cfg.agents, "Joe Nemotron") == "general"
        assert resolve_agent_reference(cfg.agents, "missing") is None

    def test_max_tool_rounds_and_hooks_are_parsed(self):
        raw = {
            "max_tool_rounds": 17,
            "hooks": {
                "post_write_file": ["ruff format {path}"],
            },
        }
        cfg = _parse_config(raw)
        assert cfg.max_tool_rounds == 17
        assert cfg.hooks.post_write_file == ["ruff format {path}"]


class TestEnvOverrides:
    def test_endpoint_override(self):
        raw = {"default_endpoint": "nim-super"}
        with patch.dict(os.environ, {"NEMOCODE_ENDPOINT": "nim-nano"}):
            cfg = _parse_config(raw)
            from nemocode.config import _apply_env_overrides

            cfg = _apply_env_overrides(cfg)
            assert cfg.default_endpoint == "nim-nano"

    def test_formation_override(self):
        raw = {}
        with patch.dict(os.environ, {"NEMOCODE_FORMATION": "super-nano"}):
            cfg = _parse_config(raw)
            from nemocode.config import _apply_env_overrides

            cfg = _apply_env_overrides(cfg)
            assert cfg.active_formation == "super-nano"


class TestProjectConfig:
    def test_load_project_config(self, tmp_path: Path):
        project_config = {
            "project": {
                "name": "TestProject",
                "tech_stack": ["python"],
                "conventions": ["Use ruff"],
            },
            "default_endpoint": "nim-nano",
        }
        (tmp_path / ".nemocode.yaml").write_text(yaml.dump(project_config))

        # Point defaults to a non-existent file so only project config loads
        with patch("nemocode.config._DEFAULTS_PATH", tmp_path / "nonexistent.yaml"):
            with patch("nemocode.config._USER_CONFIG_PATH", tmp_path / "nonexistent2.yaml"):
                cfg = load_config(tmp_path)
                assert cfg.project.name == "TestProject"
                assert cfg.default_endpoint == "nim-nano"

    def test_directory_config_overrides_project_config(self, tmp_path: Path):
        nested_dir = tmp_path / "apps" / "api"
        nested_dir.mkdir(parents=True)

        (tmp_path / ".nemocode.yaml").write_text(
            yaml.dump(
                {
                    "default_endpoint": "project-endpoint",
                    "max_tool_rounds": 12,
                    "project": {
                        "name": "RootProject",
                        "description": "project-level",
                    },
                }
            )
        )
        (nested_dir / ".nemocode").mkdir()
        (nested_dir / ".nemocode" / "config.yaml").write_text(
            yaml.dump(
                {
                    "default_endpoint": "dir-endpoint",
                    "project": {
                        "description": "directory-level",
                        "conventions": ["Keep handlers thin"],
                    },
                }
            )
        )

        with (
            patch("nemocode.config._DEFAULTS_PATH", tmp_path / "nonexistent.yaml"),
            patch("nemocode.config._USER_CONFIG_PATH", tmp_path / "nonexistent2.yaml"),
        ):
            cfg = load_config(nested_dir)

        assert cfg.default_endpoint == "dir-endpoint"
        assert cfg.max_tool_rounds == 12
        assert cfg.project.name == "RootProject"
        assert cfg.project.description == "directory-level"
        assert cfg.project.conventions == ["Keep handlers thin"]

    def test_directory_configs_merge_from_project_root_to_cwd(self, tmp_path: Path):
        package_dir = tmp_path / "packages"
        feature_dir = package_dir / "sdk"
        feature_dir.mkdir(parents=True)

        (tmp_path / ".nemocode.yaml").write_text(
            yaml.dump(
                {
                    "project": {
                        "name": "RootProject",
                    },
                }
            )
        )
        (tmp_path / ".nemocode").mkdir()
        (tmp_path / ".nemocode" / "config.yaml").write_text(
            yaml.dump(
                {
                    "max_tool_rounds": 9,
                    "project": {
                        "description": "root directory config",
                    },
                }
            )
        )
        (package_dir / ".nemocode").mkdir()
        (package_dir / ".nemocode" / "config.yaml").write_text(
            yaml.dump(
                {
                    "project": {
                        "tech_stack": ["python"],
                    },
                }
            )
        )
        (feature_dir / ".nemocode").mkdir()
        (feature_dir / ".nemocode" / "config.yaml").write_text(
            yaml.dump(
                {
                    "project": {
                        "conventions": ["Run tests locally"],
                    },
                }
            )
        )

        with (
            patch("nemocode.config._DEFAULTS_PATH", tmp_path / "nonexistent.yaml"),
            patch("nemocode.config._USER_CONFIG_PATH", tmp_path / "nonexistent2.yaml"),
        ):
            cfg = load_config(feature_dir)

        assert cfg.max_tool_rounds == 9
        assert cfg.project.name == "RootProject"
        assert cfg.project.description == "root directory config"
        assert cfg.project.tech_stack == ["python"]
        assert cfg.project.conventions == ["Run tests locally"]

    def test_load_config_configures_default_token_counter_model(self, tmp_path: Path):
        (tmp_path / ".nemocode.yaml").write_text(
            yaml.dump(
                {
                    "default_endpoint": "local-dev",
                    "endpoints": {
                        "local-dev": {
                            "tier": "dev-hosted",
                            "base_url": "https://example.com/v1",
                            "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
                        }
                    },
                }
            )
        )

        with (
            patch("nemocode.config._DEFAULTS_PATH", tmp_path / "nonexistent.yaml"),
            patch("nemocode.config._USER_CONFIG_PATH", tmp_path / "nonexistent2.yaml"),
            patch("nemocode.config.configure_token_counting") as configure_token_counting,
        ):
            load_config(tmp_path)

        configure_token_counting.assert_called_once_with("nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8")

    def test_load_project_agent_markdown(self, tmp_path: Path):
        agent_dir = tmp_path / ".nemocode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "review.md").write_text(
            "---\n"
            "description: Review code\n"
            "mode: subagent\n"
            "role: reviewer\n"
            "prefer_tiers:\n"
            "  - super\n"
            "tools:\n"
            "  - fs_read\n"
            "  - git_read\n"
            "---\n\n"
            "Inspect the diff and highlight regressions.\n"
        )

        config_root = tmp_path / "config-root"
        config_root.mkdir()

        with (
            patch("nemocode.config._DEFAULTS_PATH", tmp_path / "nonexistent.yaml"),
            patch("nemocode.config._USER_CONFIG_PATH", tmp_path / "nonexistent2.yaml"),
            patch.dict(os.environ, {"NEMOCODE_CONFIG_DIR": str(config_root)}),
        ):
            cfg = load_config(tmp_path)

        agent = cfg.agents["review"]
        assert agent.mode == AgentMode.SUBAGENT
        assert agent.role == FormationRole.REVIEWER
        assert agent.prefer_tiers == ["super"]
        assert "Inspect the diff" in agent.prompt

    def test_load_project_agent_markdown_from_nested_directory(self, tmp_path: Path):
        agent_dir = tmp_path / ".nemocode" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "review.md").write_text(
            "---\n"
            "description: Review code\n"
            "mode: subagent\n"
            "role: reviewer\n"
            "---\n\n"
            "Inspect nested project diffs.\n"
        )
        nested_dir = tmp_path / "services" / "api"
        nested_dir.mkdir(parents=True)
        (tmp_path / ".nemocode.yaml").write_text(yaml.dump({"project": {"name": "Nested"}}))

        with (
            patch("nemocode.config._DEFAULTS_PATH", tmp_path / "nonexistent.yaml"),
            patch("nemocode.config._USER_CONFIG_PATH", tmp_path / "nonexistent2.yaml"),
        ):
            cfg = load_config(nested_dir)

        assert "review" in cfg.agents
        assert "Inspect nested project diffs." in cfg.agents["review"].prompt
