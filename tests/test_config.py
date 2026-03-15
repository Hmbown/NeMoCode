# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test configuration loading, merging, and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from nemocode.config import _deep_merge, _parse_config, load_config
from nemocode.config.schema import Capability, EndpointTier


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
