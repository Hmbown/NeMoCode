# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for guided runtime setup."""

from __future__ import annotations

from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from nemocode.cli.main import app

runner = CliRunner()


class TestSetupWizard:
    def test_setup_wizard_configures_hosted_nim(self, tmp_path):
        user_config = tmp_path / "config.yaml"

        with (
            patch("nemocode.config._USER_CONFIG_DIR", tmp_path),
            patch("nemocode.core.setup_wizard._USER_CONFIG_PATH", user_config),
            patch("nemocode.core.setup_wizard.get_credential", return_value=None),
            patch("nemocode.core.setup_wizard.set_credential", return_value=(True, "keyring")),
        ):
            result = runner.invoke(app, ["setup", "wizard"], input="1\nnvapi-test\n1\n")

        assert result.exit_code == 0
        raw = yaml.safe_load(user_config.read_text())
        assert raw["default_endpoint"] == "nim-super"

    def test_setup_wizard_configures_sglang_nano4b(self, tmp_path):
        user_config = tmp_path / "config.yaml"

        with (
            patch("nemocode.config._USER_CONFIG_DIR", tmp_path),
            patch("nemocode.core.setup_wizard._USER_CONFIG_PATH", user_config),
        ):
            result = runner.invoke(app, ["setup", "wizard"], input="3\n3\n")

        assert result.exit_code == 0
        raw = yaml.safe_load(user_config.read_text())
        assert raw["default_endpoint"] == "local-sglang-nano4b"

    def test_setup_wizard_configures_trt_llm_nano4b(self, tmp_path):
        user_config = tmp_path / "config.yaml"

        with (
            patch("nemocode.config._USER_CONFIG_DIR", tmp_path),
            patch("nemocode.core.setup_wizard._USER_CONFIG_PATH", user_config),
        ):
            result = runner.invoke(app, ["setup", "wizard"], input="4\n2\n")

        assert result.exit_code == 0
        raw = yaml.safe_load(user_config.read_text())
        assert raw["default_endpoint"] == "local-trt-llm-nano4b"
