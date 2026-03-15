# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for hooks system."""

from __future__ import annotations

import pytest

from nemocode.core.hooks import HookRunner, _safe_template


class TestSafeTemplate:
    def test_basic_substitution(self):
        result = _safe_template("echo {{path}}", {"path": "/tmp/test"})
        assert result == "echo /tmp/test"

    def test_unknown_variable_left_alone(self):
        result = _safe_template("echo {{unknown}}", {"path": "/tmp"})
        assert result == "echo {{unknown}}"

    def test_multiple_vars(self):
        result = _safe_template("{{a}} and {{b}}", {"a": "1", "b": "2"})
        assert result == "1 and 2"


class TestHookRunner:
    def test_not_enabled_when_empty(self):
        runner = HookRunner({})
        assert runner.enabled is False

    def test_enabled_with_hooks(self):
        runner = HookRunner({"pre_bash_exec": ["echo test"]})
        assert runner.enabled is True

    @pytest.mark.asyncio
    async def test_run_pre_echo(self):
        runner = HookRunner({"pre_bash_exec": ["echo hello"]})
        outputs = await runner.run_pre("bash_exec", {})
        assert outputs == ["hello"]

    @pytest.mark.asyncio
    async def test_run_pre_no_matching_hook(self):
        runner = HookRunner({"pre_bash_exec": ["echo hello"]})
        outputs = await runner.run_pre("read_file", {})
        assert outputs == []

    @pytest.mark.asyncio
    async def test_run_post_with_template(self):
        runner = HookRunner({"post_write_file": ["echo wrote {{path}}"]})
        outputs = await runner.run_post("write_file", {"path": "/tmp/test.py"})
        assert outputs == ["wrote /tmp/test.py"]

    @pytest.mark.asyncio
    async def test_hook_failure_captured(self):
        runner = HookRunner({"pre_bash_exec": ["false"]})
        outputs = await runner.run_pre("bash_exec", {})
        assert len(outputs) == 1
        assert "[hook error]" in outputs[0]
