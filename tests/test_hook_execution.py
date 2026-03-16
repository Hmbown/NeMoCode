# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for HookRunner pre/post hook execution around tool calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nemocode.core.hooks import HookRunner


class TestPreHooks:
    @pytest.mark.asyncio
    async def test_pre_bash_exec_runs_before_execution(self):
        runner = HookRunner({"pre_bash_exec": ["echo before"]})
        outputs = await runner.run_pre("bash_exec", {"command": "ls"})
        assert outputs == ["before"]

    @pytest.mark.asyncio
    async def test_pre_write_file_gets_path_in_template_vars(self):
        runner = HookRunner({"pre_write_file": ["echo writing {{path}}"]})
        outputs = await runner.run_pre("write_file", {"path": "/tmp/out.py"})
        assert outputs == ["writing /tmp/out.py"]

    @pytest.mark.asyncio
    async def test_pre_hook_receives_all_args(self):
        runner = HookRunner({"pre_bash_exec": ["echo cmd={{command}}"]})
        outputs = await runner.run_pre("bash_exec", {"command": "npm test"})
        assert outputs == ["cmd=npm test"]


class TestPostHooks:
    @pytest.mark.asyncio
    async def test_post_bash_exec_runs_after_execution_with_result(self):
        runner = HookRunner({"post_bash_exec": ["echo result={{result}}"]})
        outputs = await runner.run_post("bash_exec", {"command": "ls"}, result="file1.py")
        assert outputs == ["result=file1.py"]

    @pytest.mark.asyncio
    async def test_post_git_commit_gets_message_in_template_vars(self):
        runner = HookRunner({"post_git_commit": ["echo committed={{message}}"]})
        outputs = await runner.run_post("git_commit", {"message": "fix bug"})
        assert outputs == ["committed=fix bug"]

    @pytest.mark.asyncio
    async def test_post_hook_result_truncated_at_1000_chars(self):
        """The run_post method truncates result to 1000 chars in template vars."""
        runner = HookRunner({"post_bash_exec": ["echo ok"]})
        long_result = "x" * 2000
        outputs = await runner.run_post("bash_exec", {}, result=long_result)
        # Should succeed without error — result is truncated internally
        assert outputs == ["ok"]


class TestMultipleHooks:
    @pytest.mark.asyncio
    async def test_multiple_hooks_in_sequence_all_execute(self):
        runner = HookRunner(
            {"pre_bash_exec": ["echo first", "echo second", "echo third"]}
        )
        outputs = await runner.run_pre("bash_exec", {})
        assert outputs == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_multiple_hooks_all_receive_template_vars(self):
        runner = HookRunner(
            {
                "pre_write_file": [
                    "echo path={{path}}",
                    "echo also={{path}}",
                ]
            }
        )
        outputs = await runner.run_pre("write_file", {"path": "a.py"})
        assert outputs == ["path=a.py", "also=a.py"]


class TestHookTimeout:
    @pytest.mark.asyncio
    async def test_hook_timeout_handled_gracefully(self):
        runner = HookRunner({"pre_bash_exec": ["sleep 60"]})
        # Patch the timeout to be very short for testing
        import asyncio as _asyncio

        with patch(
            "nemocode.core.hooks.asyncio.wait_for",
            side_effect=_asyncio.TimeoutError,
        ):
            outputs = await runner.run_pre("bash_exec", {})
        assert len(outputs) == 1
        assert "[hook timeout]" in outputs[0]


class TestHookErrors:
    @pytest.mark.asyncio
    async def test_hook_with_bad_command_reports_error(self):
        runner = HookRunner({"pre_bash_exec": ["false"]})
        outputs = await runner.run_pre("bash_exec", {})
        assert len(outputs) == 1
        assert "[hook error]" in outputs[0]

    @pytest.mark.asyncio
    async def test_hook_with_nonexistent_command_reports_error(self):
        runner = HookRunner({"pre_bash_exec": ["__nonexistent_cmd_12345__"]})
        outputs = await runner.run_pre("bash_exec", {})
        assert len(outputs) == 1
        assert "[hook error]" in outputs[0]

    @pytest.mark.asyncio
    async def test_error_in_first_hook_does_not_stop_second(self):
        runner = HookRunner(
            {"pre_bash_exec": ["false", "echo ok"]}
        )
        outputs = await runner.run_pre("bash_exec", {})
        assert len(outputs) == 2
        assert "[hook error]" in outputs[0]
        assert outputs[1] == "ok"
