# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the rule-based permission system."""

from __future__ import annotations

from nemocode.core.permissions import PermissionEngine, PermissionRule


class TestPermissionRule:
    def test_exact_tool_match(self):
        rule = PermissionRule(tool="bash_exec", action="allow")
        assert rule.matches("bash_exec", {}) is True
        assert rule.matches("read_file", {}) is False

    def test_glob_tool_match(self):
        rule = PermissionRule(tool="git_*", action="allow")
        assert rule.matches("git_status", {}) is True
        assert rule.matches("git_commit", {}) is True
        assert rule.matches("bash_exec", {}) is False

    def test_condition_glob_match(self):
        rule = PermissionRule(
            tool="bash_exec",
            action="allow",
            conditions={"command": "pytest *"},
        )
        assert rule.matches("bash_exec", {"command": "pytest tests/"}) is True
        assert rule.matches("bash_exec", {"command": "rm -rf /"}) is False

    def test_condition_regex_match(self):
        rule = PermissionRule(
            tool="bash_exec",
            action="deny",
            conditions={"command": "re:rm\\s+-rf"},
        )
        assert rule.matches("bash_exec", {"command": "rm -rf /tmp"}) is True
        assert rule.matches("bash_exec", {"command": "pytest"}) is False

    def test_multiple_conditions(self):
        rule = PermissionRule(
            tool="write_file",
            action="allow",
            conditions={"path": "src/*"},
        )
        assert rule.matches("write_file", {"path": "src/main.py"}) is True
        assert rule.matches("write_file", {"path": "tests/test.py"}) is False


class TestPermissionEngine:
    def test_no_rules_returns_none(self):
        engine = PermissionEngine()
        assert engine.should_auto_approve("bash_exec", {}) is None

    def test_allow_rule(self):
        engine = PermissionEngine([
            PermissionRule(tool="bash_exec", action="allow", conditions={"command": "pytest *"}),
        ])
        assert engine.should_auto_approve("bash_exec", {"command": "pytest tests/"}) is True
        assert engine.should_auto_approve("bash_exec", {"command": "rm -rf"}) is None

    def test_deny_rule(self):
        engine = PermissionEngine([
            PermissionRule(tool="git_commit", action="deny"),
        ])
        assert engine.should_auto_approve("git_commit", {}) is False

    def test_from_config(self):
        config = [
            {"tool": "bash_exec", "action": "allow", "conditions": {"command": "ruff *"}},
            {"tool": "write_file", "action": "allow", "conditions": {"path": "src/*"}},
        ]
        engine = PermissionEngine.from_config(config)
        assert engine.should_auto_approve("bash_exec", {"command": "ruff check ."}) is True
        assert engine.should_auto_approve("write_file", {"path": "src/foo.py"}) is True
        assert engine.should_auto_approve("write_file", {"path": "/etc/passwd"}) is None
