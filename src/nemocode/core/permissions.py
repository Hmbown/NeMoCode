# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Rule-based permission system.

Allows fine-grained auto-approve rules so users can whitelist
specific tool invocations (e.g. 'bash_exec with pytest *' or
'write_file within src/') without blanket auto-approve.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PermissionRule:
    """A single permission rule.

    tool: Tool name to match (exact or glob, e.g. 'bash_exec', 'git_*')
    action: 'allow' or 'deny'
    conditions: Dict of arg_name -> pattern pairs. Patterns support
                glob matching and regex (prefixed with 're:').
    """

    tool: str
    action: str = "allow"  # "allow" or "deny"
    conditions: dict[str, str] = field(default_factory=dict)

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this rule matches a tool call."""
        # Match tool name (support glob)
        if not fnmatch.fnmatch(tool_name, self.tool):
            return False

        # Check all conditions
        for arg_name, pattern in self.conditions.items():
            arg_value = str(args.get(arg_name, ""))

            if pattern.startswith("re:"):
                # Regex match
                regex = pattern[3:]
                if not re.search(regex, arg_value):
                    return False
            else:
                # Glob match
                if not fnmatch.fnmatch(arg_value, pattern):
                    return False

        return True


class PermissionEngine:
    """Evaluates permission rules to decide auto-approve vs confirm."""

    def __init__(self, rules: list[PermissionRule] | None = None) -> None:
        self._rules = rules or []

    def add_rule(self, rule: PermissionRule) -> None:
        self._rules.append(rule)

    def should_auto_approve(self, tool_name: str, args: dict[str, Any]) -> bool | None:
        """Check if a tool call should be auto-approved.

        Returns:
            True: auto-approve (rule matched with action='allow')
            False: deny (rule matched with action='deny')
            None: no matching rule — fall through to default behavior
        """
        for rule in self._rules:
            if rule.matches(tool_name, args):
                logger.debug(
                    "Permission rule matched: %s %s -> %s",
                    rule.tool,
                    rule.conditions,
                    rule.action,
                )
                return rule.action == "allow"
        return None

    @classmethod
    def from_config(cls, rules_config: list[dict[str, Any]]) -> PermissionEngine:
        """Create from config dict list.

        Config format:
        ```yaml
        permission_rules:
          - tool: bash_exec
            action: allow
            conditions:
              command: "pytest *"
          - tool: bash_exec
            action: allow
            conditions:
              command: "ruff *"
          - tool: write_file
            action: allow
            conditions:
              path: "src/*"
          - tool: git_*
            action: deny
            conditions:
              command: "re:push|force"
        ```
        """
        rules = []
        for r in rules_config:
            rules.append(
                PermissionRule(
                    tool=r.get("tool", "*"),
                    action=r.get("action", "allow"),
                    conditions=r.get("conditions", {}),
                )
            )
        return cls(rules)
