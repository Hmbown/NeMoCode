# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Review skill — analyze changes for issues."""

from __future__ import annotations

import subprocess
from typing import Any

from nemocode.skills import Skill


class ReviewSkill(Skill):
    name = "review"
    description = "Review uncommitted or staged changes for issues"
    usage = "/review [file_or_ref]"

    async def run(self, args: str, context: dict[str, Any]) -> str | None:
        console = context.get("console")
        agent = context.get("agent")

        if not agent:
            return "No agent available for review."

        # Get the diff to review
        if args:
            diff = _get_diff_for(args)
        else:
            diff = _get_working_diff()

        if not diff:
            return "No changes to review."

        prompt = (
            "Review the following code changes. Check for:\n"
            "- Correctness and logic errors\n"
            "- Edge cases and error handling\n"
            "- Security issues (injection, data exposure)\n"
            "- Performance concerns\n"
            "- Style and readability\n\n"
            "Be specific about line numbers and provide actionable suggestions.\n\n"
            f"```diff\n{diff[:12000]}\n```"
        )

        # Stream the review output
        if console:
            console.print("\n[bold blue]--- Code Review ---[/bold blue]")

        output_parts: list[str] = []
        async for event in agent.run(prompt):
            if event.kind == "text":
                output_parts.append(event.text)

        return None  # Output is streamed by the agent


def _get_working_diff() -> str:
    """Get combined staged + unstaged diff."""
    parts = []
    for cmd in (["git", "diff", "--cached"], ["git", "diff"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                parts.append(r.stdout)
        except Exception:
            pass
    return "\n".join(parts)


def _get_diff_for(ref: str) -> str:
    """Get diff for a specific file or git ref."""
    try:
        # Try as a file path first
        r = subprocess.run(
            ["git", "diff", "--", ref],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout

        # Try as a git ref (e.g. HEAD~1)
        r = subprocess.run(
            ["git", "diff", ref],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    except Exception:
        pass
    return ""
