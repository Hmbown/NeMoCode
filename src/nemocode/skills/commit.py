# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Commit skill — generate commit message from staged changes and commit."""

from __future__ import annotations

import subprocess
from typing import Any

from nemocode.skills import Skill


class CommitSkill(Skill):
    name = "commit"
    description = "Generate a commit message from staged changes and commit"
    usage = "/commit [message override]"

    async def run(self, args: str, context: dict[str, Any]) -> str | None:
        console = context.get("console")
        agent = context.get("agent")

        # Get current git state
        diff = _get_staged_diff()
        if not diff:
            # Try unstaged diff
            diff = _get_unstaged_diff()
            if not diff:
                return "No changes to commit."

            # Stage all changes
            if console:
                console.print("[dim]No staged changes. Staging all modified files...[/dim]")
            subprocess.run(["git", "add", "-u"], capture_output=True, timeout=10)
            diff = _get_staged_diff()
            if not diff:
                return "No changes to commit."

        status = _get_status()

        if args:
            # User provided a message override
            message = args
        elif agent:
            # Use the agent to generate a commit message
            prompt = (
                "Generate a concise git commit message for these changes. "
                "Follow conventional commits format (feat:, fix:, refactor:, etc). "
                "Return ONLY the commit message, nothing else.\n\n"
                f"## Git Status\n```\n{status}\n```\n\n"
                f"## Diff\n```diff\n{diff[:8000]}\n```"
            )
            message_parts: list[str] = []
            async for event in agent.run(prompt):
                if event.kind == "text":
                    message_parts.append(event.text)
            message = "".join(message_parts).strip()
            # Clean up markdown formatting if present
            message = message.strip("`").strip()
            if message.startswith("commit"):
                message = message[6:].strip()
        else:
            return "No agent available and no message provided."

        if not message:
            return "Failed to generate commit message."

        # Show message and confirm
        if console:
            console.print(f"\n[bold]Commit message:[/bold]\n  {message}\n")
            try:
                response = console.input("[yellow]Commit? [y/N]: [/yellow]")
            except (EOFError, KeyboardInterrupt):
                return "Commit cancelled."
            if response.strip().lower() not in ("y", "yes"):
                return "Commit cancelled."

        # Execute commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return f"Committed: {message}"
        return f"Commit failed: {result.stderr.strip()}"


def _get_staged_diff() -> str:
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            r2 = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r2.stdout if r2.returncode == 0 else ""
    except Exception:
        pass
    return ""


def _get_unstaged_diff() -> str:
    try:
        r = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:
        return ""


def _get_status() -> str:
    try:
        r = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""
