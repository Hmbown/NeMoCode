# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Git tools: status, diff, log, commit."""

from __future__ import annotations

import asyncio
import json
import os
import shlex

from nemocode.tools import tool


async def _git(args: list[str], cwd: str = "") -> tuple[int, str, str]:
    work_dir = cwd or os.getcwd()
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=work_dir,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


@tool(name="git_status", description="Show git working tree status.", category="git")
async def git_status() -> str:
    """Show git status."""
    rc, out, err = await _git(["status", "--porcelain", "-u"])
    if rc != 0:
        return json.dumps({"error": err.strip()})
    return out if out.strip() else "(clean working tree)"


@tool(name="git_diff", description="Show git diff (staged and unstaged changes).", category="git")
async def git_diff(staged: bool = False, path: str = "") -> str:
    """Show diff.
    staged: If true, show staged changes only.
    path: Specific file path to diff.
    """
    cmd: list[str] = ["diff", "--stat"]
    if staged:
        cmd.append("--cached")
    if path:
        cmd.extend(["--", path])
    rc, out, err = await _git(cmd)
    if rc != 0:
        return json.dumps({"error": err.strip()})

    # Also get the full diff
    cmd2: list[str] = ["diff"]
    if staged:
        cmd2.append("--cached")
    if path:
        cmd2.extend(["--", path])
    _, full_diff, _ = await _git(cmd2)

    return f"{out}\n{full_diff}" if out.strip() else "(no changes)"


@tool(name="git_log", description="Show recent git commit history.", category="git")
async def git_log(count: int = 10, oneline: bool = True) -> str:
    """Show git log.
    count: Number of commits to show.
    oneline: Use one-line format.
    """
    fmt = "--oneline" if oneline else "--format=medium"
    rc, out, err = await _git(["log", fmt, "-n", str(count)])
    if rc != 0:
        return json.dumps({"error": err.strip()})
    return out if out.strip() else "(no commits)"


@tool(
    name="git_commit",
    description="Stage files and create a git commit.",
    requires_confirmation=True,
    category="git",
)
async def git_commit(message: str, files: str = ".") -> str:
    """Create a git commit.
    message: Commit message.
    files: Space-separated list of files to stage (default: all).
    """
    # Stage — split files safely
    file_list = shlex.split(files)
    rc, _, err = await _git(["add"] + file_list)
    if rc != 0:
        return json.dumps({"error": f"git add failed: {err.strip()}"})

    # Commit — pass message as a single argument (no shell splitting)
    rc, out, err = await _git(["commit", "-m", message])
    if rc != 0:
        return json.dumps({"error": err.strip()})
    return out.strip()
