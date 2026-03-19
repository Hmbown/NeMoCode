# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Search tool: ripgrep with grep fallback."""

from __future__ import annotations

import asyncio
import json
import os
import shutil

from nemocode.tools import tool

_MAX_RESULTS = 200


@tool(name="search_files", description="Search file contents using regex patterns.", category="rg")
async def search_files(
    pattern: str = "",
    text: str = "",
    path: str = ".",
    glob: str = "",
    max_results: int = 50,
    context: int = 0,
) -> str:
    """Search files for a regex pattern.
    pattern: Regex pattern to search for.
    text: Alias for pattern used by some model-generated tool calls.
    path: Directory to search in.
    glob: File glob filter (e.g. '*.py').
    max_results: Maximum number of matches to return.
    context: Number of context lines around each match.
    """
    pattern = pattern or text
    if not pattern:
        return json.dumps({"error": "pattern is required"})

    work_dir = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    max_results = min(max_results, _MAX_RESULTS)

    # Try ripgrep first, then grep
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--no-heading", "--line-number", "--color=never", f"--max-count={max_results}"]
        if glob:
            cmd.extend(["--glob", glob])
        if context > 0:
            cmd.extend([f"-C{context}"])
        cmd.extend([pattern, work_dir])
    else:
        cmd = ["grep", "-rn", "--color=never"]
        if glob:
            cmd.extend(["--include", glob])
        if context > 0:
            cmd.extend([f"-C{context}"])
        cmd.extend([pattern, work_dir])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace")
        if not output.strip():
            return "(no matches)"
        # Truncate if needed
        lines = output.splitlines()
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"... (truncated to {max_results} results)")
        return "\n".join(lines)
    except asyncio.TimeoutError:
        return json.dumps({"error": "Search timed out after 30s"})
    except Exception as e:
        return json.dumps({"error": str(e)})
