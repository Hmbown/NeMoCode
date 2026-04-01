# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Glob tool — fast file pattern matching."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nemocode.core.tool_cache import get_tool_cache
from nemocode.tools import tool

_MAX_RESULTS = 500


@tool(
    name="glob_files",
    description="Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts').",
    category="glob",
)
async def glob_files(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
) -> str:
    """Find files matching a glob pattern.
    pattern: Glob pattern to match (e.g. '**/*.py', 'src/*.ts').
    path: Directory to search in (default: current directory).
    max_results: Maximum number of results to return (default: 100).
    """
    base = Path(path).resolve() if os.path.isabs(path) else Path(os.getcwd(), path).resolve()
    if not base.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    if not base.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})

    max_results = min(max_results, _MAX_RESULTS)

    # Check cache
    cache = get_tool_cache()
    cache_key = f"glob:{pattern}:{base}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        matches: list[Path] = []
        for p in base.glob(pattern):
            if p.is_file():
                # Skip hidden directories and common noise
                parts = p.relative_to(base).parts
                if any(
                    part.startswith(".")
                    or part in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                    for part in parts[:-1]  # don't filter the filename itself
                ):
                    continue
                matches.append(p)
                if len(matches) >= max_results:
                    break

        if not matches:
            return json.dumps({"files": [], "count": 0})

        # Sort by modification time (most recent first)
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Return relative paths
        files = []
        for m in matches:
            try:
                rel = str(m.relative_to(base))
            except ValueError:
                rel = str(m)
            files.append(rel)

        result = {"files": files, "count": len(files)}
        if len(matches) >= max_results:
            result["truncated"] = True
        result_json = json.dumps(result)
        # Cache the result
        cache.put(cache_key, result_json, ttl=60.0)
        return result_json
    except Exception as e:
        return json.dumps({"error": str(e)})
