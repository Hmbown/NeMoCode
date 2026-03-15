# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tool loader — registers tools by category from formation config."""

from __future__ import annotations

from nemocode.tools import ToolRegistry
from nemocode.tools.bash import bash_exec
from nemocode.tools.fs import edit_file, list_dir, read_file, write_file
from nemocode.tools.git import git_commit, git_diff, git_log, git_status
from nemocode.tools.http import http_fetch
from nemocode.tools.rg import search_files

_CATEGORY_MAP: dict[str, list] = {
    "fs": [read_file, write_file, edit_file, list_dir],
    "bash": [bash_exec],
    "git": [git_status, git_diff, git_log, git_commit],
    "rg": [search_files],
    "http": [http_fetch],
}


def load_tools(categories: list[str] | None = None) -> ToolRegistry:
    """Create a ToolRegistry with tools from the specified categories."""
    registry = ToolRegistry()
    cats = categories or list(_CATEGORY_MAP.keys())
    for cat in cats:
        fns = _CATEGORY_MAP.get(cat, [])
        for fn in fns:
            registry.register_function(fn)
    return registry
