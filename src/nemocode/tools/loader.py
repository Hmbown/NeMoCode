# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tool loader — registers tools by category from formation config."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from nemocode.tools import ToolRegistry
from nemocode.tools.ask_user import ask_user
from nemocode.tools.bash import bash_exec
from nemocode.tools.clarify import ask_clarify
from nemocode.tools.fs import edit_file, list_dir, read_file, write_file
from nemocode.tools.git import git_commit, git_diff, git_log, git_status
from nemocode.tools.glob import glob_files
from nemocode.tools.http import http_fetch
from nemocode.tools.memory_tool import forget_memory_tool, list_memories_tool, save_memory_tool
from nemocode.tools.multi_edit import apply_patch, multi_edit
from nemocode.tools.parse import parse_document
from nemocode.tools.rg import search_files
from nemocode.tools.tasks import create_task, list_tasks, update_task
from nemocode.tools.test import run_tests
from nemocode.tools.web import web_fetch, web_search

logger = logging.getLogger(__name__)
_PLUGIN_SUBDIR = Path(".nemocode") / "tools"

_CATEGORY_MAP: dict[str, list] = {
    "fs": [read_file, write_file, edit_file, list_dir, multi_edit, apply_patch],
    "fs_read": [read_file, list_dir],  # read-only subset for plan mode
    "bash": [bash_exec],
    "git": [git_status, git_diff, git_log, git_commit],
    "git_read": [git_status, git_diff, git_log],  # read-only subset
    "rg": [search_files],
    "glob": [glob_files],
    "http": [http_fetch],
    "test": [run_tests],
    "memory": [save_memory_tool, forget_memory_tool, list_memories_tool],
    "tasks": [create_task, update_task, list_tasks],
    "web": [web_search, web_fetch],
    "parse": [parse_document],
    "clarify": [ask_clarify, ask_user],
    "lsp": [],  # populated lazily when LSP tools are available
}


def _try_load_lsp_tools() -> list:
    """Load LSP tools if available. Returns empty list on import failure."""
    try:
        from nemocode.tools.lsp_tool import lsp_diagnostics, lsp_hover, lsp_references

        return [lsp_diagnostics, lsp_hover, lsp_references]
    except Exception:
        return []


_CATEGORY_MAP["lsp"] = _try_load_lsp_tools()


def _load_plugin_functions(project_dir: Path) -> list:
    """Load decorated tool functions from .nemocode/tools/*.py."""
    plugin_dir = project_dir / _PLUGIN_SUBDIR
    if not plugin_dir.is_dir():
        return []

    functions = []
    for path in sorted(plugin_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module_name = f"_nemocode_plugin_{abs(hash(path.resolve())):x}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.debug("Skipping plugin with no import spec: %s", path)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                sys.path.insert(0, str(path.parent))
                spec.loader.exec_module(module)
            finally:
                if sys.path and sys.path[0] == str(path.parent):
                    sys.path.pop(0)
            for value in vars(module).values():
                if callable(value) and hasattr(value, "_tool_def"):
                    functions.append(value)
        except Exception:
            logger.exception("Failed to load plugin tools from %s", path)
    return functions


def load_tools(
    categories: list[str] | None = None,
    project_dir: str | Path | None = None,
) -> ToolRegistry:
    """Create a ToolRegistry with tools from the specified categories."""
    registry = ToolRegistry()
    cats = list(_CATEGORY_MAP.keys()) if categories is None else categories
    for cat in cats:
        fns = _CATEGORY_MAP.get(cat, [])
        for fn in fns:
            registry.register_function(fn)

    if categories == []:
        return registry

    root = Path(project_dir).resolve() if project_dir is not None else Path.cwd().resolve()
    plugin_functions = _load_plugin_functions(root)
    for fn in plugin_functions:
        tool_def = fn._tool_def
        if categories is not None and tool_def.category not in cats and tool_def.name not in cats:
            continue
        if registry.get(tool_def.name) is not None:
            logger.warning(
                "Skipping plugin tool '%s' because a tool with that name already exists",
                tool_def.name,
            )
            continue
        registry.register_function(fn)
    return registry
