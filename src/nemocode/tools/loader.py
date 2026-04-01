# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tool loader — registers tools by category from formation config.

Supports lazy loading: tools are imported only when first accessed,
reducing startup time when MCP servers or heavy optional dependencies
are configured. Set ``NEMOCODE_LAZY_TOOLS=0`` to disable lazy loading.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from nemocode.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)
_PLUGIN_SUBDIR = Path(".nemocode") / "tools"

# Tool specifications: (module_path, attribute_name)
# Stored as specs so modules are not imported at startup.
_TOOL_SPECS: dict[str, tuple[str, str]] = {
    "read_file": ("nemocode.tools.fs", "read_file"),
    "write_file": ("nemocode.tools.fs", "write_file"),
    "edit_file": ("nemocode.tools.fs", "edit_file"),
    "list_dir": ("nemocode.tools.fs", "list_dir"),
    "multi_edit": ("nemocode.tools.multi_edit", "multi_edit"),
    "apply_patch": ("nemocode.tools.multi_edit", "apply_patch"),
    "bash_exec": ("nemocode.tools.bash", "bash_exec"),
    "git_status": ("nemocode.tools.git", "git_status"),
    "git_diff": ("nemocode.tools.git", "git_diff"),
    "git_log": ("nemocode.tools.git", "git_log"),
    "git_commit": ("nemocode.tools.git", "git_commit"),
    "glob_files": ("nemocode.tools.glob", "glob_files"),
    "http_fetch": ("nemocode.tools.http", "http_fetch"),
    "run_tests": ("nemocode.tools.test", "run_tests"),
    "save_memory_tool": ("nemocode.tools.memory_tool", "save_memory_tool"),
    "forget_memory_tool": ("nemocode.tools.memory_tool", "forget_memory_tool"),
    "list_memories_tool": ("nemocode.tools.memory_tool", "list_memories_tool"),
    "create_task": ("nemocode.tools.tasks", "create_task"),
    "update_task": ("nemocode.tools.tasks", "update_task"),
    "list_tasks": ("nemocode.tools.tasks", "list_tasks"),
    "web_search": ("nemocode.tools.web", "web_search"),
    "web_fetch": ("nemocode.tools.web", "web_fetch"),
    "parse_document": ("nemocode.tools.parse", "parse_document"),
    "ask_clarify": ("nemocode.tools.clarify", "ask_clarify"),
    "ask_user": ("nemocode.tools.ask_user", "ask_user"),
    "search_files": ("nemocode.tools.rg", "search_files"),
}

# LSP tool specs — loaded lazily only if the module is available.
_LSP_TOOL_SPECS: dict[str, tuple[str, str]] = {
    "lsp_diagnostics": ("nemocode.tools.lsp_tool", "lsp_diagnostics"),
    "lsp_hover": ("nemocode.tools.lsp_tool", "lsp_hover"),
    "lsp_references": ("nemocode.tools.lsp_tool", "lsp_references"),
}

_CATEGORY_TOOL_NAMES: dict[str, list[str]] = {
    "fs": ["read_file", "write_file", "edit_file", "list_dir", "multi_edit", "apply_patch"],
    "fs_read": ["read_file", "list_dir"],
    "bash": ["bash_exec"],
    "git": ["git_status", "git_diff", "git_log", "git_commit"],
    "git_read": ["git_status", "git_diff", "git_log"],
    "rg": ["search_files"],
    "glob": ["glob_files"],
    "http": ["http_fetch"],
    "test": ["run_tests"],
    "memory": ["save_memory_tool", "forget_memory_tool", "list_memories_tool"],
    "tasks": ["create_task", "update_task", "list_tasks"],
    "web": ["web_search", "web_fetch"],
    "parse": ["parse_document"],
    "clarify": ["ask_clarify", "ask_user"],
    "lsp": [],  # populated lazily when LSP tools are available
}

# Determine whether lazy loading is enabled (default: True).
_LAZY_ENABLED = os.environ.get("NEMOCODE_LAZY_TOOLS", "1") != "0"


def _lsp_available() -> bool:
    """Check whether the LSP tool module can be imported."""
    try:
        importlib.import_module("nemocode.tools.lsp_tool")
        return True
    except Exception:
        return False


if _lsp_available():
    _CATEGORY_TOOL_NAMES["lsp"] = list(_LSP_TOOL_SPECS.keys())
    _TOOL_SPECS.update(_LSP_TOOL_SPECS)


class LazyToolWrapper:
    """Wraps a tool specification and imports the actual tool on first access.

    This wrapper stores the module path and attribute name of a tool function.
    The real function is imported only when ``_tool_def`` is first accessed,
    which dramatically reduces startup time when many tools are registered
    but only a few are used in a given session.

    Attributes:
        name: The logical tool name (used for registry lookups).
        module_path: Fully-qualified module path (e.g. ``"nemocode.tools.fs"``).
        attr_name: Attribute name within the module (e.g. ``"read_file"``).
        _fn: Cached function reference after first import.
    """

    def __init__(self, name: str, module_path: str, attr_name: str) -> None:
        self.name = name
        self.module_path = module_path
        self.attr_name = attr_name
        self._fn: Callable[..., Any] | None = None

    def _load(self) -> Callable[..., Any]:
        """Import and return the wrapped tool function (cached)."""
        if self._fn is not None:
            return self._fn
        try:
            module = importlib.import_module(self.module_path)
            self._fn = getattr(module, self.attr_name)
            logger.debug(
                "Lazy-loaded tool '%s' from %s.%s", self.name, self.module_path, self.attr_name
            )
        except Exception:
            logger.exception(
                "Failed to lazy-load tool '%s' from %s.%s",
                self.name,
                self.module_path,
                self.attr_name,
            )
            raise
        return self._fn

    @property
    def _tool_def(self) -> ToolDef:
        """Return the tool definition, importing the module if needed."""
        fn = self._load()
        return fn._tool_def  # type: ignore[attr-defined]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward calls to the underlying tool function."""
        fn = self._load()
        return fn(*args, **kwargs)

    def __repr__(self) -> str:
        state = "loaded" if self._fn is not None else "pending"
        return f"<LazyToolWrapper name={self.name!r} state={state}>"


def _make_wrappers(
    tool_names: list[str],
) -> list[LazyToolWrapper]:
    """Create LazyToolWrapper instances for the given tool names."""
    wrappers: list[LazyToolWrapper] = []
    for tool_name in tool_names:
        spec = _TOOL_SPECS.get(tool_name)
        if spec is None:
            logger.warning("Unknown tool spec: %s", tool_name)
            continue
        module_path, attr_name = spec
        wrappers.append(LazyToolWrapper(tool_name, module_path, attr_name))
    return wrappers


def _load_eager_tools(tool_names: list[str]) -> list[Callable[..., Any]]:
    """Import and return tool functions eagerly (for debugging)."""
    tools: list[Callable[..., Any]] = []
    for tool_name in tool_names:
        spec = _TOOL_SPECS.get(tool_name)
        if spec is None:
            logger.warning("Unknown tool spec: %s", tool_name)
            continue
        module_path, attr_name = spec
        try:
            module = importlib.import_module(module_path)
            tools.append(getattr(module, attr_name))
        except Exception:
            logger.exception("Failed to eagerly load tool '%s'", tool_name)
    return tools


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
    """Create a ToolRegistry with tools from the specified categories.

    When ``NEMOCODE_LAZY_TOOLS`` is not ``"0"``, tools are wrapped in
    ``LazyToolWrapper`` instances and imported only on first use.  Set
    ``NEMOCODE_LAZY_TOOLS=0`` to disable lazy loading (useful for
    debugging).

    Args:
        categories: Tool category names to load.  ``None`` loads all
            categories; ``[]`` loads no built-in tools (plugins only).
        project_dir: Project root for plugin discovery.

    Returns:
        A populated ``ToolRegistry`` instance.
    """
    registry = ToolRegistry()
    cats = list(_CATEGORY_TOOL_NAMES.keys()) if categories is None else categories

    for cat in cats:
        tool_names = _CATEGORY_TOOL_NAMES.get(cat, [])
        if _LAZY_ENABLED:
            for wrapper in _make_wrappers(tool_names):
                registry.register_function(wrapper)
        else:
            for fn in _load_eager_tools(tool_names):
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
