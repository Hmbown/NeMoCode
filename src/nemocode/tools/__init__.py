# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tool registry and execution framework.

Provides @tool decorator, schema generation, and execution dispatch.
"""

from __future__ import annotations

import inspect
import json
import logging
import types
import typing
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union, get_type_hints

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[str]]
    requires_confirmation: bool = False
    category: str = ""


# Python type → JSON Schema type
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _resolve_json_type(hint: Any) -> str:
    """Map a Python type hint to a JSON Schema type string.

    Correctly handles Optional[X] (Union[X, None]) by extracting
    the inner type X rather than falling back to 'string'.
    """
    origin = getattr(hint, "__origin__", None)

    # Handle Union types (including Optional[X] which is Union[X, None])
    if origin is Union or origin is types.UnionType:
        args = typing.get_args(hint)
        # Filter out NoneType to find the real type
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            # Use the first non-None type
            return _resolve_json_type(non_none[0])
        return "string"

    # Handle generic types like list[X], dict[X, Y]
    if origin is not None:
        return _TYPE_MAP.get(origin, "string")

    # Handle plain types
    return _TYPE_MAP.get(hint, "string")


def _build_schema(fn: Callable) -> dict[str, Any]:
    """Generate a JSON Schema for function parameters from type hints."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        hint = hints.get(name, str)
        json_type = _resolve_json_type(hint)

        prop: dict[str, Any] = {"type": json_type}

        # Extract description from docstring if available
        doc = fn.__doc__ or ""
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{name}:") or stripped.startswith(f":param {name}:"):
                prop["description"] = stripped.split(":", 2)[-1].strip()
                break

        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    name: str | None = None,
    description: str | None = None,
    requires_confirmation: bool = False,
    category: str = "",
) -> Callable:
    """Decorator to register a function as a tool."""

    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        schema = _build_schema(fn)
        fn._tool_def = ToolDef(
            name=tool_name,
            description=tool_desc,
            parameters=schema,
            fn=fn,
            requires_confirmation=requires_confirmation,
            category=category,
        )
        return fn

    return decorator


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool_def: ToolDef) -> None:
        self._tools[tool_def.name] = tool_def

    def register_function(self, fn: Callable) -> None:
        if hasattr(fn, "_tool_def"):
            self.register(fn._tool_def)
        else:
            raise ValueError(f"{fn.__name__} is not decorated with @tool")

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-format tool schemas for all registered tools."""
        schemas = []
        for td in self._tools.values():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters,
                    },
                }
            )
        return schemas

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        td = self._tools.get(name)
        if td is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = await td.fn(**arguments)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(e)})

    def by_category(self, category: str) -> list[ToolDef]:
        return [td for td in self._tools.values() if td.category == category]
