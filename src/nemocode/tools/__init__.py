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


def _unwrap_optional(hint: Any) -> Any:
    """Return the inner type of Optional[X] / Union[X, None]; else hint itself."""
    origin = getattr(hint, "__origin__", None)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        if non_none:
            return non_none[0]
    return hint


def _coerce_arg(value: Any, hint: Any) -> Any:
    """Coerce a tool argument value to match its declared Python type hint.

    Models occasionally serialize tool arguments with string-typed numbers
    (e.g. `max_results="50"`) or vice versa. We coerce defensively at the
    dispatch boundary so a single mistyped arg doesn't crash the tool.
    Returns the value unchanged on any conversion failure.
    """
    if value is None:
        return None

    target = _unwrap_optional(hint)
    origin = getattr(target, "__origin__", None)

    try:
        if target is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "yes", "1", "on"):
                    return True
                if low in ("false", "no", "0", "off", ""):
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return value
        if target is int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                return int(value.strip())
            return value
        if target is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
            return value
        if target is str:
            if isinstance(value, str):
                return value
            return str(value)
        if origin is list or target is list:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    return json.loads(stripped)
                return [value]
            return value
        if origin is dict or target is dict:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    return json.loads(stripped)
            return value
    except (ValueError, TypeError, json.JSONDecodeError):
        return value
    return value


def _coerce_args(fn: Callable, arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce each kwarg in `arguments` to match `fn`'s type hints."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in sig.parameters:
            hint = hints.get(key)
            out[key] = _coerce_arg(value, hint) if hint is not None else value
        else:
            out[key] = value
    return out


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
            coerced = _coerce_args(td.fn, arguments)
            result = await td.fn(**coerced)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(e)})

    def by_category(self, category: str) -> list[ToolDef]:
        return [td for td in self._tools.values() if td.category == category]
