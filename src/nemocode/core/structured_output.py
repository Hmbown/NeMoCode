# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Structured output helpers — capability gating and response_format builders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nemocode.config.schema import Manifest, StructuredOutputConfig

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """Raised when a requested structured output mode is unsupported."""


def check_structured_output_support(
    manifest: Manifest | None,
    response_format: dict[str, Any],
) -> str | None:
    """Validate that *manifest* supports the requested *response_format*.

    Returns ``None`` if the mode is supported (or cannot be checked because
    there is no manifest).  Returns a human-readable warning string otherwise.
    """
    if manifest is None:
        return None  # No manifest — cannot gate

    fmt_type = response_format.get("type", "")
    structured: StructuredOutputConfig = manifest.structured

    if fmt_type == "json_object":
        # json_object is a lightweight mode; json_schema support implies it
        if not structured.json_schema:
            return (
                f"Model {manifest.model_id} does not advertise json_schema support. "
                f"'json_object' output may not be enforced by the backend."
            )
    elif fmt_type == "json_schema":
        if not structured.json_schema:
            return (
                f"Model {manifest.model_id} does not support json_schema structured output."
            )
    elif fmt_type == "regex":
        if not structured.regex:
            return (
                f"Model {manifest.model_id} does not support regex structured output."
            )
    elif fmt_type == "grammar":
        if not structured.grammar:
            return (
                f"Model {manifest.model_id} does not support grammar structured output."
            )

    return None


def build_json_schema_response_format(schema_input: str) -> dict[str, Any]:
    """Build a ``{"type": "json_schema", "json_schema": {...}}`` dict.

    *schema_input* may be:
    - A file path (JSON file containing the schema)
    - An inline JSON string
    """
    # Try as file path first
    path = Path(schema_input)
    if path.is_file():
        raw = path.read_text()
    else:
        raw = schema_input

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"Invalid JSON schema: {exc}. "
            f"Provide a valid JSON string or a path to a .json file."
        ) from exc

    # If the parsed object is already a full response_format envelope, use it
    if isinstance(parsed, dict) and parsed.get("type") == "json_schema":
        return parsed

    # If the parsed object has a "name" key, treat it as a json_schema spec
    if isinstance(parsed, dict) and "name" in parsed:
        return {"type": "json_schema", "json_schema": parsed}

    # Otherwise treat the whole thing as the schema body; wrap it
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "user_schema",
            "schema": parsed,
        },
    }
