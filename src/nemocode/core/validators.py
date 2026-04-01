# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Input validation functions for the NeMoCode CLI boundary.

All public validators raise ``ValueError`` with clear messages when input
is rejected.  They are intended to be called at the entry point of each
CLI command before any business logic runs.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")
_DANGEROUS_CMD_PATTERNS = re.compile(
    r"(\$\(|`|;|\|\||&&|\|>|<\||\$\{)",
)
_PATH_TRAVERSAL_RE = re.compile(r"(\.\./|\.\.\\)")

_TIMEOUT_MIN = 1
_TIMEOUT_MAX = 3599


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_file_path(path: str) -> str:
    """Validate a user-supplied file-system path.

    Args:
        path: The raw path string from the CLI.

    Returns:
        The original *path* when it passes all checks.

    Raises:
        ValueError: If the path contains null bytes, control characters,
            or path-traversal sequences.
    """
    if not path:
        raise ValueError("File path must not be empty")

    if "\x00" in path:
        raise ValueError("File path must not contain null bytes")

    if any(ord(c) < 32 and c not in ("\t", "\n", "\r") for c in path):
        raise ValueError("File path must not contain control characters")

    if _PATH_TRAVERSAL_RE.search(path):
        raise ValueError("File path must not contain path traversal sequences (../)")

    return path


def validate_timeout(value: int) -> int:
    """Validate a timeout value in seconds.

    Args:
        value: The timeout in seconds.

    Returns:
        The original *value* when it is within the allowed range.

    Raises:
        ValueError: If the timeout is not strictly between 0 and 3600.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Timeout must be an integer, got {type(value).__name__}")

    if value <= 0:
        raise ValueError(f"Timeout must be greater than 0, got {value}")

    if value >= 3600:
        raise ValueError(f"Timeout must be less than 3600 seconds, got {value}")

    return value


def validate_model_id(model_id: str) -> str:
    """Validate a model identifier against known safe patterns.

    Allowed characters: ASCII letters, digits, dots, forward slashes, and
    hyphens.  This covers Hugging-Face-style IDs such as
    ``nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16``.

    Args:
        model_id: The raw model identifier string.

    Returns:
        The original *model_id* when it matches the allowed pattern.

    Raises:
        ValueError: If the model ID contains disallowed characters or is
            empty.
    """
    if not model_id:
        raise ValueError("Model ID must not be empty")

    if len(model_id) > 512:
        raise ValueError("Model ID must not exceed 512 characters")

    if not _MODEL_ID_RE.match(model_id):
        raise ValueError(
            f"Model ID '{model_id}' contains disallowed characters. "
            "Only letters, digits, dots, slashes, and hyphens are allowed."
        )

    return model_id


def validate_json_input(data: str) -> dict:
    """Validate and parse a JSON string before sending to an API.

    Args:
        data: A JSON-formatted string.

    Returns:
        The parsed Python ``dict``.

    Raises:
        ValueError: If the string is not valid JSON or does not decode to
            a dict.
    """
    if not data or not data.strip():
        raise ValueError("JSON input must not be empty")

    try:
        result = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("JSON input must decode to a JSON object (dict)")

    return result


def validate_command_args(cmd: str) -> str:
    """Validate that a command string does not contain dangerous patterns.

    Checks for shell injection vectors such as command substitution,
    pipeline operators, and logical chaining.

    Args:
        cmd: The raw command string from the CLI.

    Returns:
        The original *cmd* when it passes all checks.

    Raises:
        ValueError: If the command contains potentially dangerous shell
            patterns.
    """
    if not cmd:
        raise ValueError("Command must not be empty")

    if len(cmd) > 10000:
        raise ValueError("Command must not exceed 10000 characters")

    if _DANGEROUS_CMD_PATTERNS.search(cmd):
        raise ValueError(
            "Command contains potentially dangerous shell patterns (e.g. $(), backticks, ;, ||, &&)"
        )

    return cmd
