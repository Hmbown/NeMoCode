# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Structured exception hierarchy for NeMoCode.

Enables callers (CLI, TUI, API) to handle errors programmatically
instead of parsing error strings.
"""

from __future__ import annotations


class NeMoCodeError(Exception):
    """Base exception for all NeMoCode errors."""


class ToolExecutionError(NeMoCodeError):
    """A tool execution failed."""

    def __init__(
        self, tool_name: str, message: str, *, original_error: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class ProviderError(NeMoCodeError):
    """A chat provider (API/backend) request failed."""

    def __init__(
        self,
        message: str,
        *,
        endpoint_name: str | None = None,
        status_code: int | None = None,
        is_retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.endpoint_name = endpoint_name
        self.status_code = status_code
        self.is_retryable = is_retryable


class NetworkError(ProviderError):
    """Network-level failure (connection refused, timeout, DNS, etc.)."""


class AuthError(ProviderError):
    """Authentication or authorization failure (401, 403, invalid key)."""


class QuotaError(ProviderError):
    """Rate limit or quota exceeded (429)."""


class ServerError(ProviderError):
    """Backend server error (5xx)."""


class StagnationError(NeMoCodeError):
    """The agent loop detected stagnation (repeated identical actions)."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class PermissionDeniedError(NeMoCodeError):
    """An operation was denied by the permission engine or sandbox."""

    def __init__(
        self, message: str, *, operation: str | None = None, path: str | None = None
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.path = path


class ConfigError(NeMoCodeError):
    """Configuration is invalid or missing."""


class SessionError(NeMoCodeError):
    """Session persistence or restoration failed."""
