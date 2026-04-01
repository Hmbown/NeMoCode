# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Structured logging configuration for NeMoCode.

Provides a JSON formatter and StructuredLogger wrapper that automatically
attaches common contextual fields (session_id, endpoint_name, tool_name)
to log records. Structured (JSON) output is opt-in via the
NEMOCODE_JSON_LOGS=1 environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON with extra fields included."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        # Merge extra fields (anything not in standard LogRecord attrs)
        _STANDARD_ATTRS = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "taskName",
            "exc_info",
            "exc_text",
            "stack_info",
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                log_data[key] = value

        return json.dumps(log_data, default=str)


def setup_logging(level: str = "INFO", json_format: bool | None = None) -> None:
    """Configure the root logger for the application.

    Args:
        level: Logging level name (e.g. "DEBUG", "INFO", "WARNING").
        json_format: If True, use JSON output. If None, read from the
            NEMOCODE_JSON_LOGS environment variable ("1" enables JSON).
    """
    if json_format is None:
        json_format = os.environ.get("NEMOCODE_JSON_LOGS", "0") == "1"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)


class StructuredLogger:
    """Logger wrapper that attaches common contextual fields automatically.

    Every log call through this logger will include the fields set at
    construction time (session_id, endpoint_name, tool_name) plus any
    extra fields passed via the ``extra`` kwarg on individual calls.

    Args:
        name: Logger name (passed to ``logging.getLogger``).
        session_id: Optional session identifier.
        endpoint_name: Optional endpoint identifier.
        tool_name: Optional tool identifier.

    Example:
        >>> logger = StructuredLogger("my.module", session_id="abc123")
        >>> logger.info("tool_executed", extra={"tool": "read_file", "duration_ms": 12})
    """

    def __init__(
        self,
        name: str,
        *,
        session_id: str = "",
        endpoint_name: str = "",
        tool_name: str = "",
    ) -> None:
        self._logger = logging.getLogger(name)
        self._session_id = session_id
        self._endpoint_name = endpoint_name
        self._tool_name = tool_name

    def _merge_extra(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self._session_id:
            result["session_id"] = self._session_id
        if self._endpoint_name:
            result["endpoint_name"] = self._endpoint_name
        if self._tool_name:
            result["tool_name"] = self._tool_name
        if extra:
            result.update(extra)
        return result

    def debug(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.debug(msg, extra=self._merge_extra(extra), **kwargs)

    def info(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.info(msg, extra=self._merge_extra(extra), **kwargs)

    def warning(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.warning(msg, extra=self._merge_extra(extra), **kwargs)

    def error(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.error(msg, extra=self._merge_extra(extra), **kwargs)

    def exception(self, msg: str, extra: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._logger.exception(msg, extra=self._merge_extra(extra), **kwargs)

    def getChild(self, suffix: str) -> StructuredLogger:
        child = StructuredLogger(f"{self._logger.name}.{suffix}")
        child._session_id = self._session_id
        child._endpoint_name = self._endpoint_name
        child._tool_name = self._tool_name
        return child
