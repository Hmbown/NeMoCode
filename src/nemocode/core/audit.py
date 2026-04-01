# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Audit trail — append-only log for tool executions, file mutations, and API calls."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(os.environ.get("NEMOCODE_CACHE_DIR", Path.home() / ".cache" / "nemocode"))
_AUDIT_FILENAME = "audit.log"
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_duration_re = re.compile(r"^(\d+)([hHdDsS])$")


def _parse_since(since: str) -> datetime:
    """Parse a duration string like '1h', '24h', '7d' into a cutoff datetime.

    Args:
        since: Duration string (e.g. "1h", "24h", "7d").

    Returns:
        A timezone-aware datetime representing the cutoff time.

    Raises:
        ValueError: If the format is not recognized.
    """
    m = _duration_re.match(since)
    if not m:
        raise ValueError(f"Invalid since format: {since!r}. Use e.g. '1h', '24h', '7d'.")
    value = int(m.group(1))
    unit = m.group(2).lower()
    delta_map = {
        "s": timedelta(seconds=value),
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
    }
    return datetime.now(timezone.utc) - delta_map[unit]


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime(_ISO_FORMAT)


class AuditLog:
    """Append-only audit log persisted as JSONL.

    The log is stored at ``{cache_dir}/nemocode/audit.log``. Each entry is a
    single JSON line containing a timestamp, event type, and relevant details.
    Thread-safe via an internal lock.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialise the audit log.

        Args:
            path: Optional explicit path to the JSONL file. Defaults to
                ``{cache_dir}/nemocode/audit.log``.
        """
        self._path = path or (_CACHE_DIR / _AUDIT_FILENAME)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        is_error: bool,
        session_id: str,
    ) -> None:
        """Record a tool execution event.

        Args:
            tool_name: Name of the tool that was executed.
            args: Arguments passed to the tool.
            result: The tool's output (truncated for storage).
            is_error: Whether the execution resulted in an error.
            session_id: The session identifier.
        """
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "type": "tool",
            "tool": tool_name,
            "args": args,
            "result_preview": result[:200],
            "is_error": is_error,
            "session": session_id,
        }
        self._append(entry)

    def log_file_mutation(
        self,
        operation: str,
        path: str,
        session_id: str,
    ) -> None:
        """Record a file mutation event.

        Args:
            operation: The mutation operation (e.g. "write", "edit", "delete").
            path: The file path that was mutated.
            session_id: The session identifier.
        """
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "type": "file",
            "operation": operation,
            "path": path,
            "session": session_id,
        }
        self._append(entry)

    def log_api_call(
        self,
        endpoint: str,
        model: str,
        tokens_used: int,
        session_id: str,
    ) -> None:
        """Record an API call event.

        Args:
            endpoint: The API endpoint that was called.
            model: The model identifier used for the call.
            tokens_used: Total tokens consumed.
            session_id: The session identifier.
        """
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "type": "api",
            "endpoint": endpoint,
            "model": model,
            "tokens_used": tokens_used,
            "session": session_id,
        }
        self._append(entry)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_entries(
        self,
        since: str | None = None,
        limit: int = 100,
        entry_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve recent audit entries with optional filtering.

        Args:
            since: Optional duration string (e.g. "1h", "24h", "7d") to
                filter entries newer than the specified duration.
            limit: Maximum number of entries to return (default 100).
            entry_type: Optional filter by entry type ("tool", "file", "api").

        Returns:
            A list of entry dicts, newest first.
        """
        cutoff = _parse_since(since) if since else None
        entries: list[dict[str, Any]] = []

        if not self._path.exists():
            return entries

        with self._lock:
            lines = self._path.read_text(encoding="utf-8", errors="replace").splitlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry_type and entry.get("type") != entry_type:
                continue

            if cutoff:
                try:
                    ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if ts < cutoff:
                    continue

            entries.append(entry)
            if len(entries) >= limit:
                break

        return entries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, entry: dict[str, Any]) -> None:
        """Append a single entry to the JSONL file (thread-safe)."""
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")

    def reset(self) -> None:
        """Truncate the log file. Intended for testing only."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_instance: AuditLog | None = None
_instance_lock = threading.Lock()


def get_audit_log() -> AuditLog:
    """Return the singleton ``AuditLog`` instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AuditLog()
    return _instance


def reset_audit_log(path: Path | None = None) -> AuditLog:
    """Reset (and optionally re-create) the singleton audit log.

    Args:
        path: Optional new path for the log file.

    Returns:
        The fresh ``AuditLog`` instance.
    """
    global _instance
    with _instance_lock:
        _instance = AuditLog(path=path)
        return _instance
