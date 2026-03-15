# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""File watcher — polls the project tree for external changes.

Uses pure stat-based polling (no external dependencies).  Watches for
file creations, modifications, and deletions while respecting common
ignore patterns (.git, __pycache__, node_modules, etc.).
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ChangeKind(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileChange:
    """Represents a single observed file-system change."""

    path: str
    kind: ChangeKind
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"FileChange({self.kind.value} {self.path})"


# ---------------------------------------------------------------------------
# Default ignore patterns
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".nemocode",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}

_DEFAULT_IGNORE_PATTERNS: list[str] = [
    "*.pyc",
    "*.pyo",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.class",
    "*.jar",
    "*.whl",
    "*.egg",
    "*.tar.gz",
    "*.zip",
    ".DS_Store",
    "Thumbs.db",
    "*.swp",
    "*.swo",
    "*~",
]


# ---------------------------------------------------------------------------
# File snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FileStat:
    mtime: float
    size: int


# ---------------------------------------------------------------------------
# FileWatcher
# ---------------------------------------------------------------------------


class FileWatcher:
    """Poll-based file system watcher.

    Parameters
    ----------
    root : Path
        The project directory to watch.
    ignore_patterns : list[str] | None
        Extra glob patterns to ignore (on top of defaults).
    poll_interval : float
        Seconds between polls (default 2.0).
    """

    def __init__(
        self,
        root: Path,
        ignore_patterns: list[str] | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        self._root = root.resolve()
        self._poll_interval = poll_interval
        self._ignore_patterns = list(_DEFAULT_IGNORE_PATTERNS)
        if ignore_patterns:
            self._ignore_patterns.extend(ignore_patterns)
        self._snapshot: dict[str, _FileStat] = {}
        self._changes: list[FileChange] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        """Take an initial snapshot and begin polling."""
        self._snapshot = self._scan()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the polling loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- public API ----------------------------------------------------

    def get_changes(self) -> list[FileChange]:
        """Return pending changes (non-blocking). Does NOT clear them."""
        return list(self._changes)

    def clear(self) -> None:
        """Clear all pending changes."""
        self._changes.clear()

    # -- polling -------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._check()
            except Exception:
                logger.debug("Watcher poll error", exc_info=True)

    async def _check(self) -> None:
        """Run a single poll cycle in a thread to avoid blocking the loop."""
        loop = asyncio.get_running_loop()
        new_snapshot = await loop.run_in_executor(None, self._scan)
        changes = self._diff(self._snapshot, new_snapshot)
        if changes:
            async with self._lock:
                self._changes.extend(changes)
        self._snapshot = new_snapshot

    # -- scanning and diffing ------------------------------------------

    def _scan(self) -> dict[str, _FileStat]:
        """Walk the tree and stat every non-ignored file."""
        result: dict[str, _FileStat] = {}
        for dirpath, dirnames, filenames in os.walk(self._root):
            # Prune ignored directories in-place
            dirnames[:] = [
                d
                for d in dirnames
                if not self._is_ignored_dir(d)
            ]
            for fname in filenames:
                if self._is_ignored_file(fname):
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                    result[full] = _FileStat(mtime=st.st_mtime, size=st.st_size)
                except OSError:
                    pass
        return result

    def _diff(
        self,
        old: dict[str, _FileStat],
        new: dict[str, _FileStat],
    ) -> list[FileChange]:
        """Compare two snapshots and return a list of changes."""
        changes: list[FileChange] = []
        now = time.time()

        # Creations and modifications
        for path, stat in new.items():
            prev = old.get(path)
            if prev is None:
                changes.append(
                    FileChange(path=path, kind=ChangeKind.CREATED, timestamp=now)
                )
            elif stat.mtime != prev.mtime or stat.size != prev.size:
                changes.append(
                    FileChange(path=path, kind=ChangeKind.MODIFIED, timestamp=now)
                )

        # Deletions
        for path in old:
            if path not in new:
                changes.append(
                    FileChange(path=path, kind=ChangeKind.DELETED, timestamp=now)
                )

        return changes

    # -- ignore helpers ------------------------------------------------

    def _is_ignored_dir(self, name: str) -> bool:
        if name in _DEFAULT_IGNORE_DIRS:
            return True
        # Allow wildcard matching for patterns like *.egg-info
        for pat in _DEFAULT_IGNORE_DIRS:
            if "*" in pat and fnmatch.fnmatch(name, pat):
                return True
        return False

    def _is_ignored_file(self, name: str) -> bool:
        for pat in self._ignore_patterns:
            if fnmatch.fnmatch(name, pat):
                return True
        return False
