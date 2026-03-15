# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Git snapshot system — create safety checkpoints before risky operations.

Provides automatic WIP commits/stashes so the user can always revert
to a known-good state after the agent makes changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_SNAPSHOTS = 20


@dataclass
class Snapshot:
    """Record of a git snapshot."""

    id: str  # short commit hash or stash ref
    kind: str  # "commit" or "stash"
    message: str
    timestamp: float = field(default_factory=time.time)
    files_changed: int = 0


class SnapshotManager:
    """Manages git snapshots for safe rollback."""

    def __init__(self, cwd: str = "") -> None:
        self._cwd = cwd or None
        self._snapshots: list[Snapshot] = []

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    async def _git(self, args: list[str]) -> tuple[int, str, str]:
        """Run a git command."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode().strip(), stderr.decode().strip()

    async def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        rc, out, _ = await self._git(["status", "--porcelain"])
        return rc == 0 and bool(out.strip())

    async def create_snapshot(self, label: str = "") -> Snapshot | None:
        """Create a WIP snapshot of current state.

        Uses git stash create (creates a stash commit without modifying worktree).
        Falls back to a WIP commit on a detached ref if stash fails.
        """
        if not await self.has_changes():
            return None

        # Count changed files
        rc, status_out, _ = await self._git(["status", "--porcelain"])
        files_changed = len(status_out.splitlines()) if rc == 0 else 0

        msg = f"nemocode-snapshot: {label or 'auto'}"

        # Try stash create (doesn't touch worktree or index)
        rc, stash_hash, _ = await self._git(["stash", "create", msg])
        if rc == 0 and stash_hash:
            # Store the stash ref
            await self._git(["stash", "store", "-m", msg, stash_hash])
            snap = Snapshot(
                id=stash_hash[:8],
                kind="stash",
                message=msg,
                files_changed=files_changed,
            )
            self._snapshots.append(snap)
            self._trim()
            logger.info("Created snapshot: stash %s (%d files)", snap.id, files_changed)
            return snap

        # Fallback: create a WIP commit, then soft-reset
        rc, _, _ = await self._git(["add", "-A"])
        if rc != 0:
            return None

        rc, _, _ = await self._git(["commit", "-m", msg, "--no-verify"])
        if rc != 0:
            return None

        rc, commit_hash, _ = await self._git(["rev-parse", "HEAD"])
        if rc != 0:
            return None

        # Soft-reset to undo the commit but keep changes staged
        await self._git(["reset", "--soft", "HEAD~1"])

        snap = Snapshot(
            id=commit_hash[:8],
            kind="commit",
            message=msg,
            files_changed=files_changed,
        )
        self._snapshots.append(snap)
        self._trim()
        logger.info("Created snapshot: commit %s (%d files)", snap.id, files_changed)
        return snap

    async def restore_snapshot(self, snapshot_id: str) -> dict:
        """Restore a snapshot by ID."""
        snap = None
        for s in self._snapshots:
            if s.id == snapshot_id:
                snap = s
                break

        if snap is None:
            return {"error": f"Snapshot not found: {snapshot_id}"}

        if snap.kind == "stash":
            # Find the stash index
            rc, stash_list, _ = await self._git(["stash", "list"])
            if rc != 0:
                return {"error": "Failed to list stashes"}

            stash_ref = None
            for line in stash_list.splitlines():
                if snap.message in line:
                    stash_ref = line.split(":")[0]  # e.g. "stash@{0}"
                    break

            if stash_ref:
                rc, out, err = await self._git(["stash", "apply", stash_ref])
                if rc != 0:
                    return {"error": f"Failed to apply stash: {err}"}
                return {"status": "ok", "restored": snap.id, "kind": "stash"}

        # For commit snapshots, cherry-pick or checkout
        rc, _, err = await self._git(["cherry-pick", "--no-commit", snap.id])
        if rc != 0:
            # Try hard checkout of files from that commit
            rc, _, err = await self._git(["checkout", snap.id, "--", "."])
            if rc != 0:
                return {"error": f"Failed to restore: {err}"}

        return {"status": "ok", "restored": snap.id, "kind": snap.kind}

    async def list_snapshots(self) -> list[dict]:
        """List all snapshots."""
        return [
            {
                "id": s.id,
                "kind": s.kind,
                "message": s.message,
                "timestamp": s.timestamp,
                "files_changed": s.files_changed,
            }
            for s in reversed(self._snapshots)
        ]

    def _trim(self) -> None:
        """Keep only the most recent snapshots."""
        while len(self._snapshots) > _MAX_SNAPSHOTS:
            self._snapshots.pop(0)
