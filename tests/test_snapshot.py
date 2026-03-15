# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the git snapshot system."""

from __future__ import annotations

import subprocess

import pytest

from nemocode.core.snapshot import SnapshotManager


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True
    )
    f = tmp_path / "initial.txt"
    f.write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True)
    return tmp_path


@pytest.mark.asyncio
async def test_no_changes_no_snapshot(git_repo):
    mgr = SnapshotManager(cwd=str(git_repo))
    snap = await mgr.create_snapshot("test")
    assert snap is None


@pytest.mark.asyncio
async def test_has_changes(git_repo):
    (git_repo / "new.txt").write_text("new content")
    mgr = SnapshotManager(cwd=str(git_repo))
    assert await mgr.has_changes() is True


@pytest.mark.asyncio
async def test_create_snapshot(git_repo):
    (git_repo / "new.txt").write_text("new content")
    mgr = SnapshotManager(cwd=str(git_repo))
    snap = await mgr.create_snapshot("test-snap")
    assert snap is not None
    assert snap.files_changed >= 1
    assert len(snap.id) == 8


@pytest.mark.asyncio
async def test_list_snapshots(git_repo):
    (git_repo / "new.txt").write_text("new content")
    mgr = SnapshotManager(cwd=str(git_repo))
    await mgr.create_snapshot("snap1")
    snaps = await mgr.list_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["message"] == "nemocode-snapshot: snap1"


@pytest.mark.asyncio
async def test_restore_nonexistent(git_repo):
    mgr = SnapshotManager(cwd=str(git_repo))
    result = await mgr.restore_snapshot("nonexistent")
    assert "error" in result
