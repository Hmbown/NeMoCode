# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for the file watcher — change detection with temp directories."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from nemocode.core.watcher import ChangeKind, FileChange, FileWatcher

# ---------------------------------------------------------------------------
# FileChange data class
# ---------------------------------------------------------------------------


class TestFileChange:
    def test_repr(self):
        fc = FileChange(path="/foo.py", kind=ChangeKind.MODIFIED)
        assert "modified" in repr(fc)
        assert "/foo.py" in repr(fc)

    def test_change_kinds(self):
        assert ChangeKind.CREATED.value == "created"
        assert ChangeKind.MODIFIED.value == "modified"
        assert ChangeKind.DELETED.value == "deleted"


# ---------------------------------------------------------------------------
# Snapshot scan and diff (synchronous, no polling)
# ---------------------------------------------------------------------------


class TestScanAndDiff:
    def test_scan_finds_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.txt").write_text("hello")

        watcher = FileWatcher(tmp_path)
        snapshot = watcher._scan()
        paths = set(snapshot.keys())
        assert str(tmp_path / "a.py") in paths
        assert str(tmp_path / "b.txt") in paths

    def test_scan_ignores_git(self, tmp_path: Path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("...")
        (tmp_path / "main.py").write_text("pass")

        watcher = FileWatcher(tmp_path)
        snapshot = watcher._scan()
        assert not any(".git" in p for p in snapshot)
        assert str(tmp_path / "main.py") in snapshot

    def test_scan_ignores_pycache(self, tmp_path: Path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-312.pyc").write_bytes(b"\x00")
        (tmp_path / "module.py").write_text("pass")

        watcher = FileWatcher(tmp_path)
        snapshot = watcher._scan()
        assert not any("__pycache__" in p for p in snapshot)

    def test_scan_ignores_node_modules(self, tmp_path: Path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lodash.js").write_text("// lodash")
        (tmp_path / "index.js").write_text("//")

        watcher = FileWatcher(tmp_path)
        snapshot = watcher._scan()
        # Check path components (not substring) to avoid false positives
        # from the pytest tmp dir name containing "node_modules"
        assert not any("/node_modules/" in p for p in snapshot)
        assert str(tmp_path / "index.js") in snapshot

    def test_scan_ignores_pyc_files(self, tmp_path: Path):
        (tmp_path / "mod.pyc").write_bytes(b"\x00")
        (tmp_path / "mod.py").write_text("pass")

        watcher = FileWatcher(tmp_path)
        snapshot = watcher._scan()
        assert not any(p.endswith(".pyc") for p in snapshot)

    def test_scan_with_custom_ignore(self, tmp_path: Path):
        (tmp_path / "data.csv").write_text("1,2,3")
        (tmp_path / "main.py").write_text("pass")

        watcher = FileWatcher(tmp_path, ignore_patterns=["*.csv"])
        snapshot = watcher._scan()
        assert not any(p.endswith(".csv") for p in snapshot)
        assert str(tmp_path / "main.py") in snapshot

    def test_diff_detects_creation(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        old: dict = {}
        (tmp_path / "new.py").write_text("x = 1")
        new = watcher._scan()

        changes = watcher._diff(old, new)
        assert len(changes) == 1
        assert changes[0].kind == ChangeKind.CREATED

    def test_diff_detects_deletion(self, tmp_path: Path):
        f = tmp_path / "gone.py"
        f.write_text("bye")

        watcher = FileWatcher(tmp_path)
        old = watcher._scan()
        f.unlink()
        new = watcher._scan()

        changes = watcher._diff(old, new)
        assert len(changes) == 1
        assert changes[0].kind == ChangeKind.DELETED

    def test_diff_detects_modification(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text("v1")

        watcher = FileWatcher(tmp_path)
        old = watcher._scan()

        # Ensure mtime changes (some file systems have 1-second granularity)
        time.sleep(0.05)
        f.write_text("v2 — modified content that changes size")
        new = watcher._scan()

        changes = watcher._diff(old, new)
        assert len(changes) >= 1
        kinds = {c.kind for c in changes}
        assert ChangeKind.MODIFIED in kinds or ChangeKind.CREATED in kinds

    def test_diff_no_changes(self, tmp_path: Path):
        (tmp_path / "stable.py").write_text("pass")
        watcher = FileWatcher(tmp_path)
        snap = watcher._scan()
        assert watcher._diff(snap, snap) == []


# ---------------------------------------------------------------------------
# Async start / stop / get_changes / clear
# ---------------------------------------------------------------------------


class TestAsyncWatcher:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_path: Path):
        (tmp_path / "init.py").write_text("pass")
        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()
        assert watcher.is_running
        await watcher.stop()
        assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_detects_new_file(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()

        # Create a file after the watcher has started
        (tmp_path / "added.py").write_text("print('hi')")
        # Wait for at least one poll cycle
        await asyncio.sleep(0.3)

        changes = watcher.get_changes()
        await watcher.stop()

        created = [c for c in changes if c.kind == ChangeKind.CREATED]
        assert len(created) >= 1
        assert any("added.py" in c.path for c in created)

    @pytest.mark.asyncio
    async def test_detects_deletion(self, tmp_path: Path):
        target = tmp_path / "to_delete.py"
        target.write_text("pass")

        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()

        target.unlink()
        await asyncio.sleep(0.3)

        changes = watcher.get_changes()
        await watcher.stop()

        deleted = [c for c in changes if c.kind == ChangeKind.DELETED]
        assert len(deleted) >= 1

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()

        (tmp_path / "temp.py").write_text("pass")
        await asyncio.sleep(0.3)

        assert len(watcher.get_changes()) > 0
        watcher.clear()
        assert watcher.get_changes() == []

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_get_changes_is_non_destructive(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()

        (tmp_path / "keep.py").write_text("pass")
        await asyncio.sleep(0.3)

        first = watcher.get_changes()
        second = watcher.get_changes()
        await watcher.stop()

        # get_changes returns a copy, so calling it twice gives the same data
        assert len(first) == len(second)

    @pytest.mark.asyncio
    async def test_ignores_nemocode_dir(self, tmp_path: Path):
        nc = tmp_path / ".nemocode"
        nc.mkdir()

        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()

        (nc / "state.json").write_text("{}")
        await asyncio.sleep(0.3)

        changes = watcher.get_changes()
        await watcher.stop()

        assert not any(".nemocode" in c.path for c in changes)

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.1)
        await watcher.start()
        await watcher.stop()
        await watcher.stop()  # should not raise
        assert not watcher.is_running
