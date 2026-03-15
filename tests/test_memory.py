# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for auto-memory system."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from nemocode.core import memory


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Redirect memory storage to a temp directory."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    with mock.patch.object(memory, "_GLOBAL_MEMORY_DIR", mem_dir):
        yield mem_dir


class TestSaveMemory:
    def test_save_global(self, temp_memory_dir):
        result = memory.save_memory("test_key", "test content", scope="global")
        assert result["status"] == "ok"
        assert (temp_memory_dir / "test_key.json").exists()

    def test_save_and_load(self, temp_memory_dir):
        memory.save_memory("pref", "user likes pytest", tags=["testing"])
        data = json.loads((temp_memory_dir / "pref.json").read_text())
        assert data["content"] == "user likes pytest"
        assert data["tags"] == ["testing"]

    def test_save_updates_timestamp(self, temp_memory_dir):
        memory.save_memory("key1", "v1")
        data1 = json.loads((temp_memory_dir / "key1.json").read_text())
        memory.save_memory("key1", "v2")
        data2 = json.loads((temp_memory_dir / "key1.json").read_text())
        assert data2["content"] == "v2"
        assert data2["created_at"] == data1["created_at"]  # preserved

    def test_sanitizes_key(self, temp_memory_dir):
        result = memory.save_memory("my key/path", "val")
        assert result["status"] == "ok"
        assert (temp_memory_dir / "my_key_path.json").exists()


class TestForgetMemory:
    def test_forget_existing(self, temp_memory_dir):
        memory.save_memory("to_delete", "temp")
        result = memory.forget_memory("to_delete")
        assert result["action"] == "deleted"
        assert not (temp_memory_dir / "to_delete.json").exists()

    def test_forget_nonexistent(self, temp_memory_dir):
        result = memory.forget_memory("nope")
        assert result["action"] == "not_found"


class TestLoadAllMemories:
    def test_empty(self, temp_memory_dir):
        result = memory.load_all_memories()
        assert result == ""

    def test_loads_memories(self, temp_memory_dir):
        memory.save_memory("key1", "value1")
        memory.save_memory("key2", "value2")
        result = memory.load_all_memories()
        assert "## Memories" in result
        assert "key1" in result
        assert "key2" in result


class TestListMemories:
    def test_list_empty(self, temp_memory_dir):
        result = memory.list_memories()
        assert result == []

    def test_list_global(self, temp_memory_dir):
        memory.save_memory("a", "val_a")
        memory.save_memory("b", "val_b")
        result = memory.list_memories(scope="global")
        assert len(result) == 2
