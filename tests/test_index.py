# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for file embedding index."""

from __future__ import annotations

import pytest

from nemocode.core.index import _CHUNK_SIZE, FileIndex


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project for indexing."""
    (tmp_path / "main.py").write_text("def hello():\n    print('hello world')\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "README.md").write_text("# Test Project\nA test project.\n")
    (tmp_path / ".git").mkdir()  # simulate git repo
    (tmp_path / "data.bin").write_bytes(b"\x00" * 100)  # non-code file
    return tmp_path


class TestFileIndex:
    def test_collect_files(self, project_dir):
        idx = FileIndex(project_dir)
        files = idx.collect_files()
        names = {f.name for f in files}
        assert "main.py" in names
        assert "utils.py" in names
        assert "README.md" in names
        assert "data.bin" not in names  # not a code extension

    def test_chunk_small_files(self, project_dir):
        idx = FileIndex(project_dir)
        files = idx.collect_files()
        chunks = idx.chunk_files(files)
        assert len(chunks) >= 3  # at least one per file
        for chunk in chunks:
            assert "path" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) <= _CHUNK_SIZE

    def test_chunk_large_file(self, tmp_path):
        large_text = "x = 1\n" * 1000  # well over _CHUNK_SIZE
        (tmp_path / "big.py").write_text(large_text)
        idx = FileIndex(tmp_path)
        files = idx.collect_files()
        chunks = idx.chunk_files(files)
        assert len(chunks) > 1  # should be split

    def test_not_built_initially(self, project_dir):
        idx = FileIndex(project_dir)
        assert idx.is_built is False

    def test_stats_before_build(self, project_dir):
        idx = FileIndex(project_dir)
        stats = idx.stats()
        assert stats["built"] is False
        assert stats["chunks"] == 0

    @pytest.mark.asyncio
    async def test_build_and_search(self, project_dir):
        idx = FileIndex(project_dir)

        # Mock embedding function — returns distinct non-zero vectors
        call_count = 0

        async def mock_embed(texts):
            nonlocal call_count
            call_count += 1
            # Use hash of text content to generate varied embeddings
            result = []
            for t in texts:
                h = hash(t) % 1000
                result.append([float(h + i + 1) for i in range(8)])
            return result

        count = await idx.build(mock_embed)
        assert count > 0
        assert idx.is_built is True

        # Search
        results = await idx.search("hello", mock_embed, top_k=2)
        assert len(results) > 0
        assert "path" in results[0]
        assert "score" in results[0]

    def test_skips_git_dir(self, project_dir):
        (project_dir / ".git" / "config").write_text("gitconfig")
        idx = FileIndex(project_dir)
        files = idx.collect_files()
        paths = [str(f) for f in files]
        assert not any(".git" in p for p in paths)
