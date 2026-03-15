# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Retrieval index — embed project files for RAG-augmented coding.

Uses numpy for cosine similarity (no FAISS dependency).
Builds a local file index, embeds on first use, searches by query.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INDEX_DIR = Path(
    os.environ.get("NEMOCODE_INDEX_DIR", "~/.local/share/nemocode/index")
).expanduser()

# File extensions to index
_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".md",
    ".txt",
    ".sql",
    ".html",
    ".css",
}

_MAX_FILE_SIZE = 100_000  # bytes
_CHUNK_SIZE = 1500  # characters per chunk
_CHUNK_OVERLAP = 200


class FileIndex:
    """In-memory file embedding index for a project."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self._chunks: list[dict[str, Any]] = []  # {path, start, end, text}
        self._embeddings: list[list[float]] = []
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    def collect_files(self) -> list[Path]:
        """Collect indexable files from the project."""
        files = []
        ignore_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".tox",
            "dist",
            "build",
        }

        for p in self.project_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _CODE_EXTENSIONS:
                # Skip ignored directories
                if any(part in ignore_dirs for part in p.parts):
                    continue
                # Skip large files
                try:
                    if p.stat().st_size <= _MAX_FILE_SIZE:
                        files.append(p)
                except OSError:
                    continue
        return files

    def chunk_files(self, files: list[Path]) -> list[dict[str, Any]]:
        """Split files into overlapping chunks for embedding."""
        chunks = []
        for path in files:
            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue

            rel_path = str(path.relative_to(self.project_root))

            if len(text) <= _CHUNK_SIZE:
                chunks.append(
                    {
                        "path": rel_path,
                        "start": 0,
                        "end": len(text),
                        "text": text,
                    }
                )
            else:
                # Split into overlapping chunks
                pos = 0
                while pos < len(text):
                    end = min(pos + _CHUNK_SIZE, len(text))
                    chunks.append(
                        {
                            "path": rel_path,
                            "start": pos,
                            "end": end,
                            "text": text[pos:end],
                        }
                    )
                    pos += _CHUNK_SIZE - _CHUNK_OVERLAP

        return chunks

    async def build(self, embed_fn) -> int:
        """Build the index by embedding all project files.

        embed_fn: async function that takes list[str] and returns list[list[float]]
        """
        files = self.collect_files()
        if not files:
            logger.info("No indexable files found")
            return 0

        self._chunks = self.chunk_files(files)
        if not self._chunks:
            return 0

        # Embed in batches
        batch_size = 50
        all_embeddings: list[list[float]] = []

        for i in range(0, len(self._chunks), batch_size):
            batch_texts = [c["text"] for c in self._chunks[i : i + batch_size]]
            try:
                embeddings = await embed_fn(batch_texts)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error("Embedding batch %d failed: %s", i // batch_size, e)
                # Fill with zeros for failed batches
                all_embeddings.extend([[0.0] * 1024 for _ in batch_texts])

        self._embeddings = all_embeddings
        self._built = True
        logger.info("Indexed %d chunks from %d files", len(self._chunks), len(files))
        return len(self._chunks)

    async def search(
        self, query: str, embed_fn, top_k: int = 5, rerank_fn=None
    ) -> list[dict[str, Any]]:
        """Search the index for relevant chunks.

        query: Search query
        embed_fn: async function for embedding the query
        top_k: Number of results to return
        rerank_fn: Optional async rerank function(query, passages, top_n)
        """
        if not self._built or not self._embeddings:
            return []

        # Embed the query
        try:
            query_emb = (await embed_fn([query]))[0]
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            return []

        # Cosine similarity using list comprehension (no numpy needed)
        scores = []
        q_norm = sum(x * x for x in query_emb) ** 0.5
        if q_norm == 0:
            return []

        for emb in self._embeddings:
            dot = sum(a * b for a, b in zip(query_emb, emb))
            e_norm = sum(x * x for x in emb) ** 0.5
            sim = dot / (q_norm * e_norm) if e_norm > 0 else 0.0
            scores.append(sim)

        # Get top results
        candidates_k = top_k * 3 if rerank_fn else top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:candidates_k]

        results = []
        for idx, score in indexed:
            chunk = self._chunks[idx]
            results.append(
                {
                    "path": chunk["path"],
                    "score": round(score, 4),
                    "text": chunk["text"][:500],
                    "start": chunk["start"],
                    "end": chunk["end"],
                }
            )

        # Optional reranking
        if rerank_fn and results:
            try:
                passages = [r["text"] for r in results]
                reranked = await rerank_fn(query, passages, top_k)
                # reranked is list of (index, score)
                reranked_results = []
                for orig_idx, rerank_score in reranked:
                    r = results[orig_idx].copy()
                    r["rerank_score"] = round(rerank_score, 4)
                    reranked_results.append(r)
                return reranked_results
            except Exception as e:
                logger.debug("Reranking failed, using embedding scores: %s", e)

        return results[:top_k]

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return {
            "built": self._built,
            "chunks": len(self._chunks),
            "files": len(set(c["path"] for c in self._chunks)),
            "embedding_dim": len(self._embeddings[0]) if self._embeddings else 0,
        }
