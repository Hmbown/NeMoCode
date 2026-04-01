# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tool result caching for read-only operations.

Provides an LRU cache with TTL support for filesystem and search tool results.
Cache can be disabled via NEMOCODE_TOOL_CACHE=0 environment variable.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any


class ToolCache:
    """LRU cache with TTL support for tool results.

    Cache entries are automatically evicted when they exceed max_size or
    when their TTL expires. The cache can be disabled via the
    NEMOCODE_TOOL_CACHE environment variable.

    Attributes:
        max_size: Maximum number of entries before LRU eviction.
        hits: Number of cache hits.
        misses: Number of cache misses.

    Example:
        >>> cache = ToolCache(max_size=100)
        >>> cache.put("key1", "value1", ttl=30.0)
        >>> cache.get("key1")
        'value1'
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize the tool cache.

        Args:
            max_size: Maximum number of entries. Defaults to 100.
        """
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled via environment variable.

        Returns:
            True if NEMOCODE_TOOL_CACHE is not set to "0".
        """
        return os.environ.get("NEMOCODE_TOOL_CACHE", "1") != "0"

    def get(self, key: str) -> str | None:
        """Get cached result if valid and not expired.

        Args:
            key: Cache key to look up.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        if not self.enabled:
            self._misses += 1
            return None

        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]
        if time.monotonic() > expiry:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end for LRU tracking
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: str, ttl: float = 60.0) -> None:
        """Cache a result with TTL.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. Defaults to 60.0.
        """
        if not self.enabled:
            return

        # Evict expired entries first
        self._evict_expired()

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate.
        """
        self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        """Invalidate all entries starting with prefix.

        Args:
            prefix: Key prefix to match for invalidation.
        """
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, size, and enabled status.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
            "enabled": self.enabled,
        }

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        now = time.monotonic()
        expired = [k for k, (_, expiry) in self._cache.items() if now > expiry]
        for key in expired:
            del self._cache[key]


# Module-level singleton for tool caching
_tool_cache: ToolCache | None = None


def get_tool_cache() -> ToolCache:
    """Get or create the module-level tool cache singleton.

    Returns:
        The global ToolCache instance.
    """
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = ToolCache(max_size=100)
    return _tool_cache


def reset_tool_cache() -> None:
    """Reset the module-level cache singleton.

    Useful for testing to ensure clean state between tests.
    """
    global _tool_cache
    _tool_cache = None
