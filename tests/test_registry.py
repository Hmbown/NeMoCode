# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test endpoint resolution and manifest lookup."""

from __future__ import annotations

import pytest

from nemocode.config.schema import (
    Capability,
    Endpoint,
    Manifest,
)
from nemocode.core.registry import Registry


class TestRegistry:
    def test_get_endpoint(self, sample_registry: Registry, sample_endpoint: Endpoint):
        ep = sample_registry.get_endpoint("test-ep")
        assert ep.name == sample_endpoint.name
        assert ep.model_id == sample_endpoint.model_id

    def test_get_unknown_endpoint(self, sample_registry: Registry):
        with pytest.raises(KeyError, match="Unknown endpoint"):
            sample_registry.get_endpoint("nonexistent")

    def test_get_manifest(self, sample_registry: Registry, sample_manifest: Manifest):
        m = sample_registry.get_manifest(sample_manifest.model_id)
        assert m is not None
        assert m.display_name == "Nemotron 3 Super"

    def test_get_manifest_for_endpoint(self, sample_registry: Registry):
        m = sample_registry.get_manifest_for_endpoint("test-ep")
        assert m is not None
        assert m.tier_in_family == "super"

    def test_get_formation(self, sample_registry: Registry):
        f = sample_registry.get_formation("solo")
        assert f.name == "Solo"

    def test_get_unknown_formation(self, sample_registry: Registry):
        with pytest.raises(KeyError, match="Unknown formation"):
            sample_registry.get_formation("nonexistent")

    def test_list_endpoints(self, sample_registry: Registry):
        eps = sample_registry.list_endpoints()
        assert len(eps) == 1

    def test_list_endpoints_by_capability(self, sample_registry: Registry):
        eps = sample_registry.list_endpoints(Capability.CODE)
        assert len(eps) == 1
        eps = sample_registry.list_endpoints(Capability.EMBED)
        assert len(eps) == 0

    def test_default_endpoint(self, sample_registry: Registry):
        assert sample_registry.default_endpoint_name() == "test-ep"

    def test_chat_provider_cached(self, sample_registry: Registry):
        # Getting the same provider twice should return the cached instance
        # (we can't actually test the API call, but we can test the cache)
        p1 = sample_registry.get_chat_provider("test-ep")
        p2 = sample_registry.get_chat_provider("test-ep")
        assert p1 is p2
