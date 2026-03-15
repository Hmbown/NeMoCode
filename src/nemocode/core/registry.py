# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Central registry — resolves endpoints to providers, manages manifests and formations."""

from __future__ import annotations

import logging

from nemocode.config import get_api_key
from nemocode.config.schema import (
    Capability,
    Endpoint,
    Formation,
    Manifest,
    NeMoCodeConfig,
)
from nemocode.providers.nim_chat import NIMChatProvider
from nemocode.providers.nim_embeddings import NIMEmbeddingProvider
from nemocode.providers.nim_parse import NIMParseProvider
from nemocode.providers.nim_rerank import NIMRerankProvider

logger = logging.getLogger(__name__)


class Registry:
    """Central resolver: endpoints → manifests → providers."""

    def __init__(self, config: NeMoCodeConfig) -> None:
        self._config = config
        self._chat_cache: dict[str, NIMChatProvider] = {}
        self._embed_cache: dict[str, NIMEmbeddingProvider] = {}
        self._rerank_cache: dict[str, NIMRerankProvider] = {}
        self._parse_cache: dict[str, NIMParseProvider] = {}

    def get_endpoint(self, name: str) -> Endpoint:
        ep = self._config.endpoints.get(name)
        if ep is None:
            raise KeyError(
                f"Unknown endpoint: {name}. Available: {list(self._config.endpoints.keys())}"
            )
        return ep

    def get_manifest(self, model_id: str) -> Manifest | None:
        return self._config.manifests.get(model_id)

    def get_manifest_for_endpoint(self, endpoint_name: str) -> Manifest | None:
        ep = self.get_endpoint(endpoint_name)
        return self.get_manifest(ep.model_id)

    def get_formation(self, name: str) -> Formation:
        f = self._config.formations.get(name)
        if f is None:
            raise KeyError(
                f"Unknown formation: {name}. Available: {list(self._config.formations.keys())}"
            )
        return f

    def get_chat_provider(self, endpoint_name: str) -> NIMChatProvider:
        if endpoint_name in self._chat_cache:
            return self._chat_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        manifest = self.get_manifest(ep.model_id)
        api_key = get_api_key(ep)
        provider = NIMChatProvider(endpoint=ep, manifest=manifest, api_key=api_key)
        self._chat_cache[endpoint_name] = provider
        return provider

    def get_embedding_provider(self, endpoint_name: str) -> NIMEmbeddingProvider:
        if endpoint_name in self._embed_cache:
            return self._embed_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        api_key = get_api_key(ep)
        provider = NIMEmbeddingProvider(endpoint=ep, api_key=api_key)
        self._embed_cache[endpoint_name] = provider
        return provider

    def get_rerank_provider(self, endpoint_name: str) -> NIMRerankProvider:
        if endpoint_name in self._rerank_cache:
            return self._rerank_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        api_key = get_api_key(ep)
        provider = NIMRerankProvider(endpoint=ep, api_key=api_key)
        self._rerank_cache[endpoint_name] = provider
        return provider

    def get_parse_provider(self, endpoint_name: str) -> NIMParseProvider:
        if endpoint_name in self._parse_cache:
            return self._parse_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        api_key = get_api_key(ep)
        provider = NIMParseProvider(endpoint=ep, api_key=api_key)
        self._parse_cache[endpoint_name] = provider
        return provider

    def list_endpoints(self, capability: Capability | None = None) -> list[Endpoint]:
        eps = list(self._config.endpoints.values())
        if capability:
            eps = [ep for ep in eps if capability in ep.capabilities]
        return eps

    def list_formations(self) -> list[Formation]:
        return list(self._config.formations.values())

    def list_manifests(self) -> list[Manifest]:
        return list(self._config.manifests.values())

    def default_endpoint_name(self) -> str:
        return self._config.default_endpoint

    def active_formation_name(self) -> str | None:
        return self._config.active_formation
