# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Central registry — resolves endpoints to providers, manages manifests and formations."""

from __future__ import annotations

import logging
from typing import Any, Callable

from nemocode.config import get_api_key
from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    Formation,
    Manifest,
    NeMoCodeConfig,
)
from nemocode.core.streaming import ChatProvider, EmbeddingProvider

logger = logging.getLogger(__name__)

ChatProviderFactory = Callable[[Endpoint, Manifest | None, str | None, str], ChatProvider]
EmbedProviderFactory = Callable[[Endpoint, str | None, str], EmbeddingProvider]
RerankProviderFactory = Callable[[Endpoint, str | None, str], Any]
ParseProviderFactory = Callable[[Endpoint, str | None, str], Any]


def _default_chat_factory(
    endpoint: Endpoint,
    manifest: Manifest | None,
    api_key: str | None,
    endpoint_name: str,
) -> ChatProvider:
    from nemocode.providers.nim_chat import NIMChatProvider

    return NIMChatProvider(
        endpoint=endpoint,
        manifest=manifest,
        api_key=api_key,
        endpoint_name=endpoint_name,
    )


def _default_embed_factory(
    endpoint: Endpoint,
    api_key: str | None,
    endpoint_name: str,
) -> EmbeddingProvider:
    from nemocode.providers.nim_embeddings import NIMEmbeddingProvider

    return NIMEmbeddingProvider(endpoint=endpoint, api_key=api_key)


def _default_rerank_factory(
    endpoint: Endpoint,
    api_key: str | None,
    endpoint_name: str,
) -> Any:
    from nemocode.providers.nim_rerank import NIMRerankProvider

    return NIMRerankProvider(endpoint=endpoint, api_key=api_key)


def _default_parse_factory(
    endpoint: Endpoint,
    api_key: str | None,
    endpoint_name: str,
) -> Any:
    from nemocode.providers.nim_parse import NIMParseProvider

    return NIMParseProvider(endpoint=endpoint, api_key=api_key)


class Registry:
    """Central resolver: endpoints -> manifests -> providers.

    Provider creation is pluggable. By default, NIM providers are used
    for all endpoint tiers. Register a custom factory via
    ``register_chat_factory(tier, factory)`` to override provider
    creation for specific endpoint tiers (e.g. OpenRouter, Together).
    """

    def __init__(
        self,
        config: NeMoCodeConfig,
        *,
        chat_factory: ChatProviderFactory | None = None,
        embed_factory: EmbedProviderFactory | None = None,
        rerank_factory: RerankProviderFactory | None = None,
        parse_factory: ParseProviderFactory | None = None,
    ) -> None:
        self._config = config
        self._chat_cache: dict[str, ChatProvider] = {}
        self._embed_cache: dict[str, EmbeddingProvider] = {}
        self._rerank_cache: dict[str, Any] = {}
        self._parse_cache: dict[str, Any] = {}
        self._chat_factories: dict[EndpointTier | None, ChatProviderFactory] = {
            None: chat_factory or _default_chat_factory,
        }
        self._embed_factory = embed_factory or _default_embed_factory
        self._rerank_factory = rerank_factory or _default_rerank_factory
        self._parse_factory = parse_factory or _default_parse_factory

    def register_chat_factory(
        self,
        tier: EndpointTier | None,
        factory: ChatProviderFactory,
    ) -> None:
        """Register a custom chat provider factory for an endpoint tier.

        Pass ``tier=None`` to override the default (fallback) factory.
        Factories are matched by exact tier first, then ``None`` fallback.
        """
        self._chat_factories[tier] = factory

    def _resolve_chat_factory(self, endpoint: Endpoint) -> ChatProviderFactory:
        if endpoint.tier in self._chat_factories:
            return self._chat_factories[endpoint.tier]
        return self._chat_factories[None]

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

    def get_chat_provider(self, endpoint_name: str) -> ChatProvider:
        if endpoint_name in self._chat_cache:
            return self._chat_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        manifest = self.get_manifest(ep.model_id)
        api_key = get_api_key(ep)
        factory = self._resolve_chat_factory(ep)
        provider = factory(ep, manifest, api_key, endpoint_name)
        self._chat_cache[endpoint_name] = provider
        return provider

    def get_embedding_provider(self, endpoint_name: str) -> EmbeddingProvider:
        if endpoint_name in self._embed_cache:
            return self._embed_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        api_key = get_api_key(ep)
        provider = self._embed_factory(ep, api_key, endpoint_name)
        self._embed_cache[endpoint_name] = provider
        return provider

    def get_rerank_provider(self, endpoint_name: str) -> Any:
        if endpoint_name in self._rerank_cache:
            return self._rerank_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        api_key = get_api_key(ep)
        provider = self._rerank_factory(ep, api_key, endpoint_name)
        self._rerank_cache[endpoint_name] = provider
        return provider

    def get_parse_provider(self, endpoint_name: str) -> Any:
        if endpoint_name in self._parse_cache:
            return self._parse_cache[endpoint_name]
        ep = self.get_endpoint(endpoint_name)
        api_key = get_api_key(ep)
        provider = self._parse_factory(ep, api_key, endpoint_name)
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
