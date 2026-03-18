# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for DGX Spark-aware routing in router.py."""

from __future__ import annotations

import pytest

from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    NeMoCodeConfig,
    ToolPermissions,
)
from nemocode.core.router import (
    _has_spark_endpoints,
    get_auto_endpoint,
    route_to_formation,
)


@pytest.fixture
def spark_config() -> NeMoCodeConfig:
    """Config with Spark NIM endpoints + hosted fallbacks."""
    return NeMoCodeConfig(
        default_endpoint="spark-nim-super",
        endpoints={
            "spark-nim-super": Endpoint(
                name="Spark NIM Super",
                tier=EndpointTier.LOCAL_NIM,
                base_url="http://localhost:8000/v1",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                capabilities=[Capability.CHAT, Capability.CODE],
            ),
            "spark-nim-nano9b": Endpoint(
                name="Spark NIM Nano 9B",
                tier=EndpointTier.LOCAL_NIM,
                base_url="http://localhost:8002/v1",
                model_id="nvidia/nemotron-nano-9b-v2",
                capabilities=[Capability.CHAT, Capability.CODE],
            ),
            "nim-super": Endpoint(
                name="Hosted Super",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                capabilities=[Capability.CHAT, Capability.CODE],
            ),
            "nim-nano": Endpoint(
                name="Hosted Nano",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                capabilities=[Capability.CHAT, Capability.CODE],
            ),
        },
        formations={
            "spark": Formation(
                name="Spark",
                description="DGX Spark local",
                slots=[
                    FormationSlot(
                        endpoint="spark-nim-super",
                        role=FormationRole.EXECUTOR,
                    ),
                    FormationSlot(
                        endpoint="spark-nim-nano9b",
                        role=FormationRole.FAST,
                    ),
                ],
            ),
            "super-nano": Formation(
                name="Super-Nano",
                description="Hosted pair",
                slots=[
                    FormationSlot(
                        endpoint="nim-super",
                        role=FormationRole.EXECUTOR,
                    ),
                    FormationSlot(
                        endpoint="nim-nano",
                        role=FormationRole.FAST,
                    ),
                ],
            ),
        },
        permissions=ToolPermissions(),
    )


@pytest.fixture
def no_spark_config() -> NeMoCodeConfig:
    """Config without any Spark endpoints."""
    return NeMoCodeConfig(
        default_endpoint="nim-super",
        endpoints={
            "nim-super": Endpoint(
                name="Hosted Super",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                capabilities=[Capability.CHAT, Capability.CODE],
            ),
            "nim-nano": Endpoint(
                name="Hosted Nano",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                capabilities=[Capability.CHAT, Capability.CODE],
            ),
        },
        formations={
            "super-nano": Formation(
                name="Super-Nano",
                description="Hosted pair",
                slots=[
                    FormationSlot(
                        endpoint="nim-super",
                        role=FormationRole.EXECUTOR,
                    ),
                    FormationSlot(
                        endpoint="nim-nano",
                        role=FormationRole.FAST,
                    ),
                ],
            ),
        },
        permissions=ToolPermissions(),
    )


class TestHasSparkEndpoints:
    def test_spark_config(self, spark_config):
        assert _has_spark_endpoints(spark_config) is True

    def test_no_spark_config(self, no_spark_config):
        assert _has_spark_endpoints(no_spark_config) is False

    def test_sglang_spark_config(self):
        config = NeMoCodeConfig(
            default_endpoint="spark-sglang-super",
            endpoints={
                "spark-sglang-super": Endpoint(
                    name="Spark SGLang Super",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8000/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT, Capability.CODE],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _has_spark_endpoints(config) is True


class TestRouteToFormation:
    def test_complex_prefers_spark(self, spark_config):
        # A complex task on Spark should prefer the local spark formation
        result = route_to_formation(
            "refactor the authentication module across all services",
            spark_config,
        )
        assert result == "spark"

    def test_complex_no_spark_uses_super_nano(self, no_spark_config):
        result = route_to_formation(
            "refactor the authentication module across all services",
            no_spark_config,
        )
        assert result == "super-nano"


class TestGetAutoEndpoint:
    def test_simple_prefers_spark_nano(self, spark_config):
        result = get_auto_endpoint("what is this?", spark_config)
        assert result == "spark-nim-nano9b"

    def test_simple_no_spark_uses_hosted_nano(self, no_spark_config):
        result = get_auto_endpoint("what is this?", no_spark_config)
        assert result == "nim-nano"

    def test_complex_prefers_spark_super(self, spark_config):
        # Non-simple tasks on Spark should use local Super
        result = get_auto_endpoint(
            "add error handling to the parser and write tests",
            spark_config,
        )
        assert result == "spark-nim-super"

    def test_complex_no_spark_uses_default(self, no_spark_config):
        result = get_auto_endpoint(
            "add error handling to the parser and write tests",
            no_spark_config,
        )
        # Should return None (use default)
        assert result is None

    def test_prefers_sglang_when_only_sglang_is_available(self):
        config = NeMoCodeConfig(
            default_endpoint="spark-sglang-super",
            endpoints={
                "spark-sglang-super": Endpoint(
                    name="Spark SGLang Super",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8000/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT, Capability.CODE],
                ),
                "spark-sglang-nano9b": Endpoint(
                    name="Spark SGLang Nano 9B",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8001/v1",
                    model_id="nvidia/nemotron-nano-9b-v2",
                    capabilities=[Capability.CHAT, Capability.CODE],
                ),
            },
            formations={
                "spark-sglang": Formation(
                    name="DGX Spark (SGLang)",
                    description="SGLang local",
                    slots=[
                        FormationSlot(
                            endpoint="spark-sglang-super",
                            role=FormationRole.EXECUTOR,
                        ),
                    ],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert get_auto_endpoint("what is this?", config) == "spark-sglang-nano9b"
        assert (
            get_auto_endpoint(
                "refactor the authentication module across all services",
                config,
            )
            == "spark-sglang-super"
        )
        assert (
            route_to_formation("refactor the authentication module across all services", config)
            == "spark-sglang"
        )
