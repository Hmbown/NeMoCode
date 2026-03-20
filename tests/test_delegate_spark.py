# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for Spark-aware endpoint selection in delegate.py."""

from __future__ import annotations

import pytest

from nemocode.config.schema import (
    Capability,
    Endpoint,
    EndpointTier,
    NeMoCodeConfig,
    ToolPermissions,
)
from nemocode.tools.delegate import _pick_endpoint


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
                capabilities=[Capability.CHAT],
            ),
            "spark-nim-nano9b": Endpoint(
                name="Spark NIM Nano 9B",
                tier=EndpointTier.LOCAL_NIM,
                base_url="http://localhost:8002/v1",
                model_id="nvidia/nemotron-nano-9b-v2",
                capabilities=[Capability.CHAT],
            ),
            "nim-super": Endpoint(
                name="Hosted Super",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                capabilities=[Capability.CHAT],
            ),
            "nim-nano": Endpoint(
                name="Hosted Nano",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                capabilities=[Capability.CHAT],
            ),
        },
        permissions=ToolPermissions(),
    )


@pytest.fixture
def hosted_only_config() -> NeMoCodeConfig:
    """Config with only hosted endpoints (no Spark)."""
    return NeMoCodeConfig(
        default_endpoint="nim-super",
        endpoints={
            "nim-super": Endpoint(
                name="Hosted Super",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                capabilities=[Capability.CHAT],
            ),
            "nim-nano": Endpoint(
                name="Hosted Nano",
                tier=EndpointTier.DEV_HOSTED,
                base_url="https://integrate.api.nvidia.com/v1",
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                capabilities=[Capability.CHAT],
            ),
        },
        permissions=ToolPermissions(),
    )


class TestPickEndpointSpark:
    def test_nano9b_prefers_spark(self, spark_config):
        result = _pick_endpoint(spark_config, "nano-9b")
        assert result == "spark-nim-nano9b"

    def test_super_prefers_spark(self, spark_config):
        result = _pick_endpoint(spark_config, "super")
        assert result == "spark-nim-super"

    def test_nano_falls_back_to_hosted(self, hosted_only_config):
        result = _pick_endpoint(hosted_only_config, "nano")
        assert result == "nim-nano"

    def test_super_falls_back_to_hosted(self, hosted_only_config):
        result = _pick_endpoint(hosted_only_config, "super")
        assert result == "nim-super"

    def test_unknown_tier_uses_default(self, hosted_only_config):
        result = _pick_endpoint(hosted_only_config, "unknown-tier")
        assert result == "nim-super"  # default endpoint

    def test_explicit_endpoint_wins(self, spark_config):
        result = _pick_endpoint(spark_config, ["nano-9b", "super"], explicit_endpoint="nim-super")
        assert result == "nim-super"

    def test_hosted_default_prefers_hosted_even_if_local_exists(self):
        config = NeMoCodeConfig(
            default_endpoint="nim-super",
            endpoints={
                "spark-sglang-nano9b": Endpoint(
                    name="Spark SGLang Nano 9B",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8001/v1",
                    model_id="nvidia/nemotron-nano-9b-v2",
                    capabilities=[Capability.CHAT],
                ),
                "nim-nano-9b": Endpoint(
                    name="Hosted Nano 9B",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/nemotron-nano-9b-v2",
                    capabilities=[Capability.CHAT],
                ),
                "nim-super": Endpoint(
                    name="Hosted Super",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _pick_endpoint(config, "nano-9b") == "nim-nano-9b"


class TestPickEndpointVllm:
    def test_vllm_spark_endpoints(self):
        """vLLM Spark endpoints should also be preferred."""
        config = NeMoCodeConfig(
            default_endpoint="spark-vllm-super",
            endpoints={
                "spark-vllm-super": Endpoint(
                    name="Spark vLLM Super",
                    tier=EndpointTier.LOCAL_VLLM,
                    base_url="http://localhost:8000/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
                "spark-vllm-nano9b": Endpoint(
                    name="Spark vLLM Nano 9B",
                    tier=EndpointTier.LOCAL_VLLM,
                    base_url="http://localhost:8001/v1",
                    model_id="nvidia/nemotron-nano-9b-v2",
                    capabilities=[Capability.CHAT],
                ),
                "nim-super": Endpoint(
                    name="Hosted Super",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _pick_endpoint(config, "super") == "spark-vllm-super"
        assert _pick_endpoint(config, "nano-9b") == "spark-vllm-nano9b"


class TestPickEndpointSGLang:
    def test_sglang_spark_endpoints(self):
        """SGLang Spark endpoints should also be preferred."""
        config = NeMoCodeConfig(
            default_endpoint="spark-sglang-super",
            endpoints={
                "spark-sglang-super": Endpoint(
                    name="Spark SGLang Super",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8000/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
                "spark-sglang-nano9b": Endpoint(
                    name="Spark SGLang Nano 9B",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8001/v1",
                    model_id="nvidia/nemotron-nano-9b-v2",
                    capabilities=[Capability.CHAT],
                ),
                "nim-super": Endpoint(
                    name="Hosted Super",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _pick_endpoint(config, "super") == "spark-sglang-super"
        assert _pick_endpoint(config, "nano-9b") == "spark-sglang-nano9b"

    def test_sglang_local_nano4b_is_preferred(self):
        config = NeMoCodeConfig(
            default_endpoint="local-sglang-nano4b",
            endpoints={
                "local-sglang-nano4b": Endpoint(
                    name="Local SGLang Nano 4B",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:9001/v1",
                    model_id="nvidia/llama-3.1-nemotron-nano-4b-v1.1",
                    capabilities=[Capability.CHAT],
                ),
                "nim-nano-4b": Endpoint(
                    name="Hosted Nano 4B",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/llama-3.1-nemotron-nano-4b-v1.1",
                    capabilities=[Capability.CHAT],
                ),
                "nim-super": Endpoint(
                    name="Hosted Super",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _pick_endpoint(config, ["nano-4b", "super"]) == "local-sglang-nano4b"


class TestPickEndpointTrtLlm:
    def test_trt_llm_spark_endpoints(self):
        config = NeMoCodeConfig(
            default_endpoint="spark-trt-llm-super",
            endpoints={
                "spark-trt-llm-super": Endpoint(
                    name="Spark TensorRT-LLM Super 120B",
                    tier=EndpointTier.LOCAL_TRT_LLM,
                    base_url="http://localhost:8000/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
                "spark-trt-llm-nano4b": Endpoint(
                    name="Spark TensorRT-LLM Nano 4B",
                    tier=EndpointTier.LOCAL_TRT_LLM,
                    base_url="http://localhost:8001/v1",
                    model_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
                    capabilities=[Capability.CHAT],
                ),
                "nim-super": Endpoint(
                    name="Hosted Super",
                    tier=EndpointTier.DEV_HOSTED,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _pick_endpoint(config, "super") == "spark-trt-llm-super"
        assert _pick_endpoint(config, ["nano-4b", "super"]) == "spark-trt-llm-nano4b"

    def test_default_local_family_prefers_trt_llm_over_other_local_backends(self):
        config = NeMoCodeConfig(
            default_endpoint="spark-trt-llm-super",
            endpoints={
                "spark-sglang-super": Endpoint(
                    name="Spark SGLang Super",
                    tier=EndpointTier.LOCAL_SGLANG,
                    base_url="http://localhost:8000/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
                "spark-trt-llm-super": Endpoint(
                    name="Spark TensorRT-LLM Super 120B",
                    tier=EndpointTier.LOCAL_TRT_LLM,
                    base_url="http://localhost:8100/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
                "spark-vllm-super": Endpoint(
                    name="Spark vLLM Super",
                    tier=EndpointTier.LOCAL_VLLM,
                    base_url="http://localhost:8200/v1",
                    model_id="nvidia/nemotron-3-super-120b-a12b",
                    capabilities=[Capability.CHAT],
                ),
            },
            permissions=ToolPermissions(),
        )
        assert _pick_endpoint(config, "super") == "spark-trt-llm-super"
