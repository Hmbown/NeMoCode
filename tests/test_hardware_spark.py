# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for DGX Spark hardware detection and recommendations."""

from __future__ import annotations

import pytest

from nemocode.config.schema import Manifest, NemotronArch
from nemocode.core.hardware import (
    GPUInfo,
    HardwareProfile,
    _is_dgx_spark_gpu,
)


@pytest.fixture
def spark_profile() -> HardwareProfile:
    """Properly configured DGX Spark profile with is_dgx_spark=True."""
    return HardwareProfile(
        gpus=[GPUInfo(name="NVIDIA GB10", vram_gb=128, index=0)],
        total_vram_gb=128,
        cpu_cores=12,
        ram_gb=128,
        has_microphone=True,
        has_speakers=True,
        platform_name="Linux",
        is_dgx_spark=True,
        unified_memory_gb=128,
    )


@pytest.fixture
def rtx4090_profile() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(name="NVIDIA GeForce RTX 4090", vram_gb=24, index=0)],
        total_vram_gb=24,
        cpu_cores=16,
        ram_gb=64,
        platform_name="Linux",
    )


class TestIsDgxSparkGpu:
    def test_gb10(self):
        assert _is_dgx_spark_gpu("NVIDIA GB10") is True

    def test_dgx_spark_name(self):
        assert _is_dgx_spark_gpu("DGX Spark GPU") is True

    def test_grace_blackwell(self):
        assert _is_dgx_spark_gpu("Grace Blackwell Superchip") is True

    def test_not_spark(self):
        assert _is_dgx_spark_gpu("NVIDIA GeForce RTX 4090") is False

    def test_case_insensitive(self):
        assert _is_dgx_spark_gpu("nvidia gb10") is True


class TestEffectiveGpuMemory:
    def test_spark_uses_75_percent(self, spark_profile):
        # 128GB * 0.75 = 96GB
        assert spark_profile.effective_gpu_memory_gb == 96.0

    def test_standard_gpu_uses_vram(self, rtx4090_profile):
        assert rtx4090_profile.effective_gpu_memory_gb == 24.0

    def test_no_gpu_zero(self):
        p = HardwareProfile(gpus=[], total_vram_gb=0)
        assert p.effective_gpu_memory_gb == 0.0


class TestSparkRecommendations:
    def test_spark_recommends_spark_formation(self, spark_profile):
        assert spark_profile.recommend_formation() == "spark"

    def test_spark_can_run_super(self, spark_profile):
        manifest = Manifest(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            display_name="Super",
            arch=NemotronArch.NEMOTRON_3,
            min_gpu_memory_gb=80,
        )
        can, reason = spark_profile.can_run_model(manifest)
        assert can is True
        assert "unified memory" in reason

    def test_spark_cant_run_ultra(self, spark_profile):
        manifest = Manifest(
            model_id="nvidia/nemotron-3-ultra",
            display_name="Ultra",
            arch=NemotronArch.NEMOTRON_3,
            min_gpu_memory_gb=640,
        )
        can, reason = spark_profile.can_run_model(manifest)
        assert can is False
        assert "unified memory" in reason

    def test_spark_local_models_include_nano4b(self, spark_profile):
        models = spark_profile.recommend_local_models()
        assert "nvidia/nemotron-3-super-120b-a12b" in models
        assert "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8" in models
        assert "nvidia/nemotron-nano-9b-v2" in models
        assert "nvidia/llama-nemotron-reasoning-super-49b" not in models
        assert "nvidia/llama-nemotron-reasoning-8b" not in models
        assert "nvidia/llama-nemotron-embed-1b-v2" in models

    def test_rtx4090_local_models(self, rtx4090_profile):
        models = rtx4090_profile.recommend_local_models()
        assert "nvidia/nemotron-3-nano-30b-a3b" in models
        assert "nvidia/nemotron-3-super-120b-a12b" not in models


class TestSparkConcurrentConfigs:
    def test_spark_has_configs(self, spark_profile):
        configs = spark_profile.spark_concurrent_configs()
        assert len(configs) == 4
        models = [c["model_id"] for c in configs]
        assert "nvidia/nemotron-3-super-120b-a12b" in models
        assert "nvidia/nemotron-nano-9b-v2" in models

    def test_non_spark_empty(self, rtx4090_profile):
        assert rtx4090_profile.spark_concurrent_configs() == []


class TestSparkSummary:
    def test_spark_summary_mentions_spark(self, spark_profile):
        s = spark_profile.summary()
        assert "DGX Spark" in s
        assert "Unified Memory" in s
        assert "128" in s

    def test_spark_to_dict_roundtrip(self, spark_profile):
        d = spark_profile.to_dict()
        assert d["is_dgx_spark"] is True
        assert d["unified_memory_gb"] == 128
