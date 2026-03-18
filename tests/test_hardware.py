# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test hardware detection and recommendations."""

from __future__ import annotations

import pytest

from nemocode.config.schema import Manifest, NemotronArch
from nemocode.core.hardware import (
    GPUInfo,
    HardwareProfile,
)


@pytest.fixture
def dgx_spark_profile() -> HardwareProfile:
    """Simulate a DGX Spark with one GPU."""
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
def workstation_profile() -> HardwareProfile:
    """Simulate a workstation with 1x H100."""
    return HardwareProfile(
        gpus=[GPUInfo(name="NVIDIA H100 80GB HBM3", vram_gb=80, index=0)],
        total_vram_gb=80,
        cpu_cores=64,
        ram_gb=512,
        has_microphone=False,
        has_speakers=True,
        platform_name="Linux",
    )


@pytest.fixture
def no_gpu_profile() -> HardwareProfile:
    """Simulate a machine with no GPU."""
    return HardwareProfile(
        gpus=[],
        total_vram_gb=0,
        cpu_cores=8,
        ram_gb=16,
        has_microphone=True,
        has_speakers=True,
        platform_name="Darwin",
    )


@pytest.fixture
def super_manifest() -> Manifest:
    return Manifest(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        display_name="Nemotron 3 Super",
        arch=NemotronArch.NEMOTRON_3,
        min_gpu_memory_gb=80,
        min_gpus=1,
    )


@pytest.fixture
def nano_manifest() -> Manifest:
    return Manifest(
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        display_name="Nemotron 3 Nano",
        arch=NemotronArch.NEMOTRON_3,
        min_gpu_memory_gb=24,
        min_gpus=1,
    )


class TestCanRunModel:
    def test_h100_can_run_super(
        self, workstation_profile: HardwareProfile, super_manifest: Manifest
    ):
        can, reason = workstation_profile.can_run_model(super_manifest)
        assert can is True

    def test_no_gpu_cannot_run(self, no_gpu_profile: HardwareProfile, super_manifest: Manifest):
        can, reason = no_gpu_profile.can_run_model(super_manifest)
        assert can is False
        assert "No NVIDIA GPU" in reason

    def test_dgx_spark_can_run_nano(
        self, dgx_spark_profile: HardwareProfile, nano_manifest: Manifest
    ):
        can, reason = dgx_spark_profile.can_run_model(nano_manifest)
        assert can is True


class TestRecommendations:
    def test_no_gpu_recommends_solo(self, no_gpu_profile: HardwareProfile):
        assert no_gpu_profile.recommend_formation() == "solo"

    def test_h100_recommends_super_nano(self, workstation_profile: HardwareProfile):
        assert workstation_profile.recommend_formation() == "super-nano"

    def test_dgx_spark_recommends_spark(self, dgx_spark_profile: HardwareProfile):
        assert dgx_spark_profile.recommend_formation() == "spark"

    def test_no_gpu_no_local_models(self, no_gpu_profile: HardwareProfile):
        models = no_gpu_profile.recommend_local_models()
        assert len(models) == 0

    def test_h100_local_models(self, workstation_profile: HardwareProfile):
        models = workstation_profile.recommend_local_models()
        assert "nvidia/nemotron-3-super-120b-a12b" in models
        assert "nvidia/nemotron-3-nano-30b-a3b" in models

    def test_8xh100_recommends_ultra(self):
        profile = HardwareProfile(
            gpus=[GPUInfo(name="H100", vram_gb=80, index=i) for i in range(8)],
            total_vram_gb=640,
            cpu_cores=128,
            ram_gb=2048,
            platform_name="Linux",
        )
        assert profile.recommend_formation() == "ultra-swarm"


class TestSummary:
    def test_summary_has_gpu_info(self, workstation_profile: HardwareProfile):
        summary = workstation_profile.summary()
        assert "NVIDIA H100" in summary
        assert "80.0 GB VRAM" in summary

    def test_summary_no_gpu(self, no_gpu_profile: HardwareProfile):
        summary = no_gpu_profile.summary()
        assert "None detected" in summary

    def test_to_dict_roundtrip(self, workstation_profile: HardwareProfile):
        d = workstation_profile.to_dict()
        assert d["total_vram_gb"] == 80
        assert len(d["gpus"]) == 1
