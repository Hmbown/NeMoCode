# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Hardware detection and capability awareness.

Detects NVIDIA GPUs, RAM, audio devices, and recommends formations/models.
Special handling for DGX Spark (GB10 Grace Blackwell, 128GB unified memory).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from nemocode.config.schema import Manifest

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.environ.get("NEMOCODE_CACHE_DIR", "~/.cache/nemocode")).expanduser()
_CACHE_FILE = _CACHE_DIR / "hardware.json"
_CACHE_TTL_SECONDS = 3600  # 1 hour

# DGX Spark identifiers — GB10 Grace Blackwell Superchip
_DGX_SPARK_GPU_NAMES = {"gb10", "dgx spark", "grace blackwell"}


@dataclass
class GPUInfo:
    name: str = ""
    vram_gb: float = 0.0
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    index: int = 0
    utilization_pct: float = 0.0
    temperature_c: int = 0


@dataclass
class HardwareProfile:
    gpus: list[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    cpu_cores: int = 0
    ram_gb: float = 0.0
    has_microphone: bool = False
    has_speakers: bool = False
    platform_name: str = ""
    is_dgx_spark: bool = False
    unified_memory_gb: float = 0.0  # Spark's shared CPU/GPU memory

    @property
    def effective_gpu_memory_gb(self) -> float:
        """Total GPU-usable memory, accounting for unified memory on Spark.

        DGX Spark shares 128GB between CPU and GPU. In practice, ~96-100GB
        is available for model inference when the system isn't loaded.
        Standard GPUs report VRAM directly.
        """
        if self.is_dgx_spark and self.unified_memory_gb > 0:
            # Conservative estimate: 75% of unified memory available for inference
            return self.unified_memory_gb * 0.75
        return self.total_vram_gb

    def can_run_model(self, manifest: Manifest) -> tuple[bool, str]:
        """Check if this hardware can run a model locally."""
        if not manifest.min_gpu_memory_gb:
            return True, "No GPU requirement specified"

        if not self.gpus:
            return False, "No NVIDIA GPU detected"

        min_gpus = manifest.min_gpus or 1
        if len(self.gpus) < min_gpus:
            return False, f"Need {min_gpus} GPU(s), have {len(self.gpus)}"

        available = self.effective_gpu_memory_gb
        if available < manifest.min_gpu_memory_gb:
            mem_type = "unified memory" if self.is_dgx_spark else "VRAM"
            return False, (
                f"Need {manifest.min_gpu_memory_gb}GB, have {available:.0f}GB {mem_type} available"
            )

        if self.is_dgx_spark:
            return True, (f"{available:.0f}GB unified memory available (128GB shared CPU/GPU)")

        return True, f"{available:.0f}GB VRAM available across {len(self.gpus)} GPU(s)"

    def recommend_formation(self) -> str:
        """Suggest the best formation for this hardware."""
        if not self.gpus:
            return "solo"  # Hosted only

        # DGX Spark: always recommend the Spark formation
        if self.is_dgx_spark:
            return "spark"

        available = self.effective_gpu_memory_gb
        if available >= 640:
            return "ultra-swarm"
        elif available >= 80:
            return "super-nano"  # Can run Super locally + Nano
        elif available >= 24:
            return "local"  # Can run Nano locally
        else:
            return "solo"  # GPU too small for any model, use hosted

    def recommend_local_models(self) -> list[str]:
        """Which models can run locally on this hardware."""
        models: list[str] = []
        available = self.effective_gpu_memory_gb

        if available >= 640:
            models.append("nvidia/nemotron-3-ultra")
        if available >= 80:
            models.append("nvidia/nemotron-3-super-120b-a12b")
        if available >= 48:
            models.append("nvidia/llama-nemotron-reasoning-super-49b")
        if available >= 24:
            models.append("nvidia/nemotron-3-nano-30b-a3b")
        if available >= 16:
            models.append("nvidia/nemotron-nano-12b-v2-vl")
        if available >= 10:
            models.append("nvidia/nemotron-nano-9b-v2")
            models.append("nvidia/llama-nemotron-reasoning-8b")
            models.append("nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
        if available >= 4:
            models.append("nvidia/llama-nemotron-embed-1b-v2")
            models.append("nvidia/llama-nemotron-rerank-1b-v2")
            models.append("nvidia/parakeet-ctc-1.1b")
            models.append("nvidia/fastpitch-hifigan-tts")

        return models

    def spark_concurrent_configs(self) -> list[dict[str, str]]:
        """Recommended concurrent container configs for DGX Spark.

        Returns list of (model_id, port, memory_note) dicts showing
        which models can run simultaneously on Spark's 128GB.
        """
        if not self.is_dgx_spark:
            return []

        # Proven concurrent configs for 128GB unified memory.
        # Super ~80GB + Nano 9B ~10GB + Embed ~2GB + Rerank ~2GB = ~94GB
        return [
            {
                "model_id": "nvidia/nemotron-3-super-120b-a12b",
                "port": "8000",
                "memory": "~80GB",
                "role": "Main brain",
            },
            {
                "model_id": "nvidia/nemotron-nano-9b-v2",
                "port": "8002",
                "memory": "~10GB",
                "role": "Mini-agent worker",
            },
            {
                "model_id": "nvidia/llama-nemotron-embed-1b-v2",
                "port": "8004",
                "memory": "~2GB",
                "role": "Embedding",
            },
            {
                "model_id": "nvidia/llama-nemotron-rerank-1b-v2",
                "port": "8005",
                "memory": "~2GB",
                "role": "Reranking",
            },
        ]

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [f"Platform: {self.platform_name}"]
        lines.append(f"CPU cores: {self.cpu_cores}")
        lines.append(f"RAM: {self.ram_gb:.1f} GB")

        if self.is_dgx_spark:
            lines.append("")
            lines.append("DGX Spark (GB10 Grace Blackwell Superchip)")
            lines.append(f"  Unified Memory: {self.unified_memory_gb:.0f} GB (shared CPU/GPU)")
            lines.append(f"  Effective GPU Memory: ~{self.effective_gpu_memory_gb:.0f} GB")
            lines.append("  Compute: 1 PFLOP FP4 / 1000 TOPS")
            lines.append("  CUDA Arch: SM 12.1 (Blackwell desktop)")
            for gpu in self.gpus:
                lines.append(f"  GPU: {gpu.name}")
                if gpu.driver_version:
                    lines.append(f"  Driver: {gpu.driver_version} / CUDA: {gpu.cuda_version}")
        elif self.gpus:
            lines.append(f"GPUs: {len(self.gpus)}")
            for gpu in self.gpus:
                lines.append(f"  [{gpu.index}] {gpu.name} — {gpu.vram_gb:.1f} GB VRAM")
            lines.append(f"Total VRAM: {self.total_vram_gb:.1f} GB")
        else:
            lines.append("GPUs: None detected")

        lines.append(f"Microphone: {'Yes' if self.has_microphone else 'No'}")
        lines.append(f"Speakers: {'Yes' if self.has_speakers else 'No'}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpus": [
                {
                    "name": g.name,
                    "vram_gb": g.vram_gb,
                    "compute_capability": g.compute_capability,
                    "driver_version": g.driver_version,
                    "cuda_version": g.cuda_version,
                    "index": g.index,
                }
                for g in self.gpus
            ],
            "total_vram_gb": self.total_vram_gb,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "has_microphone": self.has_microphone,
            "has_speakers": self.has_speakers,
            "platform": self.platform_name,
            "is_dgx_spark": self.is_dgx_spark,
            "unified_memory_gb": self.unified_memory_gb,
        }


def _safe_float(text: str | None, *, strip_suffix: str = "", default: float = 0.0) -> float:
    """Parse a float from nvidia-smi output, handling 'N/A' and other non-numeric values.

    nvidia-smi may report 'N/A', 'Not Supported', or empty strings for fields
    that are unavailable on certain GPU models or driver versions.
    """
    if not text:
        return default
    cleaned = text.strip()
    if strip_suffix:
        cleaned = cleaned.replace(strip_suffix, "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def _is_dgx_spark_gpu(gpu_name: str) -> bool:
    """Check if a GPU name indicates a DGX Spark."""
    name_lower = gpu_name.lower()
    return any(spark_name in name_lower for spark_name in _DGX_SPARK_GPU_NAMES)


def _detect_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        result = subprocess.run(
            [nvidia_smi, "-q", "-x"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        root = ElementTree.fromstring(result.stdout)
        gpus = []
        for i, gpu_elem in enumerate(root.findall("gpu")):
            name = (gpu_elem.findtext("product_name") or "").strip()
            fb_mem = gpu_elem.find("fb_memory_usage")
            vram_text = fb_mem.findtext("total") if fb_mem else "0 MiB"
            vram_mb = _safe_float(vram_text, strip_suffix="MiB")
            driver = (root.findtext("driver_version") or "").strip()
            cuda = (root.findtext("cuda_version") or "").strip()

            util = gpu_elem.find("utilization")
            gpu_util_text = util.findtext("gpu_util") if util else "0 %"
            gpu_util = _safe_float(gpu_util_text, strip_suffix="%")

            temp = gpu_elem.find("temperature")
            temp_text = temp.findtext("gpu_temp") if temp else "0 C"
            temp_c = int(_safe_float(temp_text, strip_suffix="C"))

            gpus.append(
                GPUInfo(
                    name=name,
                    vram_gb=vram_mb / 1024,
                    driver_version=driver,
                    cuda_version=cuda,
                    index=i,
                    utilization_pct=gpu_util,
                    temperature_c=temp_c,
                )
            )
        return gpus
    except Exception as e:
        logger.debug("GPU detection failed: %s", e)
        return []


def _detect_dgx_spark(gpus: list[GPUInfo], ram_gb: float) -> tuple[bool, float]:
    """Detect DGX Spark and its unified memory.

    Returns (is_spark, unified_memory_gb).

    DGX Spark has unified LPDDR5x memory shared between the Grace CPU
    and Blackwell GPU. nvidia-smi may report the GPU portion of this
    as fb_memory, but the actual usable pool for inference is the full
    unified memory minus OS/system overhead.
    """
    if not gpus:
        return False, 0.0

    is_spark = any(_is_dgx_spark_gpu(gpu.name) for gpu in gpus)

    if is_spark:
        # DGX Spark ships with 128GB unified LPDDR5x.
        # System RAM detection returns the full 128GB.
        # If ram_gb is close to 128, use that; otherwise fall back.
        if 120 <= ram_gb <= 136:
            return True, ram_gb
        else:
            # Fallback: assume 128GB standard config
            return True, 128.0

    # Also check for DGX Spark via system identifiers (Linux)
    try:
        dmi_path = Path("/sys/class/dmi/id/product_name")
        if dmi_path.exists():
            product = dmi_path.read_text().strip().lower()
            if "dgx spark" in product or "dgx-spark" in product:
                unified = ram_gb if 120 <= ram_gb <= 136 else 128.0
                return True, unified
    except Exception:
        pass

    return False, 0.0


def _detect_ram() -> float:
    """Detect system RAM in GB."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        elif system == "Windows":
            # Use wmic
            result = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if "TotalPhysicalMemory" in line:
                    return int(line.split("=")[1]) / (1024**3)
    except Exception as e:
        logger.debug("RAM detection failed: %s", e)

    # Fallback
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 0.0


def _detect_cpu_cores() -> int:
    return os.cpu_count() or 0


def _detect_audio_devices() -> tuple[bool, bool]:
    """Detect microphone and speaker availability."""
    system = platform.system()
    has_mic = False
    has_speakers = False

    try:
        if system == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                audio_items = data.get("SPAudioDataType", [])
                for item in audio_items:
                    item_str = json.dumps(item).lower()
                    if "input" in item_str or "microphone" in item_str:
                        has_mic = True
                    if "output" in item_str or "speaker" in item_str:
                        has_speakers = True
        elif system == "Linux":
            # Check for ALSA devices
            result = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            has_mic = result.returncode == 0 and "card" in result.stdout.lower()

            result = subprocess.run(
                ["aplay", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            has_speakers = result.returncode == 0 and "card" in result.stdout.lower()
    except Exception as e:
        logger.debug("Audio detection failed: %s", e)

    # Try sounddevice as fallback
    if not (has_mic or has_speakers):
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for d in devices:
                if d.get("max_input_channels", 0) > 0:
                    has_mic = True
                if d.get("max_output_channels", 0) > 0:
                    has_speakers = True
        except Exception:
            pass

    return has_mic, has_speakers


def _cache_is_fresh() -> bool:
    """Check if the hardware cache exists and is younger than the TTL."""
    if not _CACHE_FILE.exists():
        return False
    try:
        age = time.time() - _CACHE_FILE.stat().st_mtime
        return age < _CACHE_TTL_SECONDS
    except OSError:
        return False


def detect_hardware(use_cache: bool = True) -> HardwareProfile:
    """Detect all hardware capabilities. Uses cache if available and fresh."""
    if use_cache and _cache_is_fresh():
        try:
            data = json.loads(_CACHE_FILE.read_text())
            return _from_dict(data)
        except Exception:
            pass

    gpus = _detect_gpus()
    ram = _detect_ram()
    cpu_cores = _detect_cpu_cores()
    has_mic, has_speakers = _detect_audio_devices()

    is_spark, unified_mem = _detect_dgx_spark(gpus, ram)

    profile = HardwareProfile(
        gpus=gpus,
        total_vram_gb=sum(g.vram_gb for g in gpus),
        cpu_cores=cpu_cores,
        ram_gb=ram,
        has_microphone=has_mic,
        has_speakers=has_speakers,
        platform_name=platform.system(),
        is_dgx_spark=is_spark,
        unified_memory_gb=unified_mem,
    )

    # Cache the result
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(profile.to_dict(), indent=2))
    except Exception as e:
        logger.debug("Cache write failed: %s", e)

    return profile


def _from_dict(data: dict[str, Any]) -> HardwareProfile:
    gpus = [GPUInfo(**g) for g in data.get("gpus", [])]
    return HardwareProfile(
        gpus=gpus,
        total_vram_gb=data.get("total_vram_gb", 0.0),
        cpu_cores=data.get("cpu_cores", 0),
        ram_gb=data.get("ram_gb", 0.0),
        has_microphone=data.get("has_microphone", False),
        has_speakers=data.get("has_speakers", False),
        platform_name=data.get("platform", ""),
        is_dgx_spark=data.get("is_dgx_spark", False),
        unified_memory_gb=data.get("unified_memory_gb", 0.0),
    )


def refresh_hardware() -> HardwareProfile:
    """Force re-detection of hardware (ignores cache)."""
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    return detect_hardware(use_cache=False)
