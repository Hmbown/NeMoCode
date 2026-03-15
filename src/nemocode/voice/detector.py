# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Voice capability detection — check audio hardware and available backends."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from nemocode.core.hardware import detect_hardware


@dataclass
class VoiceCapabilities:
    has_microphone: bool = False
    has_speakers: bool = False
    stt_backend: str = "none"  # riva | whisper | system | none
    tts_backend: str = "none"  # riva | piper | system | none
    available: bool = False
    reason: str = ""


def detect_voice_capabilities() -> VoiceCapabilities:
    """Detect voice mode availability and best backends."""
    hw = detect_hardware()
    caps = VoiceCapabilities(
        has_microphone=hw.has_microphone,
        has_speakers=hw.has_speakers,
    )

    if not hw.has_microphone:
        caps.reason = "No microphone detected"
        return caps

    # Detect STT backend
    caps.stt_backend = _detect_stt_backend()

    # Detect TTS backend
    caps.tts_backend = _detect_tts_backend()

    if caps.stt_backend == "none":
        caps.reason = "No STT backend available. Install faster-whisper: pip install faster-whisper"
        return caps

    caps.available = True
    caps.reason = f"STT: {caps.stt_backend}, TTS: {caps.tts_backend}"
    return caps


def _detect_stt_backend() -> str:
    """Detect best available STT backend."""
    import importlib.util

    # Riva NIM STT requires grpcio + a running Riva endpoint.
    # Detection deferred until Riva NIM SDK provides a discovery API.

    # 1. Check for faster-whisper
    if importlib.util.find_spec("faster_whisper"):
        return "whisper"

    # 3. System fallback
    import platform

    if platform.system() == "Darwin":
        return "system"  # macOS has built-in speech recognition

    return "none"


def _detect_tts_backend() -> str:
    """Detect best available TTS backend."""
    # Riva NIM TTS requires grpcio + a running Riva endpoint.
    # Detection deferred until Riva NIM SDK provides a discovery API.

    # 1. Check for piper-tts
    if shutil.which("piper"):
        return "piper"

    # 3. System fallback
    import platform

    if platform.system() == "Darwin" and shutil.which("say"):
        return "system"
    if platform.system() == "Linux" and shutil.which("espeak"):
        return "system"

    return "none"
