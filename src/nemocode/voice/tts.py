# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Text-to-speech abstraction — Riva NIM -> piper -> system fallback."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract TTS provider."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak text aloud."""
        ...


class PiperTTS(TTSProvider):
    """Local TTS using piper-tts."""

    async def speak(self, text: str) -> None:
        piper = shutil.which("piper")
        if not piper:
            raise RuntimeError("piper not found in PATH")

        # Generate raw audio with piper, write to temp file, play with system player
        proc = await asyncio.create_subprocess_exec(
            piper,
            "--output-raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate(input=text.encode())

        import platform

        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(stdout)
            tmp_path = f.name

        try:
            if platform.system() == "Darwin":
                # afplay needs a file, not stdin. Convert raw to playable format.
                play_proc = await asyncio.create_subprocess_exec(
                    "afplay",
                    tmp_path,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                # aplay can read from file with explicit format
                play_proc = await asyncio.create_subprocess_exec(
                    "aplay",
                    "-r",
                    "22050",
                    "-c",
                    "1",
                    "-f",
                    "S16_LE",
                    tmp_path,
                    stderr=asyncio.subprocess.PIPE,
                )
            await play_proc.wait()
        finally:
            import os

            os.unlink(tmp_path)


class SystemTTS(TTSProvider):
    """System-level TTS (macOS say, Linux espeak)."""

    async def speak(self, text: str) -> None:
        import platform

        system = platform.system()

        if system == "Darwin":
            proc = await asyncio.create_subprocess_exec("say", text)
            await proc.wait()
        elif system == "Linux":
            espeak = shutil.which("espeak") or shutil.which("espeak-ng")
            if espeak:
                proc = await asyncio.create_subprocess_exec(espeak, text)
                await proc.wait()
            else:
                raise RuntimeError("No TTS available (install espeak)")
        else:
            raise RuntimeError(f"System TTS not supported on {system}")


def get_tts_provider(backend: str = "auto") -> TTSProvider:
    """Get the best available TTS provider."""
    if backend in ("piper", "auto"):
        if shutil.which("piper"):
            return PiperTTS()

    if backend in ("system", "auto"):
        return SystemTTS()

    raise RuntimeError(f"No TTS backend available (requested: {backend})")
