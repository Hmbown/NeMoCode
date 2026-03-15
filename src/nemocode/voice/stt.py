# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Speech-to-text abstraction — Riva NIM -> faster-whisper -> system fallback."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from abc import ABC, abstractmethod
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # 16-bit
_CHANNELS = 1


class STTProvider(ABC):
    """Abstract STT provider."""

    @abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        ...

    @abstractmethod
    async def stream_transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream transcription from audio chunks."""
        ...


class WhisperSTT(STTProvider):
    """Local STT using faster-whisper."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, compute_type="int8")
        return self._model

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file. Runs in thread pool to avoid blocking event loop."""
        model = self._load_model()
        segments, _ = await asyncio.to_thread(model.transcribe, audio_path)
        return " ".join(s.text for s in segments).strip()

    async def stream_transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Buffer audio chunks and transcribe periodically."""
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) > _SAMPLE_RATE * _SAMPLE_WIDTH * 3:  # ~3 seconds
                text = await self._transcribe_buffer(buffer)
                if text:
                    yield text
                buffer.clear()

        if buffer:
            text = await self._transcribe_buffer(buffer)
            if text:
                yield text

    async def _transcribe_buffer(self, buffer: bytearray) -> str:
        """Write buffer as proper WAV and transcribe it."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(_CHANNELS)
                wf.setsampwidth(_SAMPLE_WIDTH)
                wf.setframerate(_SAMPLE_RATE)
                wf.writeframes(bytes(buffer))
        try:
            return await self.transcribe(path)
        finally:
            import os

            os.unlink(path)


class SystemSTT(STTProvider):
    """System-level STT using macOS speech recognition."""

    async def transcribe(self, audio_path: str) -> str:
        import platform

        if platform.system() != "Darwin":
            raise NotImplementedError("System STT only available on macOS")

        # Use macOS NSSpeechRecognizer via osascript (no external deps)
        proc = await asyncio.create_subprocess_exec(
            "say", "--interactive", audio_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    async def stream_transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        raise NotImplementedError("System STT does not support streaming")


def get_stt_provider(backend: str = "auto") -> STTProvider:
    """Get the best available STT provider."""
    if backend in ("whisper", "auto"):
        try:
            return WhisperSTT()
        except ImportError:
            pass

    if backend in ("system", "auto"):
        return SystemSTT()

    raise RuntimeError(f"No STT backend available (requested: {backend})")
