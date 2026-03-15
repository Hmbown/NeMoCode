# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Voice session manager — handles push-to-talk, VAD, and streaming pipeline."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from nemocode.voice.detector import detect_voice_capabilities
from nemocode.voice.stt import STTProvider, get_stt_provider
from nemocode.voice.tts import TTSProvider, get_tts_provider

logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages a voice interaction session."""

    def __init__(
        self,
        stt_backend: str = "auto",
        tts_backend: str = "auto",
        on_transcript: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._stt: STTProvider | None = None
        self._tts: TTSProvider | None = None
        self._stt_backend = stt_backend
        self._tts_backend = tts_backend
        self._on_transcript = on_transcript
        self._active = False
        self._listening = False

    async def start(self) -> bool:
        """Initialize voice session. Returns True if successful."""
        caps = detect_voice_capabilities()
        if not caps.available:
            logger.warning("Voice mode unavailable: %s", caps.reason)
            return False

        try:
            self._stt = get_stt_provider(self._stt_backend)
            self._tts = get_tts_provider(self._tts_backend)
            self._active = True
            return True
        except Exception as e:
            logger.error("Failed to initialize voice session: %s", e)
            return False

    async def stop(self) -> None:
        """Stop the voice session."""
        self._active = False
        self._listening = False

    async def speak(self, text: str) -> None:
        """Speak text using TTS."""
        if self._tts and self._active:
            try:
                await self._tts.speak(text)
            except Exception as e:
                logger.error("TTS failed: %s", e)

    async def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file."""
        if not self._stt:
            raise RuntimeError("STT not initialized")
        return await self._stt.transcribe(audio_path)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_listening(self) -> bool:
        return self._listening
