# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM Speech providers — ASR (Parakeet) and TTS (FastPitch).

Parakeet ASR: Transcribe audio files to text via /v1/audio/transcriptions.
FastPitch TTS: Synthesize text to audio via /v1/audio/speech.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from nemocode.providers import NIMProviderBase

logger = logging.getLogger(__name__)


class NIMASRProvider(NIMProviderBase):
    """Automatic Speech Recognition via NVIDIA Parakeet NIM."""

    async def transcribe(
        self,
        audio_path: Path,
        language: str = "en",
        response_format: str = "json",
    ) -> dict[str, Any]:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, FLAC, MP3, etc.).
            language: ISO 639-1 language code.
            response_format: Output format — json, text, verbose_json.

        Returns:
            Dict with 'text' key containing the transcription.
        """
        url = f"{self._base_url}/audio/transcriptions"
        mime_types = {
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
        }
        suffix = audio_path.suffix.lower()
        content_type = mime_types.get(suffix, "audio/wav")

        async with httpx.AsyncClient(timeout=120) as client:
            with audio_path.open("rb") as f:
                files = {"file": (audio_path.name, f, content_type)}
                data = {
                    "model": self.endpoint.model_id,
                    "language": language,
                    "response_format": response_format,
                }
                resp = await client.post(
                    url,
                    files=files,
                    data=data,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()


class NIMTTSProvider(NIMProviderBase):
    """Text-to-Speech via NVIDIA FastPitch + HiFi-GAN NIM."""

    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
        voice: str = "English-US.Female-1",
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            output_path: If provided, write audio bytes to this file.
            voice: Voice identifier.
            response_format: Audio format — wav, mp3, opus, flac.
            speed: Playback speed multiplier (0.25–4.0).

        Returns:
            Raw audio bytes.
        """
        url = f"{self._base_url}/audio/speech"
        body = {
            "model": self.endpoint.model_id,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=self._headers())
            resp.raise_for_status()
            audio_bytes = resp.content

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            logger.info("Wrote %d bytes to %s", len(audio_bytes), output_path)

        return audio_bytes
