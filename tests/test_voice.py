# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test voice mode — STT/TTS fallback chain and detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nemocode.voice.detector import (
    _detect_stt_backend,
    _detect_tts_backend,
    detect_voice_capabilities,
)


class TestVoiceDetection:
    def test_no_microphone(self):
        with patch("nemocode.voice.detector.detect_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(has_microphone=False, has_speakers=True)
            caps = detect_voice_capabilities()
            assert caps.available is False
            assert "microphone" in caps.reason.lower()

    def test_with_microphone_and_whisper(self):
        with patch("nemocode.voice.detector.detect_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(has_microphone=True, has_speakers=True)
            with patch("nemocode.voice.detector._detect_stt_backend", return_value="whisper"):
                with patch("nemocode.voice.detector._detect_tts_backend", return_value="system"):
                    caps = detect_voice_capabilities()
                    assert caps.available is True
                    assert caps.stt_backend == "whisper"

    def test_no_stt_backend(self):
        with patch("nemocode.voice.detector.detect_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(has_microphone=True, has_speakers=True)
            with patch("nemocode.voice.detector._detect_stt_backend", return_value="none"):
                caps = detect_voice_capabilities()
                assert caps.available is False


class TestSTTDetection:
    def test_whisper_available(self):
        mock_spec = MagicMock()
        with patch(
            "importlib.util.find_spec",
            side_effect=lambda x: mock_spec if x == "faster_whisper" else None,
        ):
            result = _detect_stt_backend()
            assert result == "whisper"


class TestTTSDetection:
    def test_system_tts_macos(self):
        with patch("platform.system", return_value="Darwin"):
            with patch(
                "shutil.which", side_effect=lambda x: "/usr/bin/say" if x == "say" else None
            ):
                result = _detect_tts_backend()
                assert result == "system"
