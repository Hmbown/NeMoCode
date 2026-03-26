# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for non-text-generation model CLIs: embed, rerank, speech."""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from nemocode.cli.main import app

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


# ──────────────────────────────────────────────────────────────────────
# CLI Help Tests (smoke tests — verify commands registered and parseable)
# ──────────────────────────────────────────────────────────────────────


class TestCLIHelp:
    def test_embed_help(self):
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert "embed" in _strip_ansi(result.stdout).lower()

    def test_embed_text_help(self):
        result = runner.invoke(app, ["embed", "text", "--help"])
        assert result.exit_code == 0

    def test_embed_similarity_help(self):
        result = runner.invoke(app, ["embed", "similarity", "--help"])
        assert result.exit_code == 0
        assert "cosine" in _strip_ansi(result.stdout).lower()

    def test_rerank_help(self):
        result = runner.invoke(app, ["rerank", "--help"])
        assert result.exit_code == 0
        assert "rerank" in _strip_ansi(result.stdout).lower()

    def test_rerank_query_help(self):
        result = runner.invoke(app, ["rerank", "query", "--help"])
        assert result.exit_code == 0

    def test_speech_help(self):
        result = runner.invoke(app, ["speech", "--help"])
        assert result.exit_code == 0
        assert "speech" in _strip_ansi(result.stdout).lower()

    def test_speech_transcribe_help(self):
        result = runner.invoke(app, ["speech", "transcribe", "--help"])
        assert result.exit_code == 0
        assert "transcribe" in _strip_ansi(result.stdout).lower()

    def test_speech_synthesize_help(self):
        result = runner.invoke(app, ["speech", "synthesize", "--help"])
        assert result.exit_code == 0
        assert "synthesize" in _strip_ansi(result.stdout).lower()

    def test_speech_test_help(self):
        result = runner.invoke(app, ["speech", "test", "--help"])
        assert result.exit_code == 0


# ──────────────────────────────────────────────────────────────────────
# Embedding Provider Tests
# ──────────────────────────────────────────────────────────────────────


class TestEmbeddingProvider:
    def test_embed_returns_vectors(self):
        """NIMEmbeddingProvider.embed() should return list of float vectors."""
        from nemocode.providers.nim_embeddings import NIMEmbeddingProvider

        mock_endpoint = MagicMock()
        mock_endpoint.model_id = "nvidia/llama-nemotron-embed-1b-v2"
        mock_endpoint.base_url = "http://localhost:8000/v1"
        mock_endpoint.extra_headers = {}

        provider = NIMEmbeddingProvider(endpoint=mock_endpoint, api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }

        with patch("nemocode.providers.nim_embeddings.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            import asyncio

            result = asyncio.run(provider.embed(["hello", "world"]))

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


# ──────────────────────────────────────────────────────────────────────
# Reranking Provider Tests
# ──────────────────────────────────────────────────────────────────────


class TestRerankProvider:
    def test_rerank_returns_scored_pairs(self):
        """NIMRerankProvider.rerank() should return (index, score) tuples."""
        from nemocode.providers.nim_rerank import NIMRerankProvider

        mock_endpoint = MagicMock()
        mock_endpoint.model_id = "nvidia/llama-nemotron-rerank-1b-v2"
        mock_endpoint.base_url = "http://localhost:8000/v1"
        mock_endpoint.extra_headers = {}

        provider = NIMRerankProvider(endpoint=mock_endpoint, api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "rankings": [
                {"index": 1, "logit": 0.95},
                {"index": 0, "logit": 0.32},
            ]
        }

        with patch("nemocode.providers.nim_rerank.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            import asyncio

            result = asyncio.run(
                provider.rerank("CUDA programming", ["Python is a language", "CUDA is parallel"])
            )

        assert len(result) == 2
        assert result[0] == (1, 0.95)
        assert result[1] == (0, 0.32)


# ──────────────────────────────────────────────────────────────────────
# Speech Provider Tests
# ──────────────────────────────────────────────────────────────────────


class TestSpeechProviders:
    def test_tts_returns_audio_bytes(self):
        """NIMTTSProvider.synthesize() should return audio bytes."""
        from nemocode.providers.nim_speech import NIMTTSProvider

        mock_endpoint = MagicMock()
        mock_endpoint.model_id = "nvidia/fastpitch-hifigan-tts"
        mock_endpoint.base_url = "http://localhost:8000/v1"
        mock_endpoint.extra_headers = {}

        provider = NIMTTSProvider(endpoint=mock_endpoint, api_key="test-key")

        fake_audio = b"RIFF" + b"\x00" * 100

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.content = fake_audio

        with patch("nemocode.providers.nim_speech.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            import asyncio

            result = asyncio.run(provider.synthesize("testing"))

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:4] == b"RIFF"

    def test_tts_writes_file(self, tmp_path):
        """NIMTTSProvider.synthesize() should write to output_path."""
        from nemocode.providers.nim_speech import NIMTTSProvider

        mock_endpoint = MagicMock()
        mock_endpoint.model_id = "nvidia/fastpitch-hifigan-tts"
        mock_endpoint.base_url = "http://localhost:8000/v1"
        mock_endpoint.extra_headers = {}

        provider = NIMTTSProvider(endpoint=mock_endpoint, api_key="test-key")
        output_file = tmp_path / "output.wav"

        fake_audio = b"RIFF" + b"\x00" * 100

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.content = fake_audio

        with patch("nemocode.providers.nim_speech.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            import asyncio

            asyncio.run(provider.synthesize("testing", output_path=output_file))

        assert output_file.exists()
        assert output_file.read_bytes() == fake_audio

    def test_asr_returns_transcription(self, tmp_path):
        """NIMASRProvider.transcribe() should return text."""
        from nemocode.providers.nim_speech import NIMASRProvider

        mock_endpoint = MagicMock()
        mock_endpoint.model_id = "nvidia/parakeet-ctc-1.1b"
        mock_endpoint.base_url = "http://localhost:8000/v1"
        mock_endpoint.extra_headers = {}

        provider = NIMASRProvider(endpoint=mock_endpoint, api_key="test-key")

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "hello world"}

        with patch("nemocode.providers.nim_speech.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            import asyncio

            result = asyncio.run(provider.transcribe(audio_file))

        assert result["text"] == "hello world"


# ──────────────────────────────────────────────────────────────────────
# CLI Integration Tests (mocked providers)
# ──────────────────────────────────────────────────────────────────────


class TestEmbedCLI:
    @patch("nemocode.cli.commands.embed.Registry")
    @patch("nemocode.cli.commands.embed.load_config")
    def test_embed_text_dims(self, mock_config, mock_registry_cls):
        mock_provider = MagicMock()
        mock_provider.embed = AsyncMock(return_value=[[0.1] * 1024])
        mock_registry = MagicMock()
        mock_registry.get_embedding_provider.return_value = mock_provider
        mock_registry_cls.return_value = mock_registry

        result = runner.invoke(app, ["embed", "text", "hello world", "--dims"])
        assert result.exit_code == 0
        assert "1024" in _strip_ansi(result.stdout)

    @patch("nemocode.cli.commands.embed.Registry")
    @patch("nemocode.cli.commands.embed.load_config")
    def test_embed_similarity(self, mock_config, mock_registry_cls):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        mock_provider = MagicMock()
        mock_provider.embed = AsyncMock(return_value=[vec_a, vec_b])
        mock_registry = MagicMock()
        mock_registry.get_embedding_provider.return_value = mock_provider
        mock_registry_cls.return_value = mock_registry

        result = runner.invoke(app, ["embed", "similarity", "hello", "hello"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "1.0000" in out
        assert "Very similar" in out


class TestRerankCLI:
    @patch("nemocode.cli.commands.rerank.Registry")
    @patch("nemocode.cli.commands.rerank.load_config")
    def test_rerank_passages(self, mock_config, mock_registry_cls):
        mock_provider = MagicMock()
        mock_provider.rerank = AsyncMock(return_value=[(0, 0.95), (1, 0.12)])
        mock_registry = MagicMock()
        mock_registry.get_rerank_provider.return_value = mock_provider
        mock_registry_cls.return_value = mock_registry

        result = runner.invoke(
            app,
            [
                "rerank", "query", "CUDA programming",
                "-p", "CUDA is parallel",
                "-p", "Python is a language",
            ],
        )
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "0.9500" in out

    @patch("nemocode.cli.commands.rerank.Registry")
    @patch("nemocode.cli.commands.rerank.load_config")
    def test_rerank_json_output(self, mock_config, mock_registry_cls):
        mock_provider = MagicMock()
        mock_provider.rerank = AsyncMock(return_value=[(0, 0.9)])
        mock_registry = MagicMock()
        mock_registry.get_rerank_provider.return_value = mock_provider
        mock_registry_cls.return_value = mock_registry

        result = runner.invoke(
            app,
            ["rerank", "query", "test query", "-p", "passage one", "--json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(_strip_ansi(result.stdout))
        assert parsed[0]["score"] == 0.9
        assert parsed[0]["passage"] == "passage one"


class TestSpeechCLI:
    @patch("nemocode.providers.nim_speech.httpx.AsyncClient")
    @patch("nemocode.config.get_api_key", return_value="test-key")
    @patch("nemocode.cli.commands.speech.load_config")
    def test_synthesize_creates_output(self, mock_config, mock_api_key, mock_http, tmp_path):
        from nemocode.config.schema import Endpoint, EndpointTier

        mock_ep = Endpoint(
            name="nim-tts",
            tier=EndpointTier.DEV_HOSTED,
            base_url="http://localhost:8000/v1",
            model_id="nvidia/fastpitch-hifigan-tts",
            capabilities=[],
        )
        mock_cfg = MagicMock()
        mock_cfg.endpoints = {"nim-tts": mock_ep}
        mock_config.return_value = mock_cfg

        fake_audio = b"RIFF" + b"\x00" * 100
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.content = fake_audio

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_http.return_value = mock_client

        output_file = tmp_path / "test_output.wav"
        result = runner.invoke(
            app, ["speech", "synthesize", "testing", "-o", str(output_file)]
        )

        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "Speech Synthesized" in out
        assert "testing" in out
