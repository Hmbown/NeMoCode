# NIM for Speech

> ASR, TTS, and Neural Machine Translation inference microservices.

## Canonical URL

`https://docs.nvidia.com/nim/speech/latest/`

## Overview

Speech NIMs provide production-ready inference for automatic speech recognition (ASR), text-to-speech (TTS), and neural machine translation (NMT). They replace the older Riva speech platform with optimized NIM containers.

## ASR (Automatic Speech Recognition)

### Available Models

| Model | Languages | Description |
|---|---|---|
| **Parakeet-TDT-0.6B-V2** | English | Top-ranked on HuggingFace OpenASR leaderboard |
| **Parakeet-CTC-1.1B** | English | CTC decoder variant |
| **Parakeet-RNNT-1.1B** | English | RNN-T decoder variant |
| **Nemotron ASR Streaming** | English | Optimized for real-time streaming |
| **Canary** | EN, ES, DE, FR | Multilingual with punctuation/capitalization |
| **Conformer CTC** | 14+ languages | General-purpose CTC model |
| **Whisper Large v3** | 100+ languages | OpenAI Whisper, optimized by NVIDIA |

### Features
- Streaming and batch modes
- Punctuation and capitalization
- Word-level timestamps
- Speaker diarization
- Custom vocabulary / hot words

### API
- gRPC (recommended for streaming)
- REST/HTTP (for batch)

## TTS (Text-to-Speech)

### Features
- Multiple voices and emotional styles
- Voice cloning from reference audio
- Batch synthesis for high-throughput
- SSML support for speech control
- Multi-language support

### Models
- NVIDIA TTS models with voice cloning
- Magpie-TTS with DPO/GRPO preference optimization
- Multiple vocoder options

## NMT (Neural Machine Translation)

- 36 language pairs supported
- Bidirectional translation
- Batch and streaming modes

## Deployment

```bash
# Example: Deploy Parakeet ASR NIM
docker run --gpus all -p 50051:50051 \
  nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:latest
```

## Migration from Riva

Speech NIMs replace Riva for new deployments. Key differences:
- NIM uses standard container deployment (vs. Riva's custom server)
- Individual model containers (vs. Riva's monolithic server)
- Simplified API surface
- Better scaling with Kubernetes
