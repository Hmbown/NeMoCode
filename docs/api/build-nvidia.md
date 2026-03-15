# build.nvidia.com Guide

> NVIDIA's AI model catalog and API playground.

## Canonical URL

`https://build.nvidia.com/explore/discover`

## Overview

build.nvidia.com is NVIDIA's portal for discovering, trying, and deploying AI models. It provides:

1. **Model Catalog** — Browse all available NVIDIA and partner models
2. **API Playground** — Try models interactively in the browser
3. **API Keys** — Generate keys for programmatic access
4. **Deployment Options** — Self-host via NIM or use cloud endpoints

## Model Categories

| Category | Description | Examples |
|---|---|---|
| Reasoning | LLMs for text generation and reasoning | Nemotron, Llama, Mistral, DeepSeek |
| Vision | Visual understanding models | Llama 4 multimodal, Qwen-VL |
| Visual Design | Image/3D generation | FLUX, Stable Diffusion, TRELLIS |
| Retrieval | Embedding and reranking | NV-EmbedQA, NV-RerankQA |
| Speech | ASR, TTS, translation | Parakeet, Canary, Whisper |
| Biology | Drug discovery and protein science | AlphaFold2, ProteinMPNN |
| Simulation | Physics and engineering | DoMINO |
| Climate & Weather | Earth science | FourCastNet, CorrDiff |
| Safety & Moderation | Content filtering | NemoGuard models |

## Model Providers

NVIDIA hosts models from multiple providers:
- **NVIDIA** — Nemotron, Cosmos, Parakeet, NemoGuard
- **Meta** — Llama family
- **Mistral** — Mistral, Mixtral
- **DeepSeek** — DeepSeek R1, V3
- **Alibaba** — Qwen family
- **Moonshot** — Kimi
- **MiniMax** — MiniMax models
- **Black Forest Labs** — FLUX
- **Stability AI** — Stable Diffusion

## Workflow

1. **Browse** models at `build.nvidia.com/explore/discover`
2. **Try** in the interactive playground
3. **Get API Key** for programmatic access
4. **Integrate** using OpenAI-compatible SDKs
5. **Scale** by self-hosting with NIM containers

## GPU Instances Available

For self-hosting via DGX Cloud or partner clouds:
B300, B200, H200, H100, A100, L40S, RTX PRO 6000
