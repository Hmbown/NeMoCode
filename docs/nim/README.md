# NVIDIA NIM (Inference Microservices)

> Pre-built, optimized inference containers for deploying AI models at scale.

## Canonical URLs

| Resource | URL |
|---|---|
| NIM Documentation Hub | `https://docs.nvidia.com/nim/index.html` |
| NIM for LLMs | `https://docs.nvidia.com/nim/large-language-models/latest/index.html` |
| NIM API Reference | `https://docs.api.nvidia.com/nim/reference` |
| build.nvidia.com (try NIMs) | `https://build.nvidia.com/explore/discover` |

## What is NIM?

NIM packages AI models with optimized inference engines (TensorRT-LLM, Triton) into containers that can be deployed anywhere — cloud, data center, or workstation. Each NIM exposes standard APIs (OpenAI-compatible for LLMs, gRPC for speech/retrieval).

## NIM Categories

| Category | Doc | Description |
|---|---|---|
| [LLMs](llm.md) | `docs/nim/llm.md` | Large Language Models — chat, completion, reasoning, tool calling |
| [Speech](speech.md) | `docs/nim/speech.md` | ASR, TTS, NMT — Parakeet, Canary, Whisper, voice cloning |
| [Retrieval](retrieval.md) | `docs/nim/retrieval.md` | Text embedding and reranking for RAG |
| [Vision & Multimodal](vision.md) | `docs/nim/vision.md` | NV-CLIP, Cosmos, Visual GenAI, Digital Human |
| [Safety](safety.md) | `docs/nim/safety.md` | NemoGuard — content safety, topic control, jailbreak detection |
| [Specialized](specialized.md) | `docs/nim/specialized.md` | BioNeMo, MONAI, Earth-2, Simulation |

## Deployment Options

| Option | Description |
|---|---|
| **Cloud API** | Free serverless endpoints at build.nvidia.com |
| **Self-hosted** | Pull NIM container from NGC, run on your GPUs |
| **Kubernetes** | Deploy via Helm charts with NIM Operator |
| **Air-gapped** | Offline deployment for restricted environments |

## Key Features

- OpenAI-compatible API for LLMs (drop-in replacement)
- Automatic TensorRT-LLM optimization
- Multi-GPU and multi-node support
- Dynamic batching and model parallelism
- LoRA adapter hot-swapping
- Structured output / guided decoding
- Function/tool calling
- Observability (Prometheus metrics, OpenTelemetry)

## GPU Support

NIMs run on: H100, H200, A100, L40S, RTX 6000 Ada, RTX PRO 6000, and DGX Spark platforms.
