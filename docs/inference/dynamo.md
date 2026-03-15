# NVIDIA Dynamo

> High-throughput, low-latency distributed inference orchestration.

## Canonical URLs

| Resource | URL |
|---|---|
| Documentation | `https://docs.nvidia.com/dynamo/` |
| GitHub | `https://github.com/ai-dynamo/dynamo` |

## Overview

Dynamo is NVIDIA's inference orchestration framework built in Rust and Python. It coordinates distributed LLM inference across multiple GPUs and nodes, optimizing for throughput and latency.

## Key Features

### Disaggregated Prefill/Decode
Separate prompt processing (prefill) from token generation (decode) across different GPU pools:
- **Prefill nodes**: Optimized for high-throughput prompt processing
- **Decode nodes**: Optimized for low-latency token generation

### Dynamic GPU Scheduling
Automatically route requests to the best available GPU based on:
- Current load
- Model placement
- Request requirements (latency vs. throughput)

### KV-Aware Routing
Route follow-up requests to the same GPU that has the KV cache, avoiding recomputation.

### Backend Support
| Backend | Description |
|---|---|
| TensorRT-LLM | Optimized NVIDIA inference |
| SGLang | Efficient LLM serving |
| vLLM | Popular open-source serving |

## Architecture

```
          Load Balancer
               ↓
     ┌─────────────────────┐
     │   Dynamo Router     │  ← Request routing & scheduling
     └──────────┬──────────┘
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
┌──────────┐         ┌──────────┐
│ Prefill  │         │  Decode  │
│  Pool    │ ──KV──→ │  Pool    │
│ (GPUs)   │  cache  │ (GPUs)   │
└──────────┘         └──────────┘
```

## Supported Features

- Multimodal models (text + image/video)
- Tool/function calling
- LoRA adapter management
- Diffusion models
- Streaming responses
- OpenAI-compatible API

## Use Cases

| Scenario | Why Dynamo |
|---|---|
| High-volume LLM serving | Disaggregated serving maximizes GPU utilization |
| Multi-model deployment | Dynamic scheduling across model variants |
| Latency-sensitive applications | KV-aware routing minimizes recomputation |
| Scale-out inference | Distributed across multiple nodes |

## Integration

Dynamo integrates with:
- **NIM** — Can be used as the orchestration layer within NIM
- **NeMo Agent Toolkit** — High-throughput backend for agent workloads
- **Kubernetes** — Cloud-native deployment with auto-scaling
