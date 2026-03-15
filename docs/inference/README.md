# Inference Stack

> NVIDIA's inference optimization and serving technologies.

## Stack Overview

```
                    Application Layer
                         ↓
              ┌─────────────────────┐
              │   NIM Containers    │  ← Packaged, ready-to-deploy
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   NVIDIA Dynamo     │  ← Distributed inference orchestration
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   Triton Server     │  ← Multi-framework model serving
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   TensorRT-LLM      │  ← LLM-specific optimization
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   TensorRT          │  ← General inference optimization
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   CUDA / cuDNN      │  ← GPU compute foundation
              └─────────────────────┘
```

## Components

| Component | Doc | Description |
|---|---|---|
| [Triton](triton.md) | `docs/inference/triton.md` | Multi-framework model serving platform |
| [TensorRT-LLM](tensorrt-llm.md) | `docs/inference/tensorrt-llm.md` | LLM inference optimization engine |
| [Dynamo](dynamo.md) | `docs/inference/dynamo.md` | Distributed inference orchestration |
| [TensorRT](tensorrt.md) | `docs/inference/tensorrt.md` | General deep learning inference optimizer |

## When to Use What

| Scenario | Use |
|---|---|
| Deploy a model quickly | **NIM** (includes everything pre-optimized) |
| Serve multiple model types | **Triton** (supports TensorRT, PyTorch, ONNX, etc.) |
| Optimize LLM inference | **TensorRT-LLM** |
| Distributed LLM serving | **Dynamo** + TensorRT-LLM |
| Optimize non-LLM models | **TensorRT** |
| Custom serving logic | **Triton** with Python backend or BLS |

## Canonical URLs

| Resource | URL |
|---|---|
| Triton | `https://docs.nvidia.com/deeplearning/triton-inference-server/` |
| TensorRT-LLM | `https://nvidia.github.io/TensorRT-LLM/` |
| Dynamo | `https://docs.nvidia.com/dynamo/` |
| TensorRT | `https://docs.nvidia.com/deeplearning/tensorrt/index.html` |
| CUDA | `https://docs.nvidia.com/cuda/doc/index.html` |
| cuDNN | `https://docs.nvidia.com/cudnn/index.html` |
