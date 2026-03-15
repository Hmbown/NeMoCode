# NeMo AutoModel

> PyTorch SPMD training with day-0 HuggingFace model support.

## Overview

NeMo AutoModel enables training of large language models using Single Program Multiple Data (SPMD) parallelism strategies. It provides seamless integration with HuggingFace model definitions while leveraging NVIDIA's optimized distributed training infrastructure.

## Canonical URL

`https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html` (AutoModel section)

## Supported Models (as of 2026-03)

- DeepSeek V3
- Qwen3, Qwen2-VL, Qwen3.5-VL
- Gemma 3, Gemma 3n
- Llama 4 (including multimodal)
- Mistral
- Many more via HuggingFace integration

## Key Features

- **HuggingFace Compatibility**: Use HuggingFace model definitions directly
- **Distributed Training**: Automatic parallelism (Tensor, Pipeline, FSDP, Expert)
- **VLM Support**: Vision-Language Models (Qwen2-VL, Qwen3.5-VL, Llama 4, Gemma 3n)
- **Megatron-Core Integration**: Access to optimized kernels and communication primitives

## Parallelism Strategies

| Strategy | Use Case |
|---|---|
| Tensor Parallelism (TP) | Split individual layers across GPUs |
| Pipeline Parallelism (PP) | Split model layers across GPU groups |
| FSDP | Shard optimizer states and gradients |
| Expert Parallelism (EP) | Distribute MoE experts across GPUs |
| Sequence Parallelism (SP) | Split long sequences across GPUs |
