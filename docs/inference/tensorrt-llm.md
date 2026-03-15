# TensorRT-LLM

> High-performance inference optimization for large language models.

## Canonical URLs

| Resource | URL |
|---|---|
| GitHub | `https://github.com/NVIDIA/TensorRT-LLM` |
| Documentation | `https://nvidia.github.io/TensorRT-LLM/` |

## Overview

TensorRT-LLM compiles LLMs into optimized TensorRT engines with state-of-the-art inference performance. It's the inference backend used by NIM for LLMs.

## Key Features

### Quantization
| Method | Precision | Description |
|---|---|---|
| FP8 | 8-bit float | Best balance of speed and quality (Hopper+) |
| FP4 | 4-bit float | Maximum compression (Blackwell) |
| INT4 AWQ | 4-bit int | Weight-only quantization |
| INT8 SmoothQuant | 8-bit int | Weight + activation quantization |
| INT4 GPTQ | 4-bit int | Post-training quantization |

### Attention Optimizations
- Flash Attention 2
- Paged Attention (vLLM-style)
- Fused Multi-Head Attention
- Sparse Attention

### KV Cache Management
- Paged KV cache for memory efficiency
- KV cache reuse across requests
- Prefix caching for shared prompts

### Parallelism
| Strategy | Description |
|---|---|
| Tensor Parallelism | Split layers across GPUs |
| Pipeline Parallelism | Split model stages across GPUs |
| Helix Parallelism | Optimized hybrid parallelism |

### Speculative Decoding
Use a small draft model to generate candidate tokens, verified by the large model. 2-3x speedup.

### Disaggregated Serving
Separate prefill (prompt processing) from decode (token generation) for better resource utilization.

## Supported Models

Extensive model support including:
- Llama family (2, 3, 3.1, 3.3, 4)
- Mistral / Mixtral
- DeepSeek R1, V3
- Qwen family
- GPT variants
- Falcon
- Gemma
- Phi
- Multimodal models (VLMs)
- Full list: `https://nvidia.github.io/TensorRT-LLM/` (Supported Models section)

## CLI Tools

```bash
# Benchmark
trtllm-bench --model llama-3.1-8b --tp 1

# Evaluate
trtllm-eval --model llama-3.1-8b --benchmark mmlu

# Serve (OpenAI-compatible)
trtllm-serve --model llama-3.1-8b --port 8000
```

## Model Recipes

Pre-built optimization recipes for popular models:
- DeepSeek R1
- Llama 3.3 / Llama 4
- Qwen3
- Kimi K2

## Quick Start

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(
    ["What is TensorRT-LLM?"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=256)
)
print(outputs[0].text)
```

## Integration

TensorRT-LLM engines can be served via:
- **NIM** — Pre-packaged containers (easiest)
- **Triton** — TensorRT-LLM backend
- **trtllm-serve** — Built-in OpenAI-compatible server
- **Dynamo** — Distributed orchestration
