# NeMo Export & Deploy

> Model export and deployment utilities for taking NeMo models to production.

## Canonical URL

`https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html` (Export/Deploy section)

## Overview

NeMo Export & Deploy handles the conversion of trained NeMo models into optimized formats for inference, and provides utilities for deploying them via various serving backends.

## Export Formats

| Format | Target Runtime |
|---|---|
| TensorRT-LLM | NVIDIA TensorRT-LLM engine |
| ONNX | ONNX Runtime, TensorRT |
| TorchScript | PyTorch serving |
| HuggingFace | HuggingFace Transformers ecosystem |
| vLLM | vLLM serving |

## Optimization Techniques

- **Quantization**: FP8, INT8 (SmoothQuant), INT4 (AWQ, GPTQ)
- **Pruning**: Structured and unstructured weight pruning
- **Distillation**: Knowledge distillation from larger to smaller models
- **Speculative Decoding**: Draft-target model pairs for faster inference

## Deployment Targets

| Target | Description |
|---|---|
| NIM | NVIDIA Inference Microservices (recommended for production) |
| Triton | NVIDIA Triton Inference Server |
| TensorRT-LLM | Direct TensorRT-LLM serving |
| vLLM | Open-source LLM serving |

## Workflow

1. **Train/Fine-tune** with NeMo Framework
2. **Optimize** (quantize, prune, distill)
3. **Export** to target format
4. **Deploy** via NIM, Triton, or other serving solution
5. **Monitor** with built-in observability tools
