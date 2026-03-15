# TensorRT

> NVIDIA's general deep learning inference optimizer.

## Canonical URL

`https://docs.nvidia.com/deeplearning/tensorrt/index.html`

## Overview

TensorRT is NVIDIA's SDK for optimizing deep learning models for inference. It compiles models into optimized engines that leverage GPU-specific features (Tensor Cores, reduced precision) for maximum throughput and minimum latency.

Current version: TensorRT 10.15.1

## Key Capabilities

- **Layer Fusion**: Combine multiple operations into single kernels
- **Precision Calibration**: FP16, INT8, FP8 with accuracy preservation
- **Dynamic Shapes**: Handle variable input dimensions
- **Kernel Auto-Tuning**: Select optimal kernels per GPU architecture
- **Memory Optimization**: Efficient memory pooling and reuse

## Supported Input Formats

| Format | Description |
|---|---|
| ONNX | Industry-standard model interchange |
| PyTorch (via Torch-TensorRT) | Direct PyTorch integration |
| TensorFlow (via TF-TRT) | TensorFlow integration |
| Custom Plugins | User-defined layers |

## APIs

| API | Language | Description |
|---|---|---|
| C++ API | C++ | Full-featured, production use |
| Python API | Python | Rapid prototyping |

## Tools

| Tool | Description |
|---|---|
| **trtexec** | Command-line benchmarking and engine building |
| **Polygraphy** | Model debugging and accuracy analysis |
| **ONNX GraphSurgeon** | ONNX graph modification |

## Optimization Workflow

```
Model (PyTorch/TF/ONNX)
     ↓
Export to ONNX (optional)
     ↓
TensorRT Builder (optimization + compilation)
     ↓
TensorRT Engine (optimized for target GPU)
     ↓
TensorRT Runtime (inference execution)
```

## Relationship to Other Components

| Component | Relationship |
|---|---|
| **TensorRT-LLM** | Uses TensorRT under the hood for LLM-specific optimizations |
| **Triton** | Serves TensorRT engines via `tensorrt` backend |
| **NIM** | NIM containers include TensorRT-optimized models |
| **Torch-TensorRT** | Integrates TensorRT into PyTorch workflows |

## CUDA Math Libraries

TensorRT leverages NVIDIA's CUDA math libraries:

| Library | Purpose |
|---|---|
| cuBLAS | Matrix multiplication |
| cuDNN | Neural network primitives |
| cuSPARSE | Sparse operations |
| cuFFT | Fourier transforms |
| cuRAND | Random number generation |
