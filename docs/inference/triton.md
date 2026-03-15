# Triton Inference Server

> Multi-framework model serving platform.

## Canonical URLs

| Resource | URL |
|---|---|
| GitHub | `https://github.com/triton-inference-server/server` |
| Documentation | `https://docs.nvidia.com/deeplearning/triton-inference-server/` |
| NGC Container | `nvcr.io/nvidia/tritonserver:26.02-py3` |

## Overview

Triton Inference Server enables deployment of AI models from multiple frameworks (TensorRT, PyTorch, ONNX, TensorFlow, Python, and more) with a single, standardized API. It handles dynamic batching, concurrent execution, and model management automatically.

## Supported Frameworks

| Framework | Backend | Use Case |
|---|---|---|
| TensorRT | `tensorrt` | Optimized NVIDIA inference |
| TensorRT-LLM | `tensorrtllm` | LLM inference |
| PyTorch | `pytorch` | Direct PyTorch model serving |
| ONNX Runtime | `onnxruntime` | Cross-platform inference |
| OpenVINO | `openvino` | Intel hardware support |
| Python | `python` | Custom Python logic |
| RAPIDS FIL | `fil` | Tree-based ML models |

## Key Features

### Dynamic Batching
Automatically combines individual requests into batches for GPU efficiency.

### Concurrent Model Execution
Run multiple models on the same GPU simultaneously.

### Model Ensembles
Chain models together in a pipeline (e.g., preprocessing → model → postprocessing).

### Business Logic Scripting (BLS)
Write custom request handling logic in Python within Triton.

### Sequence Batching
Handle stateful models with implicit state management.

## API Protocols

| Protocol | Library | Use Case |
|---|---|---|
| HTTP/REST | `tritonclient.http` | Simple integration |
| gRPC | `tritonclient.grpc` | High-performance |
| KServe | Standard | Kubernetes-native inference |

## Model Repository

```
model_repository/
├── model_a/
│   ├── config.pbtxt          # Model configuration
│   └── 1/                    # Version 1
│       └── model.plan        # TensorRT engine
├── model_b/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx        # ONNX model
└── ensemble/
    └── config.pbtxt          # Pipeline definition
```

## Quick Start

```bash
# Start Triton with a model repository
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:26.02-py3 \
  tritonserver --model-repository=/models
```

```python
# Python client
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = [httpclient.InferInput("input", shape, "FP32")]
inputs[0].set_data_from_numpy(data)
result = client.infer("model_name", inputs)
```

## Deployment Options

| Method | Description |
|---|---|
| Docker | NGC container with GPU support |
| Bare metal | Build from source or install packages |
| Kubernetes/Helm | Official Helm charts |
| Jetson/JetPack | Edge deployment on NVIDIA Jetson |

## Performance Tools

| Tool | Purpose |
|---|---|
| Model Analyzer | Profile models and find optimal configurations |
| Performance Analyzer | Benchmark throughput and latency |
| Model Navigator | Automate model optimization |
