# NGC Catalog

> NVIDIA's registry for containers, models, Helm charts, and resources.

## Canonical URL

`https://catalog.ngc.nvidia.com`

## Overview

NGC (NVIDIA GPU Cloud) Catalog is the central registry for all NVIDIA AI software. It hosts container images, pre-trained models, model scripts, Helm charts, and SDKs.

## Content Types

| Type | Description | Example |
|---|---|---|
| **Containers** | GPU-optimized Docker images | NeMo Framework, Triton Server |
| **Models** | Pre-trained model checkpoints | Nemotron, Parakeet, NV-CLIP |
| **Helm Charts** | Kubernetes deployment manifests | NIM Operator, Triton Helm |
| **Resources** | Scripts, notebooks, configs | Training scripts, fine-tuning examples |

## Key Container Images

| Image | Tag Pattern | Description |
|---|---|---|
| `nvcr.io/nvidia/nemo` | `26.xx` | NeMo Framework training container |
| `nvcr.io/nvidia/tritonserver` | `26.xx-py3` | Triton Inference Server |
| `nvcr.io/nim/*` | Various | NIM inference containers |
| `nvcr.io/nvidia/pytorch` | `26.xx-py3` | Optimized PyTorch |
| `nvcr.io/nvidia/tensorrt` | `26.xx-py3` | TensorRT development |

## Authentication

```bash
# Login to NGC registry
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API key from ngc.nvidia.com>
```

## NGC CLI

```bash
# Install NGC CLI
pip install ngc-cli

# Search for models
ngc registry model list --query "parakeet"

# Pull a model
ngc registry model download-version nvidia/nemo/parakeet-tdt-0.6b-v2:1.0

# Pull a container
docker pull nvcr.io/nvidia/nemo:26.02
```

## Organization Structure

| Org | Content |
|---|---|
| `nvidia` | Official NVIDIA containers and models |
| `nim` | NIM inference microservice containers |
| `nvidia/nemo` | NeMo-specific models and resources |
