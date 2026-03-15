# NVIDIA API Catalog

> Access NVIDIA AI models via cloud APIs at build.nvidia.com.

## Canonical URLs

| Resource | URL |
|---|---|
| API Catalog (browse models) | `https://build.nvidia.com/explore/discover` |
| API Documentation | `https://docs.api.nvidia.com/` |
| NIM API Reference | `https://docs.api.nvidia.com/nim/reference` |

## Overview

The NVIDIA API Catalog provides free serverless API endpoints for trying NVIDIA and partner AI models. Models can also be self-hosted on your own GPUs using NIM containers.

## Sections

| Doc | Description |
|---|---|
| [Authentication](authentication.md) | API keys, auth flow, rate limits |
| [Endpoints](endpoints.md) | Key API endpoints by category |
| [build.nvidia.com Guide](build-nvidia.md) | Using the API catalog portal |

## API Categories

| Category | Description |
|---|---|
| **NIM Reference** | LLM, Visual, Multimodal, Retrieval, Healthcare, Earth-2 APIs |
| **Cloud Functions** | Serverless GPU functions |
| **NGC Reference** | Container and model registry APIs |
| **TensorRT Cloud** | Cloud-based TensorRT optimization |
| **Riva** | Legacy speech AI APIs |

## Quick Start

```bash
# 1. Get API key from build.nvidia.com
# 2. Call any model endpoint
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-4-340b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

## Available GPU Instances (for self-hosting)

B300, B200, H200, H100, A100, L40S, RTX PRO 6000
