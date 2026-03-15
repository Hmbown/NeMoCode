# NVIDIA API Endpoints Reference

> Key API endpoints by category.

## Base URLs

| Service | Base URL |
|---|---|
| Cloud API (LLMs) | `https://integrate.api.nvidia.com/v1` |
| API Documentation | `https://docs.api.nvidia.com/` |
| NIM Reference | `https://docs.api.nvidia.com/nim/reference` |

## LLM Endpoints (OpenAI-compatible)

### Chat Completions
```
POST https://integrate.api.nvidia.com/v1/chat/completions
```

### Models List
```
GET https://integrate.api.nvidia.com/v1/models
```

### Embeddings
```
POST https://integrate.api.nvidia.com/v1/embeddings
```

## Request Format (Chat Completions)

```json
{
  "model": "nvidia/nemotron-4-340b-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is NeMo?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 0.95,
  "stream": true
}
```

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "nvidia/nemotron-4-340b-instruct",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "NeMo is..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 150, "total_tokens": 170}
}
```

## Available Model Categories on build.nvidia.com

| Category | Example Models |
|---|---|
| **Reasoning** | Nemotron, Llama, DeepSeek, Qwen |
| **Vision** | Llama 4 multimodal, Qwen-VL |
| **Visual Design** | FLUX, Stable Diffusion, TRELLIS |
| **Retrieval** | NV-EmbedQA, NV-RerankQA |
| **Speech** | Parakeet, Canary, TTS |
| **Biology** | AlphaFold2, ProteinMPNN, MolMIM |
| **Safety** | NemoGuard, Content Safety |

## Self-hosted NIM Endpoints

When running NIMs locally, the same API format applies with your local URL:

```bash
# Local NIM
curl http://localhost:8000/v1/chat/completions ...

# Cloud API
curl https://integrate.api.nvidia.com/v1/chat/completions ...
```

The API is identical — only the base URL changes.
