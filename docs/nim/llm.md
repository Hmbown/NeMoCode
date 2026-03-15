# NIM for Large Language Models

> Deploy optimized LLMs with OpenAI-compatible APIs.

## Canonical URL

`https://docs.nvidia.com/nim/large-language-models/latest/index.html`

## Overview

NIM for LLMs packages language models with TensorRT-LLM optimization into containers that expose OpenAI-compatible chat/completion endpoints. Supports single-model, multi-model, reasoning, reward, and text-to-SQL NIMs.

## API Compatibility

NIM for LLMs implements the OpenAI Chat Completions API:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

## Key Features

### Model Profiles
Pre-optimized configurations for different GPU setups (1xH100, 2xH100, 8xA100, etc.).

### Function/Tool Calling
```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
  }]
}
```

### Structured Generation
Force output to match a JSON schema or regex pattern via `guided_json` or `guided_regex` parameters.

### LoRA Adapter Support
Hot-swap fine-tuned LoRA adapters without restarting the NIM.

### Reasoning Models
Support for chain-of-thought reasoning models that show their work.

### Reward Models
Score outputs for quality — useful for RLHF pipelines and best-of-N sampling.

## Deployment

```bash
# Pull and run a NIM container
docker run --gpus all -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-70b-instruct:latest
```

## Documentation Sections

- Getting Started / Quick Start
- Deployment guides (Docker, Helm, air-gap, multi-node, proxy, DGX Spark, WSL2)
- Model profiles and supported models
- Function calling and structured generation
- Fine-tuned model deployment
- Observability and benchmarking
- Llama Stack integration
