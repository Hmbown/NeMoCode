# NeMo Guardrails

> Programmable safety and control rails for LLM applications.

## Canonical URLs

| Resource | URL |
|---|---|
| Documentation | `https://docs.nvidia.com/nemo/guardrails/index.html` |
| GitHub | `https://github.com/NVIDIA/NeMo-Guardrails` |
| PyPI | `pip install nemoguardrails` |

## Overview

NeMo Guardrails is an open-source framework for adding programmable safety, security, and control to LLM-powered applications. It uses the Colang domain-specific language to define conversation flows and safety rules.

## Five Rail Types

| Rail | When | Purpose |
|---|---|---|
| **Input** | Before LLM | Filter/modify user input (jailbreak detection, PII masking) |
| **Dialog** | During conversation | Control conversation flow (topic boundaries, required steps) |
| **Retrieval** | During RAG | Filter/modify retrieved context (relevance, safety) |
| **Execution** | During tool use | Control which tools can be called and with what parameters |
| **Output** | After LLM | Filter/modify model output (factuality, safety, formatting) |

## Colang DSL

Guardrails uses Colang (versions 1.0 and 2.0) to define conversation flows:

```colang
# Colang 2.0 example
define flow greeting
  user said "hello" or user said "hi"
  bot said "Hello! How can I help you today?"

define flow block harmful content
  user asked about harmful topic
  bot said "I can't help with that. Let me assist you with something else."
  bot refuse to respond
```

## Built-in Safety Features

- **Jailbreak Detection**: Detect and block prompt injection attempts
- **Hallucination Detection**: Fact-check responses against provided context
- **Topic Control**: Keep conversations within defined boundaries
- **Content Safety**: Filter harmful, toxic, or inappropriate content
- **PII Protection**: Detect and mask personal information

## Supported LLM Backends

- NVIDIA NIM endpoints
- OpenAI GPT-3.5/4
- Llama-2, Llama-3
- Falcon, Vicuna, Mosaic
- Any OpenAI-compatible API

## Integration Options

```python
# Python API
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
response = rails.generate(messages=[{"role": "user", "content": "Hello!"}])
```

```python
# LangChain integration
from nemoguardrails.integrations.langchain import RunnableRails
chain = RunnableRails(config) | llm_chain
```

```bash
# Server mode (REST API)
nemoguardrails server --config ./config
# POST http://localhost:8000/v1/chat/completions
```

## Docker Deployment

```bash
docker run -p 8000:8000 -v ./config:/app/config \
  nemoguardrails/nemoguardrails:latest server
```

## Configuration Structure

```
config/
├── config.yml          # Main config (LLM, rails to enable)
├── prompts.yml         # Custom prompts for rails
├── rails/
│   ├── input.co        # Input rail definitions (Colang)
│   ├── output.co       # Output rail definitions
│   └── dialog.co       # Dialog flow definitions
└── kb/                 # Knowledge base documents (optional)
```

## License

Apache 2.0
