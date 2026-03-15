# Nemotron Models

> NVIDIA's flagship large language model family.

## Overview

Nemotron is NVIDIA's family of large language models, optimized for enterprise use cases including instruction following, code generation, reasoning, and safety evaluation.

## Model Variants

### Nemotron-4 Family
| Model | Parameters | Description |
|---|---|---|
| Nemotron-4-340B-Instruct | 340B | Flagship instruction-following model |
| Nemotron-4-340B-Base | 340B | Pre-trained base model |
| Nemotron-4-340B-Reward | 340B | Reward model for RLHF |

### Nemotron-Mini Family
| Model | Parameters | Description |
|---|---|---|
| Nemotron-Mini-4B | 4B | Compact model for edge/mobile deployment |

### Nemotron Safety Models
| Model | Description |
|---|---|
| Nemotron Safety Guard | General-purpose safety classification |
| See also: [NemoGuard NIMs](../nim/safety.md) | Deployed safety models |

## Key Capabilities

- **Instruction Following**: High-quality response to complex instructions
- **Code Generation**: Multi-language code generation and debugging
- **Reasoning**: Chain-of-thought and multi-step reasoning
- **Multilingual**: Support for multiple languages
- **Safety**: Built-in safety training and evaluation

## Access

| Method | Details |
|---|---|
| Cloud API | `https://build.nvidia.com` — search "Nemotron" |
| NIM | Available as NIM containers on NGC |
| HuggingFace | `https://huggingface.co/nvidia` |
| NeMo Framework | Train/fine-tune with NeMo |

## API Example

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-4-340b-instruct",
    "messages": [{"role": "user", "content": "Explain transformers."}],
    "max_tokens": 1024
  }'
```

## Training Data & Methodology

- Pre-trained on diverse multilingual web data
- Fine-tuned with SFT on high-quality instruction data
- Aligned with RLHF (PPO) and DPO
- SteerLM for controllable generation
- Safety training with red-team data
