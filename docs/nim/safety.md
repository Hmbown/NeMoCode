# NIM for Safety (NemoGuard)

> Content safety, topic control, and jailbreak detection models.

## Canonical URLs

| Resource | URL |
|---|---|
| NIM Hub (safety section) | `https://docs.nvidia.com/nim/index.html` |
| Multimodal Safety | `https://docs.nvidia.com/nim/multimodal-safety/latest/` |
| NeMo Guardrails (framework) | See [Agentic > Guardrails](../agentic/guardrails.md) |

## Overview

NemoGuard NIMs provide deployable safety models that can be integrated into AI pipelines to filter harmful content, enforce topic boundaries, and detect prompt injection attacks.

## Available Models

### Llama 3.1 NemoGuard TopicControl
- **Purpose**: Enforce conversation topic boundaries
- **Use case**: Ensure chatbot stays on-topic for enterprise deployments
- **Input**: User message + allowed topic list
- **Output**: On-topic / off-topic classification

### Llama 3.1 NemoGuard ContentSafety
- **Purpose**: Detect harmful, toxic, or unsafe content
- **Use case**: Filter user inputs and model outputs
- **Categories**: Violence, sexual content, self-harm, hate speech, harassment, etc.
- **Output**: Safety scores per category

### Nemotron Safety Guard
- **Purpose**: General-purpose safety classification
- **Use case**: Pre/post-processing safety filter
- **Output**: Safe/unsafe with category labels

### NemoGuard JailbreakDetect
- **Purpose**: Detect prompt injection and jailbreak attempts
- **Use case**: Protect LLM deployments from adversarial inputs
- **Detection**: Template-based, role-play, encoding-based, and novel jailbreak patterns

### Multimodal Safety NIM
- **Purpose**: Safety assessment for image + text inputs
- **Use case**: Filter multimodal content in vision-language pipelines
- **Docs**: `https://docs.nvidia.com/nim/multimodal-safety/latest/`

## Integration Pattern

```
User Input → JailbreakDetect → TopicControl → LLM NIM → ContentSafety → Response
```

1. Check input for jailbreak attempts
2. Verify input is on-topic
3. Generate response with LLM
4. Check output for harmful content
5. Return filtered response

## Relationship to NeMo Guardrails

NemoGuard NIMs are the **models** — they do the classification.
NeMo Guardrails is the **framework** — it orchestrates the safety pipeline using Colang DSL.

Use them together: Guardrails calls NemoGuard NIMs as part of its rail execution.
