# NeMo Microservices

> Containerized production APIs for the NeMo platform.

## Canonical URL

`https://docs.nvidia.com/nemo/microservices/latest/index.html`

## Overview

NeMo Microservices provide production-ready, containerized APIs for fine-tuning, evaluation, data generation, safety, and model management. They are the deployment layer of the NeMo platform.

## Services

### NeMo Customizer
Fine-tuning service supporting multiple techniques.

| Feature | Description |
|---|---|
| LoRA | Low-Rank Adaptation for efficient fine-tuning |
| SFT | Supervised Fine-Tuning on instruction data |
| DPO | Direct Preference Optimization for alignment |
| Distillation | Knowledge transfer from teacher to student model |

- API: REST
- Docs: `https://docs.nvidia.com/nemo/microservices/latest/index.html` (Customizer section)

### NeMo Evaluator
Evaluation service for benchmarking models.

| Feature | Description |
|---|---|
| Academic Benchmarks | Standard benchmarks (MMLU, GSM8K, HumanEval, etc.) |
| RAG Evaluation | Retrieval-augmented generation quality metrics |
| Agentic Evaluation | Agent task completion and tool use evaluation |
| LLM-as-Judge | Use a judge model to score outputs |

### NeMo Data Designer
Synthetic data generation service (Early Access).

- Generates training data from task descriptions
- Supports multiple domains and formats
- Quality filtering and verification built-in

### NeMo Safe Synthesizer
PII replacement and differential privacy for sensitive datasets.

- Detect and replace PII in training data
- Apply differential privacy guarantees
- Generate privacy-preserving synthetic alternatives

### NeMo Retriever
Embedding, reranking, and retrieval services.

- Text embedding with OpenAI-compatible API
- Document reranking for RAG pipelines
- See also: [NIM Retrieval](../nim/retrieval.md)

### NeMo Auditor
Model safety auditing service.

- Automated red-teaming
- Bias detection and fairness analysis
- Safety benchmark evaluation

### NeMo Studio
Visual workflow management UI.

- Web-based interface for managing NeMo workflows
- Visual pipeline builder
- Experiment monitoring dashboard

### Entity Store / Data Store
Dataset and model management.

- Version-controlled dataset storage
- Model checkpoint management
- Metadata tracking and search
