# NeMo Framework

> NVIDIA's modular platform for the AI agent lifecycle — training, customization, evaluation, and deployment.

## Architecture

NeMo has two layers:

1. **NeMo Framework** — Open-source Python framework for training and developing generative AI models
2. **NeMo Microservices** — Containerized production APIs for fine-tuning, evaluation, data generation, and safety

## Canonical URLs

| Resource | URL |
|---|---|
| NeMo Hub (all products) | `https://docs.nvidia.com/nemo/index.html` |
| NeMo Framework User Guide | `https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html` |
| NeMo Microservices | `https://docs.nvidia.com/nemo/microservices/latest/index.html` |
| GitHub | `https://github.com/NVIDIA/NeMo` |

## Sub-Libraries

| Library | Description | Doc |
|---|---|---|
| [AutoModel](automodel.md) | PyTorch SPMD training with HuggingFace model support | This folder |
| [Curator](curator.md) | GPU-accelerated data preprocessing (text, image, video, audio) | This folder |
| [RL](rl.md) | Reinforcement learning (RLHF, DPO, GRPO) | This folder |
| [Run](run.md) | Experiment configuration, execution, and management | This folder |
| [Skills](skills.md) | Synthetic data generation and evaluation pipelines | This folder |
| [Export & Deploy](export-deploy.md) | Model export and deployment utilities | This folder |
| [Microservices](microservices.md) | Customizer, Evaluator, Data Designer, Guardrails, etc. | This folder |
| [Data Workflows](data-workflows.md) | Repo-aware synthetic-data planning with NeMoCode + NVIDIA services | This folder |

## Key Framework Capabilities

- **Model Training**: Pre-training, fine-tuning (LoRA, P-Tuning, Adapters, IA3), distillation
- **Distributed Training**: Tensor Parallelism, Pipeline Parallelism, FSDP, MoE via Megatron-Core
- **Model Optimization**: Quantization, Pruning, Distillation, Speculative Decoding
- **Supported Architectures**: DeepSeek V3, Qwen3, Gemma 3, Llama 4, Mistral, and many more
- **Domains**: LLMs, ASR, TTS, Vision, Multimodal

## GitHub Repositories

| Repo | URL |
|---|---|
| NeMo (core) | `https://github.com/NVIDIA/NeMo` |
| NeMo-Curator | `https://github.com/NVIDIA/NeMo-Curator` |
| NeMo-Run | `https://github.com/NVIDIA/NeMo-Run` |
| NeMo-Skills | `https://github.com/NVIDIA/NeMo-Skills` |
| NeMo-Guardrails | `https://github.com/NVIDIA/NeMo-Guardrails` |
| NeMo-Agent-Toolkit | `https://github.com/NVIDIA/nemo-agent-toolkit` |
