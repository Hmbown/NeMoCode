# NeMo RL

> Reinforcement Learning from Human Feedback (RLHF) and related alignment techniques.

## Canonical URL

`https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html` (RL section)

## Overview

NeMo RL provides implementations of alignment algorithms for fine-tuning language models to better follow instructions and produce helpful, harmless, and honest outputs.

## Supported Algorithms

| Algorithm | Description |
|---|---|
| **RLHF (PPO)** | Proximal Policy Optimization with reward model |
| **DPO** | Direct Preference Optimization — no reward model needed |
| **GRPO** | Group Relative Policy Optimization — efficient variant of PPO |
| **SteerLM** | Attribute-conditioned generation with controllable outputs |

## Training Pipeline (RLHF)

1. **Supervised Fine-Tuning (SFT)** — Train on instruction-following data
2. **Reward Model Training** — Train on human preference pairs
3. **PPO Training** — Optimize policy against reward model
4. **Evaluation** — Benchmark against safety and helpfulness metrics

## Integration with NeMo Framework

- Uses Megatron-Core for distributed training
- Supports multi-node, multi-GPU setups
- Compatible with NeMo model checkpoints
- Works with NeMo Evaluator for automated benchmarking

## Key Concepts

- **Reward Model**: Scores model outputs for quality/safety
- **KL Penalty**: Prevents policy from diverging too far from reference model
- **Value Function**: Estimates expected reward for PPO training
- **Preference Pairs**: (chosen, rejected) pairs for DPO/reward training
