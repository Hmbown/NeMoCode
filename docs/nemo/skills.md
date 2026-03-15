# NeMo Skills

> Synthetic data generation and evaluation pipelines for math, code, and reasoning.

## Canonical URLs

| Resource | URL |
|---|---|
| GitHub | `https://github.com/NVIDIA/NeMo-Skills` |
| Documentation | `https://nvidia-nemo.github.io/Skills/` |

## Overview

NeMo Skills generates high-quality synthetic training data and provides evaluation pipelines for domains requiring structured reasoning — math, code, and logical problem-solving.

## Key Capabilities

- **Synthetic Data Generation**: Create training examples with verified solutions
- **Math Pipelines**: Generate and verify mathematical problem-solution pairs
- **Code Pipelines**: Generate code problems with executable test verification
- **Reasoning Chains**: Generate step-by-step reasoning traces
- **Evaluation**: Benchmark models on standard and custom problem sets

## Pipeline Stages

1. **Problem Generation** — Create diverse problem statements
2. **Solution Generation** — Generate candidate solutions via LLM
3. **Verification** — Execute/check solutions for correctness
4. **Filtering** — Remove incorrect or low-quality pairs
5. **Formatting** — Convert to training-ready format

## Use Cases

- Bootstrapping training data for math-capable models
- Generating code instruction-following data
- Creating evaluation benchmarks for reasoning tasks
- Augmenting existing datasets with verified synthetic examples
