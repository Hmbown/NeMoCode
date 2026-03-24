# NVIDIA Ecosystem Roadmap — NeMoCode

> From repo analysis to a fine-tuned Nemotron 3 Nano serving locally on DGX Spark.

## Vision

NeMoCode should be the single CLI that takes you from a raw repository to a fine-tuned, locally-served coding assistant — all through NVIDIA's stack.

```
nemo data analyze        →  understand the repo
nemo data export-seeds   →  grounded seed artifacts
nemo data export-sft     →  training dataset
nemo customize create    →  LoRA fine-tune Nemotron 3 Nano
nemo serve               →  serve it locally on Spark
nemo chat                →  use your custom model
```

## Current State (as of 2026-03-23)

### What Works

| Capability | Status | Commands |
|---|---|---|
| NIM cloud inference (17 models) | Done | `nemo chat`, `nemo code` |
| Multi-model formations | Done | `nemo formation ls/run` |
| 6 inference backends | Done | `nemo setup {nim,vllm,sglang,trt-llm,spark,brev}` |
| Credential management | Done | `nemo auth setup/set/show/test` |
| Repo analysis | Done | `nemo data analyze` |
| Seed export | Done | `nemo data export-seeds` |
| Data Designer preview/jobs | Done | `nemo data preview`, `nemo data job {create,status,logs,results}` |
| SFT dataset export | Done | `nemo data export-sft` |
| NGC API key in credential store | Done | `nemo auth set NGC_CLI_API_KEY` |

### What's Missing (This Roadmap)

| Gap | Impact |
|---|---|
| No fine-tuning CLI | Can't close the loop from data → model |
| No model pulling | Users copy-paste Docker commands manually |
| No adapter serving | No way to use a fine-tuned model locally |
| No end-to-end pipeline | Each step is manual |
| No guardrails enforcement | Safety endpoint exists but unused |
| No structured output | Config exists in manifests but inactive |
| No token/cost tracking | No visibility into API spend |

## Linear Issues

### Critical Path: Repo → Synthetic Data → Fine-Tune → Serve

| Issue | Title | Priority | Depends On |
|---|---|---|---|
| [SHA-3685](https://linear.app/shannon-labs/issue/SHA-3685) | `nemo customize`: LoRA fine-tuning CLI for Nemotron 3 Nano | Urgent | — |
| [SHA-3687](https://linear.app/shannon-labs/issue/SHA-3687) | NGC model manager: `nemo model pull` for NIM containers | High | — |
| [SHA-3690](https://linear.app/shannon-labs/issue/SHA-3690) | `nemo data pipeline`: analyze → seeds → synth → finetune | High | SHA-3685 |
| [SHA-3691](https://linear.app/shannon-labs/issue/SHA-3691) | `nemo serve` with LoRA adapter on DGX Spark | High | SHA-3685, SHA-3687 |

### Ecosystem Polish

| Issue | Title | Priority |
|---|---|---|
| [SHA-3686](https://linear.app/shannon-labs/issue/SHA-3686) | NeMo Guardrails provider: safety in chat/code pipeline | Medium |
| [SHA-3688](https://linear.app/shannon-labs/issue/SHA-3688) | Structured output: activate JSON schema/grammar in NIM chat | Medium |
| [SHA-3689](https://linear.app/shannon-labs/issue/SHA-3689) | Token counting and cost tracking for NIM API usage | Medium |

## Execution Order

```
Phase 1 (parallel):
  SHA-3685  nemo customize (Customizer CLI)     ← unlocks everything
  SHA-3687  nemo model pull (NGC containers)     ← unlocks local serving

Phase 2 (after Phase 1):
  SHA-3690  nemo data pipeline (end-to-end)      ← blocked by 3685
  SHA-3691  nemo serve (adapter serving)          ← blocked by 3685 + 3687

Phase 3 (any order):
  SHA-3686  Guardrails provider
  SHA-3688  Structured output
  SHA-3689  Token/cost tracking
```

## Agent Execution Guide

Each Linear issue is self-contained and agent-ready. Every issue includes:

- **What to build** — commands, API endpoints, file locations
- **Patterns to follow** — which existing files to read for conventions
- **How to verify** — test commands and official docs to check against
- **Environment setup** — `pip install -e ".[dev]"` and relevant env vars

### Key Files an Agent Should Read First

| File | Purpose |
|---|---|
| `src/nemocode/core/nvidia_client.py` | HTTP client pattern for NVIDIA microservices |
| `src/nemocode/cli/commands/data.py` | CLI command pattern with Typer + Rich |
| `src/nemocode/core/data_workflows.py` | Repo analysis and seed generation logic |
| `src/nemocode/providers/nim_chat.py` | NIM provider pattern (streaming, retry, manifests) |
| `src/nemocode/defaults.yaml` | Full endpoint/model/formation catalog |
| `src/nemocode/config/schema.py` | Pydantic config models |
| `src/nemocode/core/credentials.py` | Unified credential store (keyring + env) |

### Environment Facts

- Machine: DGX Spark (NVIDIA GB10, 128GB unified memory)
- Docker: installed
- Python: 3.11+
- NGC CLI: not yet installed (agents should install if needed)
- `nemo_microservices` / `nemo_curator`: not installed (agents should install if needed)
- GitHub remote: `origin → https://github.com/Hmbown/NeMoCode.git`
- Working branch for data features: `feature/data-workflow-cli`

### API Keys

| Key | Purpose | How to Set |
|---|---|---|
| `NVIDIA_API_KEY` | Cloud NIM inference (build.nvidia.com) | `nemo auth set NVIDIA_API_KEY` |
| `NGC_CLI_API_KEY` | Container pulls, Customizer, Data Designer, Evaluator | `nemo auth set NGC_CLI_API_KEY` |

Both are in the unified credential store. Get NGC key at: https://ngc.nvidia.com/setup/api-key

### NVIDIA Docs Reference

| Service | Docker Setup | API Reference |
|---|---|---|
| Data Designer | [docker-compose](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/docker-compose.html) | [API](https://docs.nvidia.com/nemo/microservices/latest/api/data-designer.html) |
| Evaluator | [docker-compose](https://docs.nvidia.com/nemo/microservices/latest/evaluate/docker-compose.html) | [API](https://docs.nvidia.com/nemo/microservices/latest/api/evaluator.html) |
| Customizer | [docker-compose](https://docs.nvidia.com/nemo/microservices/latest/customize/docker-compose.html) | [API](https://docs.nvidia.com/nemo/microservices/latest/api/customization.html) |
| Safe Synthesizer | [docker-compose](https://docs.nvidia.com/nemo/microservices/latest/generate-private-synthetic-data/docker-compose.html) | [API](https://docs.nvidia.com/nemo/microservices/latest/api/safe-synthesizer.html) |
| Curator | [docs](https://docs.nvidia.com/nemo/curator/latest/index.html) | — |
