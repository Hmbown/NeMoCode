# NeMoCode

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-771%20passed-success.svg)](https://github.com/Hmbown/NeMoCode)
[![PyPI](https://img.shields.io/pypi/v/nemocode.svg)](https://pypi.org/project/nemocode/)

Agentic coding CLI for [NVIDIA NIM](https://build.nvidia.com). Reads your code, makes edits, runs commands — powered by any model on the NIM API or your own GPU via [vLLM](https://docs.vllm.ai/), [SGLang](https://sgl-project.github.io/), or [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/).

> **Community project** — not affiliated with or endorsed by NVIDIA.

## Install

From PyPI:
```bash
pip install nemocode
```

Or from source (editable):
```bash
pip install -e .
```

## Quick Start

Run the guided setup wizard:

```bash
nemo setup
```

The wizard defaults to **hosted NVIDIA NIM**, prompts for `NVIDIA_API_KEY`, and can also configure a local `vllm`, `sglang`, or `trt-llm` backend for you.

### Hosted NVIDIA NIM (default)

Get a free API key from [build.nvidia.com](https://build.nvidia.com):

```bash
export NVIDIA_API_KEY="nvapi-..."
nemo code
```

Hosted Nemotron endpoints use `NVIDIA_API_KEY` by default. The setup wizard can store it in your system keyring.

### Local vLLM, SGLang, or TensorRT-LLM

Serve a model locally on any NVIDIA GPU:

```bash
# vLLM
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --host 0.0.0.0 --port 8000
nemo code -e local-vllm-nano9b

# SGLang (best for Nemotron 3 Super long context on DGX Spark)
python -m sglang.launch_server \
  --model nvidia/nemotron-3-super-120b-a12b \
  --host 0.0.0.0 --port 8000
nemo code -e local-sglang-super

# TensorRT-LLM
docker run --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6 \
  trtllm-serve nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \
  --trust_remote_code --port 8000
nemo code -e local-trt-llm-nano4b
```

TensorRT-LLM launch flags can vary by image release and whether a model needs a prebuilt
engine. The bundled NeMoCode presets assume the OpenAI-compatible `trtllm-serve` path with
`nvidia/nemotron-3-super-120b-a12b` and `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8`.

No GPU? Rent one via [Brev](https://console.brev.dev):

```bash
nemo setup brev
```

## Usage

```bash
nemo code                              # interactive REPL
nemo code "fix the bug in auth.py" -y  # one-shot, auto-approve tools
nemo chat "explain this error"         # chat, no tools
cat log.txt | nemo code "diagnose"     # pipe input
nemo code -f super-nano "refactor"     # multi-model formation
nemo code --tui                        # full-screen TUI
```

## Plan Mode

Plan mode is a read-only planning phase with an approval gate before execution.

- **Read-only**: Plan mode only reads files, searches code, and explores — no writes, shell commands, or commits.
- **Approval gate**: The planner proposes a concrete plan. You review and approve, revise with feedback, or cancel.
- **Execution**: Once approved, a build agent executes the plan with full tool access.

Switch modes in the REPL with Tab or `/mode`:

| Mode | Behavior |
|------|----------|
| `code` | Ask before tool calls (default) |
| `plan` | Read-only planning + approval gate |
| `auto` | Auto-approve everything |

Launch directly in plan mode:
```bash
nemo code --agent plan "implement user auth"
```

The plan agent can also spawn read-only research subagents to help with exploration.

## Endpoints

Works with any OpenAI-compatible API. Pre-configured:

| Endpoint | Model | Access |
|----------|-------|--------|
| `nim-super` | Nemotron 3 Super (12B/120B MoE, 1M ctx) | NIM API key |
| `nim-nano` | Nemotron 3 Nano (3B/30B MoE, 1M ctx) | NIM API key |
| `nim-nano-9b` | Nemotron Nano 9B v2 | NIM API key |
| `nim-nano-4b` | Nemotron Nano 4B v1.1 | NIM API key |
| `nim-vlm` | Nemotron Nano 12B VL (vision) | NIM API key |
| `nim-embed` | Nemotron Embed 1B v2 | NIM API key |
| `nim-rerank` | Nemotron Rerank 1B v2 | NIM API key |
| `openrouter-super` | Super via OpenRouter | OpenRouter key |
| `together-super` | Super via Together AI | Together key |
| `local-trt-llm-super` | Nemotron 3 Super 120B via TensorRT-LLM | GPU + Docker + TensorRT-LLM |
| `local-trt-llm-nano4b` | Nemotron 3 Nano 4B FP8 via TensorRT-LLM | GPU + Docker + TensorRT-LLM |
| `local-vllm-*` | Any model on local vLLM | GPU + vLLM |
| `local-sglang-*` | Any model on local SGLang | GPU + SGLang |
| `local-nim-*` | Local NIM container | GPU + Docker |

## Formations

Multi-model pipelines — Super plans, Nano executes, Super reviews:

```bash
nemo code -f super-nano "implement caching"
```

| Formation | Pipeline |
|-----------|----------|
| `solo` | Super does everything (default) |
| `super-nano` | Super plans + reviews, Nano executes |
| `spark` | All-local on DGX Spark (Super + Nano 9B) |
| `spark-sglang` | Super via SGLang on Spark (best long context) |
| `spark-trt-llm` | Super 120B + Nano 4B via TensorRT-LLM on Spark |
| `vision` | VLM reads screenshots, Super writes code |
| `local` | Nano on local GPU, no internet needed |

## Agents & Sub-agent Orchestration

NeMoCode supports named agent profiles for top-level sessions and delegated sub-agents.

- **Primary agents**: `build` (default full-access), `plan` (read-only planning)
- **Sub-agents**: `general`, `explore`, `review`, `debug`, `test`, `doc`, `code-search`, `fast`
- Inspect them with `nemo agent ls` and `nemo agent show <name>`
- Switch primary agents with `nemo code --agent <name>` or `/agent <name>` in the REPL/TUI

### Sub-agent tools

In coding sessions, these orchestration tools are available:

| Tool | Purpose |
|------|---------|
| `delegate` | Spawn a sub-agent and wait for the result |
| `spawn_agent` | Spawn a background sub-agent for parallel work |
| `wait_agent` | Wait for a spawned sub-agent to finish |
| `close_agent` | Close or cancel a sub-agent handle |
| `resume_agent` | Reopen a previously closed sub-agent handle |

Sub-agents inherit read-only mode when delegated from plan mode.

### Custom agents

Define custom agents in `.nemocode.yaml` under `agents:` or as markdown files under `.nemocode/agents/*.md`:

```markdown
---
description: Review code for bugs and regressions
mode: subagent
role: reviewer
prefer_tiers:
  - super
tools:
  - fs_read
  - git_read
  - rg
---

Review the requested changes. Focus on correctness, regressions, and missing tests.
```

## Setup Commands

```bash
nemo setup          # guided wizard
nemo setup --list   # show all setup topics
nemo setup wizard   # force the interactive wizard
nemo setup trt-llm  # TensorRT-LLM serving guide
nemo setup vllm     # vLLM serving guide
nemo setup sglang   # SGLang serving guide
nemo setup nim      # NIM container guide
nemo setup brev     # rent a cloud GPU
```

## More Commands

```bash
nemo endpoint ls / test     # manage endpoints
nemo model ls / show        # inspect model manifests
nemo formation ls / show    # inspect formations
nemo agent ls / show        # inspect agent profiles
nemo hardware recommend     # GPU-based recommendations
nemo doctor                 # run diagnostics to check setup
nemo session ls             # past conversations
nemo obs pricing            # token pricing
nemo init                   # create .nemocode.yaml without overriding user defaults
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and PR guidelines.

```bash
pip install -e ".[dev]"
ruff check src/ tests/ && ruff format --check src/ tests/
pytest tests/ -v
```

## License

[MIT](LICENSE). NVIDIA, Nemotron, and NIM are trademarks of NVIDIA Corporation.

## Architecture

```
  User Input (CLI / TUI)
         |
  CodeAgent (orchestrator)
    |  agent profiles, project context, git state, memories
    v
  Scheduler (formation pipeline driver)
    |  single-model or plan/execute/review formation
    |  stagnation detection, auto-compaction, permission engine
    v
  Registry (endpoint / manifest / formation resolver)
    |
    v
  Providers (NIM Chat, Embeddings, Rerank)
    |  OpenAI-compatible API, SSE streaming, retry with backoff
    v
  ToolRegistry (18+ tools)
    - fs: read, write, edit, multi_edit
    - git: status, diff, log, commit, snapshot
    - search: rg, glob
    - bash: shell command execution
    - agent: delegate, spawn, wait, close, resume
    - memory: save, recall
    - web, parse, test, ask_user, clarify, LSP, MCP
```

## Comparison

| Feature | NeMoCode | Claude Code | Cursor | OpenCode |
|---------|----------|-------------|--------|----------|
| Open source | MIT | No | No | MIT |
| Terminal-first | Yes | Yes | IDE | Yes |
| Multi-model formations | Yes | No | No | No |
| Local GPU serving | vLLM, SGLang, TRT-LLM | No | No | No |
| Hardware detection | Yes (GPU/RAM/Spark) | No | No | No |
| 1M token context | Yes (Nemotron 3) | 200K | 128K | Varies |
| Sub-agent orchestration | Yes | No | No | Yes |
| LSP integration | Yes | No | Yes | Yes |
| Vision (screenshots) | Yes (VLM) | Yes | Yes | Yes |
| Plugin system | Yes | No | No | Yes |
| NVIDIA NIM native | Yes | No | No | No |
