# NeMoCode

Agentic coding CLI for [NVIDIA NIM](https://build.nvidia.com). Reads your code, makes edits, runs commands — powered by any model on the NIM API or your own GPU via [vLLM](https://docs.vllm.ai/) or [SGLang](https://sgl-project.github.io/).

> **Community project** — not affiliated with or endorsed by NVIDIA.

## Install

From source (editable):
```bash
pip install -e .
```

Or from PyPI:
```bash
pip install nemocode
```

## Setup

Run the guided setup wizard:

```bash
nemo setup
```

It defaults to hosted NVIDIA NIM, prompts for `NVIDIA_API_KEY`, and can also configure a local `vllm` or `sglang` endpoint and model for you.

If you just want hosted NIM manually, get a free API key from [build.nvidia.com](https://build.nvidia.com):

```bash
export NVIDIA_API_KEY="nvapi-..."
nemo code
```

Hosted Nemotron/NIM endpoints in NeMoCode use `NVIDIA_API_KEY` by default.

Or serve a model locally with [vLLM](https://docs.vllm.ai/) or [SGLang](https://sgl-project.github.io/) on any NVIDIA GPU:

```bash
# vLLM
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --trust-remote-code --mamba_ssm_cache_dtype float32 \
  --enable-auto-tool-choice \
  --tool-parser-plugin nemotron_toolcall_parser.py \
  --tool-call-parser nemotron_json
nemo code -e local-vllm-nano9b

# SGLang (best for Nemotron 3 Super long context on DGX Spark)
python -m sglang.launch_server \
  --model nvidia/nemotron-3-super-120b-a12b \
  --quantization nvfp4 --trust-remote-code
nemo code -e spark-sglang-super
```

No GPU? Rent one via [Brev](https://console.brev.dev) — L40S from $1.03/hr:

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
```

## Endpoints

Works with any OpenAI-compatible API. Pre-configured:

| Endpoint | Model | Access |
|----------|-------|--------|
| `nim-super` | Nemotron 3 Super (12B/120B MoE, 1M ctx) | NIM API key |
| `nim-nano` | Nemotron 3 Nano (3B/30B MoE, 1M ctx) | NIM API key |
| `nim-nano-9b` | Nemotron Nano 9B v2 | NIM API key |
| `nim-nano-4b` | Nemotron Nano 4B v1.1 (new!) | NIM API key |
| `nim-vlm` | Nemotron Nano 12B VL (vision) | NIM API key |
| `nim-embed` | Nemotron Embed 1B v2 | NIM API key |
| `nim-rerank` | Nemotron Rerank 1B v2 | NIM API key |
| `openrouter-super` | Super via OpenRouter | OpenRouter key |
| `together-super` | Super via Together AI | Together key |
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
| `vision` | VLM reads screenshots, Super writes code |
| `local` | Nano on local GPU, no internet needed |

## Agents

NeMoCode now supports named agent profiles for top-level sessions and delegated sub-agents.

- Built-in primary agents: `build`, `plan`
- Built-in sub-agents: `general`, `explore`, `review`, `debug`, `test`, `doc`, `code-search`, `fast`
- Inspect them with `nemo agent ls` and `nemo agent show <name>`
- Switch primary agents with `nemo code --agent <name>` or `/agent <name>` in the REPL/TUI
- Sub-agent orchestration tools are now available in coding sessions: `delegate`, `spawn_agent`, `wait_agent`, `close_agent`, and `resume_agent`
- Define custom agents in `.nemocode.yaml` under `agents:` or in markdown files under `.nemocode/agents/*.md`

Example markdown agent:

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

## Local GPU setup

```bash
nemo setup          # guided wizard
nemo setup --list   # show all setup topics
nemo setup wizard   # force the interactive wizard
nemo setup vllm     # vLLM serving guide
nemo setup sglang   # SGLang serving guide
nemo setup nim      # NIM container guide
nemo setup brev     # rent a cloud GPU
```

## More commands

```bash
nemo endpoint ls / test     # manage endpoints
nemo model ls / show        # inspect model manifests
nemo formation ls / show    # inspect formations
nemo agent ls / show        # inspect agent profiles
nemo hardware recommend     # GPU-based recommendations
nemo doctor                 # run diagnostics to check setup
nemo session ls             # past conversations
nemo obs pricing            # token pricing
nemo init                   # create .nemocode.yaml without overriding your user default endpoint
```

## Contributing

```bash
pip install -e ".[dev]"
ruff check src/ tests/ && ruff format --check src/ tests/
pytest tests/ -v
```

## License

[MIT](LICENSE). NVIDIA, Nemotron, and NIM are trademarks of NVIDIA Corporation.
