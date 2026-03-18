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

Get a free API key from [build.nvidia.com](https://build.nvidia.com):

```bash
export NVIDIA_API_KEY="nvapi-..."
nemo code
```

Or serve a model locally with [vLLM](https://docs.vllm.ai/) on any NVIDIA GPU:

```bash
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --trust-remote-code --mamba_ssm_cache_dtype float32 \
  --enable-auto-tool-choice \
  --tool-parser-plugin nemotron_toolcall_parser.py \
  --tool-call-parser nemotron_json
nemo code -e local-vllm-nano9b
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
| `nim-super` | Nemotron 3 Super (12B/120B MoE) | NIM API key |
| `nim-nano` | Nemotron 3 Nano (3B/30B MoE) | NIM API key |
| `nim-nano-9b` | Nemotron Nano 9B v2 | NIM API key |
| `openrouter-super` | Super via OpenRouter | OpenRouter key |
| `together-super` | Super via Together AI | Together key |
| `local-vllm-*` | Any model on local vLLM | GPU + vLLM |
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
| `local` | Nano on local GPU, no internet needed |

## Local GPU setup

```bash
nemo setup          # show all options
nemo setup vllm     # vLLM serving guide
nemo setup nim      # NIM container guide
nemo setup brev     # rent a cloud GPU
```

## More commands

```bash
nemo endpoint ls / test     # manage endpoints
nemo model ls / show        # inspect model manifests
nemo formation ls / show    # inspect formations
nemo hardware recommend     # GPU-based recommendations
nemo doctor                 # run diagnostics to check setup
nemo session ls             # past conversations
nemo obs pricing            # token pricing
nemo init                   # create .nemocode.yaml
```

## Contributing

```bash
pip install -e ".[dev]"
ruff check src/ tests/ && ruff format --check src/ tests/
pytest tests/ -v
```

## License

[MIT](LICENSE). NVIDIA, Nemotron, and NIM are trademarks of NVIDIA Corporation.
