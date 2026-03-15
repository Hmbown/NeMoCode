# NeMoCode

Agentic coding CLI for [NVIDIA Nemotron 3](https://build.nvidia.com). Reads your code, makes edits, runs commands.

> **Community project** — not affiliated with or endorsed by NVIDIA.

## Install

```bash
pip install -e .
```

## Setup

Get a free API key from [build.nvidia.com](https://build.nvidia.com), then:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

Or use a local model with [Ollama](https://ollama.com) (no API key needed):

```bash
ollama pull nemotron-mini
nemo code -e local-ollama
```

## Usage

```bash
nemo code                              # interactive REPL
nemo code "fix the bug in auth.py" -y  # one-shot with auto-approve
nemo chat "explain this error"         # chat without tools
cat log.txt | nemo code "diagnose"     # pipe input
nemo code -f super-nano "refactor"     # multi-model formation
```

## What it does

NeMoCode connects to Nemotron 3 models via the Chat Completions API and gives them tools to read/write files, run shell commands, search code, and use git. It streams responses and shows tool calls as they happen.

**Tools:** `read_file`, `write_file`, `edit_file`, `list_dir`, `bash_exec`, `search_files`, `git_status`, `git_diff`, `git_log`, `git_commit`, `http_fetch`

## Endpoints

Works with any OpenAI-compatible API. Pre-configured:

| Endpoint | Model | How |
|----------|-------|-----|
| `nim-super` | Nemotron 3 Super (12B/120B) | [build.nvidia.com](https://build.nvidia.com) API key |
| `nim-nano` | Nemotron 3 Nano (3B/30B) | Same API key |
| `nim-nano-9b` | Nemotron Nano 9B v2 | Same API key |
| `openrouter-super` | Super via OpenRouter | OpenRouter key |
| `together-super` | Super via Together AI | Together key |
| `local-ollama` | Any local Ollama model | `ollama serve` |
| `local-vllm-*` | Local vLLM | `vllm serve ...` |
| `local-nim-*` | Local NIM container | Docker + GPU |

Switch endpoints: `nemo code -e openrouter-super` or `/endpoint` in the REPL.

## Formations

Multi-model pipelines where different models handle different roles:

| Formation | What happens |
|-----------|-------------|
| `solo` | Super does everything (default) |
| `super-nano` | Super plans + reviews, Nano executes |
| `local` | Nano on local hardware, no internet |

```bash
nemo code -f super-nano "implement caching"
# Super writes the plan -> Nano executes -> Super reviews
```

## Local inference

```bash
# Ollama (Mac/Linux/Windows, no GPU required)
ollama pull nemotron-mini
nemo code -e local-ollama

# vLLM (Linux, NVIDIA GPU required)
pip install vllm
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes
nemo code -e local-vllm-nano9b

# NIM container (Linux, NVIDIA GPU required)
nemo setup nim
```

## More commands

```bash
nemo endpoint ls          # list endpoints
nemo endpoint test X      # test connectivity
nemo model ls             # list model manifests
nemo hardware recommend   # GPU-based recommendations
nemo formation ls         # list formations
nemo session ls           # past sessions
nemo obs pricing          # token pricing
nemo setup                # local inference setup guides
nemo init                 # create .nemocode.yaml
```

## Configuration

Layered: `defaults.yaml` -> `~/.config/nemocode/config.yaml` -> `.nemocode.yaml` -> env vars.

```bash
export NEMOCODE_ENDPOINT=nim-nano      # override default endpoint
export NEMOCODE_FORMATION=super-nano   # activate a formation
```

## Contributing

```bash
git clone https://github.com/Hmbown/NeMoCode.git
cd NeMoCode
pip install -e ".[dev]"
ruff check src/ tests/    # must pass
pytest tests/ -v          # must pass
```

PRs welcome. Add SPDX headers to new `.py` files. Open an issue first for big changes.

## License

[MIT](LICENSE). NVIDIA, Nemotron, and NIM are trademarks of NVIDIA Corporation.
