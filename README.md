```
 _   _       __  __        ____          _
| \ | | ___ |  \/  | ___  / ___|___   __| | ___
|  \| |/ _ \| |\/| |/ _ \| |   / _ \ / _` |/ _ \
| |\  |  __/| |  | | (_) | |__| (_) | (_| |  __/
|_| \_|\___||_|  |_|\___/ \____\___/ \__,_|\___|
```

# NeMoCode

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/Hmbown/NeMoCode/actions)

NeMoCode is a community-driven, open-source agentic coding CLI built around NVIDIA's Nemotron 3 model family. It is **not an official NVIDIA product** but rather a community project that leverages Nemotron 3's unique capabilities like Latent MoE, Mamba-Transformer hybrid architecture, and native NVFP4 precision. NeMoCode provides a terminal-first experience for AI-assisted software development with features like multi-model formations, manifest-aware configuration, and hardware-aware recommendations.

## Quick Start

```bash
# Install from source
git clone https://github.com/Hmbown/NeMoCode.git
cd nemocode
pip install -e .

# Get an API key from build.nvidia.com
export NVIDIA_API_KEY="nvapi-..."

# Start coding
nemo code "add retry logic to the API client"

# Interactive REPL
nemo code

# Simple chat (no tools)
nemo chat "explain the Mamba architecture"
```

For detailed setup instructions, visit [build.nvidia.com](https://build.nvidia.com).

## Features

- **Agentic coding** -- streaming tool-calling loop with file I/O, shell, git, and search
- **Interactive REPL** -- multi-line input, slash commands, context window tracking, cost display
- **Formations** -- multi-model pipelines (planner/executor/reviewer) matching NVIDIA deployment patterns
- **Manifest-aware** -- typed config for MoE, reasoning budgets, thinking traces, NVFP4, Mamba
- **Retry logic** -- exponential backoff with Retry-After header support for 429/5xx errors
- **Hardware detection** -- GPU/VRAM/CPU profiling with model compatibility recommendations
- **Session persistence** -- save/load conversations with full message history
- **Path sandboxing** -- file writes are confined to the project directory
- **MCP support** -- connect to Model Context Protocol tool servers
- **First-run experience** -- guided setup for new users

## Formation Architecture

NeMoCode implements NVIDIA's recommended deployment patterns through formations—named multi-model setups that optimize for different workloads:

| Formation    | Pattern          | Description                                                                 |
|--------------|------------------|-----------------------------------------------------------------------------|
| `solo`       | Super only       | Day-to-day coding tasks using Nemotron 3 Super                              |
| `super-nano` | Super + Nano     | Super handles planning/review, Nano executes fast subtasks                  |
| `ultra-swarm`| Ultra + Super    | Ultra plans complex tasks, Super executes (when Ultra available)            |
| `retrieval`  | Embed + Rerank + Super | RAG-augmented coding with contextual knowledge retrieval          |
| `doc-ops`    | Parse + Embed + Super | Document processing workflows                                               |
| `local`      | Nano on local NIM| Air-gapped/offline work using locally hosted Nemotron 3 Nano                |

Formations are configured via manifest files that encode model-specific capabilities like Latent MoE, MTP, and precision settings.

## The Nemotron 3 Family

| Model | Params (active/total) | Architecture                          | Context | Status       |
|-------|----------------------|---------------------------------------|---------|--------------|
| **Nano** | 3B / 30B         | Hybrid Mamba-Transformer MoE          | 1M      | Live         |
| **Super**| 12B / 120B       | + Latent MoE + MTP + NVFP4            | 1M      | Live         |
| **Ultra**| ~50B / ~500B     | + Latent MoE + MTP + NVFP4            | 1M      | H1 2026      |

NVIDIA's recommended pattern: **Super for complex reasoning and planning, Nano for fast targeted subtasks.** NeMoCode's formations encode this directly.

## Architecture

```
+------------------------------------------------------------+
|                     CLI (Typer)                             |
|  nemo chat | nemo code | nemo endpoint | nemo formation   |
+------------------------------------------------------------+
|                     TUI (Textual)                          |
|  Interactive chat | Tool viz | Formation/endpoint switch   |
+------------------------------------------------------------+
|                  Workflows                                 |
|  code_agent | research | rag | doc_ops (future)           |
+------------------------------------------------------------+
|         Core: Registry | Scheduler | Sessions              |
|  Endpoints -> Manifests -> Formations -> Tool orchestration|
+--------------+---------------------------------------------+
|  Providers   |              Tools                          |
|  nim_chat    |  fs (read/write/edit/ls)                    |
|  nim_embed   |  bash (shell exec)                          |
|  nim_rerank  |  git (status/diff/log/commit)               |
|  nim_parse   |  rg (ripgrep search)                        |
|  (per-cap)   |  http (URL fetch)                           |
+--------------+---------------------------------------------+
```

## Core Concepts

### Endpoints
Where a model lives. Tiers:
- `dev-hosted` -- build.nvidia.com API Catalog (DGX Cloud preview)
- `prod-hosted` -- AI Enterprise / partner dedicated NIM
- `local-nim` -- Self-hosted NIM container
- `inference-partner` -- Together AI, DeepInfra, Fireworks, Baseten, etc.
- `openrouter` -- OpenRouter gateway

### Manifests
Architecture-specific model quirks as typed config. For Nemotron 3, this includes:
```yaml
nvidia/nemotron-3-super-120b-a12b:
  moe:
    total_params_b: 120
    active_params_b: 12
    uses_latent_moe: true       # 4x expert specialization
    uses_mtp: true              # Native speculative decoding
    uses_mamba: true            # Hybrid Mamba-Transformer
    precision: "nvfp4"          # Pretrained on Blackwell
  reasoning:
    supports_thinking: true
    thinking_param: "enable_thinking"
    thinking_budget_param: "reasoning_budget"
    no_think_tag: "/no_think"
    supports_budget_control: true
```

### Usage
```bash
# Interactive REPL (primary experience)
nemo code

# One-shot coding
nemo code "fix the type error in auth.py"

# Auto-approve tool calls
nemo code "create a test file" -y

# Pipe input
cat error.log | nemo code "diagnose this"

# Use a formation
nemo code -f super-nano "implement rate limiting"

# Simple chat (no tools)
nemo chat "explain the Mamba architecture"

# Endpoints
nemo endpoint ls
nemo endpoint test nim-super

# Models
nemo model ls
nemo model show nvidia/nemotron-3-super-120b-a12b

# Formations
nemo formation ls
nemo formation show solo

# Hardware
nemo hardware show
nemo hardware recommend

# Sessions
nemo session ls

# Pricing
nemo obs pricing
```

## Comparison with Other Agentic Coding Tools

| Feature                  | NeMoCode | Claude Code | Codex CLI | Aider | OpenCode |
|--------------------------|----------|-------------|-----------|-------|----------|
| NVIDIA Nemotron 3 native | Yes      | No          | No        | No    | No       |
| Multi-model formations   | Yes      | No          | No        | No    | No       |
| Manifest-aware (MoE, MTP, Mamba) | Yes | No | No | No | No |
| Hardware-aware recommendations | Yes | No | No | No | No |
| MCP tool server support  | Yes      | Yes         | No        | No    | Yes      |
| Chat Completions API only| Yes      | No          | Yes       | Yes   | Yes      |
| Open source              | MIT      | Proprietary | Apache 2.0| Apache 2.0 | MIT |

## Available Endpoints (Out of Box)

**Hosted (API Catalog):** nim-super, nim-nano, nim-ultra (placeholder), nim-embed, nim-rerank, nim-parse

**Third-party:** openrouter-super, together-super, deepinfra-super

**Local:** local-nim-super, local-nim-nano

## Configuration

Merges: **defaults -> ~/.config/nemocode/config.yaml -> .nemocode.yaml -> env**

Env overrides: `NEMOCODE_ENDPOINT`, `NEMOCODE_FORMATION`

### Getting an API Key

1. Visit [build.nvidia.com](https://build.nvidia.com)
2. Create an account or sign in
3. Navigate to any Nemotron 3 model and click "Get API Key"
4. Export it: `export NVIDIA_API_KEY="nvapi-..."`

Or store it persistently:
```bash
echo 'export NVIDIA_API_KEY="nvapi-..."' >> ~/.config/nemocode/env.sh
source ~/.config/nemocode/env.sh
```

## Design Principles

1. **Nemotron 3 native.** Not Llama fine-tunes. Manifests encode Latent MoE, MTP, Mamba, NVFP4, reasoning budget control.
2. **Chat Completions as LCD.** No Responses API dependency.
3. **Model quirks in typed config.** `enable_thinking`, `thinking_budget`, `force_nonempty_content`, `/no_think` -- never string hacks.
4. **App-side orchestration.** Fan-out, retries, coordination in NeMoCode, not model-native parallel tool calling.
5. **Capability-first providers.** `chat | embed | rerank | parse | speech` with transport adapters underneath.
6. **Honest about hardware.** Super needs 80GB GPU. Ultra needs 8xH100. Nano runs on DGX Spark.

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Ensure `ruff check src/ tests/` passes with zero errors
4. Ensure `pytest tests/ -v` passes with all tests green
5. Add SPDX license headers to any new `.py` files
6. Submit a pull request

We follow a standard GitHub flow. Please discuss major changes via issues before submitting PRs.

## License

[MIT](LICENSE)

NeMoCode is a community project. It is **not** an official NVIDIA product.
NVIDIA, Nemotron, NIM, and related marks are trademarks of NVIDIA Corporation.

## Roadmap

- [x] Endpoint registry + config schema
- [x] Chat + streaming (manifest-aware)
- [x] Tool execution (fs, bash, git, rg, http)
- [x] Formations + scheduler
- [x] CLI (Typer) + TUI (Textual)
- [x] Session persistence
- [x] Hardware detection and recommendations
- [ ] NeMo Agent Toolkit (AIQ) integration
- [ ] NemoClaw integration (post-GTC announcement)
- [ ] RAG workflow (embed + rerank)
- [ ] Doc-ops workflow (parse + embed)
- [ ] Guardrails + observability
- [ ] LoRA-as-formation-primitive