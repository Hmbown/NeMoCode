# Contributing to NeMoCode

Thanks for your interest in contributing to NeMoCode. This guide covers how to set up a development environment, make changes, and submit pull requests.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) or pip for package management
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/) (for live testing)

## Setup

```bash
git clone https://github.com/Hmbown/NeMoCode.git
cd NeMoCode
pip install -e ".[dev]"
```

Set your API key:
```bash
export NVIDIA_API_KEY="nvapi-..."
```

Or use the built-in setup wizard:
```bash
nemo setup
```

## Development Workflow

1. Create a branch from `main`
2. Make your changes
3. Run checks (see below)
4. Open a pull request

### Running Checks

```bash
ruff check src/ tests/       # Lint
ruff format src/ tests/       # Format
pytest tests/ -q              # Tests
```

### Pre-commit Hooks

Install pre-commit hooks to run checks automatically:
```bash
pip install pre-commit
pre-commit install
```

## Code Style

- **Formatter:** ruff (configured in `pyproject.toml`)
- **Lint rules:** E, F, I, W, UP, B
- **Line length:** 100
- **Imports:** sorted isort-style (ruff handles this)
- **Type hints:** use modern `X | Y` syntax, not `Union[X, Y]`
- **Docstrings:** Google style on all public functions and classes
- **SPDX headers:** all `.py` files must start with:
  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
  # SPDX-License-Identifier: MIT
  ```

## Testing

- Tests live in `tests/` mirroring the `src/nemocode/` structure
- Use `pytest-asyncio` for async tests (auto mode is configured)
- Mock all external API calls (httpx) — no real network calls in tests
- Keep tests fast (< 30s total)

## Project Structure

```
src/nemocode/
  cli/         # CLI commands, TUI, rendering
  config/      # Configuration schema and loading
  core/        # Core engine (scheduler, sessions, context)
  providers/   # NIM API providers (chat, embeddings, rerank)
  skills/      # Built-in skills (commit, review)
  tools/       # Tool registry and implementations
  workflows/   # High-level agent orchestration
```

## Adding a New Tool

1. Create a function in `src/nemocode/tools/` with the `@tool` decorator
2. Add parameters with type hints (JSON Schema is auto-generated)
3. Return a JSON string with results or error
4. Register it in `tools/__init__.py` if not auto-discovered

```python
from nemocode.tools import tool

@tool(description="One-line description shown to the model")
def my_tool(arg: str, count: int = 1) -> str:
    """My tool does something useful.
    arg: The input string to process
    count: Number of times to process
    """
    return json.dumps({"result": arg * count})
```

## Adding a New Agent

Create a markdown file in your project or config directory:

```markdown
---
name: my-agent
role: executor
tools: [read_file, edit_file, bash_exec]
---

You are a specialized agent for...
(custom system prompt here)
```

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/):

```
feat(tools): add my_tool for XYZ
fix(scheduler): resolve race condition in session save
docs(readme): update installation instructions
refactor(providers): extract base class for chat providers
test(core): add session persistence roundtrip test
```

## Pull Requests

- PRs must reference an existing issue
- All checks must pass (lint, format, tests)
- Keep PRs focused — one logical change per PR
- Include tests for new functionality
- Update docstrings for any changed public APIs

## NVIDIA Model Policy

Per `AGENTS.md`:
- Prefer native NVIDIA Nemotron models over Llama-Nemotron variants
- Do not introduce Llama-based models as coding backends
- For Nano 4B, use Nemotron 3 Nano 4B family (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`)

## Questions?

Open a [discussion](https://github.com/Hmbown/NeMoCode/issues) on GitHub.
