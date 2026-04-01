# NeMoCode — Next Agent Handoff & Quality Improvement Brief

## Project Overview

**NeMoCode** is a terminal-first agentic coding CLI for the NVIDIA Nemotron 3 model ecosystem. It provides an interactive REPL, full-screen TUI, one-shot mode, multi-model formations, and sub-agent orchestration. Pure Python 3.11+, built with Typer, Textual, Rich, httpx, Pydantic v2.

- **Root**: `/home/hmbown/Projects/NeMoCode`
- **Source**: `src/nemocode/` (91 Python files, ~17,500+ lines)
- **Tests**: `tests/` (70+ files, 1030 passing)
- **Entry point**: `nemo = "nemocode.cli.main:app"`
- **Current version**: `0.1.26`

## What Was Already Fixed (Do NOT Re-do These)

A previous agent completed a comprehensive quality pass. These items are **already done**:

### Critical (4 fixed)
- `_resolve_path()` in `fs.py` now raises `PermissionError` for paths outside project root
- `NEMOCODE_SCRATCH_DIR` env var is validated against path traversal
- Module-level mutable globals properly scoped; sandbox enforced at source
- TUI confirmation documented; defense-in-depth via path sandboxing

### High (6 fixed)
- Scheduler `_run_tool` logs full tracebacks via `traceback.format_exc()`
- `bash_exec` `cwd` validated to be within project root (max 10 levels deep)
- Stagnation detection upgraded from MD5(12) to SHA256(16) hashes
- `response_format` dict validated in `nim_chat.py` before API call
- Missing type hints added (`_tui_theme`, `sessions` property, `events.Key`, generic params)
- CodeAgent plan approval session transfer verified correct

### Medium (8 fixed)
- `read_file` has 10MB hard limit to prevent OOM
- `list_dir` `max_depth` hard-capped at 5
- `FormationSlot.reasoning_mode` changed to `Literal["low_effort", "full", "off"] | None`
- `NeMoCodeConfig` has `@model_validator` for cross-field reference checking
- `_from_dict` filters extra/unknown fields from cached hardware JSON
- GPU detection failure logging upgraded from `debug` to `warning`
- `bash_exec` env sanitization expanded (PRIVATE_KEY, CREDENTIALS, AUTH_TOKEN, etc.)
- `AgentEventKind` enum added to prevent typo-based UI breakage

### Additional
- `_format_elapsed_compact` clamps negative values to 0
- All 1030 tests pass, 0 lint errors (excluding pre-existing E501 line-length)
- All imports sorted, unused imports removed, formatting applied

---

## Quality Pass — Completed Items ✅

The following items have been implemented and verified:

### Priority 1: Robustness & Error Handling ✅

- **1.2 Structured Error Types** ✅ — `src/nemocode/core/errors.py`
  - Exception hierarchy: `NeMoCodeError`, `ToolExecutionError`, `ProviderError`, `NetworkError`, `AuthError`, `QuotaError`, `ServerError`, `StagnationError`, `PermissionDeniedError`, `ConfigError`, `SessionError`
  - Scheduler updated to use `ProviderError` and `ToolExecutionError`

- **1.1 Graceful Degradation** ✅ — `src/nemocode/providers/nim_chat.py`, `src/nemocode/core/registry.py`
  - `_classify_error()` distinguishes network, auth, quota, server errors
  - Actionable suggestions (`nemo auth`, `nemo doctor`, `nemo endpoint status`)
  - `NEMOCODE_VERBOSE=1` enables full request/response dumps
  - `Registry.test_endpoint()` for connectivity testing

- **1.3 Circuit Breaker** ✅ — `src/nemocode/core/circuit_breaker.py`
  - `CircuitBreaker` with CLOSED/OPEN/HALF_OPEN states
  - Configurable failure threshold and cooldown
  - Integrated into `NIMChatProvider.stream()` and `complete()`

- **1.4 Input Validation** ✅ — `src/nemocode/core/validators.py`
  - `validate_file_path()`, `validate_timeout()`, `validate_model_id()`, `validate_json_input()`, `validate_command_args()`
  - Applied across CLI commands (chat, code, serve, model, endpoint, customize)

### Priority 2: Performance & Scalability ✅

- **2.1 Streaming Response Buffering** ✅ — `src/nemocode/core/scheduler.py`
  - Batched yielding with configurable delay (`NEMOCODE_STREAM_BATCH_MS`, default 50ms)
  - Tool call and error events yielded immediately; text/thinking batched

- **2.2 Tool Result Caching** ✅ — `src/nemocode/core/tool_cache.py`
  - LRU cache (100 entries) with TTL support
  - Applied to `read_file`, `glob_files`, `search_files`
  - Cache invalidated on file writes
  - Opt-out via `NEMOCODE_TOOL_CACHE=0`

- **2.3 Lazy Tool Loading** ✅ — `src/nemocode/tools/loader.py`
  - Tools imported on-demand when first referenced
  - Opt-out via `NEMOCODE_LAZY_TOOLS=0`

- **2.4 Context Window Optimization** ✅ — `src/nemocode/core/context.py`
  - Priority-based compaction: system > user > recent tool results > assistant > old tool results
  - `compact_to_target()` for specific token targets

### Priority 3: Developer Experience & Observability ✅

- **3.1 Structured Logging** ✅ — `src/nemocode/core/logging_config.py`
  - `JsonFormatter` for JSON output (`NEMOCODE_JSON_LOGS=1`)
  - `StructuredLogger` with auto-attached context fields
  - Tool execution, API calls, circuit breaker events all logged with structured fields

- **3.3 Health Check** ✅ — `src/nemocode/cli/commands/health.py`
  - `nemo health` command checks: API keys, local ports, hardware/GPU, config validity, disk space
  - Rich-formatted output with status table

- **3.4 Audit Trail** ✅ — `src/nemocode/core/audit.py`, `src/nemocode/cli/commands/audit.py`
  - JSONL audit log at `~/.cache/nemocode/audit.log`
  - `nemo audit` command with `--since`, `--limit`, `--type` filters
  - Tool executions logged automatically from scheduler

### Priority 4: Security Hardening ✅

- **4.1 Tool Sandbox Levels** ✅ — `src/nemocode/core/sandbox.py`
  - `SandboxLevel` enum: STRICT, STANDARD, PERMISSIVE
  - `Sandbox` class with `validate_path()`, `can_execute_command()`, `can_write()`, `can_read()`
  - Integrated into `bash.py` and `fs.py`
  - Configurable via `NEMOCODE_SANDBOX_LEVEL` env var

- **4.4 Secret Scanning** ✅ — `src/nemocode/core/secrets.py`
  - `SecretScanner` with patterns for AWS keys, GitHub tokens, Slack tokens, JWTs, private keys, etc.
  - Integrated into `bash.py` (stdout/stderr) and `fs.py` (read_file output)
  - Opt-out via `scanner.enabled = False`

### Priority 6: Documentation ✅

- **6.3 Changelog** ✅ — `CHANGELOG.md`
  - Keep a Changelog format with Unreleased, 0.1.27, and 0.1.26 sections

---

## Remaining Items (Not Yet Implemented)

### Priority 3: Developer Experience
- **3.2 Telemetry & Metrics Export** — OpenTelemetry span export, Prometheus metrics, per-tool latency histograms, token cost tracking
- **3.4 Shell Completion** — Typer shell completion for bash/zsh with endpoint, formation, model, profile completions

### Priority 4: Security
- **4.2 API Key Rotation & Scoping** — Multiple API key profiles, key expiration warnings, automatic rotation hooks, per-endpoint key isolation

### Priority 5: Testing & CI
- **5.1 Integration Tests** — Full REPL session, formation flow, TUI flow tests
- **5.2 Property-Based Testing** — Hypothesis tests for config loading, path sandbox, context compaction, tool schemas
- **5.3 Mutation Testing** — mutmut or cosmic-ray setup, >90% mutation score target
- **5.4 Benchmark Suite** — Startup time, tool latency, context compaction speed, memory usage

### Priority 6: Documentation
- **6.1 API Reference Documentation** — Generate from docstrings
- **6.2 Architecture Decision Records** — docs/adr/ for key architectural decisions
- **6.4 Contributing Guide Enhancements** — How to add tools/providers, code review checklist, release process

### Priority 7: Feature Completeness
- **7.1 Multi-File Edit Atomicity** — Rollback on failure using undo stack
- **7.2 Plan Mode Visual Diff Preview** — Side-by-side diff in TUI before approval
- **7.3 Session Export/Import** — Portable session format for sharing/debugging
- **7.4 Custom Tool Plugin System** — PyPI-distributed tools, git URL sources, version pinning, dependency resolution

---

## How to Work

1. **Pick one priority area** and work through its items
2. **Make changes incrementally** — small, focused commits
3. **Run tests after every change**: `python -m pytest tests/ -q --tb=short`
4. **Run lint after every change**: `ruff check src/nemocode/ && ruff format src/nemocode/`
5. **Update this file** as you complete items — mark them done
6. **Ask the user** if you need clarification on any item
7. **Do NOT break existing functionality** — all 1030 tests must pass at all times

## Quick Reference Commands

```bash
# Run all tests
python -m pytest tests/ -q --tb=short

# Run specific test file
python -m pytest tests/test_scheduler.py -v

# Lint check
ruff check src/nemocode/

# Auto-fix lint issues
ruff check src/nemocode/ --fix

# Format check
ruff format --check src/nemocode/

# Format all files
ruff format src/nemocode/

# Run with coverage
python -m pytest tests/ --cov=src/nemocode --cov-report=term-missing

# Type check (if mypy is installed)
mypy src/nemocode/
```

## Current Test Status
- **1030 passed**, 4 skipped, 0 failures
- **0 non-E501 lint errors** (14 pre-existing E501 line-length warnings in unmodified files)
- **All formatting clean**

## New Files Created in This Pass
- `src/nemocode/core/errors.py` — Structured exception hierarchy
- `src/nemocode/core/circuit_breaker.py` — Circuit breaker pattern
- `src/nemocode/core/secrets.py` — Secret scanning for tool output
- `src/nemocode/core/sandbox.py` — Configurable sandbox levels
- `src/nemocode/core/tool_cache.py` — LRU tool result cache
- `src/nemocode/core/audit.py` — Audit trail logging
- `src/nemocode/core/validators.py` — Input validation functions
- `src/nemocode/core/logging_config.py` — Structured logging setup
- `src/nemocode/cli/commands/health.py` — Health check command
- `src/nemocode/cli/commands/audit.py` — Audit log viewer command
- `CHANGELOG.md` — Conventional changelog
