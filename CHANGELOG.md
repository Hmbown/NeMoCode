# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Structured error types (`NeMoCodeError`, `ToolExecutionError`, `ProviderError`, etc.)
- Circuit breaker pattern for API calls
- Secret scanning in tool output
- Configurable sandbox levels (strict, standard, permissive)
- Health check command (`nemo health`)
- Audit trail with `nemo audit` command
- Tool result caching for read operations
- Lazy tool loading
- Context window optimization with priority-based compaction
- Graceful degradation with actionable error messages
- Structured logging with JSON output option
- Input validation at CLI boundary
- Streaming response buffering

## [0.1.27] - 2026-04-01

### Security

- `_resolve_path()` in `fs.py` now raises `PermissionError` for paths outside project root
- `NEMOCODE_SCRATCH_DIR` env var is validated against path traversal
- Module-level mutable globals properly scoped; sandbox enforced at source
- `bash_exec` `cwd` validated to be within project root (max 10 levels deep)
- `bash_exec` env sanitization expanded (PRIVATE_KEY, CREDENTIALS, AUTH_TOKEN, etc.)
- `read_file` has 10MB hard limit to prevent OOM
- `list_dir` `max_depth` hard-capped at 5

### Added

- Scheduler `_run_tool` logs full tracebacks via `traceback.format_exc()`
- TUI confirmation documented; defense-in-depth via path sandboxing
- `AgentEventKind` enum added to prevent typo-based UI breakage

### Changed

- Stagnation detection upgraded from MD5(12) to SHA256(16) hashes
- `FormationSlot.reasoning_mode` changed to `Literal["low_effort", "full", "off"] | None`
- GPU detection failure logging upgraded from `debug` to `warning`

### Fixed

- `response_format` dict validated in `nim_chat.py` before API call
- Missing type hints added (`_tui_theme`, `sessions` property, `events.Key`, generic params)
- CodeAgent plan approval session transfer verified correct
- `_from_dict` filters extra/unknown fields from cached hardware JSON
- `_format_elapsed_compact` clamps negative values to 0

### Security

- `NeMoCodeConfig` has `@model_validator` for cross-field reference checking

## [0.1.26] - 2026-03-15

### Added

- Initial release
