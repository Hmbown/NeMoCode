# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Repo-aware planning for NVIDIA synthetic data workflows."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".nemocode",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}

# Prefixes that also indicate directories to skip (e.g., .venv-spark, .venv-sglang)
_IGNORE_DIR_PREFIXES = (".venv-", "venv-", ".venv_", "venv_")


def _is_ignored_dir(part: str) -> bool:
    """Check if a path component should be ignored."""
    if part in _IGNORE_DIRS:
        return True
    return part.startswith(_IGNORE_DIR_PREFIXES)

_SECRETS_PATTERNS = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}

_BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".whl",
    ".egg",
}

_LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
}

_LANGUAGE_BY_EXT = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".md": "markdown",
    ".mjs": "javascript",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".swift": "swift",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
}

_DATASET_EXTENSIONS = {
    ".arrow",
    ".csv",
    ".jsonl",
    ".parquet",
    ".tsv",
}


@dataclass
class RepoProfile:
    root: str
    total_files: int
    code_files: int
    doc_files: int
    test_files: int
    dataset_files: list[str]
    primary_languages: list[str]
    frameworks: list[str]
    top_directories: list[str]
    has_tests: bool
    has_docs: bool


def analyze_repo(root: Path) -> RepoProfile:
    """Build a lightweight profile of a repository for data workflow planning."""
    root = root.resolve()

    language_counts: Counter[str] = Counter()
    top_dir_counts: Counter[str] = Counter()
    dataset_files: list[str] = []
    total_files = 0
    code_files = 0
    doc_files = 0
    test_files = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue

        if any(_is_ignored_dir(part) for part in rel.parts):
            continue

        total_files += 1
        suffix = path.suffix.lower()
        language = _LANGUAGE_BY_EXT.get(suffix)
        if language:
            code_files += 1
            language_counts[language] += 1

        if _is_doc_file(rel):
            doc_files += 1
        if _is_test_file(rel):
            test_files += 1
        if suffix in _DATASET_EXTENSIONS:
            dataset_files.append(str(rel))

        if len(rel.parts) > 1:
            top_dir_counts[rel.parts[0]] += 1

    frameworks = _detect_frameworks(root)
    primary_languages = [name for name, _ in language_counts.most_common(5)]
    top_directories = [name for name, _ in top_dir_counts.most_common(6)]

    return RepoProfile(
        root=str(root),
        total_files=total_files,
        code_files=code_files,
        doc_files=doc_files,
        test_files=test_files,
        dataset_files=sorted(dataset_files)[:25],
        primary_languages=primary_languages,
        frameworks=frameworks,
        top_directories=top_directories,
        has_tests=test_files > 0,
        has_docs=doc_files > 0,
    )


def build_repo_data_plan(profile: RepoProfile) -> dict[str, Any]:
    """Create a repo-to-data workflow plan centered on NVIDIA services."""
    repo_summary = _repo_summary(profile)
    task_types = _suggest_task_types(profile)
    repo_areas = profile.top_directories or ["src", "tests", "docs"]
    languages = profile.primary_languages or ["python"]
    repo_context = [
        f"Repository root: {profile.root}",
        f"Primary languages: {', '.join(languages)}",
        "Framework signals: "
        f"{', '.join(profile.frameworks) if profile.frameworks else 'none detected'}",
        f"Top directories: {', '.join(repo_areas)}",
        f"Tests present: {'yes' if profile.has_tests else 'no'}",
        f"Docs present: {'yes' if profile.has_docs else 'no'}",
    ]

    data_designer_preview = {
        "config": {
            "model_configs": [
                {
                    "alias": "code",
                    "provider": "nvidiabuild",
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                    "inference_parameters": {
                        "temperature": 0.4,
                        "top_p": 0.95,
                        "max_tokens": 1200,
                    },
                },
                {
                    "alias": "judge",
                    "provider": "nvidiabuild",
                    "model": "nvidia/nemotron-nano-9b-v2",
                    "inference_parameters": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "max_tokens": 700,
                    },
                },
            ],
            "columns": [
                {
                    "name": "repo_context",
                    "sampler_type": "category",
                    "params": {"values": [repo_summary]},
                },
                {
                    "name": "repo_area",
                    "sampler_type": "category",
                    "params": {"values": repo_areas},
                },
                {
                    "name": "language",
                    "sampler_type": "category",
                    "params": {"values": languages},
                },
                {
                    "name": "task_type",
                    "sampler_type": "category",
                    "params": {"values": task_types},
                },
                {
                    "name": "difficulty",
                    "sampler_type": "category",
                    "params": {"values": ["easy", "medium", "hard"]},
                },
                {
                    "name": "user_request",
                    "output_type": "text",
                    "model_alias": "code",
                    "prompt": (
                        "Generate a realistic software engineering request for a "
                        "{{language}} repository task in {{repo_area}}. "
                        "Focus on {{task_type}} work at {{difficulty}} difficulty. "
                        "Repository context: {{repo_context}}. "
                        "Return only the user request text."
                    ),
                },
                {
                    "name": "acceptance_criteria",
                    "output_type": "text",
                    "model_alias": "judge",
                    "prompt": (
                        "Write concise acceptance criteria for the following request: "
                        "{{user_request}}. Keep it repo-aware and testable."
                    ),
                },
            ],
        }
    }

    recommended_stack = {
        "data_designer": {
            "recommendation": "yes",
            "why": (
                "Turn repo structure plus NeMoCode-generated seed artifacts into synthetic "
                "coding, test, and documentation tasks."
            ),
            "service_url": "http://localhost:8080",
            "preview_endpoint": "/v1/data-designer/preview",
            "jobs_endpoint": "/v1/data-designer/jobs",
        },
        "evaluator": {
            "recommendation": "yes",
            "why": (
                "Score generated tasks, RAG flows, or agent behaviors before promoting "
                "synthetic data into training or benchmark sets."
            ),
            "service_url": "http://localhost:8080",
            "jobs_endpoint": "/v2/evaluation/jobs",
        },
        "safe_synthesizer": {
            "recommendation": "conditional",
            "why": (
                "Use only when private CSV, CRM, ticketing, support, or tabular product data "
                "must be anonymized or synthesized before mixing with repo-derived data."
            ),
            "service_url": "http://localhost:8080/v1beta1/safe-synthesizer",
        },
        "curator": {
            "recommendation": "later",
            "why": (
                "Best once you are scaling beyond repo-local prompts into larger corpora, "
                "crawl dumps, design docs, tickets, or multimodal training assets."
            ),
            "package": "nemo-curator",
        },
    }

    return {
        "repo_profile": asdict(profile),
        "repo_summary": repo_summary,
        "recommended_stack": recommended_stack,
        "repo_to_data_mvp": {
            "goal": (
                "Analyze the repository, extract grounded seed artifacts, and use NeMo "
                "Data Designer to generate realistic repo-specific tasks."
            ),
            "seed_artifacts": [
                {
                    "name": "repo_profile.yaml",
                    "purpose": "Machine-readable repo summary for prompt grounding.",
                },
                {
                    "name": "file_manifest.jsonl",
                    "purpose": "Paths and metadata for candidate source, test, and docs files.",
                },
                {
                    "name": "task_taxonomy.yaml",
                    "purpose": "Repo-specific task families and acceptance checks.",
                },
            ],
            "task_families": task_types,
            "recommended_next_steps": [
                "Run `nemo setup data` to install the documented NVIDIA prerequisites.",
                "Launch NeMo Data Designer and verify `POST /v1/data-designer/preview`.",
                "Generate repo-aware seeds from source, tests, and docs.",
                "Preview a small batch, then promote to async job generation.",
                "Use NeMo Evaluator to score the generated tasks before wider use.",
            ],
        },
        "data_designer_starter": {
            "provider_registry": {
                "default": "nvidiabuild",
                "providers": [
                    {
                        "name": "nvidiabuild",
                        "endpoint": "https://integrate.api.nvidia.com/v1",
                        "api_key": "NIM_API_KEY",
                    }
                ],
            },
            "context": repo_context,
            "starter_preview_request": data_designer_preview,
        },
    }


def render_plan(plan: dict[str, Any], output_format: str) -> str:
    """Serialize a repo data plan for file output."""
    if output_format == "json":
        return json.dumps(plan, indent=2) + "\n"

    import yaml

    return yaml.safe_dump(plan, sort_keys=False)


def _is_doc_file(rel: Path) -> bool:
    return "docs" in rel.parts or rel.suffix.lower() in {".md", ".rst"}


def _is_test_file(rel: Path) -> bool:
    name = rel.name.lower()
    return (
        "tests" in rel.parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".spec.ts")
        or name.endswith(".test.ts")
        or name.endswith(".spec.js")
        or name.endswith(".test.js")
    )


def _detect_frameworks(root: Path) -> list[str]:
    frameworks: list[str] = []

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text())
        except Exception:
            data = {}
        deps = _flatten_dependency_names(data)
        frameworks.extend(_match_frameworks(deps))

    package_json = root / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text())
        except Exception:
            data = {}
        deps = set(data.get("dependencies", {})) | set(data.get("devDependencies", {}))
        frameworks.extend(_match_frameworks(deps))

    cargo_toml = root / "Cargo.toml"
    if cargo_toml.exists():
        text = cargo_toml.read_text(errors="replace").lower()
        if "axum" in text:
            frameworks.append("Axum")
        if "tokio" in text:
            frameworks.append("Tokio")

    if (root / ".github" / "workflows").exists():
        frameworks.append("GitHub Actions")

    return _dedupe(frameworks)


def _flatten_dependency_names(data: dict[str, Any]) -> set[str]:
    names: set[str] = set()

    project = data.get("project")
    if isinstance(project, dict):
        for dep in project.get("dependencies", []):
            if isinstance(dep, str) and dep:
                names.add(dep.split()[0].split("[")[0].split(">=")[0].split("==")[0])

        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for values in optional.values():
                for dep in values or []:
                    if isinstance(dep, str) and dep:
                        names.add(dep.split()[0].split("[")[0].split(">=")[0].split("==")[0])

    tool = data.get("tool", {})
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    poetry_deps = poetry.get("dependencies", {}) if isinstance(poetry, dict) else {}
    if isinstance(poetry_deps, dict):
        names.update(poetry_deps)

    return {name.lower() for name in names}


def _match_frameworks(deps: set[str]) -> list[str]:
    frameworks = []
    if "django" in deps:
        frameworks.append("Django")
    if "fastapi" in deps:
        frameworks.append("FastAPI")
    if "flask" in deps:
        frameworks.append("Flask")
    if "pytest" in deps:
        frameworks.append("Pytest")
    if "rich" in deps:
        frameworks.append("Rich")
    if "textual" in deps:
        frameworks.append("Textual")
    if "typer" in deps:
        frameworks.append("Typer")
    if "react" in deps:
        frameworks.append("React")
    if "next" in deps:
        frameworks.append("Next.js")
    if "vite" in deps:
        frameworks.append("Vite")
    if "express" in deps:
        frameworks.append("Express")
    if "jest" in deps:
        frameworks.append("Jest")
    if "vitest" in deps:
        frameworks.append("Vitest")
    if "electron" in deps:
        frameworks.append("Electron")
    return frameworks


def _repo_summary(profile: RepoProfile) -> str:
    languages = ", ".join(profile.primary_languages) if profile.primary_languages else "unknown"
    frameworks = ", ".join(profile.frameworks) if profile.frameworks else "none detected"
    top_dirs = ", ".join(profile.top_directories) if profile.top_directories else "root-heavy"
    return (
        f"Repo uses {languages}. Framework signals: {frameworks}. "
        f"Top directories: {top_dirs}. "
        f"Tests present: {'yes' if profile.has_tests else 'no'}. "
        f"Docs present: {'yes' if profile.has_docs else 'no'}."
    )


def _suggest_task_types(profile: RepoProfile) -> list[str]:
    task_types = [
        "bugfix_request",
        "feature_request",
        "refactor_request",
    ]
    if profile.has_tests:
        task_types.extend(["test_generation", "regression_hunt"])
    if profile.has_docs:
        task_types.extend(["repo_qa", "docs_sync"])
    if len(profile.primary_languages) > 1:
        task_types.append("cross_stack_change")
    if profile.dataset_files:
        task_types.append("data_pipeline_change")
    return _dedupe(task_types)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


# ---------------------------------------------------------------------------
# Seed export
# ---------------------------------------------------------------------------

_MAX_FILE_SIZE_BYTES = 256 * 1024  # 256 KB — skip large files in manifests


def _should_exclude(rel: Path) -> bool:
    """Return True if the file should be excluded from seed artifacts."""
    name = rel.name.lower()
    if name in _SECRETS_PATTERNS or name.startswith(".env"):
        return True
    if rel.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    if name in _LOCKFILE_NAMES:
        return True
    if any(part.startswith(".") and part not in {".github"} for part in rel.parts[:-1]):
        return False  # allow dotfiles in non-hidden dirs
    return False


def _file_category(rel: Path) -> str:
    if _is_test_file(rel):
        return "test"
    if _is_doc_file(rel):
        return "doc"
    lang = _LANGUAGE_BY_EXT.get(rel.suffix.lower())
    if lang:
        return "source"
    return "other"


@dataclass
class SeedExportResult:
    output_dir: str
    profile_path: str
    manifest_path: str
    taxonomy_path: str
    context_packs_path: str
    file_count: int


def export_seeds(root: Path, output_dir: Path | None = None) -> SeedExportResult:
    """Scan a repo and write grounded seed artifacts for Data Designer.

    Writes to ``output_dir`` (default: ``<root>/.nemocode/data/``):
    - repo_profile.yaml — machine-readable repo summary
    - file_manifest.jsonl — paths + metadata for source, test, docs files
    - task_taxonomy.yaml — repo-specific task families
    - context_packs.jsonl — sampled file snippets for prompt grounding
    """
    root = root.resolve()
    if output_dir is None:
        output_dir = root / ".nemocode" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = analyze_repo(root)

    # 1. repo_profile.yaml
    import yaml

    profile_path = output_dir / "repo_profile.yaml"
    profile_path.write_text(yaml.safe_dump(asdict(profile), sort_keys=False))

    # 2. file_manifest.jsonl
    manifest_path = output_dir / "file_manifest.jsonl"
    manifest_entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(_is_ignored_dir(part) for part in rel.parts):
            continue
        if _should_exclude(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_FILE_SIZE_BYTES:
            continue
        entry = {
            "path": str(rel),
            "category": _file_category(rel),
            "language": _LANGUAGE_BY_EXT.get(rel.suffix.lower(), ""),
            "size_bytes": size,
        }
        manifest_entries.append(entry)

    with manifest_path.open("w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    # 3. task_taxonomy.yaml
    taxonomy_path = output_dir / "task_taxonomy.yaml"
    task_types = _suggest_task_types(profile)
    taxonomy = {
        "repo_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_families": task_types,
        "difficulty_levels": ["easy", "medium", "hard"],
        "areas": profile.top_directories or ["src", "tests", "docs"],
        "languages": profile.primary_languages or ["python"],
    }
    taxonomy_path.write_text(yaml.safe_dump(taxonomy, sort_keys=False))

    # 4. context_packs.jsonl — sample file snippets for prompt grounding
    context_packs_path = output_dir / "context_packs.jsonl"
    _write_context_packs(root, manifest_entries, context_packs_path)

    return SeedExportResult(
        output_dir=str(output_dir),
        profile_path=str(profile_path),
        manifest_path=str(manifest_path),
        taxonomy_path=str(taxonomy_path),
        context_packs_path=str(context_packs_path),
        file_count=len(manifest_entries),
    )


_CONTEXT_PACK_MAX_LINES = 80
_CONTEXT_PACK_MAX_FILES = 50


def _write_context_packs(
    root: Path,
    manifest_entries: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Sample the first N lines of source/test/doc files as context packs."""
    source_files = [e for e in manifest_entries if e["category"] in ("source", "test", "doc")]
    # Prioritize by category: source first, then test, then doc
    source_files.sort(key=lambda e: {"source": 0, "test": 1, "doc": 2}.get(e["category"], 3))
    sampled = source_files[:_CONTEXT_PACK_MAX_FILES]

    with out_path.open("w") as f:
        for entry in sampled:
            file_path = root / entry["path"]
            try:
                lines = file_path.read_text(errors="replace").splitlines()[:_CONTEXT_PACK_MAX_LINES]
            except OSError:
                continue
            pack = {
                "path": entry["path"],
                "category": entry["category"],
                "language": entry["language"],
                "snippet": "\n".join(lines),
            }
            f.write(json.dumps(pack) + "\n")


# ---------------------------------------------------------------------------
# SFT / instruction-tuning dataset export
# ---------------------------------------------------------------------------


@dataclass
class SFTExportResult:
    output_path: str
    record_count: int


def _extract_python_definitions(source: str) -> list[dict[str, str]]:
    """Extract function and class definitions with their docstrings from Python source."""
    import ast

    defs: list[dict[str, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return defs

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
            name = node.name
            if name.startswith("_") and not name.startswith("__"):
                continue  # skip private helpers for cleaner data
            docstring = ast.get_docstring(node) or ""
            start = node.lineno - 1
            end = min(node.end_lineno or start + 30, len(lines))
            body = "\n".join(lines[start:end])
            if len(body) > 2000:
                body = body[:2000] + "\n    # ... (truncated)"
            defs.append({"kind": kind, "name": name, "docstring": docstring, "code": body})
        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            start = node.lineno - 1
            end = min(node.end_lineno or start + 50, len(lines))
            body = "\n".join(lines[start:end])
            if len(body) > 2500:
                body = body[:2500] + "\n    # ... (truncated)"
            defs.append({"kind": "class", "name": node.name, "docstring": docstring, "code": body})

    return defs


# High-signal templates only. These are grounded in the actual implementation and
# avoid synthetic "gold answers" that are mostly boilerplate.
_SFT_TEMPLATES = [
    {
        "task_type": "implementation_reconstruction",
        "user": (
            "Implement the `{name}` {kind} in `{filepath}` for NeMoCode.\n\n"
            "Required signature:\n```python\n{signature}\n```\n\n"
            "Behavior to preserve:\n{behavior_hint}"
        ),
        "assistant": "```python\n{code}\n```",
    },
    {
        "task_type": "code_explanation",
        "user": "Explain what the `{name}` {kind} does in `{filepath}`. Walk through the logic step by step.",
        "assistant": (
            "The `{name}` {kind} lives in `{filepath}` and belongs to NeMoCode's {module_area} layer.\n\n"
            "Signature:\n```python\n{signature}\n```\n\n"
            "What it does:\n"
            "- {behavior_hint}\n"
            "- {feature_summary}\n"
            "- It fits into the {module_area} layer, which handles {area_description}.\n\n"
            "Implementation:\n```python\n{code}\n```"
        ),
    },
    {
        "task_type": "repo_qa",
        "user": "What is the purpose of `{filepath}` in the NeMoCode project? Specifically, what does `{name}` do?",
        "assistant": (
            "`{filepath}` is part of NeMoCode's {module_area} layer.\n\n"
            "The `{name}` {kind} {behavior_hint}\n\n"
            "Signature:\n```python\n{signature}\n```\n\n"
            "Implementation:\n```python\n{code}\n```\n\n"
            "This area of the project is responsible for {area_description}."
        ),
    },
]

_DEFAULT_SFT_TASK_TYPES = {template["task_type"] for template in _SFT_TEMPLATES}

_EXTENSION_IDEAS = [
    "custom error messages",
    "a timeout parameter",
    "retry logic with exponential backoff",
    "a dry-run mode",
    "JSON output format",
    "streaming results",
    "batch processing multiple inputs",
    "caching for repeated calls",
    "progress reporting via callbacks",
    "configurable logging levels",
]


def _definition_line_count(code: str) -> int:
    return len([line for line in code.splitlines() if line.strip()])


def _definition_signature(code: str) -> str:
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("def ", "async def ", "class ")):
            return stripped
    return code.splitlines()[0].strip() if code.splitlines() else ""


def _behavior_hint(docstring: str, kind: str) -> str:
    if docstring:
        sentence = docstring.split(".")[0].strip()
        if sentence:
            return sentence[0].upper() + sentence[1:] + "."
    return f"implements repo-specific {kind}-level behavior."


def _feature_summary(code: str) -> str:
    features: list[str] = []
    first_line = _definition_signature(code)
    if first_line.startswith("async def"):
        features.append("It is asynchronous.")
    if "httpx." in code or "requests." in code:
        features.append("It performs HTTP client work.")
    if "subprocess." in code or "Popen(" in code:
        features.append("It launches or manages subprocesses.")
    if "typer." in code:
        features.append("It is wired into the CLI surface.")
    if "Path(" in code or ".read_text(" in code or ".write_text(" in code:
        features.append("It reads or writes repository files.")
    if "yield " in code:
        features.append("It produces values incrementally.")
    if "return " in code and "yield " not in code:
        features.append("It returns a computed value to its caller.")
    if "try:" in code:
        features.append("It includes explicit error handling.")
    if not features:
        features.append("Its behavior is primarily encoded directly in the implementation.")
    return " ".join(features)


def _pick_context_pack(
    filepath: str,
    module_area: str,
    context_packs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not context_packs:
        return None

    target_name = Path(filepath).name
    target_stem = Path(filepath).stem
    target_parts = filepath.split("/")

    for pack in context_packs:
        pack_path = str(pack.get("path", ""))
        if not pack_path:
            continue
        if Path(pack_path).name == target_name:
            return pack

    for pack in context_packs:
        pack_path = str(pack.get("path", ""))
        if target_stem and target_stem in Path(pack_path).stem:
            return pack

    for pack in context_packs:
        pack_path = str(pack.get("path", ""))
        pack_parts = pack_path.split("/")
        if target_parts and pack_parts and target_parts[0] == pack_parts[0]:
            return pack

    for pack in context_packs:
        pack_path = str(pack.get("path", ""))
        if module_area and module_area in pack_path.split("/"):
            return pack

    return context_packs[0]

_AREA_DESCRIPTIONS = {
    "providers": "communication with LLM backends (NIM, Ollama, vLLM, etc.)",
    "cli": "the command-line interface and user-facing commands",
    "core": "the central business logic, registry, and orchestration",
    "tools": "the agent tool implementations (filesystem, git, bash, etc.)",
    "config": "configuration loading, schema validation, and defaults",
    "workflows": "high-level agent workflows and multi-step tasks",
    "skills": "slash-command skills like /commit and /review",
    "tests": "the automated regression and behavioral verification suite",
    "scripts": "supporting scripts for training, deployment, and maintenance workflows",
}


def export_sft(
    root: Path,
    output_path: Path | None = None,
    seed_dir: Path | None = None,
    max_records: int = 0,
    *,
    include_tests: bool = False,
    task_types: set[str] | None = None,
) -> SFTExportResult:
    """Export a code-grounded JSONL dataset for SFT / instruction tuning.

    Scans real implementation files, extracts function and class definitions, then
    generates grounded Q&A pairs from those definitions. By default, test files are
    excluded because they tend to dominate the repo and produce lower-signal SFT
    targets than the implementation itself.

    This produces training data suitable for fine-tuning a coding assistant
    that understands this specific codebase, with a bias toward implementation-
    reconstruction and repo-grounded explanation tasks.
    """
    import random

    root = root.resolve()
    if output_path is None:
        output_path = root / ".nemocode" / "data" / "sft_dataset.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_dir is None:
        default_seed_dir = root / ".nemocode" / "data"
        if default_seed_dir.exists():
            seed_dir = default_seed_dir

    profile = analyze_repo(root)
    repo_summary = _repo_summary(profile)

    selected_task_types = task_types or _DEFAULT_SFT_TASK_TYPES

    def _candidate_python_files() -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        for py_file in sorted(root.glob("*.py")):
            if py_file not in seen:
                seen.add(py_file)
                candidates.append(py_file)

        candidate_dirs = [root / "src", root / "scripts"]
        if include_tests:
            candidate_dirs.append(root / "tests")

        for src_dir in candidate_dirs:
            if not src_dir.exists():
                continue
            for py_file in sorted(src_dir.rglob("*.py")):
                if py_file not in seen:
                    seen.add(py_file)
                    candidates.append(py_file)

        return candidates

    def _load_context_packs(seed_root: Path | None) -> list[dict[str, Any]]:
        if seed_root is None:
            return []
        context_path = seed_root / "context_packs.jsonl"
        if not context_path.exists():
            return []

        packs: list[dict[str, Any]] = []
        try:
            for line in context_path.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    pack = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if pack.get("path") and pack.get("snippet"):
                    packs.append(pack)
        except OSError:
            return []
        return packs

    # Scan actual implementation files (not venvs, not generated)
    source_files: list[tuple[str, str]] = []  # (relative_path, content)
    for py_file in _candidate_python_files():
        rel = py_file.relative_to(root)
        if any(_is_ignored_dir(part) for part in rel.parts):
            continue
        if py_file.name == "__init__.py" and py_file.stat().st_size < 50:
            continue
        try:
            content = py_file.read_text(errors="replace")
            if content.strip():
                source_files.append((str(rel), content))
        except OSError:
            continue

    context_packs = _load_context_packs(seed_dir)

    # Extract definitions from all source files
    all_defs: list[dict[str, Any]] = []
    for filepath, content in source_files:
        defs = _extract_python_definitions(content)
        for d in defs:
            if not include_tests and filepath.startswith("tests/"):
                continue
            if not include_tests and (d["name"].startswith("test_") or d["name"].startswith("Test")):
                continue
            if _definition_line_count(d["code"]) < 2:
                continue
            d["filepath"] = filepath
            # Determine module area
            parts = filepath.split("/")
            module_area = "core"
            if parts and parts[0] == "tests":
                module_area = "tests"
            for part in parts:
                if part in _AREA_DESCRIPTIONS:
                    module_area = part
                    break
            d["module_area"] = module_area
            all_defs.append(d)

    # Shuffle and generate records
    random.seed(42)
    random.shuffle(all_defs)

    records: list[dict[str, Any]] = []
    for defn in all_defs:
        # Generate one record per template for each definition
        for template in _SFT_TEMPLATES:
            if max_records and len(records) >= max_records:
                break
            if template["task_type"] not in selected_task_types:
                continue

            name = defn["name"]
            kind = defn["kind"]
            filepath = defn["filepath"]
            code = defn["code"]
            docstring = defn["docstring"]
            module_area = defn["module_area"]
            signature = _definition_signature(code)

            # Build template variables
            area_description = _AREA_DESCRIPTIONS.get(
                module_area, "project functionality"
            )
            behavior_hint = _behavior_hint(docstring, kind)
            feature_summary = _feature_summary(code)

            fmt = {
                "name": name, "kind": kind, "filepath": filepath,
                "code": code, "signature": signature,
                "behavior_hint": behavior_hint,
                "feature_summary": feature_summary,
                "module_area": module_area, "area_description": area_description,
            }

            try:
                user_msg = template["user"].format(**fmt)
                assistant_msg = template["assistant"].format(**fmt)
            except (KeyError, IndexError):
                continue

            system_msg = (
                f"You are a senior software engineer working on NeMoCode, "
                f"a terminal-first agentic coding CLI for NVIDIA Nemotron models. "
                f"Repository context: {repo_summary}"
            )
            pack = _pick_context_pack(filepath, module_area, context_packs)
            if pack:
                system_msg += (
                    f"\n\nRelevant file: {pack['path']}\n"
                    f"```\n{pack['snippet']}\n```"
                )

            record = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "metadata": {
                    "task_type": template["task_type"],
                    "definition": name,
                    "filepath": filepath,
                    "module_area": module_area,
                    "source": "nemocode-export-sft-v2",
                },
            }
            records.append(record)

        if max_records and len(records) >= max_records:
            break

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return SFTExportResult(
        output_path=str(output_path),
        record_count=len(records),
    )


def build_preview_config(profile: RepoProfile) -> dict[str, Any]:
    """Build a Data Designer preview config from a repo profile.

    Returns the config dict suitable for passing to DataDesignerClient.preview().
    """
    plan = build_repo_data_plan(profile)
    return plan["data_designer_starter"]["starter_preview_request"]["config"]


# ---------------------------------------------------------------------------
# NIM API-based synthetic data generation
# ---------------------------------------------------------------------------

_DEFAULT_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1"
_DEFAULT_NIM_MODEL = "nvidia/nemotron-3-super-120b-a12b"

_GENERATE_SYSTEM_PROMPT = (
    "You are a synthetic training-data generator for a coding assistant. "
    "Given a real code snippet from a repository and a task description, "
    "generate a realistic instruction-tuning example.\n\n"
    "Output ONLY valid JSON with exactly two keys:\n"
    '  "user_request": a natural developer question or task request about the code\n'
    '  "assistant_response": a detailed, expert response that demonstrates understanding '
    "of the actual code shown\n\n"
    "Do not include markdown fences around the JSON. Output raw JSON only."
)

_GENERATE_USER_TEMPLATE = (
    "Repository context: {repo_summary}\n\n"
    "Code snippet from `{path}` ({language}):\n"
    "```\n{snippet}\n```\n\n"
    "Task type: {task_type}\n"
    "Difficulty: {difficulty}\n\n"
    "Generate a realistic training example based on this code."
)


@dataclass
class GenerateResult:
    output_path: str
    record_count: int
    skipped: int
    endpoint: str
    model: str


def _resolve_nim_api_key() -> str | None:
    """Resolve an API key for the NIM endpoint."""
    try:
        from nemocode.core.credentials import get_credential

        for name in ("NVIDIA_API_KEY", "NGC_CLI_API_KEY", "NIM_API_KEY"):
            key = get_credential(name)
            if key:
                return key
    except Exception:
        pass
    import os
    for name in ("NVIDIA_API_KEY", "NGC_CLI_API_KEY", "NIM_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def generate_sft_via_nim(
    seed_dir: Path,
    output_path: Path | None = None,
    num_records: int = 100,
    endpoint: str = _DEFAULT_NIM_ENDPOINT,
    model: str = _DEFAULT_NIM_MODEL,
    *,
    progress_callback: Any | None = None,
) -> GenerateResult:
    """Generate synthetic SFT data by calling the NIM API (Nemotron Super).

    This is the lightweight "CLI-to-API proxy" path — no Docker or Data Designer
    service needed. Just an NVIDIA_API_KEY for build.nvidia.com (or a local endpoint).

    Reads seed artifacts from *seed_dir* (context_packs.jsonl, task_taxonomy.yaml),
    calls the model to generate realistic Q&A pairs, and writes Customizer-ready JSONL.
    """
    import random
    import time

    import httpx
    import yaml

    seed_dir = Path(seed_dir).resolve()

    # Load seed artifacts
    context_packs_path = seed_dir / "context_packs.jsonl"
    taxonomy_path = seed_dir / "task_taxonomy.yaml"

    if not context_packs_path.exists():
        raise FileNotFoundError(
            f"context_packs.jsonl not found in {seed_dir}. "
            "Run `nemo data export-seeds` first."
        )

    context_packs: list[dict[str, Any]] = []
    with context_packs_path.open() as f:
        for line in f:
            if line.strip():
                context_packs.append(json.loads(line))

    if not context_packs:
        raise ValueError("context_packs.jsonl is empty. Run `nemo data export-seeds` first.")

    # Load taxonomy (optional — use defaults if missing)
    task_types = ["bugfix_request", "feature_request", "code_explanation", "refactor_request"]
    difficulties = ["easy", "medium", "hard"]
    repo_summary = ""

    if taxonomy_path.exists():
        taxonomy = yaml.safe_load(taxonomy_path.read_text()) or {}
        task_types = taxonomy.get("task_families", task_types)
        difficulties = taxonomy.get("difficulty_levels", difficulties)

    # Try to load repo_profile for summary
    profile_path = seed_dir / "repo_profile.yaml"
    if profile_path.exists():
        profile_data = yaml.safe_load(profile_path.read_text()) or {}
        langs = ", ".join(profile_data.get("primary_languages", []))
        frameworks = ", ".join(profile_data.get("frameworks", []))
        repo_summary = f"Languages: {langs}. Frameworks: {frameworks}."

    # Resolve output path
    if output_path is None:
        output_path = seed_dir / "sft_generated.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve API key
    api_key = _resolve_nim_api_key()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Generate records
    records: list[dict[str, Any]] = []
    skipped = 0

    with httpx.Client(timeout=120.0) as client:
        for i in range(num_records):
            pack = context_packs[i % len(context_packs)]
            task_type = random.choice(task_types)
            difficulty = random.choice(difficulties)

            user_prompt = _GENERATE_USER_TEMPLATE.format(
                repo_summary=repo_summary,
                path=pack.get("path", "unknown"),
                language=pack.get("language", "unknown"),
                snippet=pack.get("snippet", "")[:2000],
                task_type=task_type,
                difficulty=difficulty,
            )

            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _GENERATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 2048,
            }

            try:
                resp = client.post(
                    f"{endpoint.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=body,
                )

                if resp.status_code == 429:
                    # Rate limited — wait and retry once
                    time.sleep(5)
                    resp = client.post(
                        f"{endpoint.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=body,
                    )

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                # Strip markdown fences if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

                parsed = json.loads(content)
                user_request = parsed.get("user_request", "")
                assistant_response = parsed.get("assistant_response", "")

                if not user_request or not assistant_response:
                    skipped += 1
                    continue

                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert coding assistant with deep knowledge of "
                                "this repository's codebase, architecture, and conventions."
                            ),
                        },
                        {"role": "user", "content": user_request},
                        {"role": "assistant", "content": assistant_response},
                    ],
                    "metadata": {
                        "task_type": task_type,
                        "difficulty": difficulty,
                        "source_file": pack.get("path", ""),
                        "source": "nemocode-generate-nim-v1",
                    },
                }
                records.append(record)

            except (httpx.HTTPError, json.JSONDecodeError, KeyError):
                skipped += 1
                continue

            if progress_callback is not None:
                progress_callback(i + 1, num_records)

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return GenerateResult(
        output_path=str(output_path),
        record_count=len(records),
        skipped=skipped,
        endpoint=endpoint,
        model=model,
    )
