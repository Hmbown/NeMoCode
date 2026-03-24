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

        if any(part in _IGNORE_DIRS for part in rel.parts):
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
        if any(part in _IGNORE_DIRS for part in rel.parts):
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


def export_sft(
    root: Path,
    output_path: Path | None = None,
    seed_dir: Path | None = None,
    max_records: int = 0,
) -> SFTExportResult:
    """Export a JSONL dataset suitable for SFT / instruction tuning.

    Each record contains:
    - system: grounding context
    - user: a realistic repo-specific request
    - assistant: acceptance criteria + context
    - metadata: repo area, language, task type, difficulty

    If ``seed_dir`` exists and contains context_packs.jsonl, file snippets
    are woven into the prompts for grounding. Otherwise falls back to
    profile-level context.

    This is for repo-specific assistant tuning, not raw code dumping.
    """
    root = root.resolve()
    if output_path is None:
        output_path = root / ".nemocode" / "data" / "sft_dataset.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_dir is None:
        seed_dir = root / ".nemocode" / "data"

    profile = analyze_repo(root)
    task_types = _suggest_task_types(profile)
    repo_summary = _repo_summary(profile)
    areas = profile.top_directories or ["src", "tests", "docs"]
    languages = profile.primary_languages or ["python"]
    difficulties = ["easy", "medium", "hard"]

    # Load context packs if available
    context_packs: list[dict[str, Any]] = []
    packs_path = seed_dir / "context_packs.jsonl"
    if packs_path.exists():
        for line in packs_path.read_text().splitlines():
            if line.strip():
                try:
                    context_packs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    records: list[dict[str, Any]] = []
    pack_idx = 0

    for task_type in task_types:
        for area in areas:
            for lang in languages:
                for difficulty in difficulties:
                    # Pick a context pack if available
                    context_snippet = ""
                    if context_packs:
                        pack = context_packs[pack_idx % len(context_packs)]
                        pack_idx += 1
                        context_snippet = (
                            f"\n\nRelevant file: {pack['path']}\n"
                            f"```{pack.get('language', '')}\n"
                            f"{pack['snippet'][:500]}\n```"
                        )

                    system_msg = (
                        f"You are a coding assistant for a {lang} repository. "
                        f"Repository context: {repo_summary}{context_snippet}"
                    )

                    user_msg = (
                        f"I need help with a {difficulty} {task_type.replace('_', ' ')} "
                        f"in the {area} area of the repository. "
                        f"The project uses {lang}."
                    )

                    assistant_msg = (
                        f"I'll help with this {task_type.replace('_', ' ')}. "
                        f"Based on the repository structure, here's my approach:\n\n"
                        f"1. Identify the relevant files in {area}/\n"
                        f"2. Analyze the current implementation\n"
                        f"3. Implement the changes with proper {lang} conventions\n"
                        f"4. Add or update tests to verify the changes\n"
                        f"5. Ensure documentation is consistent\n\n"
                        f"Acceptance criteria:\n"
                        f"- Changes are scoped to {area}/\n"
                        f"- All existing tests continue to pass\n"
                        f"- New behavior has test coverage\n"
                        f"- Code follows project conventions"
                    )

                    record = {
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                        "metadata": {
                            "repo_root": str(root),
                            "task_type": task_type,
                            "area": area,
                            "language": lang,
                            "difficulty": difficulty,
                            "source": "nemocode-export-sft",
                        },
                    }
                    records.append(record)

                    if max_records and len(records) >= max_records:
                        break
                if max_records and len(records) >= max_records:
                    break
            if max_records and len(records) >= max_records:
                break
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
