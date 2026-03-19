# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo init — create .nemocode.yaml in current directory."""

from __future__ import annotations

import copy
from pathlib import Path

import typer
import yaml
from rich.console import Console

console = Console()

_TEMPLATE = {
    "project": {
        "name": "",
        "description": "",
        "tech_stack": [],
        "conventions": [],
        "context_files": ["README.md"],
        "ignore": ["*.pyc", "__pycache__/", ".venv/", "node_modules/"],
    },
    "permissions": {
        "auto_approve_shell": False,
        "require_confirmation": ["bash_exec", "git_commit"],
    },
}


def init_cmd(
    name: str = typer.Option(None, "--name", help="Project name"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
    endpoint: str = typer.Option(None, "--endpoint", help="Project-specific default endpoint"),
    formation: str = typer.Option(
        None, "--formation", help="Project-specific active formation"
    ),
) -> None:
    """Create a .nemocode.yaml project config in the current directory."""
    config_path = Path.cwd() / ".nemocode.yaml"

    if config_path.exists() and not force:
        console.print(f"[yellow]{config_path} already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(1)

    config = copy.deepcopy(_TEMPLATE)

    # Auto-detect project name
    if name:
        config["project"]["name"] = name
    else:
        config["project"]["name"] = Path.cwd().name

    # Auto-detect tech stack
    cwd = Path.cwd()
    tech_stack = []
    if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
        tech_stack.append("python")
    if (cwd / "package.json").exists():
        tech_stack.append("javascript")
    if (cwd / "Cargo.toml").exists():
        tech_stack.append("rust")
    if (cwd / "go.mod").exists():
        tech_stack.append("go")
    if (cwd / "Dockerfile").exists() or (cwd / "docker-compose.yml").exists():
        tech_stack.append("docker")
    config["project"]["tech_stack"] = tech_stack

    # Auto-detect context files
    context_files = []
    for fname in ["README.md", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]:
        if (cwd / fname).exists():
            context_files.append(fname)
    config["project"]["context_files"] = context_files

    if endpoint:
        config["default_endpoint"] = endpoint
    if formation:
        config["active_formation"] = formation

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created {config_path}[/green]")
    console.print(
        "[dim]Project config now inherits your user-level runtime defaults unless you set"
        " --endpoint or --formation here.[/dim]"
    )
