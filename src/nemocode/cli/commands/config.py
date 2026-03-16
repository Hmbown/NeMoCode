# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo config — display resolved configuration."""

from __future__ import annotations

import typer
import yaml
from rich.console import Console
from rich.syntax import Syntax

from nemocode.config import load_config

console = Console()
config_app = typer.Typer(help="Configuration inspection.")


@config_app.command("show")
def config_show() -> None:
    """Print the resolved configuration (merged defaults + project + env overrides)."""
    cfg = load_config()
    data = cfg.model_dump(mode="json", exclude_defaults=False)
    formatted = yaml.dump(data, default_flow_style=False, sort_keys=False, width=100)
    syntax = Syntax(formatted, "yaml", theme="monokai", word_wrap=True)
    console.print(syntax)
