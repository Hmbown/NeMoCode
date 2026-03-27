# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo guardrails — inspect and test guardrails configuration."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemocode.config import get_api_key, load_config
from nemocode.config.schema import GuardrailsConfig
from nemocode.providers.nim_guardrails import NIMGuardrailsProvider

console = Console()
guardrails_app = typer.Typer(help="Inspect and test NeMo Guardrails configuration.")


def _resolve_text(text: str | None, input_file: Path | None, stdin: bool) -> str:
    if input_file:
        return input_file.read_text()
    if stdin:
        return sys.stdin.read()
    return text or ""


@guardrails_app.command("config")
def guardrails_config() -> None:
    """Show the resolved guardrails configuration and endpoint."""
    cfg = load_config()
    endpoint_name = cfg.guardrails.endpoint
    endpoint = cfg.endpoints.get(endpoint_name)

    table = Table(title="Guardrails Config")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("enabled", str(cfg.guardrails.enabled))
    table.add_row("endpoint", endpoint_name)
    table.add_row("timeout", str(cfg.guardrails.timeout))
    table.add_row("reject_categories", ", ".join(cfg.guardrails.reject_categories))
    table.add_row("endpoint_found", str(endpoint is not None))
    if endpoint:
        table.add_row("base_url", endpoint.base_url)
        table.add_row("model_id", endpoint.model_id)
    console.print(table)


@guardrails_app.command("check")
def guardrails_check(
    text: str | None = typer.Argument(None, help="Text to check."),
    input_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        exists=True,
        dir_okay=False,
        help="Read text from a file.",
    ),
    stdin: bool = typer.Option(False, "--stdin", help="Read text from stdin."),
    endpoint: str | None = typer.Option(None, "--endpoint", help="Override guardrails endpoint."),
    timeout: float | None = typer.Option(None, "--timeout", help="Override timeout in seconds."),
    reject_category: list[str] | None = typer.Option(
        None,
        "--reject-category",
        help="Reject category override. Repeat for multiple.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw result as JSON."),
) -> None:
    """Run guardrails classification against text."""
    body = _resolve_text(text, input_file, stdin).strip()
    if not body:
        console.print("[red]No input text provided.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    endpoint_name = endpoint or cfg.guardrails.endpoint
    ep = cfg.endpoints.get(endpoint_name)
    if not ep:
        console.print(f"[red]Guardrails endpoint not found:[/red] {endpoint_name}")
        raise typer.Exit(1)

    gr_config = GuardrailsConfig(
        enabled=True,
        endpoint=endpoint_name,
        timeout=timeout if timeout is not None else cfg.guardrails.timeout,
        reject_categories=list(reject_category) if reject_category else cfg.guardrails.reject_categories,
    )
    provider = NIMGuardrailsProvider(endpoint=ep, config=gr_config, api_key=get_api_key(ep))

    try:
        result = asyncio.run(provider.check(body))
    except Exception as exc:
        console.print(f"[red]Guardrails check failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    payload = {
        "safe": result.safe,
        "blocked_categories": result.blocked_categories,
        "categories": result.categories,
        "error": result.error,
    }
    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    summary = [
        f"[bold]Endpoint:[/bold] {endpoint_name}",
        f"[bold]Safe:[/bold] {'yes' if result.safe else 'no'}",
    ]
    if result.blocked_categories:
        summary.append(
            f"[bold]Blocked:[/bold] {', '.join(result.blocked_categories)}"
        )
    if result.error:
        summary.append(f"[bold]Error:[/bold] {result.error}")

    console.print(
        Panel(
            "\n".join(summary),
            title="Guardrails Check",
            border_style="green" if result.safe else "red",
        )
    )

    if result.categories:
        table = Table(title="Category Scores")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right")
        for name, score in sorted(result.categories.items(), key=lambda item: item[1], reverse=True):
            table.add_row(name, f"{score:.4f}")
        console.print(table)
