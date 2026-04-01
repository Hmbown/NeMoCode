# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo rerank — rerank passages by relevance using NVIDIA NIM endpoints."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from nemocode.config import load_config
from nemocode.core.registry import Registry

console = Console()
rerank_app = typer.Typer(help="Rerank passages by relevance to a query.")


@rerank_app.command("query")
def rerank_query(
    query: str = typer.Argument(..., help="The query to rank passages against."),
    passage: Annotated[
        Optional[list[str]],
        typer.Option(
            "--passage",
            "-p",
            help="Passages to rerank. Repeat for multiple.",
        ),
    ] = None,
    endpoint: str = typer.Option(
        "nim-rerank",
        "--endpoint",
        "-e",
        help="Reranking endpoint to use.",
    ),
    top_n: int = typer.Option(
        0,
        "--top-n",
        "-n",
        help="Return only top N results (0 = all).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read passages from stdin (one per line).",
    ),
) -> None:
    """Rerank passages by relevance to a query.

    Examples:
        nemo rerank query "CUDA programming" -p "CUDA is a parallel computing platform" -p "Python is a language"
        nemo rerank query "error handling" --stdin < passages.txt
    """
    passages = list(passage) if passage else []
    if stdin:
        passages.extend(line.strip() for line in sys.stdin if line.strip())
    if not passages:
        console.print("[yellow]No passages provided. Use -p or --stdin.[/yellow]")
        raise typer.Exit(1)

    cfg = load_config()
    registry = Registry(cfg)

    try:
        provider = registry.get_rerank_provider(endpoint)
    except KeyError as exc:
        console.print(f"[red]Endpoint error:[/red] {exc}")
        raise typer.Exit(1) from exc

    n = top_n if top_n > 0 else len(passages)

    try:
        rankings = asyncio.run(provider.rerank(query, passages, top_n=n))
    except Exception as exc:
        console.print(f"[red]Reranking failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        results = [
            {"rank": rank + 1, "index": idx, "score": score, "passage": passages[idx]}
            for rank, (idx, score) in enumerate(rankings)
        ]
        console.print(json.dumps(results, indent=2))
        return

    console.print(f"[bold]Query:[/bold] {query}\n")

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Passage")

    for rank, (idx, score) in enumerate(rankings):
        preview = passages[idx][:100] + "..." if len(passages[idx]) > 100 else passages[idx]
        table.add_row(str(rank + 1), f"{score:.4f}", preview)

    console.print(table)
    console.print(f"\n[dim]Endpoint: {endpoint} | {len(rankings)}/{len(passages)} results[/dim]")
