# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo embed — generate text embeddings via NVIDIA NIM endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from nemocode.config import load_config
from nemocode.core.registry import Registry

console = Console()
embed_app = typer.Typer(help="Generate text embeddings.")


@embed_app.command("text")
def embed_text(
    texts: list[str] = typer.Argument(..., help="Text(s) to embed."),
    endpoint: str = typer.Option(
        "nim-embed",
        "--endpoint",
        "-e",
        help="Embedding endpoint to use.",
    ),
    input_type: str = typer.Option(
        "query",
        "--input-type",
        "-t",
        help="Input type: 'query' for search queries, 'passage' for documents.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write embeddings to a JSONL file.",
    ),
    dimensions: bool = typer.Option(
        False,
        "--dims",
        help="Only print the vector dimension, not the full embedding.",
    ),
) -> None:
    """Embed text into vectors using NVIDIA embedding models.

    Examples:
        nemo embed text "What is CUDA?"
        nemo embed text "first text" "second text" --input-type passage
    """
    if not texts:
        console.print("[yellow]No text provided.[/yellow]")
        raise typer.Exit(1)

    cfg = load_config()
    registry = Registry(cfg)

    try:
        provider = registry.get_embedding_provider(endpoint)
    except KeyError as exc:
        console.print(f"[red]Endpoint error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        embeddings = asyncio.run(provider.embed(texts, input_type=input_type))
    except Exception as exc:
        console.print(f"[red]Embedding failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    if dimensions:
        for i, emb in enumerate(embeddings):
            console.print(f"[cyan]Text {i + 1}:[/cyan] {len(emb)} dimensions")
        return

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for text, emb in zip(texts, embeddings):
                f.write(json.dumps({"text": text, "embedding": emb}) + "\n")
        console.print(f"[green]Wrote {len(embeddings)} embeddings to {output}[/green]")
    else:
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            preview = text[:60] + "..." if len(text) > 60 else text
            console.print(f"[bold cyan]Text {i + 1}:[/bold cyan] {preview}")
            console.print(f"  Dimensions: {len(emb)}")
            console.print(f"  First 5: {emb[:5]}")
            console.print(f"  Norm: {sum(x * x for x in emb) ** 0.5:.4f}")

    console.print(f"\n[dim]Endpoint: {endpoint} | Input type: {input_type}[/dim]")


@embed_app.command("similarity")
def embed_similarity(
    text_a: str = typer.Argument(..., help="First text."),
    text_b: str = typer.Argument(..., help="Second text."),
    endpoint: str = typer.Option(
        "nim-embed",
        "--endpoint",
        "-e",
        help="Embedding endpoint to use.",
    ),
) -> None:
    """Compute cosine similarity between two texts."""
    cfg = load_config()
    registry = Registry(cfg)

    try:
        provider = registry.get_embedding_provider(endpoint)
    except KeyError as exc:
        console.print(f"[red]Endpoint error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        embeddings = asyncio.run(provider.embed([text_a, text_b], input_type="query"))
    except Exception as exc:
        console.print(f"[red]Embedding failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    vec_a, vec_b = embeddings[0], embeddings[1]
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(x * x for x in vec_a) ** 0.5
    norm_b = sum(x * x for x in vec_b) ** 0.5
    similarity = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    console.print(f"[bold cyan]Text A:[/bold cyan] {text_a[:80]}")
    console.print(f"[bold cyan]Text B:[/bold cyan] {text_b[:80]}")
    console.print(f"\n[bold]Cosine Similarity:[/bold] {similarity:.4f}")

    if similarity > 0.8:
        console.print("[green]Very similar[/green]")
    elif similarity > 0.5:
        console.print("[yellow]Somewhat similar[/yellow]")
    else:
        console.print("[dim]Not very similar[/dim]")
