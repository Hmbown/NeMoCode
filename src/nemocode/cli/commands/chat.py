# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo chat — streaming chat, no tools."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console

from nemocode.config import load_config
from nemocode.core.registry import Registry
from nemocode.core.streaming import Message, Role

console = Console()


def chat_cmd(
    prompt: str = typer.Argument(None, help="Chat message (or pipe via stdin)"),
    endpoint: str = typer.Option(None, "-e", "--endpoint", help="Endpoint to use"),
    think: bool = typer.Option(True, "--think/--no-think", help="Show/hide thinking trace"),
    output_format: str = typer.Option(None, "--output-format", help="Output format: json, text"),
) -> None:
    """Simple streaming chat with a Nemotron model. No tools."""
    # Read from stdin if no prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            console.print("[yellow]Usage: nemo chat 'your message here'[/yellow]")
            raise typer.Exit(1)

    asyncio.run(_chat(prompt, endpoint, think, output_format))


async def _chat(
    prompt: str,
    endpoint_name: str | None,
    show_thinking: bool,
    output_format: str | None = None,
) -> None:
    cfg = load_config()
    registry = Registry(cfg)
    ep_name = endpoint_name or cfg.default_endpoint
    provider = registry.get_chat_provider(ep_name)

    ep = registry.get_endpoint(ep_name)
    console.print(f"[dim]{ep.name} ({ep.model_id})[/dim]\n")

    messages = [
        Message(
            role=Role.SYSTEM, content="You are a helpful assistant powered by NVIDIA Nemotron 3."
        ),
        Message(role=Role.USER, content=prompt),
    ]

    # Build response_format from --output-format flag
    response_format: dict[str, str] | None = None
    if output_format and output_format.lower() == "json":
        response_format = {"type": "json_object"}

    text_buf = ""
    think_buf = ""
    last_usage = None

    async for chunk in provider.stream(messages, response_format=response_format):
        if chunk.thinking and show_thinking:
            think_buf += chunk.thinking
            console.print(f"[dim]{chunk.thinking}[/dim]", end="")
        if chunk.text:
            text_buf += chunk.text
            console.print(chunk.text, end="")
        if chunk.usage:
            last_usage = chunk.usage

    console.print()  # Final newline

    if last_usage:
        console.print(
            f"\n[dim]Tokens: {last_usage.get('prompt_tokens', 0)} prompt + "
            f"{last_usage.get('completion_tokens', 0)} completion = "
            f"{last_usage.get('total_tokens', 0)} total[/dim]"
        )
