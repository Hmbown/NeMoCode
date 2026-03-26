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
from nemocode.core.structured_output import (
    StructuredOutputError,
    build_json_schema_response_format,
    check_structured_output_support,
)

console = Console()


def chat_cmd(
    prompt: str = typer.Argument(None, help="Chat message (or pipe via stdin)"),
    endpoint: str = typer.Option(None, "-e", "--endpoint", help="Endpoint to use"),
    think: bool = typer.Option(True, "--think/--no-think", help="Show/hide thinking trace"),
    output_format: str = typer.Option(None, "--output-format", help="Output format: json, text"),
    json_schema: str = typer.Option(
        None, "--json-schema", help="JSON schema file or inline JSON for structured output"
    ),
    guardrails: bool | None = typer.Option(
        None, "--guardrails/--no-guardrails", help="Enable/disable content safety"
    ),
) -> None:
    """Simple streaming chat with a Nemotron model. No tools."""
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            console.print("[yellow]Usage: nemo chat 'your message here'[/yellow]")
            raise typer.Exit(1)

    if json_schema and output_format:
        console.print("[red]Cannot use --json-schema and --output-format together.[/red]")
        raise typer.Exit(1)

    asyncio.run(_chat(prompt, endpoint, think, output_format, json_schema, guardrails))


async def _chat(
    prompt: str,
    endpoint_name: str | None,
    show_thinking: bool,
    output_format: str | None = None,
    json_schema_input: str | None = None,
    guardrails_flag: bool | None = None,
) -> None:
    cfg = load_config()
    registry = Registry(cfg)
    ep_name = endpoint_name or cfg.default_endpoint
    provider = registry.get_chat_provider(ep_name)

    ep = registry.get_endpoint(ep_name)
    manifest = registry.get_manifest(ep.model_id)
    console.print(f"[dim]{ep.name} ({ep.model_id})[/dim]\n")

    use_guardrails = guardrails_flag if guardrails_flag is not None else cfg.guardrails.enabled

    messages = [
        Message(
            role=Role.SYSTEM, content="You are a helpful assistant powered by NVIDIA Nemotron 3."
        ),
        Message(role=Role.USER, content=prompt),
    ]

    response_format: dict[str, Any] | None = None
    if json_schema_input:
        try:
            response_format = build_json_schema_response_format(json_schema_input)
        except StructuredOutputError as exc:
            console.print(f"[red]{exc}[/red]")
            return
    elif output_format and output_format.lower() == "json":
        response_format = {"type": "json_object"}

    # Capability gating: warn if the model doesn't support the requested mode
    if response_format:
        warning = check_structured_output_support(manifest, response_format)
        if warning:
            console.print(f"[yellow]Warning: {warning}[/yellow]")

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

    console.print()

    if use_guardrails and text_buf.strip():
        from nemocode.config import get_api_key
        from nemocode.providers.nim_guardrails import NIMGuardrailsProvider

        gr_ep_name = cfg.guardrails.endpoint
        gr_ep = cfg.endpoints.get(gr_ep_name)
        if gr_ep:
            gr_provider = NIMGuardrailsProvider(
                endpoint=gr_ep,
                config=cfg.guardrails,
                api_key=get_api_key(gr_ep),
            )
            result = await gr_provider.check(text_buf)
            if not result.safe:
                console.print(
                    f"\n[red]Content safety: blocked categories: "
                    f"{', '.join(result.blocked_categories)}[/red]"
                )
            elif result.error:
                console.print(f"\n[yellow]Guardrails warning: {result.error}[/yellow]")
        else:
            console.print(
                f"\n[yellow]Guardrails endpoint '{gr_ep_name}' not found, skipping.[/yellow]"
            )

    if last_usage:
        console.print(
            f"\n[dim]Tokens: {last_usage.get('prompt_tokens', 0)} prompt + "
            f"{last_usage.get('completion_tokens', 0)} completion = "
            f"{last_usage.get('total_tokens', 0)} total[/dim]"
        )
