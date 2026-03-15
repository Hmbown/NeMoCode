# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo endpoint — manage and test endpoints."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from nemocode.config import get_api_key, load_config
from nemocode.core.registry import Registry

console = Console()
endpoint_app = typer.Typer(help="Manage NIM endpoints.")


@endpoint_app.command("ls")
def endpoint_ls() -> None:
    """List all configured endpoints."""
    cfg = load_config()
    table = Table(title="Endpoints")
    table.add_column("Name", style="cyan")
    table.add_column("Tier", style="green")
    table.add_column("Model")
    table.add_column("Capabilities")
    table.add_column("Key", style="dim")

    for name, ep in cfg.endpoints.items():
        caps = ", ".join(c.value for c in ep.capabilities)
        has_key = "ok" if get_api_key(ep) else "missing"
        key_style = "green" if has_key == "ok" else "red"
        marker = " *" if name == cfg.default_endpoint else ""
        table.add_row(
            f"{name}{marker}",
            ep.tier.value,
            ep.model_id,
            caps,
            f"[{key_style}]{has_key}[/{key_style}]",
        )

    console.print(table)
    console.print("\n[dim]* = default endpoint[/dim]")


@endpoint_app.command("test")
def endpoint_test(
    name: str = typer.Argument(..., help="Endpoint name to test"),
) -> None:
    """Test connectivity to an endpoint."""
    asyncio.run(_test_endpoint(name))


async def _test_endpoint(name: str) -> None:
    cfg = load_config()
    registry = Registry(cfg)

    try:
        ep = registry.get_endpoint(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    api_key = get_api_key(ep)
    if ep.api_key_env and not api_key:
        console.print(f"[red]Missing API key: {ep.api_key_env}[/red]")
        console.print(f"[dim]Set it with: nemo auth set {ep.api_key_env}[/dim]")
        raise typer.Exit(1)

    console.print(f"Testing {name} ({ep.base_url})...")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            headers.update(ep.extra_headers)
            resp = await client.get(f"{ep.base_url.rstrip('/')}/models", headers=headers)
            if resp.status_code == 200:
                console.print(f"[green]Connected. Status: {resp.status_code}[/green]")
                data = resp.json()
                models = data.get("data", [])
                if models:
                    console.print(f"[dim]Models available: {len(models)}[/dim]")
            else:
                console.print(f"[yellow]Response: {resp.status_code}[/yellow]")
                console.print(f"[dim]{resp.text[:500]}[/dim]")
    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")


@endpoint_app.command("add")
def endpoint_add(
    name: str = typer.Argument(..., help="Name for the endpoint"),
    base_url: str = typer.Option(..., help="Base URL"),
    model_id: str = typer.Option(..., help="Model ID"),
    tier: str = typer.Option("local-nim", help="Endpoint tier"),
    api_key_env: str = typer.Option(None, help="Environment variable for API key"),
) -> None:
    """Add a custom endpoint to user config."""
    import yaml

    from nemocode.config import _USER_CONFIG_PATH, ensure_config_dir

    ensure_config_dir()
    existing = {}
    if _USER_CONFIG_PATH.exists():
        with open(_USER_CONFIG_PATH) as f:
            existing = yaml.safe_load(f) or {}

    if "endpoints" not in existing:
        existing["endpoints"] = {}

    existing["endpoints"][name] = {
        "name": name,
        "tier": tier,
        "base_url": base_url,
        "model_id": model_id,
        "capabilities": ["chat", "code"],
    }
    if api_key_env:
        existing["endpoints"][name]["api_key_env"] = api_key_env

    with open(_USER_CONFIG_PATH, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)

    console.print(f"[green]Added endpoint '{name}'[/green]")
