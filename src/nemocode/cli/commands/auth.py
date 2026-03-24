# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo auth — credential management."""

from __future__ import annotations

import asyncio
import getpass

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.credentials import (
    KNOWN_KEYS,
    get_credential,
    list_credentials,
    set_credential,
    storage_backend,
    test_credential,
)

console = Console()
auth_app = typer.Typer(help="Manage API credentials.")


@auth_app.command("setup")
def auth_setup() -> None:
    """Interactive guided credential setup."""
    console.print("[bold]NeMoCode Credential Setup[/bold]\n")
    console.print(f"Storage backend: [cyan]{storage_backend()}[/cyan]\n")

    console.print("Configure your API keys. Press Enter to skip any key.\n")

    for key_name, description in KNOWN_KEYS.items():
        existing = get_credential(key_name)
        if existing:
            console.print(f"  {key_name}: [green]already configured[/green]")
            update = console.input("  Update? [y/N]: ")
            if update.lower() not in ("y", "yes"):
                continue

        console.print(f"\n  [bold]{description}[/bold]")
        if key_name == "NVIDIA_API_KEY":
            console.print("  [dim]Get a free key at https://build.nvidia.com[/dim]")
        elif key_name == "NGC_CLI_API_KEY":
            console.print("  [dim]Generate at https://ngc.nvidia.com/setup/api-key[/dim]")
            console.print(
                "  [dim]Required for: Docker image pulls, Data Designer, "
                "Evaluator, Customizer[/dim]"
            )

        value = getpass.getpass(f"  {key_name}: ")
        if not value.strip():
            console.print("  [dim]Skipped[/dim]")
            continue

        success, method = set_credential(key_name, value.strip())
        if success:
            console.print(f"  [green]Stored in {method}[/green]")
        else:
            console.print(
                "  [yellow]Could not store in keyring. Set as environment variable:[/yellow]"
            )
            console.print(f"  [dim]export {key_name}='<your-key-here>'[/dim]")

    console.print("\n[green]Setup complete.[/green]")
    console.print(
        "[dim]Run 'nemo auth show' to verify or 'nemo auth test' to test connectivity.[/dim]"
    )


@auth_app.command("set")
def auth_set(
    key_name: str = typer.Argument(..., help="Credential key name (e.g. NVIDIA_API_KEY)"),
) -> None:
    """Set a specific API key (prompts securely)."""
    if key_name not in KNOWN_KEYS:
        console.print(f"[yellow]Unknown key: {key_name}[/yellow]")
        console.print(f"[dim]Known keys: {', '.join(KNOWN_KEYS.keys())}[/dim]")
        # Allow setting anyway
        console.print("[dim]Proceeding anyway...[/dim]")

    value = getpass.getpass(f"{key_name}: ")
    if not value.strip():
        console.print("[yellow]No value provided.[/yellow]")
        raise typer.Exit(1)

    success, method = set_credential(key_name, value.strip())
    if success:
        console.print(f"[green]Stored {key_name} in {method}[/green]")
    else:
        console.print("[yellow]Keyring unavailable. Set as environment variable:[/yellow]")
        console.print(f"[dim]export {key_name}='<your-key-here>'[/dim]")


@auth_app.command("show")
def auth_show() -> None:
    """Show which credentials are configured (values masked)."""
    console.print(f"[dim]Storage backend: {storage_backend()}[/dim]\n")

    creds = list_credentials()
    table = Table(title="Credentials")
    table.add_column("Key", style="cyan")
    table.add_column("Description")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Value")

    for key_name, info in creds.items():
        if info["configured"]:
            status = "[green]configured[/green]"
            source = info["source"]
            value = info["masked_value"] or ""
        else:
            status = "[red]missing[/red]"
            source = "-"
            value = "-"
        table.add_row(key_name, info["description"], status, source, value)

    console.print(table)


@auth_app.command("test")
def auth_test() -> None:
    """Test all configured credentials with API calls."""
    asyncio.run(_test_all())


async def _test_all() -> None:
    console.print("[bold]Testing credentials...[/bold]\n")

    for key_name in KNOWN_KEYS:
        value = get_credential(key_name)
        if not value:
            console.print(f"  {key_name}: [dim]not configured, skipping[/dim]")
            continue

        console.print(f"  {key_name}: ", end="")
        result = await test_credential(key_name)
        status = result["status"]
        if status == "ok":
            console.print("[green]OK[/green]")
        elif status == "invalid":
            console.print(f"[red]Invalid — {result.get('message', '')}[/red]")
        elif status == "error":
            console.print(f"[yellow]Error — {result.get('message', '')}[/yellow]")
        else:
            console.print(f"[dim]{status}[/dim]")
