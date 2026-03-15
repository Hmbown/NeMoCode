# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo session — conversation persistence."""

from __future__ import annotations

import json
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.persistence import _SESSION_DIR

console = Console()
session_app = typer.Typer(help="Manage conversation sessions.")


@session_app.command("ls")
def session_ls(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show"),
) -> None:
    """List past sessions."""
    if not _SESSION_DIR.exists():
        console.print("[dim]No sessions found.[/dim]")
        return

    sessions = sorted(
        _SESSION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Date")
    table.add_column("Endpoint")
    table.add_column("Messages")
    table.add_column("Tokens")

    for sp in sessions[:limit]:
        try:
            data = json.loads(sp.read_text())
            sid = sp.stem
            date = datetime.fromtimestamp(data.get("created_at", 0)).strftime("%Y-%m-%d %H:%M")
            endpoint = data.get("endpoint_name", "")
            msg_count = data.get("message_count", 0)
            tokens = data.get("usage", {}).get("total_tokens", 0)
            table.add_row(sid, date, endpoint, str(msg_count), f"{tokens:,}")
        except Exception:
            continue

    console.print(table)


@session_app.command("export")
def session_export(
    session_id: str = typer.Argument(..., help="Session ID"),
    fmt: str = typer.Option("md", "--format", "-f", help="Export format: md or json"),
) -> None:
    """Export a session."""
    session_path = _SESSION_DIR / f"{session_id}.json"
    if not session_path.exists():
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    data = json.loads(session_path.read_text())

    if fmt == "json":
        console.print_json(json.dumps(data, indent=2))
    else:
        console.print(f"# Session {session_id}\n")
        for msg in data.get("messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            console.print(f"## {role.upper()}\n")
            console.print(f"{content}\n")
