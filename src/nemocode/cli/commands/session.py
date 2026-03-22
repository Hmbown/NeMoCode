# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo session — conversation persistence."""

from __future__ import annotations

import json
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.persistence import list_sessions, load_session

console = Console()
session_app = typer.Typer(help="Manage conversation sessions.")


@session_app.command("ls")
def session_ls(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show"),
) -> None:
    """List past sessions."""
    sessions = list_sessions(limit=limit)
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Date")
    table.add_column("Endpoint")
    table.add_column("Messages")
    table.add_column("Tokens")
    table.add_column("Branch", style="dim")
    table.add_column("CWD", style="dim")

    for s in sessions:
        sid = s.get("id", "")
        ts = s.get("updated_at", s.get("created_at", 0))
        date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
        endpoint = s.get("endpoint_name", "")
        msg_count = s.get("message_count", 0)
        tokens = s.get("total_tokens", 0)
        meta = s.get("metadata") or {}
        branch = meta.get("git_branch", "")
        cwd = meta.get("cwd", "")
        if cwd and len(cwd) > 40:
            cwd = "~" + cwd[-39:]
        table.add_row(sid, date, endpoint, str(msg_count), f"{tokens:,}", branch, cwd)

    console.print(table)


@session_app.command("export")
def session_export(
    session_id: str = typer.Argument(..., help="Session ID"),
    fmt: str = typer.Option("md", "--format", "-f", help="Export format: md or json"),
) -> None:
    """Export a session."""
    session = load_session(session_id)
    if session is None:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    if fmt == "json":
        data = {
            "id": session.id,
            "endpoint_name": session.endpoint_name,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [{"role": m.role.value, "content": m.content} for m in session.messages],
        }
        console.print_json(json.dumps(data, indent=2))
    else:
        console.print(f"# Session {session_id}\n")
        for msg in session.messages:
            console.print(f"## {msg.role.value.upper()}\n")
            console.print(f"{msg.content}\n")
