# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo audit — view the audit trail."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.audit import get_audit_log

console = Console()
audit_app = typer.Typer(help="View the NeMoCode audit trail.")


@audit_app.callback(invoke_without_command=True)
def audit(
    ctx: typer.Context,
    since: str | None = typer.Option(
        None,
        "--since",
        help="Filter entries newer than this duration (e.g. 1h, 24h, 7d).",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum number of entries to display.",
    ),
    entry_type: str | None = typer.Option(
        None,
        "--type",
        help="Filter by entry type: tool, file, or api.",
    ),
) -> None:
    """Show recent audit entries in a formatted table."""
    if ctx.invoked_subcommand is not None:
        return

    if entry_type and entry_type not in ("tool", "file", "api"):
        console.print(f"[red]Invalid type: {entry_type!r}. Must be one of: tool, file, api[/red]")
        raise typer.Exit(code=1)

    log = get_audit_log()
    entries = log.get_entries(since=since, limit=limit, entry_type=entry_type)

    if not entries:
        console.print("[dim]No audit entries found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", style="dim", width=22)
    table.add_column("Type", width=6)
    table.add_column("Details")
    table.add_column("Session", style="cyan", width=12)

    for entry in entries:
        ts = entry.get("ts", "")
        etype = entry.get("type", "")
        session = entry.get("session", "")

        if etype == "tool":
            tool = entry.get("tool", "")
            err = " [red](error)[/red]" if entry.get("is_error") else ""
            details = f"{tool}{err}"
        elif etype == "file":
            op = entry.get("operation", "")
            path = entry.get("path", "")
            details = f"{op} {path}"
        elif etype == "api":
            endpoint = entry.get("endpoint", "")
            model = entry.get("model", "")
            tokens = entry.get("tokens_used", 0)
            details = f"{endpoint} ({model}) — {tokens} tokens"
        else:
            details = str(entry)

        table.add_row(ts, etype, details, session)

    console.print(table)
    console.print(f"\n[dim]Showing {len(entries)} entry/entries.[/dim]")


if __name__ == "__main__":
    audit_app()
