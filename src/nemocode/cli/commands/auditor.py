# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo auditor — manage NeMo Auditor targets, configs, and jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.nvidia_client import AuditorClient

console = Console()
auditor_app = typer.Typer(help="Manage NeMo Auditor targets, configs, and jobs.")
target_app = typer.Typer(help="Manage audit targets.")
config_app = typer.Typer(help="Manage audit configs.")
job_app = typer.Typer(help="Manage audit jobs.")

auditor_app.add_typer(target_app, name="target")
auditor_app.add_typer(config_app, name="config")
auditor_app.add_typer(job_app, name="job")


def _client(base_url: str | None) -> AuditorClient:
    return AuditorClient(base_url=base_url)


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        console.print(f"[red]Spec file not found:[/red] {path}")
        raise typer.Exit(1) from exc
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {path}:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not isinstance(data, dict):
        console.print("[red]Spec must be a JSON object.[/red]")
        raise typer.Exit(1)
    return data


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _print_table(title: str, payload: Any, columns: list[tuple[str, str]]) -> None:
    rows = _rows(payload)
    if not rows:
        console.print(f"[dim]No {title.lower()} found.[/dim]")
        return

    table = Table(title=title)
    for header, _ in columns:
        table.add_column(header, style="cyan" if header == columns[0][0] else None)

    for row in rows:
        table.add_row(*[str(row.get(key, "")) for _, key in columns])
    console.print(table)


def _print_json(payload: Any) -> None:
    console.print(json.dumps(payload, indent=2))


@target_app.command("ls")
def target_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.list_targets(page=page, page_size=page_size)
    except Exception as exc:
        console.print(f"[red]List failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_table("Audit Targets", payload, [("ID", "id"), ("Name", "name"), ("Type", "type")])


@target_app.command("get")
def target_get(
    target_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.get_target(target_id)
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@target_app.command("create")
def target_create(
    spec: Path = typer.Argument(..., exists=True, dir_okay=False),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.create_target(_load_json_file(spec))
    except Exception as exc:
        console.print(f"[red]Create failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@target_app.command("delete")
def target_delete(
    target_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        client.delete_target(target_id)
    except Exception as exc:
        console.print(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Deleted target {target_id}[/green]")


@config_app.command("ls")
def config_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.list_configs(page=page, page_size=page_size)
    except Exception as exc:
        console.print(f"[red]List failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_table("Audit Configs", payload, [("ID", "id"), ("Name", "name"), ("Type", "type")])


@config_app.command("get")
def config_get(
    config_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.get_config(config_id)
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@config_app.command("create")
def config_create(
    spec: Path = typer.Argument(..., exists=True, dir_okay=False),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.create_config(_load_json_file(spec))
    except Exception as exc:
        console.print(f"[red]Create failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@config_app.command("delete")
def config_delete(
    config_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        client.delete_config(config_id)
    except Exception as exc:
        console.print(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Deleted config {config_id}[/green]")


@job_app.command("ls")
def job_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.list_jobs(page=page, page_size=page_size)
    except Exception as exc:
        console.print(f"[red]List failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_table("Audit Jobs", payload, [("ID", "id"), ("Status", "status"), ("Created", "created_at")])


@job_app.command("status")
def job_status(
    job_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.get_job_status(job_id)
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@job_app.command("create")
def job_create(
    spec: Path = typer.Argument(..., exists=True, dir_okay=False),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.create_job(_load_json_file(spec))
    except Exception as exc:
        console.print(f"[red]Create failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@job_app.command("logs")
def job_logs(
    job_id: str = typer.Argument(...),
    limit: int = typer.Option(100, "--limit", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        payload = client.get_job_logs(job_id, limit=limit)
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    entries = payload.get("data", [payload] if isinstance(payload, dict) else [])
    if not entries:
        console.print("[dim]No logs available yet.[/dim]")
        return
    for entry in entries:
        if isinstance(entry, dict):
            console.print(entry.get("message", json.dumps(entry)))
        else:
            console.print(str(entry))


@job_app.command("results")
def job_results(
    job_id: str = typer.Argument(...),
    download: bool = typer.Option(False, "--download"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    if download:
        try:
            data = client.download_report(job_id)
        except Exception as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        dest = output or Path(f".nemocode/data/audit-{job_id}-report.bin")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        console.print(f"[green]Downloaded audit report to {dest}[/green]")
        return

    try:
        payload = client.get_job_results(job_id)
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_json(payload)


@job_app.command("delete")
def job_delete(
    job_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_AUDITOR_BASE_URL"),
) -> None:
    client = _client(base_url)
    try:
        client.delete_job(job_id)
    except Exception as exc:
        console.print(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Deleted audit job {job_id}[/green]")
