# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo entity — manage NeMo Entity Store objects."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from nemocode.core.nvidia_client import EntityStoreClient

console = Console()
entity_app = typer.Typer(help="Manage NeMo Entity Store namespaces, projects, datasets, and models.")
namespace_app = typer.Typer(help="Manage namespaces.")
project_app = typer.Typer(help="Manage projects.")
dataset_app = typer.Typer(help="Manage datasets.")
model_app = typer.Typer(help="Manage registered models.")

entity_app.add_typer(namespace_app, name="namespace")
entity_app.add_typer(project_app, name="project")
entity_app.add_typer(dataset_app, name="dataset")
entity_app.add_typer(model_app, name="model")


def _client(base_url: str | None) -> EntityStoreClient:
    return EntityStoreClient(base_url=base_url)


def _parse_json_arg(value: str | None, label: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON for {label}:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not isinstance(data, dict):
        console.print(f"[red]{label} must decode to a JSON object.[/red]")
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


def _create_resource(
    resource: str,
    base_url: str | None,
    spec: dict[str, Any],
) -> None:
    client = _client(base_url)
    try:
        created = client.create_resource(resource, spec)
    except Exception as exc:
        console.print(f"[red]Create failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(json.dumps(created, indent=2))


def _list_resource(
    resource: str,
    base_url: str | None,
    page: int,
    page_size: int,
    title: str,
    columns: list[tuple[str, str]],
) -> None:
    client = _client(base_url)
    try:
        payload = client.list_resources(resource, page=page, page_size=page_size)
    except Exception as exc:
        console.print(f"[red]List failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_table(title, payload, columns)


def _get_resource(resource: str, resource_id: str, base_url: str | None) -> None:
    client = _client(base_url)
    try:
        payload = client.get_resource(resource, resource_id)
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(json.dumps(payload, indent=2))


def _delete_resource(resource: str, resource_id: str, base_url: str | None) -> None:
    client = _client(base_url)
    try:
        client.delete_resource(resource, resource_id)
    except Exception as exc:
        console.print(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Deleted {resource} {resource_id}[/green]")


@namespace_app.command("ls")
def namespace_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _list_resource("namespace", base_url, page, page_size, "Namespaces", [("Name", "name"), ("Description", "description")])


@namespace_app.command("create")
def namespace_create(
    name: str = typer.Argument(...),
    description: str = typer.Option("", "--description"),
    metadata: str | None = typer.Option(None, "--metadata", help="JSON object string."),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    spec: dict[str, Any] = {"name": name}
    if description:
        spec["description"] = description
    meta = _parse_json_arg(metadata, "metadata")
    if meta:
        spec["metadata"] = meta
    _create_resource("namespace", base_url, spec)


@namespace_app.command("get")
def namespace_get(
    namespace_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _get_resource("namespace", namespace_id, base_url)


@namespace_app.command("delete")
def namespace_delete(
    namespace_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _delete_resource("namespace", namespace_id, base_url)


@project_app.command("ls")
def project_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _list_resource("project", base_url, page, page_size, "Projects", [("Name", "name"), ("Namespace", "namespace"), ("Description", "description")])


@project_app.command("create")
def project_create(
    name: str = typer.Argument(...),
    namespace: str = typer.Option("default", "--namespace"),
    description: str = typer.Option("", "--description"),
    metadata: str | None = typer.Option(None, "--metadata", help="JSON object string."),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    spec: dict[str, Any] = {"name": name, "namespace": namespace}
    if description:
        spec["description"] = description
    meta = _parse_json_arg(metadata, "metadata")
    if meta:
        spec["metadata"] = meta
    _create_resource("project", base_url, spec)


@project_app.command("get")
def project_get(
    project_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _get_resource("project", project_id, base_url)


@project_app.command("delete")
def project_delete(
    project_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _delete_resource("project", project_id, base_url)


@dataset_app.command("ls")
def dataset_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _list_resource("dataset", base_url, page, page_size, "Datasets", [("Name", "name"), ("Namespace", "namespace"), ("Project", "project")])


@dataset_app.command("create")
def dataset_create(
    name: str = typer.Argument(...),
    namespace: str = typer.Option("default", "--namespace"),
    project: str | None = typer.Option(None, "--project"),
    description: str = typer.Option("", "--description"),
    uri: str | None = typer.Option(None, "--uri"),
    metadata: str | None = typer.Option(None, "--metadata", help="JSON object string."),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    spec: dict[str, Any] = {"name": name, "namespace": namespace}
    if project:
        spec["project"] = project
    if description:
        spec["description"] = description
    if uri:
        spec["uri"] = uri
    meta = _parse_json_arg(metadata, "metadata")
    if meta:
        spec["metadata"] = meta
    _create_resource("dataset", base_url, spec)


@dataset_app.command("get")
def dataset_get(
    dataset_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _get_resource("dataset", dataset_id, base_url)


@dataset_app.command("delete")
def dataset_delete(
    dataset_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _delete_resource("dataset", dataset_id, base_url)


@model_app.command("ls")
def model_ls(
    page: int = typer.Option(1, "--page", min=1),
    page_size: int = typer.Option(20, "--page-size", min=1),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _list_resource("model", base_url, page, page_size, "Models", [("Name", "name"), ("Namespace", "namespace"), ("Version", "version")])


@model_app.command("create")
def model_create(
    name: str = typer.Argument(...),
    namespace: str = typer.Option("default", "--namespace"),
    version: str | None = typer.Option(None, "--version"),
    uri: str | None = typer.Option(None, "--uri"),
    description: str = typer.Option("", "--description"),
    metadata: str | None = typer.Option(None, "--metadata", help="JSON object string."),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    spec: dict[str, Any] = {"name": name, "namespace": namespace}
    if version:
        spec["version"] = version
    if uri:
        spec["uri"] = uri
    if description:
        spec["description"] = description
    meta = _parse_json_arg(metadata, "metadata")
    if meta:
        spec["metadata"] = meta
    _create_resource("model", base_url, spec)


@model_app.command("get")
def model_get(
    model_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _get_resource("model", model_id, base_url)


@model_app.command("delete")
def model_delete(
    model_id: str = typer.Argument(...),
    base_url: str | None = typer.Option(None, "--base-url", envvar="NEMOCODE_ENTITY_BASE_URL"),
) -> None:
    _delete_resource("model", model_id, base_url)
