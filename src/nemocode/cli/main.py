# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NeMoCode CLI — main entry point.

Commands: nemo chat | code | endpoint | model | formation | auth | hardware | init | session
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="nemo",
    help="NeMoCode — Terminal-first agentic coding CLI for NVIDIA Nemotron 3.",
    no_args_is_help=False,
    add_completion=True,
)


def _register_commands() -> None:
    from nemocode.cli.commands.auth import auth_app
    from nemocode.cli.commands.chat import chat_cmd
    from nemocode.cli.commands.code import code_cmd
    from nemocode.cli.commands.endpoint import endpoint_app
    from nemocode.cli.commands.formation import formation_app
    from nemocode.cli.commands.hardware import hardware_app
    from nemocode.cli.commands.init_cmd import init_cmd
    from nemocode.cli.commands.model import model_app
    from nemocode.cli.commands.obs import obs_app
    from nemocode.cli.commands.session import session_app
    from nemocode.cli.commands.setup import setup_app

    app.command("chat")(chat_cmd)
    app.command("code")(code_cmd)
    app.add_typer(endpoint_app, name="endpoint")
    app.add_typer(model_app, name="model")
    app.add_typer(formation_app, name="formation")
    app.add_typer(auth_app, name="auth")
    app.add_typer(hardware_app, name="hardware")
    app.command("init")(init_cmd)
    app.add_typer(session_app, name="session")
    app.add_typer(obs_app, name="obs")
    app.add_typer(setup_app, name="setup")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Launch NeMoCode. Without a subcommand, starts the interactive TUI."""
    if ctx.invoked_subcommand is None:
        from nemocode.core.first_run import check_and_run_first_run

        check_and_run_first_run()
        _launch_tui()


def _launch_tui() -> None:
    """Launch the interactive TUI."""
    try:
        from nemocode.tui.app import NeMoCodeApp

        tui_app = NeMoCodeApp()
        tui_app.run()
    except ImportError as e:
        from rich.console import Console

        console = Console()
        console.print(f"[red]TUI requires textual: {e}[/red]")
        console.print("Install with: pip install 'nemocode[dev]'")
        raise typer.Exit(1)


_register_commands()

if __name__ == "__main__":
    app()
