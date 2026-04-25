# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NeMoCode CLI — main entry point.

Commands: nemo chat | code | customize | data | embed | rerank | speech | serve | endpoint | model | formation | auth | hardware | init | session
"""

from __future__ import annotations

import os
from pathlib import Path

import typer


def _load_dotenv() -> None:
    """Load .env from the nearest project root (cwd up to git root)."""
    start = Path.cwd()
    for parent in [start, *start.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            try:
                for raw in candidate.read_text().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[len("export "):].strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            except OSError:
                pass
            return
        if (parent / ".git").is_dir() or parent == Path.home():
            return


_load_dotenv()


app = typer.Typer(
    name="nemo",
    help="NeMoCode — Terminal-first agentic coding CLI for NVIDIA NIM (DeepSeek + Nemotron + frontier models).",
    no_args_is_help=False,
    add_completion=True,
)


def _register_commands() -> None:
    from nemocode.cli.commands.agent import agent_app
    from nemocode.cli.commands.auditor import auditor_app
    from nemocode.cli.commands.auth import auth_app
    from nemocode.cli.commands.chat import chat_cmd
    from nemocode.cli.commands.code import code_cmd
    from nemocode.cli.commands.config import config_app
    from nemocode.cli.commands.customize import customize_app
    from nemocode.cli.commands.data import data_app
    from nemocode.cli.commands.doctor import doctor_app
    from nemocode.cli.commands.embed import embed_app
    from nemocode.cli.commands.endpoint import endpoint_app
    from nemocode.cli.commands.entity import entity_app
    from nemocode.cli.commands.evaluator import evaluator_app
    from nemocode.cli.commands.formation import formation_app
    from nemocode.cli.commands.guardrails import guardrails_app
    from nemocode.cli.commands.hardware import hardware_app
    from nemocode.cli.commands.init_cmd import init_cmd
    from nemocode.cli.commands.model import model_app
    from nemocode.cli.commands.obs import obs_app
    from nemocode.cli.commands.rerank import rerank_app
    from nemocode.cli.commands.safe_synth import safe_synth_app
    from nemocode.cli.commands.session import session_app
    from nemocode.cli.commands.setup import setup_app
    from nemocode.cli.commands.serve import serve_app
    from nemocode.cli.commands.speech import speech_app

    app.command("chat")(chat_cmd)
    app.command("code")(code_cmd)
    app.add_typer(agent_app, name="agent")
    app.add_typer(auditor_app, name="auditor")
    app.add_typer(customize_app, name="customize")
    app.add_typer(data_app, name="data")
    app.add_typer(embed_app, name="embed")
    app.add_typer(entity_app, name="entity")
    app.add_typer(evaluator_app, name="evaluator")
    app.add_typer(guardrails_app, name="guardrails")
    app.add_typer(rerank_app, name="rerank")
    app.add_typer(safe_synth_app, name="safe-synth")
    app.add_typer(speech_app, name="speech")
    app.add_typer(serve_app, name="serve")
    app.add_typer(endpoint_app, name="endpoint")
    app.add_typer(model_app, name="model")
    app.add_typer(formation_app, name="formation")
    app.add_typer(auth_app, name="auth")
    app.add_typer(hardware_app, name="hardware")
    app.add_typer(doctor_app, name="doctor")
    app.add_typer(config_app, name="config")
    app.command("init")(init_cmd)
    app.add_typer(session_app, name="session")
    app.add_typer(obs_app, name="obs")
    app.add_typer(setup_app, name="setup")


def _version_callback(value: bool) -> None:
    if value:
        from nemocode import __version__

        typer.echo(f"NeMoCode v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Launch NeMoCode. Without a subcommand, starts the interactive REPL."""
    if ctx.invoked_subcommand is None:
        from nemocode.core.first_run import check_and_run_first_run

        check_and_run_first_run()

        # Launch the REPL (same as `nemo code` with no args)
        from nemocode.cli.commands.repl import start_repl

        start_repl(endpoint=None, formation=None, agent_name=None, think=True, yes=False)


_register_commands()

if __name__ == "__main__":
    app()
