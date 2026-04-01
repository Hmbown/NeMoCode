# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""nemo speech — ASR transcription and TTS synthesis via NVIDIA NIM endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from nemocode.config import load_config
from nemocode.config.schema import Endpoint

console = Console()
speech_app = typer.Typer(help="Speech recognition (ASR) and text-to-speech (TTS).")


def _get_speech_endpoint(cfg, endpoint_name: str) -> Endpoint:
    """Resolve a speech endpoint from config."""
    if endpoint_name not in cfg.endpoints:
        raise typer.BadParameter(
            f"Unknown endpoint: {endpoint_name}. Available speech endpoints: nim-asr, nim-tts"
        )
    return cfg.endpoints[endpoint_name]


@speech_app.command("transcribe")
def speech_transcribe(
    audio_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Audio file to transcribe (WAV, FLAC, MP3, etc.).",
    ),
    endpoint: str = typer.Option(
        "nim-asr",
        "--endpoint",
        "-e",
        help="ASR endpoint to use.",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        "-l",
        help="ISO 639-1 language code.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write transcription to a text file.",
    ),
) -> None:
    """Transcribe an audio file to text using NVIDIA Parakeet ASR.

    Examples:
        nemo speech transcribe recording.wav
        nemo speech transcribe meeting.mp3 --language en --output transcript.txt
    """
    from nemocode.config import get_api_key
    from nemocode.providers.nim_speech import NIMASRProvider

    cfg = load_config()
    ep = _get_speech_endpoint(cfg, endpoint)
    api_key = get_api_key(ep)
    provider = NIMASRProvider(endpoint=ep, api_key=api_key)

    console.print(f"[dim]Transcribing {audio_file.name} via {endpoint}...[/dim]")

    try:
        result = asyncio.run(provider.transcribe(audio_file, language=language))
    except Exception as exc:
        console.print(f"[red]Transcription failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    text = result.get("text", "")
    if not text:
        console.print("[yellow]No transcription returned.[/yellow]")
        raise typer.Exit(0)

    console.print(
        Panel(
            text,
            title="[bold]Transcription[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        console.print(f"[green]Wrote transcription to {output}[/green]")

    console.print(f"\n[dim]Endpoint: {endpoint} | Language: {language}[/dim]")


@speech_app.command("synthesize")
def speech_synthesize(
    text: str = typer.Argument(..., help="Text to synthesize into speech."),
    output: Path = typer.Option(
        Path("output.wav"),
        "--output",
        "-o",
        help="Output audio file path.",
    ),
    endpoint: str = typer.Option(
        "nim-tts",
        "--endpoint",
        "-e",
        help="TTS endpoint to use.",
    ),
    voice: str = typer.Option(
        "English-US.Female-1",
        "--voice",
        "-v",
        help="Voice identifier.",
    ),
    audio_format: str = typer.Option(
        "wav",
        "--format",
        "-f",
        help="Audio format: wav, mp3, opus, flac.",
    ),
    speed: float = typer.Option(
        1.0,
        "--speed",
        "-s",
        help="Playback speed (0.25–4.0).",
    ),
) -> None:
    """Synthesize text to speech using NVIDIA FastPitch TTS.

    Examples:
        nemo speech synthesize "Hello, world!"
        nemo speech synthesize "Testing NeMoCode" -o test.wav --voice English-US.Male-1
    """
    from nemocode.config import get_api_key
    from nemocode.providers.nim_speech import NIMTTSProvider

    cfg = load_config()
    ep = _get_speech_endpoint(cfg, endpoint)
    api_key = get_api_key(ep)
    provider = NIMTTSProvider(endpoint=ep, api_key=api_key)

    console.print(f"[dim]Synthesizing speech via {endpoint}...[/dim]")

    try:
        audio_bytes = asyncio.run(
            provider.synthesize(
                text,
                output_path=output,
                voice=voice,
                response_format=audio_format,
                speed=speed,
            )
        )
    except Exception as exc:
        console.print(f"[red]Synthesis failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    size_kb = len(audio_bytes) / 1024
    console.print(
        Panel(
            f"[bold]Text:[/bold] {text}\n"
            f"[bold]Voice:[/bold] {voice}\n"
            f"[bold]Format:[/bold] {audio_format}\n"
            f"[bold]Speed:[/bold] {speed}x\n"
            f"[bold]Size:[/bold] {size_kb:.1f} KB\n"
            f"[bold]Output:[/bold] {output}",
            title="[bold]Speech Synthesized[/bold]",
            border_style="bright_green",
            expand=False,
        )
    )


@speech_app.command("test")
def speech_test(
    endpoint: str = typer.Option(
        "nim-tts",
        "--endpoint",
        "-e",
        help="Speech endpoint to test (nim-asr or nim-tts).",
    ),
) -> None:
    """Quick connectivity test for a speech endpoint."""
    import httpx

    cfg = load_config()
    ep = _get_speech_endpoint(cfg, endpoint)

    console.print(f"Testing {endpoint} at {ep.base_url}...")

    try:
        resp = httpx.get(f"{ep.base_url}/models", timeout=10)
        if resp.status_code == 200:
            console.print(f"[green]Connected to {endpoint}[/green]")
        else:
            console.print(f"[yellow]Got HTTP {resp.status_code} from {endpoint}[/yellow]")
    except Exception as exc:
        console.print(f"[red]Cannot reach {endpoint}:[/red] {exc}")
        raise typer.Exit(1) from exc
