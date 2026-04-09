from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect


class DryEffect(AudioEffect):
    """Passthrough — returns the original audio unchanged.

    Use in branch recipes to mix the original input with processed branches.
    """

    name = "dry"
    description = "Passthrough — include original audio in branch mixes"

    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        return audio

    def register_command(self, app: typer.Typer) -> None:
        @app.command(name=self.name, help=self.description)
        def dry_command(
            input_file: Annotated[Path, typer.Argument(help="Input audio file")],
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file path")
            ] = Path("output.wav"),
        ) -> None:
            audio = load_audio(input_file)
            save_audio(audio, output)

        return dry_command
