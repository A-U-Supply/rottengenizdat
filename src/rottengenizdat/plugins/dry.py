from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect


_DRY_HELP = """\
Passthrough — returns the original audio unchanged.

Use 'dry' as a step in branch chains and recipes to mix the unprocessed
original with RAVE-mangled branches. Control the blend using the weight
parameter in recipes.

In a recipe TOML (under a steps entry):
  effect = "dry"
  weight = 0.7        # 70% original

In a chain command:
  rotten chain input.wav "dry" "rave -m vintage -t 1.3" --branch -o out.wav

On its own (mostly useful for testing):
  rotten dry input.wav -o copy.wav
"""


class DryEffect(AudioEffect):
    """Passthrough — returns the original audio unchanged.

    Use in branch recipes to mix the original input with processed branches.
    """

    name = "dry"
    description = _DRY_HELP

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
