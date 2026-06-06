"""AudioGen plugin — text-to-sound generation (Meta)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rottengenizdat.core import AudioBuffer, save_audio
from rottengenizdat.plugin import AudioEffect

console = Console()

_AUDIOGEN_HELP = """\
AudioGen — text-to-sound generation (Meta).

Describe a sound in words and AudioGen generates it. Unlike MusicGen
(which generates music), AudioGen focuses on environmental sounds,
sound effects, and non-musical audio: thunder, footsteps, animal
sounds, machinery, ambient textures.

KNOBS:

  -p, --prompt TEXT    Text description of the sound to generate.
                       "thunder and rain", "footsteps on gravel",
                       "church bells in the distance", "old engine"
  --duration SEC       Output duration in seconds. Default 4.
  -t, --temperature    Sampling temperature. 0.8 = focused, 1.0 = default,
                       1.5 = wild, 2.0 = chaotic.
  --top-k K            Top-k sampling. 0 = off (default), 250 = focused.
  --top-p P            Nucleus sampling. 0.0 = off (default), 0.9 = typical.
  --seed N             Random seed for reproducibility.

MODEL downloads from HuggingFace on first use (~1.5GB). Cache at
~/.cache/huggingface/.

EXAMPLES:

  Basic generation:
    rotten audiogen -p "thunder and rain" --duration 8 -o storm.wav

  Multiple sounds at once:
    rotten audiogen -p "footsteps on gravel, distant church bells" --duration 6 -o scene.wav

  Hot and chaotic:
    rotten audiogen -p "industrial machinery malfunction" -t 1.8 --duration 5 -o chaos.wav
"""


def _load_model():
    """Load AudioGen model from HuggingFace."""
    from audiocraft.models import AudioGen

    return AudioGen.get_pretrained("facebook/audiogen-medium")


class AudioGenEffect(AudioEffect):
    """AudioGen — text-to-sound generation."""

    name = "audiogen"
    description = _AUDIOGEN_HELP

    def process(
        self,
        prompt: str = "thunder and rain",
        duration: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> AudioBuffer:
        """Generate sound from a text prompt.

        Args:
            prompt: Text description of the sound.
            duration: Output duration in seconds.
            temperature: Sampling temperature.
            top_k: Top-k sampling limit (0 = off).
            top_p: Nucleus sampling (0.0 = off).
            seed: Random seed.
        """
        model = _load_model()

        if seed is not None:
            torch.manual_seed(seed)

        params: dict = {
            "duration": duration,
            "temperature": temperature,
        }
        if top_k > 0:
            params["top_k"] = top_k
        if top_p > 0.0:
            params["top_p"] = top_p
        model.set_generation_params(**params)

        out = model.generate([prompt])

        # out shape: (batch, channels, samples) at 32kHz
        samples = out[0]
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)

        return AudioBuffer(samples=samples.float(), sample_rate=model.sample_rate)

    def register_command(self, app: typer.Typer) -> None:
        """Register the audiogen subcommand."""

        @app.command(name=self.name, help=self.description)
        def audiogen_command(
            output: Annotated[
                Path,
                typer.Option("--output", "-o", help="Output file path"),
            ] = Path("output.wav"),
            prompt: Annotated[
                str,
                typer.Option("--prompt", "-p", help="Text description of the sound to generate"),
            ] = "thunder and rain",
            duration: Annotated[
                float,
                typer.Option("--duration", help="Output duration in seconds"),
            ] = 4.0,
            temperature: Annotated[
                float,
                typer.Option(
                    "--temperature",
                    "-t",
                    help="Sampling temperature. 0.8 = focused, 1.0 = default, 1.5 = wild",
                ),
            ] = 1.0,
            top_k: Annotated[
                int,
                typer.Option("--top-k", help="Top-k sampling. 0 = off, 250 = focused"),
            ] = 0,
            top_p: Annotated[
                float,
                typer.Option("--top-p", help="Nucleus sampling. 0.0 = off, 0.9 = typical"),
            ] = 0.0,
            seed: Annotated[
                Optional[int],
                typer.Option("--seed", help="Random seed for reproducibility"),
            ] = None,
        ) -> None:
            gen_info = f"prompt='{prompt}' duration={duration}s temp={temperature}"
            if top_k > 0:
                gen_info += f" top_k={top_k}"
            if top_p > 0:
                gen_info += f" top_p={top_p}"
            console.print(f"[bold]Generating:[/bold] {gen_info}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Generating (this takes a while on CPU)...", total=None)
                result = self.process(
                    prompt=prompt,
                    duration=duration,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                )

            save_audio(result, output)
            console.print(f"[green]Saved:[/green] {output} ({result.duration:.1f}s)")

        return audiogen_command
