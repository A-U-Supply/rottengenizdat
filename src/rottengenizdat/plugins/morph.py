from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect
from rottengenizdat.plugins.rave import RaveEffect

console = Console()

_MORPH_HELP = """\
Latent space interpolation — morph one sound into another.

Encodes two audio files through a RAVE model, blends their latent
representations at different ratios (using spherical linear interpolation
for perceptually smooth crossfades), and decodes the result. You hear
one sound become the other through the model's neural brain.

If you provide --steps N, generates N+1 outputs sweeping from 0.0 to 1.0.
With just --ratio you get a single output at that blend point.

EXAMPLES:

  # 50/50 blend halfway between two files
  rotten morph kick.wav snare.wav -m percussion -o hybride.wav

  # Subtle shift (20% toward B)
  rotten morph a.wav b.wav -m vintage --ratio 0.2 -o slightly.wav

  # 11-step sweep — watch the transition
  rotten morph a.wav b.wav -m VCTK --steps 10 -o morph-grid/

  # Morph with all the standard rave knobs on top
  rotten morph a.wav b.wav -m vintage -t 1.2 -n 0.1 --steps 5 -o grid/
"""


class MorphEffect(AudioEffect):
    """Morph between two audio files via latent space interpolation."""

    name = "morph"
    description = _MORPH_HELP

    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        """Not used — morph takes two files via its own CLI."""
        return audio

    def register_command(self, app: typer.Typer) -> None:
        @app.command(name=self.name, help=self.description)
        def morph_command(
            file_a: Annotated[
                Path, typer.Argument(help="First input audio file (start point)")
            ],
            file_b: Annotated[
                Path, typer.Argument(help="Second input audio file (end point)")
            ],
            output: Annotated[
                Path,
                typer.Option("--output", "-o", help="Output file path (or directory when using --steps)"),
            ] = Path("morph.wav"),
            model: Annotated[
                str,
                typer.Option(
                    "--model",
                    "-m",
                    help="Pretrained RAVE model",
                ),
            ] = "percussion",
            temperature: Annotated[
                float,
                typer.Option(
                    "--temperature",
                    "-t",
                    help="Scale latent vectors post-interpolation. <1 = subtle, >1 = extreme",
                ),
            ] = 1.0,
            ratio: Annotated[
                float,
                typer.Option(
                    "--ratio",
                    help="Blend ratio (0.0 = all A, 0.5 = halfway, 1.0 = all B)",
                ),
            ] = 0.5,
            steps: Annotated[
                Optional[int],
                typer.Option(
                    "--steps",
                    help="Generate N+1 outputs sweeping from 0.0 to 1.0. Output becomes a directory",
                ),
            ] = None,
            noise_amount: Annotated[
                float,
                typer.Option(
                    "--noise",
                    "-n",
                    help="Gaussian noise added to latent space (0.0-1.0)",
                ),
            ] = 0.0,
            dims: Annotated[
                Optional[str],
                typer.Option(
                    "--dims",
                    "-d",
                    help="Latent dims to manipulate (e.g. '0,1,2,3')",
                ),
            ] = None,
            reverse: Annotated[
                bool,
                typer.Option("--reverse", "-r", help="Flip latent time axis"),
            ] = False,
            shuffle_chunks: Annotated[
                int,
                typer.Option("--shuffle", help="Shuffle latent in chunks of N frames"),
            ] = 0,
            quantize_step: Annotated[
                float,
                typer.Option("--quantize", "-q", help="Snap latent values to grid"),
            ] = 0.0,
        ) -> None:
            for f in [file_a, file_b]:
                if not f.exists():
                    console.print(f"[red]File not found: {f}[/red]")
                    raise typer.Exit(1)

            console.print(f"[bold]Loading:[/bold] {file_a}")
            audio_a = load_audio(file_a)
            console.print(f"  {audio_a.duration:.1f}s, {audio_a.channels}ch, {audio_a.sample_rate}Hz")

            console.print(f"[bold]Loading:[/bold] {file_b}")
            audio_b = load_audio(file_b)
            console.print(f"  {audio_b.duration:.1f}s, {audio_b.channels}ch, {audio_b.sample_rate}Hz")

            effect = RaveEffect()

            if steps is not None:
                output.mkdir(parents=True, exist_ok=True)
                console.print(f"[bold]Morphing:[/bold] model={model} steps={steps}")
                for i in range(steps + 1):
                    r = i / steps
                    result = effect.interpolate(
                        audio_a, audio_b,
                        model_name=model,
                        ratio=r,
                        temperature=temperature,
                        noise=noise_amount,
                        dims=dims,
                        reverse=reverse,
                        shuffle_chunks=shuffle_chunks,
                        quantize=quantize_step,
                    )
                    out_path = output / f"morph_{r:.2f}.wav"
                    save_audio(result, out_path)
                    console.print(f"  [green]Saved:[/green] {out_path}")
            else:
                console.print(
                    f"[bold]Morphing:[/bold] model={model} ratio={ratio} "
                    f"temp={temperature}"
                )
                result = effect.interpolate(
                    audio_a, audio_b,
                    model_name=model,
                    ratio=ratio,
                    temperature=temperature,
                    noise=noise_amount,
                    dims=dims,
                    reverse=reverse,
                    shuffle_chunks=shuffle_chunks,
                    quantize=quantize_step,
                )
                save_audio(result, output)
                console.print(f"[green]Saved:[/green] {output}")

        return morph_command
