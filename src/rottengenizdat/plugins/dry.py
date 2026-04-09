from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.inputs import InputMode, combine_inputs
from rottengenizdat.plugin import AudioEffect

console = Console()

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
  rotten dry a.wav b.wav --mode concat -o combined.wav
"""


class DryEffect(AudioEffect):
    """Passthrough — returns the original audio unchanged."""

    name = "dry"
    description = _DRY_HELP

    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        return audio

    def register_command(self, app: typer.Typer) -> None:
        @app.command(name=self.name, help=self.description)
        def dry_command(
            input_files: Annotated[
                Optional[list[Path]], typer.Argument(help="Input audio file(s)")
            ] = None,
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file or directory path")
            ] = Path("output.wav"),
            sample_sale: Annotated[
                bool, typer.Option("--sample-sale", "-ss", help="Include random samples from #sample-sale")
            ] = False,
            sample_sale_count: Annotated[
                int, typer.Option("--sample-sale-count", help="How many samples to pull (implies --sample-sale)")
            ] = 0,
            input_mode: Annotated[
                Optional[str], typer.Option("--mode", help="Input combination mode: splice, concat, independent")
            ] = None,
            splice_min: Annotated[
                float, typer.Option("--splice-min", help="Min splice segment seconds")
            ] = 0.25,
            splice_max: Annotated[
                float, typer.Option("--splice-max", help="Max splice segment seconds")
            ] = 4.0,
        ) -> None:
            all_buffers: list[AudioBuffer] = []
            all_names: list[str] = []

            for f in (input_files or []):
                if not f.exists():
                    console.print(f"[red]File not found: {f}[/red]")
                    raise typer.Exit(1)
                all_buffers.append(load_audio(f))
                all_names.append(f.stem)

            ss_count = sample_sale_count if sample_sale_count > 0 else (1 if sample_sale else 0)
            if ss_count > 0:
                import rottengenizdat.sample_sale as _ss
                console.print(f"[bold]Fetching {ss_count} sample(s) from #sample-sale...[/bold]")
                index = _ss.sync_index()
                picks = _ss.pick_random_samples(index, ss_count)
                for entry in picks:
                    console.print(f"  [dim]Selected:[/dim] {entry.filename or entry.url or entry.id}")
                    path = _ss.download_sample(entry)
                    all_buffers.append(load_audio(path))
                    all_names.append(entry.filename or entry.id)

            if not all_buffers:
                console.print("[red]No input files provided. Pass audio files or use --sample-sale.[/red]")
                raise typer.Exit(1)

            resolved_mode = InputMode.resolve(input_mode, len(all_buffers))
            if len(all_buffers) > 1:
                console.print(f"[bold]Combining {len(all_buffers)} inputs[/bold] (mode={resolved_mode.value})")
            combined = combine_inputs(all_buffers, resolved_mode, splice_min=splice_min, splice_max=splice_max)

            if resolved_mode == InputMode.INDEPENDENT:
                output.mkdir(parents=True, exist_ok=True)
                for i, (audio, src_name) in enumerate(zip(combined, all_names)):
                    out_path = output / f"{i+1:03d}-{src_name}.wav"
                    save_audio(self.process(audio), out_path)
                    console.print(f"[green]Saved:[/green] {out_path}")
            else:
                audio = combined[0]
                save_audio(self.process(audio), output)
                console.print(f"[green]Saved:[/green] {output}")

        return dry_command
