"""MusicGen plugin — text-to-music generation via HuggingFace transformers."""

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

_MUSICGEN_HELP = """\
MusicGen — text-to-music generation (Meta).

Describe a sound in words and MusicGen generates it. Multiple model sizes
available — small (300M) is fast, medium (1.5B) is higher quality, large
(3.3B) is best but needs a GPU to be practical. The melody variant can
be conditioned on an input file's chromagram.

KNOBS:

  -p, --prompt TEXT    Text description of the sound to generate.
                       "dark ambient drone", "funeral drums",
                       "cartoon brass band", "spectral choir" etc.

  -m, --model MODEL    Model size: small, medium, large, melody.
                       small = 300M (fast, default). melody = conditioned
                       on input file.

  --duration SEC       Output duration in seconds. Default 4.

  -t, --temperature    Sampling temperature. 0.8 = focused, 1.0 = default,
                       1.5 = wild, 2.0 = chaotic.

  --guidance-scale G   Classifier-free guidance strength (1.0–10.0).
                       Higher = closer to prompt, lower = more variation.
                       Default 3.0.

  --top-k K            Top-k sampling. Limits to K most likely tokens.
                       0 = off (default), 250 = focused.

  --top-p P            Nucleus sampling. 0.0 = off (default), 0.9 = typical.

  --seed N             Random seed for reproducibility.

MODELS download from HuggingFace on first use (~1.5GB for small, 6GB for
medium, 12GB for large). Cache at HF_HOME or ~/.cache/huggingface/.

EXAMPLES:

  Basic generation:
    rotten musicgen -p "dark ambient drone" --duration 8 -o drone.wav

  Hot and wild:
    rotten musicgen -p "broken cassette orchestra" -t 1.8 --duration 6 -o broken.wav

  Focused prompt adherence:
    rotten musicgen -p "funeral drums in a cave" --guidance-scale 7 -o drums.wav

  Melody-conditioned (re-imagine your audio):
    rotten musicgen -p "haunted carnival" -m melody --duration 8 input.wav -o carnival.wav
"""

MODEL_IDS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


def _load_model(model_size: str = "small"):
    """Load MusicGen model and processor from HuggingFace."""
    from transformers import AutoProcessor

    if model_size not in MODEL_IDS:
        raise ValueError(f"Unknown model size '{model_size}'. Choose: {', '.join(MODEL_IDS)}")

    model_id = MODEL_IDS[model_size]

    if model_size == "melody":
        from transformers import MusicgenMelodyForConditionalGeneration

        model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_id)
    else:
        from transformers import MusicgenForConditionalGeneration

        model = MusicgenForConditionalGeneration.from_pretrained(model_id)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


class MusicGenEffect(AudioEffect):
    """MusicGen — text-to-music generation."""

    name = "musicgen"
    description = _MUSICGEN_HELP

    def process(
        self,
        audio: Optional[AudioBuffer] = None,
        prompt: str = "dark ambient drone",
        model_size: str = "small",
        duration: float = 4.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> AudioBuffer:
        """Generate audio from a text prompt.

        Args:
            audio: Optional input for melody conditioning (melody model only).
            prompt: Text description.
            model_size: Model size (small, medium, large, melody).
            duration: Output duration in seconds.
            temperature: Sampling temperature.
            guidance_scale: Classifier-free guidance strength.
            top_k: Top-k sampling limit (0 = off).
            top_p: Nucleus sampling (0.0 = off).
            seed: Random seed.
        """
        model, processor, model_id = _load_model(model_size)

        # Calculate tokens: small model uses 50 tokens/s at 32kHz
        sample_rate = model.config.audio_encoder.sampling_rate
        tokens_per_second = 50
        max_new_tokens = int(duration * tokens_per_second)

        if seed is not None:
            torch.manual_seed(seed)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "guidance_scale": guidance_scale,
        }
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        if top_p > 0.0:
            gen_kwargs["top_p"] = top_p

        # Melody conditioning — only needs a short clip for chromagram
        if model_size == "melody" and audio is not None:
            # Warn and auto-trim long inputs (chromagram extraction is slow)
            if audio.duration > 30:
                console.print(
                    f"[yellow]Input is {audio.duration:.0f}s — trimming to first 30s "
                    f"for chromagram extraction. Use a shorter clip for faster results.[/yellow]"
                )
                audio = AudioBuffer(
                    samples=audio.samples[:, :int(30 * audio.sample_rate)],
                    sample_rate=audio.sample_rate,
                )

            inputs = processor(
                audio=audio.to_mono().samples.squeeze(0).numpy(),
                sampling_rate=audio.sample_rate,
                text=[prompt],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor(text=[prompt], return_tensors="pt", padding=True)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        # out shape: (batch, 1, samples) at 32kHz
        samples = out[0]  # (1, samples)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)

        return AudioBuffer(samples=samples.float(), sample_rate=sample_rate)

    def register_command(self, app: typer.Typer) -> None:
        """Register the musicgen subcommand."""

        @app.command(name=self.name, help=self.description)
        def musicgen_command(
            input_file: Annotated[
                Optional[Path],
                typer.Argument(help="Optional input audio file (for melody conditioning with -m melody)"),
            ] = None,
            output: Annotated[
                Path,
                typer.Option("--output", "-o", help="Output file path"),
            ] = Path("output.wav"),
            prompt: Annotated[
                str,
                typer.Option("--prompt", "-p", help="Text description of the sound to generate"),
            ] = "dark ambient drone",
            model_size: Annotated[
                str,
                typer.Option(
                    "--model",
                    "-m",
                    help=f"Model size: {', '.join(MODEL_IDS)}",
                ),
            ] = "small",
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
            guidance_scale: Annotated[
                float,
                typer.Option(
                    "--guidance-scale",
                    help="CFG strength. Higher = closer to prompt (1.0–10.0)",
                ),
            ] = 3.0,
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
            from rottengenizdat.core import load_audio

            audio = None
            if input_file is not None:
                if not input_file.exists():
                    console.print(f"[red]File not found: {input_file}[/red]")
                    raise typer.Exit(1)
                console.print(f"[bold]Loading:[/bold] {input_file}")
                audio = load_audio(input_file)
                console.print(f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")

            gen_info = f"model={model_size} prompt='{prompt}' duration={duration}s temp={temperature}"
            if guidance_scale != 3.0:
                gen_info += f" guidance={guidance_scale}"
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
                    audio=audio,
                    prompt=prompt,
                    model_size=model_size,
                    duration=duration,
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                )

            save_audio(result, output)
            console.print(f"[green]Saved:[/green] {output} ({result.duration:.1f}s)")

        return musicgen_command
