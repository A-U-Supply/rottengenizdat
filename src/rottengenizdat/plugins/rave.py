from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import requests
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect

console = Console()

RAVE_API_BASE = "https://play.forum.ircam.fr/rave-vst-api"

AVAILABLE_MODELS = [
    "VCTK",
    "darbouka_onnx",
    "nasa",
    "percussion",
    "vintage",
    "isis",
    "musicnet",
    "sol_ordinario",
    "sol_full",
    "sol_ordinario_fast",
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rottengenizdat" / "models" / "rave"


def download_model(
    model_name: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Download a pretrained RAVE model, using cache if available."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / f"{model_name}.ts"
    if model_path.exists():
        return model_path

    url = f"{RAVE_API_BASE}/get_model/{model_name}"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Downloading {model_name}...", total=None)
        response = requests.get(url)
        response.raise_for_status()
        model_path.write_bytes(response.content)

    return model_path


def load_rave_model(
    model_name: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> torch.jit.ScriptModule:
    """Download (if needed) and load a RAVE model."""
    model_path = download_model(model_name, cache_dir)
    model = torch.jit.load(str(model_path)).eval()
    return model


class RaveEffect(AudioEffect):
    """RAVE — Realtime Audio Variational autoEncoder.

    Encode audio into latent space, manipulate, decode.
    Even a straight round-trip produces uncanny results.
    """

    name = "rave"
    description = "RAVE variational autoencoder — latent space audio mangling"

    def process(
        self,
        audio: AudioBuffer,
        model_name: str = "percussion",
        temperature: float = 1.0,
        noise: float = 0.0,
        mix: float = 1.0,
        **kwargs,
    ) -> AudioBuffer:
        """Process audio through RAVE.

        Args:
            audio: Input audio buffer.
            model_name: Pretrained model to use.
            temperature: Scale latent vectors (>1 = more extreme, <1 = subtle).
            noise: Amount of random noise to add to latent space (0-1).
            mix: Wet/dry blend (0.0 = fully dry/original, 1.0 = fully wet/RAVE).
        """
        model = load_rave_model(model_name)

        with torch.no_grad():
            mono = audio.to_mono()
            x = mono.as_model_input()  # (1, 1, samples)
            z = model.encode(x)

            if temperature != 1.0:
                z = z * temperature

            if noise > 0.0:
                z = z + torch.randn_like(z) * noise

            x_hat = model.decode(z)

        wet = AudioBuffer.from_model_output(x_hat, sample_rate=audio.sample_rate)

        if mix >= 1.0:
            return wet

        # Blend dry (original) and wet (RAVE) signals
        dry = mono
        # Match lengths — RAVE output may differ slightly in length
        min_len = min(dry.num_samples, wet.num_samples)
        dry_samples = dry.samples[:, :min_len]
        wet_samples = wet.samples[:, :min_len]
        blended = dry_samples * (1.0 - mix) + wet_samples * mix
        return AudioBuffer(samples=blended, sample_rate=audio.sample_rate)

    def register_command(self, app: typer.Typer) -> None:
        """Register the rave subcommand."""

        @app.command(name=self.name, help=self.description)
        def rave_command(
            input_file: Annotated[
                Path, typer.Argument(help="Input audio file")
            ],
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file path")
            ] = Path("output.wav"),
            model: Annotated[
                str,
                typer.Option(
                    "--model",
                    "-m",
                    help=f"Pretrained model: {', '.join(AVAILABLE_MODELS)}",
                ),
            ] = "percussion",
            temperature: Annotated[
                float,
                typer.Option(
                    "--temperature",
                    "-t",
                    help="Latent scaling (>1 extreme, <1 subtle)",
                ),
            ] = 1.0,
            noise_amount: Annotated[
                float,
                typer.Option(
                    "--noise",
                    "-n",
                    help="Random noise in latent space (0-1)",
                ),
            ] = 0.0,
            mix: Annotated[
                float,
                typer.Option(
                    "--mix",
                    "-w",
                    help="Wet/dry blend (0.0=original, 0.5=half, 1.0=full RAVE)",
                ),
            ] = 1.0,
            sweep: Annotated[
                Optional[str],
                typer.Option(
                    "--sweep",
                    help="Sweep a parameter, e.g. temperature=0.5,1.0,1.5",
                ),
            ] = None,
        ) -> None:
            if not input_file.exists():
                console.print(f"[red]File not found: {input_file}[/red]")
                raise typer.Exit(1)

            console.print(f"[bold]Loading:[/bold] {input_file}")
            audio = load_audio(input_file)
            console.print(
                f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz"
            )

            if sweep:
                self._run_sweep(audio, output, model, temperature, noise_amount, mix, sweep)
            else:
                console.print(
                    f"[bold]Processing:[/bold] rave model={model} temp={temperature} noise={noise_amount} mix={mix}"
                )
                result = self.process(
                    audio,
                    model_name=model,
                    temperature=temperature,
                    noise=noise_amount,
                    mix=mix,
                )
                save_audio(result, output)
                console.print(f"[green]Saved:[/green] {output}")

        return rave_command

    def _run_sweep(
        self,
        audio: AudioBuffer,
        output_dir: Path,
        model: str,
        temperature: float,
        noise_amount: float,
        mix: float,
        sweep: str,
    ) -> None:
        """Run a parameter sweep and save grid of outputs."""
        param_name, values_str = sweep.split("=", 1)
        values = [float(v) for v in values_str.split(",")]

        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Sweeping:[/bold] {param_name} = {values}")

        for val in values:
            kwargs: dict = {
                "model_name": model,
                "temperature": temperature,
                "noise": noise_amount,
                "mix": mix,
            }
            kwargs[param_name] = val

            result = self.process(audio, **kwargs)
            filename = f"{param_name}_{val:.2f}.wav"
            save_audio(result, output_dir / filename)
            console.print(f"  [green]Saved:[/green] {output_dir / filename}")
