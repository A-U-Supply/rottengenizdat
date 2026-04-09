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


_RAVE_HELP = """\
RAVE variational autoencoder — latent space audio mangling.

Encode audio into a neural network's latent space, twist the knobs,
decode. Even a straight round-trip (no knobs) produces uncanny results
because the model reinterprets your audio through whatever it was
trained on. A percussion model turns everything into ghost drums.
A voice model makes your synths sing.

MODELS — each was trained on different audio and imparts its character:

  percussion ........ Drum/percussion instruments. Makes everything
                      rhythmic and percussive. Good default for beats.
  vintage ........... Warm analog character — tape hiss, vinyl crackle
                      energy. Great for subtle lo-fi textures.
  VCTK .............. Human speech (VCTK corpus). Gives audio a vocal,
                      mouth-shaped quality. Eerie on non-vocal input.
  nasa .............. NASA mission recordings — radio chatter, space
                      noise, telemetry. Alien and otherworldly.
  musicnet .......... Multi-instrument classical (MusicNet dataset).
                      Orchestral reinterpretation of anything.
  isis .............. Instrumental sounds. Adds tonal, resonant color.
  sol_ordinario ..... Solo strings, ordinario technique. Bowed string
                      character — good for haunting, organic textures.
  sol_full .......... Solo strings, full technique range. Broader string
                      vocabulary than sol_ordinario.
  sol_ordinario_fast  Fast variant of sol_ordinario. Same character,
                      quicker inference.
  darbouka_onnx ..... Darbouka drum. Middle-Eastern percussion color.

KNOBS — manipulate the latent space between encode and decode:

  -t, --temperature   Scale latent vectors. 1.0 = identity. <1 pulls
                      toward the model's average (subtle, smoothed).
                      >1 pushes extremes (chaotic, distorted).
                      0.5 = gentle, 1.5 = aggressive, 2.0+ = destroyed.

  -n, --noise         Add gaussian noise to latent space (0.0-1.0).
                      Injects randomness. 0.1 = texture, 0.3 = haze,
                      0.8 = static wash. Stacks with temperature.

  -w, --mix           Wet/dry blend. 0.0 = fully original, 1.0 = fully
                      RAVE (default). 0.3 = ghostly undertone,
                      0.5 = half-and-half, 0.7 = mostly transformed.

  -d, --dims          Comma-separated latent dimension indices to
                      manipulate (e.g. "0,3,7"). Unselected dims keep
                      their original encoded values. Models typically
                      have 16 dims. Lower dims (0-3) tend to carry more
                      structural info; higher dims (12-15) carry finer
                      detail. Omit to manipulate all dims.

  -r, --reverse       Flip the temporal axis of the latent. The model
                      hears your audio backwards but decodes forward —
                      not the same as reversing the audio file.

  --shuffle N         Cut the latent timeline into chunks of N frames
                      and shuffle them randomly. Creates temporal
                      dislocation — the structure fragments.
                      4-8 = mild stutter, 2-3 = heavy chop.

  -q, --quantize      Snap latent values to a grid of this step size.
                      Crushes the continuous latent into steps.
                      0.1 = subtle, 0.5 = crunchy, 1.0 = obliterated.

  --sweep             Generate a grid of outputs sweeping one parameter.
                      e.g. --sweep temperature=0.5,1.0,1.5,2.0
                      Output path (-o) becomes a directory.

EXAMPLES:

  Straight round-trip (no knobs — still transforms!):
    rotten rave input.wav -m vintage -o out.wav

  Subtle lo-fi warmth:
    rotten rave input.wav -m vintage -t 0.7 -d 0,1 -w 0.4 -o warm.wav

  Aggressive percussion reinterpretation:
    rotten rave input.wav -m percussion -t 1.5 -o drums.wav

  Alien vocal texture:
    rotten rave input.wav -m VCTK -t 1.3 -n 0.2 -o eerie.wav

  Space noise with reversed latent:
    rotten rave input.wav -m nasa -t 1.4 -r -o space.wav

  Only mangle high dimensions, keep structure:
    rotten rave input.wav -m musicnet -d 8,9,10,11,12,13,14,15 -t 1.5 -o detail.wav

  Bitcrushed latent:
    rotten rave input.wav -m percussion -q 0.8 -t 1.5 -o crushed.wav

  Explore temperature range:
    rotten rave input.wav -m vintage --sweep temperature=0.3,0.7,1.0,1.5,2.0 -o grid/
"""


class RaveEffect(AudioEffect):
    """RAVE — Realtime Audio Variational autoEncoder.

    Encode audio into latent space, manipulate, decode.
    Even a straight round-trip produces uncanny results.
    """

    name = "rave"
    description = _RAVE_HELP

    def process(
        self,
        audio: AudioBuffer,
        model_name: str = "percussion",
        temperature: float = 1.0,
        noise: float = 0.0,
        mix: float = 1.0,
        dims: Optional[str] = None,
        reverse: bool = False,
        shuffle_chunks: int = 0,
        quantize: float = 0.0,
        **kwargs,
    ) -> AudioBuffer:
        """Process audio through RAVE.

        Args:
            audio: Input audio buffer.
            model_name: Pretrained model to use.
            temperature: Scale latent vectors (>1 = more extreme, <1 = subtle).
            noise: Amount of random noise to add to latent space (0-1).
            mix: Wet/dry blend (0.0 = fully dry/original, 1.0 = fully wet/RAVE).
            dims: Comma-separated latent dim indices to manipulate (e.g. "0,3,7").
                  Unselected dims retain the original encoded values. None = all dims.
            reverse: Reverse the temporal axis of the latent representation.
            shuffle_chunks: Shuffle latent in temporal chunks of this many frames (0 = off).
            quantize: Snap latent values to a grid of this step size (0.0 = off).
        """
        model = load_rave_model(model_name)

        with torch.no_grad():
            mono = audio.to_mono()
            x = mono.as_model_input()  # (1, 1, samples)
            z = model.encode(x)

            # Save original for dim restoration after manipulations
            z_original = z.clone() if dims is not None else None

            # Build dim mask (True = dims to leave untouched)
            if dims is not None:
                dim_indices = [int(d) for d in dims.split(",")]
                mask = torch.ones(z.shape[1], dtype=torch.bool)
                mask[dim_indices] = False

            if temperature != 1.0:
                z = z * temperature

            if noise > 0.0:
                z = z + torch.randn_like(z) * noise

            if quantize > 0.0:
                z = torch.round(z / quantize) * quantize

            if reverse:
                z = z.flip(dims=[-1])

            if shuffle_chunks > 0:
                n_frames = z.shape[-1]
                n_chunks = n_frames // shuffle_chunks
                if n_chunks > 1:
                    chunks = list(z.split(shuffle_chunks, dim=-1))
                    indices = torch.randperm(len(chunks))
                    z = torch.cat([chunks[i] for i in indices], dim=-1)

            # Restore unselected dims from original encoding
            if dims is not None:
                z[:, mask, :] = z_original[:, mask, :]

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
                Path, typer.Argument(help="Input audio file (wav, flac, mp3, etc.)")
            ],
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file path (or directory when using --sweep)")
            ] = Path("output.wav"),
            model: Annotated[
                str,
                typer.Option(
                    "--model",
                    "-m",
                    help="Pretrained RAVE model. Each imparts its own character. "
                    f"Choices: {', '.join(AVAILABLE_MODELS)}",
                ),
            ] = "percussion",
            temperature: Annotated[
                float,
                typer.Option(
                    "--temperature",
                    "-t",
                    help="Scale latent vectors. <1 = subtle/smoothed, "
                    "1.0 = identity, >1 = extreme/chaotic. "
                    "Try 0.5-0.8 for gentle, 1.2-2.0 for aggressive",
                ),
            ] = 1.0,
            noise_amount: Annotated[
                float,
                typer.Option(
                    "--noise",
                    "-n",
                    help="Gaussian noise added to latent space (0.0-1.0). "
                    "0.1 = texture, 0.3 = haze, 0.8 = wash",
                ),
            ] = 0.0,
            mix: Annotated[
                float,
                typer.Option(
                    "--mix",
                    "-w",
                    help="Wet/dry blend. 0.0 = original, 0.3 = ghostly, "
                    "0.5 = half, 1.0 = full RAVE",
                ),
            ] = 1.0,
            dims: Annotated[
                Optional[str],
                typer.Option(
                    "--dims",
                    "-d",
                    help="Latent dims to manipulate (e.g. '0,1,2,3'). "
                    "Others keep original values. Lower dims = structure, "
                    "higher = detail. Omit for all dims",
                ),
            ] = None,
            reverse: Annotated[
                bool,
                typer.Option(
                    "--reverse",
                    "-r",
                    help="Flip latent time axis — not the same as reversing the audio file",
                ),
            ] = False,
            shuffle_chunks: Annotated[
                int,
                typer.Option(
                    "--shuffle",
                    help="Shuffle latent in chunks of N frames. "
                    "4-8 = mild stutter, 2-3 = heavy chop",
                ),
            ] = 0,
            quantize_step: Annotated[
                float,
                typer.Option(
                    "--quantize",
                    "-q",
                    help="Snap latent values to grid of this step size. "
                    "0.1 = subtle, 0.5 = crunchy, 1.0 = obliterated",
                ),
            ] = 0.0,
            sweep: Annotated[
                Optional[str],
                typer.Option(
                    "--sweep",
                    help="Generate grid of outputs sweeping one parameter. "
                    "e.g. 'temperature=0.5,1.0,1.5,2.0'. Output path becomes a directory",
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
                self._run_sweep(
                    audio, output, model, temperature, noise_amount, mix,
                    dims, reverse, shuffle_chunks, quantize_step, sweep,
                )
            else:
                console.print(
                    f"[bold]Processing:[/bold] rave model={model} temp={temperature} "
                    f"noise={noise_amount} mix={mix} dims={dims} reverse={reverse} "
                    f"shuffle={shuffle_chunks} quantize={quantize_step}"
                )
                result = self.process(
                    audio,
                    model_name=model,
                    temperature=temperature,
                    noise=noise_amount,
                    mix=mix,
                    dims=dims,
                    reverse=reverse,
                    shuffle_chunks=shuffle_chunks,
                    quantize=quantize_step,
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
        dims: Optional[str],
        reverse: bool,
        shuffle_chunks: int,
        quantize_step: float,
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
                "dims": dims,
                "reverse": reverse,
                "shuffle_chunks": shuffle_chunks,
                "quantize": quantize_step,
            }
            kwargs[param_name] = val

            result = self.process(audio, **kwargs)
            filename = f"{param_name}_{val:.2f}.wav"
            save_audio(result, output_dir / filename)
            console.print(f"  [green]Saved:[/green] {output_dir / filename}")
