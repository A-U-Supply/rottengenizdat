from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import requests
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.inputs import InputMode, combine_inputs
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


def _has_prior(model: torch.jit.ScriptModule) -> bool:
    """Check if a model has a working prior for unconditional generation."""
    if not hasattr(model, "_prior"):
        return False
    try:
        with torch.no_grad():
            seed = torch.randn(1, 1, 1)
            z = model._prior.forward(seed)
            return z.shape[1] >= 4  # sensible latent dims
    except Exception:
        return False


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

  -g, --generate      Generate new audio from the model's prior (no input
                      file needed). Uses the model's generative prior if
                      available (vintage, VCTK, nasa), otherwise falls
                      back to random latent generation.

  --duration N        Duration in seconds for --generate (default 2.0).

  --morph-with PATH   Path to a second audio file for latent interpolation.
                      Encodes both files, blends their latent representations,
                      decodes. Use --morph-ratio to set the blend (0-1).

  --morph-ratio R     Blend ratio for --morph-with (0.0 = file A, 1.0 = file B).

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

  Generate 4 seconds of new sound from the model's prior:
    rotten rave -g -m vintage --duration 4 -t 1.2 -o dream.wav

  Generate a drum pattern from scratch (random latent fallback):
    rotten rave -g -m percussion --duration 2 -t 1.5 -q 0.1 -o drums.wav

  Morph between two files in latent space:
    rotten rave kick.wav --morph-with snare.wav -m percussion --morph-ratio 0.5 -o hybrid.wav

  Multi-step morph with dedicated morph command:
    rotten morph kick.wav snare.wav -m vintage --steps 5 -o morph-grid/
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
                n_latent = z.shape[1]
                dim_indices = [int(d) for d in dims.split(",")]
                dim_indices = [i for i in dim_indices if i < n_latent]
                if not dim_indices:
                    dims = None  # all requested dims out of range, skip masking
                else:
                    mask = torch.ones(n_latent, dtype=torch.bool)
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

    def generate(
        self,
        model_name: str = "percussion",
        duration: float = 2.0,
        sample_rate: int = 44100,
        temperature: float = 1.0,
        noise: float = 0.0,
        dims: Optional[str] = None,
        reverse: bool = False,
        shuffle_chunks: int = 0,
        quantize: float = 0.0,
        **kwargs,
    ) -> AudioBuffer:
        """Generate audio from the model's prior (or random latent as fallback).

        Args:
            model_name: Pretrained model to use.
            duration: Output duration in seconds.
            sample_rate: Sample rate for output audio.
            temperature: Scale latent vectors (>1 = more extreme, <1 = subtle).
            noise: Amount of random noise to add to latent space (0-1).
            dims: Comma-separated latent dim indices to manipulate.
            reverse: Reverse the temporal axis of the latent.
            shuffle_chunks: Shuffle latent in temporal chunks (0 = off).
            quantize: Snap latent values to a grid of this step size (0.0 = off).
        """
        model = load_rave_model(model_name)

        # Rough frames-per-second: each latent frame decodes to ~2048 samples
        n_frames = max(1, int(duration * sample_rate / 2048))

        with torch.no_grad():
            if _has_prior(model):
                # Prior uses its own internal temperature — seed magnitude doesn't
                # matter, it just needs the right shape as a write buffer.
                seed = torch.randn(1, 1, n_frames)
                z = model._prior.forward(seed)
            else:
                # Fallback: generate random latent directly
                x_test = torch.randn(1, 1, 4096)
                z_test = model.encode(x_test)
                n_latent = z_test.shape[1]
                z = torch.randn(1, n_latent, n_frames)

            # Apply temperature as latent scaling (same as process() pipeline)
            if temperature != 1.0:
                z = z * temperature

            # Save original for dim restoration
            z_original = z.clone() if dims is not None else None

            # Build dim mask
            if dims is not None:
                n_latent = z.shape[1]
                dim_indices = [int(d) for d in dims.split(",")]
                dim_indices = [i for i in dim_indices if i < n_latent]
                if not dim_indices:
                    dims = None
                else:
                    mask = torch.ones(n_latent, dtype=torch.bool)
                    mask[dim_indices] = False

            # Apply noise after prior (adds texture to generated output)
            if noise > 0.0:
                z = z + torch.randn_like(z) * noise

            if quantize > 0.0:
                z = torch.round(z / quantize) * quantize

            if reverse:
                z = z.flip(dims=[-1])

            if shuffle_chunks > 0:
                n_frames_actual = z.shape[-1]
                n_chunks = n_frames_actual // shuffle_chunks
                if n_chunks > 1:
                    chunks = list(z.split(shuffle_chunks, dim=-1))
                    indices = torch.randperm(len(chunks))
                    z = torch.cat([chunks[i] for i in indices], dim=-1)

            if dims is not None:
                z[:, mask, :] = z_original[:, mask, :]

            x_hat = model.decode(z)

        buf = AudioBuffer.from_model_output(x_hat, sample_rate=sample_rate)
        buf = buf.to_mono()

        # Prior output is often very quiet — normalize to a listenable level.
        # Use RMS normalization with a peak ceiling to avoid clipping spikes.
        rms = buf.samples.square().mean().sqrt()
        if rms > 0:
            target_rms = 0.15  # ~ -16dB RMS, typical for speech
            gain = min(target_rms / rms, 0.95 / (buf.samples.abs().max() + 1e-8))
            if gain > 1.0:
                buf = AudioBuffer(
                    samples=buf.samples * gain,
                    sample_rate=buf.sample_rate,
                )

        return buf

    @staticmethod
    def interpolate(
        audio_a: AudioBuffer,
        audio_b: AudioBuffer,
        model_name: str,
        ratio: float = 0.5,
        temperature: float = 1.0,
        noise: float = 0.0,
        dims: Optional[str] = None,
        reverse: bool = False,
        shuffle_chunks: int = 0,
        quantize: float = 0.0,
    ) -> AudioBuffer:
        """Interpolate between two audio files in latent space.

        Encodes both files, blends their latent representations at *ratio*,
        then decodes the result. Ratio 0.0 = all A, 1.0 = all B.

        Uses spherical linear interpolation (slerp) for a perceptually
        smoother crossfade than linear blending.

        Args:
            audio_a: First input audio buffer.
            audio_b: Second input audio buffer.
            model_name: Pretrained model to use.
            ratio: Blend point (0.0 = A, 0.5 = halfway, 1.0 = B).
            temperature: Scale latent vectors post-interpolation.
            noise: Random noise added to latent space (0-1).
            dims: Comma-separated latent dim indices to manipulate.
            reverse: Reverse the temporal axis of the latent.
            shuffle_chunks: Shuffle latent in temporal chunks (0 = off).
            quantize: Snap latent values to a grid (0.0 = off).
        """
        model = load_rave_model(model_name)

        with torch.no_grad():
            mono_a = audio_a.to_mono()
            mono_b = audio_b.to_mono()
            x_a = mono_a.as_model_input()
            x_b = mono_b.as_model_input()

            z_a = model.encode(x_a)
            z_b = model.encode(x_b)

            # Match temporal length — use the shorter one
            min_frames = min(z_a.shape[-1], z_b.shape[-1])
            z_a = z_a[:, :, :min_frames]
            z_b = z_b[:, :, :min_frames]

            # Spherical linear interpolation (slerp) for smoother morphing
            if ratio <= 0.0:
                z = z_a
            elif ratio >= 1.0:
                z = z_b
            else:
                # Normalize for slerp
                norm_a = torch.norm(z_a, dim=1, keepdim=True).clamp(min=1e-8)
                norm_b = torch.norm(z_b, dim=1, keepdim=True).clamp(min=1e-8)
                za_norm = z_a / norm_a
                zb_norm = z_b / norm_b

                # Angle between vectors
                dot = (za_norm * zb_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                omega = torch.acos(dot)

                sin_omega = torch.sin(omega).clamp(min=1e-8)
                weight_a = torch.sin((1.0 - ratio) * omega) / sin_omega
                weight_b = torch.sin(ratio * omega) / sin_omega

                # Blend magnitudes linearly
                norm_blend = norm_a * (1.0 - ratio) + norm_b * ratio

                z = (za_norm * weight_a + zb_norm * weight_b) * norm_blend

            # Apply knobs
            z_original = z.clone() if dims is not None else None

            if dims is not None:
                n_latent = z.shape[1]
                dim_indices = [int(d) for d in dims.split(",")]
                dim_indices = [i for i in dim_indices if i < n_latent]
                if not dim_indices:
                    dims = None
                else:
                    mask = torch.ones(n_latent, dtype=torch.bool)
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
                n_frames_actual = z.shape[-1]
                n_chunks = n_frames_actual // shuffle_chunks
                if n_chunks > 1:
                    chunks = list(z.split(shuffle_chunks, dim=-1))
                    indices = torch.randperm(len(chunks))
                    z = torch.cat([chunks[i] for i in indices], dim=-1)

            if dims is not None:
                z[:, mask, :] = z_original[:, mask, :]

            x_hat = model.decode(z)

        # Use A's sample rate (both should match)
        buf = AudioBuffer.from_model_output(x_hat, sample_rate=audio_a.sample_rate)
        return buf.to_mono()

    @staticmethod
    def encode_latent(audio: AudioBuffer, model_name: str) -> tuple[torch.Tensor, int]:
        """Encode audio to latent representation, returning (z, sample_rate).

        The latent tensor can be saved to disk and decoded later — potentially
        through a DIFFERENT model for cross-model transfer.
        """
        model = load_rave_model(model_name)
        mono = audio.to_mono()
        x = mono.as_model_input()
        with torch.no_grad():
            z = model.encode(x)
        return z.cpu().clone(), audio.sample_rate

    @staticmethod
    def decode_latent(
        z: torch.Tensor,
        model_name: str,
        sample_rate: int = 44100,
        temperature: float = 1.0,
        noise: float = 0.0,
        dims: Optional[str] = None,
        reverse: bool = False,
        shuffle_chunks: int = 0,
        quantize: float = 0.0,
    ) -> AudioBuffer:
        """Decode a latent tensor through a RAVE model (potentially different from encoder).

        This enables cross-model transfer: encode through percussion, decode
        through VCTK. The latent shape must be compatible with the decode model.
        """
        model = load_rave_model(model_name)

        with torch.no_grad():
            z = z.clone()

            z_original = z.clone() if dims is not None else None

            if dims is not None:
                n_latent = z.shape[1]
                dim_indices = [int(d) for d in dims.split(",")]
                dim_indices = [i for i in dim_indices if i < n_latent]
                if not dim_indices:
                    dims = None
                else:
                    mask = torch.ones(n_latent, dtype=torch.bool)
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
                n_frames_actual = z.shape[-1]
                n_chunks = n_frames_actual // shuffle_chunks
                if n_chunks > 1:
                    chunks = list(z.split(shuffle_chunks, dim=-1))
                    indices = torch.randperm(len(chunks))
                    z = torch.cat([chunks[i] for i in indices], dim=-1)

            if dims is not None:
                z[:, mask, :] = z_original[:, mask, :]

            x_hat = model.decode(z)

        buf = AudioBuffer.from_model_output(x_hat, sample_rate=sample_rate)
        return buf.to_mono()

    def register_command(self, app: typer.Typer) -> None:
        """Register the rave subcommand."""

        @app.command(name=self.name, help=self.description)
        def rave_command(
            input_files: Annotated[
                Optional[list[Path]], typer.Argument(help="Input audio file(s) (wav, flac, mp3, etc.)")
            ] = None,
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file path (or directory when using --sweep or --mode independent)")
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
            generate: Annotated[
                bool, typer.Option("--generate", "-g", help="Generate new audio from the model's prior (no input file needed)")
            ] = False,
            duration: Annotated[
                float, typer.Option("--duration", help="Duration in seconds for generated output (default 2.0)")
            ] = 2.0,
            morph_with: Annotated[
                Optional[str], typer.Option("--morph-with", help="Path to a second audio file for latent interpolation")
            ] = None,
            morph_ratio: Annotated[
                float, typer.Option("--morph-ratio", help="Blend ratio for --morph-with (0.0 = file A, 1.0 = file B)")
            ] = 0.5,
            repeat_count: Annotated[
                int, typer.Option("--repeat", help="Feed output back as input N times. Compounds the model's artifacts")
            ] = 1,
            grid: Annotated[
                Optional[str], typer.Option("--grid", help="Multi-param grid sweep. e.g. 'temperature=0.5,1.0,1.5;noise=0,0.1,0.2'. Output becomes a directory")
            ] = None,
        ) -> None:
            # --- Generation mode (no input file needed) ---
            if generate:
                has_prior = False
                try:
                    mdl = load_rave_model(model)
                    has_prior = _has_prior(mdl)
                except Exception:
                    pass
                gen_label = "prior" if has_prior else "random latent"
                console.print(
                    f"[bold]Generating:[/bold] rave model={model} temp={temperature} "
                    f"duration={duration}s mode={gen_label} "
                    f"noise={noise_amount} dims={dims} reverse={reverse} "
                    f"shuffle={shuffle_chunks} quantize={quantize_step}"
                )
                result = self.generate(
                    model_name=model,
                    duration=duration,
                    temperature=temperature,
                    noise=noise_amount,
                    dims=dims,
                    reverse=reverse,
                    shuffle_chunks=shuffle_chunks,
                    quantize=quantize_step,
                )
                save_audio(result, output)
                console.print(f"[green]Saved:[/green] {output}")
                return

            # --- Morph mode (interpolate between two files) ---
            if morph_with is not None and input_files:
                b_path = Path(morph_with)
                if not b_path.exists():
                    console.print(f"[red]File not found: {b_path}[/red]")
                    raise typer.Exit(1)
                console.print(
                    f"[bold]Morphing:[/bold] {input_files[0]} <-> {b_path} "
                    f"ratio={morph_ratio} model={model} temp={temperature}"
                )
                audio_a = load_audio(input_files[0])
                audio_b = load_audio(b_path)
                result = self.interpolate(
                    audio_a, audio_b,
                    model_name=model,
                    ratio=morph_ratio,
                    temperature=temperature,
                    noise=noise_amount,
                    dims=dims,
                    reverse=reverse,
                    shuffle_chunks=shuffle_chunks,
                    quantize=quantize_step,
                )
                save_audio(result, output)
                console.print(f"[green]Saved:[/green] {output}")
                return

            all_buffers: list[AudioBuffer] = []
            all_names: list[str] = []

            for f in (input_files or []):
                if not f.exists():
                    console.print(f"[red]File not found: {f}[/red]")
                    raise typer.Exit(1)
                if f.is_dir():
                    console.print(
                        f"[red]'{f}' is a directory, not an audio file.[/red]\n"
                        f"  Did you mean: [bold]rotten recipe run {f}/<name>.toml ...[/bold]?"
                    )
                    raise typer.Exit(1)
                console.print(f"[bold]Loading:[/bold] {f}")
                try:
                    buf = load_audio(f)
                except Exception as e:
                    console.print(f"[red]Cannot read '{f}': {e}[/red]")
                    raise typer.Exit(1)
                console.print(
                    f"  {buf.duration:.1f}s, {buf.channels}ch, {buf.sample_rate}Hz"
                )
                all_buffers.append(buf)
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
                    if sweep:
                        self._run_sweep(
                            audio, out_path, model, temperature, noise_amount, mix,
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
                        for _ in range(repeat_count - 1):
                            result = self.process(
                                result,
                                model_name=model,
                                temperature=temperature,
                                noise=noise_amount,
                                mix=1.0,  # no dry blend on repeats
                                dims=dims,
                                reverse=reverse,
                                shuffle_chunks=shuffle_chunks,
                                quantize=quantize_step,
                            )
                        save_audio(result, out_path)
                        console.print(f"[green]Saved:[/green] {out_path}")
            else:
                audio = combined[0]
                if sweep:
                    self._run_sweep(
                        audio, output, model, temperature, noise_amount, mix,
                        dims, reverse, shuffle_chunks, quantize_step, sweep,
                    )
                elif grid:
                    self._run_grid(
                        audio, output, model, temperature, noise_amount, mix,
                        dims, reverse, shuffle_chunks, quantize_step, grid,
                        repeat=repeat_count,
                    )
                else:
                    console.print(
                        f"[bold]Processing:[/bold] rave model={model} temp={temperature} "
                        f"noise={noise_amount} mix={mix} dims={dims} reverse={reverse} "
                        f"shuffle={shuffle_chunks} quantize={quantize_step}"
                        + (f" repeat={repeat_count}" if repeat_count > 1 else "")
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
                    for _ in range(repeat_count - 1):
                        result = self.process(
                            result,
                            model_name=model,
                            temperature=temperature,
                            noise=noise_amount,
                            mix=1.0,
                            dims=dims,
                            reverse=reverse,
                            shuffle_chunks=shuffle_chunks,
                            quantize=quantize_step,
                        )
                    save_audio(result, output)
                    console.print(f"[green]Saved:[/green] {output}")

        # Register encode/decode subcommands
        self._register_extra_commands(app)

        return rave_command

    def _run_grid(
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
        grid: str,
        repeat: int = 1,
    ) -> None:
        """Run a multi-parameter grid sweep and save all combinations."""
        from itertools import product

        # Parse "temperature=0.5,1.0,1.5;noise=0,0.1,0.2"
        param_sets: list[tuple[str, list[float]]] = []
        for part in grid.split(";"):
            part = part.strip()
            if not part:
                continue
            name, vals = part.split("=", 1)
            param_sets.append((name.strip(), [float(v.strip()) for v in vals.split(",")]))

        names = [p[0] for p in param_sets]
        value_lists = [p[1] for p in param_sets]

        output_dir.mkdir(parents=True, exist_ok=True)
        total = 1
        for vl in value_lists:
            total *= len(vl)
        console.print(f"[bold]Grid sweep:[/bold] {names} = {[[v for v in vl] for vl in value_lists]} ({total} combinations)")

        count = 0
        for combo in product(*value_lists):
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
            for name, val in zip(names, combo):
                kwargs[name] = val

            filename_parts = [f"{n}_{v:.2f}" for n, v in zip(names, combo)]
            filename = "__".join(filename_parts) + ".wav"

            result = self.process(audio, **kwargs)
            for _ in range(repeat - 1):
                result = self.process(result, **{**kwargs, "mix": 1.0})
            save_audio(result, output_dir / filename)
            count += 1
            console.print(f"  [{count}/{total}] [green]Saved:[/green] {output_dir / filename}")

    def _register_extra_commands(self, app: typer.Typer) -> None:
        """Register encode/decode subcommands for latent export/import."""

        @app.command(name="encode", help="Encode audio to latent representation (for cross-model decode)")
        def encode_command(
            input_file: Annotated[
                Path, typer.Argument(help="Input audio file")
            ],
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output .pt file path")
            ] = Path("latent.pt"),
            model: Annotated[
                str,
                typer.Option("--model", "-m", help=f"Pretrained RAVE model. Choices: {', '.join(AVAILABLE_MODELS)}"),
            ] = "percussion",
        ) -> None:
            if not input_file.exists():
                console.print(f"[red]File not found: {input_file}[/red]")
                raise typer.Exit(1)
            console.print(f"[bold]Encoding:[/bold] {input_file} -> {model}")
            audio = load_audio(input_file)
            z, sr = self.encode_latent(audio, model)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"z": z, "sample_rate": sr, "model": model}, output)
            console.print(f"[green]Saved:[/green] {output} (latent: {list(z.shape)}, sr={sr})")

        @app.command(name="decode", help="Decode a latent file through a RAVE model (cross-model transfer)")
        def decode_command(
            latent_file: Annotated[
                Path, typer.Argument(help="Latent .pt file from 'rotten encode'")
            ],
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output audio file path")
            ] = Path("decoded.wav"),
            model: Annotated[
                Optional[str],
                typer.Option("--model", "-m", help="RAVE model to decode through. Defaults to the model used for encoding"),
            ] = None,
            temperature: Annotated[
                float, typer.Option("--temperature", "-t", help="Scale latent vectors")
            ] = 1.0,
            noise_amount: Annotated[
                float, typer.Option("--noise", "-n", help="Gaussian noise added to latent space")
            ] = 0.0,
            dims: Annotated[
                Optional[str], typer.Option("--dims", "-d", help="Latent dims to manipulate")
            ] = None,
            reverse: Annotated[
                bool, typer.Option("--reverse", "-r", help="Flip latent time axis")
            ] = False,
            shuffle_chunks: Annotated[
                int, typer.Option("--shuffle", help="Shuffle latent in chunks of N frames")
            ] = 0,
            quantize_step: Annotated[
                float, typer.Option("--quantize", "-q", help="Snap latent values to grid")
            ] = 0.0,
        ) -> None:
            if not latent_file.exists():
                console.print(f"[red]File not found: {latent_file}[/red]")
                raise typer.Exit(1)

            data = torch.load(latent_file, map_location="cpu", weights_only=True)
            z = data["z"]
            sr = data.get("sample_rate", 44100)
            src_model = data.get("model", "unknown")
            decode_model = model or src_model

            console.print(
                f"[bold]Decoding:[/bold] {latent_file} (encoded with {src_model}) -> {decode_model} "
                f"temp={temperature} noise={noise_amount}"
            )
            result = self.decode_latent(
                z, decode_model, sr,
                temperature=temperature,
                noise=noise_amount,
                dims=dims,
                reverse=reverse,
                shuffle_chunks=shuffle_chunks,
                quantize=quantize_step,
            )
            save_audio(result, output)
            console.print(f"[green]Saved:[/green] {output} ({result.duration:.1f}s)")

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
