"""VampNet plugin — masked acoustic token modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.console import Console

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect

console = Console()

_VAMP_HELP = """\
VampNet — masked acoustic token modeling (audio mad-libs).

Encodes audio into EnCodec tokens, randomly masks portions, and has a
transformer hallucinate the gaps. The mask ratio is the perfect "how much
to mangle" knob: 0.1 = subtle variation, 0.5 = half replaced, 0.9 = fever
dream.

Periodic masking creates rhythmic patterns — keep every Nth token unmasked
for a gated/stutter effect. Combined with random masking for texture.

KNOBS:

  --rand-intensity R   Random mask ratio (0.0–1.0). 0.1 = subtle,
                       0.5 = half, 0.9 = hallucinated. (default 0.5)

  --periodic-p N       Keep every Nth frame unmasked. Creates rhythmic
                       gating. 1 = off, 4–8 = stutter, 2–3 = heavy chop.

  --prefix-s S         Seconds of the original to keep unmasked at the
                       start. Acts as an anchor/seed for the hallucination.

  --suffix-s S         Seconds of the original to keep unmasked at the end.

  --upper-mask N       Mask N upper codebooks entirely. Preserves structural
                       codebooks while hallucinating detail. 0 = none.

  -t, --temperature    Sampling temperature for token prediction. 1.0 = normal,
                       0.8 = conservative, 1.5 = wild. (default 1.0)

  --feedback-steps N   Iterative refinement passes. More = cleaner output,
                       fewer = glitchier. (default 1)

  --typical / --no-typical
                       Use typical filtering instead of top-k sampling.
                       Typically produces more coherent output. (default on)


MODELS download automatically from HuggingFace (hugggof/vampnet, ~400MB)
and cache at ./models/vampnet/.


EXAMPLES:

  Half hallucinated drum pattern:
    rotten vamp input.wav --rand-intensity 0.5 -t 1.1 -o half.wav

  Heavy glitch with periodic gating (keep every 4th frame):
    rotten vamp input.wav --rand-intensity 0.9 --periodic-p 4 -t 1.3 -o glitch.wav

  Gentle variation with seed prefix:
    rotten vamp input.wav --rand-intensity 0.2 --prefix-s 0.5 -t 0.9 -o gentle.wav

  Three refinement passes (cleaner, slower):
    rotten vamp input.wav --rand-intensity 0.7 --feedback-steps 3 -t 1.0 -o refined.wav
"""


def _ensure_models() -> tuple[Path, Path, Path]:
    """Download and return paths to VampNet model checkpoints.

    Returns (codec_path, coarse_path, c2f_path).
    """
    try:
        from vampnet import download_codec, download_default
    except ImportError:
        console.print(
            "[red]VampNet is not installed.[/red]\n"
            "  Install with: [bold]uv pip install git+https://github.com/hugofloresgarcia/vampnet.git[/bold]"
        )
        raise typer.Exit(1)

    return download_codec(), *download_default()


def _load_iface() -> "Interface":
    """Load the VampNet interface without beat tracker (not needed for masking)."""
    from vampnet.interface import Interface

    codec_path, coarse_path, c2f_path = _ensure_models()
    return Interface(
        coarse_ckpt=coarse_path,
        coarse2fine_ckpt=c2f_path,
        codec_ckpt=codec_path,
        wavebeat_ckpt=None,
    )


class VampEffect(AudioEffect):
    """VampNet — masked acoustic token modeling."""

    name = "vamp"
    description = _VAMP_HELP

    def process(
        self,
        audio: AudioBuffer,
        rand_intensity: float = 0.5,
        periodic_prompt: int = 0,
        prefix_s: float = 0.0,
        suffix_s: float = 0.0,
        upper_codebook_mask: int = 0,
        temperature: float = 1.0,
        feedback_steps: int = 1,
        typical_filtering: bool = True,
        **kwargs,
    ) -> AudioBuffer:
        """Process audio through VampNet.

        Args:
            audio: Input audio buffer.
            rand_intensity: Random mask ratio (0.0–1.0).
            periodic_prompt: Keep every Nth frame unmasked (0 = off).
            prefix_s: Unmasked seconds at the start.
            suffix_s: Unmasked seconds at the end.
            upper_codebook_mask: Number of upper codebooks to mask entirely.
            temperature: Sampling temperature for the transformer.
            feedback_steps: Number of iterative refinement passes.
            typical_filtering: Use typical filtering for sampling.
        """
        import audiotools as at

        iface = _load_iface()

        # Convert AudioBuffer → AudioSignal
        samples = audio.to_mono().samples
        sig = at.AudioSignal(samples, audio.sample_rate)

        with torch.no_grad():
            codes = iface.encode(sig)
            mask = iface.build_mask(
                codes,
                sig,
                rand_mask_intensity=rand_intensity,
                prefix_s=prefix_s,
                suffix_s=suffix_s,
                periodic_prompt=periodic_prompt if periodic_prompt > 0 else 1,
                upper_codebook_mask=upper_codebook_mask,
            )
            output_codes = iface.vamp(
                codes,
                mask,
                return_mask=False,
                temperature=temperature,
                feedback_steps=feedback_steps,
                typical_filtering=typical_filtering,
            )
            out_sig = iface.decode(output_codes)

        # AudioSignal.audio_data is (batch, channels, samples) — AudioBuffer
        # expects (channels, samples), so squeeze the batch dim.
        samples = out_sig.audio_data
        if samples.dim() == 3:
            samples = samples.squeeze(0)

        return AudioBuffer(
            samples=samples,
            sample_rate=out_sig.sample_rate,
        )

    def register_command(self, app: typer.Typer) -> None:
        """Register the vamp subcommand."""

        @app.command(name=self.name, help=self.description)
        def vamp_command(
            input_files: Annotated[
                Optional[list[Path]],
                typer.Argument(help="Input audio file(s) (wav, flac, mp3, etc.)"),
            ] = None,
            output: Annotated[
                Path,
                typer.Option("--output", "-o", help="Output file path"),
            ] = Path("output.wav"),
            rand_intensity: Annotated[
                float,
                typer.Option(
                    "--rand-intensity",
                    help="Random mask ratio (0.0–1.0). 0.1 = subtle, 0.5 = half, 0.9 = hallucinated",
                ),
            ] = 0.5,
            periodic_prompt: Annotated[
                int,
                typer.Option(
                    "--periodic-p",
                    help="Keep every Nth frame unmasked. 1 = off, 4-8 = stutter, 2-3 = heavy chop",
                ),
            ] = 1,
            prefix_s: Annotated[
                float,
                typer.Option(
                    "--prefix-s",
                    help="Seconds of original to keep unmasked at the start (seed anchor)",
                ),
            ] = 0.0,
            suffix_s: Annotated[
                float,
                typer.Option(
                    "--suffix-s",
                    help="Seconds of original to keep unmasked at the end",
                ),
            ] = 0.0,
            upper_codebook_mask: Annotated[
                int,
                typer.Option(
                    "--upper-mask",
                    help="Mask N upper codebooks entirely (0 = none, preserves structure)",
                ),
            ] = 0,
            temperature: Annotated[
                float,
                typer.Option(
                    "--temperature",
                    "-t",
                    help="Sampling temperature. 1.0 = normal, 0.8 = conservative, 1.5 = wild",
                ),
            ] = 1.0,
            feedback_steps: Annotated[
                int,
                typer.Option(
                    "--feedback-steps",
                    help="Iterative refinement passes. More = cleaner, fewer = glitchier",
                ),
            ] = 1,
            typical_filtering: Annotated[
                bool,
                typer.Option(
                    "--typical / --no-typical",
                    help="Use typical filtering for more coherent output",
                ),
            ] = True,
            sample_sale: Annotated[
                bool,
                typer.Option("--sample-sale", "-ss", help="Include random samples from #sample-sale"),
            ] = False,
            sample_sale_count: Annotated[
                int,
                typer.Option("--sample-sale-count", help="How many samples to pull (implies --sample-sale)"),
            ] = 0,
        ) -> None:
            all_buffers: list[AudioBuffer] = []

            for f in (input_files or []):
                if not f.exists():
                    console.print(f"[red]File not found: {f}[/red]")
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

            if not all_buffers:
                console.print("[red]No input files provided. Pass audio files or use --sample-sale.[/red]")
                raise typer.Exit(1)

            for i, audio in enumerate(all_buffers):
                if len(all_buffers) > 1:
                    out_path = output.parent / f"{output.stem}-{i+1:03d}{output.suffix}"
                else:
                    out_path = output

                console.print(
                    f"[bold]Vamping:[/bold] rand={rand_intensity} periodic={periodic_prompt} "
                    f"prefix={prefix_s}s temp={temperature} feedback={feedback_steps} "
                    f"typical={typical_filtering} upper_mask={upper_codebook_mask}"
                )
                result = self.process(
                    audio,
                    rand_intensity=rand_intensity,
                    periodic_prompt=periodic_prompt,
                    prefix_s=prefix_s,
                    suffix_s=suffix_s,
                    upper_codebook_mask=upper_codebook_mask,
                    temperature=temperature,
                    feedback_steps=feedback_steps,
                    typical_filtering=typical_filtering,
                )
                save_audio(result, out_path)
                console.print(f"[green]Saved:[/green] {out_path}")

        return vamp_command
