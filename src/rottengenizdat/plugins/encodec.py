"""EnCodec plugin — neural audio codec compression artifacts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect

console = Console()

_ENCODEC_HELP = """\
EnCodec — Meta's neural audio codec.

Encodes audio into a compressed token representation, then reconstructs
it. Lower bandwidth = fewer codebooks = more destroyed. But the real fun
is in partial destruction: encode at high quality, then randomly drop
codebooks before decoding.

Each codebook captures a different aspect of the audio — dropping them
creates artifacts from subtle grain to full disintegration.

KNOBS:

  --bandwidth BW       Target bandwidth: 1.5, 3.0, 6.0, 12.0, 24.0
                       Lower = more artifacts. 1.5 = 2 codebooks (robot),
                       24.0 = 32 codebooks (transparent). Default 6.0.

  --drop-ratio R       Randomly drop this fraction of codebooks (0.0–1.0).
                       Encode at full quality, then destroy. 0.3 = subtle
                       grain, 0.7 = crunchy, 0.95 = obliterated.

  --keep-first N       Keep only the first N codebooks, drop the rest.
                       Lower codebooks = structure, higher = detail.

  --scramble           Randomize codebook order. Same information, wrong
                       arrangement. Creates spectral smearing.

MODELS auto-download from Facebook (~89MB) via torch hub on first use.
Cache at ~/.cache/torch/hub/checkpoints/.


EXAMPLES:

  Crushed lo-fi (minimum bandwidth):
    rotten encodec input.wav --bandwidth 1.5 -o crushed.wav

  Half the codebooks randomly dropped:
    rotten encodec input.wav --bandwidth 24 --drop-ratio 0.5 -o half.wav

  Keep only structure, drop all detail:
    rotten encodec input.wav --bandwidth 24 --keep-first 4 -o structure.wav

  Scrambled codebooks:
    rotten encodec input.wav --bandwidth 24 --scramble -o scrambled.wav

  Chain with RAVE for double neural mangling:
    rotten chain input.wav "encodec --bandwidth 3.0" "rave -m percussion -t 1.2" -o crushed-drums.wav
"""


def _load_codec(bandwidth: float = 6.0) -> "EncodecModel":
    """Load the EnCodec model, downloading if needed."""
    import encodec

    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    return model


class EncodecEffect(AudioEffect):
    """EnCodec — neural audio codec compression."""

    name = "encodec"
    description = _ENCODEC_HELP

    def process(
        self,
        audio: AudioBuffer,
        bandwidth: float = 6.0,
        drop_ratio: float = 0.0,
        keep_first: int = 0,
        scramble: bool = False,
        **kwargs,
    ) -> AudioBuffer:
        """Process audio through EnCodec with optional codebook destruction.

        Args:
            audio: Input audio buffer.
            bandwidth: Target bandwidth (1.5, 3.0, 6.0, 12.0, 24.0).
            drop_ratio: Randomly drop this fraction of codebooks (0.0–1.0).
            keep_first: Keep only first N codebooks (0 = keep all).
            scramble: Randomize codebook order.
        """
        import encodec

        mono = audio.to_mono()

        # EnCodec works at 24kHz — resample if needed
        target_sr = 24000
        if audio.sample_rate != target_sr:
            mono = mono.resample(target_sr)

        # Always encode at max bandwidth to get all codebooks for manipulation,
        # then decode at the requested bandwidth. But if bandwidth is the only
        # knob (no drop/keep/scramble), just use it directly.
        encode_bw = max(bandwidth, 24.0) if (drop_ratio > 0 or keep_first > 0 or scramble) else bandwidth

        model = _load_codec(encode_bw)

        with torch.no_grad():
            x = mono.as_model_input()  # (1, 1, samples)

            result = model.encode(x)
            frames, scale = result[0]  # frames: (1, n_codebooks, n_frames)

            # --- Codebook destruction ---
            n_codebooks = frames.shape[1]

            if keep_first > 0 and keep_first < n_codebooks:
                # Zero out codebooks beyond keep_first
                drop_mask = torch.zeros(n_codebooks, dtype=torch.bool)
                drop_mask[keep_first:] = True
                frames = frames[:, ~drop_mask, :]

            if drop_ratio > 0 and frames.shape[1] > 1:
                # Randomly drop codebooks
                n_keep = max(1, int(frames.shape[1] * (1.0 - drop_ratio)))
                keep_idx = sorted(random.sample(range(frames.shape[1]), n_keep))
                frames = frames[:, keep_idx, :]

            if scramble and frames.shape[1] > 1:
                # Shuffle codebook order
                perm = torch.randperm(frames.shape[1])
                frames = frames[:, perm, :]

            # Decode at the original bandwidth level if only using bandwidth knob
            if drop_ratio == 0 and keep_first == 0 and not scramble:
                model.set_target_bandwidth(bandwidth)

            decoded = model.decode([(frames, None)])

            # Trim to original length (encode/decode may change it slightly)
            decoded = decoded[:, :, :mono.num_samples]

        buf = AudioBuffer.from_model_output(decoded, sample_rate=target_sr)

        # Resample back to original rate
        if audio.sample_rate != target_sr:
            buf = buf.resample(audio.sample_rate)

        return buf

    def register_command(self, app: typer.Typer) -> None:
        """Register the encodec subcommand."""

        @app.command(name=self.name, help=self.description)
        def encodec_command(
            input_files: Annotated[
                Optional[list[Path]],
                typer.Argument(help="Input audio file(s) (wav, flac, mp3, etc.)"),
            ] = None,
            output: Annotated[
                Path,
                typer.Option("--output", "-o", help="Output file path"),
            ] = Path("output.wav"),
            bandwidth: Annotated[
                float,
                typer.Option(
                    "--bandwidth",
                    "-b",
                    help="Target bandwidth: 1.5 (crushed), 3.0, 6.0, 12.0, 24.0 (clean)",
                ),
            ] = 6.0,
            drop_ratio: Annotated[
                float,
                typer.Option(
                    "--drop-ratio",
                    help="Randomly drop this fraction of codebooks (0.0–1.0). Uses max bandwidth encode",
                ),
            ] = 0.0,
            keep_first: Annotated[
                int,
                typer.Option(
                    "--keep-first",
                    help="Keep only the first N codebooks, drop the rest. Lower = structure, higher = detail",
                ),
            ] = 0,
            scramble: Annotated[
                bool,
                typer.Option(
                    "--scramble",
                    help="Randomize codebook order for spectral smearing",
                ),
            ] = False,
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
                    f"[bold]Compressing:[/bold] bw={bandwidth} drop={drop_ratio} "
                    f"keep_first={keep_first} scramble={scramble}"
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Encoding + mangling + decoding...", total=None)
                    result = self.process(
                        audio,
                        bandwidth=bandwidth,
                        drop_ratio=drop_ratio,
                        keep_first=keep_first,
                        scramble=scramble,
                    )
                save_audio(result, out_path)
                console.print(f"[green]Saved:[/green] {out_path}")

        return encodec_command
