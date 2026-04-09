from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


@dataclass
class AudioBuffer:
    """Audio data container: a tensor of samples plus sample rate."""

    samples: torch.Tensor  # shape: (channels, num_samples)
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    def to_mono(self) -> AudioBuffer:
        if self.channels == 1:
            return self
        mono = self.samples.mean(dim=0, keepdim=True)
        return AudioBuffer(samples=mono, sample_rate=self.sample_rate)

    def resample(self, target_sr: int) -> AudioBuffer:
        if target_sr == self.sample_rate:
            return self
        resampled = torchaudio.functional.resample(
            self.samples, self.sample_rate, target_sr
        )
        return AudioBuffer(samples=resampled, sample_rate=target_sr)

    def as_model_input(self) -> torch.Tensor:
        """Return tensor shaped (1, channels, num_samples) for model input."""
        return self.samples.unsqueeze(0)

    @classmethod
    def from_model_output(cls, tensor: torch.Tensor, sample_rate: int) -> AudioBuffer:
        """Create AudioBuffer from model output tensor (1, channels, num_samples)."""
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        return cls(samples=tensor, sample_rate=sample_rate)


def load_audio(path: Path, target_sr: int | None = None) -> AudioBuffer:
    """Load an audio file and optionally resample.

    Uses soundfile for I/O; torchaudio.functional.resample for resampling.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(
            f"'{path}' is a directory, not an audio file. "
            f"Did you mean: rotten recipe run {path}/<name>.toml ...?"
        )
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (num_samples, channels); transpose to (channels, num_samples)
    samples = torch.from_numpy(data.T)
    buf = AudioBuffer(samples=samples, sample_rate=sr)
    if target_sr is not None:
        buf = buf.resample(target_sr)
    return buf


def save_audio(buf: AudioBuffer, path: Path) -> None:
    """Save an AudioBuffer to a file.

    Uses soundfile for I/O; the format is inferred from the file extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (num_samples, channels); transpose from (channels, num_samples)
    data: np.ndarray = buf.samples.numpy().T
    sf.write(str(path), data, buf.sample_rate)


def concat_buffers(buffers: list[AudioBuffer]) -> AudioBuffer:
    """Concatenate multiple AudioBuffers end-to-end.

    Buffers with different sample rates are resampled to match the first
    buffer's sample rate. All buffers are converted to mono before concatenation.

    Args:
        buffers: Non-empty list of AudioBuffers to concatenate in order.

    Returns:
        A single AudioBuffer containing all inputs joined end-to-end.
    """
    if not buffers:
        raise ValueError("concat_buffers requires at least one buffer")
    target_sr = buffers[0].sample_rate
    parts: list[torch.Tensor] = []
    for buf in buffers:
        b = buf.resample(target_sr).to_mono()
        parts.append(b.samples)
    joined = torch.cat(parts, dim=1)
    return AudioBuffer(samples=joined, sample_rate=target_sr)
