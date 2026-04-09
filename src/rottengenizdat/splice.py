from __future__ import annotations

import random

import torch

from rottengenizdat.core import AudioBuffer

DEFAULT_MIN_SECONDS = 0.25
DEFAULT_MAX_SECONDS = 4.0


def splice_buffers(
    buffers: list[AudioBuffer],
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> AudioBuffer:
    """Chop all buffers into random segments, shuffle, reassemble.

    Each buffer is sliced into random-length segments (between min_seconds
    and max_seconds). All segments from all buffers are collected, shuffled,
    and concatenated into a single AudioBuffer. All input samples are used.

    Buffers with different sample rates are resampled to match the first
    buffer's sample rate.

    Args:
        buffers: One or more AudioBuffers to splice together.
        min_seconds: Minimum segment duration in seconds.
        max_seconds: Maximum segment duration in seconds.

    Returns:
        A new AudioBuffer containing all input samples in shuffled segments.
    """
    if not buffers:
        raise ValueError("splice_buffers requires at least one buffer")

    target_sr = buffers[0].sample_rate

    segments: list[torch.Tensor] = []
    for buf in buffers:
        b = buf.resample(target_sr).to_mono()
        samples = b.samples
        num_samples = b.num_samples
        pos = 0
        while pos < num_samples:
            seg_seconds = random.uniform(min_seconds, max_seconds)
            seg_samples = int(seg_seconds * target_sr)
            remaining = num_samples - pos
            if remaining <= seg_samples or (remaining - seg_samples) < int(min_seconds * target_sr):
                segments.append(samples[:, pos:])
                break
            segments.append(samples[:, pos : pos + seg_samples])
            pos += seg_samples

    random.shuffle(segments)
    joined = torch.cat(segments, dim=1)
    return AudioBuffer(samples=joined, sample_rate=target_sr)
