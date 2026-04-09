import torch
import pytest
from rottengenizdat.core import AudioBuffer


@pytest.fixture
def sine_wave() -> AudioBuffer:
    """A 1-second 440Hz sine wave at 44100Hz sample rate."""
    sr = 44100
    t = torch.linspace(0, 1, sr)
    samples = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # (1, 44100)
    return AudioBuffer(samples=samples, sample_rate=sr)


@pytest.fixture
def stereo_sine() -> AudioBuffer:
    """A 1-second stereo sine wave (440Hz left, 880Hz right)."""
    sr = 44100
    t = torch.linspace(0, 1, sr)
    left = torch.sin(2 * torch.pi * 440 * t)
    right = torch.sin(2 * torch.pi * 880 * t)
    samples = torch.stack([left, right])  # (2, 44100)
    return AudioBuffer(samples=samples, sample_rate=sr)
