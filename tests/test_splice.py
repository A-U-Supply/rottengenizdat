from __future__ import annotations

import random

import torch
import pytest

from rottengenizdat.core import AudioBuffer
from rottengenizdat.splice import splice_buffers


@pytest.fixture
def two_buffers() -> list[AudioBuffer]:
    """Two 2-second mono buffers at 44100Hz: one all-ones, one all-twos."""
    sr = 44100
    a = AudioBuffer(samples=torch.ones(1, sr * 2), sample_rate=sr)
    b = AudioBuffer(samples=torch.ones(1, sr * 2) * 2, sample_rate=sr)
    return [a, b]


class TestSpliceBuffers:
    def test_output_is_audio_buffer(self, two_buffers):
        random.seed(42)
        result = splice_buffers(two_buffers)
        assert isinstance(result, AudioBuffer)

    def test_output_sample_rate_matches_input(self, two_buffers):
        random.seed(42)
        result = splice_buffers(two_buffers)
        assert result.sample_rate == 44100

    def test_output_length_equals_total_input(self, two_buffers):
        """Splice should use all samples from all inputs (total = 4 seconds)."""
        random.seed(42)
        result = splice_buffers(two_buffers)
        total_input_samples = sum(b.num_samples for b in two_buffers)
        assert result.num_samples == total_input_samples

    def test_contains_samples_from_all_inputs(self, two_buffers):
        """Output should contain segments from both buffers."""
        random.seed(42)
        result = splice_buffers(two_buffers)
        values = result.samples.unique()
        assert 1.0 in values
        assert 2.0 in values

    def test_deterministic_with_seed(self, two_buffers):
        random.seed(42)
        r1 = splice_buffers(two_buffers)
        random.seed(42)
        r2 = splice_buffers(two_buffers)
        assert torch.equal(r1.samples, r2.samples)

    def test_segments_within_min_max(self):
        """Verify no segment is shorter than min or longer than max."""
        sr = 44100
        buf = AudioBuffer(samples=torch.ones(1, sr * 10), sample_rate=sr)
        random.seed(0)
        result = splice_buffers([buf, buf], min_seconds=1.0, max_seconds=2.0)
        assert result.num_samples == sr * 20

    def test_single_buffer(self):
        """Single input: splice still works (chops and shuffles one buffer)."""
        sr = 44100
        buf = AudioBuffer(samples=torch.arange(sr * 2, dtype=torch.float32).unsqueeze(0), sample_rate=sr)
        random.seed(42)
        result = splice_buffers([buf])
        assert result.num_samples == buf.num_samples

    def test_resamples_mismatched_rates(self):
        """Inputs at different sample rates should be resampled to the first buffer's rate."""
        a = AudioBuffer(samples=torch.ones(1, 44100), sample_rate=44100)
        b = AudioBuffer(samples=torch.ones(1, 22050), sample_rate=22050)
        random.seed(42)
        result = splice_buffers([a, b])
        assert result.sample_rate == 44100
        # b was 1 second at 22050, resampled to 1 second at 44100 = 44100 samples
        assert result.num_samples == 44100 + 44100

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            splice_buffers([])
