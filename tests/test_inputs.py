from __future__ import annotations

import random

import torch
import pytest

from rottengenizdat.core import AudioBuffer
from rottengenizdat.inputs import combine_inputs, InputMode


@pytest.fixture
def three_buffers() -> list[AudioBuffer]:
    sr = 44100
    return [
        AudioBuffer(samples=torch.ones(1, sr) * (i + 1), sample_rate=sr)
        for i in range(3)
    ]


class TestInputMode:
    def test_default_single_input(self, three_buffers):
        """Single input with no explicit mode passes through."""
        mode = InputMode.resolve(None, 1)
        assert mode == InputMode.PASSTHROUGH

    def test_default_multi_input(self, three_buffers):
        """Multiple inputs with no explicit mode defaults to splice."""
        mode = InputMode.resolve(None, 3)
        assert mode == InputMode.SPLICE

    def test_explicit_concat(self):
        mode = InputMode.resolve("concat", 3)
        assert mode == InputMode.CONCAT

    def test_explicit_independent(self):
        mode = InputMode.resolve("independent", 3)
        assert mode == InputMode.INDEPENDENT

    def test_explicit_splice(self):
        mode = InputMode.resolve("splice", 2)
        assert mode == InputMode.SPLICE


class TestCombineInputs:
    def test_passthrough(self, three_buffers):
        results = combine_inputs([three_buffers[0]], InputMode.PASSTHROUGH)
        assert len(results) == 1
        assert torch.equal(results[0].samples, three_buffers[0].samples)

    def test_concat(self, three_buffers):
        results = combine_inputs(three_buffers, InputMode.CONCAT)
        assert len(results) == 1
        assert results[0].num_samples == 44100 * 3

    def test_independent(self, three_buffers):
        results = combine_inputs(three_buffers, InputMode.INDEPENDENT)
        assert len(results) == 3
        for i, buf in enumerate(results):
            expected_val = float(i + 1)
            assert torch.allclose(buf.samples, torch.ones(1, 44100) * expected_val)

    def test_splice(self, three_buffers):
        random.seed(42)
        results = combine_inputs(three_buffers, InputMode.SPLICE)
        assert len(results) == 1
        assert results[0].num_samples == 44100 * 3

    def test_splice_with_params(self, three_buffers):
        random.seed(42)
        results = combine_inputs(
            three_buffers, InputMode.SPLICE, splice_min=0.5, splice_max=1.0
        )
        assert len(results) == 1
        assert results[0].num_samples == 44100 * 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No input"):
            combine_inputs([], InputMode.PASSTHROUGH)
