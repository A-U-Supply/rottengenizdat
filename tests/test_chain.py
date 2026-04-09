from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch

from rottengenizdat.core import AudioBuffer
from rottengenizdat.chain import mix_buffers, parse_step, run_branch, run_chain


class TestParseStep:
    def test_basic(self):
        name, kwargs = parse_step("rave -m percussion")
        assert name == "rave"
        assert kwargs["model_name"] == "percussion"

    def test_multiple_flags(self):
        name, kwargs = parse_step("rave -m vintage -t 1.5 -n 0.3")
        assert name == "rave"
        assert kwargs["model_name"] == "vintage"
        assert kwargs["temperature"] == 1.5
        assert kwargs["noise"] == 0.3

    def test_long_flags(self):
        name, kwargs = parse_step("rave --model percussion --temperature 2.0")
        assert name == "rave"
        assert kwargs["model_name"] == "percussion"
        assert kwargs["temperature"] == 2.0

    def test_dims(self):
        name, kwargs = parse_step("rave -m percussion -d 0,3,7")
        assert kwargs["dims"] == "0,3,7"

    def test_reverse_flag(self):
        name, kwargs = parse_step("rave -m percussion -r")
        assert kwargs["reverse"] is True

    def test_no_reverse_by_default(self):
        name, kwargs = parse_step("rave -m percussion")
        assert kwargs.get("reverse", False) is False

    def test_mix_flag(self):
        name, kwargs = parse_step("rave -m percussion -w 0.5")
        assert kwargs["mix"] == 0.5

    def test_noise_flag(self):
        name, kwargs = parse_step("rave -m percussion -n 0.1")
        assert kwargs["noise"] == pytest.approx(0.1)

    def test_shuffle_flag(self):
        name, kwargs = parse_step("rave -m percussion --shuffle 4")
        assert kwargs["shuffle_chunks"] == 4

    def test_quantize_flag(self):
        name, kwargs = parse_step("rave -m percussion -q 0.25")
        assert kwargs["quantize"] == pytest.approx(0.25)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_step("")

    def test_unknown_flag_raises(self):
        with pytest.raises(ValueError, match="Unknown flag"):
            parse_step("rave --bogus thing")


class TestMixBuffers:
    def test_averages_two_buffers(self):
        a = AudioBuffer(samples=torch.ones(1, 100), sample_rate=44100)
        b = AudioBuffer(samples=torch.ones(1, 100) * 3, sample_rate=44100)
        mixed = mix_buffers([a, b])
        assert torch.allclose(mixed.samples, torch.ones(1, 100) * 2)

    def test_handles_different_lengths(self):
        a = AudioBuffer(samples=torch.ones(1, 100), sample_rate=44100)
        b = AudioBuffer(samples=torch.ones(1, 80) * 3, sample_rate=44100)
        mixed = mix_buffers([a, b])
        assert mixed.num_samples == 80

    def test_single_buffer_passthrough(self):
        a = AudioBuffer(samples=torch.ones(1, 50) * 7, sample_rate=44100)
        mixed = mix_buffers([a])
        assert torch.allclose(mixed.samples, a.samples)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mix_buffers([])

    def test_preserves_sample_rate(self):
        a = AudioBuffer(samples=torch.ones(1, 10), sample_rate=22050)
        b = AudioBuffer(samples=torch.ones(1, 10), sample_rate=22050)
        mixed = mix_buffers([a, b])
        assert mixed.sample_rate == 22050


class TestRunChain:
    @patch("rottengenizdat.chain.discover_plugins")
    def test_sequential_chain(self, mock_discover, sine_wave):
        # Each step doubles amplitude; two steps → 4×
        fake_cls = MagicMock()
        fake_instance = MagicMock()
        fake_instance.process = MagicMock(
            side_effect=lambda audio, **kw: AudioBuffer(
                samples=audio.samples * 2, sample_rate=audio.sample_rate
            )
        )
        fake_cls.return_value = fake_instance
        mock_discover.return_value = {"rave": fake_cls}

        result = run_chain(sine_wave, ["rave -m percussion", "rave -m vintage"])
        assert torch.allclose(result.samples, sine_wave.samples * 4)

    @patch("rottengenizdat.chain.discover_plugins")
    def test_single_step(self, mock_discover, sine_wave):
        fake_cls = MagicMock()
        fake_instance = MagicMock()
        fake_instance.process = MagicMock(
            side_effect=lambda audio, **kw: AudioBuffer(
                samples=audio.samples * 3, sample_rate=audio.sample_rate
            )
        )
        fake_cls.return_value = fake_instance
        mock_discover.return_value = {"rave": fake_cls}

        result = run_chain(sine_wave, ["rave -m percussion"])
        assert torch.allclose(result.samples, sine_wave.samples * 3)

    @patch("rottengenizdat.chain.discover_plugins")
    def test_unknown_effect_raises(self, mock_discover, sine_wave):
        mock_discover.return_value = {}
        with pytest.raises(ValueError, match="Unknown effect"):
            run_chain(sine_wave, ["bogus -m x"])


class TestRunBranch:
    @patch("rottengenizdat.chain.discover_plugins")
    def test_branch_averages(self, mock_discover, sine_wave):
        fake_cls = MagicMock()
        call_count = [0]

        def process_fn(audio, **kw):
            call_count[0] += 1
            return AudioBuffer(
                samples=audio.samples * call_count[0], sample_rate=audio.sample_rate
            )

        fake_instance = MagicMock()
        fake_instance.process = MagicMock(side_effect=process_fn)
        fake_cls.return_value = fake_instance
        mock_discover.return_value = {"rave": fake_cls}

        result = run_branch(sine_wave, ["rave -m a", "rave -m b"])
        # Branch 1: ×1, Branch 2: ×2 → average ×1.5
        expected = sine_wave.samples * 1.5
        assert torch.allclose(result.samples, expected)

    @patch("rottengenizdat.chain.discover_plugins")
    def test_branch_uses_original_input(self, mock_discover, sine_wave):
        """Each branch must receive the ORIGINAL audio, not the previous branch's output."""
        fake_cls = MagicMock()
        received_inputs = []

        def process_fn(audio, **kw):
            received_inputs.append(audio.samples.clone())
            return AudioBuffer(samples=audio.samples, sample_rate=audio.sample_rate)

        fake_instance = MagicMock()
        fake_instance.process = MagicMock(side_effect=process_fn)
        fake_cls.return_value = fake_instance
        mock_discover.return_value = {"rave": fake_cls}

        run_branch(sine_wave, ["rave -m a", "rave -m b"])
        assert len(received_inputs) == 2
        assert torch.allclose(received_inputs[0], received_inputs[1])

    @patch("rottengenizdat.chain.discover_plugins")
    def test_unknown_effect_raises(self, mock_discover, sine_wave):
        mock_discover.return_value = {}
        with pytest.raises(ValueError, match="Unknown effect"):
            run_branch(sine_wave, ["bogus -m x"])
