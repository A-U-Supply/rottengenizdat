from __future__ import annotations

import random
from pathlib import Path

import torch
import pytest
from typer.testing import CliRunner

from rottengenizdat.cli import app
from rottengenizdat.core import AudioBuffer, save_audio
from rottengenizdat.inputs import combine_inputs, InputMode

runner = CliRunner()


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


class TestRecipeRunMultiInput:
    def test_multiple_local_files(self, tmp_path: Path):
        """recipe run accepts multiple input files."""
        recipe = tmp_path / "test.toml"
        recipe.write_text(
            '[recipe]\nname = "test"\nmode = "sequential"\n'
            '[[steps]]\neffect = "dry"\n'
        )
        sr = 44100
        for name in ["a.wav", "b.wav"]:
            buf = AudioBuffer(samples=torch.randn(1, sr), sample_rate=sr)
            save_audio(buf, tmp_path / name)

        result = runner.invoke(app, [
            "recipe", "run", str(recipe),
            str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
            "--mode", "concat",
            "-o", str(tmp_path / "out.wav"),
        ])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "out.wav").exists()

    def test_independent_mode_creates_directory(self, tmp_path: Path):
        recipe = tmp_path / "test.toml"
        recipe.write_text(
            '[recipe]\nname = "test"\nmode = "sequential"\n'
            '[[steps]]\neffect = "dry"\n'
        )
        sr = 44100
        for name in ["a.wav", "b.wav"]:
            buf = AudioBuffer(samples=torch.randn(1, sr), sample_rate=sr)
            save_audio(buf, tmp_path / name)

        out_dir = tmp_path / "outputs"
        result = runner.invoke(app, [
            "recipe", "run", str(recipe),
            str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
            "--mode", "independent",
            "-o", str(out_dir),
        ])
        assert result.exit_code == 0, result.stdout
        assert out_dir.is_dir()
        output_files = list(out_dir.glob("*.wav"))
        assert len(output_files) == 2


class TestPluginMultiInput:
    def test_dry_multiple_local_files(self, tmp_path: Path):
        sr = 44100
        for name in ["a.wav", "b.wav"]:
            buf = AudioBuffer(samples=torch.randn(1, sr), sample_rate=sr)
            save_audio(buf, tmp_path / name)

        result = runner.invoke(app, [
            "dry",
            str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
            "--mode", "concat",
            "-o", str(tmp_path / "out.wav"),
        ])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "out.wav").exists()


class TestChainMultiInput:
    def test_multiple_local_files_concat(self, tmp_path: Path):
        sr = 44100
        for name in ["a.wav", "b.wav"]:
            buf = AudioBuffer(samples=torch.randn(1, sr), sample_rate=sr)
            save_audio(buf, tmp_path / name)

        result = runner.invoke(app, [
            "chain",
            str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
            "--",
            "dry",
            "--mode", "concat",
            "-o", str(tmp_path / "out.wav"),
        ])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "out.wav").exists()


class TestIntegrationMultiInput:
    def test_splice_three_files_through_dry(self, tmp_path: Path):
        """End-to-end: three files spliced and run through dry recipe."""
        sr = 44100
        for i, name in enumerate(["x.wav", "y.wav", "z.wav"]):
            buf = AudioBuffer(
                samples=torch.ones(1, sr) * (i + 1), sample_rate=sr
            )
            save_audio(buf, tmp_path / name)

        recipe = tmp_path / "dry.toml"
        recipe.write_text(
            '[recipe]\nname = "dry"\nmode = "sequential"\n'
            '[[steps]]\neffect = "dry"\n'
        )

        result = runner.invoke(app, [
            "recipe", "run", str(recipe),
            str(tmp_path / "x.wav"),
            str(tmp_path / "y.wav"),
            str(tmp_path / "z.wav"),
            "--mode", "splice",
            "--splice-min", "0.1",
            "--splice-max", "0.5",
            "-o", str(tmp_path / "spliced.wav"),
        ])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "spliced.wav").exists()
        # Output should contain all input samples (3 seconds total)
        from rottengenizdat.core import load_audio as _load
        out = _load(tmp_path / "spliced.wav")
        assert out.num_samples == sr * 3
