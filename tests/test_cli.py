from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf
import torch
from typer.testing import CliRunner

from rottengenizdat.cli import app

runner = CliRunner()


class TestCLI:
    def test_help_shows_banner(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "rottengenizdat" in result.output.lower() or "rotten" in result.output.lower()

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app)
        assert result.exit_code == 0


class TestRaveSubcommand:
    def test_rave_appears_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "rave" in result.output.lower()

    @patch("rottengenizdat.plugins.rave.load_rave_model")
    def test_rave_processes_file(self, mock_load, tmp_path):
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            side_effect=lambda x: torch.randn(1, 16, x.shape[-1] // 2048)
        )
        mock_model.decode = MagicMock(
            side_effect=lambda z: torch.randn(1, 1, z.shape[-1] * 2048)
        )
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_load.return_value = mock_model

        sr = 44100
        samples_np = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))
        input_path = tmp_path / "test_input.wav"
        sf.write(str(input_path), samples_np, sr)

        output_path = tmp_path / "test_output.wav"
        result = runner.invoke(app, [
            "rave",
            str(input_path),
            "-o", str(output_path),
            "-m", "percussion",
            "-t", "1.5",
        ])
        assert result.exit_code == 0
        assert output_path.exists()

    @patch("rottengenizdat.plugins.rave.load_rave_model")
    def test_rave_sweep(self, mock_load, tmp_path):
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            side_effect=lambda x: torch.randn(1, 16, x.shape[-1] // 2048)
        )
        mock_model.decode = MagicMock(
            side_effect=lambda z: torch.randn(1, 1, z.shape[-1] * 2048)
        )
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_load.return_value = mock_model

        sr = 44100
        samples_np = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))
        input_path = tmp_path / "test_input.wav"
        sf.write(str(input_path), samples_np, sr)

        output_dir = tmp_path / "grid"
        result = runner.invoke(app, [
            "rave",
            str(input_path),
            "-o", str(output_dir),
            "-m", "percussion",
            "--sweep", "temperature=0.5,1.0,1.5",
        ])
        assert result.exit_code == 0
        assert (output_dir / "temperature_0.50.wav").exists()
        assert (output_dir / "temperature_1.00.wav").exists()
        assert (output_dir / "temperature_1.50.wav").exists()
