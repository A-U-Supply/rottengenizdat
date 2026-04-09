import torch
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rottengenizdat.core import AudioBuffer
from rottengenizdat.plugins.rave import RaveEffect, AVAILABLE_MODELS, download_model


class TestRaveEffect:
    @pytest.fixture
    def mock_model(self):
        """A mock TorchScript RAVE model."""
        model = MagicMock()
        model.encode = MagicMock(
            side_effect=lambda x: torch.randn(1, 16, x.shape[-1] // 2048)
        )
        model.decode = MagicMock(
            side_effect=lambda z: torch.randn(1, 1, z.shape[-1] * 2048)
        )
        model.eval = MagicMock(return_value=model)
        return model

    @patch("rottengenizdat.plugins.rave.load_rave_model")
    def test_process_roundtrip(self, mock_load, mock_model, sine_wave: AudioBuffer):
        mock_load.return_value = mock_model
        effect = RaveEffect()
        result = effect.process(sine_wave, model_name="percussion")
        assert isinstance(result, AudioBuffer)
        assert result.sample_rate == sine_wave.sample_rate
        mock_model.encode.assert_called_once()
        mock_model.decode.assert_called_once()

    @patch("rottengenizdat.plugins.rave.load_rave_model")
    def test_temperature_scales_latents(self, mock_load, mock_model, sine_wave: AudioBuffer):
        mock_load.return_value = mock_model
        decoded_z = []
        def capture_decode(z):
            decoded_z.append(z.clone())
            return torch.randn(1, 1, z.shape[-1] * 2048)
        mock_model.decode = MagicMock(side_effect=capture_decode)

        effect = RaveEffect()
        effect.process(sine_wave, model_name="percussion", temperature=2.0)
        assert len(decoded_z) == 1

    @patch("rottengenizdat.plugins.rave.load_rave_model")
    def test_latent_noise_adds_noise(self, mock_load, mock_model, sine_wave: AudioBuffer):
        mock_load.return_value = mock_model
        effect = RaveEffect()
        result = effect.process(sine_wave, model_name="percussion", noise=0.5)
        assert isinstance(result, AudioBuffer)


class TestAvailableModels:
    def test_known_models_listed(self):
        assert "percussion" in AVAILABLE_MODELS
        assert "vintage" in AVAILABLE_MODELS
        assert "nasa" in AVAILABLE_MODELS
        assert "VCTK" in AVAILABLE_MODELS


class TestModelDownload:
    @patch("rottengenizdat.plugins.rave.requests.get")
    def test_download_caches_model(self, mock_get, tmp_path: Path):
        mock_response = MagicMock()
        mock_response.content = b"fake model data"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        path = download_model("percussion", cache_dir=tmp_path)
        assert path.exists()
        assert path.name == "percussion.ts"

        mock_get.reset_mock()
        path2 = download_model("percussion", cache_dir=tmp_path)
        mock_get.assert_not_called()
        assert path == path2

    def test_download_unknown_model_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown model"):
            download_model("nonexistent_model", cache_dir=tmp_path)
