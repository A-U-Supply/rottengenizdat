import torch
import pytest
from pathlib import Path
from rottengenizdat.core import AudioBuffer, load_audio, save_audio


class TestAudioBuffer:
    def test_duration(self, sine_wave: AudioBuffer):
        assert sine_wave.duration == pytest.approx(1.0, abs=0.001)

    def test_channels(self, sine_wave: AudioBuffer):
        assert sine_wave.channels == 1

    def test_channels_stereo(self, stereo_sine: AudioBuffer):
        assert stereo_sine.channels == 2

    def test_num_samples(self, sine_wave: AudioBuffer):
        assert sine_wave.num_samples == 44100

    def test_to_mono(self, stereo_sine: AudioBuffer):
        mono = stereo_sine.to_mono()
        assert mono.channels == 1
        assert mono.num_samples == stereo_sine.num_samples

    def test_resample(self, sine_wave: AudioBuffer):
        resampled = sine_wave.resample(22050)
        assert resampled.sample_rate == 22050
        assert resampled.num_samples == 22050

    def test_as_model_input(self, sine_wave: AudioBuffer):
        """Model input shape is (1, channels, num_samples)."""
        tensor = sine_wave.as_model_input()
        assert tensor.shape == (1, 1, 44100)

    def test_from_model_output(self, sine_wave: AudioBuffer):
        tensor = torch.randn(1, 1, 44100)
        buf = AudioBuffer.from_model_output(tensor, sample_rate=44100)
        assert buf.channels == 1
        assert buf.num_samples == 44100


class TestAudioIO:
    def test_save_and_load_wav(self, sine_wave: AudioBuffer, tmp_path: Path):
        path = tmp_path / "test.wav"
        save_audio(sine_wave, path)
        loaded = load_audio(path)
        assert loaded.sample_rate == sine_wave.sample_rate
        assert loaded.num_samples == sine_wave.num_samples
        assert loaded.channels == sine_wave.channels

    def test_load_with_resample(self, sine_wave: AudioBuffer, tmp_path: Path):
        path = tmp_path / "test.wav"
        save_audio(sine_wave, path)
        loaded = load_audio(path, target_sr=22050)
        assert loaded.sample_rate == 22050

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_audio(Path("/nonexistent/audio.wav"))
