import torch
import pytest
from rottengenizdat.core import AudioBuffer
from rottengenizdat.plugin import AudioEffect, discover_plugins


class FakeEffect(AudioEffect):
    """A test plugin that doubles the amplitude."""

    name = "fake"
    description = "doubles amplitude"

    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        return AudioBuffer(
            samples=audio.samples * 2,
            sample_rate=audio.sample_rate,
        )


class TestAudioEffect:
    def test_process(self, sine_wave: AudioBuffer):
        effect = FakeEffect()
        result = effect.process(sine_wave)
        assert torch.allclose(result.samples, sine_wave.samples * 2)

    def test_name(self):
        effect = FakeEffect()
        assert effect.name == "fake"

    def test_description(self):
        effect = FakeEffect()
        assert effect.description == "doubles amplitude"


class TestDiscoverPlugins:
    def test_discovers_plugins(self):
        plugins = discover_plugins()
        assert isinstance(plugins, dict)

    def test_plugins_are_audio_effects(self):
        plugins = discover_plugins()
        for plugin_cls in plugins.values():
            assert issubclass(plugin_cls, AudioEffect)
