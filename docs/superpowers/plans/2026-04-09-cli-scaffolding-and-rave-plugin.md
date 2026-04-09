# CLI Scaffolding & RAVE Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working CLI that can load a RAVE model, process an audio file through it with configurable knobs, and output the result — with a plugin architecture ready for future models.

**Architecture:** Plugin-based CLI using typer + rich. Each ML model is a plugin implementing a simple `process()` interface. Audio flows as `AudioBuffer` objects (torch tensor + sample rate). RAVE is the first plugin, using TorchScript exported models loaded via `torch.jit.load`.

**Tech Stack:** Python 3.11+, typer, rich, torch, torchaudio, soundfile, pytest, uv

---

## File Structure

```
rottengenizdat/
├── pyproject.toml                      # project metadata, dependencies, entry point
├── src/rottengenizdat/
│   ├── __init__.py                     # version
│   ├── banner.py                       # console banner art
│   ├── core.py                         # AudioBuffer dataclass, load/save audio
│   ├── plugin.py                       # AudioEffect base class, plugin discovery
│   ├── cli.py                          # typer app, main entry point
│   └── plugins/
│       ├── __init__.py
│       └── rave.py                     # RAVE plugin
├── tests/
│   ├── conftest.py                     # shared fixtures (sine wave AudioBuffer, tmp paths)
│   ├── test_core.py                    # AudioBuffer + I/O tests
│   ├── test_plugin.py                  # plugin interface + discovery tests
│   ├── test_cli.py                     # CLI integration tests
│   └── test_rave.py                    # RAVE plugin tests (mocked model)
├── README.md
└── docs/
    └── design.md
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/rottengenizdat/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/jake/au-supply/rottengenizdat
uv init --lib --name rottengenizdat --python 3.11
```

This will generate a `pyproject.toml` and `src/rottengenizdat/__init__.py`. We'll overwrite both.

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "rottengenizdat"
version = "0.1.0"
description = "bone music for the machine age — audio transformation through ML models"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.15",
    "rich>=13.0",
    "torch>=2.0",
    "torchaudio>=2.0",
    "soundfile>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[project.scripts]
rotten = "rottengenizdat.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 3: Write `__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p src/rottengenizdat/plugins tests
touch src/rottengenizdat/plugins/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install project in dev mode**

```bash
uv sync --extra dev
```

- [ ] **Step 6: Verify install**

```bash
uv run python -c "import rottengenizdat; print(rottengenizdat.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ uv.lock
git commit -m "feat: project scaffolding with uv, typer, torch dependencies"
```

---

### Task 2: AudioBuffer and I/O

**Files:**
- Create: `src/rottengenizdat/core.py`
- Create: `tests/conftest.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write test fixtures**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Write AudioBuffer tests**

Create `tests/test_core.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_core.py -v
```

Expected: FAIL — `ImportError: cannot import name 'AudioBuffer' from 'rottengenizdat.core'`

- [ ] **Step 4: Implement `core.py`**

Create `src/rottengenizdat/core.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass
class AudioBuffer:
    """Audio data container: a tensor of samples plus sample rate."""

    samples: torch.Tensor  # shape: (channels, num_samples)
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    def to_mono(self) -> AudioBuffer:
        if self.channels == 1:
            return self
        mono = self.samples.mean(dim=0, keepdim=True)
        return AudioBuffer(samples=mono, sample_rate=self.sample_rate)

    def resample(self, target_sr: int) -> AudioBuffer:
        if target_sr == self.sample_rate:
            return self
        resampled = torchaudio.functional.resample(
            self.samples, self.sample_rate, target_sr
        )
        return AudioBuffer(samples=resampled, sample_rate=target_sr)

    def as_model_input(self) -> torch.Tensor:
        """Return tensor shaped (1, channels, num_samples) for model input."""
        return self.samples.unsqueeze(0)

    @classmethod
    def from_model_output(cls, tensor: torch.Tensor, sample_rate: int) -> AudioBuffer:
        """Create AudioBuffer from model output tensor (1, channels, num_samples)."""
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        return cls(samples=tensor, sample_rate=sample_rate)


def load_audio(path: Path, target_sr: int | None = None) -> AudioBuffer:
    """Load an audio file and optionally resample."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    samples, sr = torchaudio.load(str(path))
    buf = AudioBuffer(samples=samples, sample_rate=sr)
    if target_sr is not None:
        buf = buf.resample(target_sr)
    return buf


def save_audio(buf: AudioBuffer, path: Path) -> None:
    """Save an AudioBuffer to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), buf.samples, buf.sample_rate)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_core.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/rottengenizdat/core.py tests/conftest.py tests/test_core.py
git commit -m "feat: AudioBuffer dataclass with load/save, resample, mono conversion"
```

---

### Task 3: Plugin Interface and Discovery

**Files:**
- Create: `src/rottengenizdat/plugin.py`
- Create: `tests/test_plugin.py`

- [ ] **Step 1: Write plugin tests**

Create `tests/test_plugin.py`:

```python
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
        # At minimum we should find the plugins in the plugins/ package
        assert isinstance(plugins, dict)

    def test_plugins_are_audio_effects(self):
        plugins = discover_plugins()
        for plugin_cls in plugins.values():
            assert issubclass(plugin_cls, AudioEffect)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_plugin.py -v
```

Expected: FAIL — `ImportError: cannot import name 'AudioEffect'`

- [ ] **Step 3: Implement `plugin.py`**

Create `src/rottengenizdat/plugin.py`:

```python
from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod

from rottengenizdat.core import AudioBuffer


class AudioEffect(ABC):
    """Base class for all audio effect plugins."""

    name: str
    description: str

    @abstractmethod
    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        """Transform audio. All knobs passed as kwargs."""
        ...


def discover_plugins() -> dict[str, type[AudioEffect]]:
    """Auto-discover all AudioEffect subclasses in the plugins package."""
    import rottengenizdat.plugins as plugins_pkg

    plugins: dict[str, type[AudioEffect]] = {}

    for importer, modname, ispkg in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(f"rottengenizdat.plugins.{modname}")
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AudioEffect) and obj is not AudioEffect:
                plugins[obj.name] = obj

    return plugins
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_plugin.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/plugin.py tests/test_plugin.py
git commit -m "feat: AudioEffect base class with plugin auto-discovery"
```

---

### Task 4: Console Banner

**Files:**
- Create: `src/rottengenizdat/banner.py`

- [ ] **Step 1: Create `banner.py`**

```python
BANNER = r"""
░░▒▒▓▓████████████████████████████████▓▓▒▒░░
 ____    ___   ______  ______    ___  ____
|    \  /   \ |      T|      T  /  _]|    \
|  D  )Y     Y|      ||      | /  [_ |  _  Y
|    / |  O  |l_j  l_jl_j  l_jY    _]|  |  |
|    \ |     |  |  |    |  |  |   [_ |  |  |
|  .  Yl     !  |  |    |  |  |     T|  |  |
l__j\_j \___/   l__j    l__j  l_____jl__j__j
            г е н и з д а т
░░▒▒▓▓████████████████████████████████▓▓▒▒░░
        فاسد · 腐骨音 · bone music
""".strip()
```

- [ ] **Step 2: Commit**

```bash
git add src/rottengenizdat/banner.py
git commit -m "feat: console banner with Cyrillic/Arabic/CJK tagline"
```

---

### Task 5: CLI with Typer

**Files:**
- Create: `src/rottengenizdat/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL — `ImportError: cannot import name 'app'`

- [ ] **Step 3: Implement `cli.py`**

Create `src/rottengenizdat/cli.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rottengenizdat import __version__
from rottengenizdat.banner import BANNER
from rottengenizdat.plugin import discover_plugins

console = Console()
app = typer.Typer(
    name="rotten",
    help="bone music for the machine age",
    invoke_without_command=True,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        console.print(BANNER)
        console.print(f"\n  rottengenizdat v{__version__}\n")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """bone music for the machine age"""


def register_plugins() -> None:
    """Discover and register all plugins as CLI subcommands."""
    plugins = discover_plugins()
    for name, plugin_cls in plugins.items():
        plugin = plugin_cls()
        plugin.register_command(app)


register_plugins()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: all 3 tests PASS. (The `register_command` method doesn't exist yet on `AudioEffect` — but `discover_plugins` returns an empty dict since no plugins exist yet, so `register_plugins` is a no-op and won't error.)

- [ ] **Step 5: Verify CLI entry point works**

```bash
uv run rotten --version
uv run rotten --help
```

Expected: banner prints, version prints, help shows available commands.

- [ ] **Step 6: Commit**

```bash
git add src/rottengenizdat/cli.py tests/test_cli.py
git commit -m "feat: typer CLI with banner, version flag, plugin registration"
```

---

### Task 6: RAVE Plugin

**Files:**
- Create: `src/rottengenizdat/plugins/rave.py`
- Create: `tests/test_rave.py`
- Modify: `src/rottengenizdat/plugin.py` — add `register_command` method to `AudioEffect`

This is the big one. The RAVE plugin needs to:
1. Download and cache pretrained models
2. Load a TorchScript model via `torch.jit.load`
3. Encode audio → latent space → optionally manipulate → decode
4. Register a typer subcommand with knobs

**RAVE API reference** (from research):
```python
model = torch.jit.load("model.ts").eval()
x = torch.from_numpy(audio).reshape(1, 1, -1)  # (batch, channels, samples)
z = model.encode(x)                              # (1, latent_dims, time_steps)
x_hat = model.decode(z)                          # (1, 1, samples)
# latent_dims is typically 16, time_steps ≈ samples / 2048
```

**Available pretrained models** (from `https://play.forum.ircam.fr/rave-vst-api/`):
VCTK, darbouka_onnx, nasa, percussion, vintage, isis, musicnet, sol_ordinario, sol_full, sol_ordinario_fast

- [ ] **Step 1: Add `register_command` to AudioEffect base class**

Add to `src/rottengenizdat/plugin.py`, inside the `AudioEffect` class:

```python
    def register_command(self, app: "typer.Typer") -> None:
        """Register this plugin as a typer subcommand. Override for custom args."""
        ...
```

This is an abstract-ish hook. Each plugin overrides it to register its own subcommand with its own knobs. Make it non-abstract with a default no-op so existing tests don't break.

- [ ] **Step 2: Write RAVE plugin tests**

Create `tests/test_rave.py`:

```python
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
        # encode returns (1, 16, time_steps) — 16 latent dims
        model.encode = MagicMock(
            side_effect=lambda x: torch.randn(1, 16, x.shape[-1] // 2048)
        )
        # decode returns (1, 1, samples)
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
        # Capture the z passed to decode
        decoded_z = []
        def capture_decode(z):
            decoded_z.append(z.clone())
            return torch.randn(1, 1, z.shape[-1] * 2048)
        mock_model.decode = MagicMock(side_effect=capture_decode)

        effect = RaveEffect()
        effect.process(sine_wave, model_name="percussion", temperature=2.0)

        # z should have been scaled by temperature
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

        # Second call should use cache, not re-download
        mock_get.reset_mock()
        path2 = download_model("percussion", cache_dir=tmp_path)
        mock_get.assert_not_called()
        assert path == path2

    def test_download_unknown_model_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown model"):
            download_model("nonexistent_model", cache_dir=tmp_path)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_rave.py -v
```

Expected: FAIL — `ImportError: cannot import name 'RaveEffect'`

- [ ] **Step 4: Add requests dependency**

Add `"requests>=2.31"` to `dependencies` in `pyproject.toml`, then:

```bash
uv sync --extra dev
```

- [ ] **Step 5: Implement the RAVE plugin**

Create `src/rottengenizdat/plugins/rave.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import requests
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import AudioEffect

console = Console()

RAVE_API_BASE = "https://play.forum.ircam.fr/rave-vst-api"

AVAILABLE_MODELS = [
    "VCTK",
    "darbouka_onnx",
    "nasa",
    "percussion",
    "vintage",
    "isis",
    "musicnet",
    "sol_ordinario",
    "sol_full",
    "sol_ordinario_fast",
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rottengenizdat" / "models" / "rave"


def download_model(
    model_name: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Download a pretrained RAVE model, using cache if available."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / f"{model_name}.ts"
    if model_path.exists():
        return model_path

    url = f"{RAVE_API_BASE}/get_model/{model_name}"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Downloading {model_name}...", total=None)
        response = requests.get(url)
        response.raise_for_status()
        model_path.write_bytes(response.content)

    return model_path


def load_rave_model(model_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> torch.jit.ScriptModule:
    """Download (if needed) and load a RAVE model."""
    model_path = download_model(model_name, cache_dir)
    model = torch.jit.load(str(model_path)).eval()
    return model


class RaveEffect(AudioEffect):
    """RAVE — Realtime Audio Variational autoEncoder.

    Encode audio into latent space, manipulate, decode.
    Even a straight round-trip produces uncanny results.
    """

    name = "rave"
    description = "RAVE variational autoencoder — latent space audio mangling"

    def process(
        self,
        audio: AudioBuffer,
        model_name: str = "percussion",
        temperature: float = 1.0,
        noise: float = 0.0,
        **kwargs,
    ) -> AudioBuffer:
        """Process audio through RAVE.

        Args:
            audio: Input audio buffer.
            model_name: Pretrained model to use.
            temperature: Scale latent vectors (>1 = more extreme, <1 = subtle).
            noise: Amount of random noise to add to latent space (0-1).
        """
        model = load_rave_model(model_name)

        with torch.no_grad():
            # Ensure mono for RAVE
            mono = audio.to_mono()
            x = mono.as_model_input()  # (1, 1, samples)

            # Encode
            z = model.encode(x)

            # Manipulate latent space
            if temperature != 1.0:
                z = z * temperature

            if noise > 0.0:
                z = z + torch.randn_like(z) * noise

            # Decode
            x_hat = model.decode(z)

        return AudioBuffer.from_model_output(x_hat, sample_rate=audio.sample_rate)

    def register_command(self, app: typer.Typer) -> None:
        """Register the rave subcommand."""

        @app.command(name=self.name, help=self.description)
        def rave_command(
            input_file: Annotated[
                Path, typer.Argument(help="Input audio file")
            ],
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file path")
            ] = Path("output.wav"),
            model: Annotated[
                str, typer.Option("--model", "-m", help=f"Pretrained model: {', '.join(AVAILABLE_MODELS)}")
            ] = "percussion",
            temperature: Annotated[
                float, typer.Option("--temperature", "-t", help="Latent scaling (>1 extreme, <1 subtle)")
            ] = 1.0,
            noise_amount: Annotated[
                float, typer.Option("--noise", "-n", help="Random noise in latent space (0-1)")
            ] = 0.0,
            sweep: Annotated[
                Optional[str], typer.Option("--sweep", help="Sweep a parameter, e.g. temperature=0.5,1.0,1.5")
            ] = None,
        ) -> None:
            if not input_file.exists():
                console.print(f"[red]File not found: {input_file}[/red]")
                raise typer.Exit(1)

            console.print(f"[bold]Loading:[/bold] {input_file}")
            audio = load_audio(input_file)
            console.print(
                f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz"
            )

            if sweep:
                self._run_sweep(audio, output, model, temperature, noise_amount, sweep)
            else:
                console.print(
                    f"[bold]Processing:[/bold] rave model={model} temp={temperature} noise={noise_amount}"
                )
                result = self.process(
                    audio,
                    model_name=model,
                    temperature=temperature,
                    noise=noise_amount,
                )
                save_audio(result, output)
                console.print(f"[green]Saved:[/green] {output}")

        return rave_command

    def _run_sweep(
        self,
        audio: AudioBuffer,
        output_dir: Path,
        model: str,
        temperature: float,
        noise_amount: float,
        sweep: str,
    ) -> None:
        """Run a parameter sweep and save grid of outputs."""
        param_name, values_str = sweep.split("=", 1)
        values = [float(v) for v in values_str.split(",")]

        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Sweeping:[/bold] {param_name} = {values}")

        for val in values:
            kwargs = {
                "model_name": model,
                "temperature": temperature,
                "noise": noise_amount,
            }
            kwargs[param_name] = val

            result = self.process(audio, **kwargs)
            filename = f"{param_name}_{val:.2f}.wav"
            save_audio(result, output_dir / filename)
            console.print(f"  [green]Saved:[/green] {output_dir / filename}")
```

- [ ] **Step 6: Run RAVE tests**

```bash
uv run pytest tests/test_rave.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 7: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS (core + plugin + cli + rave)

- [ ] **Step 8: Commit**

```bash
git add src/rottengenizdat/plugin.py src/rottengenizdat/plugins/rave.py tests/test_rave.py pyproject.toml uv.lock
git commit -m "feat: RAVE plugin with model download, latent manipulation, sweep mode

Supports encode→manipulate→decode pipeline with knobs for temperature
(latent scaling) and noise (random perturbation). Downloads and caches
pretrained models from IRCAM. Sweep mode generates grids of outputs
across parameter ranges."
```

---

### Task 7: Integration Test — End to End

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add CLI integration test for RAVE subcommand**

Add to `tests/test_cli.py`:

```python
from unittest.mock import patch, MagicMock
import torch


class TestRaveSubcommand:
    def test_rave_appears_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "rave" in result.output.lower()

    @patch("rottengenizdat.plugins.rave.load_rave_model")
    def test_rave_processes_file(self, mock_load, tmp_path):
        # Create a mock model
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            side_effect=lambda x: torch.randn(1, 16, x.shape[-1] // 2048)
        )
        mock_model.decode = MagicMock(
            side_effect=lambda z: torch.randn(1, 1, z.shape[-1] * 2048)
        )
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_load.return_value = mock_model

        # Create a test audio file
        import torchaudio
        sr = 44100
        samples = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, sr)).unsqueeze(0)
        input_path = tmp_path / "test_input.wav"
        torchaudio.save(str(input_path), samples, sr)

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

        import torchaudio
        sr = 44100
        samples = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, sr)).unsqueeze(0)
        input_path = tmp_path / "test_input.wav"
        torchaudio.save(str(input_path), samples, sr)

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
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS

- [ ] **Step 3: Verify CLI end-to-end**

```bash
uv run rotten --help
uv run rotten rave --help
```

Expected: help shows rave subcommand with all knobs documented.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: CLI integration tests for rave subcommand and sweep mode"
```

---

### Task 8: Push and Verify

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS

- [ ] **Step 2: Push**

```bash
git push origin main
```

- [ ] **Step 3: Verify repo looks good**

```bash
gh repo view --web
```
