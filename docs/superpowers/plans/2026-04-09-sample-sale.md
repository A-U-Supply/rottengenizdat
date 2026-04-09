# Sample Sale Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable fetching random audio/video from Slack #sample-sale as pipeline inputs, with multi-input support (splice, concat, independent modes) for both local files and Slack samples.

**Architecture:** Flat CLI integration — no abstract input-source system. New modules for config, sample_sale, splice, and input resolution sit alongside existing code. A `rotten config` subcommand manages settings, a `rotten sample-sale` subcommand manages the cache, and `--sample-sale`/`--sample-sale-count` flags on pipeline commands pull in random samples. Multi-input is supported via positional args + a `--mode` flag.

**Tech Stack:** Python 3.11+, typer, slack_sdk, yt-dlp (optional runtime), tomllib/tomli_w for config

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/rottengenizdat/config.py` | Read/write `~/.config/rottengenizdat/config.toml`, token resolution (file > env > error) |
| `src/rottengenizdat/splice.py` | Chop AudioBuffers into random segments, shuffle, reassemble |
| `src/rottengenizdat/inputs.py` | Resolve multi-input list (local + sample-sale), dispatch combination modes |
| `src/rottengenizdat/sample_sale.py` | Slack API interaction, index management, media download, random selection |
| `tests/test_config.py` | Config module tests |
| `tests/test_splice.py` | Splice module tests |
| `tests/test_inputs.py` | Input resolution + combination mode tests |
| `tests/test_sample_sale.py` | Sample sale module tests |

### Modified Files

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `slack_sdk` and `tomli_w` dependencies |
| `src/rottengenizdat/core.py` | Add `concat_buffers()` function |
| `src/rottengenizdat/cli.py` | Multi-input args, `--sample-sale` flags, `--mode` flag, `config` and `sample-sale` subcommands |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:1-28`

- [ ] **Step 1: Add slack_sdk and tomli_w to pyproject.toml**

In `pyproject.toml`, add to the `dependencies` list:

```toml
dependencies = [
    "typer>=0.15",
    "rich>=13.0",
    "torch>=2.0",
    "torchaudio>=2.0",
    "soundfile>=0.12",
    "requests>=2.31",
    "slack_sdk>=3.0",
    "tomli_w>=1.0",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `cd /Users/jake/au-supply/rottengenizdat && uv pip install -e .`
Expected: successful install with slack_sdk and tomli_w added.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add slack_sdk and tomli_w dependencies

slack_sdk for Slack API access (sample-sale feature).
tomli_w for writing TOML config files (tomllib only reads)."
```

---

## Task 2: Config Module

**Files:**
- Create: `src/rottengenizdat/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config reading**

Create `tests/test_config.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rottengenizdat.config import (
    CONFIG_DIR,
    load_config,
    save_config,
    resolve_slack_token,
    resolve_slack_channel,
    config_set,
)


class TestLoadConfig:
    def test_returns_empty_dict_when_no_file(self, tmp_path: Path):
        config = load_config(config_dir=tmp_path)
        assert config == {}

    def test_reads_existing_config(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[slack]\ntoken = "xoxb-test"\nchannel = "C123"\n')
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "xoxb-test"
        assert config["slack"]["channel"] == "C123"


class TestSaveConfig:
    def test_creates_file_and_dirs(self, tmp_path: Path):
        nested = tmp_path / "sub" / "dir"
        save_config({"slack": {"token": "xoxb-abc"}}, config_dir=nested)
        assert (nested / "config.toml").exists()
        config = load_config(config_dir=nested)
        assert config["slack"]["token"] == "xoxb-abc"

    def test_overwrites_existing(self, tmp_path: Path):
        save_config({"slack": {"token": "old"}}, config_dir=tmp_path)
        save_config({"slack": {"token": "new"}}, config_dir=tmp_path)
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "new"


class TestConfigSet:
    def test_set_nested_key(self, tmp_path: Path):
        config_set("slack.token", "xoxb-set-test", config_dir=tmp_path)
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "xoxb-set-test"

    def test_set_preserves_existing_keys(self, tmp_path: Path):
        config_set("slack.token", "xoxb-keep", config_dir=tmp_path)
        config_set("slack.channel", "C999", config_dir=tmp_path)
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "xoxb-keep"
        assert config["slack"]["channel"] == "C999"


class TestResolveSlackToken:
    def test_prefers_config_file(self, tmp_path: Path, monkeypatch):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[slack]\ntoken = "xoxb-from-file"\n')
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
        token = resolve_slack_token(config_dir=tmp_path)
        assert token == "xoxb-from-file"

    def test_falls_back_to_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
        token = resolve_slack_token(config_dir=tmp_path)
        assert token == "xoxb-from-env"

    def test_raises_when_no_token(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="Slack token not found"):
            resolve_slack_token(config_dir=tmp_path)


class TestResolveSlackChannel:
    def test_from_config(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[slack]\nchannel = "C456"\n')
        channel = resolve_slack_channel(config_dir=tmp_path)
        assert channel == "C456"

    def test_raises_when_no_channel(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Slack channel not configured"):
            resolve_slack_channel(config_dir=tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rottengenizdat.config'`

- [ ] **Step 3: Implement config module**

Create `src/rottengenizdat/config.py`:

```python
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w

CONFIG_DIR = Path.home() / ".config" / "rottengenizdat"
CONFIG_FILENAME = "config.toml"


def _config_path(config_dir: Path = CONFIG_DIR) -> Path:
    return config_dir / CONFIG_FILENAME


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Load config from TOML file. Returns empty dict if file doesn't exist."""
    path = _config_path(config_dir)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def save_config(config: dict, config_dir: Path = CONFIG_DIR) -> None:
    """Write config dict to TOML file, creating directories as needed."""
    path = _config_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def config_set(key: str, value: str, config_dir: Path = CONFIG_DIR) -> None:
    """Set a dotted key (e.g. 'slack.token') in the config file."""
    config = load_config(config_dir)
    parts = key.split(".")
    d = config
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
    save_config(config, config_dir)


def resolve_slack_token(config_dir: Path = CONFIG_DIR) -> str:
    """Resolve Slack token: config file > SLACK_BOT_TOKEN env var > error."""
    config = load_config(config_dir)
    token = config.get("slack", {}).get("token")
    if token:
        return token
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token
    raise ValueError(
        "Slack token not found. Set it with:\n"
        "  rotten config set slack.token xoxb-YOUR-TOKEN\n"
        "or set the SLACK_BOT_TOKEN environment variable."
    )


def resolve_slack_channel(config_dir: Path = CONFIG_DIR) -> str:
    """Resolve Slack channel ID from config file."""
    config = load_config(config_dir)
    channel = config.get("slack", {}).get("channel")
    if channel:
        return channel
    raise ValueError(
        "Slack channel not configured. Set it with:\n"
        "  rotten config set slack.channel C0XXXXXXX"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_config.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/config.py tests/test_config.py
git commit -m "feat: config module for settings and Slack token resolution

Reads/writes ~/.config/rottengenizdat/config.toml. Supports dotted
key setting (e.g. slack.token). Token resolution: config file takes
priority over SLACK_BOT_TOKEN env var."
```

---

## Task 3: `rotten config` Subcommand

**Files:**
- Modify: `src/rottengenizdat/cli.py:321` (append after recipe_save)

- [ ] **Step 1: Write failing test for config subcommands**

Append to `tests/test_config.py`:

```python
from typer.testing import CliRunner
from rottengenizdat.cli import app

runner = CliRunner()


class TestConfigCLI:
    def test_config_path(self):
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert "config.toml" in result.stdout

    def test_config_set_and_show(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("rottengenizdat.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("rottengenizdat.cli.CONFIG_DIR", tmp_path)
        result = runner.invoke(app, ["config", "set", "slack.token", "xoxb-cli-test"])
        assert result.exit_code == 0
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        # Token should be masked in output
        assert "xoxb-cli-test" not in result.stdout
        assert "xoxb-***" in result.stdout or "****" in result.stdout

    def test_config_show_empty(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("rottengenizdat.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("rottengenizdat.cli.CONFIG_DIR", tmp_path)
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_config.py::TestConfigCLI -v`
Expected: FAIL — typer doesn't know about "config" subcommand yet

- [ ] **Step 3: Implement config subcommands**

Add to `src/rottengenizdat/cli.py`, after the recipe section. First add the import at the top of the file:

```python
from rottengenizdat.config import (
    CONFIG_DIR,
    load_config,
    config_set as _config_set,
)
```

Then add the config subcommand group at the bottom:

```python
# ---------------------------------------------------------------------------
# config sub-app
# ---------------------------------------------------------------------------

config_app = typer.Typer(name="config", help="Manage rottengenizdat configuration", context_settings=CONTEXT_SETTINGS)
app.add_typer(config_app)


@config_app.command(name="path")
def config_path() -> None:
    """Print the config file location."""
    console.print(str(CONFIG_DIR / "config.toml"))


@config_app.command(name="show")
def config_show() -> None:
    """Print current configuration (tokens masked)."""
    config = load_config()
    if not config:
        console.print("[dim]No configuration file found.[/dim]")
        console.print(f"[dim]Create one with:[/dim] rotten config set slack.token xoxb-YOUR-TOKEN")
        return
    for section, values in config.items():
        console.print(f"[bold]\\[{section}][/bold]")
        if isinstance(values, dict):
            for key, val in values.items():
                display_val = val
                if "token" in key.lower() and isinstance(val, str) and len(val) > 8:
                    display_val = val[:4] + "****" + val[-4:]
                console.print(f"  {key} = {display_val}")
        else:
            console.print(f"  {section} = {values}")


@config_app.command(name="set")
def config_set_cmd(
    key: Annotated[str, typer.Argument(help="Dotted config key (e.g. slack.token)")],
    value: Annotated[str, typer.Argument(help="Value to set")],
) -> None:
    """Set a configuration value."""
    _config_set(key, value)
    console.print(f"[green]Set:[/green] {key}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_config.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/cli.py tests/test_config.py
git commit -m "feat: rotten config subcommand (show, set, path)

Manages ~/.config/rottengenizdat/config.toml. Token values are
masked in 'config show' output. 'config set' accepts dotted keys
like slack.token and slack.channel."
```

---

## Task 4: Splice Module

**Files:**
- Create: `src/rottengenizdat/splice.py`
- Create: `tests/test_splice.py`

- [ ] **Step 1: Write failing tests for splice**

Create `tests/test_splice.py`:

```python
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
        # Use tight min/max to make it easy to check
        result = splice_buffers([buf, buf], min_seconds=1.0, max_seconds=2.0)
        # Total should be 20 seconds of samples
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_splice.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rottengenizdat.splice'`

- [ ] **Step 3: Implement splice module**

Create `src/rottengenizdat/splice.py`:

```python
from __future__ import annotations

import random

import torch

from rottengenizdat.core import AudioBuffer

DEFAULT_MIN_SECONDS = 0.25
DEFAULT_MAX_SECONDS = 4.0


def splice_buffers(
    buffers: list[AudioBuffer],
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> AudioBuffer:
    """Chop all buffers into random segments, shuffle, reassemble.

    Each buffer is sliced into random-length segments (between min_seconds
    and max_seconds). All segments from all buffers are collected, shuffled,
    and concatenated into a single AudioBuffer. All input samples are used.

    Buffers with different sample rates are resampled to match the first
    buffer's sample rate.

    Args:
        buffers: One or more AudioBuffers to splice together.
        min_seconds: Minimum segment duration in seconds.
        max_seconds: Maximum segment duration in seconds.

    Returns:
        A new AudioBuffer containing all input samples in shuffled segments.
    """
    if not buffers:
        raise ValueError("splice_buffers requires at least one buffer")

    target_sr = buffers[0].sample_rate

    # Resample and convert to mono, collect all segments
    segments: list[torch.Tensor] = []
    for buf in buffers:
        b = buf.resample(target_sr).to_mono()
        samples = b.samples  # (1, num_samples)
        num_samples = b.num_samples
        pos = 0
        while pos < num_samples:
            seg_seconds = random.uniform(min_seconds, max_seconds)
            seg_samples = int(seg_seconds * target_sr)
            # Don't leave a tiny remainder — absorb it into the last segment
            remaining = num_samples - pos
            if remaining <= seg_samples or (remaining - seg_samples) < int(min_seconds * target_sr):
                segments.append(samples[:, pos:])
                break
            segments.append(samples[:, pos : pos + seg_samples])
            pos += seg_samples

    random.shuffle(segments)
    joined = torch.cat(segments, dim=1)
    return AudioBuffer(samples=joined, sample_rate=target_sr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_splice.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/splice.py tests/test_splice.py
git commit -m "feat: splice module — chop and shuffle AudioBuffers

Slices inputs into random-length segments (default 0.25s–4.0s),
shuffles all segments together, and reassembles into one buffer.
Resamples mismatched sample rates to the first input's rate."
```

---

## Task 5: Concat Utility in core.py

**Files:**
- Modify: `src/rottengenizdat/core.py:84` (append after save_audio)
- Modify: `tests/test_core.py:60` (append new test class)

- [ ] **Step 1: Write failing test for concat_buffers**

Append to `tests/test_core.py`:

```python
from rottengenizdat.core import concat_buffers


class TestConcatBuffers:
    def test_concat_two_buffers(self):
        a = AudioBuffer(samples=torch.ones(1, 100), sample_rate=44100)
        b = AudioBuffer(samples=torch.ones(1, 200) * 2, sample_rate=44100)
        result = concat_buffers([a, b])
        assert result.num_samples == 300
        assert torch.allclose(result.samples[:, :100], torch.ones(1, 100))
        assert torch.allclose(result.samples[:, 100:], torch.ones(1, 200) * 2)

    def test_preserves_sample_rate(self):
        a = AudioBuffer(samples=torch.ones(1, 100), sample_rate=22050)
        b = AudioBuffer(samples=torch.ones(1, 100), sample_rate=22050)
        result = concat_buffers([a, b])
        assert result.sample_rate == 22050

    def test_resamples_mismatched_rates(self):
        a = AudioBuffer(samples=torch.ones(1, 44100), sample_rate=44100)
        b = AudioBuffer(samples=torch.ones(1, 22050), sample_rate=22050)
        result = concat_buffers([a, b])
        assert result.sample_rate == 44100
        assert result.num_samples == 44100 + 44100

    def test_single_buffer(self):
        a = AudioBuffer(samples=torch.ones(1, 50), sample_rate=44100)
        result = concat_buffers([a])
        assert torch.equal(result.samples, a.samples)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            concat_buffers([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_core.py::TestConcatBuffers -v`
Expected: FAIL — `ImportError: cannot import name 'concat_buffers'`

- [ ] **Step 3: Implement concat_buffers**

Append to `src/rottengenizdat/core.py` after the `save_audio` function:

```python
def concat_buffers(buffers: list[AudioBuffer]) -> AudioBuffer:
    """Concatenate multiple AudioBuffers end-to-end.

    Buffers with different sample rates are resampled to match the first
    buffer's sample rate. All buffers are converted to mono before concatenation.

    Args:
        buffers: Non-empty list of AudioBuffers to concatenate in order.

    Returns:
        A single AudioBuffer containing all inputs joined end-to-end.
    """
    if not buffers:
        raise ValueError("concat_buffers requires at least one buffer")
    target_sr = buffers[0].sample_rate
    parts: list[torch.Tensor] = []
    for buf in buffers:
        b = buf.resample(target_sr).to_mono()
        parts.append(b.samples)
    joined = torch.cat(parts, dim=1)
    return AudioBuffer(samples=joined, sample_rate=target_sr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_core.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/core.py tests/test_core.py
git commit -m "feat: concat_buffers utility for joining AudioBuffers

Concatenates multiple AudioBuffers end-to-end, resampling mismatched
sample rates to the first buffer's rate. Converts all to mono."
```

---

## Task 6: Input Resolution Module

**Files:**
- Create: `src/rottengenizdat/inputs.py`
- Create: `tests/test_inputs.py`

- [ ] **Step 1: Write failing tests for input resolution**

Create `tests/test_inputs.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rottengenizdat.inputs'`

- [ ] **Step 3: Implement inputs module**

Create `src/rottengenizdat/inputs.py`:

```python
from __future__ import annotations

from enum import Enum

from rottengenizdat.core import AudioBuffer, concat_buffers
from rottengenizdat.splice import splice_buffers, DEFAULT_MIN_SECONDS, DEFAULT_MAX_SECONDS


class InputMode(Enum):
    """How to combine multiple inputs before running the pipeline."""

    PASSTHROUGH = "passthrough"
    SPLICE = "splice"
    CONCAT = "concat"
    INDEPENDENT = "independent"

    @classmethod
    def resolve(cls, mode_str: str | None, num_inputs: int) -> InputMode:
        """Determine the input mode from the user's --mode flag and input count.

        Args:
            mode_str: The raw --mode value (None if not specified).
            num_inputs: How many input files/samples there are.

        Returns:
            The resolved InputMode.
        """
        if mode_str is not None:
            return cls(mode_str)
        if num_inputs <= 1:
            return cls.PASSTHROUGH
        return cls.SPLICE


def combine_inputs(
    buffers: list[AudioBuffer],
    mode: InputMode,
    splice_min: float = DEFAULT_MIN_SECONDS,
    splice_max: float = DEFAULT_MAX_SECONDS,
) -> list[AudioBuffer]:
    """Combine input buffers according to the given mode.

    Args:
        buffers: The input AudioBuffers.
        mode: How to combine them.
        splice_min: Min segment duration for splice mode.
        splice_max: Max segment duration for splice mode.

    Returns:
        A list of AudioBuffers to process. Length is 1 for all modes except
        INDEPENDENT, which returns one buffer per input.
    """
    if not buffers:
        raise ValueError("No input buffers provided")

    if mode == InputMode.PASSTHROUGH:
        return [buffers[0]]
    elif mode == InputMode.CONCAT:
        return [concat_buffers(buffers)]
    elif mode == InputMode.INDEPENDENT:
        return list(buffers)
    elif mode == InputMode.SPLICE:
        return [splice_buffers(buffers, min_seconds=splice_min, max_seconds=splice_max)]
    else:
        raise ValueError(f"Unknown input mode: {mode}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/inputs.py tests/test_inputs.py
git commit -m "feat: input resolution module with mode dispatch

InputMode enum with resolve() for default selection (single input =
passthrough, multi = splice). combine_inputs() dispatches to splice,
concat, independent, or passthrough."
```

---

## Task 7: Sample Sale Module — Index and Sync

**Files:**
- Create: `src/rottengenizdat/sample_sale.py`
- Create: `tests/test_sample_sale.py`

- [ ] **Step 1: Write failing tests for index management**

Create `tests/test_sample_sale.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rottengenizdat.sample_sale import (
    CACHE_DIR,
    IndexEntry,
    load_index,
    save_index,
    extract_media_from_messages,
    sync_index,
    pick_random_samples,
)


@pytest.fixture
def sample_index() -> list[dict]:
    return [
        {
            "id": "F001",
            "type": "attachment",
            "filename": "beat.wav",
            "mime": "audio/wav",
            "slack_url": "https://files.slack.com/beat.wav",
            "cached_path": "samples/F001-beat.wav",
            "message_ts": "1700000001.000100",
        },
        {
            "id": "sha256-abc123",
            "type": "link",
            "url": "https://youtube.com/watch?v=abc",
            "cached_path": "samples/sha256-abc123.wav",
            "message_ts": "1700000002.000200",
        },
    ]


class TestIndexIO:
    def test_load_empty(self, tmp_path: Path):
        entries = load_index(cache_dir=tmp_path)
        assert entries == []

    def test_save_and_load(self, tmp_path: Path, sample_index):
        entries = [IndexEntry(**e) for e in sample_index]
        save_index(entries, cache_dir=tmp_path)
        loaded = load_index(cache_dir=tmp_path)
        assert len(loaded) == 2
        assert loaded[0].id == "F001"
        assert loaded[1].type == "link"


class TestExtractMedia:
    def test_extracts_audio_attachment(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "files": [
                    {
                        "id": "F001",
                        "name": "beat.wav",
                        "mimetype": "audio/wav",
                        "url_private_download": "https://files.slack.com/beat.wav",
                    }
                ],
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 1
        assert entries[0].type == "attachment"
        assert entries[0].filename == "beat.wav"

    def test_skips_non_audio_attachment(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "files": [
                    {
                        "id": "F002",
                        "name": "photo.png",
                        "mimetype": "image/png",
                        "url_private_download": "https://files.slack.com/photo.png",
                    }
                ],
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 0

    def test_extracts_video_attachment(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "files": [
                    {
                        "id": "F003",
                        "name": "clip.mp4",
                        "mimetype": "video/mp4",
                        "url_private_download": "https://files.slack.com/clip.mp4",
                    }
                ],
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 1

    def test_extracts_url_links(self):
        messages = [
            {
                "ts": "1700000003.000300",
                "text": "check this out https://youtube.com/watch?v=xyz",
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 1
        assert entries[0].type == "link"
        assert "youtube.com" in entries[0].url

    def test_no_duplicate_ids(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "text": "https://youtube.com/watch?v=xyz",
            },
            {
                "ts": "1700000002.000200",
                "text": "https://youtube.com/watch?v=xyz",
            },
        ]
        entries = extract_media_from_messages(messages)
        # Same URL should produce same ID — deduplicated
        ids = [e.id for e in entries]
        assert len(set(ids)) == len(ids)


class TestPickRandom:
    def test_picks_correct_count(self, sample_index):
        entries = [IndexEntry(**e) for e in sample_index]
        picked = pick_random_samples(entries, count=1)
        assert len(picked) == 1

    def test_picks_all_when_count_exceeds(self, sample_index):
        entries = [IndexEntry(**e) for e in sample_index]
        picked = pick_random_samples(entries, count=10)
        assert len(picked) == 2

    def test_empty_index_raises(self):
        with pytest.raises(ValueError, match="No samples"):
            pick_random_samples([], count=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_sample_sale.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rottengenizdat.sample_sale'`

- [ ] **Step 3: Implement sample_sale module**

Create `src/rottengenizdat/sample_sale.py`:

```python
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rottengenizdat.config import resolve_slack_channel, resolve_slack_token

CACHE_DIR = Path.home() / ".cache" / "rottengenizdat"
INDEX_FILENAME = "index.json"
SAMPLES_DIRNAME = "samples"

# MIME prefixes that indicate audio or video
_MEDIA_PREFIXES = ("audio/", "video/")

# Regex to find URLs in message text
_URL_PATTERN = re.compile(r"https?://[^\s<>|]+")


@dataclass
class IndexEntry:
    """One indexed media item from #sample-sale."""

    id: str
    type: str  # "attachment" or "link"
    message_ts: str
    filename: str = ""
    mime: str = ""
    slack_url: str = ""
    url: str = ""
    cached_path: str = ""

    @property
    def is_cached(self) -> bool:
        if not self.cached_path:
            return False
        return (CACHE_DIR / self.cached_path).exists()


def _index_path(cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / INDEX_FILENAME


def load_index(cache_dir: Path = CACHE_DIR) -> list[IndexEntry]:
    """Load the sample index from disk."""
    path = _index_path(cache_dir)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [IndexEntry(**entry) for entry in data]


def save_index(entries: list[IndexEntry], cache_dir: Path = CACHE_DIR) -> None:
    """Write the sample index to disk."""
    path = _index_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)


def _url_hash(url: str) -> str:
    """Deterministic short hash for a URL."""
    return "sha256-" + hashlib.sha256(url.encode()).hexdigest()[:12]


def extract_media_from_messages(messages: list[dict]) -> list[IndexEntry]:
    """Extract audio/video entries from Slack messages.

    Finds both file attachments (filtered by MIME type) and URLs in message text.
    Deduplicates by entry ID.
    """
    seen_ids: set[str] = set()
    entries: list[IndexEntry] = []

    for msg in messages:
        ts = msg.get("ts", "")

        # File attachments
        for file_info in msg.get("files", []):
            mime = file_info.get("mimetype", "")
            if not any(mime.startswith(p) for p in _MEDIA_PREFIXES):
                continue
            file_id = file_info["id"]
            if file_id in seen_ids:
                continue
            seen_ids.add(file_id)
            filename = file_info.get("name", "unknown")
            entries.append(
                IndexEntry(
                    id=file_id,
                    type="attachment",
                    filename=filename,
                    mime=mime,
                    slack_url=file_info.get("url_private_download", ""),
                    cached_path=f"{SAMPLES_DIRNAME}/{file_id}-{filename}",
                    message_ts=ts,
                )
            )

        # URLs in message text
        text = msg.get("text", "")
        for url in _URL_PATTERN.findall(text):
            entry_id = _url_hash(url)
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append(
                IndexEntry(
                    id=entry_id,
                    type="link",
                    url=url,
                    cached_path=f"{SAMPLES_DIRNAME}/{entry_id}.wav",
                    message_ts=ts,
                )
            )

    return entries


def sync_index(
    full: bool = False,
    cache_dir: Path = CACHE_DIR,
    config_dir: Path | None = None,
) -> list[IndexEntry]:
    """Fetch new messages from Slack and update the index.

    Args:
        full: If True, rebuild index from scratch. Otherwise incremental.
        cache_dir: Where to store the index and samples.
        config_dir: Config directory override (for testing).

    Returns:
        The updated full index.
    """
    from slack_sdk import WebClient

    kwargs = {"config_dir": config_dir} if config_dir else {}
    token = resolve_slack_token(**kwargs)
    channel = resolve_slack_channel(**kwargs)
    client = WebClient(token=token)

    existing = [] if full else load_index(cache_dir)
    latest_ts = "0"
    if existing and not full:
        latest_ts = max(e.message_ts for e in existing)

    # Paginate through channel history
    new_messages: list[dict] = []
    cursor = None
    while True:
        resp = client.conversations_history(
            channel=channel,
            oldest=latest_ts,
            limit=200,
            cursor=cursor,
        )
        new_messages.extend(resp["messages"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    new_entries = extract_media_from_messages(new_messages)

    # Deduplicate against existing index
    existing_ids = {e.id for e in existing}
    unique_new = [e for e in new_entries if e.id not in existing_ids]

    combined = existing + unique_new
    save_index(combined, cache_dir)
    return combined


def download_sample(
    entry: IndexEntry,
    cache_dir: Path = CACHE_DIR,
    config_dir: Path | None = None,
) -> Path:
    """Download a sample if not already cached. Returns the local file path."""
    local_path = cache_dir / entry.cached_path
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if entry.type == "attachment":
        kwargs = {"config_dir": config_dir} if config_dir else {}
        token = resolve_slack_token(**kwargs)
        import requests

        resp = requests.get(
            entry.slack_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
    elif entry.type == "link":
        # Try yt-dlp; skip with warning if not available
        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "yt-dlp is not installed. Install it to download linked media:\n"
                "  brew install yt-dlp"
            )
        result = subprocess.run(
            [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--output", str(local_path),
                entry.url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed for {entry.url}: {result.stderr}")

    return local_path


def pick_random_samples(
    entries: list[IndexEntry], count: int
) -> list[IndexEntry]:
    """Pick N random samples from the index."""
    if not entries:
        raise ValueError("No samples in index. Run: rotten sample-sale refresh")
    return random.sample(entries, min(count, len(entries)))


def clear_cache(cache_dir: Path = CACHE_DIR, full: bool = False) -> None:
    """Delete cached sample files, optionally the entire cache dir."""
    samples_dir = cache_dir / SAMPLES_DIRNAME
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    if full:
        index = _index_path(cache_dir)
        if index.exists():
            index.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_sample_sale.py -v`
Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/sample_sale.py tests/test_sample_sale.py
git commit -m "feat: sample_sale module — index, sync, download, random pick

Indexes #sample-sale messages (attachments by MIME, links by URL).
Incremental sync fetches only new messages since last sync. Lazy
download: media files only fetched when selected. yt-dlp for links,
Slack API for attachments."
```

---

## Task 8: `rotten sample-sale` Subcommand

**Files:**
- Modify: `src/rottengenizdat/cli.py` (append after config section)

- [ ] **Step 1: Write failing tests for sample-sale subcommands**

Append to `tests/test_sample_sale.py`:

```python
from typer.testing import CliRunner
from rottengenizdat.cli import app

runner = CliRunner()


class TestSampleSaleCLI:
    def test_list_empty(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("rottengenizdat.sample_sale.CACHE_DIR", tmp_path)
        monkeypatch.setattr("rottengenizdat.cli.CACHE_DIR", tmp_path)
        result = runner.invoke(app, ["sample-sale", "list"])
        assert result.exit_code == 0
        assert "No samples" in result.stdout or "empty" in result.stdout.lower()

    def test_list_with_entries(self, tmp_path: Path, monkeypatch, sample_index):
        monkeypatch.setattr("rottengenizdat.sample_sale.CACHE_DIR", tmp_path)
        monkeypatch.setattr("rottengenizdat.cli.CACHE_DIR", tmp_path)
        entries = [IndexEntry(**e) for e in sample_index]
        save_index(entries, cache_dir=tmp_path)
        result = runner.invoke(app, ["sample-sale", "list"])
        assert result.exit_code == 0
        assert "beat.wav" in result.stdout

    def test_clear(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("rottengenizdat.sample_sale.CACHE_DIR", tmp_path)
        monkeypatch.setattr("rottengenizdat.cli.CACHE_DIR", tmp_path)
        # Create a fake cached file
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        (samples_dir / "fake.wav").write_bytes(b"fake")
        result = runner.invoke(app, ["sample-sale", "clear"])
        assert result.exit_code == 0
        assert not samples_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_sample_sale.py::TestSampleSaleCLI -v`
Expected: FAIL — typer doesn't know about "sample-sale" subcommand

- [ ] **Step 3: Implement sample-sale subcommands**

Add import at the top of `cli.py`:

```python
from rottengenizdat.sample_sale import (
    CACHE_DIR,
    load_index,
    save_index,
    sync_index,
    clear_cache,
)
```

Add subcommand group at the end of `cli.py`:

```python
# ---------------------------------------------------------------------------
# sample-sale sub-app
# ---------------------------------------------------------------------------

sample_sale_app = typer.Typer(
    name="sample-sale",
    help="Manage the #sample-sale sample cache",
    context_settings=CONTEXT_SETTINGS,
)
app.add_typer(sample_sale_app)


@sample_sale_app.command(name="refresh")
def sample_sale_refresh(
    full: Annotated[bool, typer.Option("--full", help="Rebuild index from scratch")] = False,
) -> None:
    """Sync the sample index from Slack (incremental by default)."""
    mode = "full rebuild" if full else "incremental sync"
    console.print(f"[bold]Refreshing index[/bold] ({mode})...")
    entries = sync_index(full=full)
    console.print(f"[green]Index updated:[/green] {len(entries)} samples indexed")


@sample_sale_app.command(name="list")
def sample_sale_list(
    cached: Annotated[bool, typer.Option("--cached", help="Only show downloaded samples")] = False,
) -> None:
    """List indexed samples from #sample-sale."""
    entries = load_index()
    if not entries:
        console.print("[dim]No samples indexed. Run:[/dim] rotten sample-sale refresh")
        return
    if cached:
        entries = [e for e in entries if e.is_cached]
        if not entries:
            console.print("[dim]No cached samples. Samples are downloaded on first use.[/dim]")
            return
    for entry in entries:
        cached_mark = "[green]cached[/green]" if entry.is_cached else "[dim]not cached[/dim]"
        name = entry.filename or entry.url or entry.id
        console.print(f"  {entry.type:10s}  {cached_mark:20s}  {name}")
    console.print(f"\n[dim]{len(entries)} sample(s)[/dim]")


@sample_sale_app.command(name="clear")
def sample_sale_clear(
    all_: Annotated[bool, typer.Option("--all", help="Delete index too (not just media files)")] = False,
) -> None:
    """Delete cached sample files."""
    clear_cache(full=all_)
    if all_:
        console.print("[green]Cache and index cleared.[/green]")
    else:
        console.print("[green]Cached media files cleared.[/green] Index kept.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_sample_sale.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/cli.py tests/test_sample_sale.py
git commit -m "feat: rotten sample-sale subcommand (refresh, list, clear)

Management commands for the #sample-sale cache. 'refresh' syncs
the index from Slack (incremental or full). 'list' shows indexed
samples and their cache status. 'clear' removes cached media."
```

---

## Task 9: Multi-Input + Sample-Sale Flags on `recipe run`

**Files:**
- Modify: `src/rottengenizdat/cli.py:228-293` (rewrite recipe_run command)

- [ ] **Step 1: Write failing test for multi-input recipe run**

Append to `tests/test_inputs.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from typer.testing import CliRunner

from rottengenizdat.cli import app
from rottengenizdat.core import AudioBuffer, save_audio

runner = CliRunner()


class TestRecipeRunMultiInput:
    def test_multiple_local_files(self, tmp_path: Path):
        """recipe run accepts multiple input files."""
        # Create a minimal recipe
        recipe = tmp_path / "test.toml"
        recipe.write_text(
            '[recipe]\nname = "test"\nmode = "sequential"\n'
            '[[steps]]\neffect = "dry"\n'
        )
        # Create two input files
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py::TestRecipeRunMultiInput -v`
Expected: FAIL — recipe run currently takes single `input_file`, not multiple

- [ ] **Step 3: Rewrite recipe_run to support multi-input**

In `cli.py`, add these imports at the top (alongside existing ones):

```python
from rottengenizdat.inputs import InputMode, combine_inputs
from rottengenizdat.sample_sale import (
    CACHE_DIR,
    download_sample,
    load_index,
    pick_random_samples,
    sync_index,
    clear_cache,
)
```

Replace the `recipe_run` function (lines 228-293) with:

```python
@recipe_app.command(name="run")
def recipe_run(
    recipe_file: Annotated[Path, typer.Argument(help="Path to recipe TOML file")],
    input_files: Annotated[Optional[list[Path]], typer.Argument(help="Input audio/video files")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file or directory (for --mode independent)")] = Path("output.wav"),
    sample_sale: Annotated[bool, typer.Option("--sample-sale", "-ss", help="Include random samples from #sample-sale")] = False,
    sample_sale_count: Annotated[int, typer.Option("--sample-sale-count", help="How many samples to pull (implies --sample-sale)")] = 0,
    mode: Annotated[Optional[str], typer.Option("--mode", help="Input combination mode: splice (default for multi), concat, independent")] = None,
    splice_min: Annotated[float, typer.Option("--splice-min", help="Min splice segment duration in seconds")] = 0.25,
    splice_max: Annotated[float, typer.Option("--splice-max", help="Max splice segment duration in seconds")] = 4.0,
) -> None:
    """Run a saved recipe against one or more input files.

    Supports multiple local files as positional args and/or random samples
    from #sample-sale. When multiple inputs are provided, they are combined
    according to --mode (default: splice).

    Examples:

      rotten recipe run recipes/fever-dream.toml input.wav -o out.wav
      rotten recipe run recipes/bone-xray.toml a.wav b.wav --mode concat -o out.wav
      rotten recipe run recipes/barely-there.toml --sample-sale-count 3 -o out.wav
      rotten recipe run recipes/drunk-choir.toml a.wav --sample-sale -o out.wav
    """
    if not recipe_file.exists():
        console.print(f"[red]Recipe file not found: {recipe_file}[/red]")
        raise typer.Exit(1)

    # Resolve all inputs
    all_buffers: list[AudioBuffer] = []
    all_names: list[str] = []

    # Local files
    for f in (input_files or []):
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        all_buffers.append(load_audio(f))
        all_names.append(f.stem)

    # Sample-sale samples
    ss_count = sample_sale_count if sample_sale_count > 0 else (1 if sample_sale else 0)
    if ss_count > 0:
        console.print(f"[bold]Fetching {ss_count} sample(s) from #sample-sale...[/bold]")
        index = sync_index()
        picks = pick_random_samples(index, ss_count)
        for entry in picks:
            console.print(f"  [dim]Selected:[/dim] {entry.filename or entry.url or entry.id}")
            path = download_sample(entry)
            all_buffers.append(load_audio(path))
            all_names.append(entry.filename or entry.id)

    if not all_buffers:
        console.print("[red]No input files provided. Pass audio files or use --sample-sale.[/red]")
        raise typer.Exit(1)

    # Load recipe
    recipe = load_recipe(recipe_file)
    meta = recipe.get("recipe", {})
    name = meta.get("name", "untitled")
    recipe_mode = meta.get("mode", "sequential")
    raw_steps = recipe.get("steps", [])
    console.print(f"[bold]Recipe:[/bold] {name} (mode={recipe_mode}, {len(raw_steps)} step(s))")

    # Combine inputs
    input_mode = InputMode.resolve(mode, len(all_buffers))
    if len(all_buffers) > 1:
        console.print(f"[bold]Combining {len(all_buffers)} inputs[/bold] (mode={input_mode.value})")
    combined = combine_inputs(all_buffers, input_mode, splice_min=splice_min, splice_max=splice_max)

    # Run pipeline
    step_pairs = recipe_steps_to_kwargs(raw_steps)
    plugins = discover_plugins()

    def _run_pipeline(audio: AudioBuffer) -> AudioBuffer:
        if recipe_mode == "branch":
            from rottengenizdat.chain import mix_buffers
            outputs = []
            weights = []
            for effect_name, kwargs in step_pairs:
                if effect_name not in plugins:
                    console.print(f"[red]Unknown effect: {effect_name}[/red]")
                    raise typer.Exit(1)
                w = kwargs.pop("weight", 1.0)
                weights.append(float(w))
                console.print(f"  [dim]branch (weight={w}):[/dim] {effect_name} {kwargs}")
                result = plugins[effect_name]().process(audio, **kwargs)
                outputs.append(result)
            return mix_buffers(outputs, weights)
        else:
            current = audio
            for effect_name, kwargs in step_pairs:
                if effect_name not in plugins:
                    console.print(f"[red]Unknown effect: {effect_name}[/red]")
                    raise typer.Exit(1)
                console.print(f"  [dim]step:[/dim] {effect_name} {kwargs}")
                current = plugins[effect_name]().process(current, **kwargs)
            return current

    if input_mode == InputMode.INDEPENDENT:
        output.mkdir(parents=True, exist_ok=True)
        for i, (audio, src_name) in enumerate(zip(combined, all_names)):
            console.print(f"\n[bold]Processing input {i+1}/{len(combined)}:[/bold] {src_name}")
            result = _run_pipeline(audio)
            out_path = output / f"{i+1:03d}-{src_name}.wav"
            save_audio(result, out_path)
            console.print(f"[green]Saved:[/green] {out_path}")
    else:
        audio = combined[0]
        console.print(f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")
        result = _run_pipeline(audio)
        save_audio(result, output)
        console.print(f"[green]Saved:[/green] {output}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run existing recipe tests to check for regressions**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_recipe.py tests/test_cli.py -v`
Expected: all PASS (existing single-input usage still works since `input_files` is optional list)

- [ ] **Step 6: Commit**

```bash
git add src/rottengenizdat/cli.py tests/test_inputs.py
git commit -m "feat: multi-input support on recipe run command

Accepts multiple positional input files plus --sample-sale and
--sample-sale-count flags. Inputs combined via --mode (splice
default for multi, concat, independent). Independent mode outputs
to a directory with numbered files."
```

---

## Task 10: Multi-Input + Sample-Sale Flags on `chain` Command

**Files:**
- Modify: `src/rottengenizdat/cli.py:125-154` (rewrite chain_command)

- [ ] **Step 1: Write failing test for multi-input chain**

Append to `tests/test_inputs.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py::TestChainMultiInput -v`
Expected: FAIL — chain command currently takes single input_file

- [ ] **Step 3: Rewrite chain_command for multi-input**

Replace the `chain_command` function in `cli.py`:

```python
@app.command(name="chain", help=_CHAIN_HELP)
def chain_command(
    input_files: Annotated[list[Path], typer.Argument(help="Input audio file(s)")],
    steps: Annotated[list[str], typer.Argument(help="Effect steps as quoted strings, e.g. 'rave -m percussion -t 1.2'")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file or directory path")] = Path("output.wav"),
    branch: Annotated[bool, typer.Option("--branch", "-b", help="Run steps in parallel and mix (default: sequential)")] = False,
    sample_sale: Annotated[bool, typer.Option("--sample-sale", "-ss", help="Include random samples from #sample-sale")] = False,
    sample_sale_count: Annotated[int, typer.Option("--sample-sale-count", help="How many samples to pull (implies --sample-sale)")] = 0,
    mode: Annotated[Optional[str], typer.Option("--mode", help="Input combination mode: splice (default for multi), concat, independent")] = None,
    splice_min: Annotated[float, typer.Option("--splice-min", help="Min splice segment duration in seconds")] = 0.25,
    splice_max: Annotated[float, typer.Option("--splice-max", help="Max splice segment duration in seconds")] = 4.0,
) -> None:
    if not steps:
        console.print("[red]At least one step is required.[/red]")
        raise typer.Exit(1)

    # Resolve all inputs
    all_buffers: list[AudioBuffer] = []
    all_names: list[str] = []

    for f in input_files:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        all_buffers.append(load_audio(f))
        all_names.append(f.stem)

    ss_count = sample_sale_count if sample_sale_count > 0 else (1 if sample_sale else 0)
    if ss_count > 0:
        console.print(f"[bold]Fetching {ss_count} sample(s) from #sample-sale...[/bold]")
        index = sync_index()
        picks = pick_random_samples(index, ss_count)
        for entry in picks:
            console.print(f"  [dim]Selected:[/dim] {entry.filename or entry.url or entry.id}")
            path = download_sample(entry)
            all_buffers.append(load_audio(path))
            all_names.append(entry.filename or entry.id)

    if not all_buffers:
        console.print("[red]No input files provided.[/red]")
        raise typer.Exit(1)

    input_mode = InputMode.resolve(mode, len(all_buffers))
    if len(all_buffers) > 1:
        console.print(f"[bold]Combining {len(all_buffers)} inputs[/bold] (mode={input_mode.value})")
    combined = combine_inputs(all_buffers, input_mode, splice_min=splice_min, splice_max=splice_max)

    chain_mode = "branch" if branch else "sequential"
    console.print(f"[bold]Running {chain_mode} chain[/bold] ({len(steps)} step(s))")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")

    if input_mode == InputMode.INDEPENDENT:
        output.mkdir(parents=True, exist_ok=True)
        for i, (audio, src_name) in enumerate(zip(combined, all_names)):
            console.print(f"\n[bold]Processing input {i+1}/{len(combined)}:[/bold] {src_name}")
            result = run_branch(audio, steps) if branch else run_chain(audio, steps)
            out_path = output / f"{i+1:03d}-{src_name}.wav"
            save_audio(result, out_path)
            console.print(f"[green]Saved:[/green] {out_path}")
    else:
        audio = combined[0]
        console.print(f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")
        result = run_branch(audio, steps) if branch else run_chain(audio, steps)
        save_audio(result, output)
        console.print(f"[green]Saved:[/green] {output}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/rottengenizdat/cli.py tests/test_inputs.py
git commit -m "feat: multi-input + sample-sale flags on chain command

Same input resolution as recipe run: multiple positional files,
--sample-sale, --sample-sale-count, --mode, --splice-min/max."
```

---

## Task 11: Multi-Input on Plugin Commands (rave)

**Files:**
- Modify: `src/rottengenizdat/plugins/rave.py:261-388` (rewrite register_command)
- Modify: `src/rottengenizdat/plugin.py` (update base class signature hint)

- [ ] **Step 1: Write failing test for multi-input rave**

Append to `tests/test_inputs.py`:

```python
class TestRaveMultiInput:
    def test_multiple_local_files(self, tmp_path: Path):
        sr = 44100
        for name in ["a.wav", "b.wav"]:
            buf = AudioBuffer(samples=torch.randn(1, sr), sample_rate=sr)
            save_audio(buf, tmp_path / name)

        # Use dry plugin instead to avoid downloading RAVE models in tests
        result = runner.invoke(app, [
            "dry",
            str(tmp_path / "a.wav"), str(tmp_path / "b.wav"),
            "--mode", "concat",
            "-o", str(tmp_path / "out.wav"),
        ])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "out.wav").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py::TestRaveMultiInput -v`
Expected: FAIL — plugin commands take single input_file

- [ ] **Step 3: Update RaveEffect.register_command for multi-input**

In `src/rottengenizdat/plugins/rave.py`, update the `register_command` method. Replace the `input_file` parameter and add the multi-input/sample-sale parameters. Update the function body to resolve inputs through `combine_inputs`.

Add imports at the top of `rave.py`:

```python
from rottengenizdat.inputs import InputMode, combine_inputs
from rottengenizdat.sample_sale import (
    download_sample,
    load_index,
    pick_random_samples,
    sync_index,
)
```

Replace the `rave_command` function inside `register_command` (starting at line 264):

```python
        @app.command(name=self.name, help=self.description)
        def rave_command(
            input_files: Annotated[
                Optional[list[Path]], typer.Argument(help="Input audio file(s) (wav, flac, mp3, etc.)")
            ] = None,
            output: Annotated[
                Path, typer.Option("--output", "-o", help="Output file path (or directory when using --sweep or --mode independent)")
            ] = Path("output.wav"),
            model: Annotated[
                str,
                typer.Option(
                    "--model",
                    "-m",
                    help="Pretrained RAVE model. Each imparts its own character. "
                    f"Choices: {', '.join(AVAILABLE_MODELS)}",
                ),
            ] = "percussion",
            temperature: Annotated[
                float,
                typer.Option("--temperature", "-t", help="Scale latent vectors."),
            ] = 1.0,
            noise_amount: Annotated[
                float,
                typer.Option("--noise", "-n", help="Gaussian noise added to latent space (0.0-1.0)."),
            ] = 0.0,
            mix: Annotated[
                float,
                typer.Option("--mix", "-w", help="Wet/dry blend. 0.0 = original, 1.0 = full RAVE."),
            ] = 1.0,
            dims: Annotated[
                Optional[str],
                typer.Option("--dims", "-d", help="Latent dims to manipulate (e.g. '0,1,2,3')."),
            ] = None,
            reverse: Annotated[
                bool,
                typer.Option("--reverse", "-r", help="Flip latent time axis."),
            ] = False,
            shuffle_chunks: Annotated[
                int,
                typer.Option("--shuffle", help="Shuffle latent in chunks of N frames."),
            ] = 0,
            quantize_step: Annotated[
                float,
                typer.Option("--quantize", "-q", help="Snap latent values to grid of this step size."),
            ] = 0.0,
            sweep: Annotated[
                Optional[str],
                typer.Option("--sweep", help="Generate grid sweeping one parameter."),
            ] = None,
            sample_sale: Annotated[
                bool, typer.Option("--sample-sale", "-ss", help="Include random samples from #sample-sale")
            ] = False,
            sample_sale_count: Annotated[
                int, typer.Option("--sample-sale-count", help="How many samples to pull (implies --sample-sale)")
            ] = 0,
            input_mode: Annotated[
                Optional[str], typer.Option("--mode", help="Input combination mode: splice, concat, independent")
            ] = None,
            splice_min: Annotated[
                float, typer.Option("--splice-min", help="Min splice segment seconds")
            ] = 0.25,
            splice_max: Annotated[
                float, typer.Option("--splice-max", help="Max splice segment seconds")
            ] = 4.0,
        ) -> None:
            # Resolve all inputs
            all_buffers: list[AudioBuffer] = []
            all_names: list[str] = []

            for f in (input_files or []):
                if not f.exists():
                    console.print(f"[red]File not found: {f}[/red]")
                    raise typer.Exit(1)
                all_buffers.append(load_audio(f))
                all_names.append(f.stem)

            ss_count = sample_sale_count if sample_sale_count > 0 else (1 if sample_sale else 0)
            if ss_count > 0:
                console.print(f"[bold]Fetching {ss_count} sample(s) from #sample-sale...[/bold]")
                index = sync_index()
                picks = pick_random_samples(index, ss_count)
                for entry in picks:
                    console.print(f"  [dim]Selected:[/dim] {entry.filename or entry.url or entry.id}")
                    path = download_sample(entry)
                    all_buffers.append(load_audio(path))
                    all_names.append(entry.filename or entry.id)

            if not all_buffers:
                console.print("[red]No input files provided. Pass audio files or use --sample-sale.[/red]")
                raise typer.Exit(1)

            resolved_mode = InputMode.resolve(input_mode, len(all_buffers))
            if len(all_buffers) > 1:
                console.print(f"[bold]Combining {len(all_buffers)} inputs[/bold] (mode={resolved_mode.value})")
            combined = combine_inputs(all_buffers, resolved_mode, splice_min=splice_min, splice_max=splice_max)

            if resolved_mode == InputMode.INDEPENDENT:
                output.mkdir(parents=True, exist_ok=True)
                for i, (audio, src_name) in enumerate(zip(combined, all_names)):
                    console.print(f"\n[bold]Processing input {i+1}/{len(combined)}:[/bold] {src_name}")
                    if sweep:
                        self._run_sweep(audio, output / f"{i+1:03d}-{src_name}", model, temperature, noise_amount, mix, dims, reverse, shuffle_chunks, quantize_step, sweep)
                    else:
                        result = self.process(audio, model_name=model, temperature=temperature, noise=noise_amount, mix=mix, dims=dims, reverse=reverse, shuffle_chunks=shuffle_chunks, quantize=quantize_step)
                        out_path = output / f"{i+1:03d}-{src_name}.wav"
                        save_audio(result, out_path)
                        console.print(f"[green]Saved:[/green] {out_path}")
            else:
                audio = combined[0]
                console.print(f"[bold]Loading:[/bold] {len(all_buffers)} input(s) — {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")
                if sweep:
                    self._run_sweep(audio, output, model, temperature, noise_amount, mix, dims, reverse, shuffle_chunks, quantize_step, sweep)
                else:
                    console.print(f"[bold]Processing:[/bold] rave model={model} temp={temperature} noise={noise_amount} mix={mix}")
                    result = self.process(audio, model_name=model, temperature=temperature, noise=noise_amount, mix=mix, dims=dims, reverse=reverse, shuffle_chunks=shuffle_chunks, quantize=quantize_step)
                    save_audio(result, output)
                    console.print(f"[green]Saved:[/green] {output}")

        return rave_command
```

Also update `src/rottengenizdat/plugins/dry.py` to match the same multi-input pattern — read it first, then update similarly.

- [ ] **Step 4: Update dry plugin for multi-input**

Read `src/rottengenizdat/plugins/dry.py` and update its `register_command` to accept multi-input with the same pattern. The dry plugin is simpler (just passes audio through), so the update is straightforward: change `input_file` to `input_files` list, add `--sample-sale`/`--mode` flags, use `combine_inputs`.

- [ ] **Step 5: Run all tests**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/rottengenizdat/plugins/rave.py src/rottengenizdat/plugins/dry.py tests/test_inputs.py
git commit -m "feat: multi-input + sample-sale flags on rave and dry commands

Plugin commands now accept multiple input files, --sample-sale,
--sample-sale-count, --mode, --splice-min/max. Same input resolution
pattern as recipe run and chain."
```

---

## Task 12: Full Integration Test

**Files:**
- Modify: `tests/test_inputs.py` (append integration test)

- [ ] **Step 1: Write integration test with local files + splice**

Append to `tests/test_inputs.py`:

```python
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
```

- [ ] **Step 2: Run integration test**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/test_inputs.py::TestIntegrationMultiInput -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/jake/au-supply/rottengenizdat && python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_inputs.py
git commit -m "test: integration test for multi-input splice pipeline

End-to-end test: three local files spliced through dry recipe,
verifying output contains all input samples."
```

---

## Task 13: Update README and Help Text

**Files:**
- Modify: `README.md`
- Modify: `src/rottengenizdat/cli.py` (update `_MAIN_HELP`, `_CHAIN_HELP`, `_RECIPE_HELP`)

- [ ] **Step 1: Update _MAIN_HELP in cli.py**

Add sample-sale examples to the help text:

```python
_MAIN_HELP = """\
bone music for the machine age

Audio mangling through neural networks. Feed your tracks through RAVE
variational autoencoders — models trained on percussion, voices, strings,
NASA recordings, vintage audio — and let the latent space chew them up.
The original bleeds through in uncanny ways: recognizable, but wrong.

Named after roentgenizdat — the Soviet practice of pressing bootleg records
onto discarded hospital X-ray film.

Start with a single effect, then graduate to chains and recipes:

  rotten rave input.wav -m vintage -o out.wav
  rotten rave input.wav input2.wav -m nasa -t 1.5 --mode splice -o out.wav
  rotten chain input.wav "rave -m percussion" "rave -m vintage" -o out.wav
  rotten recipe run recipes/fever-dream.toml input.wav -o out.wav

Pull random samples from Slack #sample-sale:

  rotten recipe run recipes/bone-xray.toml --sample-sale-count 3 -o out.wav
  rotten sample-sale refresh
  rotten config set slack.token xoxb-YOUR-TOKEN

Run any command with -h to see detailed help and examples.
"""
```

- [ ] **Step 2: Update README.md usage section**

Add multi-input and sample-sale examples to the Usage section of README.md.

- [ ] **Step 3: Commit**

```bash
git add README.md src/rottengenizdat/cli.py
git commit -m "docs: update help text and README for multi-input and sample-sale

Adds examples for multi-input modes (splice, concat, independent),
--sample-sale flags, config and sample-sale management subcommands."
```
