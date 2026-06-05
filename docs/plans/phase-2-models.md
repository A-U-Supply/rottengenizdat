# Phase 2: Model Plugin Expansion

Date: 2026-06-05

## Summary

Add five new neural audio plugins to rottengenizdat, following the same
plugin architecture as RAVE (`AudioEffect` → `register_command`).

Each plugin is a self-contained `src/rottengenizdat/plugins/<name>.py`
that registers as a `rotten <name>` subcommand with its own knobs.

## Priority order

| # | Model | Effort | Why first |
|---|-------|--------|-----------|
| 1 | **VampNet** | Medium | Mask-ratio knob is the most intuitive "mangle" control; tiny models available; pure PyTorch, no extra deps |
| 2 | **EnCodec** | Small | Ships inside AudioCraft; trivial "compress until it breaks" effect; also needed as tokenizer for VampNet and MusicGen |
| 3 | **MusicGen** (AudioCraft) | Medium | Text-to-music generation; melody-conditioned re-imagination; huge wow factor |
| 4 | **Riffusion** | Large | Stable Diffusion pipeline is heavy; spectrogram round-trip is gloriously broken |
| 5 | **DDSP** | Medium | Timbre transfer; pip-installable with pretrained models; different vibe from the rest |

## 1. VampNet

**Repo:** https://github.com/hugofloresgarcia/vampnet
**Package:** `vampnet` (PyPI)
**Paper:** masked acoustic token modeling

**What it does:** Represents audio as EnCodec tokens, randomly masks
portions, and has a transformer hallucinate the gaps. Audio mad-libs.

**Key knobs:**
- `--mask-ratio` (0.0–1.0): How much to replace. 0.1 = subtle variation, 0.5 = half replaced, 0.9 = fever dream
- `--mask-pattern` (random, periodic, contiguous): How gaps are distributed
- `--temperature` (0.0–2.0+): Sampling temperature for token prediction
- `--prefix-s` (float): How many seconds of the original to keep unmasked as a "seed"
- `--periodic-p` (int): Period for periodic masking (creates rhythmic patterns)
- `--num-steps` (int): Iterative decoding steps (more = cleaner, fewer = glitchier)

**CLI design:**
```
rotten vamp input.wav --mask-ratio 0.7 -m random -t 1.2 -o out.wav
rotten vamp input.wav --mask-ratio 0.9 --periodic-p 4 -o glitch-drums.wav
rotten vamp input.wav --mask-ratio 0.5 --prefix-s 1.0 -o seeded.wav
```

**Model download:** HuggingFace `hugofloresgarcia/vampnet` (~400MB).
Auto-download on first use, cache at `~/.cache/rottengenizdat/models/vampnet/`.

**Dependencies:** `vampnet`, which pulls in `transformers`, `torch`, etc.
Already compatible with our stack.

## 2. EnCodec

**Package:** `audiocraft` (or `encodec` directly)
**Repo:** part of facebookresearch/audiocraft

**What it does:** Neural audio codec. Compress & reconstruct. Crank the
bandwidth down and the artifacts become the art.

**Key knobs:**
- `--bandwidth` (1.5, 3.0, 6.0, 12.0, 24.0): Lower = more artifacts
- `--bandwidth` 1.5 = robotic, 3.0 = crunchy mp3, 6.0 = transparent-ish

**CLI design:**
```
rotten encodec input.wav --bandwidth 1.5 -o crushed.wav
rotten encodec input.wav --bandwidth 3.0 -o lo-fi.wav
```

**Note:** EnCodec is also the tokenizer used by VampNet and MusicGen.
Making it a standalone plugin is trivial and useful on its own.

**Dependencies:** `audiocraft` (~200MB install, shared with MusicGen).

## 3. MusicGen (AudioCraft)

**Package:** `audiocraft`
**Repo:** facebookresearch/audiocraft

**What it does:** Text-to-music generation. Also supports
melody-conditioned generation: feed it a melody (chromagram from your
audio) + a text prompt, get a re-imagined version.

**Key knobs:**
- `--prompt` / `-p`: Text description ("cartoon brass band", "dark ambient drone")
- `--duration` (int): Seconds to generate
- `--temperature` (0.0–2.0): Sampling temperature
- `--top-k` / `--top-p`: Nucleus sampling controls
- `--cfg-coef` (1.0–7.0): Classifier-free guidance strength
- `--melody` (path): Optional input file for melody conditioning

**Models:** `small` (300M params), `medium` (1.5B), `large` (3.3B),
`melody` (1.5B with chromagram conditioning). Auto-download via
HuggingFace.

**CLI design:**
```
rotten musicgen -p "haunted carnival waltz" --duration 8 -o carnival.wav
rotten musicgen -p "dry funeral drums" --duration 4 -t 1.5 -o drums.wav
rotten musicgen -p "spectral choir" --melody input.wav --duration 6 -o choir.wav
```

**Dependencies:** `audiocraft` (same package as EnCodec).

## 4. Riffusion

**Repo:** https://github.com/riffusion/riffusion
**Package:** `diffusers` + `torch`

**What it does:** Renders audio as a spectrogram image, runs it through
Stable Diffusion with a text prompt, converts back to audio. The
spectrogram→image→SD→spectrogram→audio round-trip adds glorious artifacts.

**Key knobs:**
- `--prompt` / `-p`: Text guidance ("saxophone solo", "ocean waves")
- `--strength` (0.0–1.0): Denoising strength. 0.2 = subtle shift, 0.8 = unrecognizable
- `--guidance-scale` (1–20): How strongly to follow the prompt
- `--steps` (int): Diffusion steps (fewer = faster but rougher)
- `--seed` (int): Reproducible noise seed

**CLI design:**
```
rotten riff input.wav -p "haunted cathedral organ" --strength 0.6 -o organ.wav
rotten riff input.wav -p "underwater whale song" --strength 0.8 --guidance-scale 10 -o whale.wav
```

**Dependencies:** Heavy — `diffusers`, `transformers`, `accelerate`,
`torchvision`, etc. ~4GB model download for SD 1.5. Warning in docs
about first-run download size.

## 5. DDSP

**Package:** `ddsp` (Google Magenta)
**Repo:** https://github.com/magenta/ddsp

**What it does:** Timbre transfer. Decomposes audio into pitch +
loudness + timbre, swaps the timbre for a different instrument, resynthesizes.
Violin → flute, voice → trumpet, anything → anything.

**Key knobs:**
- `--instrument` (flute, trumpet, violin, tenor_sax, etc.)
- `--loudness-fidelity` (0.0–1.0): How much to preserve the original dynamics
- `--pitch-fidelity` (0.0–1.0): How much to preserve the original pitch contour
- `--mix` (0.0–1.0): Wet/dry blend

**CLI design:**
```
rotten ddsp input.wav --instrument trumpet -o trumpet.wav
rotten ddsp input.wav --instrument flute --loudness-fidelity 0.3 -o breathy.wav
rotten ddsp input.wav --instrument violin --mix 0.5 -o half-violin.wav
```

**Dependencies:** `ddsp`, `tensorflow` (yes, TF — only model in the stack
that uses it, annoying but manageable).

## Implementation approach

Each plugin follows the same pattern:

1. **Create** `src/rottengenizdat/plugins/<name>.py`
2. **Subclass** `AudioEffect` with `name`, `description`, `process()`, `register_command()`
3. **Add** a `<name>` subcommand to the CLI via plugin discovery
4. **Auto-download** models on first use, cache in `~/.cache/rottengenizdat/models/<name>/`
5. **Tests** in `tests/test_<name>.py` with mocked models
6. **Document** in `_<NAME>_HELP` constant with examples

## Dependency strategy

- VampNet: `pip install vampnet` (~400MB models)
- EnCodec: `pip install audiocraft` (shared with MusicGen)
- MusicGen: same `audiocraft` package, models via HuggingFace (1–3GB each)
- Riffusion: `pip install diffusers transformers` (~4GB Stable Diffusion)
- DDSP: `pip install ddsp tensorflow` (TF only, ~200MB models)

All declared as optional dependencies in `pyproject.toml` so users can
pick which plugins to install:

```toml
[project.optional-dependencies]
vampnet = ["vampnet"]
audiocraft = ["audiocraft"]
riffusion = ["diffusers", "transformers", "accelerate"]
ddsp = ["ddsp", "tensorflow"]
all-plugins = ["rottengenizdat[vampnet,audiocraft,riffusion,ddsp]"]
```
