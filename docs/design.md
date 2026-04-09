# rottengenizdat — Design Document

Date: 2026-04-09

## Overview

A Python CLI for audio transformation through ML models. Takes existing music (loops, stems, full tracks) and runs it through neural networks that mangle, translate, and re-imagine it. The aesthetic embraces artifacts — the interesting zone is where you recognize the input but barely, or in weird ways.

Named after [roentgenizdat](https://en.wikipedia.org/wiki/Ribs_(recordings)): Soviet-era bootleg Western music pressed onto discarded hospital X-ray film. The records sounded warped and ghostly, and you could see someone's skeleton through the translucent disc while it played.

## Design Principles

- **Pipeline-first**: effects are chainable (`audio in → knobs → audio out`)
- **Embrace artifacts**: the models being "bad at it" is a feature, not a bug
- **Knobs over presets**: expose the parameters, let the user explore
- **Batch exploration**: sweep parameters and generate grids of outputs for comparison
- **Plugin architecture**: each model is self-contained, new models are easy to add

## CLI Design

Built with **typer** (CLI parsing) and **rich** (terminal output).

### Interaction modes (in priority order)

1. **Pipeline** — chain transforms:
   ```
   rotten input.wav rave --model whale --temperature 1.2 -o out.wav
   ```

2. **Batch/sweep** — explore parameter space:
   ```
   rotten input.wav rave --model whale --sweep temperature=0.3,0.7,1.0,1.5,2.0 -o grid/
   ```

3. **Presets** (future) — named combinations of models and parameters:
   ```
   rotten input.wav --preset "xray-hymnal" -o out.wav
   ```

### Console banner

The CLI displays this banner on `--help` and `--version`:

```
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
```

## Project Structure

```
rottengenizdat/
├── src/rottengenizdat/
│   ├── __init__.py
│   ├── cli.py              # typer app, discovers & registers plugins
│   ├── core.py             # AudioBuffer type, I/O, shared utilities
│   ├── plugin.py           # base plugin interface
│   ├── banner.py           # console banner art
│   └── plugins/
│       ├── __init__.py
│       ├── rave.py          # RAVE plugin (phase 1)
│       └── ...              # future plugins
├── tests/
├── docs/
│   └── design.md            # this file
├── pyproject.toml
└── README.md
```

## Plugin Interface

Each model plugin implements a minimal interface:

```python
class AudioEffect:
    """Base class for all audio effect plugins."""
    name: str                    # CLI subcommand name
    description: str             # help text for --help

    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        """Transform audio. All knobs passed as kwargs."""
        ...
```

- Plugins are auto-discovered from the `plugins/` directory
- Each plugin registers as a typer subcommand
- Plugins declare their own CLI arguments (knobs)
- `AudioBuffer` wraps a torch tensor + sample rate + metadata

## Model Catalog

### Phase 1: RAVE

**RAVE** (Realtime Audio Variational autoEncoder) — IRCAM, Antoine Caillon

- **What it does**: Encodes audio into a compressed latent representation, lets you manipulate the latent vectors, and decodes back to audio. Even a straight encode→decode round-trip produces uncanny "almost right" audio. Then you start scaling latent dimensions, interpolating, doing random walks through latent space.
- **Knobs**: model selection (pretrained: whale, singing, guitar, saxophone, church organ, speech, birds, water, etc.), temperature, latent dimension scaling, interpolation between two audio clips
- **Why first**: immediate results, pretrained models available, latent space manipulation gives fine-grained control from subtle to extreme
- **Repo**: https://github.com/acids-ircam/RAVE
- **Package**: `acids-rave`
- **Pretrained models**: https://acids-ircam.github.io/rave_models_download
- **Also available as**: VST (rave_vst), Max/MSP external (nn~), Neutone FX plugin

### Future: Timbre/Style Transfer

**DDSP** (Differentiable Digital Signal Processing) — Google Magenta

- **What it does**: Timbre transfer. Re-synthesizes audio as a different instrument. Violin → flute, voice → trumpet. Gloriously uncanny in the middle ranges.
- **Knobs**: target timbre, loudness fidelity, pitch fidelity
- **Why**: explicit knobs for how much to transfer, produces recognizable-but-wrong output
- **Repo**: https://github.com/magenta/ddsp

**NSynth** (Neural Synthesizer) — Google Magenta

- **What it does**: Interpolates between instrument timbres. Blend a guitar with a cat at 60/40.
- **Knobs**: source instruments, blend ratio, note parameters
- **Why**: the OG weird ML audio tool. Produces genuinely alien timbres.
- **Repo**: https://github.com/magenta/magenta (part of magenta)

### Future: Re-imagination

**MusicGen** — Meta / AudioCraft

- **What it does**: Generates music conditioned on text descriptions and/or melody (chromagram). Feed it your track + "cartoon brass band" and it re-imagines the harmonic contour in that style.
- **Knobs**: text description, conditioning strength, temperature, duration
- **Why**: the "add a cartoon trumpet to this track" effect. Text-guided style transfer.
- **Repo**: https://github.com/facebookresearch/audiocraft

**Riffusion** — Stable Diffusion on spectrograms

- **What it does**: Generates music by generating spectrogram images with Stable Diffusion, then converting back to audio. Text-guided. The spectrogram round-trip adds its own flavor of wrong.
- **Knobs**: text prompt, guidance scale, denoising strength (how much of the original to preserve)
- **Why**: spectrograms as images is conceptually bonkers and the results show it
- **Repo**: https://github.com/riffusion/riffusion

**VampNet** — masked acoustic token modeling

- **What it does**: Represents audio as neural tokens, masks out portions, and lets the model fill in the blanks. Audio mad-libs.
- **Knobs**: mask ratio (how much to replace), mask pattern (random, periodic, contiguous), temperature
- **Why**: the mask ratio is a perfect "how much to mangle" knob. Low mask = subtle variation. High mask = hallucination.
- **Repo**: https://github.com/hugofloresgarcia/vampnet

### Future: Decompose + Mangle

**Demucs** — Meta

- **What it does**: Source separation. Splits a full mix into stems (vocals, drums, bass, other). Not a transformation itself, but enables surgical processing — run different effects on each stem and recombine.
- **Knobs**: stem selection, per-stem effect routing
- **Why**: "make the vocals sound like a whale but keep the drums." Prerequisite for targeted transformations.
- **Repo**: https://github.com/facebookresearch/demucs

**Basic Pitch** — Spotify

- **What it does**: Audio → MIDI transcription. Extracts melody/notes as MIDI, which can then be re-synthesized through anything.
- **Knobs**: onset threshold, note confidence, pitch bend sensitivity
- **Why**: the extraction step for melody-based re-synthesis pipelines
- **Repo**: https://github.com/spotify/basic-pitch

### Future: Compression/Bottleneck

**Encodec** — Meta

- **What it does**: Neural audio codec. Compresses and reconstructs audio with a learned codec. Crank the compression and the reconstruction artifacts become the art. Your track through a neural bottleneck.
- **Knobs**: bandwidth/bitrate (lower = more artifacts), number of codebook levels
- **Why**: the simplest "make it weird" effect. Just compress until it sounds wrong.
- **Repo**: https://github.com/facebookresearch/audiocraft (part of audiocraft)

### Future: Voice/Vocal

**Bark** — Suno

- **What it does**: Text-to-audio that also does music and sound effects. Feed it lyrics, get unhinged vocal performances.
- **Knobs**: speaker preset, temperature, semantic/coarse/fine token temperatures
- **Why**: unpredictable and often hilarious vocal generation
- **Repo**: https://github.com/suno-ai/bark

**SO-VITS-SVC / RVC** — singing voice conversion

- **What it does**: Converts one singing voice to sound like another. Train on a dataset of a specific voice and it'll convert any vocal input.
- **Knobs**: pitch shift, speaker embedding, feature ratio
- **Why**: make your vocalist sound like someone (or something) else entirely

### Future: Vintage/Glitchy

**SampleRNN**

- **What it does**: Early deep audio RNN. Train on your tracks, get fever-dream continuations. Very raw, very glitchy.
- **Knobs**: temperature, sequence length, priming audio
- **Why**: the oldest and rawest approach. Beautiful artifacts from a simpler time.
- **Repo**: https://github.com/soroushmehr/sampleRNN_ICLR2017

**WaveNet** — DeepMind

- **What it does**: The OG neural audio model. Autoregressive, sample-by-sample. Eerie, slow, produces dreamlike audio.
- **Knobs**: temperature, conditioning signals
- **Why**: historically important, aesthetically unique. Slow generation is part of the charm.

### Also Considered

- **Magenta MusicVAE** — interpolation between musical sequences (MIDI level)
- **Magenta ImprovRNN / DrumsRNN** — already used in midi-bot
- **AudioLDM** — latent diffusion for audio
- **AudioSR** — audio super-resolution (could be used in reverse for intentional degradation)
- **Tortoise TTS** — voice cloning for weird vocal re-synthesis
- **Whisper** — already used in glottisdale for transcription; encoder features could enable other things
- **HuBERT / Wav2Vec2** — speech representations for voice transformation

## Technology Stack

- **Python** — all ML models are Python-native
- **typer** — CLI framework (modern, type-hint-based, built on click)
- **rich** — terminal output (progress bars, tables, panels)
- **uv** — project and dependency management
- **pytest** — testing
- **PyTorch** — ML runtime (shared across all models)
- **torchaudio** — audio I/O and processing

## Integration Points

This tool is designed to eventually plug into the existing au-supply audio ecosystem:

- **glottisdale** — syllable collage library. rottengenizdat output → glottisdale input, or vice versa
- **midi-bot** — Magenta MIDI generation. MIDI output → Basic Pitch extraction → rottengenizdat re-synthesis
- **hymnal-bot** — MIDI-driven vocal tracks. rottengenizdat as a post-processing step
- **puke-box** — synthesis experiments. Cross-pollination of outputs
- **sparagmos** — image destruction with sonification. Image → audio → rottengenizdat → back to image
- **muzzik-bot** — YouTube audio downloader. Downloaded audio → rottengenizdat input
