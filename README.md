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

# rottengenizdat

Audio transformation through ML models. A CLI for taking existing music — loops, stems, full tracks — and running it through neural networks that mangle, translate, and re-imagine it. The aesthetic embraces artifacts. Named after [roentgenizdat](https://en.wikipedia.org/wiki/Ribs_(recordings)), the Soviet practice of pressing bootleg rock and roll onto discarded hospital X-ray film. Your song, re-pressed through a medium that transforms it, where the original bleeds through in uncanny ways.

## What it does

Pipeline-chainable audio effects powered by ML models, ranging from subtle timbre shifts to full re-imagination:

```
rotten input.wav rave --model whale --temperature 1.2 -o out.wav
rotten input.wav rave --model whale --sweep temperature=0.5,1.0,1.5 -o grid/
```

## Models & Effects

See [docs/design.md](docs/design.md) for the full catalog of planned models and the design spec.

### Phase 1
- **RAVE** (IRCAM) — Realtime Audio Variational autoEncoder. Encode audio into latent space, twist the knobs, decode. Pretrained models for instruments, voices, natural sounds.

### Planned
- DDSP (timbre transfer) · MusicGen (melody-conditioned generation) · Demucs (stem separation) · Encodec (neural compression artifacts) · Riffusion (spectrogram diffusion) · VampNet (masked infilling) · NSynth (timbre interpolation) · and more

## Install

```
uv pip install rottengenizdat
```

## Usage

```
rotten --help
rotten rave input.wav --model whale -o output.wav
rotten rave input.wav --model singing --temperature 1.5 -o output.wav
rotten rave input.wav --sweep temperature=0.3,0.7,1.0,1.5,2.0 -o grid/
```

## Architecture

Plugin-based. Each model is a self-contained effect conforming to a simple interface (`audio in → knobs → audio out`). The CLI auto-discovers plugins and lets you chain them.

## License

MIT
