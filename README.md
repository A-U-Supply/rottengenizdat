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

Audio mangling through neural networks. Feed your tracks — loops, stems, full mixes — through RAVE variational autoencoders that chew them up and spit them back out wrong. The original bleeds through in uncanny ways: recognizable, but transformed.

Named after [roentgenizdat](https://en.wikipedia.org/wiki/Ribs_(recordings)), the Soviet practice of pressing bootleg rock and roll onto discarded hospital X-ray film. Your song, re-pressed through a medium that transforms it.

## Install

```
uv pip install rottengenizdat
```

## Quick Start

```bash
# Basic: run audio through a RAVE model
rotten rave input.wav -m vintage -o out.wav

# With knobs: temperature, noise, selective dimensions
rotten rave input.wav -m nasa -t 1.5 -n 0.2 -d 0,1,2,3 -o space.wav

# Chain multiple effects in sequence
rotten chain input.wav "rave -m percussion -t 1.2" "rave -m vintage" -o out.wav

# Run a built-in recipe
rotten recipe run recipes/bone-xray.toml input.wav -o out.wav

# Explore a parameter range
rotten rave input.wav -m vintage --sweep temperature=0.3,0.7,1.0,1.5,2.0 -o grid/
```

Run any command with `-h` for detailed help: `rotten -h`, `rotten rave -h`, `rotten recipe -h`.

## Multi-Input

Feed multiple files into any command. They're combined before processing:

```
rotten rave a.wav b.wav c.wav -m percussion --mode splice -o out.wav
rotten recipe run recipes/bone-xray.toml a.wav b.wav --mode concat -o out.wav
rotten recipe run recipes/fever-dream.toml a.wav b.wav --mode independent -o out/
```

**Modes:**
- `splice` (default) — chop all inputs into random segments, shuffle, reassemble
- `concat` — join inputs end-to-end in order
- `independent` — process each input separately, output to a directory

Splice parameters: `--splice-min 0.25 --splice-max 4.0` (seconds)

## #sample-sale Integration

Pull random audio/video from your Slack #sample-sale channel:

```
# Configure once
rotten config set slack.token xoxb-YOUR-TOKEN
rotten config set slack.channel C0XXXXXXX

# Fetch and use samples
rotten recipe run recipes/fever-dream.toml --sample-sale-count 3 -o out.wav
rotten rave --sample-sale -m vintage -o out.wav

# Manage the cache
rotten sample-sale refresh        # sync index from Slack
rotten sample-sale list           # show indexed samples
rotten sample-sale clear          # delete cached files
```

Samples are cached locally (~/.cache/rottengenizdat/samples/). The index syncs incrementally on each use. Supports file attachments (direct download) and links (via yt-dlp).

## How It Works

RAVE (Realtime Audio Variational autoEncoder) encodes audio into a compact latent space, where each model has learned to represent sound based on what it was trained on. The latent space typically has 16 dimensions and a temporal axis. rottengenizdat lets you manipulate this latent representation — scaling, adding noise, quantizing, reversing, shuffling — then decode it back to audio. Even a straight round-trip with no manipulation produces uncanny results, because the model reinterprets your audio through its own training data.

## Models

Each pretrained RAVE model was trained on different audio and imparts its own character:

| Model | Trained On | Character |
|-------|-----------|-----------|
| `percussion` | Drum/percussion instruments | Makes everything rhythmic and percussive. Good default for beats. |
| `vintage` | Vintage/analog recordings | Warm analog character — tape hiss, vinyl crackle energy. Great for subtle lo-fi. |
| `VCTK` | Human speech (VCTK corpus) | Gives audio a vocal, mouth-shaped quality. Eerie on non-vocal input. |
| `nasa` | NASA mission recordings | Radio chatter, space noise, telemetry. Alien and otherworldly. |
| `musicnet` | Classical music (MusicNet) | Multi-instrument orchestral reinterpretation of anything. |
| `isis` | Instrumental sounds | Adds tonal, resonant color. |
| `sol_ordinario` | Solo strings (ordinario) | Bowed string character — haunting, organic textures. |
| `sol_full` | Solo strings (full technique) | Broader string vocabulary than sol_ordinario. |
| `sol_ordinario_fast` | Solo strings (ordinario) | Fast variant of sol_ordinario. Same character, quicker inference. |
| `darbouka_onnx` | Darbouka drum | Middle-Eastern percussion color. |

Models are downloaded automatically from [IRCAM](https://play.forum.ircam.fr/rave-vst-api) on first use and cached at `~/.cache/rottengenizdat/models/rave/`.

## Knobs

Manipulate the latent space between encode and decode:

### Temperature (`-t, --temperature`)

Scale latent vectors. 1.0 is identity (no change). Values below 1 pull toward the model's average — subtle, smoothed, safer. Values above 1 push extremes — chaotic, distorted, unhinged.

| Value | Effect |
|-------|--------|
| 0.3–0.5 | Gentle. Model's average character, smoothed out. |
| 0.7–0.9 | Warm. Recognizable but softened. |
| 1.0 | Identity. Model's natural reinterpretation. |
| 1.2–1.5 | Aggressive. Artifacts and distortion emerge. |
| 2.0+ | Destroyed. Barely recognizable as the input. |

```bash
rotten rave input.wav -m vintage -t 0.7 -o warm.wav      # gentle
rotten rave input.wav -m percussion -t 1.5 -o hard.wav    # aggressive
rotten rave input.wav -m nasa -t 2.5 -o gone.wav          # destroyed
```

### Noise (`-n, --noise`)

Add gaussian noise to the latent space (0.0–1.0). Injects randomness that stacks with temperature.

| Value | Effect |
|-------|--------|
| 0.0 | Clean (default). |
| 0.1 | Texture — subtle grain and variation. |
| 0.3 | Haze — audible randomness, dreamy. |
| 0.5 | Heavy fog — structure starts to dissolve. |
| 0.8 | Static wash — barely signal left. |

```bash
rotten rave input.wav -m VCTK -n 0.1 -o textured.wav
rotten rave input.wav -m nasa -t 1.3 -n 0.3 -o hazy.wav
```

### Mix (`-w, --mix`)

Wet/dry blend between original and RAVE output. Default is 1.0 (fully wet).

| Value | Effect |
|-------|--------|
| 0.0 | Fully original (no RAVE). |
| 0.3 | Ghostly undertone — original with a shadow. |
| 0.5 | Half and half. |
| 0.7 | Mostly RAVE with original peeking through. |
| 1.0 | Fully RAVE (default). |

```bash
rotten rave input.wav -m vintage -t 0.7 -w 0.3 -o ghost.wav
```

### Dims (`-d, --dims`)

Select which latent dimensions to manipulate. Unselected dims keep their original encoded values, preserving those aspects of the sound. Models typically have 16 dims (indexed 0–15).

- **Lower dims (0–3)** tend to carry structural/tonal information
- **Higher dims (12–15)** tend to carry finer detail and texture
- Omit to manipulate all dims

```bash
# Only touch the structure, leave detail alone
rotten rave input.wav -m musicnet -d 0,1,2,3 -t 1.3 -o structure.wav

# Only mangle the fine detail
rotten rave input.wav -m musicnet -d 12,13,14,15 -t 1.5 -o detail.wav

# Surgical: just two dims
rotten rave input.wav -m vintage -d 0,1 -t 0.7 -o subtle.wav
```

### Reverse (`-r, --reverse`)

Flip the temporal axis of the latent representation. This is not the same as reversing the audio file — the model hears your audio backwards in latent space but decodes it forwards, creating temporal smearing and ghostly pre-echoes.

```bash
rotten rave input.wav -m nasa -r -o reversed-latent.wav
```

### Shuffle (`--shuffle N`)

Cut the latent timeline into chunks of N frames and shuffle them randomly. Creates temporal dislocation where the structure fragments and reassembles wrong.

| Value | Effect |
|-------|--------|
| 8 | Mild stutter — noticeable but gentle. |
| 4 | Moderate chop — clearly dislocated. |
| 2–3 | Heavy fragmentation. |

```bash
rotten rave input.wav -m musicnet --shuffle 4 -o chopped.wav
rotten rave input.wav -m percussion --shuffle 2 -t 1.3 -o shattered.wav
```

### Quantize (`-q, --quantize`)

Snap latent values to a grid of the given step size. Crushes the continuous latent space into discrete steps — like bitcrushing, but in the model's brain.

| Value | Effect |
|-------|--------|
| 0.1 | Subtle stepping. |
| 0.3 | Noticeable crunch. |
| 0.5 | Heavy quantization. |
| 1.0 | Obliterated — coarsest possible grid. |

```bash
rotten rave input.wav -m percussion -q 0.5 -o crunchy.wav
rotten rave input.wav -m sol_full -q 1.0 -t 1.5 -o crushed.wav
```

### Sweep (`--sweep`)

Generate a grid of outputs sweeping one parameter across multiple values. The output path becomes a directory containing one file per value.

```bash
rotten rave input.wav -m vintage --sweep temperature=0.3,0.7,1.0,1.5,2.0 -o grid/
# Creates: grid/temperature_0.30.wav, grid/temperature_0.70.wav, ...

rotten rave input.wav -m nasa --sweep noise=0.0,0.1,0.3,0.5,0.8 -o noise-grid/
```

## Chains

Chain multiple effects sequentially or in parallel branches.

### Sequential (default)

Audio flows through each step in order. The output of step 1 becomes the input of step 2. Use this to stack transformations:

```bash
# Percussion reinterpretation, then vintage warmth on top
rotten chain input.wav "rave -m percussion -t 1.2" "rave -m vintage" -o stacked.wav

# Three passes through the same model with rising temperature
rotten chain input.wav "rave -m vintage -t 0.9" "rave -m vintage -t 1.0" "rave -m vintage -t 1.1" -o mirrors.wav
```

### Branch (`--branch, -b`)

Each step receives the ORIGINAL audio independently. All outputs are mixed together (equal weights by default). Use this to blend multiple reinterpretations:

```bash
# Blend percussion and vintage interpretations
rotten chain input.wav "rave -m percussion" "rave -m vintage" --branch -o blended.wav

# 50/50 original and RAVE (use 'dry' for the unprocessed signal)
rotten chain input.wav "dry" "rave -m vintage -t 1.3" --branch -o half.wav
```

### Step syntax

Each step is a quoted string using the same flags as the `rave` command:

```
"rave -m MODEL -t TEMP -n NOISE -w MIX -d DIMS -r --shuffle N -q STEP"
"dry"
```

## Recipes

Recipes are TOML files that store named chains for easy reuse. rottengenizdat ships with 14 built-in recipes ranging from barely-noticeable to total sonic destruction.

```
░░░ SUBTLE ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ CHAOTIC ███

barely-there                              fever-dream
needle-drop                            bitcrushed-god
ghost-in-the-machine                    drunk-choir
haunted-dub                            time-sick
organ-donor                           hall-of-mirrors
space-sickness                       parallel-universe
nature-documentary                   bone-xray
```

### Recipe Catalog

**barely-there** — 90% original + 10% vintage whisper on 2 dims. A/B it to even tell. The gentlest touch. *(branch)*

**needle-drop** — Like playing a well-worn record. Warm vintage model on 2 dims at low temp, mixed 40% wet. *(sequential)*

**ghost-in-the-machine** — Heard through a wall. Two models in sequence, each touching only a few dims. Timbre goes wrong but structure stays. *(sequential)*

**haunted-dub** — 70% your track + 30% vintage ghost with noise. Like hearing the reverb tail of a song that was never played. *(branch)*

**organ-donor** — 50% original, transplanted with orchestral string DNA from sol_ordinario and sol_full models. *(branch)*

**space-sickness** — 60% original + reversed NASA ghosts + faint quantized percussion shadow. *(branch)*

**nature-documentary** — NASA + orchestral strings = alien wildlife soundtrack. Three models in parallel. *(branch)*

**bone-xray** — The namesake. Three models (percussion, vintage, musicnet) fighting over your track — like a bootleg pressed onto three X-ray films at once. *(branch)*

**parallel-universe** — Four models, each only touching 4 latent dims. Four alternate realities blended together. *(branch)*

**hall-of-mirrors** — Same model (vintage) three times in sequence, temperature creeping up. A photocopy of a photocopy of a photocopy. *(sequential)*

**drunk-choir** — Two VCTK voice models + isis, each with noise. Your track sung back by confused neural networks. *(branch)*

**time-sick** — Temporal nausea: reverse the latent, shuffle it into chunks, quantize. Structure is there but the timeline is having a seizure. *(sequential)*

**bitcrushed-god** — Extreme quantization through two models. Your track reduced to its coarsest neural skeleton, then that skeleton reinterpreted. *(sequential)*

**fever-dream** — Every knob cranked. Three models in sequence with reverse, shuffle, noise, high temp. What comes out is barely audio. *(sequential)*

### Running recipes

```bash
rotten recipe run recipes/bone-xray.toml input.wav -o out.wav
rotten recipe run recipes/barely-there.toml vocal.wav -o gentle.wav
rotten recipe run recipes/fever-dream.toml drums.wav -o destroyed.wav
```

### Writing your own

Save from the command line:

```bash
rotten recipe save my-recipe.toml "rave -m vintage -d 0,1 -t 0.6 -w 0.4" --name "warm vinyl"
rotten recipe save chaos.toml "rave -m nasa -t 2.0 -r" "rave -m percussion -t 1.5" --name "double chaos"
rotten recipe save blend.toml "dry" "rave -m vintage -t 1.3" --branch --name "haunted blend"
```

Or write TOML directly:

```toml
[recipe]
name = "my-recipe"
mode = "branch"       # or "sequential"

[[steps]]
effect = "dry"
weight = 0.7

[[steps]]
effect = "rave"
model = "vintage"
temperature = 1.3
noise = 0.15
dims = "0,1,2,3,4,5"
weight = 0.3
```

**Recipe fields for each step:**

| Field | Type | Description |
|-------|------|-------------|
| `effect` | string | `"rave"` or `"dry"` |
| `model` | string | RAVE model name (rave only) |
| `temperature` | float | Latent scaling |
| `noise` | float | Gaussian noise amount |
| `mix` | float | Wet/dry blend |
| `dims` | string | Latent dims to manipulate (e.g. `"0,1,2,3"`) |
| `reverse` | bool | Reverse latent time axis |
| `shuffle_chunks` | int | Temporal chunk shuffle size |
| `quantize` | float | Latent quantization step |
| `weight` | float | Branch mixing weight (branch mode only) |

## Architecture

Plugin-based. Each model is a self-contained effect conforming to `AudioEffect` (`audio in + knobs -> audio out`). The CLI auto-discovers plugins and lets you chain them. See [docs/design.md](docs/design.md) for the full design spec and planned models.

### Planned models

DDSP (timbre transfer), MusicGen (melody-conditioned generation), Demucs (stem separation), Encodec (neural compression artifacts), Riffusion (spectrogram diffusion), VampNet (masked infilling), NSynth (timbre interpolation), and more.

## License

MIT
