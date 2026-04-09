# Recipes

Recipes are TOML files that store named effect chains for reuse. Instead of typing out a long `rotten chain` command every time, save it as a recipe and run it by name. Each recipe defines a series of processing steps and a mode that controls how those steps combine.

## Running a recipe

```bash
rotten recipe run recipes/bone-xray.toml input.wav -o out.wav

# Multiple inputs — spliced into a collage, then processed
rotten recipe run recipes/fever-dream.toml a.wav b.wav -o collage.wav

# Multiple inputs concatenated end-to-end
rotten recipe run recipes/barely-there.toml a.wav b.wav c.wav --mode concat -o long.wav

# Each input processed independently, one output per input
rotten recipe run recipes/drunk-choir.toml a.wav b.wav --mode independent -o out/

# Pull random samples from #sample-sale
rotten recipe run recipes/bone-xray.toml --sample-sale-count 3 -o random.wav

# Mix a local file with a random sample
rotten recipe run recipes/hall-of-mirrors.toml vocals.wav --sample-sale -o mixed.wav
```

## How recipes work

A recipe has two parts: **metadata** and **steps**.

```toml
[recipe]
name = "my-recipe"
mode = "sequential"       # or "branch"

[[steps]]
effect = "rave"
model = "vintage"
temperature = 1.0

[[steps]]
effect = "rave"
model = "percussion"
temperature = 1.2
```

### Modes

**sequential** -- each step's output becomes the next step's input. Audio flows through the chain like a signal path. Use this to stack transformations, building complexity with each pass. The `hall-of-mirrors` recipe runs the same model three times in sequence with rising temperature -- a photocopy of a photocopy of a photocopy.

**branch** -- every step receives the *original* audio independently. All outputs are mixed together using `weight` values (normalized, default 1.0 per step). Use this to blend multiple reinterpretations of the same source. The `bone-xray` recipe runs three different models in parallel and mixes them equally.

### The `dry` effect

The `dry` effect passes audio through unchanged. In branch mode, this lets you blend your original signal with processed versions:

```toml
[[steps]]
effect = "dry"
weight = 0.7           # 70% original

[[steps]]
effect = "rave"
model = "vintage"
temperature = 1.3
weight = 0.3           # 30% RAVE
```

## Step fields

Each `[[steps]]` entry supports these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `effect` | string | yes | `"rave"` or `"dry"` |
| `model` | string | rave only | RAVE model name (see below) |
| `temperature` | float | no | Latent scaling. 1.0 = identity, <1 = subtle, >1 = chaotic |
| `noise` | float | no | Gaussian noise in latent space (0.0--1.0) |
| `mix` | float | no | Wet/dry blend per step. 0.0 = original, 1.0 = fully processed (default) |
| `dims` | string | no | Comma-separated latent dims to manipulate, e.g. `"0,1,2,3"`. Omit for all |
| `reverse` | bool | no | Reverse the latent time axis |
| `shuffle_chunks` | int | no | Shuffle latent in chunks of N frames |
| `quantize` | float | no | Snap latent values to a grid of this step size |
| `weight` | float | no | Mixing weight in branch mode (normalized across all steps; ignored in sequential) |

### Available models

| Model | Character |
|-------|-----------|
| `percussion` | Drum/percussion -- makes everything rhythmic |
| `vintage` | Warm analog -- tape hiss, vinyl crackle energy |
| `VCTK` | Human speech -- vocal, mouth-shaped quality |
| `nasa` | NASA mission audio -- radio chatter, alien, otherworldly |
| `musicnet` | Classical orchestral -- multi-instrument reinterpretation |
| `isis` | Instrumental sounds -- tonal, resonant color |
| `sol_ordinario` | Solo strings (ordinario) -- haunting, organic |
| `sol_full` | Solo strings (full technique) -- broader vocabulary |
| `sol_ordinario_fast` | Fast variant of sol_ordinario |
| `darbouka_onnx` | Darbouka drum -- Middle-Eastern percussion |

## Built-in recipes

Ordered from subtle to chaotic:

---

### barely-there
*branch, 2 steps*

90% original + 10% vintage whisper on 2 dims. A/B it to even tell. The gentlest touch.

```toml
[[steps]]
effect = "dry"
weight = 0.9

[[steps]]
effect = "rave"
model = "vintage"
dims = "0,1"
temperature = 0.7
weight = 0.1
```

**What's happening:** Branch mode splits the signal. The dry path keeps 90% of your audio untouched. The rave path only manipulates dims 0 and 1 at a low temperature, then gets mixed in at 10%. The result is an almost imperceptible warmth.

---

### needle-drop
*sequential, 1 step*

Like playing a well-worn record. Warm vintage model on 2 dims at low temp, mixed 40% wet.

```toml
[[steps]]
effect = "rave"
model = "vintage"
dims = "0,1"
temperature = 0.6
mix = 0.4
```

**What's happening:** A single pass through vintage RAVE, but only touching 2 dimensions and blended 40% with the original via `mix`. The low temperature (0.6) pulls toward the model's average character -- warm, smoothed, like analog degradation.

---

### ghost-in-the-machine
*sequential, 2 steps*

Heard through a wall. Two models in sequence, each touching only a few dims. Timbre goes wrong but structure stays.

```toml
[[steps]]
effect = "rave"
model = "vintage"
dims = "0,1,2,3"
temperature = 0.7

[[steps]]
effect = "rave"
model = "VCTK"
dims = "0,1"
temperature = 0.5
mix = 0.3
```

**What's happening:** First pass runs 4 dims through vintage at a low temp -- subtly warps the tonal structure. Second pass runs that result through VCTK (speech model) on just 2 dims at very low temp, mixed 30% wet. The vocal quality bleeds through faintly, like hearing your track through a wall where someone is talking.

---

### haunted-dub
*branch, 2 steps*

70% your track + 30% vintage ghost with noise. Like hearing the reverb tail of a song that was never played.

```toml
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

**What's happening:** The dry path preserves most of your signal. The rave path runs 6 dims through vintage at a slightly aggressive temperature with a touch of noise -- the noise adds grain and unpredictability to the latent, and the temperature pushes past identity. Mixed at 30%, it sounds like a ghostly dub delay.

---

### organ-donor
*branch, 3 steps*

50% original, transplanted with orchestral string DNA from sol_ordinario and sol_full models.

```toml
[[steps]]
effect = "dry"
weight = 0.5

[[steps]]
effect = "rave"
model = "sol_ordinario"
temperature = 0.9
weight = 0.3

[[steps]]
effect = "rave"
model = "sol_full"
temperature = 1.1
dims = "2,4,6,8,10,12,14"
weight = 0.2
```

**What's happening:** Half your original signal stays clean. The sol_ordinario path adds haunting bowed-string character at a gentle temp. The sol_full path hits alternating odd-indexed dims at a slightly aggressive temp, contributing a different string texture. Together, it's your track with orchestral DNA grafted in.

---

### space-sickness
*branch, 3 steps*

60% original + reversed NASA ghosts + faint quantized percussion shadow.

```toml
[[steps]]
effect = "dry"
weight = 0.6

[[steps]]
effect = "rave"
model = "nasa"
temperature = 1.4
noise = 0.2
reverse = true
weight = 0.25

[[steps]]
effect = "rave"
model = "percussion"
quantize = 0.6
dims = "8,9,10,11,12,13,14,15"
weight = 0.15
```

**What's happening:** Your track dominates at 60%. The NASA path reverses the latent timeline and adds noise -- radio ghosts that play backwards under your audio. The percussion path quantizes only the upper dims (fine detail), adding a faint rhythmic grid underneath. The combination is disorienting but mostly recognizable.

---

### nature-documentary
*branch, 3 steps*

NASA + orchestral strings = alien wildlife soundtrack. Three models in parallel.

```toml
[[steps]]
effect = "rave"
model = "nasa"
temperature = 1.3
noise = 0.1

[[steps]]
effect = "rave"
model = "sol_ordinario"
temperature = 0.8

[[steps]]
effect = "rave"
model = "sol_full"
temperature = 1.0
dims = "0,2,4,6,8,10"
```

**What's happening:** No dry signal -- this is fully processed. Three models each reinterpret the original independently: NASA adds space-radio character, sol_ordinario adds gentle string bowing, sol_full adds broader string color on alternating dims. Mixed equally, it sounds like the soundtrack to a nature documentary about alien wildlife.

---

### bone-xray
*branch, 3 steps*

The namesake. Three models (percussion, vintage, musicnet) fighting over your track -- like a bootleg pressed onto three X-ray films at once.

```toml
[[steps]]
effect = "rave"
model = "percussion"
temperature = 1.2

[[steps]]
effect = "rave"
model = "vintage"
temperature = 0.9

[[steps]]
effect = "rave"
model = "musicnet"
temperature = 1.0
```

**What's happening:** Three RAVE models each reinterpret your full audio with no dim restrictions. Percussion makes it rhythmic, vintage makes it warm and crackly, musicnet makes it orchestral. All three are mixed equally. Each model's training data fights for dominance -- like a bootleg pressed onto three X-ray films at once.

---

### parallel-universe
*branch, 4 steps*

Four models, each only touching 4 latent dims. Four alternate realities blended together.

```toml
[[steps]]
effect = "rave"
model = "vintage"
dims = "0,1,2,3"
temperature = 1.3

[[steps]]
effect = "rave"
model = "musicnet"
dims = "4,5,6,7"
temperature = 1.3

[[steps]]
effect = "rave"
model = "sol_ordinario"
dims = "8,9,10,11"
temperature = 1.3

[[steps]]
effect = "rave"
model = "percussion"
dims = "12,13,14,15"
temperature = 1.3
```

**What's happening:** Each model takes ownership of a different quarter of the 16-dim latent space. Vintage controls the structural dimensions, musicnet the mid-range, sol_ordinario the upper-mid detail, percussion the finest detail. Each branch preserves a different aspect of your audio and transforms the rest -- four parallel realities mixed into one.

---

### hall-of-mirrors
*sequential, 3 steps*

Same model (vintage) three times in sequence, temperature creeping up. A photocopy of a photocopy of a photocopy.

```toml
[[steps]]
effect = "rave"
model = "vintage"
temperature = 0.9
dims = "0,1,2,3,4,5"

[[steps]]
effect = "rave"
model = "vintage"
temperature = 1.0

[[steps]]
effect = "rave"
model = "vintage"
temperature = 1.1
```

**What's happening:** The same model processes the audio three times in sequence. First pass: gentle, only 6 dims, below identity temp. Second pass: full dims at identity temp -- the model's natural reinterpretation of the already-processed audio. Third pass: slightly above identity, pushing the accumulated drift further. Each pass is like photocopying a photocopy -- detail degrades and the model's character compounds.

---

### drunk-choir
*branch, 3 steps*

Two VCTK voice models + isis, each with noise. Your track sung back by confused neural networks.

```toml
[[steps]]
effect = "rave"
model = "VCTK"
temperature = 1.4
noise = 0.2

[[steps]]
effect = "rave"
model = "VCTK"
temperature = 0.8
noise = 0.4
dims = "4,5,6,7,8"

[[steps]]
effect = "rave"
model = "isis"
temperature = 1.1
noise = 0.15
```

**What's happening:** Two VCTK (speech) instances reinterpret your audio with different temperatures, noise levels, and dim selections -- one aggressive and clear, one subdued and hazy on fewer dims. The isis path adds instrumental resonance with a touch of noise. Mixed together, it sounds like a confused choir of neural networks trying to sing your track, each slightly drunk on noise.

---

### time-sick
*sequential, 2 steps*

Temporal nausea: reverse the latent, shuffle it into chunks, quantize. Structure is there but the timeline is having a seizure.

```toml
[[steps]]
effect = "rave"
model = "musicnet"
reverse = true

[[steps]]
effect = "rave"
model = "musicnet"
shuffle_chunks = 4
quantize = 0.3
```

**What's happening:** First pass reverses the latent timeline through musicnet -- the model decodes your audio as if time runs backwards, creating ghostly pre-echoes and smeared attacks. Second pass takes that reversed output, chops the latent into 4-frame chunks and shuffles them randomly, then quantizes to a 0.3-step grid. The structure survives but time is broken.

---

### bitcrushed-god
*sequential, 2 steps*

Extreme quantization through two models. Your track reduced to its coarsest neural skeleton, then that skeleton reinterpreted.

```toml
[[steps]]
effect = "rave"
model = "percussion"
quantize = 1.0
temperature = 1.5

[[steps]]
effect = "rave"
model = "sol_full"
quantize = 0.5
temperature = 0.8
```

**What's happening:** First pass quantizes to the coarsest possible grid (1.0) at high temperature through percussion -- your audio is reduced to a crude rhythmic skeleton. Second pass quantizes that skeleton again at 0.5 through sol_full at a lower temp -- the string model tries to make sense of the rubble, smoothing it slightly but adding its own artifacts. Like bitcrushing, but in the model's brain.

---

### fever-dream
*sequential, 3 steps*

Every knob cranked. Three models in sequence with reverse, shuffle, noise, high temp. What comes out is barely audio.

```toml
[[steps]]
effect = "rave"
model = "nasa"
temperature = 2.5
noise = 0.8
reverse = true

[[steps]]
effect = "rave"
model = "percussion"
temperature = 2.0
noise = 0.5
shuffle_chunks = 3
quantize = 0.8

[[steps]]
effect = "rave"
model = "VCTK"
temperature = 1.8
shuffle_chunks = 2
```

**What's happening:** Maximum destruction in three stages. First: NASA at temp 2.5 with heavy noise and reversed latent -- your audio is obliterated into alien radio static. Second: percussion at temp 2.0 with noise, 3-frame chunk shuffling, and heavy quantization -- the rubble is chopped and crushed into a rhythmic grid. Third: VCTK at temp 1.8 with 2-frame shuffle -- the speech model tries to vocalize the wreckage. What comes out is barely recognizable as the input.

---

## Writing your own

### From the command line

The `rotten recipe save` command writes a TOML file from step strings (same syntax as `rotten chain`):

```bash
# Sequential chain (default)
rotten recipe save my-recipe.toml \
  "rave -m percussion -t 1.2" \
  "rave -m vintage" \
  --name "my recipe"

# Branch mode with dry blending
rotten recipe save blend.toml \
  "dry" \
  "rave -m vintage -t 1.3" \
  --branch --name "haunted blend"
```

### Writing TOML directly

For full control over every field, write the TOML by hand:

```toml
[recipe]
name = "my-custom-recipe"
mode = "branch"

# Keep 60% of the original
[[steps]]
effect = "dry"
weight = 0.6

# Add 40% RAVE color
[[steps]]
effect = "rave"
model = "vintage"
temperature = 1.1
noise = 0.1
dims = "0,1,2,3"
weight = 0.4
```

Save it anywhere and point `rotten recipe run` at it:

```bash
rotten recipe run my-custom-recipe.toml input.wav -o out.wav
```

### Design tips

**Choose a mode based on intent:**
- `sequential` for cumulative degradation -- each step builds on the last
- `branch` for blending perspectives -- each step offers an independent reinterpretation

**Combine models strategically.** Each model imparts the character of its training data. Pair models that complement each other: NASA + strings = alien wildlife (nature-documentary). VCTK + VCTK = confused choir (drunk-choir). Same model repeated = compounding drift (hall-of-mirrors).

**Use dims for surgical control.** Lower dims (0--3) carry structural/tonal info; manipulating only these preserves detail but warps the skeleton. Higher dims (12--15) carry finer texture; manipulating only these adds grain without breaking the structure. Splitting dims across branches (like parallel-universe) gives each model a different piece of the audio to own.

**Escalate temperature in sequential chains.** Start gentle (0.7--0.9), pass through identity (1.0), push further (1.1--1.5). Each pass compounds -- small temperature increases add up across steps.

**Use `dry` + weights for subtle blending.** A dry step at weight 0.7--0.9 keeps your audio mostly intact while the rave steps add color. This is the easiest way to control how much RAVE you want.

**Use `mix` for per-step wet/dry.** Unlike `weight` (which controls branch mixing), `mix` blends the step's output with its *input*. In sequential mode, `mix = 0.3` means 70% of the previous step's output passes through unchanged, with 30% RAVE layered in.

**Comments help.** TOML supports `#` comments. The built-in recipes all include a short description at the top explaining the intent and aesthetic. Future-you will appreciate it.
