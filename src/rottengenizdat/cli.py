from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import click
import typer
from rich.console import Console

from rottengenizdat import __version__
from rottengenizdat.banner import BANNER
from rottengenizdat.chain import run_branch, run_chain
from rottengenizdat.config import (
    CONFIG_DIR,
    load_config,
    config_set as _config_set,
)
from rottengenizdat.sample_sale import (
    CACHE_DIR,
    load_index,
    sync_index,
    clear_cache,
    download_sample,
    pick_random_samples,
)
from rottengenizdat.inputs import InputMode, combine_inputs
from rottengenizdat.core import AudioBuffer, load_audio, save_audio
from rottengenizdat.plugin import discover_plugins
from rottengenizdat.recipe import (
    load_recipe,
    recipe_steps_to_kwargs,
    save_recipe,
)

console = Console()
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

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

app = typer.Typer(
    name="rotten",
    help=_MAIN_HELP,
    invoke_without_command=True,
    context_settings=CONTEXT_SETTINGS,
)


def version_callback(value: bool) -> None:
    if value:
        console.print(BANNER)
        console.print(f"\n  rottengenizdat v{__version__}\n")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """bone music for the machine age

    Audio mangling through neural networks. Feed your tracks through RAVE
    variational autoencoders and let the latent space chew them up.
    Run any command with -h to see detailed help and examples.
    """
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


def register_plugins() -> None:
    """Discover and register all plugins as CLI subcommands."""
    plugins = discover_plugins()
    for name, plugin_cls in plugins.items():
        plugin = plugin_cls()
        plugin.register_command(app)


register_plugins()


# ---------------------------------------------------------------------------
# chain command
# ---------------------------------------------------------------------------


_CHAIN_HELP = """\
Chain multiple effects sequentially or in parallel branches.

SEQUENTIAL MODE (default) feeds audio through each step in order — the
output of step 1 becomes the input of step 2. Stack transformations:

  # Percussion reinterpretation, then vintage warmth on top
  rotten chain input.wav "rave -m percussion -t 1.2" "rave -m vintage" -o out.wav

  # Three passes with rising temperature — photocopy of a photocopy
  rotten chain input.wav "rave -m vintage -t 0.8" "rave -m vintage -t 1.0" "rave -m vintage -t 1.3" -o mirrors.wav

BRANCH MODE (-b) sends the ORIGINAL audio to every step independently,
then mixes all outputs. Blend multiple reinterpretations:

  # 50/50 blend of original and RAVE vintage
  rotten chain input.wav "dry" "rave -m vintage -t 1.3" --branch -o half.wav

  # Three models fighting over the same source
  rotten chain input.wav "rave -m percussion" "rave -m nasa" "rave -m VCTK" -b -o chaos.wav

MULTI-INPUT — feed multiple files, combined before chaining:

  # Splice two files into a random collage, then chain through effects
  rotten chain a.wav b.wav "rave -m percussion" -o spliced.wav

  # Concatenate three files, then process the long result
  rotten chain a.wav b.wav c.wav -- "rave -m vintage" --mode concat -o long.wav

  # Process each input independently, outputs to a directory
  rotten chain a.wav b.wav -- "rave -m nasa -t 1.5" --mode independent -o out/

  Input combination modes (--mode):
    splice ......... Default for multiple inputs. Chop all files into
                     random segments (0.25s–4.0s), shuffle, reassemble.
    concat ......... Join files end-to-end in order.
    independent .... Process each file separately, output to directory.

  Tune splice: --splice-min 0.1 --splice-max 2.0 (seconds)

#SAMPLE-SALE — pull random audio from your Slack channel:

  # Grab 3 random samples, splice them, chain through effects
  rotten chain --sample-sale-count 3 "rave -m percussion" -o collage.wav

  # Mix a local file with one random sample
  rotten chain input.wav --sample-sale "rave -m vintage" --mode concat -o mixed.wav

STEP SYNTAX — each step is a quoted string with rave flags:
  -m MODEL, -t TEMP, -n NOISE, -w MIX, -d DIMS, -r, --shuffle N, -q STEP
  Use 'dry' as a step in branch mode to mix in the unprocessed original.

Save reusable chains as recipes:
  rotten recipe save my-chain.toml "rave -m percussion" "rave -m vintage" --name "my chain"
"""


@app.command(
    name="chain",
    help=_CHAIN_HELP,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def chain_command(ctx: typer.Context) -> None:
    """Chain multiple effects sequentially or in parallel branches.

    Accepts multiple input files followed (optionally separated by '--') by
    one or more step strings.  Named options like --mode, -o, --branch can
    appear anywhere in the argument list.

    Examples::

      rotten chain input.wav "rave -m percussion" -o out.wav
      rotten chain a.wav b.wav -- "rave -m vintage" --mode concat -o out.wav
    """
    # click/typer cannot split two variadic positional groups, so we parse the
    # raw ctx.args ourselves.  '--' is consumed by click and not present in
    # ctx.args; all other tokens land here unprocessed.

    raw: list[str] = list(ctx.args)

    # 1. Extract named chain options from the token stream.
    output: Path = Path("output.wav")
    branch: bool = False
    sample_sale: bool = False
    sample_sale_count: int = 0
    mode: Optional[str] = None
    splice_min: float = 0.25
    splice_max: float = 4.0

    positional: list[str] = []
    idx = 0
    while idx < len(raw):
        tok = raw[idx]
        if tok in ("-o", "--output") and idx + 1 < len(raw):
            output = Path(raw[idx + 1]); idx += 2
        elif tok.startswith("--output="):
            output = Path(tok.split("=", 1)[1]); idx += 1
        elif tok in ("-b", "--branch"):
            branch = True; idx += 1
        elif tok in ("-ss", "--sample-sale"):
            sample_sale = True; idx += 1
        elif tok == "--sample-sale-count" and idx + 1 < len(raw):
            sample_sale_count = int(raw[idx + 1]); idx += 2
        elif tok.startswith("--sample-sale-count="):
            sample_sale_count = int(tok.split("=", 1)[1]); idx += 1
        elif tok == "--mode" and idx + 1 < len(raw):
            mode = raw[idx + 1]; idx += 2
        elif tok.startswith("--mode="):
            mode = tok.split("=", 1)[1]; idx += 1
        elif tok == "--splice-min" and idx + 1 < len(raw):
            splice_min = float(raw[idx + 1]); idx += 2
        elif tok.startswith("--splice-min="):
            splice_min = float(tok.split("=", 1)[1]); idx += 1
        elif tok == "--splice-max" and idx + 1 < len(raw):
            splice_max = float(raw[idx + 1]); idx += 2
        elif tok.startswith("--splice-max="):
            splice_max = float(tok.split("=", 1)[1]); idx += 1
        else:
            positional.append(tok); idx += 1

    # 2. Split positional tokens: leading existing-file paths → input_files,
    #    first non-path token onward → step strings.
    input_file_strs: list[str] = []
    step_strs: list[str] = []
    past_inputs = False
    for tok in positional:
        if not past_inputs and Path(tok).exists():
            input_file_strs.append(tok)
        else:
            past_inputs = True
            step_strs.append(tok)

    input_files: list[Path] = [Path(f) for f in input_file_strs]
    steps: list[str] = step_strs

    if not steps:
        console.print("[red]At least one step is required.[/red]")
        raise typer.Exit(1)

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
        import rottengenizdat.sample_sale as _ss
        console.print(f"[bold]Fetching {ss_count} sample(s) from #sample-sale...[/bold]")
        index = _ss.sync_index()
        picks = _ss.pick_random_samples(index, ss_count)
        for entry in picks:
            console.print(f"  [dim]Selected:[/dim] {entry.filename or entry.url or entry.id}")
            path = _ss.download_sample(entry)
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
    for i_step, step in enumerate(steps, 1):
        console.print(f"  {i_step}. {step}")

    if input_mode == InputMode.INDEPENDENT:
        output.mkdir(parents=True, exist_ok=True)
        for i_buf, (audio, src_name) in enumerate(zip(combined, all_names)):
            console.print(f"\n[bold]Processing input {i_buf+1}/{len(combined)}:[/bold] {src_name}")
            result = run_branch(audio, steps) if branch else run_chain(audio, steps)
            out_path = output / f"{i_buf+1:03d}-{src_name}.wav"
            save_audio(result, out_path)
            console.print(f"[green]Saved:[/green] {out_path}")
    else:
        audio = combined[0]
        console.print(f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")
        result = run_branch(audio, steps) if branch else run_chain(audio, steps)
        save_audio(result, output)
        console.print(f"[green]Saved:[/green] {output}")


# ---------------------------------------------------------------------------
# recipe sub-app
# ---------------------------------------------------------------------------

_RECIPE_HELP = """\
Manage and run saved effect chains (recipes).

Recipes are TOML files that store a named sequence of effects — either
sequential (each step feeds the next) or branch (all steps run on the
original, then mix). rottengenizdat ships with 14 built-in recipes in
the recipes/ directory, ranging from barely-noticeable to total sonic
destruction.

  ░░░ SUBTLE ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ CHAOTIC ███
  barely-there                              fever-dream
  needle-drop                            bitcrushed-god
  ghost-in-the-machine                    drunk-choir
  haunted-dub                            time-sick
  organ-donor                           hall-of-mirrors
  space-sickness                       parallel-universe
  nature-documentary                   bone-xray

Recipes (subtle → chaotic):

  barely-there ........ 90% original + 10% vintage whisper on 2 dims.
                        A/B it to even tell. The gentlest touch.
  needle-drop ......... Like playing a well-worn record. Warm vintage
                        model on 2 dims at low temp, mixed 40% wet.
  ghost-in-the-machine  Heard through a wall. Two models in sequence,
                        each touching only a few dims. Timbre goes wrong
                        but structure stays.
  haunted-dub ......... 70% your track + 30% vintage ghost with noise.
                        Like hearing the reverb tail of a song that was
                        never played.
  organ-donor ......... 50% original, transplanted with orchestral string
                        DNA from sol_ordinario and sol_full models.
  space-sickness ...... 60% original + reversed NASA ghosts + faint
                        quantized percussion shadow.
  nature-documentary .. NASA + orchestral strings = alien wildlife
                        soundtrack. Three models in parallel.
  bone-xray ........... The namesake. Three models (percussion, vintage,
                        musicnet) fighting over your track — like a
                        bootleg pressed onto three X-ray films at once.
  parallel-universe ... Four models, each only touching 4 latent dims.
                        Four alternate realities blended together.
  hall-of-mirrors ..... Same model (vintage) three times in sequence,
                        temperature creeping up. A photocopy of a
                        photocopy of a photocopy.
  drunk-choir ......... Two VCTK voice models + isis, each with noise.
                        Your track sung back by confused neural networks.
  time-sick ........... Temporal nausea: reverse the latent, shuffle it
                        into chunks, quantize. Structure is there but the
                        timeline is having a seizure.
  bitcrushed-god ...... Extreme quantization through two models. Your
                        track reduced to its coarsest neural skeleton,
                        then that skeleton reinterpreted.
  fever-dream ......... Every knob cranked. Three models in sequence with
                        reverse, shuffle, noise, high temp. What comes
                        out is barely audio.

RUNNING RECIPES:

  # Single input — standard usage
  rotten recipe run recipes/bone-xray.toml input.wav -o out.wav
  rotten recipe run recipes/fever-dream.toml drums.wav -o destroyed.wav

  # Multiple inputs — splice them into a collage, then run recipe
  rotten recipe run recipes/drunk-choir.toml a.wav b.wav c.wav -o choir.wav

  # Multiple inputs — concatenate in order
  rotten recipe run recipes/barely-there.toml a.wav b.wav --mode concat -o gentle.wav

  # Multiple inputs — process each independently, output to directory
  rotten recipe run recipes/bone-xray.toml a.wav b.wav --mode independent -o out/

  # Random samples from Slack #sample-sale
  rotten recipe run recipes/fever-dream.toml --sample-sale-count 3 -o collage.wav

  # Mix local file with a random sample
  rotten recipe run recipes/bone-xray.toml input.wav --sample-sale -o mixed.wav

  Input combination modes (--mode):
    splice ......... Default for multiple inputs. Chop all files into
                     random segments (0.25s–4.0s), shuffle, reassemble.
    concat ......... Join files end-to-end in order.
    independent .... Process each file separately, output to directory.

  Tune splice: --splice-min 0.1 --splice-max 2.0 (seconds)

SAVING YOUR OWN:

  rotten recipe save my.toml "rave -m percussion" "rave -m vintage" --name "my recipe"
  rotten recipe save blend.toml "dry" "rave -m vintage -t 1.3" -b --name "haunted blend"
"""

recipe_app = typer.Typer(name="recipe", help=_RECIPE_HELP, context_settings=CONTEXT_SETTINGS)
app.add_typer(recipe_app)


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
    according to --mode (default: splice — chop and shuffle all inputs).

    Examples:

      # Single file through a recipe
      rotten recipe run recipes/bone-xray.toml input.wav -o xray.wav

      # Two files spliced into a collage, then processed (default mode)
      rotten recipe run recipes/fever-dream.toml a.wav b.wav -o collage.wav

      # Three files concatenated end-to-end, then processed
      rotten recipe run recipes/barely-there.toml a.wav b.wav c.wav --mode concat -o long.wav

      # Each file processed independently, one output per input
      rotten recipe run recipes/drunk-choir.toml a.wav b.wav --mode independent -o out/

      # Pull 3 random samples from #sample-sale, splice and process
      rotten recipe run recipes/bone-xray.toml --sample-sale-count 3 -o random.wav

      # Mix a local file with a random Slack sample
      rotten recipe run recipes/hall-of-mirrors.toml vocals.wav --sample-sale -o mixed.wav

      # Tune the splice segment sizes
      rotten recipe run recipes/fever-dream.toml a.wav b.wav --splice-min 0.1 --splice-max 1.0 -o choppy.wav
    """
    if not recipe_file.exists():
        console.print(f"[red]Recipe file not found: {recipe_file}[/red]")
        raise typer.Exit(1)

    # Resolve all inputs
    all_buffers: list[AudioBuffer] = []
    all_names: list[str] = []

    for f in (input_files or []):
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        all_buffers.append(load_audio(f))
        all_names.append(f.stem)

    # Sample-sale samples
    ss_count = sample_sale_count if sample_sale_count > 0 else (1 if sample_sale else 0)
    if ss_count > 0:
        import rottengenizdat.sample_sale as _ss
        console.print(f"[bold]Fetching {ss_count} sample(s) from #sample-sale...[/bold]")
        index = _ss.sync_index()
        picks = _ss.pick_random_samples(index, ss_count)
        for entry in picks:
            console.print(f"  [dim]Selected:[/dim] {entry.filename or entry.url or entry.id}")
            path = _ss.download_sample(entry)
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


@recipe_app.command(name="save")
def recipe_save(
    recipe_file: Annotated[Path, typer.Argument(help="Destination TOML file path")],
    steps: Annotated[list[str], typer.Argument(help="Effect steps as quoted strings")],
    name: Annotated[str, typer.Option("--name", help="Recipe name")] = "untitled",
    branch: Annotated[bool, typer.Option("--branch", "-b", help="Save as branch mode")] = False,
) -> None:
    """Save a sequence of effect steps as a recipe TOML file.

    Parses the same step syntax used by 'rotten chain' and writes a
    reusable TOML recipe. Use --branch to save as a parallel branch
    mix instead of a sequential chain.

    Examples:

      rotten recipe save warm.toml "rave -m vintage -d 0,1 -t 0.6 -w 0.4" --name "warm vinyl"
      rotten recipe save chaos.toml "rave -m nasa -t 2.0 -r" "rave -m percussion -t 1.5" --name "chaos"
      rotten recipe save blend.toml "dry" "rave -m vintage -t 1.3" -b --name "haunted blend"
    """
    if not steps:
        console.print("[red]At least one step is required.[/red]")
        raise typer.Exit(1)
    mode = "branch" if branch else "sequential"
    save_recipe(recipe_file, name, mode, steps)
    console.print(f"[green]Recipe saved:[/green] {recipe_file} (name={name}, mode={mode})")


# ---------------------------------------------------------------------------
# config sub-app
# ---------------------------------------------------------------------------

_CONFIG_HELP = """\
Manage rottengenizdat configuration.

Settings are stored in ~/.config/rottengenizdat/config.toml. Currently
used for Slack integration (#sample-sale), but can hold any config.

SETUP (one-time):

  # Set your Slack bot token (get one at api.slack.com/apps)
  rotten config set slack.token xoxb-YOUR-BOT-TOKEN

  # Set the #sample-sale channel ID (right-click channel → Copy link → ID is last segment)
  rotten config set slack.channel C0XXXXXXX

  # Verify your config
  rotten config show

  # See where the config file lives
  rotten config path

Token resolution order: config file > SLACK_BOT_TOKEN env var > error.
Tokens are masked in 'config show' output for safety.
"""

config_app = typer.Typer(name="config", help=_CONFIG_HELP, context_settings=CONTEXT_SETTINGS)
app.add_typer(config_app)


@config_app.command(name="path")
def config_path() -> None:
    """Print the config file location.

    Prints the full path to the config TOML file, whether or not it exists.
    Useful for finding the file to edit manually or for scripting.

    Example:
      rotten config path
      # → /Users/you/.config/rottengenizdat/config.toml
    """
    import rottengenizdat.cli as _self
    console.print(str(_self.CONFIG_DIR / "config.toml"))


@config_app.command(name="show")
def config_show() -> None:
    """Print current configuration (tokens masked).

    Shows all settings from the config file. Token values are partially
    masked (first 4 and last 4 characters shown) so you can verify the
    right token is configured without exposing the full value.

    Examples:
      rotten config show
      # → [slack]
      #     token = xoxb****abcd
      #     channel = C0XXXXXXX
    """
    import rottengenizdat.cli as _self
    config = load_config(config_dir=_self.CONFIG_DIR)
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
    """Set a configuration value.

    Keys use dotted notation to address nested TOML sections.
    Creates the config file and parent directories if they don't exist.

    Examples:
      rotten config set slack.token xoxb-YOUR-BOT-TOKEN
      rotten config set slack.channel C0XXXXXXX
    """
    import rottengenizdat.cli as _self
    _config_set(key, value, config_dir=_self.CONFIG_DIR)
    console.print(f"[green]Set:[/green] {key}")


# ---------------------------------------------------------------------------
# sample-sale sub-app
# ---------------------------------------------------------------------------

_SAMPLE_SALE_HELP = """\
Manage the #sample-sale sample cache.

rottengenizdat can pull random audio and video from your Slack
#sample-sale channel and feed them into any pipeline. This subcommand
manages the local index and media cache.

HOW IT WORKS:

  1. An index of all media in #sample-sale is synced from Slack
     (file attachments filtered by audio/video MIME type, plus URLs
     in message text for yt-dlp download).

  2. When you use --sample-sale or --sample-sale-count on a pipeline
     command, random entries are picked from the index.

  3. Media files are downloaded lazily — only when a sample is
     actually selected for use, not during indexing.

  4. Downloaded files are cached at ~/.cache/rottengenizdat/samples/
     so the same sample doesn't need re-downloading.

  5. The index syncs incrementally on every --sample-sale use
     (fetches only new messages since last sync).

COMMANDS:

  # Sync the index from Slack (incremental — only new messages)
  rotten sample-sale refresh

  # Full rebuild of the index from scratch
  rotten sample-sale refresh --full

  # Show all indexed samples and their cache status
  rotten sample-sale list

  # Show only samples that are already downloaded locally
  rotten sample-sale list --cached

  # Delete downloaded media files (keeps the index)
  rotten sample-sale clear

  # Delete everything — media files and the index
  rotten sample-sale clear --all

USING SAMPLES IN PIPELINES:

  # One random sample through a recipe
  rotten recipe run recipes/bone-xray.toml --sample-sale -o out.wav

  # Three random samples spliced and processed
  rotten recipe run recipes/fever-dream.toml --sample-sale-count 3 -o collage.wav

  # Mix a local file with random samples
  rotten rave input.wav --sample-sale-count 2 -m vintage --mode splice -o mixed.wav

  # Random samples through a chain
  rotten chain --sample-sale-count 2 "rave -m percussion" "rave -m nasa" -o chain.wav

REQUIREMENTS:

  - Slack bot token configured: rotten config set slack.token xoxb-...
  - Slack channel ID configured: rotten config set slack.channel C0XXXXXXX
  - yt-dlp installed (optional, for downloading linked media): brew install yt-dlp
"""

sample_sale_app = typer.Typer(
    name="sample-sale",
    help=_SAMPLE_SALE_HELP,
    context_settings=CONTEXT_SETTINGS,
)
app.add_typer(sample_sale_app)


@sample_sale_app.command(name="refresh")
def sample_sale_refresh(
    full: Annotated[bool, typer.Option("--full", help="Rebuild index from scratch")] = False,
) -> None:
    """Sync the sample index from Slack (incremental by default).

    Fetches message history from #sample-sale and indexes any audio/video
    attachments and URLs. By default only fetches messages newer than the
    last sync. Use --full to rebuild the entire index from scratch.

    Examples:
      rotten sample-sale refresh           # quick incremental sync
      rotten sample-sale refresh --full    # full rebuild from channel history
    """
    import rottengenizdat.sample_sale as _ss
    mode = "full rebuild" if full else "incremental sync"
    console.print(f"[bold]Refreshing index[/bold] ({mode})...")
    entries = sync_index(full=full, cache_dir=_ss.CACHE_DIR)
    console.print(f"[green]Index updated:[/green] {len(entries)} samples indexed")


@sample_sale_app.command(name="list")
def sample_sale_list(
    cached: Annotated[bool, typer.Option("--cached", help="Only show downloaded samples")] = False,
) -> None:
    """List indexed samples from #sample-sale.

    Shows all media items found in the channel — file attachments and
    URLs. Each entry shows its type (attachment/link), whether the
    media file has been downloaded locally, and the filename or URL.

    Examples:
      rotten sample-sale list              # all indexed samples
      rotten sample-sale list --cached     # only downloaded ones
    """
    import rottengenizdat.sample_sale as _ss
    entries = load_index(cache_dir=_ss.CACHE_DIR)
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
    """Delete cached sample files.

    By default, removes downloaded media files but keeps the index
    (so the next --sample-sale use can pick from the full index
    without re-syncing, and only re-downloads what it needs).

    Use --all to also delete the index, forcing a full re-sync.

    Examples:
      rotten sample-sale clear             # delete media, keep index
      rotten sample-sale clear --all       # delete everything
    """
    import rottengenizdat.sample_sale as _ss
    clear_cache(cache_dir=_ss.CACHE_DIR, full=all_)
    if all_:
        console.print("[green]Cache and index cleared.[/green]")
    else:
        console.print("[green]Cached media files cleared.[/green] Index kept.")
