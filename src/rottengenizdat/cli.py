from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import click
import typer
from rich.console import Console

from rottengenizdat import __version__
from rottengenizdat.banner import BANNER
from rottengenizdat.chain import run_branch, run_chain
from rottengenizdat.core import load_audio, save_audio
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
  rotten rave input.wav -m nasa -t 1.5 -n 0.2 -o out.wav
  rotten chain input.wav "rave -m percussion" "rave -m vintage" -o out.wav
  rotten recipe run recipes/fever-dream.toml input.wav -o out.wav

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

Sequential mode (default) feeds audio through each step in order — the
output of step 1 becomes the input of step 2. Use this to stack
transformations, building up layers of mangling:

  rotten chain input.wav "rave -m percussion -t 1.2" "rave -m vintage" -o out.wav

Branch mode (-b) sends the ORIGINAL audio to every step independently,
then mixes all the outputs together. Use this to blend multiple
reinterpretations of the same source:

  rotten chain input.wav "dry" "rave -m vintage -t 1.3" --branch -o out.wav

Each step is a quoted string using the same flags as the rave command:
  -m MODEL, -t TEMP, -n NOISE, -w MIX, -d DIMS, -r, --shuffle N, -q STEP

Use 'dry' as a step in branch mode to mix in the unprocessed original.

For reusable chains, save them as recipes:

  rotten recipe save my-chain.toml "rave -m percussion" "rave -m vintage" --name "my chain"
"""


@app.command(name="chain", help=_CHAIN_HELP)
def chain_command(
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    steps: Annotated[list[str], typer.Argument(help="Effect steps as quoted strings, e.g. 'rave -m percussion -t 1.2'")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("output.wav"),
    branch: Annotated[bool, typer.Option("--branch", "-b", help="Run steps in parallel and mix (default: sequential)")] = False,
) -> None:
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)
    if not steps:
        console.print("[red]At least one step is required.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Loading:[/bold] {input_file}")
    audio = load_audio(input_file)
    console.print(f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")

    mode = "branch" if branch else "sequential"
    console.print(f"[bold]Running {mode} chain[/bold] ({len(steps)} step(s))")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")

    if branch:
        result = run_branch(audio, steps)
    else:
        result = run_chain(audio, steps)

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

Run a recipe:
  rotten recipe run recipes/bone-xray.toml input.wav -o out.wav

Save your own:
  rotten recipe save my.toml "rave -m percussion" "rave -m vintage" --name "my recipe"
"""

recipe_app = typer.Typer(name="recipe", help=_RECIPE_HELP, context_settings=CONTEXT_SETTINGS)
app.add_typer(recipe_app)


@recipe_app.command(name="run")
def recipe_run(
    recipe_file: Annotated[Path, typer.Argument(help="Path to recipe TOML file")],
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("output.wav"),
) -> None:
    """Run a saved recipe against an input file.

    Loads a recipe TOML, prints the chain of effects, and processes the
    input audio. Sequential recipes feed each step into the next. Branch
    recipes run all steps on the original and mix with per-step weights.

    Examples:

      rotten recipe run recipes/barely-there.toml vocal.wav -o gentle.wav
      rotten recipe run recipes/fever-dream.toml drums.wav -o destroyed.wav
      rotten recipe run my-custom-recipe.toml loop.wav -o out.wav
    """
    if not recipe_file.exists():
        console.print(f"[red]Recipe file not found: {recipe_file}[/red]")
        raise typer.Exit(1)
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    recipe = load_recipe(recipe_file)
    meta = recipe.get("recipe", {})
    name = meta.get("name", "untitled")
    mode = meta.get("mode", "sequential")
    raw_steps = recipe.get("steps", [])

    console.print(f"[bold]Recipe:[/bold] {name} (mode={mode}, {len(raw_steps)} step(s))")

    console.print(f"[bold]Loading:[/bold] {input_file}")
    audio = load_audio(input_file)
    console.print(f"  {audio.duration:.1f}s, {audio.channels}ch, {audio.sample_rate}Hz")

    step_pairs = recipe_steps_to_kwargs(raw_steps)
    plugins = discover_plugins()

    if mode == "branch":
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
        final = mix_buffers(outputs, weights)
    else:
        current = audio
        for effect_name, kwargs in step_pairs:
            if effect_name not in plugins:
                console.print(f"[red]Unknown effect: {effect_name}[/red]")
                raise typer.Exit(1)
            console.print(f"  [dim]step:[/dim] {effect_name} {kwargs}")
            current = plugins[effect_name]().process(current, **kwargs)
        final = current

    save_audio(final, output)
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
