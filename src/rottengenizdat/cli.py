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

app = typer.Typer(
    name="rotten",
    help="bone music for the machine age",
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
    """bone music for the machine age"""
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


@app.command(name="chain", help="Chain multiple effects sequentially or in parallel branches")
def chain_command(
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    steps: Annotated[list[str], typer.Argument(help="Effect steps as quoted strings, e.g. 'rave -m percussion'")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("output.wav"),
    branch: Annotated[bool, typer.Option("--branch", "-b", help="Run steps in parallel and mix")] = False,
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

recipe_app = typer.Typer(name="recipe", help="Manage and run saved effect chains", context_settings=CONTEXT_SETTINGS)
app.add_typer(recipe_app)


@recipe_app.command(name="run")
def recipe_run(
    recipe_file: Annotated[Path, typer.Argument(help="Path to recipe TOML file")],
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path("output.wav"),
) -> None:
    """Run a saved recipe against an input file."""
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
    """Save a sequence of effect steps as a recipe TOML file."""
    if not steps:
        console.print("[red]At least one step is required.[/red]")
        raise typer.Exit(1)
    mode = "branch" if branch else "sequential"
    save_recipe(recipe_file, name, mode, steps)
    console.print(f"[green]Recipe saved:[/green] {recipe_file} (name={name}, mode={mode})")
