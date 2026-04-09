from __future__ import annotations

from pathlib import Path

from rich.console import Console

from rottengenizdat.core import AudioBuffer
from rottengenizdat.plugin import AudioEffect

console = Console()

_MAX_DEPTH = 8


class RecipeEffect(AudioEffect):
    """Run another recipe as a step — enables recipe nesting and layering."""

    name = "recipe"
    description = "Run another recipe TOML as an effect step."

    def process(
        self,
        audio: AudioBuffer,
        *,
        path: str,
        _depth: int = 0,
        **kwargs,
    ) -> AudioBuffer:
        """Load and execute a recipe, returning its output.

        Args:
            audio: Input audio buffer.
            path: Path to the recipe TOML file.
            _depth: Internal recursion depth counter (prevents infinite nesting).
        """
        if _depth >= _MAX_DEPTH:
            raise RecursionError(
                f"Recipe nesting depth exceeded {_MAX_DEPTH} — "
                f"check for circular references in {path}"
            )

        from rottengenizdat.chain import mix_buffers
        from rottengenizdat.plugin import discover_plugins
        from rottengenizdat.recipe import load_recipe, recipe_steps_to_kwargs

        recipe_path = Path(path)
        if not recipe_path.exists():
            raise FileNotFoundError(f"Nested recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        meta = recipe.get("recipe", {})
        name = meta.get("name", "untitled")
        recipe_mode = meta.get("mode", "sequential")
        raw_steps = recipe.get("steps", [])

        console.print(
            f"  [dim]{'  ' * _depth}nested recipe:[/dim] {name} "
            f"(mode={recipe_mode}, {len(raw_steps)} step(s))"
        )

        step_pairs = recipe_steps_to_kwargs(raw_steps)
        plugins = discover_plugins()

        if recipe_mode == "branch":
            outputs = []
            weights = []
            for effect_name, step_kwargs in step_pairs:
                if effect_name not in plugins:
                    raise ValueError(f"Unknown effect in nested recipe: {effect_name!r}")
                w = step_kwargs.pop("weight", 1.0)
                weights.append(float(w))
                # Pass depth through for nested recipe steps
                if effect_name == "recipe":
                    step_kwargs["_depth"] = _depth + 1
                result = plugins[effect_name]().process(audio, **step_kwargs)
                outputs.append(result)
            return mix_buffers(outputs, weights)
        else:
            current = audio
            for effect_name, step_kwargs in step_pairs:
                if effect_name not in plugins:
                    raise ValueError(f"Unknown effect in nested recipe: {effect_name!r}")
                if effect_name == "recipe":
                    step_kwargs["_depth"] = _depth + 1
                current = plugins[effect_name]().process(current, **step_kwargs)
            return current
