from __future__ import annotations

import tomllib
from pathlib import Path

from rottengenizdat.chain import parse_step

# Keys in a recipe step dict that map directly to process() kwargs via the
# standard flag → kwarg name mapping used in parse_step().  We handle them
# here explicitly so we don't duplicate the flag-map logic.
_STEP_KEY_MAP: dict[str, str] = {
    "model": "model_name",
    "temperature": "temperature",
    "noise": "noise",
    "mix": "mix",
    "dims": "dims",
    "reverse": "reverse",
    "shuffle_chunks": "shuffle_chunks",
    "quantize": "quantize",
}


def load_recipe(path: Path) -> dict:
    """Load a recipe from a TOML file.

    Args:
        path: Path to the ``.toml`` recipe file.

    Returns:
        The parsed recipe dict, with top-level keys ``recipe`` and ``steps``.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def save_recipe(path: Path, name: str, mode: str, steps: list[str]) -> None:
    """Save a recipe to a TOML file from CLI step strings.

    Parses each step string with :func:`~rottengenizdat.chain.parse_step` and
    writes a TOML file in the canonical recipe format.

    Args:
        path: Destination path for the ``.toml`` file.
        name: Human-readable recipe name (stored in ``[recipe]``).
        mode: Either ``"sequential"`` or ``"branch"``.
        steps: List of step strings, e.g. ``["rave -m percussion -t 1.2"]``.
    """
    path = Path(path)
    lines: list[str] = []

    lines.append("[recipe]")
    lines.append(f'name = "{name}"')
    lines.append(f'mode = "{mode}"')
    lines.append("")

    for step_str in steps:
        effect_name, kwargs = parse_step(step_str)
        lines.append("[[steps]]")
        lines.append(f'effect = "{effect_name}"')
        # Write kwargs in a stable order; model first when present
        if "model_name" in kwargs:
            lines.append(f'model = "{kwargs["model_name"]}"')
        for kwarg_key, value in kwargs.items():
            if kwarg_key == "model_name":
                continue
            if isinstance(value, str):
                lines.append(f'{kwarg_key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{kwarg_key} = {str(value).lower()}")
            else:
                lines.append(f"{kwarg_key} = {value}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def recipe_steps_to_kwargs(steps: list[dict]) -> list[tuple[str, dict]]:
    """Convert recipe step dicts to ``(effect_name, kwargs)`` pairs.

    Each step dict must have an ``effect`` key; remaining keys are mapped to
    :py:meth:`~rottengenizdat.plugin.AudioEffect.process` kwargs.

    Args:
        steps: List of step dicts as loaded from a recipe TOML file.

    Returns:
        List of ``(effect_name, kwargs)`` tuples ready to pass to a plugin's
        ``process()`` method.

    Examples:
        >>> recipe_steps_to_kwargs([{"effect": "rave", "model": "percussion"}])
        [('rave', {'model_name': 'percussion'})]
    """
    result: list[tuple[str, dict]] = []
    for step in steps:
        effect_name = step["effect"]
        kwargs: dict = {}
        for key, value in step.items():
            if key == "effect":
                continue
            kwarg_name = _STEP_KEY_MAP.get(key, key)
            kwargs[kwarg_name] = value
        result.append((effect_name, kwargs))
    return result
