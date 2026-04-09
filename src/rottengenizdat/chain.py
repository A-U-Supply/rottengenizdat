from __future__ import annotations

import shlex

import torch

from rottengenizdat.core import AudioBuffer
from rottengenizdat.plugin import discover_plugins

# Map CLI flag names (without leading dashes) to process() kwarg names and types.
# type is one of: str, float, int, bool
_FLAG_MAP: dict[str, tuple[str, str]] = {
    "m": ("model_name", "str"),
    "model": ("model_name", "str"),
    "t": ("temperature", "float"),
    "temperature": ("temperature", "float"),
    "n": ("noise", "float"),
    "noise": ("noise", "float"),
    "w": ("mix", "float"),
    "mix": ("mix", "float"),
    "d": ("dims", "str"),
    "dims": ("dims", "str"),
    "r": ("reverse", "bool"),
    "reverse": ("reverse", "bool"),
    "shuffle": ("shuffle_chunks", "int"),
    "q": ("quantize", "float"),
    "quantize": ("quantize", "float"),
}


def parse_step(step_str: str) -> tuple[str, dict]:
    """Parse a step string like 'rave -m percussion -t 1.2' into (effect_name, kwargs).

    The first token is the effect name; remaining tokens are parsed as CLI flags
    and converted to the appropriate Python types.

    Args:
        step_str: A shell-quoted string describing one processing step.

    Returns:
        A tuple of (effect_name, kwargs) where kwargs maps to process() parameter names.

    Examples:
        >>> parse_step("rave -m percussion -t 1.5")
        ('rave', {'model_name': 'percussion', 'temperature': 1.5})
        >>> parse_step("rave -r")
        ('rave', {'reverse': True})
    """
    tokens = shlex.split(step_str)
    if not tokens:
        raise ValueError("Empty step string")

    effect_name = tokens[0]
    kwargs: dict = {}

    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            flag = token[2:]
        elif token.startswith("-") and len(token) > 1:
            flag = token[1:]
        else:
            raise ValueError(f"Unexpected token in step string: {token!r}")

        if flag not in _FLAG_MAP:
            raise ValueError(f"Unknown flag: {token!r} in step {step_str!r}")

        kwarg_name, kwarg_type = _FLAG_MAP[flag]

        if kwarg_type == "bool":
            # Boolean flags carry no value — presence means True
            kwargs[kwarg_name] = True
            i += 1
        else:
            i += 1
            if i >= len(tokens):
                raise ValueError(f"Flag {token!r} requires a value")
            raw = tokens[i]
            if kwarg_type == "float":
                kwargs[kwarg_name] = float(raw)
            elif kwarg_type == "int":
                kwargs[kwarg_name] = int(raw)
            else:
                kwargs[kwarg_name] = raw
            i += 1

    return effect_name, kwargs


def run_chain(audio: AudioBuffer, steps: list[str]) -> AudioBuffer:
    """Run audio through a sequential chain of effects.

    Each step in *steps* is a step string (see :func:`parse_step`). The audio
    flows through each effect in order, with each effect receiving the output
    of the previous one.

    Args:
        audio: Input audio buffer.
        steps: List of step strings, e.g. ``["rave -m percussion", "rave -m vintage"]``.

    Returns:
        The processed audio buffer after all steps.
    """
    plugins = discover_plugins()
    for step_str in steps:
        effect_name, kwargs = parse_step(step_str)
        if effect_name not in plugins:
            raise ValueError(f"Unknown effect: {effect_name!r}")
        plugin = plugins[effect_name]()
        audio = plugin.process(audio, **kwargs)
    return audio


def run_branch(audio: AudioBuffer, steps: list[str]) -> AudioBuffer:
    """Run audio through parallel branches and average the results.

    Each step in *steps* receives the ORIGINAL input independently. All branch
    outputs are then mixed together (averaged) via :func:`mix_buffers`.

    Args:
        audio: Input audio buffer (passed unchanged to every branch).
        steps: List of step strings, one per branch.

    Returns:
        The averaged mix of all branch outputs.
    """
    plugins = discover_plugins()
    outputs: list[AudioBuffer] = []
    for step_str in steps:
        effect_name, kwargs = parse_step(step_str)
        if effect_name not in plugins:
            raise ValueError(f"Unknown effect: {effect_name!r}")
        plugin = plugins[effect_name]()
        result = plugin.process(audio, **kwargs)
        outputs.append(result)
    return mix_buffers(outputs)


def mix_buffers(
    buffers: list[AudioBuffer], weights: list[float] | None = None
) -> AudioBuffer:
    """Mix multiple AudioBuffers with optional weights.

    If buffers differ in length the result is truncated to the shortest.
    Weights are normalized to sum to 1.0.

    Args:
        buffers: Non-empty list of :class:`~rottengenizdat.core.AudioBuffer` objects.
        weights: Optional per-buffer weights. If None, equal weighting is used.

    Returns:
        A new AudioBuffer whose samples are the weighted sum of all inputs.
    """
    if not buffers:
        raise ValueError("mix_buffers requires at least one buffer")
    if weights is None:
        weights = [1.0] * len(buffers)
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    min_len = min(b.num_samples for b in buffers)
    mixed = torch.zeros_like(buffers[0].samples[:, :min_len])
    for buf, w in zip(buffers, weights):
        mixed = mixed + buf.samples[:, :min_len] * w
    return AudioBuffer(samples=mixed, sample_rate=buffers[0].sample_rate)
