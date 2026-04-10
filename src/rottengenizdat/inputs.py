from __future__ import annotations

from enum import Enum

from rottengenizdat.core import AudioBuffer, concat_buffers
from rottengenizdat.splice import splice_buffers, DEFAULT_MIN_SECONDS, DEFAULT_MAX_SECONDS


class InputMode(Enum):
    """How to combine multiple inputs before running the pipeline."""

    PASSTHROUGH = "passthrough"
    SPLICE = "splice"
    CONCAT = "concat"
    INDEPENDENT = "independent"
    BLEND = "blend"

    @classmethod
    def resolve(cls, mode_str: str | None, num_inputs: int) -> InputMode:
        """Determine the input mode from the user's --mode flag and input count."""
        if mode_str is not None:
            return cls(mode_str)
        if num_inputs <= 1:
            return cls.PASSTHROUGH
        return cls.SPLICE


def combine_inputs(
    buffers: list[AudioBuffer],
    mode: InputMode,
    splice_min: float = DEFAULT_MIN_SECONDS,
    splice_max: float = DEFAULT_MAX_SECONDS,
) -> list[AudioBuffer]:
    """Combine input buffers according to the given mode.

    Returns a list of AudioBuffers. Length is 1 for all modes except
    INDEPENDENT, which returns one buffer per input.
    """
    if not buffers:
        raise ValueError("No input buffers provided")

    if mode == InputMode.PASSTHROUGH:
        return [buffers[0]]
    elif mode == InputMode.CONCAT:
        return [concat_buffers(buffers)]
    elif mode in (InputMode.INDEPENDENT, InputMode.BLEND):
        return list(buffers)
    elif mode == InputMode.SPLICE:
        return [splice_buffers(buffers, min_seconds=splice_min, max_seconds=splice_max)]
    else:
        raise ValueError(f"Unknown input mode: {mode}")
