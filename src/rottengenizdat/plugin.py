from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod

from rottengenizdat.core import AudioBuffer


class AudioEffect(ABC):
    """Base class for all audio effect plugins."""

    name: str
    description: str

    @abstractmethod
    def process(self, audio: AudioBuffer, **kwargs) -> AudioBuffer:
        """Transform audio. All knobs passed as kwargs."""
        ...

    def register_command(self, app: "typer.Typer") -> None:
        """Register this plugin as a typer subcommand. Override for custom args."""
        ...


def discover_plugins() -> dict[str, type[AudioEffect]]:
    """Auto-discover all AudioEffect subclasses in the plugins package."""
    import rottengenizdat.plugins as plugins_pkg

    plugins: dict[str, type[AudioEffect]] = {}

    for importer, modname, ispkg in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(f"rottengenizdat.plugins.{modname}")
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AudioEffect) and obj is not AudioEffect:
                plugins[obj.name] = obj

    return plugins
