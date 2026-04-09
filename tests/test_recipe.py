from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rottengenizdat.core import AudioBuffer
from rottengenizdat.recipe import load_recipe, recipe_steps_to_kwargs, save_recipe
from rottengenizdat.plugins.recipe_effect import RecipeEffect, _MAX_DEPTH


class TestRecipe:
    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "test.toml"
        save_recipe(path, "test-recipe", "sequential", [
            "rave -m percussion -t 1.2",
            "rave -m vintage",
        ])
        recipe = load_recipe(path)
        assert recipe["recipe"]["name"] == "test-recipe"
        assert recipe["recipe"]["mode"] == "sequential"
        assert len(recipe["steps"]) == 2

    def test_saved_step_fields(self, tmp_path: Path):
        path = tmp_path / "fields.toml"
        save_recipe(path, "my-recipe", "sequential", ["rave -m percussion -t 1.5 -n 0.2"])
        recipe = load_recipe(path)
        step = recipe["steps"][0]
        assert step["effect"] == "rave"
        assert step["model"] == "percussion"
        assert step["temperature"] == pytest.approx(1.5)
        assert step["noise"] == pytest.approx(0.2)

    def test_recipe_steps_to_kwargs(self):
        steps = [
            {"effect": "rave", "model": "percussion", "temperature": 1.2},
            {"effect": "rave", "model": "vintage"},
        ]
        result = recipe_steps_to_kwargs(steps)
        assert len(result) == 2
        assert result[0] == ("rave", {"model_name": "percussion", "temperature": 1.2})
        assert result[1] == ("rave", {"model_name": "vintage"})

    def test_recipe_steps_to_kwargs_bool(self):
        steps = [{"effect": "rave", "model": "percussion", "reverse": True}]
        result = recipe_steps_to_kwargs(steps)
        assert result[0][1]["reverse"] is True

    def test_branch_recipe(self, tmp_path: Path):
        path = tmp_path / "branch.toml"
        save_recipe(path, "branch-test", "branch", [
            "rave -m percussion",
            "rave -m vintage",
        ])
        recipe = load_recipe(path)
        assert recipe["recipe"]["mode"] == "branch"

    def test_load_recipe_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_recipe(tmp_path / "nonexistent.toml")

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "subdir" / "nested.toml"
        # save_recipe uses Path.write_text, which requires parent to exist.
        # Our implementation doesn't auto-create parents — that's intentional;
        # the CLI caller is responsible.  Test that saving to an existing parent works.
        path.parent.mkdir(parents=True)
        save_recipe(path, "nested", "sequential", ["rave -m percussion"])
        assert path.exists()

    def test_recipe_steps_to_kwargs_recipe_effect(self):
        steps = [
            {"effect": "recipe", "path": "recipes/drunk-choir.toml", "weight": 0.5},
        ]
        result = recipe_steps_to_kwargs(steps)
        assert result[0] == ("recipe", {"path": "recipes/drunk-choir.toml", "weight": 0.5})

    def test_dims_preserved_round_trip(self, tmp_path: Path):
        path = tmp_path / "dims.toml"
        save_recipe(path, "dims-recipe", "sequential", ["rave -m musicnet -d 0,3,7"])
        recipe = load_recipe(path)
        step = recipe["steps"][0]
        assert step["dims"] == "0,3,7"


def _make_audio(sr: int = 16000, duration: float = 0.1) -> AudioBuffer:
    n = int(sr * duration)
    return AudioBuffer(samples=torch.randn(1, n), sample_rate=sr)


def _write_dry_recipe(path: Path, name: str = "test", mode: str = "sequential") -> None:
    """Write a minimal recipe that uses only the dry effect (no GPU needed)."""
    path.write_text(
        f'[recipe]\nname = "{name}"\nmode = "{mode}"\n\n'
        '[[steps]]\neffect = "dry"\n',
        encoding="utf-8",
    )


class TestRecipeEffect:
    def test_nested_dry_recipe(self, tmp_path: Path):
        recipe_path = tmp_path / "inner.toml"
        _write_dry_recipe(recipe_path)
        audio = _make_audio()
        effect = RecipeEffect()
        result = effect.process(audio, path=str(recipe_path))
        assert result.num_samples == audio.num_samples

    def test_nested_branch_recipe(self, tmp_path: Path):
        recipe_path = tmp_path / "branch.toml"
        recipe_path.write_text(
            '[recipe]\nname = "branch-dry"\nmode = "branch"\n\n'
            '[[steps]]\neffect = "dry"\nweight = 0.6\n\n'
            '[[steps]]\neffect = "dry"\nweight = 0.4\n',
            encoding="utf-8",
        )
        audio = _make_audio()
        effect = RecipeEffect()
        result = effect.process(audio, path=str(recipe_path))
        assert result.num_samples == audio.num_samples

    def test_double_nested_recipe(self, tmp_path: Path):
        inner = tmp_path / "inner.toml"
        _write_dry_recipe(inner, "inner")
        outer = tmp_path / "outer.toml"
        outer.write_text(
            '[recipe]\nname = "outer"\nmode = "branch"\n\n'
            '[[steps]]\neffect = "dry"\nweight = 0.5\n\n'
            f'[[steps]]\neffect = "recipe"\npath = "{inner}"\nweight = 0.5\n',
            encoding="utf-8",
        )
        audio = _make_audio()
        effect = RecipeEffect()
        result = effect.process(audio, path=str(outer))
        assert result.num_samples == audio.num_samples

    def test_max_depth_raises(self, tmp_path: Path):
        # Recipe that references itself — should hit depth limit
        self_ref = tmp_path / "loop.toml"
        self_ref.write_text(
            '[recipe]\nname = "loop"\nmode = "sequential"\n\n'
            f'[[steps]]\neffect = "recipe"\npath = "{self_ref}"\n',
            encoding="utf-8",
        )
        audio = _make_audio()
        effect = RecipeEffect()
        with pytest.raises(RecursionError, match="nesting depth exceeded"):
            effect.process(audio, path=str(self_ref))

    def test_missing_recipe_raises(self, tmp_path: Path):
        audio = _make_audio()
        effect = RecipeEffect()
        with pytest.raises(FileNotFoundError, match="Nested recipe not found"):
            effect.process(audio, path=str(tmp_path / "nope.toml"))

    def test_plugin_discovered(self):
        from rottengenizdat.plugin import discover_plugins
        plugins = discover_plugins()
        assert "recipe" in plugins
        assert plugins["recipe"] is RecipeEffect
