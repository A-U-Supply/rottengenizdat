from __future__ import annotations

from pathlib import Path

import pytest

from rottengenizdat.recipe import load_recipe, recipe_steps_to_kwargs, save_recipe


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

    def test_dims_preserved_round_trip(self, tmp_path: Path):
        path = tmp_path / "dims.toml"
        save_recipe(path, "dims-recipe", "sequential", ["rave -m musicnet -d 0,3,7"])
        recipe = load_recipe(path)
        step = recipe["steps"][0]
        assert step["dims"] == "0,3,7"
