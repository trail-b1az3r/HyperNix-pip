"""scriptgen 0.72.8rc1: a script that runs, more options, and easier to drive.

The generated script used to stop in main() with "bring your own
loader", and its default optimiser could not be built (the Pressure
Cooker classes live in submodules, and take momentum_beta, not betas).
These run a generated script for real, on a tiny HyperNix model on the
CPU, and check the recipes, the wizard, the CLI and the window.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from hypernix.scriptgen import cli as sg_cli
from hypernix.scriptgen.app import FormModel
from hypernix.scriptgen.highlight import spans
from hypernix.scriptgen.params import ALL_PARAMS, GROUPS, defaults, search, suggest, validate_all
from hypernix.scriptgen.recipes import RECIPES, apply_recipe, get_recipe
from hypernix.scriptgen.templates import CONFIG_FORMATS, as_config, generate

SRC = Path(__file__).resolve().parents[2] / "src"


class TestTheOptions:
    def test_lora_is_its_own_section(self):
        lora = next(g for g in GROUPS if g.key == "adapters")
        assert {p.name for p in lora.params} >= {"use_lora", "lora_rank", "lora_alpha", "lora_targets"}

    def test_the_new_settings(self):
        for name in ("device", "packing", "val_split", "max_samples", "report_to",
                     "early_stopping_patience", "push_to_hub", "training_loop", "sweep_trials"):
            assert name in ALL_PARAMS

    def test_the_default_model_is_the_newest_hypernix(self):
        assert defaults()["model_id"] == "Compactbot/hypernix-3.1-mini"
        assert defaults()["sequence_length"] == 512

    def test_a_dependency_can_name_values(self):
        smoothing = ALL_PARAMS["label_smoothing"]
        assert not smoothing.shown_with({"loss_function": "cross_entropy"})
        assert smoothing.shown_with({"loss_function": "label_smoothed_ce"})
        assert ALL_PARAMS["lora_rank"].shown_with({"use_lora": True})
        assert not ALL_PARAMS["lora_rank"].shown_with({"use_lora": False})

    def test_new_cross_checks(self):
        errors, _ = validate_all(defaults() | {"push_to_hub": True, "hub_repo": ""})
        assert any("Hub repo" in e for e in errors)
        _, warnings = validate_all(defaults() | {"early_stopping_patience": 3, "val_split": 0.0})
        assert any("never stop early" in w for w in warnings)
        _, warnings = validate_all(defaults())
        assert any("No dataset yet" in w for w in warnings)


class TestFindingASetting:
    def test_search_is_over_names_labels_and_hints(self):
        assert [p.name for p in search("warmup")] == ["warmup_ratio", "warmup_steps"]
        assert "lora_rank" in [p.name for p in search("adapter width")]
        assert len(search("")) == len(ALL_PARAMS)

    @pytest.mark.parametrize("typo,meant", [("lr", "learning_rate"), ("learnin_rate", "learning_rate"),
                                            ("epoch", "epochs"), ("grad-accum", "gradient_accumulation")])
    def test_a_typo_suggests_what_it_meant(self, typo, meant):
        assert meant in suggest(typo)

    def test_set_with_a_typo_says_what_it_meant(self):
        with pytest.raises(SystemExit, match="Did you mean: learning_rate"):
            sg_cli._parse_set(["learnin_rate=1e-4"])


class TestRecipes:
    def test_every_recipe_validates(self):
        for recipe in RECIPES:
            errors, _ = validate_all(apply_recipe(recipe.key) | {"dataset": "data.jsonl"})
            assert errors == [], recipe.key

    def test_every_recipe_names_real_settings(self):
        for recipe in RECIPES:
            assert set(recipe.values) <= set(ALL_PARAMS), recipe.key

    def test_a_recipe_starts_from_the_defaults(self):
        values = apply_recipe("quick-test")
        assert values["device"] == "cpu" and values["max_steps"] == 20
        assert values["lr_scheduler"] == defaults()["lr_scheduler"]

    def test_an_unknown_recipe_lists_the_real_ones(self):
        with pytest.raises(KeyError, match="quick-test"):
            get_recipe("fast")

    def test_changes_are_counted_from_the_recipe(self):
        model = FormModel()
        model.apply_recipe("quick-test")
        assert model.changed_from_defaults() == {}
        model.set("learning_rate", 1e-4)
        assert list(model.changed_from_defaults()) == ["learning_rate"]
        model.reset_param("learning_rate")
        assert model.get("learning_rate") == 3e-4

    def test_presets_remember_the_recipe(self):
        model = FormModel()
        model.apply_recipe("lora-low-vram")
        other = FormModel()
        other.load_json(model.to_json())
        assert other.recipe == "lora-low-vram" and other.get("use_lora") is True


class TestTheFormats:
    def test_every_format_renders(self):
        for fmt in CONFIG_FORMATS:
            assert as_config(defaults(), fmt=fmt)

    def test_toml_parses(self):
        parsed = tomllib.loads(as_config(defaults(), fmt="toml"))
        assert parsed["learning_rate"] == 2e-5 and parsed["lr_range"] == [1e-5, 5e-5]

    def test_yaml_parses_where_yaml_is_installed(self):
        yaml = pytest.importorskip("yaml")
        assert yaml.safe_load(as_config(defaults(), fmt="yaml"))["packing"] is True

    def test_args_are_the_generated_scripts_own_flags(self):
        text = as_config(defaults() | {"dataset": "my data.jsonl"}, fmt="args")
        assert text.startswith("python train.py") and "--dataset 'my data.jsonl'" in text


class TestTheGeneratedScript:
    def test_both_bodies_parse(self):
        ast.parse(generate(defaults()))
        ast.parse(generate(defaults() | {"training_loop": "skeleton"}))

    def test_notes_cannot_end_the_docstring(self):
        ast.parse(generate(defaults() | {"notes": 'ends here """ and a \\ backslash'}))

    def test_the_cookers_come_from_their_own_modules(self):
        source = generate(defaults())
        assert '"PressureCookerV5": "pressure_cooker_v5"' in source
        assert "betas=(config[\"beta1\"]" not in source.split("if name in _COOKERS:")[1].split("common =")[0]


torch = None


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    global torch
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_model

    from hypernix.training.brewer import BrewerConfig, BrewerModel

    root = tmp_path_factory.mktemp("sg")
    folder = root / "tiny"
    folder.mkdir()
    config = BrewerConfig(vocab_size=512, n_layers=2, n_heads=4, n_kv_heads=2, d_model=64,
                          max_seq_len=128, use_sliding_window=False, name="tiny")
    torch.manual_seed(0)
    save_model(BrewerModel(config), str(folder / "model.safetensors"))
    config.save(folder / "config.json")
    with (root / "data.jsonl").open("w") as handle:
        for i in range(120):
            handle.write(json.dumps({"text": f"Alder kept {i} boats and traded wool with the coast."}) + "\n")
    return root


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          cwd=script.parent, env=env, timeout=600)


class TestItRuns:
    """The point of 0.72.8rc1's generator: the script trains."""

    def _script(self, tiny, **overrides):
        values = apply_recipe("quick-test") | {
            "model_id": str(tiny / "tiny"), "dataset": str(tiny / "data.jsonl"),
            "sequence_length": 64, "max_steps": 6, "logging_steps": 2, "eval_steps": 3,
            "save_steps": 3, "output_dir": str(tiny / "run")} | overrides
        script = tiny / "train.py"
        script.write_text(generate(values))
        return script

    def test_it_trains_evaluates_and_saves_a_model_that_loads(self, tiny):
        done = _run(self._script(tiny))
        assert done.returncode == 0, done.stderr[-2000:]
        assert "step 6/6" in done.stderr and "final held-out loss" in done.stderr
        final = tiny / "run" / "final"
        assert (final / "model.safetensors").is_file() and (final / "config.json").is_file()
        rows = [json.loads(line) for line in (tiny / "run" / "metrics.jsonl").read_text().splitlines()]
        assert any("eval_loss" in r for r in rows) and rows[-1]["step"] == 6
        from hypernix.models import brewer_adapter

        model, _ = brewer_adapter.load(final)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_the_default_optimiser_and_lora_and_flags(self, tiny):
        script = self._script(tiny)
        done = _run(script, "--optimizer", "PressureCookerV5", "--use_lora", "true",
                    "--lora_rank", "4", "--output_dir", str(tiny / "lora"))
        assert done.returncode == 0, done.stderr[-2000:]
        assert "LoRA: 8 layers adapted" in done.stderr
        # Merged into the weights: the saved model is a plain one.
        from safetensors import safe_open

        with safe_open(str(tiny / "lora" / "final" / "model.safetensors"), "pt") as handle:
            assert not any("lora_" in key for key in handle.keys())

    def test_a_sweep_picks_a_rate_then_trains(self, tiny):
        done = _run(self._script(tiny, enable_sweep=True, sweep_trials=2, sweep_steps=2,
                                 output_dir=str(tiny / "sweep")))
        assert done.returncode == 0, done.stderr[-2000:]
        assert "training with the best learning rate" in done.stderr
        assert not (tiny / "sweep" / "sweep-lr-1.0e-05" / "final").exists()
        assert (tiny / "sweep" / "final").is_dir()

    def test_resume_carries_on_from_the_last_checkpoint(self, tiny):
        script = self._script(tiny, output_dir=str(tiny / "resume"), max_steps=4, save_steps=2)
        assert _run(script).returncode == 0
        done = _run(script, "--resume", "true", "--max_steps", "6")
        assert done.returncode == 0, done.stderr[-2000:]
        assert "resumed from checkpoint-2" in done.stderr and "step 6/6" in done.stderr

    def test_no_dataset_says_how_to_give_one(self, tiny):
        done = _run(self._script(tiny, dataset=""))
        assert done.returncode != 0 and "--dataset PATH" in done.stderr

    def test_the_skeleton_says_it_is_one(self, tiny):
        done = _run(self._script(tiny, training_loop="skeleton"))
        assert done.returncode != 0 and "skeleton" in done.stderr


class TestTheCommandLine:
    def test_a_positional_output_writes_without_a_window(self, tmp_path, capsys):
        out = tmp_path / "train.py"
        assert sg_cli.main([str(out), "--recipe", "quick-test", "--set", "dataset=d.jsonl"]) == 0
        ast.parse(out.read_text())
        assert "run it with" in capsys.readouterr().out

    def test_explain_and_listings(self, capsys):
        assert sg_cli.main(["--explain", "warmup_ratio"]) == 0
        assert "survives a change in dataset size" in capsys.readouterr().out
        assert sg_cli.main(["--list-params", "lora"]) == 0
        assert "lora_rank" in capsys.readouterr().out
        assert sg_cli.main(["--list-recipes"]) == 0
        assert "quick-test" in capsys.readouterr().out
        assert sg_cli.main(["--explain", "learnin_rate"]) == 1

    def test_changes(self, capsys):
        assert sg_cli.main(["--recipe", "quick-test", "--set", "epochs=2", "--changes"]) == 0
        out = capsys.readouterr().out
        assert "quick-test recipe" in out and "epochs" in out

    def test_hnx_scriptgen_is_a_subcommand(self, capsys):
        from hypernix.interfaces.cli import main

        assert main(["scriptgen", "--list-recipes"]) == 0
        assert "hypernix-finetune" in capsys.readouterr().out

    def test_the_wizard(self, tmp_path):
        answers = iter(["4", "", str(tmp_path / "data.jsonl"), "4", str(tmp_path / "w.py")])
        (tmp_path / "data.jsonl").write_text('{"text": "hi"}\n')
        said = []
        assert sg_cli.wizard(ask=lambda _q: next(answers), say=said.append) == 0
        source = (tmp_path / "w.py").read_text()
        ast.parse(source)
        assert "'device': 'cpu'" in source and "Run it:" in "\n".join(said)

    def test_the_wizard_asks_again_on_a_bad_answer_and_stops_cleanly(self):
        answers = iter(["9", "1"])

        def ask(_q):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None

        said = []
        assert sg_cli.wizard(ask=ask, say=said.append) == 130
        assert any("number from 1 to" in s for s in said) and "nothing written" in said[-1]


class TestTheHighlighter:
    def test_python(self):
        found = {(tag, text) for tag, text in _texts('def f(x):\n    return "s" + 1  # c\n')}
        assert {("keyword", "def"), ("name", "f"), ("keyword", "return"), ("string", '"s"'),
                ("number", "1"), ("comment", "# c")} <= found

    def test_the_markers_are_their_own_colour(self):
        tags = {tag for tag, _ in _texts(generate(defaults()))}
        assert {"marker", "keyword", "string", "number", "comment"} <= tags

    def test_configs(self):
        found = set(_texts(as_config(defaults(), fmt="yaml"), "config"))
        assert ("name", "learning_rate") in found and ("number", "2e-05") in found
        assert ("keyword", "true") in found

    def test_half_typed_python_does_not_raise(self):
        spans('x = """never closed\n', "python")


def _texts(text: str, fmt: str = "python"):
    lines = text.splitlines()
    for tag, (l1, c1), (l2, c2) in spans(text, fmt):
        if l1 == l2:
            yield tag, lines[l1 - 1][c1:c2]


class TestTheWindow:
    """Drawn for real where there is a display (CI has none; a desktop does)."""

    @pytest.fixture
    def app(self):
        from hypernix.scriptgen.widgets import tk_available

        ok, reason = tk_available()
        if not ok:
            pytest.skip(reason)
        from hypernix.scriptgen.app import ScriptGenApp

        app = ScriptGenApp(FormModel())
        app.root.update()
        yield app
        app.root.destroy()

    def test_sections_search_and_lora(self, app):
        app._show_section("adapters")
        assert set(app.widgets) >= {"use_lora"} and "lora_rank" not in app.widgets
        app._on_change("use_lora", True, rebuild=True)
        app.root.update()
        assert "lora_rank" in app.widgets
        app.search_entry.focus_set()
        app.root.update()
        app.search_var.set("warmup")
        app.root.update()
        assert "warmup_ratio" in app.widgets and "use_lora" not in app.widgets

    def test_a_bad_value_is_shown_and_not_kept(self, app):
        app._show_section("schedule")
        entry = app.widgets["warmup_ratio"]
        entry.var.set("9")
        app.root.update()
        assert app.model.get("warmup_ratio") == 0.03
        assert "at most" in app.widgets["warmup_ratio:error"].cget("text")

    def test_a_recipe_and_the_config_view(self, app):
        app.recipe_var.set("Smoke test on the CPU")
        app._on_recipe()
        assert app.model.get("device") == "cpu"
        app.mode_seg.set("config")
        app._on_mode("config")
        app.format_var.set("toml")
        app._on_format()
        tomllib.loads(app.preview.get("1.0", "end-1c"))
