"""The registry the server actually loads, and the ways it used to not.

`waiter models` asks the server; the server reads a registry file. So
"Waiter cannot see the indexed models" is never about Waiter — it is
about which file the server opened, and what it did when that file was
not perfect.

Three separate failures, all reachable from an ordinary setup:

**The server never looked.** With no ``T1_MODEL_REGISTRY_PATH`` the
loader went straight to the shipped example seed. So ``hypernix-t1
index`` would write a correct ``models.json`` and ``waiter models``
would list entries the seed file itself documents as *not real* — with
nothing anywhere connecting the two. The registry is now discovered in
the conventional places, and the indexer writes to the one the server
will read.

**A file being written was a crash.** The registry is produced by a
different process, so the server opens it mid-write in the normal course
of things. A half-flushed file raised ``JSONDecodeError`` out of
startup.

**One bad entry discarded every good one.** ``ModelEntry.from_dict``
raises ``KeyError`` on a missing required field, and that killed the
whole load — so a single typo took every other model with it, silently,
leaving an empty list and no cause.
"""
from __future__ import annotations

import json
from pathlib import Path

from hypernix.t1api.registry import (
    REGISTRY_NAMES,
    ModelRegistry,
    discover,
    read_entries,
    registry_locations,
)


def entry(model_id: str, **overrides) -> dict:
    """A complete, valid registry entry."""
    base = {
        "model_id": model_id,
        "display_name": model_id.upper(),
        "version": "1.0",
        "total_parameters": 7.0,
        "active_parameters": None,
        "architecture": "llama",
        "supported_tasks": ["chat"],
        "availability": "public",
        "minimum_plan": "free",
        "free_tier_available": True,
        "api_available": True,
        "local_available": True,
        "remote_available": False,
        "context_limit": 8192,
        "input_token_limit": 8192,
        "output_token_limit": 2048,
        "tool_call_limit": 8,
        "pricing": {},
        "routing_priority": 10,
        "fallback_model": None,
        "license": "unspecified",
        "status": "available",
    }
    base.update(overrides)
    return base


class TestABrokenFileDoesNotEndStartup:
    def test_a_half_written_file_loads_empty_rather_than_raising(self, tmp_path):
        """What the server sees if it starts while the indexer writes."""
        target = tmp_path / "models.json"
        target.write_text('[{"model_id": "a", "display_name"', encoding="utf-8")

        registry = ModelRegistry.load(target)

        assert len(registry) == 0

    def test_it_reports_the_problem_rather_than_swallowing_it(self, tmp_path):
        target = tmp_path / "models.json"
        target.write_text('[{"model_id": "a", "display_name"', encoding="utf-8")

        _found, problems = read_entries(target)

        assert len(problems) == 1
        assert "models.json" in problems[0]

    def test_one_bad_entry_does_not_discard_the_good_ones(self, tmp_path):
        """The worst of the three: a typo used to cost every model."""
        target = tmp_path / "models.json"
        target.write_text(
            json.dumps([{"model_id": "broken"}, entry("good")]), encoding="utf-8"
        )

        found, problems = read_entries(target)

        assert [e.model_id for e in found] == ["good"]
        assert len(problems) == 1
        assert "broken" in problems[0]

    def test_a_single_object_is_accepted_rather_than_a_type_error(self, tmp_path):
        """`{...}` instead of `[{...}]` used to raise "string indices
        must be integers", which names nothing the reader can fix."""
        target = tmp_path / "models.json"
        target.write_text(json.dumps(entry("solo")), encoding="utf-8")

        found, problems = read_entries(target)

        assert [e.model_id for e in found] == ["solo"]
        assert problems == []

    def test_a_models_key_wrapper_is_accepted(self, tmp_path):
        target = tmp_path / "models.json"
        target.write_text(json.dumps({"models": [entry("wrapped")]}), encoding="utf-8")

        found, _problems = read_entries(target)

        assert [e.model_id for e in found] == ["wrapped"]

    def test_an_unreadable_file_is_reported_not_raised(self, tmp_path):
        found, problems = read_entries(tmp_path / "absent.json")

        assert found == []
        assert len(problems) == 1

    def test_a_bare_scalar_is_refused_with_its_type(self, tmp_path):
        target = tmp_path / "models.json"
        target.write_text("42", encoding="utf-8")

        found, problems = read_entries(target)

        assert found == []
        assert "int" in problems[0]

    def test_the_installer_template_comment_stub_is_skipped(self, tmp_path):
        """install-t1.sh writes an explanatory `_comment` entry."""
        target = tmp_path / "models.json"
        target.write_text(
            json.dumps([{"_comment": "edit me"}, entry("real")]), encoding="utf-8"
        )

        found, problems = read_entries(target)

        assert [e.model_id for e in found] == ["real"]
        assert problems == []


class TestJSONL:
    def test_one_entry_per_line_loads(self, tmp_path):
        target = tmp_path / "models.jsonl"
        target.write_text(
            "\n".join(json.dumps(entry(m)) for m in ("alpha", "beta")) + "\n",
            encoding="utf-8",
        )

        found, problems = read_entries(target)

        assert [e.model_id for e in found] == ["alpha", "beta"]
        assert problems == []

    def test_a_truncated_last_line_keeps_the_lines_before_it(self, tmp_path):
        """A half-flushed append is exactly this shape."""
        target = tmp_path / "models.jsonl"
        target.write_text(
            json.dumps(entry("alpha")) + "\n"
            + json.dumps(entry("beta")) + "\n"
            + '{"model_id": "trunc',
            encoding="utf-8",
        )

        found, problems = read_entries(target)

        assert [e.model_id for e in found] == ["alpha", "beta"]
        assert len(problems) == 1

    def test_blank_lines_are_not_problems(self, tmp_path):
        target = tmp_path / "models.jsonl"
        target.write_text(
            json.dumps(entry("alpha")) + "\n\n\n" + json.dumps(entry("beta")) + "\n",
            encoding="utf-8",
        )

        _found, problems = read_entries(target)

        assert problems == []

    def test_the_registry_loads_a_jsonl_file(self, tmp_path):
        target = tmp_path / "models.jsonl"
        target.write_text(json.dumps(entry("alpha")) + "\n", encoding="utf-8")

        assert len(ModelRegistry.load(target)) == 1


class TestDiscovery:
    def test_both_filenames_are_looked_for(self):
        assert set(REGISTRY_NAMES) == {"models.json", "models.jsonl"}

    def test_the_config_dir_is_searched_first(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))

        first = registry_locations()[0]

        assert first == tmp_path / "models.json"

    def test_it_finds_a_registry_in_the_config_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        (tmp_path / "models.json").write_text(
            json.dumps([entry("found")]), encoding="utf-8"
        )

        assert discover() == tmp_path / "models.json"

    def test_json_wins_over_jsonl_in_the_same_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        (tmp_path / "models.jsonl").write_text("", encoding="utf-8")

        assert discover() == tmp_path / "models.json"

    def test_nothing_anywhere_is_none_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        assert discover() is None

    def test_load_with_no_path_uses_what_was_discovered(self, tmp_path, monkeypatch):
        """The bug: the loader went straight to the example seed, so an
        indexed registry was written and never read."""
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        (tmp_path / "models.json").write_text(
            json.dumps([entry("mine")]), encoding="utf-8"
        )

        registry = ModelRegistry.load()

        assert [e.model_id for e in registry.list()] == ["mine"]

    def test_an_explicit_path_still_wins(self, tmp_path, monkeypatch):
        """T1_MODEL_REGISTRY_PATH must not be overridden by discovery."""
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        (tmp_path / "models.json").write_text(
            json.dumps([entry("discovered")]), encoding="utf-8"
        )
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps([entry("explicit")]), encoding="utf-8")

        registry = ModelRegistry.load(explicit)

        assert [e.model_id for e in registry.list()] == ["explicit"]


class TestIndexingThenServingNeedsNoConfiguration:
    """The end-to-end claim: index, restart, models appear."""

    def test_the_indexer_writes_where_the_server_reads(self, tmp_path, monkeypatch):
        import sys

        from hypernix.t1api.modelindex_cli import main as index_main
        sys.path.insert(0, str(Path(__file__).parent))
        from test_hyprslug_headers import _wide_model  # noqa: PLC0415

        models = tmp_path / "models"
        models.mkdir()
        _wide_model(models / "toy-f32.gguf")
        config = tmp_path / "cfg"
        config.mkdir()
        monkeypatch.setenv("T1_CONFIG_DIR", str(config))

        code = index_main(["--dir", str(models)])

        assert code == 0
        registry = ModelRegistry.load()
        assert [e.model_id for e in registry.list()] == ["toy-f32"]

    def test_it_does_not_tell_you_to_set_a_variable_you_do_not_need(
        self, tmp_path, monkeypatch, capsys
    ):
        import sys

        from hypernix.t1api.modelindex_cli import main as index_main
        sys.path.insert(0, str(Path(__file__).parent))
        from test_hyprslug_headers import _wide_model  # noqa: PLC0415

        models = tmp_path / "models"
        models.mkdir()
        _wide_model(models / "toy-f32.gguf")
        config = tmp_path / "cfg"
        config.mkdir()
        monkeypatch.setenv("T1_CONFIG_DIR", str(config))

        index_main(["--dir", str(models)])

        assert "T1_MODEL_REGISTRY_PATH" not in capsys.readouterr().out

    def test_but_it_does_when_the_path_is_somewhere_else(
        self, tmp_path, monkeypatch, capsys
    ):
        import sys

        from hypernix.t1api.modelindex_cli import main as index_main
        sys.path.insert(0, str(Path(__file__).parent))
        from test_hyprslug_headers import _wide_model  # noqa: PLC0415

        models = tmp_path / "models"
        models.mkdir()
        _wide_model(models / "toy-f32.gguf")
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "cfg"))

        index_main(["--dir", str(models), "-o", str(tmp_path / "elsewhere.json")])

        assert "T1_MODEL_REGISTRY_PATH" in capsys.readouterr().out
