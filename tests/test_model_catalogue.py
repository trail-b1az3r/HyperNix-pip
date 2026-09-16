"""One model list, from every source the server has.

HyperLink's picker asked ``GET /bridge/lmstudio/models`` and nothing
else. Three consequences, all reported:

* a server without LM Studio had an empty picker, however many GGUFs
  were in ``~/.hypernix/models``
* what ``hypernix-t1 index`` wrote to the registry was never shown, so
  indexing a folder did not make anything appear on the phone
* the empty picker said nothing, because ``refreshModels`` turns a
  failure into ``[]`` — "no models" and "LM Studio is not running" look
  identical and need completely different things from the reader

The last one is why :class:`SourceReport` exists and why most of this
file is about it. A list is not enough; the screen has to be able to say
which sources answered.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

import pytest
from conftest import clear_t1_config

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.hyperlink.catalogue import (  # noqa: E402
    CatalogueModel,
    collect,
    local_models,
    merge,
)
from hypernix.quant.gguf import GGMLType, GGUFWriter  # noqa: E402


def _gguf(path: Path, *, name: str = "Fixture", context: int = 4096) -> Path:
    """A GGUF small enough for a test and real enough to be read."""
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.name", name)
    writer.set_metadata("llama.block_count", 1)
    writer.set_metadata("llama.context_length", context)
    shapes = {"token_embd.weight": (256, 4), "blk.0.attn_q.weight": (256, 4)}
    payload = {}
    for tensor_name, shape in shapes.items():
        count = shape[0] * shape[1]
        writer.add_tensor(tensor_name, shape, int(GGMLType.F32))
        payload[tensor_name] = struct.pack(f"<{count}f", *([0.01] * count))
    writer.write(lambda t: payload[t.name])
    return path


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    logging.disable(logging.CRITICAL)
    # Not every T1_* variable: the storage redirects in conftest.py stay,
    # or "a server with no configuration" quietly becomes "a server
    # writing to the real ~/.hypernix".
    clear_t1_config(monkeypatch)
    yield
    logging.disable(logging.NOTSET)




@pytest.fixture
def models_dir(tmp_path) -> Path:
    directory = tmp_path / "models"
    directory.mkdir()
    _gguf(directory / "qwen3-8b-q4-k-m.gguf", name="Qwen3 8B")
    _gguf(directory / "gemma-4-e2b-q8-0.gguf", name="Gemma 4 E2B")
    return directory


class TestTheDiskIsASource:
    """The one that was missing entirely."""

    def test_it_finds_the_ggufs(self, models_dir):
        found, report = local_models(models_dir)
        assert report.available
        assert {m.model_id for m in found} == {
            "qwen3-8b-q4-k-m", "gemma-4-e2b-q8-0"
        }

    def test_it_reads_the_file_rather_than_the_filename(self, models_dir):
        """A context limit parsed out of a filename is a number that is
        wrong exactly when it matters."""
        found, _ = local_models(models_dir)
        model = next(m for m in found if m.model_id == "qwen3-8b-q4-k-m")
        assert model.name == "Qwen3 8B"
        assert model.architecture == "llama"
        assert model.context_limit == 4096

    def test_a_missing_directory_is_reported_not_raised(self, tmp_path):
        found, report = local_models(tmp_path / "nope")
        assert found == []
        assert not report.available
        assert "does not exist" in report.detail

    def test_an_unreadable_file_is_listed_as_not_runnable(self, tmp_path):
        """"This model is broken" is the answer somebody is looking for
        when a model does not appear. Hiding it is the current bug in
        miniature."""
        directory = tmp_path / "models"
        directory.mkdir()
        (directory / "truncated.gguf").write_bytes(b"GGUF\x03\x00\x00\x00nonsense")
        found, report = local_models(directory)
        assert report.available
        assert len(found) == 1
        assert found[0].runnable is False
        assert found[0].detail

    def test_a_part_file_is_not_a_model(self, tmp_path):
        directory = tmp_path / "models"
        directory.mkdir()
        _gguf(directory / "real.gguf")
        (directory / "downloading.gguf.part").write_bytes(b"\x00" * 16)
        found, _ = local_models(directory)
        assert [m.model_id for m in found] == ["real"]

    def test_it_looks_in_subdirectories(self, tmp_path):
        """A HuggingFace download lands in a folder named after the repo,
        not loose in the models directory."""
        nested = tmp_path / "models" / "TheBloke" / "Qwen"
        nested.mkdir(parents=True)
        _gguf(nested / "qwen.gguf")
        found, _ = local_models(tmp_path / "models")
        assert [m.model_id for m in found] == ["qwen"]


class TestMerging:
    def test_one_model_in_two_sources_appears_once(self):
        merged = merge([
            [CatalogueModel(model_id="qwen3-8b", name="Qwen", source="registry")],
            [CatalogueModel(model_id="qwen3-8b", name="Qwen", source="local",
                            path="/models/qwen3-8b.gguf")],
        ])
        assert len(merged) == 1

    def test_the_registry_wins_and_keeps_the_others_facts(self):
        """The registry's context limit is the one the server enforces,
        so showing the file's would be showing a number not in effect —
        but the registry does not know where the file is, and the app
        needs that to load it."""
        merged = merge([
            [CatalogueModel(model_id="qwen3-8b", name="Qwen", source="registry",
                            context_limit=8192)],
            [CatalogueModel(model_id="qwen3-8b", name="Qwen", source="local",
                            path="/models/qwen3-8b.gguf", size_bytes=4_000_000,
                            context_limit=32768)],
        ])
        assert merged[0].source == "registry"
        assert merged[0].context_limit == 8192
        assert merged[0].path == "/models/qwen3-8b.gguf"
        assert merged[0].size_bytes == 4_000_000

    def test_the_other_sources_are_named(self):
        """So the app can say "on disk and loaded in LM Studio" rather
        than picking one and leaving somebody guessing which."""
        merged = merge([
            [CatalogueModel(model_id="qwen3-8b", name="Q", source="lmstudio",
                            loaded=True)],
            [CatalogueModel(model_id="qwen3-8b", name="Q", source="local",
                            path="/m.gguf")],
        ])
        assert merged[0].also_in == ["local"]
        assert merged[0].loaded is True

    def test_lm_studios_path_shaped_id_matches_the_file(self):
        """LM Studio reports a repo path; the indexer makes a slug. They
        are the same model and listing it twice is the bug the app
        already has, backwards."""
        merged = merge([
            [CatalogueModel(model_id="TheBloke/Qwen2-GGUF/qwen2-7b.Q4_K_M.gguf",
                            name="q", source="lmstudio")],
            [CatalogueModel(model_id="qwen2-7b-q4-k-m", name="q", source="local")],
        ])
        assert len(merged) == 1, [m.model_id for m in merged]

    def test_different_quants_stay_different_models(self):
        """Two quantisations of one model have different quality and
        different limits. Collapsing them would make the picker unable to
        express which is being served."""
        merged = merge([[
            CatalogueModel(model_id="qwen3-8b-q4-k-m", name="a", source="local"),
            CatalogueModel(model_id="qwen3-8b-q8-0", name="b", source="local"),
        ]])
        assert len(merged) == 2

    def test_a_loaded_model_sorts_first(self):
        """It is the one that answers without a wait."""
        merged = merge([[
            CatalogueModel(model_id="aaa", name="aaa", source="local"),
            CatalogueModel(model_id="zzz", name="zzz", source="lmstudio", loaded=True),
        ]])
        assert merged[0].model_id == "zzz"

    def test_an_entry_with_no_id_is_dropped(self):
        assert merge([[CatalogueModel(model_id="", name="?", source="local")]]) == []


class TestSourcesAreReported:
    """The fix for the silent empty picker."""

    def test_every_source_reports_even_when_it_has_nothing(self, tmp_path):
        catalogue = collect(registry=None, bridge=None, local_dir=tmp_path / "none")
        assert {s.name for s in catalogue.sources} == {"registry", "lmstudio", "local"}

    def test_an_absent_bridge_says_how_to_turn_it_on(self, tmp_path):
        catalogue = collect(bridge=None, local_dir=tmp_path)
        bridge = next(s for s in catalogue.sources if s.name == "lmstudio")
        assert not bridge.available
        assert "T1_LMSTUDIO_ENABLED" in bridge.detail

    def test_a_bridge_that_raises_does_not_take_the_disk_with_it(self, models_dir):
        """The whole point. LM Studio being down is not a reason to show
        an empty picker on a server with two models on its disk."""
        class Broken:
            base_url = "http://localhost:1234"

            def list_models(self):
                raise OSError("Connection refused")

        catalogue = collect(bridge=Broken(), local_dir=models_dir)
        assert len(catalogue.models) == 2
        bridge = next(s for s in catalogue.sources if s.name == "lmstudio")
        assert not bridge.available
        assert "Connection refused" in bridge.detail

    def test_a_broken_registry_does_not_take_the_disk_with_it(self, models_dir):
        class Broken:
            def list(self):
                raise RuntimeError("registry file is corrupt")

        catalogue = collect(registry=Broken(), local_dir=models_dir)
        assert len(catalogue.models) == 2
        registry = next(s for s in catalogue.sources if s.name == "registry")
        assert not registry.available


class TestTheEndpoint:
    def client(self, models_dir: Path, **env) -> TestClient:
        import os

        from hypernix.t1api.app import create_app

        os.environ["T1_TRUSTED_NETWORK"] = "1"
        os.environ["T1_HF_DOWNLOAD_DIR"] = str(models_dir)
        for key, value in env.items():
            os.environ[key] = value
        return TestClient(create_app(), client=("192.168.1.50", 5432))

    def test_a_keyless_phone_can_read_it(self, models_dir):
        """It is a HyperLink route, so it takes a HyperLink principal:
        a device token, a T2S key, or keyless on a trusted network."""
        response = self.client(models_dir).get("/hyperlink/models")
        assert response.status_code == 200
        assert response.json()["count"] == 2

    def test_the_local_models_are_in_it(self, models_dir):
        body = self.client(models_dir).get("/hyperlink/models").json()
        assert {m["model_id"] for m in body["models"]} == {
            "qwen3-8b-q4-k-m", "gemma-4-e2b-q8-0"
        }

    def test_it_says_why_lm_studio_contributed_nothing(self, models_dir):
        body = self.client(models_dir).get("/hyperlink/models").json()
        bridge = next(s for s in body["sources"] if s["name"] == "lmstudio")
        assert not bridge["available"]
        assert bridge["detail"]

    def test_it_needs_no_key_that_bridge_models_needed(self, models_dir):
        """`/bridge/lmstudio/models` takes an AuthContext, so a phone
        holding an HLNK_ device token cannot call it at all. This route
        is the one the app can actually use."""
        client = self.client(models_dir)
        assert client.get("/hyperlink/models").status_code == 200


class TestAnEmptyFolderSaysWhyItIsEmpty:
    """"No models" with no path on it is a screen nobody can act on.

    A directory that exists and contains no GGUFs is a source that
    *worked*: it looked where it was told and found nothing, so it
    reports available=True. That is correct, and it meant the app's
    empty state fell through to a fixed sentence -- "no models
    registered, none in ~/.hypernix/models, nothing in LM Studio" --
    which is exactly what a server showed while twenty-one model
    directories sat in the home folder of a different user than the one
    running it.

    The count of what is actually in the directory separates the cases
    that need different fixes: an empty folder, a folder full of Hugging
    Face repos with no .gguf in them, and a folder that is not the one
    you filled.
    """

    def test_an_empty_directory_says_it_is_empty(self, tmp_path):
        from hypernix.hyperlink.catalogue import local_models

        models, report = local_models(tmp_path)
        assert models == []
        assert report.available is True
        assert report.count == 0
        assert str(tmp_path) in report.detail
        assert "empty" in report.detail

    def test_entries_without_gguf_are_counted_and_named(self, tmp_path):
        """The case from the report: full of model directories, no GGUF."""
        from hypernix.hyperlink.catalogue import local_models

        for name in ("unsloth", "jinaai", "LiquidAI"):
            (tmp_path / name).mkdir()
        (tmp_path / "unsloth" / "model.safetensors").write_bytes(b"x")

        models, report = local_models(tmp_path)
        assert models == []
        assert report.available is True
        assert "3 entries" in report.detail, report.detail
        assert ".gguf" in report.detail
        assert str(tmp_path) in report.detail

    def test_one_entry_is_not_called_entries(self, tmp_path):
        from hypernix.hyperlink.catalogue import local_models

        (tmp_path / "only").mkdir()
        _, report = local_models(tmp_path)
        assert "1 entry but" in report.detail, report.detail

    def test_a_folder_with_models_still_reports_its_path(self, tmp_path):
        """The non-empty case keeps the old, shorter detail."""
        from hypernix.hyperlink.catalogue import local_models

        nested = tmp_path / "unsloth" / "Qwen"
        nested.mkdir(parents=True)
        (nested / "model.gguf").write_bytes(b"not really a gguf")

        models, report = local_models(tmp_path)
        assert len(models) == 1, "a nested .gguf should still be found"
        assert report.count == 1
        assert report.detail == str(tmp_path)


class TestPlaceholdersAreCountedAndExplained:
    """"Models registered 44" on one screen, "No models" on the next.

    Both numbers were right. `/status` reports len(registry), which
    includes the installer's example entries when
    T1_ENABLE_EXAMPLE_MODELS=1. The catalogue refuses those entries --
    correctly, since a placeholder cannot answer anything -- and used to
    refuse them silently, reporting count=0 with an empty detail. Two
    true numbers with nothing to reconcile them is worse than either.
    """

    @staticmethod
    def _entry(model_id: str, *, example: bool):
        from hypernix.t1api.registry import ModelEntry, ModelPricing, ModelStatus

        return ModelEntry(
            model_id=model_id, display_name=model_id, version="1.0",
            total_parameters=7.0, active_parameters=None, architecture="qwen3",
            supported_tasks=["chat"], availability="public",
            minimum_plan="free", free_tier_available=True, api_available=True,
            local_available=True, remote_available=True, context_limit=8000,
            input_token_limit=8000, output_token_limit=2000, tool_call_limit=4,
            pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=2.0),
            routing_priority=10, fallback_model=None, license="apache-2.0",
            status=ModelStatus.AVAILABLE, is_example_entry=example,
        )

    def _registry(self, examples: int, real: int = 0):
        from hypernix.t1api.registry import ModelRegistry

        registry = ModelRegistry(include_examples=True)
        for i in range(examples):
            registry.register(self._entry(f"example-{i}", example=True))
        for i in range(real):
            registry.register(self._entry(f"real-{i}", example=False))
        return registry

    def test_the_reported_case_reproduces(self):
        """44 registered, nothing offered."""
        from hypernix.hyperlink.catalogue import registry_models

        registry = self._registry(examples=44)
        assert len(registry) == 44
        models, report = registry_models(registry)
        assert models == []
        assert report.count == 0

    def test_all_placeholders_says_so_and_says_what_to_do(self):
        from hypernix.hyperlink.catalogue import registry_models

        _, report = registry_models(self._registry(examples=44))
        assert "44" in report.detail
        assert "placeholder" in report.detail
        assert "hypernix-t1 index" in report.detail
        assert "T1_ENABLE_EXAMPLE_MODELS" in report.detail

    def test_a_mixed_registry_still_offers_the_real_ones(self):
        """Skipping placeholders must not skip everything."""
        from hypernix.hyperlink.catalogue import registry_models

        models, report = registry_models(self._registry(examples=3, real=2))
        assert len(models) == 2
        assert report.count == 2
        assert "3 installer placeholders were skipped" in report.detail

    def test_a_clean_registry_says_nothing_extra(self):
        from hypernix.hyperlink.catalogue import registry_models

        models, report = registry_models(self._registry(examples=0, real=2))
        assert len(models) == 2
        assert report.detail == ""
