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
