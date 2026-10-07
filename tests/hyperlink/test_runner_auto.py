"""``hnx-t1 runner auto``, and linking a model saved without config.json.

``auto`` loads what the server usually runs: the model loaded last (or
most often) that is still here, on the backend used most, with that
model's last settings. The history behind it is a small JSON file in the
T1 config folder, written on every successful load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hypernix.hyperlink import runner_history as rh


def R(model, backend="auto", **kw):
    return rh.LoadRecord(model_id=model, backend=backend, **kw)


class TestChoosing:
    def test_the_last_model_with_its_last_settings(self):
        got = rh.choose([R("a", context_length=4096), R("b", context_length=2048),
                         R("a", context_length=8192)])
        assert got.model_id == "a" and got.context_length == 8192

    def test_most_used_prefers_the_habit(self):
        got = rh.choose([R("a"), R("a"), R("a"), R("b")], prefer="most")
        assert got.model_id == "a"

    def test_a_model_that_is_gone_is_passed_over(self):
        got = rh.choose([R("a"), R("b")], loadable={"a"})
        assert got.model_id == "a" and "still here" in got.why

    def test_the_backend_is_the_most_used_one(self):
        got = rh.choose([R("a", "cuda"), R("b", "cuda"), R("a", "cpu")])
        assert got.backend == "cuda"

    def test_a_backend_tie_goes_to_the_more_recent(self):
        assert rh.choose([R("a", "cpu"), R("a", "vulkan")]).backend == "vulkan"

    def test_auto_layers_stay_auto(self):
        """None meant "work it out"; replaying a computed split would pin
        it to whatever else was in VRAM last time."""
        assert rh.choose([R("a", gpu_layers=None)]).gpu_layers is None

    def test_nothing_to_go_on(self):
        assert rh.choose([]) is None
        assert rh.choose([R("a")], loadable=set()) is None

    def test_prefer_is_checked(self):
        with pytest.raises(ValueError):
            rh.choose([R("a")], prefer="newest")


class TestTheFile:
    def test_round_trip_and_cap(self, tmp_path):
        path = tmp_path / "h.json"
        for i in range(rh.MAX_RECORDS + 5):
            rh.record(R(f"m{i}"), path)
        rows = rh.read(path)
        assert len(rows) == rh.MAX_RECORDS and rows[-1].model_id == f"m{rh.MAX_RECORDS + 4}"

    def test_a_damaged_file_is_no_history(self, tmp_path):
        path = tmp_path / "h.json"
        path.write_text("{not json", encoding="utf-8")
        assert rh.read(path) == []

    def test_it_lives_in_the_t1_config_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        assert rh.history_path() == tmp_path / rh.HISTORY_FILE


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    from conftest import clear_t1_config

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "t1"))
    monkeypatch.setenv("T1_DEFAULT_MODEL", "")
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
    models = tmp_path / "models"
    models.mkdir()
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough", db_path=str(tmp_path / "t1.db"),
        module_storage_dir=str(tmp_path / "m"), hyperlink_files_dir=str(tmp_path / "f"),
        default_plan="free", hf_download_dir=str(models), web_enabled=False,
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    client = TestClient(app, client=("127.0.0.1", 5000))
    client.keymaster = km
    admin = {"Authorization": "Bearer " + km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key}
    return client, admin, models, tmp_path


def _gguf(path: Path) -> Path:
    from hypernix.quant.gguf import GGMLType, GGUFWriter

    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.add_tensor("x", (32, 1), int(GGMLType.F32))
    writer.write(lambda t: b"\0" * t.nbytes)
    return path


class TestAutoOverHTTP:
    def test_with_no_history_and_no_default_it_says_what_to_do(self, server):
        client, admin, *_ = server
        got = client.post("/runner/auto", json={}, headers=admin)
        assert got.status_code == 404
        assert "runner start" in got.json()["error"]["message"]

    def test_a_dry_run_names_the_choice_and_loads_nothing(self, server):
        client, admin, models, tmp = server
        _gguf(models / "qwen.gguf")
        _gguf(models / "llama.gguf")
        path = tmp / "t1" / rh.HISTORY_FILE
        rh.record(R("qwen", "cuda", context_length=8192), path)
        rh.record(R("llama", "cuda"), path)
        rh.record(R("gone-now"), path)
        got = client.post("/runner/auto", json={"dry_run": True}, headers=admin)
        assert got.status_code == 200, got.text
        choice = got.json()["model"]["auto"]
        assert choice["model_id"] == "llama" and choice["backend"] == "cuda"
        assert not got.json()["loaded"]
        most = client.post("/runner/auto", json={"dry_run": True, "prefer": "most"},
                           headers=admin).json()["model"]["auto"]
        assert most["model_id"] in ("qwen", "llama")

    def test_auto_loads_through_the_runner_with_the_remembered_settings(self, server, monkeypatch):
        client, admin, models, tmp = server
        _gguf(models / "qwen.gguf")
        rh.record(R("qwen", "vulkan", context_length=4096, gpu_layers=12), tmp / "t1" / rh.HISTORY_FILE)
        seen = {}

        class FakeModel:
            placement = type("P", (), {"reason": "test", "backend": "vulkan"})()

            def to_dict(self):
                return {"model_id": "qwen"}

        def fake_load(self, path, **kw):
            seen.update(kw, path=str(path))
            return FakeModel()

        from hypernix.hyperlink.managed import ManagedRunner
        monkeypatch.setattr(ManagedRunner, "load", fake_load)
        got = client.post("/runner/auto", json={}, headers=admin)
        assert got.status_code == 200, got.text
        assert seen["backend"] == "vulkan" and seen["context_length"] == 4096
        assert seen["gpu_layers"] == 12 and seen["path"].endswith("qwen.gguf")
        # ...and the load is remembered for next time.
        assert rh.read(tmp / "t1" / rh.HISTORY_FILE)[-1].model_id == "qwen"

    def test_a_read_only_key_cannot_auto_load(self, server, monkeypatch):
        client, _admin, _models, _tmp = server
        km = client.keymaster
        reader = {"Authorization": "Bearer " + km.create(
            key_type=KeyType.USER, scopes={KeyScope.READ}).key}
        got = client.post("/runner/auto", json={"dry_run": True}, headers=reader)
        assert got.status_code in (401, 403)


class TestLinkingWithoutAConfig:
    """From a report: a model in another folder could not be linked. A
    Brewer folder saved without config.json (HyperNix.3.1-mini's beta
    layout) was refused as "no .gguf in it", and never listed."""

    def test_a_config_less_brewer_folder_links_and_lists(self, server):
        client, admin, models, tmp = server
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interfaces"))
        from test_hyped_unknown_models import _brewed

        folder = _brewed(tmp / "elsewhere" / "HyperNix.3.1-mini", config_json=False)
        got = client.post("/hyperlink/models/link", json={"path": str(folder)}, headers=admin)
        assert got.status_code == 200, got.text
        assert got.json()["kind"] == "hypernix"
        listed = client.get("/hyperlink/models", headers=admin).json()["models"]
        mine = [m for m in listed if m.get("link_name") == "HyperNix.3.1-mini"]
        assert mine and mine[0]["runnable"]

    def test_linking_needs_no_model_loaded(self, server):
        client, admin, _models, tmp = server
        assert not client.get("/runner/status", headers=admin).json()["loaded"]
        other = tmp / "other"
        other.mkdir()
        got = client.post("/hyperlink/models/link",
                          json={"path": str(_gguf(other / "m.gguf"))}, headers=admin)
        assert got.status_code == 200, got.text


def test_the_cli_has_auto():
    from hypernix.t1api.runner_cli import build_parser

    args = build_parser().parse_args(["auto", "--most-used", "--dry-run"])
    assert args.command == "auto" and args.most_used and args.dry_run
