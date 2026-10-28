"""``hnx download --latest``, and the newest HyperNix model through T1's relay (0.72.8rc1).

``--latest`` downloads the newest HyperNix release (HyperNix.3.1-mini as
this is written). The runner fetches a release asked for by name, and
registers what it loads, so ``/inference/relay`` can read a document
through it -- a 512-token model, which the relay's fixed 512-token
answer used to refuse outright.
"""
from __future__ import annotations

import io
import json
import socket
import urllib.parse
from pathlib import Path

import pytest

from hypernix.hyperlink import latest as hl
from hypernix.runtime import relay


class _Listing:
    """urlopen for the Hub's /api/models, one listing per author."""

    def __init__(self, by_owner: dict[str, list[str]], fail: tuple[str, ...] = ()):
        self.by_owner, self.fail, self.asked = by_owner, fail, []

    def __call__(self, url, timeout=None):
        owner = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["author"][0]
        self.asked.append(owner)
        if owner in self.fail:
            raise OSError("offline")
        return io.BytesIO(json.dumps([{"id": r} for r in self.by_owner.get(owner, [])]).encode())


class TestTheReleaseLine:
    def test_the_newest_known_is_3_1_mini(self):
        newest = hl.latest(online=False)
        assert newest.version == "3.1" and newest.repo == "Compactbot/hypernix-3.1-mini"
        assert newest.folder == "HyperNix.3.1-mini" and newest.model_id == "hypernix.3.1-mini"

    @pytest.mark.parametrize("name", ["hypernix.3.1-mini", "Compactbot/hypernix-3.1-mini",
                                      "HyperNix.3.1-mini", "latest", "hypernix-latest"])
    def test_a_release_is_found_by_any_of_its_names(self, name):
        assert hl.find(name).version == "3.1"

    def test_other_models_are_not_releases(self):
        assert hl.find("qwen3-8b") is None and hl.find("") is None
        assert hl.find("ray0rf1re/HyperNix.3-mini").version == "3"

    def test_versions_sort_as_numbers(self):
        assert hl._from_repo("ray0rf1re/HyperNix.3.10-mini").key > hl._from_repo("ray0rf1re/HyperNix.3.9-mini").key


class TestAskingTheHub:
    def test_a_later_release_from_a_trusted_account_wins(self, monkeypatch):
        monkeypatch.setattr(hl, "safe_urlopen", _Listing({
            "ray0rf1re": ["ray0rf1re/HyperNix.3-mini", "ray0rf1re/hypernix-tools"],
            "Compactbot": ["Compactbot/hypernix-3.1-mini", "Compactbot/hypernix-4-small"],
        }))
        newest = hl.latest()
        assert newest.repo == "Compactbot/hypernix-4-small" and newest.folder == "HyperNix.4-small"

    def test_a_known_release_keeps_its_description(self, monkeypatch):
        monkeypatch.setattr(hl, "safe_urlopen", _Listing({"Compactbot": ["Compactbot/hypernix-3.1-mini"]}))
        assert "20,000" in hl.latest().note

    def test_a_strangers_hypernix_is_never_the_latest(self, monkeypatch):
        """Anybody can name a repo hypernix-9-mini; the search is by author,
        and an id from another account is dropped even if the Hub returns it."""
        monkeypatch.setattr(hl, "safe_urlopen", _Listing({"ray0rf1re": ["mallory/hypernix-9-mini"]}))
        assert hl.latest().version == "3.1"
        assert hl._from_repo("mallory/HyperNix.9-mini") is None
        assert hl._from_repo("ray0rf1re/HyperNix.9-mini-finetune") is None

    def test_offline_is_the_known_line(self, monkeypatch):
        listing = _Listing({}, fail=hl.TRUSTED_OWNERS)
        monkeypatch.setattr(hl, "safe_urlopen", listing)
        assert hl.discover() == [] and hl.latest().version == "3.1"
        assert set(listing.asked) == set(hl.TRUSTED_OWNERS)

    def test_a_plain_http_endpoint_is_not_asked(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "http://hub.test")
        monkeypatch.setattr(hl, "safe_urlopen", lambda *a, **k: pytest.fail("asked over http"))
        assert hl.discover() == []


class TestTheCommand:
    def _run(self, monkeypatch, argv, newest=None):
        from hypernix.hyperlink import brewed
        from hypernix.interfaces import cli

        fetched = []
        monkeypatch.setattr(hl, "latest", lambda online=True: newest or hl.RELEASES[-1])
        monkeypatch.setattr(brewed, "download", lambda repo, dest: fetched.append((repo, Path(dest))) or Path(dest))
        return cli._run_download(argv), fetched

    def test_latest_downloads_the_newest_into_the_models_folder(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr("hypernix.hyperlink.catalogue.DEFAULT_LOCAL_DIR", tmp_path)
        code, fetched = self._run(monkeypatch, ["--latest"])
        assert code == 0
        assert fetched == [("Compactbot/hypernix-3.1-mini", tmp_path / "HyperNix.3.1-mini")]
        out = capsys.readouterr().out
        assert "HyperNix.3.1-mini" in out and "runner start hypernix.3.1-mini" in out
        assert "/inference/relay" in out

    def test_o_chooses_the_folder(self, monkeypatch, tmp_path):
        code, fetched = self._run(monkeypatch, ["--latest", "-o", str(tmp_path / "here"), "--quiet"])
        assert code == 0 and fetched[0][1] == tmp_path / "here"

    def test_latest_with_a_source_is_refused(self, monkeypatch, capsys):
        code, fetched = self._run(monkeypatch, ["--latest", "owner/repo"])
        assert code == 2 and not fetched
        assert "--latest names the model" in capsys.readouterr().err


class TestTheRelaySizesItsAnswer:
    def test_a_512_token_model_can_relay(self):
        doc = "\n\n".join(f"Section {i}: Alder kept {i * 3} boats." * 4 for i in range(60))
        plan = relay.plan(doc, "How many boats?", context_tokens=512)
        assert not plan.single and plan.largest_call_tokens <= 512
        assert relay.default_answer_tokens(512) == 115

    def test_a_big_context_still_answers_at_length(self):
        assert relay.default_answer_tokens(128_000) == 512
        assert relay.default_answer_tokens(64) == 32

    def test_an_answer_asked_for_that_cannot_fit_is_refused(self):
        with pytest.raises(relay.ContextTooSmall):
            relay.plan("x " * 4000, "q", context_tokens=512, answer_tokens=512)


# ---------------------------------------------------------------------------
# Through T1: fetched by name, loaded, registered, relayed
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


def _tiny_release(folder: Path) -> Path:
    """A 2-layer Brewer model with HyperNix.3.1-mini's 512-token context."""
    from safetensors.torch import save_model

    from hypernix.training.brewer import BrewerConfig, BrewerModel

    folder.mkdir(parents=True)
    config = BrewerConfig(vocab_size=512, n_layers=2, n_heads=4, n_kv_heads=2, d_model=64,
                          max_seq_len=512, use_sliding_window=False, name="hypernix.3.1-mini")
    torch.manual_seed(0)
    save_model(BrewerModel(config), str(folder / "model.safetensors"))
    config.save(folder / "config.json")
    return folder


@pytest.fixture
def server(tmp_path, monkeypatch):
    from conftest import clear_t1_config

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "t1"))
    monkeypatch.setenv("T1_DEFAULT_MODEL", "")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        monkeypatch.setenv("T1_RUNNER_PORT", str(sock.getsockname()[1]))
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
    models = tmp_path / "models"
    models.mkdir()
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough", db_path=str(tmp_path / "t1.db"),
        module_storage_dir=str(tmp_path / "m"), hyperlink_files_dir=str(tmp_path / "f"),
        default_plan="free", hf_download_dir=str(models), web_enabled=False,
    )
    client = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk), client=("127.0.0.1", 5000))
    admin = {"Authorization": "Bearer " + km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key}
    yield client, admin, models
    client.post("/runner/unload", headers=admin, json={})


class TestThroughT1:
    def test_a_release_named_to_the_runner_is_fetched_into_the_models_folder(self, server, monkeypatch):
        client, admin, models = server
        fetched = []

        def fake_download(repo, dest):
            fetched.append((repo, Path(dest)))
            return _tiny_release(Path(dest))

        monkeypatch.setattr("hypernix.hyperlink.brewed.download", fake_download)
        got = client.post("/runner/load", headers=admin,
                          json={"model_id": "hypernix.3.1-mini", "backend": "hnx-cpu"})
        assert got.status_code == 200, got.text
        assert fetched == [("Compactbot/hypernix-3.1-mini", models / "HyperNix.3.1-mini")]

        # Loading registered it, so the relay can ask it.
        doc = " ".join(f"Section {i}: the town of Alder kept {i * 3} boats and traded wool." for i in range(40))
        estimate = client.post("/inference/relay", headers=admin, json={
            "model": "hypernix.3.1-mini", "document": doc, "question": "How many boats?",
            "estimate_only": True})
        assert estimate.status_code == 200, estimate.text
        relayed = client.post("/inference/relay", headers=admin, json={
            "model": "hypernix.3.1-mini", "document": doc, "question": "How many boats?"})
        assert relayed.status_code == 200, relayed.text
        body = relayed.json()
        assert len(body["hops"]) == estimate.json()["plan"]["requests"] > 2
        assert body["hops"][-1]["kind"] == "answer"

    def test_an_operators_entry_is_left_alone(self):
        from hypernix.t1api.registry import ModelRegistry
        from hypernix.t1api.routers.runner import _ensure_registered

        registry = ModelRegistry()
        current = type("Loaded", (), {"context_length": 512})()
        _ensure_registered(registry, "fresh", current, {"architecture": "hyperNix0x-v2"})
        entry = registry.get("fresh")
        assert entry.context_limit == 512 and entry.free_tier_available and entry.local_available
        # A second load, with another context, does not overwrite it.
        _ensure_registered(registry, "fresh", type("Loaded", (), {"context_length": 7})(), {})
        assert registry.get("fresh").context_limit == 512
