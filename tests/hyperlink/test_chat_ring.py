"""The two-layer ring beside the chat box (0.72.8): context inside;
usage outside when the key and server have an allowance, else GPU or CPU
load."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink import ring


def reading(cpu=12.0, gpus=()):
    return NS(cpu=NS(percent=cpu), gpus=[NS(name=n, utilization_percent=u) for n, u in gpus])


class TestTheLayers:
    def test_context(self):
        layer = ring.context_layer(2048, 8192, from_model=True).to_dict()
        assert layer["fraction"] == 0.25 and "of the model's context" in layer["label"]
        assert "budget" in ring.context_layer(10, 8000, from_model=False).label

    def test_a_full_context_is_capped_at_one(self):
        assert ring.context_layer(9000, 4096, from_model=True).to_dict()["fraction"] == 1.0

    def test_usage_takes_the_tighter_of_input_and_output(self):
        snap = NS(input_tokens_used=100, input_token_limit=1000,
                  output_tokens_used=600, output_token_limit=1000)
        layer = ring.usage_layer(snap)
        assert layer.kind == "usage" and layer.fraction == 0.6

    def test_no_limit_means_no_usage_ring(self):
        assert ring.usage_layer(NS(input_tokens_used=5, input_token_limit=0,
                                   output_tokens_used=5, output_token_limit=0)) is None
        assert ring.usage_layer(None) is None

    def test_the_busiest_gpu(self):
        layer = ring.hardware_layer(reading=reading(gpus=[("A", 20.0), ("B", 75.0)]))
        assert layer.kind == "gpu" and layer.fraction == 0.75 and layer.label.startswith("B")

    def test_cpu_when_there_is_no_gpu(self):
        layer = ring.hardware_layer(reading=reading(cpu=40.0))
        assert layer.kind == "cpu" and layer.fraction == 0.4

    def test_cpu_when_the_model_runs_on_the_cpu(self):
        layer = ring.hardware_layer(on_gpu=False, reading=reading(cpu=90.0, gpus=[("A", 5.0)]))
        assert layer.kind == "cpu"


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_HYPERLINK_SHARED_KEYLESS_CHATS", "1")
    monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")
    monkeypatch.setattr(ring, "_snapshot", lambda: reading(cpu=33.0))
    from hypernix.t1api.app import create_app

    app = create_app()
    return app, TestClient(app, client=("192.168.1.9", 5000))


class TestTheEndpoint:
    def test_with_a_loaded_model(self, app_and_client):
        app, client = app_and_client
        app.state.t1_runner = NS(base_url="http://127.0.0.1:8081", current=NS(
            model_id="tiny", context_length=4096, placement=NS(gpu_layers=0)))
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        body = client.get(f"/hyperlink/sessions/{session}/ring").json()
        assert body["inner"]["kind"] == "context" and body["inner"]["limit"] == 4096
        assert 0 < body["inner"]["fraction"] < 1  # the default prompt is already in it
        # No key, so no allowance: the model runs on the CPU, so CPU load.
        assert body["outer"] == {**body["outer"], "kind": "cpu", "fraction": 0.33}

    def test_with_nothing_loaded_it_falls_back_to_the_budget(self, app_and_client):
        _app, client = app_and_client
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        body = client.get(f"/hyperlink/sessions/{session}/ring").json()
        assert body["inner"]["limit"] == 8000 and "budget" in body["inner"]["label"]

    def test_someone_elses_chat_is_a_404(self, app_and_client):
        _app, client = app_and_client
        assert client.get("/hyperlink/sessions/nope/ring").status_code == 404


class TestTheClientsDrawIt:
    def test_the_web_app(self):
        from pathlib import Path

        web = Path(__file__).resolve().parents[2] / "src" / "hypernix" / "hyperlink" / "web"
        assert 'id="ring-inner"' in (web / "index.html").read_text() and 'id="ring-outer"' in (web / "index.html").read_text()
        assert "/ring`" in (web / "app.js").read_text()

    def test_the_ios_app(self):
        from pathlib import Path

        ios = Path(__file__).resolve().parents[2] / "ios" / "HyperLink" / "Sources"
        assert "struct UsageRing: View" in (ios / "Views" / "ChatView.swift").read_text()
        assert "/ring\"" in (ios / "Networking" / "HyperLinkClient.swift").read_text()
