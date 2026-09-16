"""A HyperNix model runner, not just the LM Studio bridge.

For most of this project's life "which thing answers a message" had one
answer, and the code said so in the field name: `backend: "lmstudio"`,
hard-coded, on every message.

Then `hypernix.hyperlink.managed` gave the server its own llama.cpp
process — and the chat path still went straight to the bridge. So
`/runner/load` would start a model that nothing could talk to: a machine
with no LM Studio installed could load a 70B through its own runner and
then refuse every single message with "this server has no chat backend
configured". The runner worked, the status screen showed the model, and
the conversation was impossible.

This is the choice that was missing. The runner comes first, and
deliberately: if somebody loaded a model, that is the model they meant,
it is the thing holding the VRAM, and quietly answering from LM Studio
instead would answer as a different model than the one on screen.
"""
from __future__ import annotations

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink.inference import (
    HYPERNIX,
    LMSTUDIO,
    BackendUnavailable,
    describe_backends,
    resolve_backend,
)


class Config:
    def __init__(self, lmstudio: bool = False, url: str = ""):
        self.lmstudio_enabled = lmstudio
        self.lmstudio_url = url
        self.lmstudio_api_key = ""
        self.lmstudio_timeout_seconds = 30.0


class Loaded:
    def __init__(self, model_id: str = "qwen3-8b"):
        self.model_id = model_id


class Runner:
    def __init__(self, loaded: Loaded | None = None, base: str = "http://127.0.0.1:8081"):
        self.base_url = base
        self.current = loaded


class TestTheRunnerComesFirst:
    def test_a_loaded_runner_serves_with_no_lm_studio_at_all(self):
        """The whole point. This machine has no LM Studio installed and
        it can still hold a conversation."""
        backend = resolve_backend(Config(lmstudio=False), Runner(Loaded()))
        assert backend.name == HYPERNIX
        assert backend.model_id == "qwen3-8b"

    def test_it_wins_even_when_lm_studio_is_available(self):
        """Somebody who loaded a model meant that model. Answering from
        LM Studio instead would answer as a different model than the one
        the status screen is showing."""
        backend = resolve_backend(Config(lmstudio=True), Runner(Loaded()))
        assert backend.name == HYPERNIX

    def test_it_points_at_the_runners_own_port(self):
        backend = resolve_backend(Config(), Runner(Loaded(), base="http://127.0.0.1:9001"))
        assert backend.base_url.startswith("http://127.0.0.1:9001")

    def test_the_url_carries_the_openai_prefix(self):
        """llama-server serves the OpenAI surface under /v1, and a
        client pointed at the bare root gets a 404 on every request."""
        backend = resolve_backend(Config(), Runner(Loaded()))
        assert backend.base_url.endswith("/v1")


class TestFallingBackToLMStudio:
    def test_an_empty_runner_falls_through(self):
        backend = resolve_backend(Config(lmstudio=True), Runner(None))
        assert backend.name == LMSTUDIO

    def test_no_runner_at_all_falls_through(self):
        """An older server, or one built without it."""
        backend = resolve_backend(Config(lmstudio=True), None)
        assert backend.name == LMSTUDIO

    def test_a_runner_that_raises_does_not_hide_the_bridge(self):
        """A broken runner must not take the working backend down with
        it — that would turn a degraded server into a dead one."""
        class Broken:
            base_url = ""

            @property
            def current(self):
                raise RuntimeError("the process table is on fire")

        backend = resolve_backend(Config(lmstudio=True), Broken())
        assert backend.name == LMSTUDIO


class TestWhenNothingCanAnswer:
    def test_it_refuses(self):
        with pytest.raises(BackendUnavailable):
            resolve_backend(Config(lmstudio=False), Runner(None))

    def test_the_refusal_names_both_ways_out(self):
        """Naming only the one that happens to be checked first is how
        somebody ends up installing LM Studio on a machine that did not
        need it."""
        with pytest.raises(BackendUnavailable) as refused:
            resolve_backend(Config(lmstudio=False), Runner(None))
        joined = " ".join(refused.value.remedies)
        assert "runner load" in joined
        assert "LM Studio" in joined

    def test_it_says_the_server_needs_no_other_software(self):
        with pytest.raises(BackendUnavailable) as refused:
            resolve_backend(Config(lmstudio=False), None)
        assert any("no other software" in r for r in refused.value.remedies)


class TestDescribingThem:
    def test_both_are_always_listed(self):
        """Including the unavailable one. "LM Studio is off" is an
        answer; a missing row is not."""
        rows = describe_backends(Config(lmstudio=False), Runner(None))
        assert {row["name"] for row in rows} == {HYPERNIX, LMSTUDIO}

    def test_a_loaded_runner_names_its_model(self):
        rows = describe_backends(Config(), Runner(Loaded("llama-70b")))
        runner_row = next(r for r in rows if r["name"] == HYPERNIX)
        assert runner_row["available"] is True
        assert runner_row["model_id"] == "llama-70b"

    def test_an_empty_runner_says_how_to_fill_it(self):
        rows = describe_backends(Config(), Runner(None))
        runner_row = next(r for r in rows if r["name"] == HYPERNIX)
        assert runner_row["available"] is False
        assert "run models itself" in runner_row["detail"]

    def test_lm_studio_switched_off_says_the_variable(self):
        rows = describe_backends(Config(lmstudio=False), None)
        row = next(r for r in rows if r["name"] == LMSTUDIO)
        assert "T1_LMSTUDIO_ENABLED" in row["detail"]


class TestThroughTheAPI:
    @pytest.fixture
    def app_and_client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")

        from hypernix.t1api.app import create_app

        app = create_app()
        return app, TestClient(app, client=("192.168.1.9", 5000))

    def test_nothing_available_is_a_501_that_explains(self, app_and_client):
        _app, client = app_and_client
        session = client.post(
            "/hyperlink/sessions", json={"title": "t"}
        ).json()["session"]["session_id"]
        response = client.post(
            f"/hyperlink/sessions/{session}/chat", json={"content": "hello"}
        )
        assert response.status_code == 501
        remedies = response.json()["error"]["details"]["remedies"]
        assert any("runner load" in r for r in remedies)

    def test_a_loaded_runner_answers_the_turn(self, app_and_client, monkeypatch):
        """The end-to-end shape of the fix: LM Studio off, a model
        loaded by this server's own runner, and a reply."""
        app, client = app_and_client

        class FakeRunner:
            base_url = "http://127.0.0.1:8081"
            current = Loaded("qwen3-8b")

        app.state.t1_runner = FakeRunner()

        asked = {}

        class StubClient:
            base_url = "http://127.0.0.1:8081/v1"

            def chat(self, messages, *, model=None, **kwargs):
                asked["model"] = model
                asked["base_url"] = self.base_url
                return {
                    "choices": [{
                        "message": {"content": "Answered by the runner."},
                        "finish_reason": "stop",
                    }],
                    "model": model,
                    "usage": {},
                }

        from hypernix.hyperlink import inference

        monkeypatch.setattr(
            inference, "_openai_client", lambda base_url, timeout=300.0: StubClient()
        )

        session = client.post(
            "/hyperlink/sessions", json={"title": "t"}
        ).json()["session"]["session_id"]
        response = client.post(
            f"/hyperlink/sessions/{session}/chat", json={"content": "hello"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == HYPERNIX
        assert body["assistant_message"]["content"] == "Answered by the runner."
        assert asked["model"] == "qwen3-8b"

    def test_the_reply_is_not_labelled_lmstudio(self, app_and_client, monkeypatch):
        """"lmstudio" on a machine with no LM Studio is the kind of
        small lie that costs somebody an afternoon."""
        app, client = app_and_client

        class FakeRunner:
            base_url = "http://127.0.0.1:8081"
            current = Loaded()

        app.state.t1_runner = FakeRunner()

        class StubClient:
            base_url = "http://127.0.0.1:8081/v1"

            def chat(self, messages, *, model=None, **kwargs):
                return {
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {},
                }

        from hypernix.hyperlink import inference

        monkeypatch.setattr(
            inference, "_openai_client", lambda base_url, timeout=300.0: StubClient()
        )

        session = client.post(
            "/hyperlink/sessions", json={"title": "t"}
        ).json()["session"]["session_id"]
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hello"})
        messages = client.get(f"/hyperlink/sessions/{session}/messages").json()
        assistant = [m for m in messages["messages"] if m["role"] == "assistant"][0]
        assert assistant["metadata"]["backend"] == HYPERNIX

    def test_the_backends_endpoint_reports_both(self, app_and_client):
        _app, client = app_and_client
        body = client.get("/hyperlink/backends").json()
        assert {row["name"] for row in body["backends"]} == {HYPERNIX, LMSTUDIO}
        assert body["active"] == ""

    def test_the_active_backend_is_named_once_one_is_loaded(self, app_and_client):
        app, client = app_and_client

        class FakeRunner:
            base_url = "http://127.0.0.1:8081"
            current = Loaded()

        app.state.t1_runner = FakeRunner()
        assert client.get("/hyperlink/backends").json()["active"] == HYPERNIX
