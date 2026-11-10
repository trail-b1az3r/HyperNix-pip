"""Strife on T1 and in HyperLink (0.73.0 alpha 1).

* ``hypernix.strife.server`` speaks the OpenAI API the way llama-server
  does, so everything that talks to the runner talks to it unchanged;
* the runner has a ``strife`` backend that starts it;
* and, end to end: T1 loads a model on Strife through ``/runner/load``,
  and a HyperLink chat turn is answered by it -- a real process, no stubs.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request

import pytest

from hypernix.strife import Engine, SamplerConfig
from hypernix.strife.fixtures import make_model
from hypernix.strife.server import build_server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, body: dict, *, raw: bool = False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            return resp.status, data if raw else json.loads(data)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    model = make_model(tmp_path_factory.mktemp("srv") / "q.gguf", arch="qwen35", seed=2)
    engine = Engine(model["path"], n_ctx=256, max_sessions=4)
    port = _free_port()
    server = build_server(engine, port=port, alias="tiny-qwen35")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", engine
    server.shutdown()
    server.server_close()
    engine.close()


class TestTheOpenAISurface:
    def test_health_and_models(self, served):
        base, _ = served
        with urllib.request.urlopen(base + "/health") as resp:
            assert json.loads(resp.read())["engine"] == "strife"
        with urllib.request.urlopen(base + "/v1/models") as resp:
            model = json.loads(resp.read())["data"][0]
        assert model["id"] == "tiny-qwen35" and model["architecture"] == "qwen35"

    def test_a_chat_is_the_engines_answer_through_the_template(self, served):
        base, engine = served
        messages = [{"role": "user", "content": "hello world"}]
        status, body = _post(base + "/v1/chat/completions",
                             {"messages": messages, "max_tokens": 6, "temperature": 0})
        assert status == 200
        expected = engine.generate(messages=messages, max_tokens=6,
                                   sampling=SamplerConfig(temperature=0, repeat_penalty=1.0))
        assert body["choices"][0]["message"]["content"] == expected.text
        assert body["usage"]["completion_tokens"] == len(expected.tokens)

    def test_streaming_adds_up_to_the_same_reply(self, served):
        base, _ = served
        req = {"messages": [{"role": "user", "content": "the model"}], "max_tokens": 6,
               "temperature": 0}
        _, whole = _post(base + "/v1/chat/completions", req)
        status, raw = _post(base + "/v1/chat/completions", dict(req, stream=True), raw=True)
        assert status == 200
        lines = [line[6:] for line in raw.decode().split("\n") if line.startswith("data: ")]
        assert lines[-1] == "[DONE]"
        chunks = [json.loads(line) for line in lines[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == whole["choices"][0]["message"]["content"]
        assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")

    def test_completions(self, served):
        base, _ = served
        status, body = _post(base + "/v1/completions", {"prompt": "hello", "max_tokens": 4,
                                                        "temperature": 0})
        assert status == 200 and body["object"] == "text_completion"

    def test_a_follow_up_reuses_the_conversation(self, served):
        base, engine = served
        first = [{"role": "user", "content": "hello world and the model"}]
        _, a = _post(base + "/v1/chat/completions", {"messages": first, "max_tokens": 3, "temperature": 0})
        follow = first + [{"role": "assistant", "content": a["choices"][0]["message"]["content"]},
                          {"role": "user", "content": "and then"}]
        _, b = _post(base + "/v1/chat/completions", {"messages": follow, "max_tokens": 3, "temperature": 0})
        assert b["strife"]["prompt_tokens_reused"] > 0

    @pytest.mark.parametrize("body,code", [({"messages": []}, 400), ({"messages": "hi"}, 400)])
    def test_bad_requests_are_400(self, served, body, code):
        base, _ = served
        status, err = _post(base + "/v1/chat/completions", body)
        assert status == code and err["error"]["message"]

    def test_unknown_route(self, served):
        base, _ = served
        status, _ = _post(base + "/v1/embeddings", {"input": "x"})
        assert status == 404


class TestTheRunner:
    def test_strife_is_a_backend_choice(self):
        from hypernix.hyperlink.managed import BACKENDS, plan_placement

        assert "strife" in BACKENDS
        plan = plan_placement(file_bytes=1 << 20, total_layers=4, gpu_layers=10, backend="strife")
        assert plan.backend == "cpu" and plan.gpu_layers == 0 and "Strife" in plan.reason

    def test_a_model_strife_cannot_run_is_refused_with_its_reason(self, tmp_path):
        from hypernix.hyperlink.managed import ManagedError, ManagedRunner

        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"GGUF" + b"\0" * 64)
        with pytest.raises(ManagedError, match="strife.model"):
            ManagedRunner(port=_free_port()).load(bad, backend="strife")

    def test_the_runner_starts_strife_and_it_answers(self, tmp_path):
        from hypernix.hyperlink.managed import ManagedRunner

        model = make_model(tmp_path / "tiny-lfm3.gguf", arch="lfm3", seed=1)
        runner = ManagedRunner(port=_free_port())
        try:
            record = runner.load(model["path"], model_id="tiny-lfm3", backend="strife", timeout=120)
            assert record.placement.backend == "cpu" and "Strife" in record.placement.reason
            status, body = _post(runner.base_url + "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
            assert status == 200 and body["model"] == "tiny-lfm3"
        finally:
            runner.unload()


class TestEndToEndOnT1AndHyperLink:
    @pytest.fixture
    def t1(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from conftest import clear_t1_config
        from fastapi.testclient import TestClient

        models = tmp_path / "models"
        models.mkdir()
        make_model(models / "tiny-glm.gguf", arch="glm4moe", seed=4)
        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        # Loading a model changes it for everybody: an operator grants that.
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")
        monkeypatch.setenv("T1_HYPERLINK_SHARED_KEYLESS_CHATS", "1")
        monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")
        monkeypatch.setenv("T1_HF_DOWNLOAD_DIR", str(models))
        monkeypatch.setenv("T1_RUNNER_PORT", str(_free_port()))
        monkeypatch.setenv("HOME", str(tmp_path))

        from hypernix.t1api.app import create_app

        app = create_app()
        client = TestClient(app, client=("192.168.1.9", 5000))
        yield app, client
        app.state.t1_runner.unload()

    def test_load_on_strife_then_a_hyperlink_chat_is_answered_by_it(self, t1):
        app, client = t1
        model_id = "tiny-glm"  # the catalogue's id for tiny-glm.gguf
        # HyperLink's system prompt and tools are thousands of tokens.
        loaded = client.post("/runner/load", json={"model_id": model_id, "backend": "strife",
                                                   "context_length": 8192})
        assert loaded.status_code == 200, loaded.text
        assert "strife" in loaded.json()["backends"]
        assert "Strife" in loaded.json()["model"]["placement"]["reason"]

        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        deadline = time.time() + 60
        while True:
            reply = client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hello"})
            if reply.status_code == 200 or time.time() > deadline:
                break
            time.sleep(0.5)
        assert reply.status_code == 200, reply.text
        body = reply.json()
        assert body["backend"] == "hypernix"
        assert isinstance(body["assistant_message"]["content"], str)
        assert app.state.t1_runner.current.model_id == model_id

    def test_a_refusal_names_the_runner_not_lm_studio(self, t1):
        _app, client = t1
        client.post("/runner/load", json={"model_id": "tiny-glm", "backend": "strife",
                                          "context_length": 64})
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        reply = client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hello"})
        assert reply.status_code >= 400
        message = reply.json()["error"]["message"]
        assert "HyperNix runner" in message and "LM Studio" not in message
