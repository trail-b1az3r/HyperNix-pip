"""Hypercompress (0.72.8): a system prompt bigger than the loaded model's
context is cut to a third (s=1) or a fifth (s=2) of it before the turn."""
from __future__ import annotations

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink import hypercompress as hc

PROMPT = "\n\n".join(
    [
        "You are HyperNix, a careful assistant. Never invent facts. Always say when you are unsure. "
        + " ".join(f"Background detail number {i} that only matters sometimes." for i in range(40)),
        "About Sam:\nSam writes Rust and likes short answers. "
        + " ".join(f"Sam once mentioned hobby {i} in passing." for i in range(30)),
        "Remembered:\n- Sam's server is called atlas.\n- Sam prefers metric units.",
    ]
)


class TestTheModule:
    def test_the_two_sizes(self):
        assert hc.target_tokens(9000, 1) == 3000
        assert hc.target_tokens(9000, 2) == 1800

    @pytest.mark.parametrize("raw", ["1", "2", "[2]", 1])
    def test_sizes_parse(self, raw):
        assert hc.parse_size(raw) in (1, 2)

    @pytest.mark.parametrize("raw", ["0", "3", "", "big"])
    def test_other_sizes_are_refused(self, raw):
        with pytest.raises(ValueError):
            hc.parse_size(raw)

    def test_a_prompt_that_fits_is_left_alone(self):
        result = hc.maybe_compress(PROMPT, context_length=100_000)
        assert not result.applied and result.text == PROMPT

    def test_an_unknown_context_is_left_alone(self):
        assert not hc.maybe_compress(PROMPT, context_length=0).applied

    def test_too_big_is_cut_to_its_share_keeping_instructions_and_every_section(self):
        context = hc.estimate_tokens(PROMPT) // 2
        for size in (1, 2):
            result = hc.maybe_compress(PROMPT, context_length=context, size=size)
            assert result.applied
            assert result.after_tokens <= hc.target_tokens(context, size)
            assert "Never invent facts" in result.text
            assert "About Sam" in result.text and "Remembered" in result.text
        assert (hc.maybe_compress(PROMPT, context_length=context, size=2).after_tokens
                < hc.maybe_compress(PROMPT, context_length=context, size=1).after_tokens)

    def test_the_conversation_counts_towards_fitting(self):
        context = hc.estimate_tokens(PROMPT) + 100
        assert not hc.maybe_compress(PROMPT, context_length=context).applied
        assert hc.maybe_compress(PROMPT, context_length=context, other_tokens=500).applied

    def test_repeats_go_first(self):
        text = "Always answer in English. " * 50 + "Be brief."
        out = hc.compress(text, 20)
        assert out.count("Always answer in English") == 1 and "Be brief." in out

    def test_one_enormous_sentence_is_cut_at_a_word(self):
        out = hc.compress("word " * 2000, 50)
        assert hc.estimate_tokens(out) <= 52 and out.endswith("…")


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
    from hypernix.t1api.app import create_app

    app = create_app()
    return app, TestClient(app, client=("192.168.1.9", 5000))


class TestTheEndpoints:
    def test_size_defaults_then_the_documented_form_sets_it(self, app_and_client):
        _app, client = app_and_client
        assert client.get("/chat/hypercompress/size").json()["size"] == 1
        body = client.get("/chat/hypercompress/size?s?=2").json()
        assert body == {**body, "size": 2, "share": "1/5", "changed": True}
        assert client.get("/chat/hypercompress/size").json()["size"] == 2
        assert client.post("/chat/hypercompress/size?s=1").json()["size"] == 1

    def test_a_bad_size_is_refused(self, app_and_client):
        _app, client = app_and_client
        assert client.get("/chat/hypercompress/size?s?=7").status_code in (400, 422)

    def test_compressing_by_hand(self, app_and_client):
        _app, client = app_and_client
        body = client.post("/chat/hypercompress", json={
            "text": PROMPT, "context_length": 1024, "force": True, "size": 2}).json()
        assert body["applied"] and body["after_tokens"] <= 1024 // 5 and body["share"] == "1/5"


class TestATurn:
    def test_a_small_model_gets_a_compressed_system_prompt(self, app_and_client, monkeypatch):
        app, client = app_and_client

        class Loaded:
            model_id = "tiny"
            context_length = 800

        class FakeRunner:
            base_url = "http://127.0.0.1:8081"
            current = Loaded()

        app.state.t1_runner = FakeRunner()
        sent = {}

        class StubClient:
            base_url = "http://127.0.0.1:8081/v1"

            def chat(self, messages, *, model=None, **kwargs):
                sent["messages"] = messages
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                        "model": model, "usage": {}}

        from hypernix.hyperlink import inference
        from hypernix.t1api.routers import hyperlink as hyperlink_router

        # Only the person's own prompt, so the test does not move every
        # time the default prompt's wording does.
        monkeypatch.setattr(hyperlink_router, "default_prompt_for", lambda config, backend: "")

        monkeypatch.setattr(inference, "_openai_client", lambda base_url, timeout=300.0: StubClient())
        client.patch("/hyperlink/preferences", json={"system_prompt": PROMPT})
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        response = client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hello"})
        assert response.status_code == 200, response.text
        system = sent["messages"][0]
        assert system["role"] == "system"
        # The tool format a model is taught is added after, whole: cutting
        # it would break tool calls. Everything before it was compressed.
        assert system["content"].count("Background detail number") < 40
        assert "Never invent facts" in system["content"]
        meta = response.json()["assistant_message"]["metadata"]["hypercompressed"]
        assert meta["applied"] and meta["context_length"] == 800 and meta["share"] == "1/3"
        assert meta["after_tokens"] <= 800 // 3 < meta["before_tokens"]
