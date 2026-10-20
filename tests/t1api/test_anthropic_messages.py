"""``/v1/messages``: the Anthropic Messages API, for Claude Code on HyperNix models.

The translation is tested against dicts; the router against a fake
OpenAI backend, the same way ``/inference`` is.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api import anthropic as wire  # noqa: E402
from hypernix.t1api import claudecode  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.registry import ModelEntry, ModelPricing, ModelRegistry, ModelStatus  # noqa: E402

BRIDGE = "hypernix.t1api.routers.inference.LMStudioBridge"


def _model(model_id: str) -> ModelEntry:
    return ModelEntry(
        model_id=model_id, display_name=model_id, version="1.0", total_parameters=7.0,
        active_parameters=None, architecture="dense", supported_tasks=["chat"], availability="public",
        minimum_plan="free", free_tier_available=True, api_available=True, local_available=True,
        remote_available=True, context_limit=8000, input_token_limit=8000, output_token_limit=2000,
        tool_call_limit=4, pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=2.0),
        routing_priority=10, fallback_model=None, license="apache-2.0", status=ModelStatus.AVAILABLE,
    )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

class TestAnthropicToOpenAI:
    def test_system_and_text(self):
        out = wire.to_openai({"system": [{"type": "text", "text": "Be brief."}], "max_tokens": 9,
                              "messages": [{"role": "user", "content": "hi"}]})
        assert out["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
        assert out["max_tokens"] == 9

    def test_a_tool_round_trip_keeps_openai_order(self):
        body = {"max_tokens": 50, "messages": [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure."},
                                              {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                               "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "text", "text": "and then?"},
                                         {"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": [{"type": "text", "text": "a.py"}]}]},
        ], "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}}
        out = wire.to_openai(body)
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["user", "assistant", "tool", "user"]      # the tool result follows its call
        call = out["messages"][1]["tool_calls"][0]
        assert call["id"] == "toolu_1" and json.loads(call["function"]["arguments"]) == {"command": "ls"}
        assert out["messages"][2] == {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py"}
        assert out["tools"][0]["function"]["name"] == "Bash"
        assert out["extra"]["tool_choice"] == "required"

    def test_an_error_result_says_so(self):
        out = wire.to_openai({"max_tokens": 5, "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "no such file", "is_error": True}]}]})
        assert out["messages"][0]["content"] == "Error: no such file"

    def test_images_become_image_url_parts(self):
        out = wire.to_openai({"max_tokens": 5, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
            {"type": "text", "text": "what is it"}]}]})
        parts = out["messages"][0]["content"]
        assert parts[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}

    def test_thinking_blocks_sent_back_are_dropped(self):
        out = wire.to_openai({"max_tokens": 5, "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "x"}, {"type": "text", "text": "ok"}]}]})
        assert out["messages"] == [{"role": "assistant", "content": "ok"}]

    def test_a_named_tool_choice(self):
        out = wire.to_openai({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}],
                              "tools": [{"name": "Read", "input_schema": {}}],
                              "tool_choice": {"type": "tool", "name": "Read"}})
        assert out["extra"]["tool_choice"] == {"type": "function", "function": {"name": "Read"}}


class TestOpenAIToAnthropic:
    def test_text(self):
        reply = wire.from_openai({"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 2}}, model="m")
        assert reply["content"] == [{"type": "text", "text": "hello"}]
        assert reply["stop_reason"] == "end_turn" and reply["usage"]["input_tokens"] == 7
        assert reply["id"].startswith("msg_") and reply["type"] == "message"

    def test_tool_calls(self):
        reply = wire.from_openai({"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_9", "function": {"name": "Bash", "arguments": "{\"command\": \"ls\"}"}}]},
            "finish_reason": "tool_calls"}]}, model="m")
        assert reply["stop_reason"] == "tool_use"
        assert reply["content"] == [{"type": "tool_use", "id": "call_9", "name": "Bash",
                                     "input": {"command": "ls"}}]

    def test_arguments_that_are_not_quite_json(self):
        reply = wire.from_openai({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "Read", "arguments": "sure: {\"path\": \"a\"} done"}}]}}]}, model="m")
        assert reply["content"][0]["input"] == {"path": "a"}
        broken = wire.from_openai({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "Read", "arguments": "{oops"}}]}}]}, model="m")
        assert broken["content"][0]["input"] == {"_unparsed_arguments": "{oops"}

    def test_length_is_max_tokens(self):
        reply = wire.from_openai({"choices": [{"message": {"content": "a"}, "finish_reason": "length"}]}, model="m")
        assert reply["stop_reason"] == "max_tokens"


def _events(chunks: list[bytes]) -> list[tuple[str, dict]]:
    out = []
    for raw in chunks:
        lines = raw.decode().strip().splitlines()
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


class TestTheStream:
    def test_text_then_a_tool_call(self):
        t = wire.StreamTranslator(model="m", input_tokens=10)
        raw = t.start()
        raw += t.feed({"choices": [{"delta": {"content": "Let me "}}]})
        raw += t.feed({"choices": [{"delta": {"content": "look."}}]})
        raw += t.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "Bash", "arguments": "{\"comm"}}]}}]})
        raw += t.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "and\":1}"}}]},
                                    "finish_reason": "tool_calls"}]})
        raw += t.end()
        events = _events(raw)
        names = [e for e, _ in events]
        assert names == ["message_start", "ping", "content_block_start", "content_block_delta",
                         "content_block_delta", "content_block_stop", "content_block_start",
                         "content_block_delta", "content_block_delta", "content_block_stop",
                         "message_delta", "message_stop"]
        assert events[2][1]["index"] == 0 and events[6][1]["index"] == 1
        assert events[6][1]["content_block"]["name"] == "Bash"
        partial = "".join(d["delta"]["partial_json"] for e, d in events if d.get("delta", {}).get("type") == "input_json_delta")
        assert json.loads(partial) == {"command": 1}
        assert events[-2][1]["delta"]["stop_reason"] == "tool_use"

    def test_an_empty_answer_still_has_a_block(self):
        t = wire.StreamTranslator(model="m")
        names = [e for e, _ in _events(t.end())]
        assert names[:2] == ["message_start", "ping"] and "content_block_start" in names


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def config(tmp_path) -> T1APIConfig:
    return T1APIConfig(token_secret="test-secret-value-that-is-long-enough",
                       db_path=str(tmp_path / "t1.sqlite3"), module_storage_dir=str(tmp_path / "modules"),
                       hyperlink_files_dir=str(tmp_path / "files"), lmstudio_url="http://lmstudio.test:1234",
                       default_plan="free", claude_code_model="model-a")


@pytest.fixture
def client(km, config, tmp_path) -> TestClient:
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)
    registry = ModelRegistry()
    registry.register(_model("model-a"))
    return TestClient(create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry),
                      client=("203.0.113.9", 5000))


@pytest.fixture
def key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


def _backend(content="hi there", tool_calls=None):
    instance = mock.MagicMock()
    instance.base_url = "http://lmstudio.test:1234"
    message = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    instance.chat.return_value = {"choices": [{"message": message, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    instance.chat_stream.return_value = iter([
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"content": " there"}, "finish_reason": "stop"}]},
    ])
    return instance


BODY = {"model": "model-a", "max_tokens": 64, "messages": [{"role": "user", "content": "hello"}]}


class TestTheRouter:
    def test_a_message(self, client, key):
        with mock.patch(BRIDGE) as bridge:
            bridge.return_value = _backend()
            r = client.post("/v1/messages", json=BODY, headers={"x-api-key": key,
                                                                 "anthropic-version": "2023-06-01"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["content"] == [{"type": "text", "text": "hi there"}] and body["model"] == "model-a"
        assert body["usage"] == {"input_tokens": 12, "output_tokens": 3, "cache_creation_input_tokens": 0,
                                 "cache_read_input_tokens": 0}

    def test_a_claude_model_name_is_the_configured_model(self, client, key):
        with mock.patch(BRIDGE) as bridge:
            bridge.return_value = _backend()
            r = client.post("/v1/messages", json={**BODY, "model": "claude-sonnet-4-5"},
                            headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200 and r.json()["model"] == "model-a"
        assert bridge.return_value.chat.call_args.kwargs["model"] == "model-a"

    def test_tools_reach_the_backend(self, client, key):
        with mock.patch(BRIDGE) as bridge:
            bridge.return_value = _backend(content=None, tool_calls=[
                {"id": "c1", "function": {"name": "Bash", "arguments": "{}"}}])
            r = client.post("/v1/messages", headers={"x-api-key": key}, json={
                **BODY, "tools": [{"name": "Bash", "input_schema": {"type": "object"}}]})
        sent = bridge.return_value.chat.call_args.kwargs["tools"]
        assert sent[0]["function"]["name"] == "Bash"
        assert r.json()["stop_reason"] == "tool_use" and r.json()["content"][0]["type"] == "tool_use"

    def test_no_credential_is_an_anthropic_401(self, client):
        r = client.post("/v1/messages", json=BODY)
        assert r.status_code == 401
        assert r.json()["type"] == "error" and r.json()["error"]["type"] == "authentication_error"

    def test_the_keyless_token_is_no_credential_not_a_bypass(self, client):
        r = client.post("/v1/messages", json=BODY, headers={"Authorization": "Bearer t1-keyless"})
        assert r.status_code == 401

    def test_an_unknown_model(self, client, key):
        r = client.post("/v1/messages", json={**BODY, "model": "nope"}, headers={"x-api-key": key})
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
        assert "model-a" in r.json()["error"]["message"]

    def test_max_tokens_is_required(self, client, key):
        r = client.post("/v1/messages", json={"model": "model-a", "messages": BODY["messages"]},
                        headers={"x-api-key": key})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"

    def test_a_stream(self, client, key):
        with mock.patch(BRIDGE) as bridge:
            bridge.return_value = _backend()
            with client.stream("POST", "/v1/messages", json={**BODY, "stream": True},
                               headers={"x-api-key": key}) as r:
                text = "".join(r.iter_text())
        assert r.status_code == 200
        events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
        assert events[0] == "message_start" and events[-1] == "message_stop"
        deltas = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
        assert "".join(d["delta"].get("text", "") for d in deltas if d.get("type") == "content_block_delta") == "hi there"
        assert bridge.return_value.chat_stream.call_args.kwargs["extra"]["stream_options"] == {"include_usage": True}

    def test_count_tokens(self, client, key):
        r = client.post("/v1/messages/count_tokens", json={"model": "model-a", "messages": BODY["messages"]},
                        headers={"x-api-key": key})
        assert r.status_code == 200 and r.json()["input_tokens"] >= 1

    def test_models(self, client, key):
        r = client.get("/v1/models", headers={"x-api-key": key})
        assert [m["id"] for m in r.json()["data"]] == ["model-a"]

    def test_it_can_be_turned_off(self, km, config, tmp_path, key):
        config.anthropic_enabled = False
        gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk2", log_to_file=False)
        off = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk, registry=ModelRegistry()))
        assert off.post("/v1/messages", json=BODY, headers={"x-api-key": key}).status_code == 404


class TestHnxClaude:
    def test_the_environment(self):
        env = claudecode.claude_env("http://127.0.0.1:8000/", "T1-key", "qwen", base={"ANTHROPIC_API_KEY": "sk-x"})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "T1-key" and "ANTHROPIC_API_KEY" not in env
        assert env["ANTHROPIC_MODEL"] == env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen"

    def test_no_key_is_the_keyless_token(self):
        assert claudecode.claude_env("http://x", "", "m", base={})["ANTHROPIC_AUTH_TOKEN"] == claudecode.KEYLESS

    def test_the_loaded_model_is_picked(self):
        models = [{"id": "a", "display_name": "a"}, {"id": "b", "display_name": "b (loaded)"}]
        assert claudecode._pick(models, "") == "b"
        with pytest.raises(RuntimeError, match="not on the server"):
            claudecode._pick(models, "c")

    def test_print_env(self, monkeypatch, capsys):
        monkeypatch.setattr(claudecode, "available_models", lambda server, key: [{"id": "m", "display_name": "m"}])
        assert claudecode.cli_main(["--server", "http://h:1", "--key", "k", "--print-env"]) == 0
        out = capsys.readouterr().out
        assert 'export ANTHROPIC_BASE_URL="http://h:1"' in out and "unset ANTHROPIC_API_KEY" in out
