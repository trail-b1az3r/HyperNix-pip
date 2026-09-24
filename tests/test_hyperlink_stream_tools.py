"""A HyperLink model has its tools on the route the app uses (0.72.6.rc2).

Reported from an iPhone: gemma-4-e4b, asked to use HyperNix's T1 API,
answered that it had no tools. Three things were true at once:

1. The app streams (``/chat/stream``), and that route never offered a
   tool. Only ``/chat`` did.
2. Even there, every tool -- the person's own T1 API, their memories --
   waited on "Let the model use tools" *and* the server's noodle switch,
   both off by default, though that toggle is about noodle's workspace
   (files and commands).
3. On the built-in runner, the tool format went in as a second system
   message in front of the first, and a backend that keeps only the
   first system message dropped the person's instructions.

The end-to-end tests run a *default* server, the one in the screenshot,
under a real uvicorn: the model's T1 calls go back into the same server
over HTTP with the phone's own token.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from hypernix.hyperlink.toolloop import CallGuard, ToolRound, stream_tool_loop
from hypernix.interfaces.noodle.tools import ToolContext

# ---------------------------------------------------------------------------
# CallGuard: stream text, never a written tool call
# ---------------------------------------------------------------------------


def _feed(pieces):
    guard = CallGuard()
    out = "".join(guard.feed(p) for p in pieces)
    return guard, out


class TestCallGuard:
    def test_ordinary_text_streams_as_it_comes(self):
        guard = CallGuard()
        assert guard.feed("The server ") == "The server "
        assert guard.feed("is fine.") == "is fine."
        assert guard.rest() == ""

    def test_a_call_is_held_from_its_marker(self):
        guard, out = _feed(["Let me check. ", '<tool_call>{"name": "x"', ', "arguments": {}}</tool_call>'])
        assert out == "Let me check. "
        assert guard.holding and "<tool_call>" in guard.text

    def test_a_marker_split_across_chunks_is_not_leaked(self):
        guard, out = _feed(["Checking <tool", '_call>{"name": "x"}'])
        assert out == "Checking "
        assert guard.holding

    def test_a_near_miss_is_released(self):
        guard, out = _feed(["a <to", "day> tag"])
        assert out == "a <today> tag"
        assert not guard.holding

    def test_a_reply_that_opens_with_json_is_held_until_the_end(self):
        guard, out = _feed(['{"name": "server_status", "arguments": {}}'])
        assert out == "" and guard.holding
        # Not a call after all: all of it still arrives.
        assert guard.rest() == '{"name": "server_status", "arguments": {}}'

    def test_other_code_blocks_arrive_whole(self):
        guard, out = _feed(["```python\n", "print(1)\n```"])
        # The closing fence could be the start of ```json until the round
        # ends; then it is sent.
        assert out + guard.rest() == "```python\nprint(1)\n```"
        assert out.startswith("```python\nprint(1)")


# ---------------------------------------------------------------------------
# stream_tool_loop, with a scripted stream
# ---------------------------------------------------------------------------


def _chunks(text, *, size=6, tool_calls=None, finish="stop"):
    for i in range(0, len(text), size):
        yield {"model": "m", "choices": [{"delta": {"content": text[i:i + size]}}]}
    for call in tool_calls or []:
        yield {"model": "m", "choices": [{"delta": {"tool_calls": [call]}}]}
    yield {"model": "m", "choices": [{"delta": {}, "finish_reason": finish}],
           "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


class Script:
    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.asked = []

    def __call__(self, messages, tools):
        self.asked.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.rounds.pop(0)()


@pytest.fixture
def context(tmp_path):
    return ToolContext(root=tmp_path, memory_enabled=False)


def _run(context, script, **kwargs):
    return list(stream_tool_loop([{"role": "user", "content": "hi"}], context,
                                 ask_stream=script, **kwargs))


class TestStreamToolLoop:
    def test_an_answer_with_no_tool_streams_in_pieces(self, context):
        script = Script(lambda: _chunks("Hello there, how can I help?"))
        events = _run(context, script, workspace=False)
        deltas = [v for k, v in events if k == "delta"]
        assert len(deltas) > 1 and "".join(deltas) == "Hello there, how can I help?"
        assert events[-1][0] == "end" and events[-1][1]["finish_reason"] == "stop"

    def test_a_written_call_runs_and_is_never_shown(self, context, monkeypatch):
        from hypernix.hyperlink import toolloop

        monkeypatch.setattr(toolloop, "_run_call", lambda call, *a: ToolRound(
            call["function"]["name"], {}, True, "hypernix 0.72.6"))
        context.memory_enabled = True  # so update_memory is on offer
        script = Script(
            lambda: _chunks('Let me look. <tool_call>{"name": "update_memory", "arguments": {"key": "a"}}</tool_call>'),
            lambda: _chunks("It is 0.72.6."),
        )
        events = _run(context, script, workspace=False)
        text = "".join(v for k, v in events if k == "delta")
        assert text == "Let me look. It is 0.72.6."
        tools = [v for k, v in events if k == "tool"]
        assert [t.tool for t in tools] == ["update_memory"]
        # The second round saw the result.
        assert any(m.get("role") == "tool" for m in script.asked[1]["messages"])
        assert events[-1][1]["rounds"][0]["tool"] == "update_memory"

    def test_structured_streamed_calls(self, context, monkeypatch):
        from hypernix.hyperlink import toolloop

        monkeypatch.setattr(toolloop, "_run_call", lambda call, *a: ToolRound(
            call["function"]["name"], json.loads(call["function"]["arguments"]), True, "ok"))
        context.memory_enabled = True
        fragments = [
            {"index": 0, "id": "c1", "function": {"name": "read_memory", "arguments": '{"ke'}},
            {"index": 0, "function": {"arguments": 'y": "a"}'}},
        ]
        script = Script(lambda: _chunks("", tool_calls=fragments, finish="tool_calls"),
                        lambda: _chunks("done"))
        events = _run(context, script, workspace=False)
        tool = next(v for k, v in events if k == "tool")
        assert (tool.tool, tool.arguments) == ("read_memory", {"key": "a"})

    def test_without_workspace_only_memory_tools_are_offered(self, context):
        context.memory_enabled = True
        script = Script(lambda: _chunks("hi"))
        _run(context, script, workspace=False)
        names = {t["function"]["name"] for t in script.asked[0]["tools"]}
        assert names == {"update_memory", "read_memory"}

    def test_stop_ends_the_stream(self, context):
        closed = []

        class Stream:
            def __iter__(self):
                yield from _chunks("one two three four five six")

            def close(self):
                closed.append(True)

        stop = {"now": False}

        def ask(messages, tools):
            stop["now"] = True
            return Stream()

        events = list(stream_tool_loop([{"role": "user", "content": "x"}], context, ask_stream=ask,
                                       workspace=False, cancelled=lambda: stop["now"]))
        assert events[-1][1]["cancelled"] is True and closed == [True]

    def test_teaching_the_format_keeps_one_system_message(self, context):
        context.memory_enabled = True
        script = Script(lambda: _chunks("hi"))
        list(stream_tool_loop([{"role": "system", "content": "Answer in French."},
                               {"role": "user", "content": "x"}], context,
                              ask_stream=script, workspace=False, teach_format=True))
        systems = [m for m in script.asked[0]["messages"] if m["role"] == "system"]
        assert len(systems) == 1
        assert systems[0]["content"].startswith("Answer in French.")
        assert "<tool_call>" in systems[0]["content"]


# ---------------------------------------------------------------------------
# End to end: a default server, a paired phone, the streaming route
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")


class Loaded:
    model_id = "gemma-4-e4b"


class FakeRunner:
    base_url = "http://127.0.0.1:1"
    current = Loaded()


class StreamingModel:
    """A GGUF on the built-in runner: tool calls written as text."""

    base_url = "http://127.0.0.1:1/v1"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat_stream(self, messages, *, model=None, tools=None, **kwargs):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        return _chunks(self.replies.pop(0) if self.replies else "ok")

    def chat(self, messages, *, model=None, tools=None, **kwargs):  # titles
        return {"choices": [{"message": {"role": "assistant", "content": "A title"},
                             "finish_reason": "stop"}], "model": model, "usage": {}}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def served(tmp_path, monkeypatch):
    from conftest import clear_t1_config

    from hypernix.security.gatekeeper import Gatekeeper
    from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
    from hypernix.t1api.app import create_app
    from hypernix.t1api.config import T1APIConfig

    clear_t1_config(monkeypatch)

    def start(**config):
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        app = create_app(
            config=T1APIConfig(
                token_secret="test-secret-value-that-is-long-enough",
                db_path=str(tmp_path / "t1.sqlite3"),
                module_storage_dir=str(tmp_path / "modules"),
                hyperlink_files_dir=str(tmp_path / "files"),
                **config,
            ),
            keymaster=km,
            gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
        )
        app.state.t1_runner = FakeRunner()
        admin = km.create(key_type=KeyType.ADMIN,
                          scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started
        servers.append((server, thread))
        base = f"http://127.0.0.1:{port}"
        code = httpx.post(f"{base}/hyperlink/pair", json={"label": "phone"},
                          headers={"Authorization": f"Bearer {admin}"}).json()["code"]
        redeemed = httpx.post(f"{base}/hyperlink/pair/redeem", json={
            "code": code, "device_name": "iPhone", "app_version": "1.0"}).json()
        token = redeemed.get("device_token") or redeemed.get("token")
        return base, {"Authorization": f"Bearer {token}"}

    servers: list = []
    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


def _use(monkeypatch, model):
    from hypernix.hyperlink import inference

    monkeypatch.setattr(inference, "_openai_client", lambda base_url, timeout=300.0: model)


def _stream(base, auth, text):
    session = httpx.post(f"{base}/hyperlink/sessions", json={"title": "t"},
                         headers=auth).json()["session"]["session_id"]
    frames = []
    with httpx.stream("POST", f"{base}/hyperlink/sessions/{session}/chat/stream",
                      json={"content": text}, headers=auth, timeout=30) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                frames.append(json.loads(line[6:]))
    return session, frames


def test_the_screenshot_a_default_server_streams_with_t1_tools(served, monkeypatch):
    base, auth = served()  # noodle off, the tools toggle never touched
    model = StreamingModel(
        '<tool_call>{"name": "server_version", "arguments": {}}</tool_call>',
        "HyperNix is running and up to date.",
    )
    _use(monkeypatch, model)
    session, frames = _stream(base, auth, "You should have access to hyperNix's t1 api commands")

    offered = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
    assert {"server_version", "list_models", "hardware", "web_search"} <= offered
    # noodle's workspace was not switched on, so no files or commands.
    assert not offered & {"create_file", "edit_file", "execute_file", "zip_files"}

    kinds = [f["type"] for f in frames]
    assert kinds[0] == "start" and "tool" in kinds and kinds.index("tool") < kinds.index("done")
    tool = next(f for f in frames if f["type"] == "tool")
    assert tool["tool"] == "server_version" and tool["ok"] is True
    text = "".join(f["text"] for f in frames if f["type"] == "delta")
    assert text == "HyperNix is running and up to date."
    assert "<tool_call>" not in text

    # The model got the endpoint's real answer, over HTTP with the phone's token.
    results = [m for m in model.seen[1]["messages"] if m.get("role") == "tool"]
    assert results and "hypernix" in results[0]["content"].lower()

    # One system message: the default prompt first, the tool format after.
    systems = [m for m in model.seen[0]["messages"] if m["role"] == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].startswith("You are a language model running")
    assert "<tool_call>" in systems[0]["content"]

    messages = httpx.get(f"{base}/hyperlink/sessions/{session}/messages", headers=auth).json()["messages"]
    reply = messages[-1]
    assert reply["content"] == "HyperNix is running and up to date."
    assert reply["metadata"]["tool_rounds"][0]["tool"] == "server_version"


def test_the_workspace_needs_both_switches(served, monkeypatch):
    base, auth = served(noodle_enabled=True)
    model = StreamingModel("hi", "hi")
    _use(monkeypatch, model)
    _stream(base, auth, "hello")
    first = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
    assert "create_file" not in first  # the person has not turned it on
    httpx.patch(f"{base}/hyperlink/preferences", json={"tools_enabled": True}, headers=auth)
    _stream(base, auth, "hello")
    second = {t["function"]["name"] for t in model.seen[1]["tools"] or []}
    assert "create_file" in second and "server_version" in second


def test_the_non_streamed_route_agrees(served, monkeypatch):
    base, auth = served()

    class Plain(StreamingModel):
        def chat(self, messages, *, model=None, tools=None, **kwargs):
            if tools is not None or not any("You name conversations" in str(m.get("content")) for m in messages):
                self.seen.append({"messages": list(messages), "tools": tools})
            return super().chat(messages, model=model, tools=tools, **kwargs)

    model = Plain()
    _use(monkeypatch, model)
    session = httpx.post(f"{base}/hyperlink/sessions", json={"title": "t"},
                         headers=auth).json()["session"]["session_id"]
    got = httpx.post(f"{base}/hyperlink/sessions/{session}/chat", json={"content": "hi"},
                     headers=auth, timeout=30)
    assert got.status_code == 200, got.text
    offered = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
    assert "server_version" in offered
