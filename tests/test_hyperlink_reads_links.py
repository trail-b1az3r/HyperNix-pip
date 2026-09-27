"""A HyperLink model reads the link it is given (0.72.6.post1).

Reported from a phone: asked to look at a GitHub repository, the model
listed its only tools, read_memory and update_memory, and said three
times that it could not browse. The phone had no key, and a keyless
caller was offered no T1 tools, so no web_search and no web_summarize;
nothing it was told said a link could be opened either.

These drive the whole path on a real server: a keyless phone on a
trusted network, a model that is told it can open links, a call to
web_summarize with the link as pasted, and the page's text coming back.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")

import uvicorn  # noqa: E402
from test_hyperlink_stream_tools import FakeRunner, StreamingModel, _free_port  # noqa: E402

from hypernix.runtime import t1tools, toolcalls  # noqa: E402

PAGE = "HyperNix-pip: a Python package for training and serving HyperNix models."


@pytest.fixture
def keyless(tmp_path, monkeypatch):
    """A server on a trusted network, reached with no key at all."""
    from conftest import clear_t1_config

    from hypernix.t1api.app import create_app
    from hypernix.t1api.config import T1APIConfig
    from hypernix.t1api.routers import web

    clear_t1_config(monkeypatch)
    fetched: list[str] = []

    def fetch(url):
        fetched.append(url)
        return PAGE

    monkeypatch.setattr(web, "_fetch_for_summary", fetch)
    app = create_app(config=T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        trusted_network=True,
    ))
    app.state.t1_runner = FakeRunner()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}", fetched
    server.should_exit = True
    thread.join(timeout=5)


def _stream(base, text):
    session = httpx.post(f"{base}/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
    frames = []
    with httpx.stream("POST", f"{base}/hyperlink/sessions/{session}/chat/stream",
                      json={"content": text}, timeout=30) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                frames.append(json.loads(line[6:]))
    return frames


def test_a_keyless_phone_can_have_a_link_read(keyless, monkeypatch):
    from hypernix.hyperlink import inference

    base, fetched = keyless
    model = StreamingModel(
        '<tool_call>{"name": "web_summarize", "arguments": '
        '{"url": "https://github.com/trail-b1az3r/HyperNix-pip#readme"}}</tool_call>',
        "It is a Python package for training and serving HyperNix models.",
    )
    monkeypatch.setattr(inference, "_openai_client", lambda base_url, timeout=300.0: model)
    frames = _stream(base, "Look at https://github.com/trail-b1az3r/HyperNix-pip#readme")

    offered = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
    assert {"web_search", "web_summarize"} <= offered
    # Read-only: nothing that writes, for a caller with no key.
    assert not offered & {"memory_create", "runner_load", "runner_unload"}

    system = next(m for m in model.seen[0]["messages"] if m["role"] == "system")["content"]
    assert "You can reach the web in this conversation" in system
    assert "Never say you cannot browse" in system

    tool = next(f for f in frames if f["type"] == "tool")
    assert tool["tool"] == "web_summarize" and tool["ok"] is True, tool
    # The link as pasted, less the fragment the endpoint refuses.
    assert fetched == ["https://github.com/trail-b1az3r/HyperNix-pip"]
    result = next(m for m in model.seen[1]["messages"] if m.get("role") == "tool"
                  or str(m.get("content", "")).startswith("<tool_response"))
    assert "training and serving" in result["content"]


class TestWhatTheModelIsTold:
    def test_the_web_tools_say_they_open_links(self):
        tools = {t["function"]["name"]: t["function"]["description"] for t in t1tools.openai_tools()}
        assert "GitHub" in tools["web_summarize"] and "link" in tools["web_summarize"]
        assert "live web" in tools["web_search"]

    def test_the_hint_only_comes_with_a_web_tool(self):
        memory_only = [{"type": "function", "function": {"name": "read_memory", "parameters": {}}}]
        assert "reach the web" not in toolcalls.tool_prompt(memory_only)
        with_web = memory_only + t1tools.openai_tools(only=["web_summarize"])
        prompt = toolcalls.tool_prompt(with_web)
        assert "open a link the person gives you" in prompt
        assert "web_search" not in prompt.split("You can reach the web", 1)[1].split(".")[0]

    def test_the_default_prompt_says_an_offered_tool_is_an_ability(self):
        from hypernix.hyperlink.default_prompt import default_prompt

        assert "do not tell them you cannot browse" in default_prompt()

    def test_a_fragment_is_dropped_before_the_call(self, monkeypatch):
        seen = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"summary": "ok"}'

        def urlopen(request, timeout=None):
            seen["body"] = json.loads(request.data)
            return Response()

        monkeypatch.setattr(t1tools.urllib.request, "urlopen", urlopen)
        t1tools.call_tool("web_summarize", {"url": " https://github.com/o/r#readme "},
                          base_url="http://127.0.0.1:1")
        assert seen["body"] == {"url": "https://github.com/o/r"}
