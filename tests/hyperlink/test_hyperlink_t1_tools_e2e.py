"""A HyperLink model calling the T1 API — end to end, over a real socket.

What this is for
----------------
Every piece of this was tested alone: reading a text-written tool call,
calling a T1 endpoint, the tool loop. What only an end-to-end test can
show is the path through all of them. A paired phone sends a message;
the server's own model writes `<tool_call>` the way a GGUF does; the
server calls *itself* over HTTP with the phone's device token; the
result goes back to the model; the phone gets the answer.

TestClient has no listening socket, so a server calling its own API from
inside a request would find nothing there. This runs the app under a
real uvicorn on a free port, and talks to it over HTTP.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")

from conftest import clear_t1_config  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


class Loaded:
    model_id = "local-gguf"


class FakeRunner:
    base_url = "http://127.0.0.1:1"
    current = Loaded()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def served(tmp_path, monkeypatch):
    clear_t1_config(monkeypatch)
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    app = create_app(
        config=T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
            hyperlink_files_dir=str(tmp_path / "files"),
            noodle_enabled=True,
        ),
        keymaster=km,
        gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
    )
    app.state.t1_runner = FakeRunner()
    admin = km.create(key_type=KeyType.ADMIN,
                      scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    base = f"http://127.0.0.1:{port}"
    yield base, admin
    server.should_exit = True
    thread.join(timeout=5)


def _pair(base, admin) -> str:
    code = httpx.post(f"{base}/hyperlink/pair", json={"label": "phone"},
                      headers={"Authorization": f"Bearer {admin}"}).json()["code"]
    redeemed = httpx.post(f"{base}/hyperlink/pair/redeem", json={
        "code": code, "device_name": "iPhone", "app_version": "1.0"})
    assert redeemed.status_code == 200, redeemed.text
    return redeemed.json()["device_token"]


class ScriptedModel:
    """A GGUF that calls a tool the way it was trained to: as text."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, *, model=None, tools=None, **kwargs):
        self.seen.append({"messages": list(messages), "tools": tools})
        content = self.replies.pop(0)
        return {"choices": [{"message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}], "model": model, "usage": {}}


def test_a_gguf_calls_the_t1_api_from_hyperlink(served, monkeypatch):
    base, admin = served
    model = ScriptedModel(
        '<tool_call>{"name": "server_version", "arguments": {}}</tool_call>',
        "The server is up to date.",
    )
    from hypernix.hyperlink import inference
    monkeypatch.setattr(inference, "_openai_client",
                        lambda base_url, timeout=300.0: model)

    device = _pair(base, admin)
    auth = {"Authorization": f"Bearer {device}"}
    assert httpx.patch(f"{base}/hyperlink/preferences",
                       json={"tools_enabled": True}, headers=auth).status_code == 200
    session = httpx.post(f"{base}/hyperlink/sessions", json={"title": "t"},
                         headers=auth).json()["session"]["session_id"]

    reply = httpx.post(f"{base}/hyperlink/sessions/{session}/chat",
                       json={"content": "Is the server current?"},
                       headers=auth, timeout=30)
    assert reply.status_code == 200, reply.text
    body = reply.json()

    # The phone gets the answer, not the tool-call markup.
    assert body["assistant_message"]["content"] == "The server is up to date."

    # The model was offered the T1 tool…
    offered = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
    assert "server_version" in offered
    # …taught the text format, since this is the built-in runner…
    assert "<tool_call>" in model.seen[0]["messages"][0]["content"]
    # …and on the second turn it had the endpoint's real answer.
    tool_messages = [m for m in model.seen[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages and "hypernix" in tool_messages[0]["content"].lower()


def test_a_device_is_not_offered_the_mutating_tools(served, monkeypatch):
    """A device token is never an admin credential, and a model acting
    for one must not be handed load and unload."""
    base, admin = served
    model = ScriptedModel("hello")
    from hypernix.hyperlink import inference
    monkeypatch.setattr(inference, "_openai_client",
                        lambda base_url, timeout=300.0: model)
    device = _pair(base, admin)
    auth = {"Authorization": f"Bearer {device}"}
    httpx.patch(f"{base}/hyperlink/preferences", json={"tools_enabled": True}, headers=auth)
    session = httpx.post(f"{base}/hyperlink/sessions", json={"title": "t"},
                         headers=auth).json()["session"]["session_id"]
    httpx.post(f"{base}/hyperlink/sessions/{session}/chat",
               json={"content": "hi"}, headers=auth, timeout=30)
    offered = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
    assert "runner_status" in offered
    assert "runner_unload" not in offered and "runner_load" not in offered
