"""T1 and MCP clients, both ways (0.72.8a3).

T1 as a server, as real clients expect it (version negotiation, the
session id, 405 for a server-to-client stream), and T1 as a *client* of
remote MCP servers such as the Hugging Face Hub's, whose tools then
appear on T1's own /mcp and in HyperLink's tool loop.

The client is tested end to end against a real T1 server on a loopback
port, so both halves of the transport are this repository's code and
nothing leaves the machine.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest

from hypernix.t1api import mcpclient
from hypernix.t1api.mcp import SUPPORTED_VERSIONS
from hypernix.t1api.mcpclient import MCPClient, MCPHub, MCPRemoteError, RemoteServer, result_text


def _enable(monkeypatch, tmp_path, **extra):
    from conftest import clear_t1_config

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")
    monkeypatch.setenv("T1_MCP_ENABLED", "1")
    monkeypatch.setenv("T1_WEB_ENABLED", "0")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _enable(monkeypatch, tmp_path)
    from hypernix.t1api.app import create_app

    return TestClient(create_app(), client=("192.168.1.9", 5000))


class TestTheServerSpeaksTheTransport:
    @pytest.mark.parametrize("version", SUPPORTED_VERSIONS)
    def test_the_version_asked_for_is_answered_in(self, client, version):
        reply = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                          "params": {"protocolVersion": version}})
        assert reply.json()["result"]["protocolVersion"] == version
        assert reply.headers.get("mcp-session-id")

    def test_an_event_stream_get_is_405(self, client):
        reply = client.get("/mcp", headers={"Accept": "text/event-stream"})
        assert reply.status_code == 405 and "POST" in reply.headers["allow"]

    def test_a_browser_still_gets_the_description(self, client):
        reply = client.get("/mcp", headers={"Accept": "text/html,application/json"})
        assert reply.status_code == 200 and reply.json()["protocol"] == "mcp"

    def test_ending_a_session(self, client):
        assert client.delete("/mcp").status_code == 204


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A real T1 server on a loopback port."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    _enable(monkeypatch, tmp_path)
    from hypernix.t1api.app import create_app

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


class TestTheClient:
    def test_against_a_real_server(self, live):
        client = MCPClient(RemoteServer(name="self", url=live, timeout=10))
        tools = client.list_tools()
        assert client.protocol == SUPPORTED_VERSIONS[0] and client.session_id
        assert "server_version" in {t["name"] for t in tools}
        result = client.call_tool("server_version", {})
        assert not result.get("isError")

    def test_an_unreachable_server_is_said_plainly(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = MCPClient(RemoteServer(name="gone", url=f"http://127.0.0.1:{port}/mcp", timeout=2))
        with pytest.raises(MCPRemoteError, match="unreachable"):
            client.list_tools()

    def test_an_event_stream_reply_is_read(self):
        client = MCPClient(RemoteServer(name="s", url="http://127.0.0.1:1/mcp"))
        asked = []
        client._post = lambda message: asked.append(message)
        stream = [b"event: message\n", b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n',
                  b"\n", b'data: {"jsonrpc":"2.0","id":"q","method":"sampling/createMessage"}\n', b"\n",
                  b'data: {"jsonrpc":"2.0","id":7,"result":{"tools":[]}}\n', b"\n"]
        assert client._from_stream(iter(stream), 7) == {"jsonrpc": "2.0", "id": 7, "result": {"tools": []}}
        assert asked and asked[0]["id"] == "q" and asked[0]["error"]["code"] == -32601

    def test_an_expired_session_starts_a_new_one(self, monkeypatch):
        client = MCPClient(RemoteServer(name="s", url="http://127.0.0.1:1/mcp"))
        client.protocol, client.session_id = "2025-11-25", "old"
        calls = []

        def post(message):
            calls.append(message.get("method"))
            if message.get("method") == "tools/list" and calls.count("tools/list") == 1:
                raise mcpclient._Expired()
            if message.get("method") == "initialize":
                client.session_id = "new"
                return {"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": "2025-06-18"}}
            if "id" not in message:
                return None
            return {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [{"name": "t"}]}}

        client._post = post
        assert client.list_tools() == [{"name": "t"}]
        assert calls == ["tools/list", "initialize", "notifications/initialized", "tools/list"]
        assert client.session_id == "new" and client.protocol == "2025-06-18"


class TestServersAndTokens:
    def test_a_name_and_a_scheme_are_checked(self):
        with pytest.raises(ValueError, match="server name"):
            RemoteServer(name="Bad Name", url="https://x")
        with pytest.raises(ValueError, match="https"):
            RemoteServer(name="plain", url="http://example.org/mcp")
        RemoteServer(name="local", url="http://127.0.0.1:9/mcp")      # this machine may be plain http

    def test_the_token_is_never_shown_and_is_kept_owner_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOME_TOKEN", "hf_secret")
        hub = MCPHub(tmp_path / "servers.json")
        hub.add(RemoteServer(name="hf", url="https://huggingface.co/mcp", token_env="SOME_TOKEN"))
        hub.add(RemoteServer(name="other", url="https://example.org/mcp", token="tok_secret"))
        rows = {r["name"]: r for r in hub.status()}
        assert rows["hf"]["token_set"] and rows["other"]["token_set"]
        assert "hf_secret" not in json.dumps(rows) and "tok_secret" not in json.dumps(rows)
        if os.name == "posix":
            assert (tmp_path / "servers.json").stat().st_mode & 0o077 == 0
        again = MCPHub(tmp_path / "servers.json")
        assert [s.name for s in again.servers] == ["hf", "other"] and again.servers[0].bearer == "hf_secret"
        assert again.remove("other") and not again.remove("other")

    def test_servers_from_the_environment_cannot_be_removed_here(self, tmp_path):
        hub = MCPHub(tmp_path / "s.json", extra=[dict(mcpclient.HUGGING_FACE)])
        with pytest.raises(ValueError, match="T1_MCP_REMOTE"):
            hub.remove("hf")
        assert not (tmp_path / "s.json").exists()

    def test_result_text(self):
        assert result_text({"content": [{"type": "text", "text": "a"},
                                        {"type": "resource_link", "name": "r", "uri": "hf://x"}]}) == "a\n[r] hf://x"
        assert result_text({"content": [], "structuredContent": {"n": 1}}) == '{\n "n": 1\n}'
        assert result_text({"content": [{"type": "text", "text": "no"}], "isError": True}) == "Error: no"


class _FakeHub(MCPHub):
    def __init__(self):
        super().__init__(None, extra=[{"name": "hf", "url": "https://huggingface.co/mcp"},
                                      {"name": "ops", "url": "https://ops.example/mcp", "admin_only": True}])
        self.calls = []

    def tools(self, name, *, refresh=False):
        return [{"name": "hub_repo_search", "description": "search", "annotations": {"readOnlyHint": True},
                 "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]

    def call(self, server, tool, arguments=None):
        self.calls.append((server, tool, arguments))
        return True, f"{server}:{tool}:{json.dumps(arguments, sort_keys=True)}"


class TestRemoteToolsReachEveryone:
    def test_on_t1s_own_mcp(self, client):
        client.app.state.t1_mcp_hub = hub = _FakeHub()
        listed = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()
        names = {t["name"] for t in listed["result"]["tools"]}
        assert "hf.hub_repo_search" in names
        called = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "hf.hub_repo_search", "arguments": {"query": "qwen"}}}).json()
        assert called["result"]["content"][0]["text"] == 'hf:hub_repo_search:{"query": "qwen"}'
        assert hub.calls == [("hf", "hub_repo_search", {"query": "qwen"})]

    def test_in_hyperlinks_tool_loop(self):
        from hypernix.hyperlink.toolloop import T1Access, _prepare, _run_call
        from hypernix.interfaces.noodle.tools import ToolContext

        hub = _FakeHub()
        access = T1Access("http://127.0.0.1:8000", "", mcp=hub)
        context = ToolContext(root=os.getcwd()) if "root" in ToolContext.__dataclass_fields__ else ToolContext()
        _messages, tools, t1_names = _prepare([{"role": "user", "content": "hi"}], context, access, False, True)
        names = {t["function"]["name"] for t in tools}
        assert "hf__hub_repo_search" in names and "ops__hub_repo_search" not in names   # admin-only stays hidden
        call = {"id": "c", "function": {"name": "hf__hub_repo_search", "arguments": '{"query": "lfm"}'}}
        record = _run_call(call, context, tools, access, t1_names)
        assert record.ok and record.content == 'hf:hub_repo_search:{"query": "lfm"}'

    def test_the_remote_routes(self, client):
        client.app.state.t1_mcp_hub = _FakeHub()
        listed = client.get("/mcp/remote").json()
        assert {r["name"] for r in listed["servers"]} >= {"hf"}
        tools = client.get("/mcp/remote/hf/tools").json()
        assert tools["tools"][0]["name"] == "hub_repo_search"
        # Calling a remote tool directly is an admin's check; trusted-network
        # partial admin is not that.
        assert client.post("/mcp/remote/hf/call", json={"tool": "hub_repo_search",
                                                        "arguments": {"query": "x"}}).status_code == 403

    def test_adding_needs_admin(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        _enable(monkeypatch, tmp_path)
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "0")
        from hypernix.t1api.app import create_app

        plain = TestClient(create_app(), client=("192.168.1.9", 5000))
        reply = plain.post("/mcp/remote", json={"name": "hf", "url": "https://huggingface.co/mcp"})
        assert reply.status_code == 403
