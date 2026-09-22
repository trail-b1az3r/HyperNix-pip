"""The T1 API as an MCP server.

Two audiences, and keeping them apart is most of the design:

* a **JSON-RPC error** is for the client library — malformed request,
  unknown method, bad params. The model never sees it.
* an **`isError` result** is for the model — "that model is not on this
  server", "the runner is busy". Something it can read and act on.

Getting that backwards produces either a model that cannot see why its
call failed, or a client library handling a protocol error that was
really a normal outcome.

The other thing tested hard here is that a tool the caller cannot run
is not *listed*. A model shown a tool it lacks the scope for will try
it, be refused, and try again — the refusal is not information it can
act on, because nothing it does changes its own scopes.
"""
from __future__ import annotations

import json

import pytest

from hypernix.t1api.mcp import (
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_PARSE_ERROR,
    PROTOCOL_VERSION,
    MCPResource,
    MCPServer,
    MCPTool,
    ToolError,
)


def echo_tool(**overrides) -> MCPTool:
    defaults = dict(
        name="echo",
        description="Echo the text back.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: f"you said {text}",
        scope="",
    )
    defaults.update(overrides)
    return MCPTool(**defaults)


@pytest.fixture
def server() -> MCPServer:
    built = MCPServer(server_version="1.2.3")
    built.add_tool(echo_tool())
    built.add_tool(echo_tool(name="read_thing", scope="read",
                             handler=lambda: "a thing"))
    built.add_tool(echo_tool(name="write_thing", scope="write", mutating=True,
                             handler=lambda: "written"))
    return built


def call(server: MCPServer, method: str, params=None, identifier=1, **kwargs):
    message = {"jsonrpc": "2.0", "id": identifier, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message, **kwargs)


# ---------------------------------------------------------------------------


class TestHandshake:
    def test_initialize_reports_the_protocol_and_the_server(self, server):
        result = call(server, "initialize",
                      {"protocolVersion": PROTOCOL_VERSION})["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert result["serverInfo"] == {"name": "hypernix-t1", "version": "1.2.3"}
        assert "tools" in result["capabilities"]

    def test_a_version_mismatch_is_reported_not_refused(self, server):
        """The client decides whether it can work with what we speak.
        Saying so is more useful than refusing."""
        result = call(server, "initialize",
                      {"protocolVersion": "1999-01-01"})["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert "1999-01-01" in result["_note"]

    def test_ping_answers(self, server):
        assert call(server, "ping")["result"] == {}

    def test_prompts_list_is_declared_and_empty(self, server):
        """Rather than absent: a client that gets method-not-found
        cannot tell "no prompts" from "this server is too old", and
        retries."""
        assert call(server, "prompts/list")["result"] == {"prompts": []}


class TestNotifications:
    def test_a_notification_gets_no_reply(self, server):
        """By JSON-RPC a message with no id must get no response.
        Replying is the commonest way to break a client: it is not
        expecting one, so it treats it as the answer to whatever it
        asks next."""
        assert server.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        ) is None

    def test_a_malformed_notification_still_gets_no_reply(self, server):
        assert server.handle({"jsonrpc": "2.0"}) is None
        assert server.handle({"jsonrpc": "1.0", "method": "x"}) is None

    def test_a_batch_of_only_notifications_returns_nothing(self, server):
        body = json.dumps([
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ])
        assert server.handle_text(body) is None


class TestTheTwoKindsOfError:
    def test_bad_json_is_a_protocol_error(self, server):
        """For the client library, not the model."""
        reply = server.handle_text("{not json")
        assert reply["error"]["code"] == JSONRPC_PARSE_ERROR

    def test_an_unknown_method_is_a_protocol_error(self, server):
        reply = call(server, "tools/frobnicate")
        assert reply["error"]["code"] == JSONRPC_METHOD_NOT_FOUND
        assert "tools/call" in reply["error"]["message"]

    def test_a_missing_jsonrpc_version_is_a_protocol_error(self, server):
        reply = server.handle({"id": 1, "method": "ping"})
        assert reply["error"]["code"] == JSONRPC_INVALID_REQUEST

    def test_non_object_params_are_a_protocol_error(self, server):
        reply = server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1, 2]}
        )
        assert reply["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_an_unknown_tool_is_a_result_the_model_can_read(self, server):
        """Not a protocol error: "that tool does not exist here" is
        something the model should see and correct."""
        reply = call(server, "tools/call",
                     {"name": "nope", "arguments": {}}, is_admin=True)
        assert "error" not in reply
        assert reply["result"]["isError"] is True
        # And it says what there *is*, so the next attempt can be right.
        assert "echo" in reply["result"]["content"][0]["text"]

    def test_a_failing_tool_is_a_result_not_an_exception(self, server):
        def angry():
            raise ToolError("the runner is busy")

        server.add_tool(echo_tool(name="angry", handler=angry, input_schema={
            "type": "object", "properties": {}}))
        reply = call(server, "tools/call", {"name": "angry", "arguments": {}})
        assert reply["result"]["isError"] is True
        assert "busy" in reply["result"]["content"][0]["text"]

    def test_an_unexpected_exception_is_a_protocol_error(self, server):
        """A handler that raises something other than ToolError has a
        bug, and a bug is not a message for the model."""
        def broken():
            raise RuntimeError("off by one")

        server.add_tool(echo_tool(name="broken", handler=broken, input_schema={
            "type": "object", "properties": {}}))
        reply = call(server, "tools/call", {"name": "broken", "arguments": {}})
        assert "error" in reply

    def test_a_missing_required_argument_is_named(self, server):
        reply = call(server, "tools/call", {"name": "echo", "arguments": {}})
        assert reply["result"]["isError"] is True
        assert "text" in reply["result"]["content"][0]["text"]


class TestScopes:
    def test_a_tool_the_caller_cannot_run_is_not_listed(self, server):
        """The whole point. A model shown a tool it cannot use will try
        it, be refused, and try again."""
        names = [
            t["name"]
            for t in call(server, "tools/list", scopes={"read"})["result"]["tools"]
        ]
        assert "read_thing" in names
        assert "write_thing" not in names

    def test_no_scopes_sees_only_the_unscoped(self, server):
        names = [
            t["name"]
            for t in call(server, "tools/list", scopes=set())["result"]["tools"]
        ]
        assert names == ["echo"]

    def test_admin_sees_everything(self, server):
        names = [
            t["name"]
            for t in call(server, "tools/list", is_admin=True)["result"]["tools"]
        ]
        assert {"echo", "read_thing", "write_thing"} == set(names)

    def test_calling_past_a_scope_is_refused_before_the_handler(self, server):
        called = []
        server.add_tool(echo_tool(
            name="guarded", scope="admin",
            handler=lambda: called.append(1) or "ran",
            input_schema={"type": "object", "properties": {}},
        ))
        reply = call(server, "tools/call",
                     {"name": "guarded", "arguments": {}}, scopes={"read"})
        assert reply["result"]["isError"] is True
        assert not called, "the handler ran despite the scope check"

    def test_the_refusal_says_retrying_will_not_help(self, server):
        reply = call(server, "tools/call",
                     {"name": "write_thing", "arguments": {}}, scopes={"read"})
        text = reply["result"]["content"][0]["text"]
        assert "operator" in text or "will change that" in text


class TestToolCalls:
    def test_a_string_result_becomes_text_content(self, server):
        reply = call(server, "tools/call",
                     {"name": "echo", "arguments": {"text": "hi"}})
        assert reply["result"]["content"] == [
            {"type": "text", "text": "you said hi"}
        ]
        assert reply["result"]["isError"] is False

    def test_a_dict_result_is_serialised(self, server):
        server.add_tool(echo_tool(
            name="facts", handler=lambda: {"a": 1},
            input_schema={"type": "object", "properties": {}},
        ))
        reply = call(server, "tools/call", {"name": "facts", "arguments": {}})
        assert json.loads(reply["result"]["content"][0]["text"]) == {"a": 1}

    def test_an_argument_the_handler_does_not_take_is_dropped(self, server):
        """A schema and a signature drift apart. Passing an unexpected
        keyword turns that drift into a TypeError the model cannot
        read."""
        reply = call(server, "tools/call",
                     {"name": "echo", "arguments": {"text": "hi", "extra": 1}})
        assert reply["result"]["isError"] is False

    def test_a_handler_taking_kwargs_gets_everything(self, server):
        seen = {}
        server.add_tool(echo_tool(
            name="greedy", handler=lambda **kw: seen.update(kw) or "ok",
            input_schema={"type": "object", "properties": {}},
        ))
        call(server, "tools/call", {"name": "greedy", "arguments": {"a": 1, "b": 2}})
        assert seen == {"a": 1, "b": 2}

    def test_a_mutating_tool_says_so(self, server):
        """So a client can decide whether to ask a human. A read is not
        the same kind of decision as a write."""
        tools = {
            t["name"]: t
            for t in call(server, "tools/list", is_admin=True)["result"]["tools"]
        }
        assert tools["write_thing"]["annotations"]["destructiveHint"] is True
        assert tools["read_thing"]["annotations"]["readOnlyHint"] is True


class TestResources:
    def test_a_resource_can_be_listed_and_read(self, server):
        server.add_resource(MCPResource(
            uri="test://thing", name="Thing", description="A thing.",
            mime_type="text/plain", reader=lambda: "contents",
        ))
        listed = call(server, "resources/list")["result"]["resources"]
        assert listed[0]["uri"] == "test://thing"

        read = call(server, "resources/read", {"uri": "test://thing"})
        assert read["result"]["contents"][0]["text"] == "contents"

    def test_an_unknown_uri_is_a_readable_error(self, server):
        reply = call(server, "resources/read", {"uri": "test://absent"})
        assert reply["result"]["isError"] is True


class TestPlugins:
    class Demo:
        name = "demo"

        def tools(self):
            return [echo_tool(name="hello", handler=lambda: "hi from the plugin",
                              input_schema={"type": "object", "properties": {}})]

        def resources(self):
            return []

    def test_a_plugin_contributes_tools_under_its_own_name(self, server):
        """Namespaced so a plugin cannot shadow a built-in, accidentally
        or otherwise."""
        server.add_plugin(self.Demo())
        assert "demo.hello" in server.tool_names
        assert server.plugins == ["demo"]

    def test_a_plugin_cannot_shadow_a_builtin(self, server):
        class Impostor:
            name = "x"

            def tools(self):
                return [echo_tool(name="echo")]

            def resources(self):
                return []

        server.add_plugin(Impostor())
        # The built-in is untouched; the plugin's lives beside it.
        assert "echo" in server.tool_names and "x.echo" in server.tool_names
        reply = call(server, "tools/call",
                     {"name": "echo", "arguments": {"text": "hi"}})
        assert reply["result"]["content"][0]["text"] == "you said hi"

    def test_a_nameless_plugin_is_refused(self, server):
        class Anonymous:
            name = ""

            def tools(self):
                return []

            def resources(self):
                return []

        with pytest.raises(ValueError, match="name"):
            server.add_plugin(Anonymous())

    def test_the_same_plugin_twice_is_refused(self, server):
        server.add_plugin(self.Demo())
        with pytest.raises(ValueError, match="already loaded"):
            server.add_plugin(self.Demo())

    def test_a_duplicate_tool_name_is_refused(self, server):
        """Which of the two is reachable would depend on registration
        order, so this is refused rather than resolved."""
        with pytest.raises(ValueError, match="unreachable"):
            server.add_tool(echo_tool())


class TestBuiltInTools:
    def test_the_t1_tools_are_all_there(self):
        from hypernix.t1api.mcptools import ServerContext, build_server

        built = build_server(ServerContext(version="9.9"))
        assert set(built.tool_names) == {
            "hardware", "list_models", "runner_load", "runner_status",
            "server_version",
        }

    def test_loading_a_model_is_marked_as_mutating(self):
        """It evicts whatever is answering. A client that cannot tell
        that from a read will let a model switch models mid-conversation
        to answer a question about model names."""
        from hypernix.t1api.mcptools import ServerContext, build_server

        built = build_server(ServerContext(version="9.9"))
        tools = {
            t["name"]: t
            for t in built.handle(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, is_admin=True
            )["result"]["tools"]
        }
        assert tools["runner_load"]["annotations"]["destructiveHint"] is True
        assert tools["list_models"]["annotations"]["readOnlyHint"] is True

    def test_the_load_description_warns_about_eviction(self):
        """The description is the interface — a model reads it on every
        call, with no docs and no second chance."""
        from hypernix.t1api.mcptools import ServerContext, build_server

        built = build_server(ServerContext(version="9.9"))
        listing = built.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, is_admin=True
        )["result"]["tools"]
        load = next(t for t in listing if t["name"] == "runner_load")
        assert "EVICTS" in load["description"]

    def test_loading_without_a_runner_says_so(self):
        from hypernix.t1api.mcptools import ServerContext, build_server

        built = build_server(ServerContext(version="9.9", runner=None))
        reply = built.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "runner_load", "arguments": {"model_id": "x"}}},
            is_admin=True,
        )
        assert reply["result"]["isError"] is True
        assert "no built-in runner" in reply["result"]["content"][0]["text"]


class TestOverHTTP:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from conftest import clear_t1_config
        from fastapi.testclient import TestClient

        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")
        monkeypatch.setenv("T1_MCP_ENABLED", "1")

        from hypernix.t1api.app import create_app

        return TestClient(create_app(), client=("192.168.1.9", 5000))

    def test_the_handshake_works_over_http(self, client):
        response = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL_VERSION},
        })
        assert response.status_code == 200
        assert response.json()["result"]["protocolVersion"] == PROTOCOL_VERSION

    def test_a_notification_gets_202_and_no_body(self, client):
        """Returning `null` here makes a client treat it as the answer
        to its next request."""
        response = client.post("/mcp", json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        assert response.status_code == 202
        assert response.text == ""

    def test_a_tool_can_be_called(self, client):
        response = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "server_version", "arguments": {}},
        })
        assert response.json()["result"]["isError"] is False

    def test_get_describes_it_for_a_person(self, client):
        """The first thing anybody does with an MCP endpoint is open it,
        and a 405 says nothing about whether the URL is right."""
        body = client.get("/mcp").json()
        assert body["protocol"] == "mcp"
        assert body["tools"]

    def test_it_is_off_unless_enabled(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from conftest import clear_t1_config
        from fastapi.testclient import TestClient

        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")
        monkeypatch.delenv("T1_MCP_ENABLED", raising=False)

        from hypernix.t1api.app import create_app

        client = TestClient(create_app(), client=("192.168.1.9", 5000))
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                                             "method": "ping"})
        assert response.status_code == 501
        assert "T1_MCP_ENABLED" in response.json()["error"]["message"]

    def test_an_enormous_body_is_refused_before_parsing(self, client):
        response = client.post(
            "/mcp",
            content=b'{"jsonrpc":"2.0","id":1,"method":"ping","pad":"'
            + b"x" * 1_000_100 + b'"}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
