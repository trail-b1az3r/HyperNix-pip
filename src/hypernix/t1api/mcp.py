"""t1api.mcp — the T1 API as an MCP server, and the plugins that extend it.

The Model Context Protocol is how an assistant discovers what a server
can do and then does it, without either side having been written
against the other. T1 already has the capabilities — models, the
runner, memory, hardware, jobs — behind a REST surface that an
assistant has to be taught, endpoint by endpoint, by whoever wires it
up. This is the same capabilities, described in a way the assistant can
read for itself.

What is here and what is not
-----------------------------
This is the protocol core: the JSON-RPC envelope, the method dispatch,
the tool and resource registries, and the plugin loader. It has **no
FastAPI import**, on purpose — the transport is forty lines in
``routers/mcp.py`` and everything that can be got wrong is in here,
where it can be tested against a dict rather than against a server.

Three decisions worth stating
-----------------------------
**Errors are results, not exceptions.** A tool that fails returns
``isError: true`` with the reason in its content, because that is a
thing the model can read and act on; a JSON-RPC error is a thing the
*client library* handles and the model never sees. The distinction
matters: "that model is not on this server" should reach the assistant,
and "your request was not valid JSON" should not.

**A tool is refused before it runs, not while it runs.** Every tool
declares the scope it needs, and dispatch checks it against the caller
before the handler is entered. A handler that checks its own
permissions is a handler that forgets to.

**Plugins are registered, never imported by name from a request.**
A plugin is a Python object handed to :meth:`MCPServer.add_plugin` at
startup by code the operator controls. There is deliberately no
"load the module this JSON names" path, because that is remote code
execution wearing a protocol.
"""
from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "PROTOCOL_VERSION",
    "MCPTool",
    "MCPResource",
    "MCPPlugin",
    "MCPServer",
    "ToolError",
    "text_content",
    "error_result",
    "JSONRPC_PARSE_ERROR",
    "JSONRPC_INVALID_REQUEST",
    "JSONRPC_METHOD_NOT_FOUND",
    "JSONRPC_INVALID_PARAMS",
    "JSONRPC_INTERNAL_ERROR",
]

#: The revision of the protocol this speaks. Sent in `initialize` and
#: checked against what the client asks for — a mismatch is reported
#: rather than papered over, because a client expecting a later
#: revision will be looking for fields that are not here.
PROTOCOL_VERSION = "2025-06-18"

# JSON-RPC 2.0 reserved codes.
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


class ToolError(Exception):
    """A tool failed in a way the *model* should be told about.

    Raised by a handler, turned into an ``isError`` result rather than a
    JSON-RPC error — see the module docstring for why those are
    different audiences.
    """


def text_content(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


def error_result(message: str) -> dict[str, Any]:
    return {"content": text_content(message), "isError": True}


@dataclass
class MCPTool:
    """One thing the server can do, described so a model can choose it.

    *scope* is checked by dispatch before the handler runs. ``""`` means
    any authenticated caller; anything else must be present in the
    caller's scopes.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    scope: str = ""
    #: True when calling this changes something. Surfaced to the client
    #: so it can decide whether to ask a human first — a read is not the
    #: same kind of decision as a write, and a protocol that describes
    #: them identically pushes that judgement onto nobody.
    mutating: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "readOnlyHint": not self.mutating,
                "destructiveHint": self.mutating,
            },
        }


@dataclass
class MCPResource:
    """Something readable by URI. Context rather than action."""

    uri: str
    name: str
    description: str
    mime_type: str
    reader: Callable[[], str]
    scope: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@runtime_checkable
class MCPPlugin(Protocol):
    """What a plugin has to be.

    A name, and a chance to contribute tools and resources. Anything
    satisfying this shape works — there is no base class to inherit and
    no entry point to register, because the operator hands the object
    over at startup and that is the whole security model.
    """

    name: str

    def tools(self) -> Iterable[MCPTool]:
        ...

    def resources(self) -> Iterable[MCPResource]:
        ...


@dataclass
class MCPServer:
    """The protocol, over a registry of tools and resources."""

    server_name: str = "hypernix-t1"
    server_version: str = ""
    instructions: str = ""

    _tools: dict[str, MCPTool] = field(default_factory=dict, init=False)
    _resources: dict[str, MCPResource] = field(default_factory=dict, init=False)
    _plugins: list[str] = field(default_factory=list, init=False)

    # -- registration ------------------------------------------------

    def add_tool(self, tool: MCPTool) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"two tools are called {tool.name!r}. A duplicate name means "
                f"one of them is unreachable, and which one depends on "
                f"registration order — so this is refused rather than "
                f"resolved."
            )
        self._tools[tool.name] = tool

    def add_resource(self, resource: MCPResource) -> None:
        if resource.uri in self._resources:
            raise ValueError(f"two resources claim {resource.uri!r}")
        self._resources[resource.uri] = resource

    def add_plugin(self, plugin: MCPPlugin) -> None:
        """Take a plugin's tools and resources.

        Namespaced under the plugin's name so a plugin cannot shadow a
        built-in tool, accidentally or otherwise. `memory.search` stays
        `memory.search` whatever a plugin calls its own.
        """
        if not getattr(plugin, "name", ""):
            raise ValueError("a plugin needs a name to namespace its tools under")
        if plugin.name in self._plugins:
            raise ValueError(f"a plugin called {plugin.name!r} is already loaded")

        for tool in plugin.tools():
            namespaced = MCPTool(
                name=f"{plugin.name}.{tool.name}",
                description=tool.description,
                input_schema=tool.input_schema,
                handler=tool.handler,
                scope=tool.scope,
                mutating=tool.mutating,
            )
            self.add_tool(namespaced)
        for resource in plugin.resources():
            self.add_resource(resource)
        self._plugins.append(plugin.name)
        logger.info("t1api.mcp: loaded plugin %s", plugin.name)

    @property
    def plugins(self) -> list[str]:
        return list(self._plugins)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    # -- dispatch ----------------------------------------------------

    def handle(
        self,
        message: Any,
        *,
        scopes: Iterable[str] = (),
        is_admin: bool = False,
    ) -> dict[str, Any] | None:
        """One JSON-RPC message in, one response out — or None.

        None for a *notification*, which by JSON-RPC has no id and must
        get no reply. Returning an error for one is the commonest way to
        break a client: it is not expecting a response, so it either
        ignores a valid one or treats it as the answer to whatever it
        asks next.
        """
        if not isinstance(message, dict):
            return self._error(None, JSONRPC_INVALID_REQUEST,
                               "a JSON-RPC message is an object")

        identifier = message.get("id")
        is_notification = "id" not in message
        method = message.get("method")

        if message.get("jsonrpc") != "2.0":
            if is_notification:
                return None
            return self._error(identifier, JSONRPC_INVALID_REQUEST,
                               'every message needs "jsonrpc": "2.0"')
        if not isinstance(method, str):
            if is_notification:
                return None
            return self._error(identifier, JSONRPC_INVALID_REQUEST,
                               "a JSON-RPC message needs a method name")

        params = message.get("params") or {}
        if not isinstance(params, dict):
            if is_notification:
                return None
            return self._error(identifier, JSONRPC_INVALID_PARAMS,
                               "params must be an object")

        if is_notification:
            self._notify(method, params)
            return None

        try:
            result = self._call(method, params, set(scopes), is_admin)
        except ToolError as exc:
            return self._ok(identifier, error_result(str(exc)))
        except Exception as exc:  # noqa: BLE001 - a bad handler is a 500, not a crash
            logger.exception("t1api.mcp: %s failed", method)
            return self._error(identifier, JSONRPC_INTERNAL_ERROR, str(exc))

        if result is _METHOD_NOT_FOUND:
            return self._error(
                identifier, JSONRPC_METHOD_NOT_FOUND,
                f"no method {method!r}; this server speaks "
                f"{', '.join(sorted(_METHODS))}",
            )
        return self._ok(identifier, result)

    def handle_text(self, body: str, **kwargs) -> dict[str, Any] | list | None:
        """A raw body in, for the transport. Handles batches."""
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            return self._error(None, JSONRPC_PARSE_ERROR, f"invalid JSON: {exc}")

        if isinstance(parsed, list):
            if not parsed:
                return self._error(None, JSONRPC_INVALID_REQUEST,
                                   "an empty batch is not a request")
            replies = [self.handle(item, **kwargs) for item in parsed]
            answered = [reply for reply in replies if reply is not None]
            # A batch of nothing but notifications gets no body at all,
            # which is what the spec says and what clients expect.
            return answered or None
        return self.handle(parsed, **kwargs)

    # -- methods -----------------------------------------------------

    def _call(self, method: str, params: dict, scopes: set[str], is_admin: bool):
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [t.describe() for t in self._visible_tools(scopes, is_admin)]}
        if method == "tools/call":
            return self._call_tool(params, scopes, is_admin)
        if method == "resources/list":
            return {
                "resources": [
                    r.describe() for r in self._resources.values()
                    if _allowed(r.scope, scopes, is_admin)
                ]
            }
        if method == "resources/read":
            return self._read_resource(params, scopes, is_admin)
        if method == "prompts/list":
            # Declared and empty rather than absent: a client that asks
            # and gets a method-not-found cannot tell "no prompts" from
            # "this server is too old", and retries.
            return {"prompts": []}
        return _METHOD_NOT_FOUND

    def _initialize(self, params: dict) -> dict[str, Any]:
        asked = str(params.get("protocolVersion") or "")
        result: dict[str, Any] = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version or "unknown",
            },
        }
        if self.instructions:
            result["instructions"] = self.instructions
        if asked and asked != PROTOCOL_VERSION:
            # Not an error: the client decides whether it can work with
            # what we speak. Saying so is more useful than refusing.
            result["_note"] = (
                f"this server speaks {PROTOCOL_VERSION}; you asked for "
                f"{asked}"
            )
        return result

    def _visible_tools(self, scopes: set[str], is_admin: bool) -> list[MCPTool]:
        """Only the tools this caller could actually run.

        Listing a tool the caller cannot call produces a model that
        tries it, gets refused, and tries again — the refusal is not
        information it can act on, because nothing it does will change
        its own scopes.
        """
        return [
            tool for name, tool in sorted(self._tools.items())
            if _allowed(tool.scope, scopes, is_admin)
        ]

    def _call_tool(self, params: dict, scopes: set[str], is_admin: bool):
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ToolError("tools/call needs a tool name")

        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(t.name for t in self._visible_tools(scopes, is_admin))
            raise ToolError(
                f"no tool called {name!r}. This server offers: {known or '(none)'}"
            )
        if not _allowed(tool.scope, scopes, is_admin):
            raise ToolError(
                f"{name} needs the {tool.scope!r} scope, which this key does "
                f"not have. Nothing you can send will change that — ask the "
                f"operator for a key with it."
            )

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")

        missing = _missing_required(tool.input_schema, arguments)
        if missing:
            raise ToolError(
                f"{name} needs {', '.join(missing)}, which "
                f"{'was' if len(missing) == 1 else 'were'} not given"
            )

        # Only what the handler declares. A schema and a signature drift
        # apart, and passing an unexpected keyword turns that drift into
        # a TypeError the model cannot read.
        accepted = _accepted_arguments(tool.handler, arguments)
        outcome = tool.handler(**accepted)

        if isinstance(outcome, dict) and "content" in outcome:
            return outcome
        if isinstance(outcome, str):
            return {"content": text_content(outcome), "isError": False}
        return {
            "content": text_content(json.dumps(outcome, indent=2, default=str)),
            "isError": False,
        }

    def _read_resource(self, params: dict, scopes: set[str], is_admin: bool):
        uri = params.get("uri")
        resource = self._resources.get(uri) if isinstance(uri, str) else None
        if resource is None:
            raise ToolError(f"no resource at {uri!r}")
        if not _allowed(resource.scope, scopes, is_admin):
            raise ToolError(f"{uri} needs the {resource.scope!r} scope")
        return {
            "contents": [
                {
                    "uri": resource.uri,
                    "mimeType": resource.mime_type,
                    "text": resource.reader(),
                }
            ]
        }

    def _notify(self, method: str, params: dict) -> None:
        # `notifications/initialized` is the only one that matters, and
        # it needs nothing done. Logged rather than ignored so an
        # unexpected one is visible.
        if method != "notifications/initialized":
            logger.debug("t1api.mcp: notification %s %s", method, params)

    # -- envelopes ---------------------------------------------------

    @staticmethod
    def _ok(identifier: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": identifier, "result": result}

    @staticmethod
    def _error(identifier: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": identifier,
            "error": {"code": code, "message": message},
        }


_METHOD_NOT_FOUND = object()

_METHODS = {
    "initialize", "ping", "tools/list", "tools/call",
    "resources/list", "resources/read", "prompts/list",
}


def _allowed(scope: str, scopes: set[str], is_admin: bool) -> bool:
    if not scope:
        return True
    if is_admin or "admin" in scopes:
        return True
    return scope in scopes


def _missing_required(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    required = schema.get("required") or []
    if not isinstance(required, list):
        return []
    return [name for name in required if name not in arguments]


def _accepted_arguments(
    handler: Callable[..., Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    """The subset of *arguments* the handler actually takes.

    A handler with ``**kwargs`` gets everything; otherwise anything it
    does not name is dropped. The alternative is a TypeError about an
    unexpected keyword, which reaches the model as an internal error
    rather than as something it can correct.
    """
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):       # builtins, C functions
        return dict(arguments)
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(arguments)
    return {name: value for name, value in arguments.items() if name in parameters}
