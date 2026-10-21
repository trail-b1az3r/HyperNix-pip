"""t1api.mcpclient — T1 as an MCP *client*, for servers like Hugging Face's.

:mod:`hypernix.t1api.mcp` makes this server an MCP server. This is the
other direction: the server connects to remote MCP servers — the Hugging
Face Hub's at ``https://huggingface.co/mcp``, or any other speaking the
streamable HTTP transport — and offers their tools

* to assistants that connect to T1's own ``/mcp`` (namespaced
  ``<server>.<tool>``, as plugins are), and
* to HyperLink's models, in the same tool loop as noodle's and T1's own
  tools (``<server>__<tool>``, since OpenAI function names allow no dot).

The transport
-------------
Streamable HTTP, revision 2025-03-26 and later: every message is a POST
that accepts ``application/json`` and ``text/event-stream``; a reply is
either one JSON body or an event stream carrying it; the
``Mcp-Session-Id`` the server hands out on ``initialize`` goes back on
every later request, and a 404 for it means the session expired, so the
client initialises again once. Server-to-client requests on the stream
(sampling, elicitation) are answered with "not supported" rather than
left hanging.

Credentials
-----------
A server's token is read from an environment variable it names
(``token_env``, e.g. ``HF_TOKEN``) or stored in the servers file, which
is written owner-only. Neither is ever returned by an endpoint: the
listing says only whether a token is set. The token is the operator's,
so a server's tools are offered to every authenticated caller unless it
is added with ``admin_only``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..security.safeurl import urlopen as safe_urlopen
from .mcp import PROTOCOL_VERSION, SUPPORTED_VERSIONS, MCPResource, MCPTool, ToolError

logger = logging.getLogger(__name__)

__all__ = ["HUGGING_FACE", "MCPClient", "MCPHub", "MCPRemoteError", "RemoteServer"]

#: The Hugging Face Hub's MCP server, as a ready entry (the token comes from HF_TOKEN).
HUGGING_FACE = {"name": "hf", "url": "https://huggingface.co/mcp", "token_env": "HF_TOKEN",
                "description": "Hugging Face Hub: models, datasets, Spaces, papers and docs"}

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,23}$")
#: How long a server's tool list is trusted before it is asked again.
TOOLS_TTL = 300.0
#: A server that just failed is not asked again for this long.
RETRY_AFTER = 60.0
MAX_RESULT_CHARS = 60_000


class MCPRemoteError(RuntimeError):
    """A remote MCP server could not be reached, or refused."""


@dataclass
class RemoteServer:
    name: str
    url: str
    token: str = ""
    token_env: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    admin_only: bool = False
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(f"a server name is lower-case letters, digits, - and _ (1-24), not {self.name!r}")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(f"{self.url} is not an http(s) URL")
        if self.url.startswith("http://") and not _local(self.url):
            raise ValueError("a remote MCP server outside this machine and the tailnet must use https")

    @property
    def bearer(self) -> str:
        return self.token or (os.environ.get(self.token_env, "") if self.token_env else "")

    def public(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("token")
        out["headers"] = sorted(self.headers)
        out["token_set"] = bool(self.bearer)
        return out


def _local(url: str) -> bool:
    from urllib.parse import urlsplit

    from .config import _is_local_or_tailscale

    return _is_local_or_tailscale(urlsplit(url).hostname or "")


class MCPClient:
    """One remote server, over streamable HTTP."""

    def __init__(self, server: RemoteServer, *, client_version: str = "") -> None:
        self.server = server
        self.client_version = client_version
        self.session_id = ""
        self.protocol = ""
        self.server_info: dict[str, Any] = {}
        self.instructions = ""
        self._ids = 0
        self._lock = threading.Lock()
        self._listing = False
        #: Listing tools happens while a person waits for a reply; a
        #: server that has not answered in this long is left out.
        self.listing_timeout = 15.0

    # -- transport ---------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "User-Agent": f"hypernix-t1/{self.client_version or 'dev'}", **self.server.headers}
        if self.server.bearer:
            headers["Authorization"] = f"Bearer {self.server.bearer}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol:
            headers["MCP-Protocol-Version"] = self.protocol
        return headers

    def _post(self, message: dict[str, Any]) -> dict[str, Any] | None:
        data = json.dumps(message).encode()
        request = urllib.request.Request(self.server.url, data=data, headers=self._headers(), method="POST")
        timeout = min(self.server.timeout, self.listing_timeout) if self._listing else self.server.timeout
        try:
            response = safe_urlopen(request, timeout=timeout)  # nosec B310 - an operator-configured server
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace").strip()
            if exc.code == 404 and self.session_id:
                raise _Expired() from exc
            if exc.code in (401, 403):
                raise MCPRemoteError(f"{self.server.name} refused the credentials ({exc.code}); "
                                     f"{'set ' + self.server.token_env if self.server.token_env else 'give it a token'}"
                                     f"{': ' + detail[:200] if detail else ''}") from exc
            raise MCPRemoteError(f"{self.server.name} answered {exc.code}: {detail[:300]}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise MCPRemoteError(f"{self.server.name} ({self.server.url}) is unreachable: "
                                 f"{getattr(exc, 'reason', exc)}") from exc
        with response:
            session = response.headers.get("Mcp-Session-Id")
            if session:
                self.session_id = session
            if response.status == 202 or "id" not in message:
                return None
            kind = (response.headers.get("Content-Type") or "").split(";")[0].strip()
            if kind == "text/event-stream":
                return self._from_stream(response, message["id"])
            body = response.read()
        if not body:
            raise MCPRemoteError(f"{self.server.name} sent an empty reply to {message.get('method')}")
        reply = json.loads(body)
        if isinstance(reply, list):
            reply = next((r for r in reply if r.get("id") == message["id"]), {})
        return reply

    def _from_stream(self, response: Any, wanted: Any) -> dict[str, Any]:
        data: list[str] = []
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
                continue
            if line or not data:
                continue
            payload = json.loads("\n".join(data))
            data = []
            for item in payload if isinstance(payload, list) else [payload]:
                if item.get("id") == wanted and ("result" in item or "error" in item):
                    return item
                if "method" in item and "id" in item:      # the server asking us something
                    self._post({"jsonrpc": "2.0", "id": item["id"],
                                "error": {"code": -32601, "message": "this client supports no requests"}})
        raise MCPRemoteError(f"{self.server.name} closed the stream without answering")

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        reply = None
        for attempt in (1, 2):
            with self._lock:
                if not self.protocol:
                    self._initialize()
                self._ids += 1
                message = {"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params or {}}
                try:
                    reply = self._post(message)
                    break
                except _Expired:
                    # The server forgot the session: start a new one, once.
                    self.session_id, self.protocol = "", ""
                    if attempt == 2:
                        raise MCPRemoteError(f"{self.server.name} keeps expiring the session") from None
        if reply is None:
            raise MCPRemoteError(f"{self.server.name} did not answer {method}")
        if "error" in reply:
            error = reply["error"] or {}
            raise MCPRemoteError(f"{self.server.name}: {method} failed: {error.get('message', error)}")
        return reply.get("result") or {}

    def _initialize(self) -> None:
        self._ids += 1
        reply = self._post({"jsonrpc": "2.0", "id": self._ids, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "hypernix-t1", "version": self.client_version or "dev"}}})
        if not reply or "error" in reply:
            raise MCPRemoteError(f"{self.server.name} refused to initialise: "
                                 f"{(reply or {}).get('error', {}).get('message', 'no reply')}")
        result = reply.get("result") or {}
        self.protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        if self.protocol not in SUPPORTED_VERSIONS:
            logger.warning("t1api.mcpclient: %s speaks %s, which this client does not know",
                           self.server.name, self.protocol)
        self.server_info = result.get("serverInfo") or {}
        self.instructions = str(result.get("instructions") or "")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # -- methods -----------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        tools, cursor = [], None
        self._listing = True
        try:
            for _ in range(20):                   # pages; a server that never stops paging is cut off
                result = self._request("tools/list", {"cursor": cursor} if cursor else {})
                tools.extend(result.get("tools") or [])
                cursor = result.get("nextCursor")
                if not cursor:
                    break
        finally:
            self._listing = False
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": arguments or {}})


class _Expired(Exception):
    pass


def result_text(result: dict[str, Any]) -> str:
    """A ``tools/call`` result as text a model reads."""
    parts = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(str(resource.get("text") or f"[resource {resource.get('uri', '')}]"))
        elif kind == "resource_link":
            parts.append(f"[{block.get('name') or 'link'}] {block.get('uri', '')}")
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} {block.get('mimeType', '')}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], indent=1))
    text = "\n".join(p for p in parts if p) or "(no output)"
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n[... {len(text) - MAX_RESULT_CHARS} more characters cut]"
    return ("Error: " + text) if result.get("isError") else text


def _function_name(server: str, tool: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool}")[:64]


class MCPHub:
    """Every remote server this T1 server is configured with."""

    def __init__(self, path: str | Path | None = None, *, extra: list[dict[str, Any]] | None = None,
                 client_version: str = "") -> None:
        self.path = Path(path) if path else None
        self.client_version = client_version
        self._servers: dict[str, RemoteServer] = {}
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._errors: dict[str, str] = {}
        self._failed_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._fixed: set[str] = set()
        for row in self._read():
            self._servers[row.name] = row
        for item in extra or []:
            server = RemoteServer(**{k: v for k, v in item.items() if k in RemoteServer.__dataclass_fields__})
            self._servers[server.name] = server
            self._fixed.add(server.name)

    def _read(self) -> list[RemoteServer]:
        if not self.path or not self.path.is_file():
            return []
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            return [RemoteServer(**{k: v for k, v in r.items() if k in RemoteServer.__dataclass_fields__})
                    for r in rows]
        except (ValueError, TypeError) as exc:
            logger.error("t1api.mcpclient: %s is not readable: %s", self.path, exc)
            return []

    def _write(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(s) for name, s in sorted(self._servers.items()) if name not in self._fixed]
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        os.replace(tmp, self.path)

    # -- configuration -------------------------------------------------

    @property
    def servers(self) -> list[RemoteServer]:
        return [self._servers[n] for n in sorted(self._servers)]

    def add(self, server: RemoteServer) -> RemoteServer:
        with self._lock:
            if server.name in self._fixed:
                raise ValueError(f"{server.name} comes from T1_MCP_REMOTE; change it there")
            self._servers[server.name] = server
            self._clients.pop(server.name, None)
            self._tools.pop(server.name, None)
            self._write()
        return server

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._fixed:
                raise ValueError(f"{name} comes from T1_MCP_REMOTE; change it there")
            gone = self._servers.pop(name, None) is not None
            self._clients.pop(name, None)
            self._tools.pop(name, None)
            if gone:
                self._write()
            return gone

    def client(self, name: str) -> MCPClient:
        server = self._servers.get(name)
        if server is None:
            raise MCPRemoteError(f"no remote MCP server called {name!r}")
        if name not in self._clients:
            self._clients[name] = MCPClient(server, client_version=self.client_version)
        return self._clients[name]

    # -- tools ---------------------------------------------------------

    def tools(self, name: str, *, refresh: bool = False) -> list[dict[str, Any]]:
        cached = self._tools.get(name)
        if cached and not refresh and time.monotonic() - cached[0] < TOOLS_TTL:
            return cached[1]
        if not refresh and time.monotonic() - self._failed_at.get(name, -1e9) < RETRY_AFTER:
            if cached:
                return cached[1]
            raise MCPRemoteError(self._errors.get(name) or f"{name} failed a moment ago")
        try:
            listed = self.client(name).list_tools()
        except MCPRemoteError as exc:
            self._errors[name] = str(exc)
            self._failed_at[name] = time.monotonic()
            if cached:
                return cached[1]
            raise
        self._errors.pop(name, None)
        self._failed_at.pop(name, None)
        self._tools[name] = (time.monotonic(), listed)
        return listed

    def _usable(self, is_admin: bool) -> list[RemoteServer]:
        return [s for s in self.servers if s.enabled and (is_admin or not s.admin_only)]

    def openai_tools(self, *, is_admin: bool = False) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
        """``(tools, names)``: the remote tools as OpenAI functions, and
        which ``(server, tool)`` each function name stands for. A server
        that cannot be reached is left out, not fatal."""
        offered, names = [], {}
        for server in self._usable(is_admin):
            try:
                listed = self.tools(server.name)
            except MCPRemoteError as exc:
                logger.info("t1api.mcpclient: %s is left out: %s", server.name, exc)
                continue
            for tool in listed:
                function = _function_name(server.name, str(tool.get("name", "")))
                names[function] = (server.name, str(tool.get("name", "")))
                offered.append({"type": "function", "function": {
                    "name": function,
                    "description": f"[{server.name}] {tool.get('description', '')}".strip()[:1024],
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}}}})
        return offered, names

    def call(self, server: str, tool: str, arguments: dict[str, Any] | None = None) -> tuple[bool, str]:
        """``(ok, text)`` — a failure is text the model can read, never a raise."""
        try:
            result = self.client(server).call_tool(tool, arguments)
        except MCPRemoteError as exc:
            return False, str(exc)
        return not result.get("isError"), result_text(result)

    def status(self) -> list[dict[str, Any]]:
        rows = []
        for server in self.servers:
            row = server.public()
            cached = self._tools.get(server.name)
            row["tools"] = [t.get("name") for t in cached[1]] if cached else None
            row["error"] = self._errors.get(server.name, "")
            row["fixed"] = server.name in self._fixed
            rows.append(row)
        return rows

    # -- as T1's own MCP server's plugins ------------------------------

    def plugins(self, *, is_admin: bool = False) -> list[Any]:
        return [_RemotePlugin(self, s) for s in self._usable(is_admin)]


class _RemotePlugin:
    """A remote server's tools as a plugin of T1's own MCP server."""

    def __init__(self, hub: MCPHub, server: RemoteServer) -> None:
        self.hub = hub
        self.server = server
        self.name = server.name

    def tools(self) -> list[MCPTool]:
        try:
            listed = self.hub.tools(self.server.name)
        except MCPRemoteError:
            return []
        out = []
        for tool in listed:
            name = str(tool.get("name", ""))

            def handler(_name=name, **arguments):
                ok, text = self.hub.call(self.server.name, _name, arguments)
                if not ok:
                    raise ToolError(text)
                return text

            annotations = tool.get("annotations") or {}
            out.append(MCPTool(name=name, description=str(tool.get("description", "")),
                               input_schema=tool.get("inputSchema") or {"type": "object"},
                               handler=handler, scope="admin" if self.server.admin_only else "",
                               mutating=not annotations.get("readOnlyHint", False)))
        return out

    def resources(self) -> list[MCPResource]:
        return []
