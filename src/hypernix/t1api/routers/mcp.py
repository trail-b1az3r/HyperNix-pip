"""``/mcp`` — the T1 API described so an assistant can read it itself.

The transport, and deliberately almost nothing else. Everything that
can be got wrong about the protocol is in :mod:`hypernix.t1api.mcp`,
which has no FastAPI in it and is tested against dicts; this turns an
HTTP request into one call to that.

Off unless ``T1_MCP_ENABLED``. It is a second, differently-shaped way
in to capabilities that already have one, and a surface nobody asked
for is a surface nobody is watching.

Authenticated like everything else
----------------------------------
The same key, the same scopes. MCP has its own ideas about
authorisation and none of them are better than the ones this server
already enforces, so the caller's scopes are handed to dispatch and it
refuses there — before a handler runs, and by *hiding* tools the caller
could not call rather than offering them and refusing. A model that is
shown a tool it cannot use will try it, be refused, and try again; the
refusal is not information it can act on.

Streamable HTTP, as real clients expect it (0.72.8a3)
-----------------------------------------------------
The version a client asks for is answered in when it is one this server
speaks (see ``SUPPORTED_VERSIONS``); ``initialize`` hands out an
``Mcp-Session-Id`` (the server keeps no state per session, so any id it
is sent back is fine); a ``GET`` asking for an event stream gets 405,
which tells a client there is no server-initiated stream rather than
leaving it parsing the browser description as one; ``DELETE`` (ending a
session) is accepted.

Remote servers
--------------
``/mcp/remote`` is this server as a *client* of other MCP servers — the
Hugging Face Hub's, or any other — whose tools then appear here as
``<server>.<tool>`` and in HyperLink's tool loop. See
:mod:`hypernix.t1api.mcpclient`. Listing them needs a key; adding,
removing and calling directly need admin.
"""
from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..auth import AuthContext
from ..config import T1APIConfig
from ..deps import get_auth_context, get_config, get_registry, get_request_id, get_runner
from ..errors import T1APIError, T1ErrorCode
from ..mcp import MCPServer
from ..mcptools import ServerContext, build_server

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])

#: Bodies larger than this are refused before parsing. A JSON-RPC call
#: is small; anything this size is a mistake or an attempt to make the
#: parser the expensive part.
MAX_BODY_BYTES = 1_000_000


def _require_enabled(config: T1APIConfig) -> None:
    if not config.mcp_enabled:
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            "MCP is disabled on this server. Set T1_MCP_ENABLED=1 to offer "
            "this server's capabilities over the Model Context Protocol.",
            http_status=501,
        )


def _server_for(request: Request, config: T1APIConfig, *, is_admin: bool = False) -> MCPServer:
    """Built per request, from the app's own registry and runner.

    Per request rather than cached on the app: the registry is reloaded
    when the file on disk changes, and a cached MCP server would go on
    describing the models that existed at startup. Building it is a few
    dataclasses.
    """
    from ... import __version__

    try:
        runner = get_runner(request)
    except Exception:  # noqa: BLE001 - a server without one is not an error
        runner = None

    built = build_server(
        ServerContext(
            config=config,
            registry=get_registry(request),
            runner=runner,
            version=__version__,
        )
    )
    # Plugins the operator attached at startup. Registered objects, never
    # a module named by the request -- see hypernix.t1api.mcp.
    hub = getattr(request.app.state, "t1_mcp_hub", None)
    remote = hub.plugins(is_admin=is_admin) if hub is not None else []
    for plugin in [*(getattr(request.app.state, "t1_mcp_plugins", []) or []), *remote]:
        try:
            built.add_plugin(plugin)
        except Exception:  # noqa: BLE001
            logger.exception(
                "t1api.mcp: plugin %s failed to register",
                getattr(plugin, "name", "?"),
            )
    return built


@router.post("")
async def mcp_endpoint(
    request: Request,
    principal: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
):
    """One JSON-RPC message (or a batch) in, one response out."""
    _require_enabled(config)

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"an MCP message over {MAX_BODY_BYTES} bytes is a mistake; "
            f"this one is {len(raw)}",
            http_status=413,
        )

    text = raw.decode("utf-8", errors="replace")
    server = _server_for(request, config, is_admin=principal.is_admin)
    reply = server.handle_text(
        text,
        scopes={str(scope) for scope in principal.scopes},
        is_admin=principal.is_admin,
    )

    # A body of nothing but notifications gets 202 and no body, which is
    # what the spec says and what clients expect. Returning `null` here
    # makes a client treat it as the answer to its next request.
    if reply is None:
        return Response(status_code=202)
    headers = {}
    if '"initialize"' in text:
        headers["Mcp-Session-Id"] = request.headers.get("mcp-session-id") or secrets.token_hex(16)
    return JSONResponse(reply, headers=headers)


@router.delete("")
async def mcp_end_session(principal: AuthContext = Depends(get_auth_context),
                          config: T1APIConfig = Depends(get_config)):
    """A client ending its session. There is nothing kept to end."""
    _require_enabled(config)
    return Response(status_code=204)


@router.get("")
async def mcp_description(
    principal: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request: Request = None,
    request_id: str = Depends(get_request_id),
):
    """What this server offers, for a human looking at it in a browser.

    Not part of MCP — the protocol is the POST. This exists because the
    first thing anybody does with an MCP endpoint is open it, and a 405
    tells them nothing about whether they have the URL right.
    """
    _require_enabled(config)
    accept = (request.headers.get("accept") or "").lower()
    if "text/event-stream" in accept and "text/html" not in accept:
        # A client opening the server-to-client stream. There is none,
        # and the transport says so with 405.
        return Response(status_code=405, headers={"Allow": "POST, DELETE"})
    server = _server_for(request, config, is_admin=principal.is_admin)
    scopes = {str(scope) for scope in principal.scopes}
    listing = server.handle(
        {"jsonrpc": "2.0", "id": "describe", "method": "tools/list"},
        scopes=scopes,
        is_admin=principal.is_admin,
    )
    return {
        "protocol": "mcp",
        "transport": "POST this URL with JSON-RPC 2.0",
        "server": server.server_name,
        "version": server.server_version,
        "plugins": server.plugins,
        "tools": listing["result"]["tools"],
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# Remote servers: this server as an MCP client
# ---------------------------------------------------------------------------

class RemoteServerBody(BaseModel):
    name: str = Field(..., description="lower-case, 1-24 characters; the tools' prefix")
    url: str = Field(..., description="the server's streamable-HTTP endpoint, e.g. https://huggingface.co/mcp")
    token_env: str = Field("", description="an environment variable holding the bearer token, e.g. HF_TOKEN")
    token: str = Field("", description="the bearer token itself; stored owner-only, never returned")
    headers: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    admin_only: bool = False
    enabled: bool = True


class RemoteCallBody(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def _hub(request: Request, config: T1APIConfig):
    if not config.mcp_remote_enabled:
        raise T1APIError(T1ErrorCode.NOT_SUPPORTED,
                         "Remote MCP servers are off here (T1_MCP_REMOTE_ENABLED=0).", http_status=501)
    return request.app.state.t1_mcp_hub


def _admin(principal: AuthContext) -> None:
    if not principal.is_admin:
        raise T1APIError(T1ErrorCode.AUTH_ADMIN_REQUIRED,
                         "Changing or calling remote MCP servers directly needs an admin key.", http_status=403)


@router.get("/remote")
async def remote_list(request: Request, principal: AuthContext = Depends(get_auth_context),
                      config: T1APIConfig = Depends(get_config)):
    """The remote servers, whether each has a token, and the tools last seen."""
    hub = _hub(request, config)
    rows = [r for r in hub.status() if principal.is_admin or not r["admin_only"]]
    return {"servers": rows, "presets": [{"name": "hf", "url": "https://huggingface.co/mcp",
                                          "token_env": "HF_TOKEN"}]}


@router.post("/remote")
async def remote_add(body: RemoteServerBody, request: Request,
                     principal: AuthContext = Depends(get_auth_context),
                     config: T1APIConfig = Depends(get_config)):
    """Add (or replace) a remote server, and list its tools to prove it answers."""
    import anyio

    from ..mcpclient import MCPRemoteError, RemoteServer

    _admin(principal)
    hub = _hub(request, config)
    try:
        server = hub.add(RemoteServer(**body.model_dump()))
    except ValueError as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc), http_status=400) from exc
    try:
        tools = await anyio.to_thread.run_sync(lambda: hub.tools(server.name, refresh=True))
        reached, error = True, ""
    except MCPRemoteError as exc:
        tools, reached, error = [], False, str(exc)
    return {"server": server.public(), "reached": reached, "error": error,
            "tools": [t.get("name") for t in tools]}


@router.delete("/remote/{name}")
async def remote_remove(name: str, request: Request, principal: AuthContext = Depends(get_auth_context),
                        config: T1APIConfig = Depends(get_config)):
    _admin(principal)
    try:
        removed = _hub(request, config).remove(name)
    except ValueError as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc), http_status=409) from exc
    if not removed:
        raise T1APIError(T1ErrorCode.NOT_FOUND, f"no remote MCP server called {name!r}", http_status=404)
    return {"removed": name}


@router.get("/remote/{name}/tools")
async def remote_tools(name: str, request: Request, refresh: bool = False,
                       principal: AuthContext = Depends(get_auth_context),
                       config: T1APIConfig = Depends(get_config)):
    """A remote server's tools, as it describes them."""
    import anyio

    from ..mcpclient import MCPRemoteError

    hub = _hub(request, config)
    server = next((s for s in hub.servers if s.name == name), None)
    if server is None or (server.admin_only and not principal.is_admin):
        raise T1APIError(T1ErrorCode.NOT_FOUND, f"no remote MCP server called {name!r}", http_status=404)
    try:
        tools = await anyio.to_thread.run_sync(lambda: hub.tools(name, refresh=refresh))
    except MCPRemoteError as exc:
        raise T1APIError(T1ErrorCode.TRANSPORT_FAILED, str(exc), http_status=502) from exc
    return {"server": name, "tools": tools}


@router.post("/remote/{name}/call")
async def remote_call(name: str, body: RemoteCallBody, request: Request,
                      principal: AuthContext = Depends(get_auth_context),
                      config: T1APIConfig = Depends(get_config)):
    """Call one remote tool directly (admin): for checking a server works."""
    import anyio

    _admin(principal)
    hub = _hub(request, config)
    ok, text = await anyio.to_thread.run_sync(lambda: hub.call(name, body.tool, body.arguments))
    return {"server": name, "tool": body.tool, "ok": ok, "content": text}
