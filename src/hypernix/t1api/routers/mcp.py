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
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

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


def _server_for(request: Request, config: T1APIConfig) -> MCPServer:
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
    for plugin in getattr(request.app.state, "t1_mcp_plugins", []) or []:
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

    server = _server_for(request, config)
    reply = server.handle_text(
        raw.decode("utf-8", errors="replace"),
        scopes={str(scope) for scope in principal.scopes},
        is_admin=principal.is_admin,
    )

    # A body of nothing but notifications gets 202 and no body, which is
    # what the spec says and what clients expect. Returning `null` here
    # makes a client treat it as the answer to its next request.
    if reply is None:
        return Response(status_code=202)
    return JSONResponse(reply)


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
    server = _server_for(request, config)
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
