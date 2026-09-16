"""``/noodle/*`` — the autonomous executor, over the API.

Noodle writes files, edits them, runs commands and packs archives. That
is the whole point of it and it is also, said plainly, arbitrary code
execution as a service — so this router is mostly about the boundary
rather than the feature.

Four things hold it in
----------------------
**Off unless the operator turned it on.** ``T1_NOODLE_ENABLED`` is unset
by default and every route 404s until it is. Not 403: a server that has
not enabled this should not advertise that it could.

**A workspace per owner, under a configured root.** Every path a tool
touches goes through ``ToolContext.resolve``, which resolves symlinks
before checking containment — so a link planted by an earlier call
cannot step outside. Two owners never share a directory.

**Execution is a second switch.** Creating and editing files is one
capability; running what was written is another, and
``T1_NOODLE_ALLOW_EXECUTE`` is separate for that reason. The default
lets an agent write a script and not run it, which is the useful half of
the feature at a fraction of the risk.

**Admin or partial admin.** Not an ordinary read token, and never a
plain keyless caller on a public origin.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from ...interfaces.noodle.tools import (
    TOOLS,
    ToolContext,
    run_tool,
    tool_schemas,
)
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_config,
    get_request_id,
    require_hyperlink_operator,
)
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    NoodleRunRequest,
    NoodleToolsResponse,
    NoodleWorkspaceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/noodle", tags=["noodle"])

#: Tool failures that are the *caller's* doing, and so a 4xx.
#:
#: The distinction that matters: a command exiting non-zero is a result,
#: not a request error — the tool did exactly what was asked and the
#: answer is "it failed". Returning 400 for that would make a working
#: `run_fish` indistinguishable from a malformed request, and a client
#: could not tell "your path is outside the workspace" from "your script
#: has a bug".
CALLER_ERRORS = frozenset({
    "bad_path", "outside_workspace", "not_found", "bad_args", "bad_command",
    "unknown_tool", "too_large", "recursive_archive", "bad_archive",
    "write_budget", "execute_disabled", "memory_disabled", "fish_missing",
    "exists", "unknown_type",
})


def _require_enabled(config: T1APIConfig) -> None:
    if not getattr(config, "noodle_enabled", False):
        # 404 rather than 403. A server that has not enabled this has no
        # reason to tell an unauthenticated stranger that it could.
        raise T1APIError(
            T1ErrorCode.NOT_FOUND,
            "Noodle is not enabled on this server (set T1_NOODLE_ENABLED=1).",
            http_status=404,
        )


def _safe_slug(owner: str) -> str:
    """A directory name from an owner id.

    Not the owner id itself: it can contain a slash — a device token's
    owner is a key id, and key ids have prefixes — and a slash in a
    directory name is a directory somewhere else.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in owner)
    return cleaned.strip("-")[:64] or "anonymous"


def _context(config: T1APIConfig, principal: HyperLinkPrincipal) -> ToolContext:
    root = Path(
        getattr(config, "noodle_workspace_dir", "")
        or (Path.home() / ".hypernix" / "noodle")
    ) / _safe_slug(principal.owner)
    return ToolContext(
        root=root,
        allow_execute=bool(getattr(config, "noodle_allow_execute", False)),
        memory_enabled=bool(getattr(config, "noodle_memory_enabled", False)),
        allow_web_search=bool(getattr(config, "noodle_allow_web_search", True)),
        execute_timeout=float(getattr(config, "noodle_execute_timeout", 60.0)),
    )


@router.get("/tools", response_model=NoodleToolsResponse)
def list_tools(
    principal: HyperLinkPrincipal = Depends(require_hyperlink_operator),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> NoodleToolsResponse:
    """What this server will let noodle do.

    The `enabled` flags matter as much as the list: a client that shows
    "run a command" on a server with execution off produces a button that
    always fails, and the honest version greys it out.
    """
    _require_enabled(config)
    context = _context(config, principal)
    return NoodleToolsResponse(
        tools=tool_schemas(),
        names=sorted(TOOLS),
        execute_enabled=context.allow_execute,
        web_search_enabled=context.allow_web_search,
        memory_enabled=context.memory_enabled,
        workspace=str(context.root),
        request_id=request_id,
    )


@router.post("/run", response_model=dict)
def run_one_tool(
    payload: NoodleRunRequest,
    principal: HyperLinkPrincipal = Depends(require_hyperlink_operator),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Run one tool and return what it produced.

    One call, one tool — this is not the agent loop. A client that wants
    the loop drives it from its own model, which is the arrangement that
    keeps the *decision* about what to run on the caller's side and only
    the doing on the server's.

    A `ToolError` becomes a 400 with the tool's own code rather than a
    500: "that path is outside the workspace" is a thing the caller did,
    and reporting it as a server fault sends them looking in the wrong
    place.
    """
    _require_enabled(config)
    if payload.tool not in TOOLS:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"No tool named {payload.tool!r}. This server has: {sorted(TOOLS)}",
        )
    context = _context(config, principal)
    # `run_tool` never raises -- it turns every failure into a result a
    # model can read and correct itself from, which is right for a tool
    # loop and wrong for HTTP. So the code is inspected here and the
    # caller's mistakes become 4xx.
    result = run_tool(context, payload.tool, payload.arguments or {})
    if not result.ok and result.code in CALLER_ERRORS:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            result.content,
            details={"tool": payload.tool, "code": result.code},
        )

    logger.info(
        "noodle: %s ran %s (ok=%s)", principal.label, payload.tool, result.ok
    )
    # `to_dict` rather than hand-built: noodle's own surfaces already use
    # that shape, and a second spelling of the same result is a client
    # that works against one and not the other.
    return {
        **result.to_dict(),
        "workspace": str(context.root),
        "request_id": request_id,
    }


@router.get("/workspace", response_model=NoodleWorkspaceResponse)
def show_workspace(
    principal: HyperLinkPrincipal = Depends(require_hyperlink_operator),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> NoodleWorkspaceResponse:
    """What is in this caller's workspace.

    Scoped to the caller's own directory, so one owner cannot enumerate
    another's. Sizes and names only — reading a file is `run` with
    `read_file`, which goes through the same containment check everything
    else does.
    """
    _require_enabled(config)
    context = _context(config, principal)
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(context.root.rglob("*")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total += size
        files.append({
            "path": str(path.relative_to(context.root)),
            "bytes": size,
            "modified_at": path.stat().st_mtime,
        })
    return NoodleWorkspaceResponse(
        workspace=str(context.root),
        files=files[:500],
        count=len(files),
        total_bytes=total,
        request_id=request_id,
    )
