"""``/privacy/*`` and ``/server/info`` (0.72.6).

``/privacy/conceal`` turns conceal mode on and off for the calling key —
see :mod:`hypernix.t1api.privacy` for exactly what it does and does not
do. ``/server/info`` is what ``waiter serv -Y`` prints: the public face
of a server, with no credential needed.
"""
from __future__ import annotations

import socket
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..auth import AuthContext
from ..deps import get_auth_context, get_auth_service, get_config, get_request_id
from ..errors import T1APIError, T1ErrorCode
from ..privacy import CONCEAL_MIN_ACCESS_LEVEL, RETENTION_HOURS

router = APIRouter(tags=["privacy"])


def _conceal(request: Request):
    return request.app.state.t1_conceal


@router.get("/privacy/conceal")
def conceal_status(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    return {**_conceal(request).status(ctx.key_id), "request_id": request_id}


@router.post("/privacy/conceal")
def conceal_on(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Conceal the calling key. Needs access level 3 or more."""
    if not ctx.meets_access_level(CONCEAL_MIN_ACCESS_LEVEL):
        raise T1APIError(
            T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
            f"Conceal needs access level {CONCEAL_MIN_ACCESS_LEVEL} or more; this key is "
            f"level {ctx.t2_access_level}.",
            details={"required_level": CONCEAL_MIN_ACCESS_LEVEL, "level": ctx.t2_access_level},
            http_status=403,
        )
    status = _conceal(request).set(ctx.key_id, True)
    # Sweep now rather than at the next tick, so turning it on is seen
    # to do something the moment it returns.
    swept = request.app.state.t1_retention.sweep_key(ctx.key_id)
    return {**status, "swept": swept.to_dict(), "request_id": request_id}


@router.delete("/privacy/conceal")
def conceal_off(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Stop concealing. Nothing already deleted comes back."""
    return {**_conceal(request).set(ctx.key_id, False), "request_id": request_id}


@router.get("/server/info")
def server_info(
    request: Request,
    config=Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Who runs this server and what it offers. Public by design."""
    import hypernix

    from ..version import T1_VERSION

    try:
        t2c = bool(get_auth_service(request).accept_t2_keys)
    except Exception:  # noqa: BLE001
        t2c = False
    url = config.server_url or config.hyperlink_public_url or ""
    return {
        "name": config.server_name or socket.gethostname(),
        "description": config.server_description,
        "owner": config.server_owner,
        "url": url,
        "public": bool(url) or config.environment == "production",
        "environment": config.environment,
        "hypernix_version": getattr(hypernix, "__version__", "unknown"),
        "t1_version": T1_VERSION.to_dict(),
        "server_id": config.server_id,
        "host_id": config.host_id,
        "features": {
            "t2c_keys": t2c,
            "conceal": True,
            "conceal_min_access_level": CONCEAL_MIN_ACCESS_LEVEL,
            "retention_hours": RETENTION_HOURS,
            "accounts": bool(getattr(getattr(request.app.state, "t1_webauth", None), "enabled", False)),
            "hyperlink": bool(getattr(config, "hyperlink_enabled", True)),
        },
        "request_id": request_id,
    }
