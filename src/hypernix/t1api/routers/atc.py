"""``/atc/*`` — API traffic control (0.73.0 pre-alpha 1).

* ``POST /atc/handshake`` -- start an end-to-end encrypted session
  (``hypernix-atc/1``). No credential: the key goes *inside* the tunnel.
* ``POST /atc/tunnel`` -- served by the ATC layer itself, not this
  router; the route here only answers when that layer is off.
* ``GET /atc/status`` -- the settings, the jail, traffic and route
  statistics. Admin.
* ``POST /atc/jail`` and ``POST /atc/release`` -- jail or free an
  address by hand. Admin.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..auth import AuthContext
from ..deps import get_auth_context, require_admin
from ..errors import T1APIError, T1ErrorCode

router = APIRouter(prefix="/atc", tags=["atc"])


def _admin(principal: AuthContext = Depends(get_auth_context)) -> AuthContext:
    return require_admin(principal)


def _atc(request: Request) -> Any:
    atc = getattr(request.app.state, "t1_atc", None)
    if atc is None:
        raise T1APIError(T1ErrorCode.NOT_SUPPORTED,
                         "ATC is off on this server (T1_ATC_ENABLED=0).",
                         http_status=501, details={"reason": "atc_off"})
    return atc


class HandshakeRequest(BaseModel):
    public_key: str = Field(..., min_length=40, max_length=64,
                            description="The client's X25519 public key, 32 bytes, base64url.")


class ClientRequest(BaseModel):
    client: str = Field(..., min_length=1, max_length=64, description="The IP address.")
    reason: str = Field(default="jailed by an administrator", max_length=200)


@router.post("/handshake")
def handshake(body: HandshakeRequest, request: Request) -> dict[str, Any]:
    """Start an encrypted session. The answer is signed by the server's
    identity key (the T2C key); check ``server_fingerprint`` against the
    one you pinned before trusting it."""
    from ...atc import e2e

    atc = _atc(request)
    if not atc.settings.e2e or atc.sessions is None:
        raise T1APIError(T1ErrorCode.NOT_SUPPORTED, "End-to-end encryption is off (T1_ATC_E2E=0).",
                         http_status=501, details={"reason": "atc_e2e_off"})
    try:
        return atc.sessions.handshake(body.public_key)
    except e2e.E2EError as exc:
        status = 501 if exc.reason == "unavailable" else 400
        raise T1APIError(T1ErrorCode.NOT_SUPPORTED if status == 501 else T1ErrorCode.VALIDATION_ERROR,
                         str(exc), http_status=status, details={"reason": exc.reason}) from exc


@router.post("/tunnel", include_in_schema=False)
def tunnel_off(request: Request) -> None:
    _atc(request)
    raise T1APIError(T1ErrorCode.NOT_SUPPORTED, "End-to-end encryption is off (T1_ATC_E2E=0).",
                     http_status=501, details={"reason": "atc_e2e_off"})


@router.get("/status")
def status(request: Request, _: AuthContext = Depends(_admin)) -> dict[str, Any]:
    atc = _atc(request)
    return atc.snapshot()


@router.post("/jail")
def jail(body: ClientRequest, request: Request, _: AuthContext = Depends(_admin)) -> dict[str, Any]:
    atc = _atc(request)
    if atc.shield.exempt(body.client):
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR,
                         "That address is exempt from the jail (this machine, unless "
                         "T1_ATC_JAIL_LOOPBACK=1).", http_status=400,
                         details={"reason": "atc_exempt"})
    seconds = atc.shield.jail(body.client, body.reason)
    return {"client": body.client, "jailed_seconds": round(seconds, 1), "reason": body.reason}


@router.post("/release")
def release(body: ClientRequest, request: Request, _: AuthContext = Depends(_admin)) -> dict[str, Any]:
    atc = _atc(request)
    if not atc.shield.release(body.client):
        raise T1APIError(T1ErrorCode.NOT_FOUND, f"{body.client} is not in the jail.",
                         http_status=404, details={"reason": "atc_not_jailed"})
    return {"client": body.client, "released": True}
