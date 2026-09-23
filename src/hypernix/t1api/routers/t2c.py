"""``/auth/t2c/*`` — v2.1 (T2C) keys: the server's public key and devices (0.72.6).

A client turns a key it already holds into a T2C kit without the server
ever seeing a secret in the clear:

1. ``GET /auth/t2c/public-key`` — the server's RSA public key.
2. The client makes a 32-byte device secret, encrypts it for that key,
   and ``POST /auth/t2c/devices`` with the key it holds. The server
   checks the key, unwraps the secret and binds the device to the key.
3. The client seals its T2 key for the public key (the inner) and from
   then on sends the day's key the kit makes. ``waiter serv -E`` does all
   three.

``GET /auth/t2c/devices`` lists the caller's devices and
``DELETE /auth/t2c/devices/{id}`` revokes one — the answer to a kit that
was copied somewhere it should not be.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth import AuthContext, T1AuthService
from ..deps import get_auth_context, get_auth_service, get_request_id
from ..errors import T1APIError, T1ErrorCode

router = APIRouter(prefix="/auth/t2c", tags=["auth"])


class T2CDeviceRequest(BaseModel):
    key: str = Field(..., min_length=4, description="The T1 or T2 key the device is for.")
    wrapped_secret: str = Field(..., min_length=16,
                                description="The device secret, RSA-OAEP-SHA256 encrypted for "
                                            "GET /auth/t2c/public-key, base64.")
    label: str = Field(default="", max_length=80)
    access_level: int = Field(default=1, ge=1, le=9,
                              description="For a T1 key, the level of the T2 spelling the client "
                                          "will seal. A T2 key's own level is used instead.")


def _unavailable(exc: Exception) -> T1APIError:
    return T1APIError(
        T1ErrorCode.NOT_SUPPORTED,
        f"T2C keys are not available on this server: {exc}",
        http_status=501,
    )


@router.get("/public-key")
def public_key(
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """The key a client seals for. Public by design; no credential needed."""
    from ...security.rotorvault import SCHEME, RotorvaultError

    try:
        pem, fingerprint = svc.t2c.public_key_pem(), svc.t2c.fingerprint()
    except (RotorvaultError, ImportError) as exc:
        raise _unavailable(exc) from exc
    return {
        "scheme": SCHEME,
        "algorithm": "RSA-OAEP-SHA256",
        "public_key_pem": pem,
        "fingerprint": fingerprint,
        "grace_days": 1,
        "request_id": request_id,
    }


def _t1_equivalent(key: str) -> tuple[str, int | None]:
    from ...security.t2keys import T2KeyGenerator, looks_like_t2

    if looks_like_t2(key):
        parsed = T2KeyGenerator.parse(key)
        return T2KeyGenerator.to_t1(parsed), parsed.access_level
    return key, None


@router.post("/devices")
def register_device(
    body: T2CDeviceRequest,
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Register a device for the key in the body. The key itself, not a
    token or a T2C key: the device is bound to exactly that key."""
    from ...security.rotorvault import RotorvaultError
    from ...security.t2c import T2CError

    if body.key.startswith(("T1S.", "T2C_", "T2CK_")):
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "Register a device with the T1 or T2 key it is for, not a token or a T2C key.",
            http_status=400,
        )
    ctx = svc.validate_key(body.key)
    try:
        bound, level = _t1_equivalent(body.key)
    except ValueError as exc:
        raise T1APIError(T1ErrorCode.AUTH_INVALID_KEY, str(exc), http_status=401) from exc
    level = level if level is not None else body.access_level
    try:
        device_id = svc.t2c.register_wrapped(
            bound_key=bound, wrapped_secret=body.wrapped_secret,
            label=body.label, access_level=level,
        )
        fingerprint = svc.t2c.fingerprint()
    except T2CError as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc), http_status=400) from exc
    except (RotorvaultError, ImportError) as exc:
        raise _unavailable(exc) from exc
    return {
        "device_id": device_id,
        "key_id": ctx.key_id,
        "access_level": level,
        "server_fingerprint": fingerprint,
        "request_id": request_id,
    }


@router.get("/devices")
def list_devices(
    ctx: AuthContext = Depends(get_auth_context),
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    devices = svc.t2c.devices_for(ctx.key_meta.key) if ctx.key_meta.key else []
    return {"devices": devices, "count": len(devices), "request_id": request_id}


@router.delete("/devices/{device_id}")
def revoke_device(
    device_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Revoke one of the caller's devices; an administrator may revoke any."""
    bound = None if ctx.is_admin else ctx.key_meta.key
    if not svc.t2c.revoke(device_id, bound_key=bound):
        raise T1APIError(T1ErrorCode.NOT_FOUND, f"No device {device_id} for this key.",
                         http_status=404)
    return {"device_id": device_id, "revoked": True, "request_id": request_id}
