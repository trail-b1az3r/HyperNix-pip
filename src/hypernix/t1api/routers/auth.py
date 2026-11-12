from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..auth import AuthContext, T1AuthService
from ..authhistory import AuthOp
from ..deps import get_auth_context, get_auth_service, get_request_id, require_admin
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    AccessLevelRequest,
    AccessLevelResponse,
    AdminRotateRequest,
    RotateResponse,
    TokenRequest,
    TokenResponse,
    ValidateKeyRequest,
    ValidateKeyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/t1/validate", response_model=ValidateKeyResponse)
def validate(
    body: ValidateKeyRequest,
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> ValidateKeyResponse:
    ctx = svc.validate_key(body.key)
    return ValidateKeyResponse(
        key_id=ctx.key_id,
        key_type=ctx.key_meta.key_type.value,
        scopes=sorted(s.value for s in ctx.scopes),
        active=ctx.key_meta.active,
        is_admin=ctx.is_admin,
        expires_at=ctx.key_meta.expires_at,
        request_id=request_id,
        key_family=ctx.t2_family or "T1",
        access_level=ctx.t2_access_level if ctx.t2_family else None,
    )


@router.post("/token", response_model=TokenResponse)
def issue_token(
    body: TokenRequest,
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> TokenResponse:
    tok = svc.issue_scoped_token(body.key, ttl_seconds=body.ttl_seconds, scopes=body.scopes)
    return TokenResponse(
        token=tok.token,
        key_id=tok.key_id,
        scopes=tok.scopes,
        expires_at=tok.expires_at,
        request_id=request_id,
    )


def _record_rotation(
    request: Request,
    *,
    actor: str,
    previous_key: str,
    new_meta: Any,
    previous_type: str = "",
) -> None:
    """Write a rotation into the auth history so it can be undone.

    Without this, ``POST /t1/auth/undo`` has nothing to work with: the
    history was only ever read, never written, so it was permanently
    empty and every undo answered "nothing to undo". The endpoint
    existed, the storage existed, and no operation ever put anything in
    it.

    Recorded against the *new* key ID, because that is the record the
    inverse acts on — restoring the previous material onto it puts the
    caller back where they were without changing any key ID, which is what
    keeps references to the key valid across an undo.

    Best-effort on purpose. The rotation has already happened and its new
    key is in the response; failing the request now would tell the caller
    their rotation failed when it did not, and lose the key with it.
    """
    history = getattr(request.app.state, "t1_auth_history", None)
    if history is None:
        return
    payload = {"previous_key": previous_key, "new_key": new_meta.key}
    if previous_type:
        payload["previous_type"] = previous_type
        payload["new_type"] = new_meta.key_type.value
    try:
        history.record(
            AuthOp.ROTATE,
            actor=actor,
            target_key_id=new_meta.key_id,
            payload=payload,
            summary=f"rotated {new_meta.key_id[:8]}…",
        )
    except Exception:  # noqa: BLE001
        logger.warning("t1api.auth: could not record the rotation", exc_info=True)


@router.post("/t1/rotate", response_model=RotateResponse)
def rotate_own_key(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> RotateResponse:
    """A caller may always rotate the key/token they authenticated with."""
    previous_key = ctx.key_meta.key
    new_meta = svc.rotate_own_key(ctx.key_id)
    _record_rotation(
        request, actor=ctx.key_id, previous_key=previous_key, new_meta=new_meta
    )
    return RotateResponse(
        key_id=new_meta.key_id,
        key=new_meta.key,
        key_type=new_meta.key_type.value,
        scopes=sorted(s.value for s in new_meta.scopes),
        rotated_from=new_meta.rotated_from,
        request_id=request_id,
    )


@router.post("/t1/admin/rotate", response_model=RotateResponse)
def admin_rotate(
    request: Request,
    body: AdminRotateRequest,
    ctx: AuthContext = Depends(get_auth_context),
    svc: T1AuthService = Depends(get_auth_service),
    request_id: str = Depends(get_request_id),
) -> RotateResponse:
    require_admin(ctx)
    keymaster = getattr(request.app.state, "t1_keymaster", None)
    target = keymaster.get(body.target_key_id) if keymaster is not None else None
    previous_key = target.key if target is not None else ""
    previous_type = target.key_type.value if target is not None else ""
    new_meta = svc.admin_rotate(
        requester=ctx, target_key_id=body.target_key_id, promote_to_admin=body.promote_to_admin
    )
    if previous_key:
        _record_rotation(
            request,
            actor=ctx.key_id,
            previous_key=previous_key,
            new_meta=new_meta,
            previous_type=previous_type if body.promote_to_admin else "",
        )
    return RotateResponse(
        key_id=new_meta.key_id,
        key=new_meta.key,
        key_type=new_meta.key_type.value,
        scopes=sorted(s.value for s in new_meta.scopes),
        rotated_from=new_meta.rotated_from,
        request_id=request_id,
    )


@router.post("/t1/admin/access-level", response_model=AccessLevelResponse)
def admin_set_access_level(
    request: Request,
    body: AccessLevelRequest,
    ctx: AuthContext = Depends(get_auth_context),
    request_id: str = Depends(get_request_id),
) -> AccessLevelResponse:
    """Grant a key an access level, 1-9 (``waiter serv -Pro KEY -K ADMIN -L N``).

    Admin only. The grant is the server's record (the key's
    ``access_level`` tag), and a T2 key's own suffix is held to it when it
    authenticates -- so the answer carries the key spelled at its new
    level, which is the spelling its holder now uses. Reversible with
    ``POST /t1/auth/undo``.
    """
    require_admin(ctx)
    keymaster = getattr(request.app.state, "t1_keymaster", None)
    if keymaster is None or not hasattr(keymaster, "set_access_level"):
        raise T1APIError(T1ErrorCode.NOT_SUPPORTED,
                         "This server's key store cannot change access levels.", http_status=501)
    t1_key, respell = _level_spelling(body.key.strip(), body.access_level)
    target = keymaster.get_by_key(t1_key)
    if target is None:
        raise T1APIError(T1ErrorCode.AUTH_INVALID_KEY,
                         "That key is not registered on this server.", http_status=404)
    raw_previous = str((target.tags or {}).get("access_level") or "")
    previous = int(raw_previous) if raw_previous.isdigit() else None
    keymaster.set_access_level(target.key_id, body.access_level)
    history = getattr(request.app.state, "t1_auth_history", None)
    if history is not None:
        try:
            history.record(
                AuthOp.LEVEL_CHANGE,
                actor=ctx.key_id,
                target_key_id=target.key_id,
                payload={"previous_level": previous, "new_level": body.access_level},
                summary=f"set access level of {target.key_id[:8]}… to {body.access_level}",
            )
        except Exception:  # noqa: BLE001 - the grant stands; only its undo entry is lost
            logger.warning("t1api.auth: could not record the level change", exc_info=True)
    logger.info("t1api.auth: %s set access level of %s to %s (was %s)",
                ctx.key_id[:8], target.key_id[:8], body.access_level, previous)
    return AccessLevelResponse(key_id=target.key_id, access_level=body.access_level,
                               previous_level=previous, key=respell, request_id=request_id)


def _level_spelling(key: str, level: int) -> tuple[str, str]:
    """``(the T1 key, the key spelled at *level*)`` for a key as its holder has it.

    A T2-family key keeps its family and password and changes only its
    last character, the level. A T1 key has no level, so it is given its
    T2 spelling. A T2C key's level is sealed on its device record and is
    refused here rather than half-changed.
    """
    from ...security.t2keys import T2KeyGenerator, T2Type

    if key.startswith("T1_"):
        try:
            return key, T2KeyGenerator.from_t1(key, access_level=level).raw
        except ValueError as exc:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc), http_status=400) from exc
    try:
        parsed = T2KeyGenerator.parse(key)
    except ValueError as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR,
                         f"Not a T1 or T2 key: {exc}", http_status=400) from exc
    if parsed.family is T2Type.T2C:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "A T2C key's level is sealed on its device record; issue a new T2C key at the "
            "level you want instead.", http_status=400)
    return T2KeyGenerator.to_t1(parsed), parsed.raw[:-1] + str(level)


__all__ = ["router"]
