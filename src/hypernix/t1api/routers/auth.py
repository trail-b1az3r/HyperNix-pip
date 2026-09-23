from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..auth import AuthContext, T1AuthService
from ..authhistory import AuthOp
from ..deps import get_auth_context, get_auth_service, get_request_id, require_admin
from ..schemas import (
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


__all__ = ["router"]
