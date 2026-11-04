"""``/t1/auth/undo`` and ``/t1/auth/redo``.

These live at ``/t1/auth/...`` because that is the path the release
specifies. The rest of this API's authentication surface is at
``/auth/...`` (with T1-specific operations under ``/auth/t1/...``), so
the same two operations are also registered there as aliases — one
implementation, two routes, because a client that has learned
``/auth/t1/rotate`` should not have to learn a differently-shaped path
for the operation that reverses it.

What undo actually does
-----------------------
It reverses the last recorded authentication change: a rotation, a
promotion, a scope change, a revocation, an SSPKID assignment, or a plan
change. The history is a stack (see :mod:`hypernix.t1api.authhistory`),
recording an inverse rather than a snapshot, and it refuses to record an
operation it could not actually reverse.

Both endpoints are admin-only and both are audited *before* the mutation
is attempted as well as after, because "who tried to undo the admin
promotion" is a question worth being able to answer even when the
attempt failed.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..audit import AuditCategory, AuditOutcome
from ..auth import AuthContext
from ..authhistory import AuthHistory, AuthOp
from ..deps import get_audit_log, get_auth_context, get_client_ip, get_request_id, require_admin
from ..errors import T1APIError, T1ErrorCode
from ..schemas import AuthHistoryResponse, AuthUndoResponse

logger = logging.getLogger(__name__)

#: The path the release specifies.
router = APIRouter(prefix="/t1/auth", tags=["auth"])
#: The same operations under the existing auth namespace.
alias_router = APIRouter(prefix="/auth/t1", tags=["auth"])


def get_auth_history(request: Request) -> AuthHistory:
    return request.app.state.t1_auth_history


def _applier(request: Request):
    """Return the function that actually reverses (or replays) an entry.

    Kept here rather than in :mod:`~hypernix.t1api.authhistory` because
    it is the only part that needs Keymaster: the history owns ordering
    and durability, this owns key semantics.
    """
    keymaster = getattr(request.app.state, "t1_keymaster", None)
    directory = getattr(request.app.state, "t1_key_directory", None)

    def apply(entry: Any, *, direction: str) -> None:
        """Put a key back the way it was, or forward again.

        Wrapped so that a key which no longer exists produces a refusal
        rather than a bare 500. It is a reachable state, not a programming
        error: the history outlives the keys it refers to, so undoing a
        rotation whose key has since been revoked is an ordinary thing for
        an operator to try, and "Internal Server Error" with no code and
        no body is the least useful possible answer to it.
        """
        try:
            _apply_inner(entry, direction=direction)
        except KeyError as exc:
            raise T1APIError(
                T1ErrorCode.NOT_FOUND,
                f"The key this {entry.op.value} applied to no longer exists, so it "
                "cannot be reversed. The history entry is left in place.",
                details={
                    "target_key_id": entry.target_key_id,
                    "op": entry.op.value,
                    "direction": direction,
                },
                http_status=409,
            ) from exc

    def _apply_inner(entry: Any, *, direction: str) -> None:
        payload = entry.payload
        # "undo" restores the previous value; "redo" re-applies the new
        # one. Choosing the field by direction is the whole of it — the
        # two paths are otherwise identical, and writing them separately
        # is how they drift.
        want_previous = direction == "undo"

        if entry.op is AuthOp.ROTATE:
            target = payload["previous_key"] if want_previous else payload["new_key"]
            if keymaster is None or not hasattr(keymaster, "restore_key"):
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server's Keymaster cannot restore key material, so a rotation "
                    "cannot be reversed here. The previous key is still recorded; restore it "
                    "manually with `gkey`.",
                    http_status=501,
                )
            keymaster.restore_key(entry.target_key_id, target)
            return

        if entry.op is AuthOp.PROMOTE:
            target = payload["previous_type"] if want_previous else payload["new_type"]
            if keymaster is None or not hasattr(keymaster, "set_key_type"):
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server's Keymaster cannot change a key's type in place.",
                    http_status=501,
                )
            keymaster.set_key_type(entry.target_key_id, target)
            return

        if entry.op is AuthOp.SCOPE_CHANGE:
            target = payload["previous_scopes"] if want_previous else payload["new_scopes"]
            if keymaster is None or not hasattr(keymaster, "set_scopes"):
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server's Keymaster cannot change a key's scopes in place.",
                    http_status=501,
                )
            keymaster.set_scopes(entry.target_key_id, target)
            return

        if entry.op is AuthOp.REVOKE:
            was_revoked = payload["previous_revoked"] if want_previous else True
            if keymaster is None or not hasattr(keymaster, "set_revoked"):
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server's Keymaster cannot un-revoke a key.",
                    http_status=501,
                )
            keymaster.set_revoked(entry.target_key_id, bool(was_revoked))
            return

        if entry.op is AuthOp.ASSIGN_SSPKID:
            target = payload["previous_sspkid"] if want_previous else payload["new_sspkid"]
            registry = getattr(request.app.state, "t1_server_key_registry", None)
            if registry is None:
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server has no SSPKID registry wired.",
                    http_status=501,
                )
            if not target:
                registry.release(entry.target_key_id)
            else:
                from ...security.t2keys import SSPKID

                registry.assign(entry.target_key_id, SSPKID.parse(target))
            return

        if entry.op is AuthOp.PLAN_CHANGE:
            target = payload["previous_plan"] if want_previous else payload["new_plan"]
            if directory is None or not hasattr(directory, "assign_plan"):
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server has no key directory that can reassign a plan.",
                    http_status=501,
                )
            directory.assign_plan(entry.target_key_id, target)
            return

        if entry.op is AuthOp.LEVEL_CHANGE:
            target = payload["previous_level"] if want_previous else payload["new_level"]
            if keymaster is None or not hasattr(keymaster, "set_access_level"):
                raise T1APIError(
                    T1ErrorCode.NOT_SUPPORTED,
                    "This server's Keymaster cannot change a key's access level in place.",
                    http_status=501,
                )
            keymaster.set_access_level(entry.target_key_id, target)
            return

        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED, f"No reverser for {entry.op.value}", http_status=501
        )

    return apply


def _run(request: Request, ctx: AuthContext, history: AuthHistory, audit: Any,
         request_id: str, *, direction: str) -> AuthUndoResponse:
    require_admin(ctx)
    peek = history.peek_undo() if direction == "undo" else history.peek_redo()
    # Audited before the attempt as well as after: "who tried to undo the
    # admin promotion" is worth answering even when the attempt failed.
    audit.record(
        f"auth.{direction}.attempt",
        category=AuditCategory.ADMIN,
        actor_key_id=ctx.key_id,
        actor_is_admin=True,
        outcome=AuditOutcome.SUCCESS,
        resource_type="auth_history",
        resource_id=peek.entry_id if peek else "",
        client_ip=get_client_ip(request),
        request_id=request_id,
    )
    try:
        entry = (
            history.undo(_applier(request))
            if direction == "undo"
            else history.redo(_applier(request))
        )
    except T1APIError:
        audit.record(
            f"auth.{direction}",
            category=AuditCategory.ADMIN,
            actor_key_id=ctx.key_id,
            actor_is_admin=True,
            outcome=AuditOutcome.FAILURE,
            resource_type="auth_history",
            resource_id=peek.entry_id if peek else "",
            client_ip=get_client_ip(request),
            request_id=request_id,
        )
        raise

    audit.record(
        f"auth.{direction}",
        category=AuditCategory.ADMIN,
        actor_key_id=ctx.key_id,
        actor_is_admin=True,
        outcome=AuditOutcome.SUCCESS,
        resource_type="auth_history",
        resource_id=entry.entry_id,
        client_ip=get_client_ip(request),
        request_id=request_id,
        details={"op": entry.op.value, "target": entry.target_key_id},
    )
    state = history.describe(limit=1)
    return AuthUndoResponse(
        direction=direction,
        entry=entry.describe(),
        can_undo=state["can_undo"],
        can_redo=state["can_redo"],
        request_id=request_id,
    )


@router.post("/undo", response_model=AuthUndoResponse)
def undo(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    history: AuthHistory = Depends(get_auth_history),
    request_id: str = Depends(get_request_id),
    audit=Depends(get_audit_log),
) -> AuthUndoResponse:
    """Reverse the most recent authentication change. Admin only."""
    return _run(request, ctx, history, audit, request_id, direction="undo")


@router.post("/redo", response_model=AuthUndoResponse)
def redo(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    history: AuthHistory = Depends(get_auth_history),
    request_id: str = Depends(get_request_id),
    audit=Depends(get_audit_log),
) -> AuthUndoResponse:
    """Replay the most recently undone authentication change. Admin only."""
    return _run(request, ctx, history, audit, request_id, direction="redo")


@router.get("/history", response_model=AuthHistoryResponse)
def history_view(
    limit: int = 50,
    ctx: AuthContext = Depends(get_auth_context),
    history: AuthHistory = Depends(get_auth_history),
    request_id: str = Depends(get_request_id),
) -> AuthHistoryResponse:
    """The undo stack. Never includes key material — see ``describe()``."""
    require_admin(ctx)
    return AuthHistoryResponse(**history.describe(limit=limit), request_id=request_id)


# The aliases under the existing /auth/t1 namespace. Same functions, so
# there is one implementation and no chance of the two drifting.
alias_router.add_api_route("/undo", undo, methods=["POST"], response_model=AuthUndoResponse)
alias_router.add_api_route("/redo", redo, methods=["POST"], response_model=AuthUndoResponse)
alias_router.add_api_route(
    "/history", history_view, methods=["GET"], response_model=AuthHistoryResponse
)

__all__ = ["router", "alias_router", "get_auth_history"]
