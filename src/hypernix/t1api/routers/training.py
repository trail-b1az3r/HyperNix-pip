"""Training administration — 0.72.4 item 5.

Read a run's progress, read its log, list its checkpoints, see what the
machine is doing, and stop/pause/resume it. Backed by
``hypernix.training.monitor``; this module is only the access decision
and the shape of the reply.

Who may call these
------------------
The spec: *"These features should be admin-only unless the server is
using the explicitly enabled trusted LAN/Tailscale keyless mode"* and
*"Do not expose destructive or sensitive training controls to ordinary
users."* Two tiers, then, and the second is a separate opt-in:

* **Reading** (runs, logs, checkpoints, resources) — an admin key, or a
  trusted origin when the operator turned trusted-network mode on.
* **Controlling** (stop, pause, resume) — an admin key, or a trusted
  origin when the operator additionally turned on *partial admin*.
  Killing six hours of training is not something a device that
  presented no credential should be able to do because it happens to be
  on the same wifi.

A public origin never qualifies for either, whatever the configuration
says: :meth:`TrustPolicy.allows_keyless` refuses ``PUBLIC`` before the
policy is consulted, so no amount of configuration can turn an
unauthenticated internet connection into training administration.

An ordinary (non-admin) key does not *lose* access it would have had
keyless from the same address — the tier is decided by the strongest of
the two, because "presenting a read key made you less trusted than
presenting nothing" is a rule nobody could design around.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from ..audit import AuditCategory, AuditLog, AuditOutcome
from ..auth import AuthContext
from ..deps import (
    get_audit_log,
    get_auth_context,
    get_origin,
    get_request_id,
    get_training_monitor,
    get_trust_policy,
)
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    TrainingCheckpointItem,
    TrainingCheckpointListResponse,
    TrainingControlResponse,
    TrainingLogResponse,
    TrainingResourcesResponse,
    TrainingRunItem,
    TrainingRunListResponse,
    TrainingRunResponse,
)

router = APIRouter(prefix="/training", tags=["training"])


# --- access ------------------------------------------------------------


def _deny(what: str, reason: str) -> T1APIError:
    return T1APIError(
        T1ErrorCode.AUTH_ADMIN_REQUIRED,
        f"{what} requires an admin-scoped T1 key.",
        details={
            "reason": reason,
            "remedy": (
                "Use an admin key, or enable trusted-network mode on the "
                "server (install-t1.sh --trusted-network) if this machine "
                "is on a network you control."
            ),
        },
        http_status=403,
    )


def require_training_read(
    request: Request, ctx: AuthContext = Depends(get_auth_context)
) -> AuthContext:
    """Admin, or a trusted origin on a server in trusted-network mode."""
    if ctx.is_admin:
        return ctx
    policy = get_trust_policy(request)
    if policy.enabled and policy.allows_keyless(get_origin(request)):
        return ctx
    raise _deny("Reading training runs", "training_read_requires_admin")


def require_training_control(
    request: Request, ctx: AuthContext = Depends(get_auth_context)
) -> AuthContext:
    """Admin, or a trusted origin where partial admin is also enabled.

    Deliberately stricter than :func:`require_training_read`. Trusted
    mode alone says "this network may read"; the destructive controls
    need the second switch, which an operator sets knowing that anything
    on the LAN can then end a training run.
    """
    if ctx.is_admin:
        return ctx
    policy = get_trust_policy(request)
    if policy.enabled and policy.allows_partial_admin(get_origin(request)):
        return ctx
    raise _deny("Controlling a training run", "training_control_requires_admin")


def _audit(
    audit: AuditLog,
    request: Request,
    ctx: AuthContext,
    request_id: str,
    action: str,
    *,
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    **details: object,
) -> None:
    from ..deps import get_client_ip

    audit.record(
        action,
        category=AuditCategory.ADMIN,
        outcome=outcome,
        actor_key_id=ctx.key_id,
        actor_is_admin=ctx.is_admin,
        resource_type="training_run",
        resource_id=str(details.get("run_id", "")),
        request_id=request_id,
        client_ip=get_client_ip(request),
        details=dict(details),
    )


def _item(run) -> TrainingRunItem:
    return TrainingRunItem(**run.to_dict())


def _require_run(monitor, run_id: str):
    run = monitor.get(run_id)
    if run is None:
        raise T1APIError(
            T1ErrorCode.NOT_FOUND,
            f"No training run named {run_id!r}.",
            details={"run_id": run_id, "root": str(monitor.root)},
            http_status=404,
        )
    return run


# --- reading -----------------------------------------------------------


@router.get("/runs", response_model=TrainingRunListResponse)
def list_runs(
    active: bool = Query(False, description="Only runs that have not ended."),
    ctx: AuthContext = Depends(require_training_read),
    monitor=Depends(get_training_monitor),
    request_id: str = Depends(get_request_id),
) -> TrainingRunListResponse:
    runs = monitor.runs()
    active_count = sum(1 for run in runs if run.is_active)
    if active:
        runs = [run for run in runs if run.is_active]
    return TrainingRunListResponse(
        runs=[_item(run) for run in runs],
        count=len(runs),
        active=active_count,
        root=str(monitor.root),
        request_id=request_id,
    )


@router.get("/resources", response_model=TrainingResourcesResponse)
def training_resources(
    ctx: AuthContext = Depends(require_training_read),
    monitor=Depends(get_training_monitor),
    request_id: str = Depends(get_request_id),
) -> TrainingResourcesResponse:
    """GPU/CPU/RAM, for the panel the run is displayed in.

    Declared before ``/runs/{run_id}`` so the literal path wins the
    match — not strictly required by Starlette's router, which prefers
    the earlier registration, which is exactly why the order matters.
    """
    return TrainingResourcesResponse(**monitor.resources(), request_id=request_id)


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
def get_run(
    run_id: str,
    ctx: AuthContext = Depends(require_training_read),
    monitor=Depends(get_training_monitor),
    request_id: str = Depends(get_request_id),
) -> TrainingRunResponse:
    return TrainingRunResponse(
        run=_item(_require_run(monitor, run_id)), request_id=request_id
    )


@router.get("/runs/{run_id}/logs", response_model=TrainingLogResponse)
def get_run_logs(
    run_id: str,
    tail: int = Query(200, ge=1, le=5000),
    ctx: AuthContext = Depends(require_training_read),
    monitor=Depends(get_training_monitor),
    request_id: str = Depends(get_request_id),
) -> TrainingLogResponse:
    """The tail of the run's log, when it has one.

    A run started by hand rather than through the launcher has no log
    path, and a run whose log has been rotated away has a path with
    nothing behind it. Both answer 200 with an empty ``log`` and a
    ``detail`` saying which — a 404 here would read as "no such run",
    which is a different and more alarming thing.
    """
    run = _require_run(monitor, run_id)
    if not run.log_path:
        return TrainingLogResponse(
            run_id=run.run_id,
            detail=(
                "This run has no log file. It was started outside the "
                "launcher, so its output went wherever it was started from."
            ),
            request_id=request_id,
        )
    path = Path(run.log_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return TrainingLogResponse(
            run_id=run.run_id,
            path=str(path),
            detail=f"Could not read {path}: {exc}",
            request_id=request_id,
        )
    window = lines[-tail:]
    return TrainingLogResponse(
        run_id=run.run_id,
        log="\n".join(window),
        lines=len(window),
        path=str(path),
        request_id=request_id,
    )


@router.get(
    "/runs/{run_id}/checkpoints", response_model=TrainingCheckpointListResponse
)
def get_run_checkpoints(
    run_id: str,
    ctx: AuthContext = Depends(require_training_read),
    monitor=Depends(get_training_monitor),
    request_id: str = Depends(get_request_id),
) -> TrainingCheckpointListResponse:
    """What the trainer said it wrote, and whether it is still there.

    ``exists`` is checked rather than assumed: a checkpoint list is
    mostly used to decide what to resume from, and a path that was
    deleted or written to a filesystem that has since gone away is the
    case worth knowing about.
    """
    run = _require_run(monitor, run_id)
    items: list[TrainingCheckpointItem] = []
    for raw in run.checkpoints:
        path = Path(raw)
        try:
            stat = path.stat()
        except OSError:
            items.append(TrainingCheckpointItem(path=raw, exists=False))
            continue
        items.append(
            TrainingCheckpointItem(
                path=raw,
                exists=True,
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
            )
        )
    return TrainingCheckpointListResponse(
        run_id=run.run_id,
        checkpoints=items,
        count=len(items),
        request_id=request_id,
    )


# --- controls ----------------------------------------------------------


_PAUSE_NOTE = (
    "The process is frozen with its GPU allocations intact. Resuming is "
    "instant; the card is not released while paused."
)


def _control(monitor, run, action: str):
    from ...training.monitor import TrainingError

    handler = {"pause": monitor.pause, "resume": monitor.resume, "stop": monitor.stop}[
        action
    ]
    try:
        return handler(run)
    except TrainingError as exc:
        raise T1APIError(
            T1ErrorCode.CONFLICT,
            str(exc),
            details={"run_id": run.run_id, "action": action, "state": run.state},
            http_status=409,
        ) from exc


@router.post("/runs/{run_id}/pause", response_model=TrainingControlResponse)
def pause_run(
    request: Request,
    run_id: str,
    ctx: AuthContext = Depends(require_training_control),
    monitor=Depends(get_training_monitor),
    audit: AuditLog = Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> TrainingControlResponse:
    run = _control(monitor, _require_run(monitor, run_id), "pause")
    _audit(audit, request, ctx, request_id, "training.pause", run_id=run_id)
    return TrainingControlResponse(
        run=_item(run), action="pause", note=_PAUSE_NOTE, request_id=request_id
    )


@router.post("/runs/{run_id}/resume", response_model=TrainingControlResponse)
def resume_run(
    request: Request,
    run_id: str,
    ctx: AuthContext = Depends(require_training_control),
    monitor=Depends(get_training_monitor),
    audit: AuditLog = Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> TrainingControlResponse:
    run = _control(monitor, _require_run(monitor, run_id), "resume")
    _audit(audit, request, ctx, request_id, "training.resume", run_id=run_id)
    return TrainingControlResponse(
        run=_item(run), action="resume", request_id=request_id
    )


@router.post("/runs/{run_id}/stop", response_model=TrainingControlResponse)
def stop_run(
    request: Request,
    run_id: str,
    ctx: AuthContext = Depends(require_training_control),
    monitor=Depends(get_training_monitor),
    audit: AuditLog = Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> TrainingControlResponse:
    """SIGTERM the run's process group.

    Terminate rather than kill: a trainer that handles SIGTERM gets the
    chance to write a final checkpoint, and the difference between
    "stopped at epoch 4" and "lost epoch 4" is the whole value of asking
    politely first.
    """
    target = _require_run(monitor, run_id)
    had_process = bool(target.pid)
    run = _control(monitor, target, "stop")
    _audit(audit, request, ctx, request_id, "training.stop", run_id=run_id)
    return TrainingControlResponse(
        run=_item(run),
        action="stop",
        note=(
            "Sent SIGTERM. Checkpoints written on shutdown may still appear."
            if had_process
            else "No process was recorded for this run, so nothing was "
            "signalled; the record is now marked stopped."
        ),
        request_id=request_id,
    )


__all__ = ["router", "require_training_read", "require_training_control"]
