"""``/runner/*`` — loading, unloading and switching the served model.

Before this, "switch model" meant walking over to the PC and using LM
Studio. The server could hold forty GGUFs in ``~/.hypernix/models`` and
serve none of them, and nothing in HyperLink could change that.

These endpoints own a llama.cpp process instead — see
:mod:`hypernix.hyperlink.managed` — so load, unload and switch are
operations rather than instructions.

Who may
-------
Changing what a shared server is running affects everybody using it, so
it is gated harder than reading is. Three ways in, and the operator
decides how far the third goes:

* an **admin** key,
* **partial admin** — a key with write, or a trusted-network caller on a
  server whose operator turned that on, and
* ``--switch-perm``: an explicit grant an operator hands to somebody
  paired over ``waiter`` or Tailscale, for exactly this. Off unless
  ``T1_RUNNER_SWITCH_PERM`` names an access level, which is the "access
  6+ if servers enable it" the request asked for.

Planning is separate from doing
-------------------------------
``POST /runner/plan`` says where a model's layers would go and changes
nothing. Loading a model evicts the one people are currently talking to,
so being able to see the consequence first is not a nicety.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from ...hyperlink.catalogue import collect
from ...hyperlink.managed import BACKENDS, ManagedError, plan_placement
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_config,
    get_hyperlink_principal,
    get_registry,
    get_request_id,
    get_runner,
)
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    RunnerLoadRequest,
    RunnerPlanResponse,
    RunnerStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runner", tags=["runner"])


def _access_level(principal: HyperLinkPrincipal) -> int:
    """The caller's numeric access level, or 0.

    Carried in the key's tags rather than invented here — the level is
    the operator's grant, and a router that computed its own would be a
    second opinion about somebody's authority.
    """
    ctx = principal.auth_context
    tags = getattr(getattr(ctx, "key_meta", None), "tags", None) or {}
    try:
        return int(tags.get("access_level") or tags.get("access") or 0)
    except (TypeError, ValueError):
        return 0


def require_switch(
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
) -> HyperLinkPrincipal:
    """Admin, partial admin, or an explicit switch grant.

    The refusal names every route in, because a 403 that does not say how
    to stop being a 403 sends somebody to the wrong place — usually to
    widening a key's scopes, which is not what this checks.
    """
    if principal.is_admin:
        return principal
    if "write" in principal.scopes or "admin" in principal.scopes:
        return principal

    required = int(getattr(config, "runner_switch_perm_level", 0) or 0)
    if required and _access_level(principal) >= required:
        return principal

    raise T1APIError(
        T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
        "Changing the loaded model affects everybody using this server.",
        details={
            "needs": "an admin key, partial administrative access, or a switch grant",
            "switch_perm_enabled": bool(required),
            "switch_perm_level": required,
            "your_access_level": _access_level(principal),
            "remedy": (
                "Use an admin key; or enable T1_TRUSTED_NETWORK_PARTIAL_ADMIN "
                "for trusted origins; or set T1_RUNNER_SWITCH_PERM to the "
                "access level you want to allow, and grant it to the key "
                "paired with waiter or over Tailscale."
            ),
        },
        http_status=403,
    )


def _resolve(model_id: str, config: T1APIConfig, registry) -> tuple[str, dict]:
    """``(path, facts)`` for *model_id*, or a 404 naming what does exist.

    Resolved through the catalogue rather than by joining the id onto the
    models directory: the catalogue is what the app listed, so a model
    the user can see is a model they can load, and a path built by string
    concatenation is a path traversal waiting to be found.
    """
    catalogue = collect(
        registry=registry, bridge=None, local_dir=config.hf_download_dir or None
    )
    for model in catalogue.models:
        if model.model_id == model_id:
            if not model.path:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    f"{model_id} is known to this server but has no file on "
                    f"disk — it came from {model.source}. Only a local GGUF "
                    f"can be loaded by the runner.",
                )
            return model.path, model.to_dict()
    known = [m.model_id for m in catalogue.models if m.path][:20]
    raise T1APIError(
        T1ErrorCode.NOT_FOUND,
        f"No model {model_id!r} on this server.",
        details={"loadable": known},
        http_status=404,
    )


@router.get("/status", response_model=RunnerStatusResponse)
def runner_status(
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    runner=Depends(get_runner),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> RunnerStatusResponse:
    """What is loaded, where its layers are, and how long it has been up.

    Readable by any HyperLink caller: knowing which model is answering is
    not an administrative secret, and a client that cannot tell is a
    client that shows the wrong model name.
    """
    current = runner.current
    return RunnerStatusResponse(
        loaded=current is not None,
        model=current.to_dict() if current else {},
        base_url=runner.base_url,
        backends=list(BACKENDS),
        request_id=request_id,
    )


@router.post("/plan", response_model=RunnerPlanResponse)
def runner_plan(
    payload: RunnerLoadRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry=Depends(get_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> RunnerPlanResponse:
    """Where this model's layers would go. Changes nothing.

    Loading evicts whatever people are currently talking to, so seeing
    the consequence first is not a nicety.
    """
    path, facts = _resolve(payload.model_id, config, registry)
    try:
        placement = plan_placement(
            file_bytes=Path(path).stat().st_size,
            total_layers=payload.total_layers or 0,
            gpu_layers=payload.gpu_layers,
            backend=payload.backend,
            context_length=payload.context_length or 0,
        )
    except ManagedError as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc)) from exc
    return RunnerPlanResponse(
        model_id=payload.model_id,
        path=path,
        placement=placement.to_dict(),
        model=facts,
        request_id=request_id,
    )


@router.post("/load", response_model=RunnerStatusResponse)
def runner_load(
    payload: RunnerLoadRequest,
    principal: HyperLinkPrincipal = Depends(require_switch),
    runner=Depends(get_runner),
    registry=Depends(get_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> RunnerStatusResponse:
    """Load a model, replacing whatever was running.

    Switching *is* loading — there is no separate verb, because two
    llama.cpp servers on one machine each try to take the VRAM the other
    has and the failure lands on the one that was working.
    """
    path, _facts = _resolve(payload.model_id, config, registry)
    try:
        current = runner.load(
            path,
            model_id=payload.model_id,
            gpu_layers=payload.gpu_layers,
            backend=payload.backend,
            context_length=payload.context_length or 0,
            total_layers=payload.total_layers or 0,
        )
    except ManagedError as exc:
        # 400 rather than 500: "there is no built llama.cpp" and "this
        # model does not fit" are both things the operator can act on,
        # and reporting them as a server fault hides the remedy.
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR, str(exc),
            details={"model_id": payload.model_id},
        ) from exc

    logger.info(
        "runner: %s loaded %s (%s)",
        principal.label, payload.model_id, current.placement.reason,
    )
    return RunnerStatusResponse(
        loaded=True, model=current.to_dict(), base_url=runner.base_url,
        backends=list(BACKENDS), request_id=request_id,
    )


@router.post("/unload", response_model=RunnerStatusResponse)
def runner_unload(
    principal: HyperLinkPrincipal = Depends(require_switch),
    runner=Depends(get_runner),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> RunnerStatusResponse:
    """Stop serving. Unloading nothing is a success, not an error."""
    stopped = runner.unload()
    logger.info("runner: %s unloaded (was running: %s)", principal.label, stopped)
    return RunnerStatusResponse(
        loaded=False, model={}, base_url=runner.base_url,
        backends=list(BACKENDS), was_running=stopped, request_id=request_id,
    )


@router.get("/hyperchat")
def runner_hyperchat(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict:
    """Whether prompts run side by side or wait in line, and how deep.

    Readable by any HyperLink caller for the same reason the status is:
    a client that cannot tell shows a spinner that does not move, when
    what it could be showing is "third in line". The queue depth is the
    single most useful number on a shared server and there is nothing
    secret about it.
    """
    from ...hyperlink.hyperchat import plan_cores

    if config.hyperchat_multi:
        budget = plan_cores(wanted=config.hyperchat_instances)
    else:
        # Not `plan_cores(wanted=1)`: that would report a one-instance
        # *pool*, and "off" needs to say so, because an operator who set
        # the flag and sees `instances: 1` should be able to tell
        # "disabled" from "this machine only has the cores for one".
        budget = replace(
            plan_cores(wanted=1), instances=1,
            note="multiple instances are off (T1_HYPERCHAT_MULTI). Prompts "
                 "are answered one at a time, in the order they arrive.",
        )

    chat = getattr(request.app.state, "t1_hyperchat", None)
    return {
        "enabled": config.hyperchat_multi,
        "max_queued": config.hyperchat_max_queued,
        "cores": budget.to_dict(),
        "live": chat.stats() if chat is not None else None,
        "request_id": request_id,
    }
