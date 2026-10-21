"""Usage and cost endpoints.

Beta 1 shipped ``/usage/current`` and ``/usage/remaining``; Beta 3 adds
the three the spec lists that were still missing — ``/usage/history``,
``/usage/cost``, ``/usage/estimate`` — plus ``/usage/by`` for the
"usage by model/key/server/module/user/account" reports.

One rule runs through all of them: **a non-admin caller can only ever
see their own usage.** Filters like ``key_id``/``account_id`` are
accepted from admins and silently pinned to the caller's own identity
otherwise — pinned rather than rejected, because the honest reading of
``GET /usage/history?key_id=someone-else`` from a non-admin is "show me
my history", not an error worth a 403. The pinning happens in
:func:`_scope_filters`, once, so no endpoint can forget it.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Query

from ..audit import AuditCategory
from ..auth import AuthContext
from ..cost import CostCalculator
from ..deps import (
    get_audit_log,
    get_auth_context,
    get_cost_calculator,
    get_key_directory,
    get_request_id,
    get_usage_meter,
)
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    UsageAggregateResponse,
    UsageAggregateRow,
    UsageCostResponse,
    UsageCurrentResponse,
    UsageEstimateRequest,
    UsageEstimateResponse,
    UsageHistoryItem,
    UsageHistoryResponse,
    UsageRemainingResponse,
    UsageReportRequest,
    UsageReportResponse,
    UsageStatementResponse,
)
from ..storage import GROUPABLE
from ..usage import UsageMeter

router = APIRouter(prefix="/usage", tags=["usage"])

# Windows accepted by the ?range= shorthand. Explicit since/until always
# wins; this is a convenience for the common "last 24h / 7d / 30d" call.
_RANGES = {"1h": 3600.0, "24h": 86400.0, "7d": 604800.0, "30d": 2592000.0, "all": None}


def _resolve_window(
    range_: str | None, since: float | None, until: float | None
) -> tuple[float | None, float | None]:
    if since is not None or until is not None:
        return since, until
    if range_ is None:
        return None, None
    if range_ not in _RANGES:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"Unknown range '{range_}'. Use one of: {', '.join(sorted(_RANGES))}, "
            "or pass explicit since/until timestamps.",
            http_status=422,
        )
    span = _RANGES[range_]
    return (None, None) if span is None else (time.time() - span, None)


def _scope_filters(
    ctx: AuthContext,
    *,
    key_id: str | None,
    model_id: str | None,
    server_id: str | None,
    module_id: str | None,
    user_id: str | None,
    account_id: str | None,
    key_directory=None,
) -> dict[str, str | None]:
    """Build the filter dict, pinned to the caller unless they're admin."""
    if ctx.is_admin:
        return {
            "key_id": key_id,
            "model_id": model_id,
            "server_id": server_id,
            "module_id": module_id,
            "user_id": user_id,
            "account_id": account_id,
        }
    # Non-admin: their own key, their own model/server/module filters, and
    # never another user's or account's rows.
    own_user, own_account = ("", "")
    if key_directory is not None:
        own_user, own_account = key_directory.usage_identity(ctx.key_id)
    return {
        "key_id": ctx.key_id,
        "model_id": model_id,
        "server_id": server_id,
        "module_id": module_id,
        "user_id": own_user or None,
        "account_id": own_account or None,
    }


@router.get("/current", response_model=UsageCurrentResponse)
def usage_current(
    ctx: AuthContext = Depends(get_auth_context),
    meter: UsageMeter = Depends(get_usage_meter),
    request_id: str = Depends(get_request_id),
) -> UsageCurrentResponse:
    data = meter.current(ctx.key_id)
    return UsageCurrentResponse(**data, request_id=request_id)


@router.get("/remaining", response_model=UsageRemainingResponse)
def usage_remaining(
    model_id: str = Query(..., description="Registered model_id to check remaining allowance for."),
    ctx: AuthContext = Depends(get_auth_context),
    meter: UsageMeter = Depends(get_usage_meter),
    request_id: str = Depends(get_request_id),
) -> UsageRemainingResponse:
    snap = meter.snapshot_for_model(ctx.key_id, model_id)
    return UsageRemainingResponse(
        model_id=model_id,
        input_remaining=snap.input_remaining,
        output_remaining=snap.output_remaining,
        is_exhausted=snap.is_exhausted,
        request_id=request_id,
    )


@router.get("/history", response_model=UsageHistoryResponse)
def usage_history(
    range_: str | None = Query(default=None, alias="range", description="1h|24h|7d|30d|all"),
    since: float | None = Query(default=None, description="POSIX timestamp, inclusive."),
    until: float | None = Query(default=None, description="POSIX timestamp, exclusive."),
    key_id: str | None = Query(default=None, description="Admin only; ignored otherwise."),
    model_id: str | None = Query(default=None),
    server_id: str | None = Query(default=None),
    module_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None, description="Admin only; ignored otherwise."),
    account_id: str | None = Query(default=None, description="Admin only; ignored otherwise."),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ctx: AuthContext = Depends(get_auth_context),
    meter: UsageMeter = Depends(get_usage_meter),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
) -> UsageHistoryResponse:
    resolved_since, resolved_until = _resolve_window(range_, since, until)
    filters = _scope_filters(
        ctx,
        key_id=key_id,
        model_id=model_id,
        server_id=server_id,
        module_id=module_id,
        user_id=user_id,
        account_id=account_id,
        key_directory=keys,
    )
    events = meter.store.history(
        since=resolved_since, until=resolved_until, limit=limit, offset=offset, **filters
    )
    return UsageHistoryResponse(
        events=[UsageHistoryItem(**e) for e in events],
        count=len(events),
        since=resolved_since,
        until=resolved_until,
        limit=limit,
        offset=offset,
        request_id=request_id,
    )


@router.get("/by", response_model=UsageAggregateResponse)
def usage_by(
    group_by: str = Query(..., description=f"One of: {', '.join(GROUPABLE)}"),
    range_: str | None = Query(default=None, alias="range"),
    since: float | None = Query(default=None),
    until: float | None = Query(default=None),
    key_id: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
    server_id: str | None = Query(default=None),
    module_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    ctx: AuthContext = Depends(get_auth_context),
    meter: UsageMeter = Depends(get_usage_meter),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
) -> UsageAggregateResponse:
    """"Usage by model / key / server / module / user / account", as one
    endpoint rather than six near-identical ones.

    ``group_by`` is checked against the storage layer's allowlist before
    it reaches SQL — it becomes an identifier, not a bound parameter, so
    it can never be free-form caller input.
    """
    if group_by not in GROUPABLE:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"Cannot group usage by '{group_by}'. Valid dimensions: {', '.join(GROUPABLE)}.",
            details={"group_by": group_by},
            http_status=422,
        )
    resolved_since, resolved_until = _resolve_window(range_, since, until)
    filters = _scope_filters(
        ctx,
        key_id=key_id,
        model_id=model_id,
        server_id=server_id,
        module_id=module_id,
        user_id=user_id,
        account_id=account_id,
        key_directory=keys,
    )
    rows = meter.store.aggregate(group_by, since=resolved_since, until=resolved_until, **filters)
    return UsageAggregateResponse(
        group_by=group_by,
        rows=[
            UsageAggregateRow(
                group_by=group_by,
                value=str(row[group_by]),
                requests=row["requests"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                total_tokens=row["total_tokens"],
            )
            for row in rows
        ],
        count=len(rows),
        since=resolved_since,
        until=resolved_until,
        request_id=request_id,
    )


@router.get("/cost", response_model=UsageCostResponse)
def usage_cost(
    group_by: str = Query(default="model_id", description=f"One of: {', '.join(GROUPABLE)}"),
    range_: str | None = Query(default=None, alias="range"),
    since: float | None = Query(default=None),
    until: float | None = Query(default=None),
    key_id: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
    server_id: str | None = Query(default=None),
    module_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    forecast: bool = Query(default=False, description="Include a spend forecast for the horizon."),
    forecast_horizon_seconds: float = Query(default=2592000.0, gt=0),
    ctx: AuthContext = Depends(get_auth_context),
    calculator: CostCalculator = Depends(get_cost_calculator),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
) -> UsageCostResponse:
    """Actual cost with optional forecast — totals, date ranges, and
    per-model/per-server breakdowns in one shape."""
    if group_by not in GROUPABLE:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"Cannot group cost by '{group_by}'. Valid dimensions: {', '.join(GROUPABLE)}.",
            http_status=422,
        )
    resolved_since, resolved_until = _resolve_window(range_, since, until)
    filters = _scope_filters(
        ctx,
        key_id=key_id,
        model_id=model_id,
        server_id=server_id,
        module_id=module_id,
        user_id=user_id,
        account_id=account_id,
        key_directory=keys,
    )
    report = calculator.cost_report(
        group_by=group_by, since=resolved_since, until=resolved_until, **filters
    )
    payload = report.to_dict()
    forecast_payload = None
    if forecast:
        forecast_payload = calculator.forecast(
            horizon_seconds=forecast_horizon_seconds, **filters
        ).to_dict()
    return UsageCostResponse(**payload, forecast=forecast_payload, request_id=request_id)


@router.get("/statement", response_model=UsageStatementResponse)
def usage_statement(
    range_: str | None = Query(default=None, alias="range"),
    since: float | None = Query(default=None),
    until: float | None = Query(default=None),
    key_id: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
    server_id: str | None = Query(default=None),
    module_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    ctx: AuthContext = Depends(get_auth_context),
    calculator: CostCalculator = Depends(get_cost_calculator),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
) -> UsageStatementResponse:
    """The itemised bill (``t1.statement/2``, 0.72.8).

    ``/usage/cost`` prices tokens along one dimension. This lists every
    feature that was used -- inference, context relays, embeddings,
    compaction, code sandboxes, web searches, client reports -- with a
    line per model and unit under each, subtotals, the total, and what
    nothing prices. Scoped to the caller's own usage unless admin, as
    every usage read is.
    """
    resolved_since, resolved_until = _resolve_window(range_, since, until)
    filters = _scope_filters(
        ctx, key_id=key_id, model_id=model_id, server_id=server_id, module_id=module_id,
        user_id=user_id, account_id=account_id, key_directory=keys,
    )
    statement = calculator.statement(since=resolved_since, until=resolved_until, **filters)
    return UsageStatementResponse(**statement, request_id=request_id)


@router.post("/estimate", response_model=UsageEstimateResponse)
def usage_estimate(
    body: UsageEstimateRequest,
    ctx: AuthContext = Depends(get_auth_context),
    calculator: CostCalculator = Depends(get_cost_calculator),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
) -> UsageEstimateResponse:
    """What an operation *would* cost. Records nothing and reserves
    nothing — see ``t1api.cost``.

    POST rather than GET because it takes a body and, more importantly,
    because a cost estimate for a specific prompt size is a question
    about a proposed operation, not a resource that can be bookmarked.
    """
    keys.assert_model_allowed(ctx.key_id, body.model_id)
    estimate = calculator.estimate(
        model_id=body.model_id,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        requests=body.requests,
    )
    return UsageEstimateResponse(**estimate, request_id=request_id)


@router.post("/report", response_model=UsageReportResponse)
def usage_report(
    body: UsageReportRequest,
    ctx: AuthContext = Depends(get_auth_context),
    meter: UsageMeter = Depends(get_usage_meter),
    keys=Depends(get_key_directory),
    audit=Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> UsageReportResponse:
    """Record tokens a client consumed running a model this server routed it to.

    **Why this exists (Beta 4).** Beta 3 could route a request and refuse
    an exhausted model, but nothing could ever *report* consumption back:
    ``UsageMeter.record`` had no HTTP surface. For a client that runs
    inference itself — ``hyped-pro`` against a remote T1 server is the
    motivating case — that meant the quota cascade never advanced, so
    every call routed to the same first model forever and per-model
    limits were unenforceable in practice.

    Three rules make this safe to expose to any authenticated key:

    * **Usage is recorded against the caller's own key**, never a
      body-supplied one, so a key can only ever spend its own budget.
    * **The model must be registered** (``UsageMeter.record`` calls
      ``registry.require``) **and allowed for this key**, so a report
      can't invent a model or a budget line.
    * **Counts are non-negative and capped** by the request schema, so a
      report can add usage but never subtract it. A client that could
      report negative tokens could refund itself quota, which would make
      every limit in the system advisory.

    Reporting is not the same as being permitted: a report that pushes a
    model past its cap succeeds and is recorded (the tokens really were
    spent), and the response says ``exhausted: true``. The refusal
    belongs on the *next* route call, not on the accounting for work
    already done.
    """
    keys.assert_model_allowed(ctx.key_id, body.model_id)
    meter.record(
        key_id=ctx.key_id,
        model_id=body.model_id,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        requests=body.requests,
        endpoint=body.endpoint,
        server_id=body.server_id,
        module_id=body.module_id,
        feature="reported",
    )
    snap = meter.snapshot_for_model(ctx.key_id, body.model_id)
    audit.record(
        "usage.report",
        category=AuditCategory.WRITE,
        actor_key_id=ctx.key_id,
        actor_is_admin=ctx.is_admin,
        resource_type="model",
        resource_id=body.model_id,
        request_id=request_id,
        details={
            "input_tokens": body.input_tokens,
            "output_tokens": body.output_tokens,
            "requests": body.requests,
            "exhausted_after": snap.is_exhausted,
        },
    )
    return UsageReportResponse(
        recorded=True,
        model_id=body.model_id,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        requests=body.requests,
        input_remaining=snap.input_remaining,
        output_remaining=snap.output_remaining,
        exhausted=snap.is_exhausted,
        request_id=request_id,
    )


__all__ = ["router"]
