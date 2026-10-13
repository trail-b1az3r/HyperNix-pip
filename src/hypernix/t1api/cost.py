"""t1api.cost — usage costing, estimates, and forecasts.

Backs the three usage endpoints Beta 1/2 left unimplemented:
``GET /usage/cost``, ``GET /usage/estimate``, and the cost half of
``GET /usage/history``. The spec's requirements for these are specific:

  Cost endpoints must support: totals, forecasts, estimates, date
  ranges, per-model costs, per-server costs.

Every number here comes from two places and nowhere else: recorded usage
events (:class:`t1api.storage.UsageStore`) and the registry entry's
``pricing`` block (:class:`t1api.registry.ModelPricing`). There is no
second price list — a model's price is a registry fact, so changing it
is a registry change, and a model that isn't registered has no price and
cannot be costed. That is the same "server decides, client only asks"
rule the rest of the API runs on, applied to money.

Two honesty rules shape the API:

* **Estimates are labelled estimates.** :meth:`CostCalculator.estimate`
  takes a *requested* token count and returns what it *would* cost. It
  never records anything and never reserves budget; a caller that wants
  a guarantee has to actually make the request.
* **Forecasts state their basis.** :meth:`CostCalculator.forecast`
  returns the observation window it extrapolated from and a confidence
  band, because "you will spend $340 this month" derived from eleven
  minutes of history is a number that should arrive with that context
  attached rather than looking like a measurement.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import T1APIError, T1ErrorCode
from .registry import ModelRegistry
from .storage import UsageStore

_DAY = 86400.0

# A forecast extrapolated from less history than this is reported with
# low confidence — it isn't withheld (an operator asking for a forecast
# on a new account should still get the arithmetic), but it is labelled.
_MIN_CONFIDENT_OBSERVATION_SECONDS = 6 * 3600.0


@dataclass
class CostLine:
    """Cost attributable to one dimension value (a model, a server, ...)."""

    dimension: str
    value: str
    requests: int
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    currency: str = "USD"

    @property
    def total_cost(self) -> float:
        return round(self.input_cost + self.output_cost, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": self.total_cost,
            "currency": self.currency,
        }


@dataclass
class CostReport:
    total_cost: float
    currency: str
    since: float | None
    until: float | None
    lines: list[CostLine] = field(default_factory=list)
    unpriced_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 6),
            "currency": self.currency,
            "since": self.since,
            "until": self.until,
            "lines": [line.to_dict() for line in self.lines],
            "unpriced_models": self.unpriced_models,
        }


@dataclass
class Forecast:
    projected_cost: float
    horizon_seconds: float
    observed_seconds: float
    observed_cost: float
    rate_per_day: float
    confidence: str  # "low" | "medium" | "high"
    currency: str = "USD"
    basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "projected_cost": round(self.projected_cost, 6),
            "horizon_seconds": self.horizon_seconds,
            "observed_seconds": round(self.observed_seconds, 2),
            "observed_cost": round(self.observed_cost, 6),
            "rate_per_day": round(self.rate_per_day, 6),
            "confidence": self.confidence,
            "currency": self.currency,
            "basis": self.basis,
        }


#: The billing format :meth:`CostCalculator.statement` writes.
STATEMENT_FORMAT = "t1.statement/2"

#: Every feature a statement can itemise, in the order it lists them.
FEATURES: dict[str, str] = {
    "inference": "Model inference (chat, completions, streams)",
    "relay": "Context relay (long prompts read in hops by small-context instances)",
    "embeddings": "Embeddings",
    "compact": "Chat compaction",
    "code": "Code sandboxes",
    "web_search": "Web search",
    "reported": "Usage reported by clients",
    "other": "Other",
}

#: Units billed in something other than tokens, and what one of each is.
UNITS: dict[str, str] = {
    "code_second": "one second of sandbox run time",
    "web_search": "one web search",
    "image": "one image",
}

# Rows recorded before 0.72.8 carry no feature; their endpoint says it.
_ENDPOINT_FEATURES: tuple[tuple[str, str], ...] = (
    ("/inference/relay", "relay"),
    ("/inference/embeddings", "embeddings"),
    ("/inference/", "inference"),
    ("/chat/compact", "compact"),
    ("/code/", "code"),
    ("/web/", "web_search"),
    ("/usage/report", "reported"),
)


def feature_of(feature: str, endpoint: str) -> str:
    """The statement feature for a usage row."""
    if feature:
        return feature if feature in FEATURES else "other"
    for prefix, name in _ENDPOINT_FEATURES:
        if endpoint.startswith(prefix):
            return name
    return "other"


class CostCalculator:
    """Prices recorded usage, estimates prospective usage, forecasts spend."""

    def __init__(self, store: UsageStore, registry: ModelRegistry,
                 feature_prices: dict[str, float] | None = None) -> None:
        self.store = store
        self.registry = registry
        #: Server-wide prices per unit (``T1_FEATURE_PRICES``).
        self.feature_prices = dict(feature_prices or {})

    # ------------------------------------------------------------------
    # Pricing primitives
    # ------------------------------------------------------------------

    def price_tokens(self, model_id: str, input_tokens: int, output_tokens: int) -> tuple[float, float, str]:
        """Price a token pair for one registered model.

        Raises ``MODEL_NOT_SUPPORTED`` for an unregistered model —
        pricing an unknown model would mean inventing a price, which is
        exactly what the registry rule exists to prevent.
        """
        entry = self.registry.require(model_id)
        pricing = entry.pricing
        input_cost = (input_tokens / 1000.0) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * pricing.output_price_per_1k
        return input_cost, output_cost, pricing.currency

    def _price_or_zero(self, model_id: str, input_tokens: int, output_tokens: int) -> tuple[float, float, str, bool]:
        """Like :meth:`price_tokens` but tolerant of historical rows.

        Usage recorded against a model that has since been removed from
        the registry still exists and must still be reportable; it is
        priced at zero and the model is listed in
        ``CostReport.unpriced_models`` so the gap is visible rather than
        silently folded into the total.
        """
        entry = self.registry.get(model_id)
        if entry is None:
            return 0.0, 0.0, "USD", False
        pricing = entry.pricing
        return (
            (input_tokens / 1000.0) * pricing.input_price_per_1k,
            (output_tokens / 1000.0) * pricing.output_price_per_1k,
            pricing.currency,
            True,
        )

    def price_units(self, model_id: str, unit: str, units: float) -> tuple[float, str, float | None]:
        """Price *units* of *unit*: ``(cost, currency, unit_price)``.

        The model's own ``unit_prices`` first, then the server's
        ``T1_FEATURE_PRICES``; a unit priced by neither has
        ``unit_price`` None and costs 0, and a statement lists it as
        unpriced rather than inventing a price.
        """
        entry = self.registry.get(model_id) if model_id else None
        if entry is not None and unit in entry.pricing.unit_prices:
            price = entry.pricing.unit_prices[unit]
            return units * price, entry.pricing.currency, price
        if unit in self.feature_prices:
            return units * self.feature_prices[unit], "USD", self.feature_prices[unit]
        return 0.0, "USD", None

    # ------------------------------------------------------------------
    # Actual cost
    # ------------------------------------------------------------------

    def cost_report(
        self,
        *,
        group_by: str = "model_id",
        since: float | None = None,
        until: float | None = None,
        **filters: str | None,
    ) -> CostReport:
        """Actual cost over a date range, broken down by one dimension.

        Costing always walks per-model rows internally even when grouping
        by something else (server, module, key): price is a property of
        the *model*, so a per-server total is the sum of that server's
        per-model costs, not a server-level rate.
        """
        rows = self.store.aggregate("model_id", since=since, until=until, **filters)
        currency = "USD"
        unpriced: list[str] = []
        total = 0.0
        for row in rows:
            input_cost, output_cost, row_currency, priced = self._price_or_zero(
                row["model_id"], row["input_tokens"], row["output_tokens"]
            )
            if not priced:
                # A row with no model is a unit-billed feature (a sandbox
                # run, a search), which /usage/statement prices; it is not
                # a model missing from the registry.
                if row["model_id"]:
                    unpriced.append(row["model_id"])
            else:
                currency = row_currency
            total += input_cost + output_cost

        if group_by == "model_id":
            lines = [
                self._line("model_id", row["model_id"], row, currency)
                for row in rows
            ]
        else:
            lines = []
            for group_row in self.store.aggregate(group_by, since=since, until=until, **filters):
                group_value = group_row[group_by]
                # Re-slice per model *within* this group so each line's
                # cost uses the right model prices.
                sub_filters = dict(filters)
                sub_filters[group_by] = group_value
                sub_rows = self.store.aggregate("model_id", since=since, until=until, **sub_filters)
                line_input_cost = 0.0
                line_output_cost = 0.0
                for sub in sub_rows:
                    ic, oc, _, _ = self._price_or_zero(
                        sub["model_id"], sub["input_tokens"], sub["output_tokens"]
                    )
                    line_input_cost += ic
                    line_output_cost += oc
                lines.append(
                    CostLine(
                        dimension=group_by,
                        value=group_value,
                        requests=group_row["requests"],
                        input_tokens=group_row["input_tokens"],
                        output_tokens=group_row["output_tokens"],
                        input_cost=line_input_cost,
                        output_cost=line_output_cost,
                        currency=currency,
                    )
                )

        lines.sort(key=lambda line: line.total_cost, reverse=True)
        return CostReport(
            total_cost=total,
            currency=currency,
            since=since,
            until=until,
            lines=lines,
            unpriced_models=sorted(set(unpriced)),
        )

    def _line(self, dimension: str, value: str, row: dict[str, Any], currency: str) -> CostLine:
        input_cost, output_cost, row_currency, _ = self._price_or_zero(
            row["model_id"], row["input_tokens"], row["output_tokens"]
        )
        return CostLine(
            dimension=dimension,
            value=value,
            requests=row["requests"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            input_cost=input_cost,
            output_cost=output_cost,
            currency=row_currency or currency,
        )

    # ------------------------------------------------------------------
    # Itemised statement (t1.statement/2)
    # ------------------------------------------------------------------

    def statement(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        **filters: str | None,
    ) -> dict[str, Any]:
        """Everything billed over a range, itemised by feature and model.

        Tokens are priced from the model's registry entry; other units
        (sandbox seconds, searches, images) from the model's
        ``unit_prices`` or the server's ``T1_FEATURE_PRICES``. Anything
        with no price is listed under ``unpriced`` and counted at 0, so
        the total never holds a number nobody set.
        """
        features: dict[str, dict[str, Any]] = {}
        items: dict[tuple[str, str, str], dict[str, Any]] = {}
        unpriced: set[tuple[str, str, str]] = set()
        currencies: set[str] = set()

        for row in self.store.line_items(since=since, until=until, **filters):
            feature = feature_of(row["feature"], row["endpoint"])
            key = (feature, row["model_id"], row["unit"])
            item = items.setdefault(key, {
                "model_id": row["model_id"], "unit": row["unit"], "requests": 0,
                "input_tokens": 0, "output_tokens": 0, "units": 0.0,
            })
            for field_name in ("requests", "input_tokens", "output_tokens", "units"):
                item[field_name] += row[field_name]

        for (feature, model_id, unit), item in sorted(items.items()):
            input_cost = output_cost = unit_cost = 0.0
            priced = True
            if item["input_tokens"] or item["output_tokens"]:
                # Tokens with no registered model (a compaction by whatever
                # the server had loaded) have no price to apply.
                input_cost, output_cost, currency, priced = self._price_or_zero(
                    model_id, item["input_tokens"], item["output_tokens"])
                if priced:
                    currencies.add(currency)
            unit_price = None
            if unit and item["units"]:
                unit_cost, currency, unit_price = self.price_units(model_id, unit, item["units"])
                if unit_price is None:
                    priced = False
                else:
                    currencies.add(currency)
            if not priced:
                unpriced.add((feature, model_id, unit))
            total = input_cost + output_cost + unit_cost
            line = {
                **item,
                "units": round(item["units"], 3),
                "total_tokens": item["input_tokens"] + item["output_tokens"],
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "unit_price": unit_price,
                "unit_cost": round(unit_cost, 6),
                "total": round(total, 6),
                "priced": priced,
            }
            group = features.setdefault(feature, {
                "feature": feature, "label": FEATURES[feature], "requests": 0,
                "input_tokens": 0, "output_tokens": 0, "units": {}, "subtotal": 0.0, "items": [],
            })
            group["items"].append(line)
            group["requests"] += item["requests"]
            group["input_tokens"] += item["input_tokens"]
            group["output_tokens"] += item["output_tokens"]
            if unit:
                group["units"][unit] = round(group["units"].get(unit, 0.0) + item["units"], 3)
            group["subtotal"] += total

        ordered = [features[name] for name in FEATURES if name in features]
        for group in ordered:
            group["items"].sort(key=lambda line: line["total"], reverse=True)
            group["subtotal"] = round(group["subtotal"], 6)
        total = round(sum(group["subtotal"] for group in ordered), 6)
        return {
            "format": STATEMENT_FORMAT,
            "currency": currencies.pop() if len(currencies) == 1 else ("mixed" if currencies else "USD"),
            "since": since,
            "until": until,
            "generated_at": time.time(),
            "features": ordered,
            "total": total,
            "requests": sum(group["requests"] for group in ordered),
            "unpriced": [
                {"feature": f, "model_id": m, "unit": u} for f, m, u in sorted(unpriced)
            ],
            "units": {unit: description for unit, description in UNITS.items()},
        }

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    def estimate(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int | None = None,
        requests: int = 1,
    ) -> dict[str, Any]:
        """What *would* this cost? Records nothing, reserves nothing.

        When *output_tokens* is omitted the model's own
        ``output_token_limit`` is used as the worst case, so an estimate
        is an upper bound rather than an optimistic guess a caller could
        be surprised by.
        """
        entry = self.registry.require(model_id)
        if input_tokens < 0 or (output_tokens is not None and output_tokens < 0):
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                "Token counts in a cost estimate cannot be negative.",
                http_status=422,
            )
        assumed_output = entry.output_token_limit if output_tokens is None else output_tokens
        per_request_input, per_request_output, currency = self.price_tokens(
            model_id, input_tokens, assumed_output
        )
        exceeds_input_limit = input_tokens > entry.input_token_limit
        exceeds_context = input_tokens > entry.context_limit
        return {
            "model_id": model_id,
            "requests": requests,
            "input_tokens": input_tokens * requests,
            "output_tokens": assumed_output * requests,
            "output_tokens_assumed": output_tokens is None,
            "input_cost": round(per_request_input * requests, 6),
            "output_cost": round(per_request_output * requests, 6),
            "estimated_cost": round((per_request_input + per_request_output) * requests, 6),
            "currency": currency,
            "exceeds_input_token_limit": exceeds_input_limit,
            "exceeds_context_limit": exceeds_context,
            "input_token_limit": entry.input_token_limit,
            "context_limit": entry.context_limit,
            "note": (
                "Upper-bound estimate: output_tokens defaulted to the model's output_token_limit."
                if output_tokens is None
                else "Estimate based on the supplied token counts."
            ),
        }

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------

    def forecast(
        self,
        *,
        horizon_seconds: float = 30 * _DAY,
        lookback_seconds: float = 7 * _DAY,
        **filters: str | None,
    ) -> Forecast:
        """Project spend over *horizon_seconds* from observed history.

        Linear extrapolation of the observed daily rate — deliberately
        the simplest defensible model. Anything fancier (seasonality,
        trend fitting) would imply a precision this data doesn't support,
        and the confidence label tells the caller how much history the
        number rests on.
        """
        now = time.time()
        since = now - lookback_seconds
        report = self.cost_report(since=since, until=now, **filters)
        first_ts = self.store.first_event_ts(**filters)

        if first_ts is None:
            return Forecast(
                projected_cost=0.0,
                horizon_seconds=horizon_seconds,
                observed_seconds=0.0,
                observed_cost=0.0,
                rate_per_day=0.0,
                confidence="low",
                currency=report.currency,
                basis="No usage recorded — nothing to extrapolate from.",
            )

        observed_seconds = max(1.0, now - max(first_ts, since))
        rate_per_second = report.total_cost / observed_seconds
        rate_per_day = rate_per_second * _DAY
        projected = rate_per_second * horizon_seconds

        if observed_seconds < _MIN_CONFIDENT_OBSERVATION_SECONDS:
            confidence = "low"
        elif observed_seconds < 3 * _DAY:
            confidence = "medium"
        else:
            confidence = "high"

        return Forecast(
            projected_cost=projected,
            horizon_seconds=horizon_seconds,
            observed_seconds=observed_seconds,
            observed_cost=report.total_cost,
            rate_per_day=rate_per_day,
            confidence=confidence,
            currency=report.currency,
            basis=(
                f"Linear extrapolation of {report.total_cost:.6f} {report.currency} observed over "
                f"{observed_seconds / 3600:.1f}h ({math.ceil(observed_seconds / _DAY)}d of history)."
            ),
        )


__all__ = ["FEATURES", "STATEMENT_FORMAT", "UNITS", "CostCalculator", "CostLine", "CostReport",
           "Forecast", "feature_of"]
