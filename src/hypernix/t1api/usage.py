"""t1api.usage — Beta 1 "basic usage tracking".

``hypernix.gatekeeper.Gatekeeper`` already tracks per-key request/token
counts and enforces sliding-window quotas — the T1 API integrates with it
rather than re-implementing key-level accounting (see
``t1api.auth.T1AuthService``). What Gatekeeper does *not* do is slice usage
by model, which the spec requires ("Support usage accounting by model").
:class:`UsageMeter` adds that slice on top of :class:`t1api.storage.UsageStore`
and cross-checks against :class:`t1api.registry.ModelRegistry` limits.

Scope note: this module reports usage and tells the caller whether a
model's absolute token cap has been reached (``MODEL_QUOTA_EXHAUSTED``
semantics from the spec). It deliberately does NOT implement the
quota-cascade / automatic-fallback routing engine — that is Beta 2
("QUOTA CASCADE", "MODEL ROUTING" in the spec). Beta 1 only needs the
model's allowance to be reported and enforced as exhausted; deciding what
to do next (fall back, stop) is the router's job.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .errors import T1APIError, T1ErrorCode
from .registry import ModelEntry, ModelRegistry
from .storage import UsageStore

logger = logging.getLogger(__name__)

_DAY_SECONDS = 86400.0


@dataclass
class ModelUsageSnapshot:
    model_id: str
    input_tokens_used: int
    output_tokens_used: int
    requests: int
    input_token_limit: int
    output_token_limit: int

    @property
    def input_remaining(self) -> int:
        return max(0, self.input_token_limit - self.input_tokens_used)

    @property
    def output_remaining(self) -> int:
        return max(0, self.output_token_limit - self.output_tokens_used)

    @property
    def is_exhausted(self) -> bool:
        """A model's allowance is FULLY EXHAUSTED if EITHER the input or
        output cap has been reached — the spec's "either/or, absolute"
        rule, not an average or a combined budget."""
        return self.input_remaining == 0 or self.output_remaining == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "input_tokens_used": self.input_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "requests": self.requests,
            "input_token_limit": self.input_token_limit,
            "output_token_limit": self.output_token_limit,
            "input_remaining": self.input_remaining,
            "output_remaining": self.output_remaining,
            "is_exhausted": self.is_exhausted,
        }


class UsageMeter:
    """Records usage events and answers "how much does this key have left
    for this model" questions against the model registry's absolute caps.

    Reset period: Beta 1 uses a fixed rolling 24h window (``reset_period``
    is a config value, not hard-coded logic — see ``t1api.config``). Plan-
    aware multi-tier reset windows (daily/monthly per plan) are Beta 2/3.
    """

    def __init__(
        self,
        store: UsageStore,
        registry: ModelRegistry,
        *,
        reset_period_seconds: float = _DAY_SECONDS,
    ) -> None:
        self.store = store
        self.registry = registry
        self.reset_period_seconds = reset_period_seconds

    def _since(self) -> float:
        import time

        return time.time() - self.reset_period_seconds

    def record(
        self,
        *,
        key_id: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 1,
        endpoint: str = "",
        server_id: str = "",
        module_id: str = "",
        feature: str = "",
        units: float = 0.0,
        unit: str = "",
    ) -> None:
        # Always validate against the registry first — never record usage
        # against a model that isn't explicitly registered.
        self.registry.require(model_id)
        self.store.record(
            key_id=key_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requests=requests,
            endpoint=endpoint,
            server_id=server_id,
            module_id=module_id,
            feature=feature,
            units=units,
            unit=unit,
        )

    def record_feature(
        self,
        *,
        key_id: str,
        feature: str,
        endpoint: str = "",
        model_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        unit: str = "",
        units: float = 0.0,
        requests: int = 1,
    ) -> None:
        """Record use of a feature that may have no registered model: a
        sandbox run (``unit="code_second"``), a web search
        (``unit="web_search"``), a compaction by whatever model the
        server had loaded.

        A model that is not registered is recorded as no model: the
        tokens still show on the statement, as unpriced, rather than
        being priced from a registry entry that does not exist or being
        dropped.
        """
        if model_id and self.registry.get(model_id) is None:
            model_id = ""
        self.store.record(
            key_id=key_id, model_id=model_id, requests=requests, endpoint=endpoint,
            input_tokens=max(0, int(input_tokens)), output_tokens=max(0, int(output_tokens)),
            feature=feature, units=max(0.0, float(units)), unit=unit,
        )

    def record_quietly(self, **kwargs: Any) -> None:
        """:meth:`record_feature`, logging rather than raising: a feature
        that already ran must not fail because its accounting did."""
        try:
            self.record_feature(**kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("t1api.usage: could not record %s usage", kwargs.get("feature"))

    def snapshot_for_model(self, key_id: str, model_id: str) -> ModelUsageSnapshot:
        entry: ModelEntry = self.registry.require(model_id)
        totals = self.store.totals_for_key(key_id, since=self._since(), model_id=model_id)
        return ModelUsageSnapshot(
            model_id=model_id,
            input_tokens_used=totals["input_tokens"],
            output_tokens_used=totals["output_tokens"],
            requests=totals["requests"],
            input_token_limit=entry.input_token_limit,
            output_token_limit=entry.output_token_limit,
        )

    def assert_not_exhausted(self, key_id: str, model_id: str) -> ModelUsageSnapshot:
        """Raise ``MODEL_QUOTA_EXHAUSTED`` if this key has hit *model_id*'s
        absolute cap for the current reset window."""
        snap = self.snapshot_for_model(key_id, model_id)
        if snap.is_exhausted:
            raise T1APIError(
                T1ErrorCode.MODEL_QUOTA_EXHAUSTED,
                f"Model '{model_id}' allowance is fully exhausted for this reset period.",
                details=snap.to_dict(),
                http_status=429,
            )
        return snap

    def current(self, key_id: str) -> dict[str, Any]:
        """All-time + current-window totals across every model, for
        GET /usage/current."""
        all_time = self.store.totals_for_key(key_id)
        window = self.store.totals_for_key(key_id, since=self._since())
        by_model = self.store.breakdown_by_model(key_id, since=self._since())
        return {
            "key_id": key_id,
            "window_seconds": self.reset_period_seconds,
            "current_window": window,
            "all_time": all_time,
            "by_model": by_model,
        }

    def remaining(self, key_id: str, model_id: str) -> dict[str, Any]:
        return self.assert_not_exhausted_safe(key_id, model_id).to_dict()

    def assert_not_exhausted_safe(self, key_id: str, model_id: str) -> ModelUsageSnapshot:
        """Like :meth:`snapshot_for_model` but does not raise — used by
        read-only endpoints (GET /usage/remaining) that should report
        exhaustion rather than error on it."""
        return self.snapshot_for_model(key_id, model_id)


__all__ = ["ModelUsageSnapshot", "UsageMeter"]
