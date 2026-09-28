"""t1api.routing — quota-aware model routing and the quota-cascade engine.

Beta 1 shipped the pieces this depends on: the model registry
(``t1api.registry``) and per-key/per-model usage metering
(``t1api.usage``). Beta 2 wires them together into the router described
in the spec's "MODEL ROUTING" / "QUOTA CASCADE" sections:

* The router only ever selects models present in the registry — it calls
  ``ModelRegistry.require()``/``.get()``, never trusts a bare string.
* A model's allowance is absolute: hitting either the input or output cap
  fully exhausts it for the reset period (enforced already by
  ``UsageMeter`` — the router reads that, doesn't reimplement it).
* Exhausting one model never touches another model's quota — accounting
  stays per-model (again, already true of ``UsageMeter``).
* Cascades are **plan-scoped configuration**, not hard-coded control flow.
  A :class:`RoutingPolicy` is a named, ordered list of
  :class:`CascadeStep` entries; :class:`RoutingTable` maps plan names to
  policies. Change the thresholds/order by editing data
  (``t1api/data/routing_policies.example.json`` or your own file), not by
  touching ``RoutingEngine``.
* Manual model selection is *never* silently substituted — selecting an
  exhausted model raises ``MODEL_QUOTA_EXHAUSTED`` unless the caller opts
  into automatic fallback (``automatic=True``), matching "Do not silently
  substitute a different model unless the user has enabled automatic
  fallback."
* Once every model in the applicable cascade is exhausted, routing stops
  and raises ``ROUTING_EXHAUSTED`` — there is no silent no-op fallback to
  "just pick something."

This module is pure Python + stdlib, same as the rest of the Beta 1 core
— no FastAPI/Pydantic dependency, fully testable on its own.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import T1APIError, T1ErrorCode
from .registry import ModelRegistry
from .usage import UsageMeter

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_EXAMPLE_POLICY_PATH = _DATA_DIR / "routing_policies.example.json"


@dataclass
class CascadeStep:
    """One entry in a routing cascade.

    ``max_input_tokens`` implements the spec's free-tier example ("if the
    prompt is <= 5,000 input tokens, route to N^3 ... if > 5,000, route to
    miniLite"): a step only matches if the request's input token count is
    within this bound (``None`` = no bound, matches any size). Steps are
    tried in list order; the router uses the first step whose model is
    both size-eligible and not exhausted.
    """

    model_id: str
    max_input_tokens: int | None = None
    note: str = ""

    def matches_size(self, input_tokens: int) -> bool:
        return self.max_input_tokens is None or input_tokens <= self.max_input_tokens


@dataclass
class RoutingPolicy:
    """A named, ordered cascade — e.g. "free_tier_default" or
    "paired_plan_default" from the spec's own examples."""

    name: str
    steps: list[CascadeStep]
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoutingPolicy:
        steps = [
            CascadeStep(
                model_id=s["model_id"],
                max_input_tokens=s.get("max_input_tokens"),
                note=s.get("note", ""),
            )
            for s in d.get("steps", [])
        ]
        return cls(name=d["name"], steps=steps, description=d.get("description", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [
                {"model_id": s.model_id, "max_input_tokens": s.max_input_tokens, "note": s.note}
                for s in self.steps
            ],
        }


class RoutingTable:
    """Maps plan names to :class:`RoutingPolicy`. Thread-safe, backed by a
    JSON file — same load-from-data-not-code pattern as
    :class:`t1api.registry.ModelRegistry`.

    A plan with no explicit policy falls back to ``default_policy`` (if
    set) rather than failing — most deployments will have one "default"
    cascade and a handful of plan-specific overrides (matching the spec's
    "PAIRED-PLAN ROUTING" section, where only the paired plan differs from
    the free-tier default).
    """

    def __init__(
        self,
        policies: dict[str, RoutingPolicy] | None = None,
        *,
        plan_to_policy: dict[str, str] | None = None,
        default_policy: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._policies: dict[str, RoutingPolicy] = dict(policies or {})
        self._plan_to_policy: dict[str, str] = dict(plan_to_policy or {})
        self._default_policy = default_policy

    @classmethod
    def load(cls, path: str | Path | None = None) -> RoutingTable:
        src = Path(path) if path is not None else _EXAMPLE_POLICY_PATH
        if not src.exists():
            logger.warning("t1api.routing: %s not found, starting with an empty routing table", src)
            return cls()
        raw = json.loads(src.read_text(encoding="utf-8"))
        policies = {p["name"]: RoutingPolicy.from_dict(p) for p in raw.get("policies", [])}
        plan_to_policy = dict(raw.get("plan_to_policy", {}))
        default_policy = raw.get("default_policy")
        logger.info("t1api.routing: loaded %d polic(y/ies) from %s", len(policies), src)
        return cls(policies, plan_to_policy=plan_to_policy, default_policy=default_policy)

    def register_policy(self, policy: RoutingPolicy) -> None:
        with self._lock:
            self._policies[policy.name] = policy

    def set_plan_policy(self, plan: str, policy_name: str) -> None:
        with self._lock:
            self._plan_to_policy[plan] = policy_name

    def policy_for_plan(self, plan: str) -> RoutingPolicy | None:
        with self._lock:
            name = self._plan_to_policy.get(plan, self._default_policy)
            if name is None:
                return None
            return self._policies.get(name)

    def get_policy(self, name: str) -> RoutingPolicy | None:
        with self._lock:
            return self._policies.get(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._policies)


@dataclass
class RoutingDecision:
    model_id: str
    reason: str
    cascade_position: int
    policy_name: str
    considered: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "reason": self.reason,
            "cascade_position": self.cascade_position,
            "policy_name": self.policy_name,
            "considered": self.considered,
        }


class RoutingEngine:
    """Ties :class:`RoutingTable` + :class:`ModelRegistry` +
    :class:`UsageMeter` together into the actual routing decision.

    Two entry points:

    * :meth:`route_automatic` — no model_id given by the caller; walk the
      plan's cascade and return the first eligible, non-exhausted model.
    * :meth:`route_manual` — caller named a model_id explicitly. Enforced
      exactly as the spec says: if it's exhausted, raise
      ``MODEL_QUOTA_EXHAUSTED`` unless ``automatic_fallback=True``, in
      which case fall through to the cascade starting *after* the
      requested model's position (if it's in the cascade) or from the top
      (if it isn't).
    """

    def __init__(self, table: RoutingTable, registry: ModelRegistry, meter: UsageMeter) -> None:
        self.table = table
        self.registry = registry
        self.meter = meter

    def _eligible_steps(self, policy: RoutingPolicy, *, input_tokens: int) -> list[CascadeStep]:
        return [s for s in policy.steps if s.matches_size(input_tokens)]

    def route_automatic(
        self, *, key_id: str, plan: str, input_tokens: int = 0
    ) -> RoutingDecision:
        policy = self.table.policy_for_plan(plan)
        if policy is None:
            raise T1APIError(
                T1ErrorCode.NOT_SUPPORTED,
                f"No routing policy configured for plan '{plan}'.",
                details={"plan": plan},
                http_status=501,
            )
        considered: list[dict[str, Any]] = []
        for i, step in enumerate(self._eligible_steps(policy, input_tokens=input_tokens)):
            entry = self.registry.get(step.model_id)
            if entry is None or not entry.is_routable:
                considered.append({"model_id": step.model_id, "skipped": "not registered/routable"})
                continue
            snap = self.meter.snapshot_for_model(key_id, step.model_id)
            considered.append({"model_id": step.model_id, "exhausted": snap.is_exhausted})
            if not snap.is_exhausted:
                return RoutingDecision(
                    model_id=step.model_id,
                    reason=step.note or "next eligible step in cascade",
                    cascade_position=i,
                    policy_name=policy.name,
                    considered=considered,
                )
        raise T1APIError(
            T1ErrorCode.ROUTING_EXHAUSTED,
            f"Every model in policy '{policy.name}' eligible for a {input_tokens}-token "
            "prompt is exhausted for this reset period.",
            details={"policy_name": policy.name, "considered": considered},
            http_status=429,
        )

    def route_manual(
        self,
        *,
        key_id: str,
        plan: str,
        model_id: str,
        input_tokens: int = 0,
        automatic_fallback: bool = False,
    ) -> RoutingDecision:
        # Manual selection must still go through the registry — a
        # client-supplied model_id never bypasses it.
        entry = self.registry.require(model_id)
        if not entry.is_routable:
            raise T1APIError(
                T1ErrorCode.MODEL_NOT_SUPPORTED,
                f"Model '{model_id}' is not routable (disabled, deprecated, or an "
                "example entry).",
                details={"model_id": model_id},
            )
        snap = self.meter.snapshot_for_model(key_id, model_id)
        if not snap.is_exhausted:
            return RoutingDecision(
                model_id=model_id, reason="manual selection", cascade_position=-1, policy_name=""
            )
        if not automatic_fallback:
            raise T1APIError(
                T1ErrorCode.MODEL_QUOTA_EXHAUSTED,
                f"Model '{model_id}' allowance is fully exhausted for this reset period.",
                details=snap.to_dict(),
                http_status=429,
            )
        # Automatic fallback enabled: fall through to the cascade, skipping
        # everything up to and including the position of the requested
        # model if it appears in the policy (so we don't just re-select the
        # same exhausted model), otherwise start from the top.
        policy = self.table.policy_for_plan(plan)
        if policy is None:
            raise T1APIError(
                T1ErrorCode.MODEL_QUOTA_EXHAUSTED,
                f"Model '{model_id}' is exhausted and no routing policy exists for plan "
                f"'{plan}' to fall back through.",
                details=snap.to_dict(),
                http_status=429,
            )
        start_index = 0
        for i, step in enumerate(policy.steps):
            if step.model_id == model_id:
                start_index = i + 1
                break
        considered: list[dict[str, Any]] = [{"model_id": model_id, "exhausted": True}]
        remaining = [s for s in policy.steps[start_index:] if s.matches_size(input_tokens)]
        for i, step in enumerate(remaining):
            entry = self.registry.get(step.model_id)
            if entry is None or not entry.is_routable:
                considered.append({"model_id": step.model_id, "skipped": "not registered/routable"})
                continue
            fallback_snap = self.meter.snapshot_for_model(key_id, step.model_id)
            considered.append({"model_id": step.model_id, "exhausted": fallback_snap.is_exhausted})
            if not fallback_snap.is_exhausted:
                return RoutingDecision(
                    model_id=step.model_id,
                    reason=f"automatic fallback from exhausted '{model_id}'",
                    cascade_position=start_index + i,
                    policy_name=policy.name,
                    considered=considered,
                )
        raise T1APIError(
            T1ErrorCode.ROUTING_EXHAUSTED,
            f"'{model_id}' and every fallback after it in policy '{policy.name}' are exhausted.",
            details={"policy_name": policy.name, "considered": considered},
            http_status=429,
        )


__all__ = [
    "CascadeStep",
    "RoutingPolicy",
    "RoutingTable",
    "RoutingDecision",
    "RoutingEngine",
]
