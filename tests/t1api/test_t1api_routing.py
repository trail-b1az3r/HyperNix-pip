"""Tests for hypernix.t1api.routing — the quota-cascade / model-routing
engine. Every test here traces directly to a worked example in the T1 API
spec's "FREE MODEL ROUTING EXAMPLE" and "PAIRED-PLAN ROUTING" sections.
"""
from __future__ import annotations

import pytest

from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.registry import ModelRegistry, ModelStatus
from hypernix.t1api.routing import RoutingEngine, RoutingTable
from hypernix.t1api.storage import UsageStore
from hypernix.t1api.usage import UsageMeter

_FREE_TIER_MODELS = [
    "nanonix-mini-plus",
    "nanonanonano-n3",
    "nanonix-mini-lite",
    "nanonix-mini",
    "nanonix-nano",
    "nanonanonanonano-n4",
]


@pytest.fixture
def registry() -> ModelRegistry:
    """Example registry with the free-tier cascade's models promoted out
    of `status=example` — simulating "real registry data swapped in",
    since example entries are deliberately not routable (see
    t1api.registry.ModelEntry.is_routable)."""
    reg = ModelRegistry.load(include_examples=True)
    for model_id in _FREE_TIER_MODELS:
        entry = reg.require(model_id)
        entry.status = ModelStatus.AVAILABLE
        entry.is_example_entry = False
        reg.register(entry)
    return reg


@pytest.fixture
def meter(registry: ModelRegistry, tmp_path) -> UsageMeter:
    store = UsageStore(tmp_path / "usage.sqlite3")
    return UsageMeter(store, registry)


@pytest.fixture
def table() -> RoutingTable:
    return RoutingTable.load()


@pytest.fixture
def engine(table: RoutingTable, registry: ModelRegistry, meter: UsageMeter) -> RoutingEngine:
    return RoutingEngine(table, registry, meter)


def _exhaust(meter: UsageMeter, registry: ModelRegistry, key_id: str, model_id: str) -> None:
    entry = registry.require(model_id)
    meter.record(key_id=key_id, model_id=model_id, input_tokens=entry.input_token_limit)


class TestRoutingTable:
    def test_loads_example_policies(self, table: RoutingTable):
        assert len(table) == 2

    def test_free_plan_resolves_to_free_tier_default(self, table: RoutingTable):
        policy = table.policy_for_plan("free")
        assert policy is not None
        assert policy.name == "free_tier_default"

    def test_paired_plan_resolves_to_paired_plan_default(self, table: RoutingTable):
        policy = table.policy_for_plan("paired")
        assert policy.name == "paired_plan_default"

    def test_unknown_plan_falls_back_to_default_policy(self, table: RoutingTable):
        policy = table.policy_for_plan("some-plan-nobody-configured")
        assert policy is not None
        assert policy.name == "free_tier_default"  # the table's default_policy


class TestFreeTeirRoutingExample:
    """Directly reproduces the spec's own "FREE MODEL ROUTING EXAMPLE"."""

    def test_fresh_key_routes_to_primary(self, engine: RoutingEngine):
        d = engine.route_automatic(key_id="k1", plan="free", input_tokens=2000)
        assert d.model_id == "nanonix-mini-plus"

    def test_short_prompt_after_primary_exhausted_routes_to_n3(self, engine, meter, registry):
        _exhaust(meter, registry, "k2", "nanonix-mini-plus")
        d = engine.route_automatic(key_id="k2", plan="free", input_tokens=2000)
        assert d.model_id == "nanonanonano-n3"

    def test_long_prompt_after_primary_exhausted_skips_n3_routes_to_minilite(self, engine, meter, registry):
        _exhaust(meter, registry, "k2", "nanonix-mini-plus")
        d = engine.route_automatic(key_id="k2", plan="free", input_tokens=9000)
        assert d.model_id == "nanonix-mini-lite"

    def test_after_primary_and_size_tier_exhausted_routes_to_nano(self, engine, meter, registry):
        for m in ["nanonix-mini-plus", "nanonanonano-n3", "nanonix-mini-lite"]:
            _exhaust(meter, registry, "k3", m)
        d = engine.route_automatic(key_id="k3", plan="free", input_tokens=2000)
        assert d.model_id == "nanonix-nano"

    def test_after_nano_exhausted_routes_to_n4_last_resort(self, engine, meter, registry):
        for m in ["nanonix-mini-plus", "nanonanonano-n3", "nanonix-mini-lite", "nanonix-nano"]:
            _exhaust(meter, registry, "k4", m)
        d = engine.route_automatic(key_id="k4", plan="free", input_tokens=2000)
        assert d.model_id == "nanonanonanonano-n4"

    def test_everything_exhausted_raises_routing_exhausted(self, engine, meter, registry):
        for m in _FREE_TIER_MODELS:
            _exhaust(meter, registry, "k5", m)
        with pytest.raises(T1APIError) as exc_info:
            engine.route_automatic(key_id="k5", plan="free", input_tokens=2000)
        assert exc_info.value.code == T1ErrorCode.ROUTING_EXHAUSTED

    def test_different_keys_have_independent_cascades(self, engine, meter, registry):
        for m in _FREE_TIER_MODELS:
            _exhaust(meter, registry, "k5", m)
        # A completely fresh key is unaffected by k5's exhaustion.
        d = engine.route_automatic(key_id="k_fresh", plan="free", input_tokens=100)
        assert d.model_id == "nanonix-mini-plus"


class TestPairedPlanRoutingExample:
    """Directly reproduces the spec's "PAIRED-PLAN ROUTING" example: after
    N^3 usage is exhausted, a paired plan falls back to nanonix-mini —
    NOT nanonix-mini-lite (the free-tier fallback)."""

    def test_paired_plan_falls_back_to_mini_not_minilite(self, engine, meter, registry):
        for m in ["nanonix-mini-plus", "nanonanonano-n3"]:
            _exhaust(meter, registry, "k6", m)
        d = engine.route_automatic(key_id="k6", plan="paired", input_tokens=2000)
        assert d.model_id == "nanonix-mini"
        assert d.model_id != "nanonix-mini-lite"

    def test_free_plan_in_same_situation_uses_minilite_not_mini(self, engine, meter, registry):
        """Sanity check the plans really are independent: the same
        exhaustion state under the free plan takes the OTHER branch."""
        for m in ["nanonix-mini-plus", "nanonanonano-n3"]:
            _exhaust(meter, registry, "k6b", m)
        d = engine.route_automatic(key_id="k6b", plan="free", input_tokens=9000)
        assert d.model_id == "nanonix-mini-lite"


class TestManualSelection:
    def test_manual_selection_of_available_model_succeeds(self, engine: RoutingEngine):
        d = engine.route_manual(key_id="k1", plan="free", model_id="nanonix-nano")
        assert d.model_id == "nanonix-nano"
        assert d.reason == "manual selection"

    def test_manual_selection_of_exhausted_model_raises_without_auto_fallback(self, engine, meter, registry):
        _exhaust(meter, registry, "k5", "nanonix-mini-plus")
        with pytest.raises(T1APIError) as exc_info:
            engine.route_manual(key_id="k5", plan="free", model_id="nanonix-mini-plus")
        assert exc_info.value.code == T1ErrorCode.MODEL_QUOTA_EXHAUSTED

    def test_manual_selection_never_silently_substitutes(self, engine, meter, registry):
        """The spec: 'Do not silently substitute a different model unless
        the user has enabled automatic fallback.' Confirm the raised error
        names the model that was actually requested, not a substitute."""
        _exhaust(meter, registry, "k5", "nanonix-mini-plus")
        with pytest.raises(T1APIError) as exc_info:
            engine.route_manual(key_id="k5", plan="free", model_id="nanonix-mini-plus")
        assert exc_info.value.details["model_id"] == "nanonix-mini-plus"

    def test_manual_selection_with_auto_fallback_walks_cascade(self, engine, meter, registry):
        for m in ["nanonix-mini-plus", "nanonanonano-n3", "nanonix-mini-lite"]:
            _exhaust(meter, registry, "k3", m)
        d = engine.route_manual(
            key_id="k3", plan="free", model_id="nanonix-mini-plus", input_tokens=2000,
            automatic_fallback=True,
        )
        assert d.model_id == "nanonix-nano"

    def test_manual_selection_of_unregistered_model_raises_model_not_supported(self, engine: RoutingEngine):
        with pytest.raises(T1APIError) as exc_info:
            engine.route_manual(key_id="k1", plan="free", model_id="totally-fake-model")
        assert exc_info.value.code == T1ErrorCode.MODEL_NOT_SUPPORTED

    def test_manual_fallback_cannot_bypass_registry_either(self, engine, meter, registry):
        """automatic_fallback=True still requires the manually-selected
        starting point to be a real registered model."""
        with pytest.raises(T1APIError) as exc_info:
            engine.route_manual(
                key_id="k1", plan="free", model_id="../../etc/passwd", automatic_fallback=True
            )
        assert exc_info.value.code == T1ErrorCode.MODEL_NOT_SUPPORTED


class TestSizeBasedRoutingBoundary:
    def test_exactly_5000_tokens_is_still_eligible_for_n3(self, engine, meter, registry):
        _exhaust(meter, registry, "k2", "nanonix-mini-plus")
        d = engine.route_automatic(key_id="k2", plan="free", input_tokens=5000)
        assert d.model_id == "nanonanonano-n3"  # <=5000 per the spec, boundary inclusive

    def test_5001_tokens_skips_n3(self, engine, meter, registry):
        _exhaust(meter, registry, "k2", "nanonix-mini-plus")
        d = engine.route_automatic(key_id="k2", plan="free", input_tokens=5001)
        assert d.model_id == "nanonix-mini-lite"


class TestUnknownPlanWithoutDefault:
    def test_raises_not_supported_when_no_policy_and_no_default(self, registry, meter):
        empty_table = RoutingTable()  # no policies, no default
        engine = RoutingEngine(empty_table, registry, meter)
        with pytest.raises(T1APIError) as exc_info:
            engine.route_automatic(key_id="k1", plan="nonexistent", input_tokens=100)
        assert exc_info.value.code == T1ErrorCode.NOT_SUPPORTED
