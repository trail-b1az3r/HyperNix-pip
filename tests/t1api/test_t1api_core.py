"""Tests for hypernix.t1api's pure-Python core: registry, storage, usage
metering, errors. None of these require fastapi/pydantic — they're the
same "framework-independent enforcement logic" pattern as
hypernix.keymaster / hypernix.gatekeeper, and are tested the same way.
"""
from __future__ import annotations

import pytest

from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.registry import (
    ModelAvailabilityFlag,
    ModelEntry,
    ModelPricing,
    ModelRegistry,
    ModelStatus,
)
from hypernix.t1api.storage import UsageStore
from hypernix.t1api.usage import UsageMeter

# ===========================================================================
# Model Registry
# ===========================================================================


class TestModelRegistryExamples:
    def test_example_seed_hidden_by_default(self):
        reg = ModelRegistry.load()
        assert len(reg) == 0

    def test_example_seed_visible_when_enabled(self):
        reg = ModelRegistry.load(include_examples=True)
        assert len(reg) == 9

    def test_example_entries_flagged(self):
        reg = ModelRegistry.load(include_examples=True)
        for entry in reg.list():
            assert entry.is_example_entry is True
            assert entry.status == ModelStatus.EXAMPLE

    def test_spec_model_ids_present(self):
        reg = ModelRegistry.load(include_examples=True)
        ids = {e.model_id for e in reg.list()}
        assert "nanonix-mini-lite" in ids  # spec's own model_id naming example
        assert "hypernix-1" in ids
        assert "ryiver-1" in ids

    def test_model_id_is_not_a_parameter_count_string(self):
        reg = ModelRegistry.load(include_examples=True)
        for entry in reg.list():
            assert not entry.model_id[0].isdigit()


class TestModelRegistryEnforcement:
    def test_require_unknown_model_raises_model_not_supported(self):
        reg = ModelRegistry()
        with pytest.raises(T1APIError) as exc_info:
            reg.require("totally-made-up-model")
        assert exc_info.value.code == T1ErrorCode.MODEL_NOT_SUPPORTED

    def test_require_known_model_returns_entry(self):
        reg = ModelRegistry.load(include_examples=True)
        entry = reg.require("nanonix-nano")
        assert entry.model_id == "nanonix-nano"

    def test_client_supplied_path_cannot_bypass_registry(self):
        """A model_id that merely LOOKS plausible but was never registered
        must still be rejected — the registry is the only source of truth."""
        reg = ModelRegistry.load(include_examples=True)
        with pytest.raises(T1APIError):
            reg.require("../../etc/some-arbitrary-backend-path")

    def test_get_returns_none_not_raise(self):
        reg = ModelRegistry()
        assert reg.get("nope") is None

    def test_register_then_require_succeeds(self):
        reg = ModelRegistry()
        entry = ModelEntry(
            model_id="test-model",
            display_name="Test Model",
            version="1.0",
            total_parameters=1.0,
            active_parameters=None,
            architecture="dense",
            supported_tasks=["chat"],
            availability=ModelAvailabilityFlag.PUBLIC,
            minimum_plan="free",
            free_tier_available=True,
            api_available=True,
            local_available=True,
            remote_available=False,
            context_limit=4096,
            input_token_limit=4096,
            output_token_limit=1024,
            tool_call_limit=None,
            pricing=ModelPricing(),
            routing_priority=1,
            fallback_model=None,
            license="mit",
            status=ModelStatus.AVAILABLE,
        )
        reg.register(entry)
        assert reg.require("test-model").display_name == "Test Model"

    def test_deregister_removes_entry(self):
        reg = ModelRegistry.load(include_examples=True)
        reg.deregister("nanonix-nano")
        with pytest.raises(T1APIError):
            reg.require("nanonix-nano")

    def test_disabled_status_excluded_from_routable_list(self):
        reg = ModelRegistry.load(include_examples=True)
        entry = reg.get("nanonix-nano")
        entry.status = ModelStatus.DISABLED
        reg.register(entry)
        routable_ids = {e.model_id for e in reg.list(routable_only=True)}
        assert "nanonix-nano" not in routable_ids
        # still directly gettable, just not "routable"
        assert reg.require("nanonix-nano").status == ModelStatus.DISABLED


# ===========================================================================
# Usage store / meter
# ===========================================================================


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load(include_examples=True)


@pytest.fixture
def store(tmp_path) -> UsageStore:
    return UsageStore(tmp_path / "usage.sqlite3")


@pytest.fixture
def meter(store: UsageStore, registry: ModelRegistry) -> UsageMeter:
    return UsageMeter(store, registry)


class TestUsageStore:
    def test_record_and_totals(self, store: UsageStore):
        store.record(key_id="k1", model_id="m1", input_tokens=10, output_tokens=5)
        store.record(key_id="k1", model_id="m1", input_tokens=20, output_tokens=15)
        totals = store.totals_for_key("k1")
        assert totals["requests"] == 2
        assert totals["input_tokens"] == 30
        assert totals["output_tokens"] == 20
        assert totals["total_tokens"] == 50

    def test_totals_scoped_by_key(self, store: UsageStore):
        store.record(key_id="k1", model_id="m1", input_tokens=10)
        store.record(key_id="k2", model_id="m1", input_tokens=999)
        assert store.totals_for_key("k1")["input_tokens"] == 10

    def test_breakdown_by_model(self, store: UsageStore):
        store.record(key_id="k1", model_id="m1", input_tokens=10)
        store.record(key_id="k1", model_id="m2", input_tokens=20)
        breakdown = {row["model_id"]: row for row in store.breakdown_by_model("k1")}
        assert breakdown["m1"]["input_tokens"] == 10
        assert breakdown["m2"]["input_tokens"] == 20

    def test_since_filter_excludes_old_events(self, store: UsageStore):
        import time

        store.record(key_id="k1", model_id="m1", input_tokens=5, ts=time.time() - 1000)
        totals = store.totals_for_key("k1", since=time.time() - 10)
        assert totals["input_tokens"] == 0


class TestUsageMeter:
    def test_record_validates_against_registry(self, meter: UsageMeter):
        with pytest.raises(T1APIError) as exc_info:
            meter.record(key_id="k1", model_id="not-registered", input_tokens=1)
        assert exc_info.value.code == T1ErrorCode.MODEL_NOT_SUPPORTED

    def test_snapshot_reflects_recorded_usage(self, meter: UsageMeter):
        meter.record(key_id="k1", model_id="nanonix-nano", input_tokens=100, output_tokens=50)
        snap = meter.snapshot_for_model("k1", "nanonix-nano")
        assert snap.input_tokens_used == 100
        assert snap.output_tokens_used == 50
        assert snap.is_exhausted is False

    def test_input_cap_exhausts_model_allowance(self, meter: UsageMeter, registry: ModelRegistry):
        entry = registry.require("nanonix-nano")
        meter.record(key_id="k1", model_id="nanonix-nano", input_tokens=entry.input_token_limit)
        snap = meter.snapshot_for_model("k1", "nanonix-nano")
        assert snap.is_exhausted is True
        assert snap.input_remaining == 0

    def test_output_cap_alone_also_exhausts(self, meter: UsageMeter, registry: ModelRegistry):
        """Either cap reaching its limit exhausts the model — not just input."""
        entry = registry.require("nanonix-nano")
        meter.record(key_id="k1", model_id="nanonix-nano", output_tokens=entry.output_token_limit)
        snap = meter.snapshot_for_model("k1", "nanonix-nano")
        assert snap.is_exhausted is True
        assert snap.output_remaining == 0
        assert snap.input_remaining > 0  # input cap untouched — independent accounting

    def test_exhausting_one_model_does_not_touch_another(self, meter: UsageMeter, registry: ModelRegistry):
        entry = registry.require("nanonix-nano")
        meter.record(key_id="k1", model_id="nanonix-nano", input_tokens=entry.input_token_limit)
        other_snap = meter.snapshot_for_model("k1", "nanonix-mini-plus")
        assert other_snap.is_exhausted is False
        assert other_snap.input_tokens_used == 0

    def test_assert_not_exhausted_raises_model_quota_exhausted(self, meter: UsageMeter, registry: ModelRegistry):
        entry = registry.require("nanonix-nano")
        meter.record(key_id="k1", model_id="nanonix-nano", input_tokens=entry.input_token_limit)
        with pytest.raises(T1APIError) as exc_info:
            meter.assert_not_exhausted("k1", "nanonix-nano")
        assert exc_info.value.code == T1ErrorCode.MODEL_QUOTA_EXHAUSTED

    def test_current_aggregates_across_models(self, meter: UsageMeter):
        meter.record(key_id="k1", model_id="nanonix-nano", input_tokens=10)
        meter.record(key_id="k1", model_id="nanonix-mini-plus", input_tokens=20)
        current = meter.current("k1")
        assert current["all_time"]["input_tokens"] == 30
        assert len(current["by_model"]) == 2

    def test_different_keys_independent_usage(self, meter: UsageMeter):
        meter.record(key_id="k1", model_id="nanonix-nano", input_tokens=10)
        meter.record(key_id="k2", model_id="nanonix-nano", input_tokens=999)
        assert meter.snapshot_for_model("k1", "nanonix-nano").input_tokens_used == 10
