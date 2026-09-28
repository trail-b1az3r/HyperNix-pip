"""Regression tests for v0.52.5 — forgiving smoke_alarm kwargs.

Reported by a downstream chat_hypernix2.py running on an i7 7th gen
Surface Pro::

    TypeError: GasAlarm.__init__() got an unexpected keyword
    argument 'cpu_preset'

…and after fallback to RadsAlarm::

    TypeError: Alarm.__init__() got an unexpected keyword
    argument 'max_steps'

The fix adds three forgiving kwargs to the base ``Alarm`` dataclass
(and propagates them through ``AutoAlarm._common_kwargs``), plus a
generational-family alias map in ``hypernix.freezer.cpu_preset`` so
the user's exact ``"i7_7th_gen"`` string resolves to a real CPU
preset (``i7-7700hq``).
"""
from __future__ import annotations

import pytest

from hypernix import smoke_alarm
from hypernix.freezer import cpu_preset

# ---------------------------------------------------------------------------
# The exact two repro lines from the user's screenshot
# ---------------------------------------------------------------------------

class TestUserRepro:
    def test_gas_alarm_accepts_cpu_preset_kwarg(self) -> None:
        # The exact downstream call.
        a = smoke_alarm.GasAlarm(
            time_budget_seconds=600,
            cpu_preset="i7_7th_gen",
        )
        assert a.cpu is not None
        assert "i7-7700hq" in a.cpu.name.lower()

    def test_rads_alarm_accepts_max_steps_kwarg(self) -> None:
        a = smoke_alarm.RadsAlarm(
            time_budget_seconds=10_000,
            max_steps=100,
        )
        assert a.recommended_steps() == 100


# ---------------------------------------------------------------------------
# max_steps semantics (the cap)
# ---------------------------------------------------------------------------

class TestMaxStepsCap:
    def test_max_steps_caps_recommended_when_smaller(self) -> None:
        a = smoke_alarm.RadsAlarm(time_budget_seconds=10_000, max_steps=42)
        assert a.recommended_steps() == 42

    def test_max_steps_does_not_inflate_recommended_when_larger(self) -> None:
        # If the natural recommendation is below max_steps, max_steps
        # is a no-op (it's a CAP, not a target).
        a = smoke_alarm.RadsAlarm(time_budget_seconds=1, max_steps=10_000)
        assert a.recommended_steps() <= 10_000

    def test_max_steps_zero_or_none_is_ignored(self) -> None:
        a = smoke_alarm.RadsAlarm(time_budget_seconds=10_000, max_steps=None)
        b = smoke_alarm.RadsAlarm(time_budget_seconds=10_000, max_steps=0)
        # Neither caps the natural value.
        assert a.recommended_steps() > 100
        assert b.recommended_steps() > 100


# ---------------------------------------------------------------------------
# cpu_preset / gpu_preset resolution on GasAlarm
# ---------------------------------------------------------------------------

class TestGasAlarmPresetResolution:
    def test_cpu_preset_string_resolves_to_cpu_object(self) -> None:
        a = smoke_alarm.GasAlarm(time_budget_seconds=600, cpu_preset="i7-7700hq")
        assert a.cpu is not None
        assert "7700HQ" in a.cpu.name

    def test_cpu_preset_object_passes_through(self) -> None:
        cpu = cpu_preset("i7-12700h")
        a = smoke_alarm.GasAlarm(time_budget_seconds=600, cpu_preset=cpu)
        assert a.cpu is cpu

    def test_gpu_preset_string_resolves_to_gpu_object(self) -> None:
        a = smoke_alarm.GasAlarm(time_budget_seconds=600, gpu_preset="rtx-3080")
        assert a.gpu is not None
        assert "3080" in a.gpu.name

    def test_explicit_cpu_takes_precedence_over_cpu_preset(self) -> None:
        cpu = cpu_preset("i9-13900k")
        a = smoke_alarm.GasAlarm(
            time_budget_seconds=600,
            cpu=cpu,
            cpu_preset="i7-7700hq",  # would resolve to a different cpu
        )
        assert a.cpu is cpu


# ---------------------------------------------------------------------------
# Generational CPU aliases (the actual root cause of the user's bug)
# ---------------------------------------------------------------------------

class TestGenerationalAliases:
    @pytest.mark.parametrize(
        "alias,expected_substr",
        [
            ("i7_7th_gen",  "7700HQ"),
            ("i7-7th-gen",  "7700HQ"),
            ("i7-12th-gen", "12700H"),
            ("i7-13th-gen", "13700H"),
            ("i9-12th-gen", "12900K"),
            ("i9-14th-gen", "14900K"),
            ("ultra-7",     "Ultra 7 155H"),
            ("ultra-9",     "Ultra 9 185H"),
            ("core-ultra",  "Ultra 7 155H"),
        ],
    )
    def test_generational_alias_resolves(self, alias: str, expected_substr: str) -> None:
        p = cpu_preset(alias)
        assert p is not None, f"alias {alias!r} did not resolve"
        assert expected_substr.lower() in p.name.lower()

    def test_unknown_alias_still_returns_none(self) -> None:
        assert cpu_preset("definitely-not-a-cpu") is None


# ---------------------------------------------------------------------------
# AutoAlarm propagates the new kwargs
# ---------------------------------------------------------------------------

class TestAutoAlarmForwarding:
    def test_auto_alarm_accepts_cpu_preset_alias(self) -> None:
        chosen = smoke_alarm.AutoAlarm(
            time_budget_seconds=600,
            cpu_preset="i7_7th_gen",
        ).pick()
        # cpu_preset alias should be treated as cpu_name and route to GasAlarm.
        assert isinstance(chosen, smoke_alarm.GasAlarm)
        assert chosen.cpu is not None

    def test_auto_alarm_propagates_max_steps_to_picked_alarm(self) -> None:
        chosen = smoke_alarm.AutoAlarm(
            time_budget_seconds=10_000,
            max_steps=77,
        ).pick()
        assert chosen.max_steps == 77
        assert chosen.recommended_steps() == 77


# ---------------------------------------------------------------------------
# All four alarm tiers accept the new kwargs without TypeError
# ---------------------------------------------------------------------------

class TestKwargAcceptanceOnEveryTier:
    @pytest.mark.parametrize(
        "ctor",
        [
            smoke_alarm.RadsAlarm,
            smoke_alarm.GasAlarm,
            smoke_alarm.ModernAlarm,
        ],
    )
    def test_every_tier_accepts_max_steps(self, ctor) -> None:
        a = ctor(time_budget_seconds=600, max_steps=10)
        assert a.max_steps == 10

    @pytest.mark.parametrize(
        "ctor",
        [
            smoke_alarm.RadsAlarm,
            smoke_alarm.GasAlarm,
            smoke_alarm.ModernAlarm,
        ],
    )
    def test_every_tier_accepts_cpu_preset(self, ctor) -> None:
        # RadsAlarm / ModernAlarm don't *use* it, but accepting it
        # silently is the whole point.
        a = ctor(time_budget_seconds=600, cpu_preset="i7-7700hq")
        assert a.cpu_preset is not None or hasattr(a, "cpu")
