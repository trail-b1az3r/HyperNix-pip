"""Regression tests for v0.52.6 — more forgiving smoke_alarm kwargs.

Reported (continuation of the i7-7th-gen Surface Pro thread) on the
0.52.5 wheel::

    TypeError: GasAlarm.__init__() missing 1 required positional
    argument: 'time_budget_seconds'

…and after fallback to RadsAlarm::

    TypeError: Alarm.__init__() got an unexpected keyword argument
    'log_every'

Fix: ``time_budget_seconds`` now defaults to 600.0, and the base
``Alarm`` accepts ``log_every`` / ``save_every`` / ``eval_every``
as additional ignored fields so downstream training-config scripts
can ``**spread`` config dicts straight into the constructor.
"""
from __future__ import annotations

import pytest

from hypernix import smoke_alarm

# ---------------------------------------------------------------------------
# The exact two repro lines from the user's screenshot
# ---------------------------------------------------------------------------

class TestUserRepro:
    def test_gas_alarm_works_without_time_budget_seconds(self) -> None:
        # The exact downstream call.
        a = smoke_alarm.GasAlarm(cpu_preset="i7_7th_gen")
        assert a.time_budget_seconds == 600.0  # default
        assert a.cpu is not None
        assert "7700HQ" in a.cpu.name

    def test_rads_alarm_accepts_log_every(self) -> None:
        a = smoke_alarm.RadsAlarm(log_every=10)
        assert a.log_every == 10


# ---------------------------------------------------------------------------
# Default time budget on every tier
# ---------------------------------------------------------------------------

class TestDefaultTimeBudget:
    @pytest.mark.parametrize(
        "ctor",
        [
            smoke_alarm.RadsAlarm,
            smoke_alarm.GasAlarm,
            smoke_alarm.ModernAlarm,
            smoke_alarm.AutoAlarm,
        ],
    )
    def test_no_args_constructor_uses_default_time_budget(self, ctor) -> None:
        a = ctor()
        assert a.time_budget_seconds == 600.0

    def test_explicit_time_budget_still_works(self) -> None:
        a = smoke_alarm.GasAlarm(time_budget_seconds=120.0, cpu_preset="i7-7700hq")
        assert a.time_budget_seconds == 120.0


# ---------------------------------------------------------------------------
# log_every / save_every / eval_every accepted everywhere
# ---------------------------------------------------------------------------

class TestLoggingCadenceKwargs:
    @pytest.mark.parametrize(
        "ctor",
        [
            smoke_alarm.RadsAlarm,
            smoke_alarm.GasAlarm,
            smoke_alarm.ModernAlarm,
        ],
    )
    def test_every_tier_accepts_log_every(self, ctor) -> None:
        a = ctor(log_every=25)
        assert a.log_every == 25

    @pytest.mark.parametrize(
        "ctor",
        [
            smoke_alarm.RadsAlarm,
            smoke_alarm.GasAlarm,
            smoke_alarm.ModernAlarm,
        ],
    )
    def test_every_tier_accepts_save_every(self, ctor) -> None:
        a = ctor(save_every=500)
        assert a.save_every == 500

    @pytest.mark.parametrize(
        "ctor",
        [
            smoke_alarm.RadsAlarm,
            smoke_alarm.GasAlarm,
            smoke_alarm.ModernAlarm,
        ],
    )
    def test_every_tier_accepts_eval_every(self, ctor) -> None:
        a = ctor(eval_every=100)
        assert a.eval_every == 100

    def test_default_logging_cadence_is_none(self) -> None:
        a = smoke_alarm.RadsAlarm()
        assert a.log_every is None
        assert a.save_every is None
        assert a.eval_every is None


# ---------------------------------------------------------------------------
# AutoAlarm forwards everything to the picked tier
# ---------------------------------------------------------------------------

class TestAutoAlarmForwarding:
    def test_auto_alarm_no_args_uses_default_budget(self) -> None:
        chosen = smoke_alarm.AutoAlarm().pick()
        assert chosen.time_budget_seconds == 600.0

    def test_auto_alarm_propagates_logging_cadence(self) -> None:
        chosen = smoke_alarm.AutoAlarm(
            cpu_preset="i7_7th_gen",
            log_every=20,
            save_every=200,
            eval_every=50,
        ).pick()
        assert chosen.log_every == 20
        assert chosen.save_every == 200
        assert chosen.eval_every == 50

    def test_auto_alarm_combined_user_config_dict(self) -> None:
        # Simulate the realistic call shape: a downstream training
        # config dict gets unpacked straight into the constructor.
        cfg = {
            "cpu_preset": "i7_7th_gen",
            "max_steps": 200,
            "log_every": 10,
            "save_every": 100,
            "eval_every": 50,
        }
        chosen = smoke_alarm.AutoAlarm(**cfg).pick()
        assert isinstance(chosen, smoke_alarm.GasAlarm)
        assert chosen.max_steps == 200
        assert chosen.recommended_steps() <= 200
        assert chosen.log_every == 10
