"""ups — checkpoint-panic guard.

The module had no tests, no CLI, and no console script: every piece of it
worked and there was no way to run any of it. These cover the behaviour
plus the three defects that fell out of reading it — a network call held
under a lock, a background poller with no way to stop, and a "one-shot"
helper that left one running.
"""
from __future__ import annotations

import threading
import time

import pytest

from hypernix.system import ups as ups_mod
from hypernix.system.ups import UPS


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may touch the network.

    Autouse rather than opt-in: a test that accidentally makes a real
    request would be slow, flaky, and dependent on the weather.
    """
    monkeypatch.setattr(ups_mod, "_query_open_meteo", lambda *a, **k: None)
    monkeypatch.setattr(ups_mod, "_autodetect_coords", lambda *a, **k: None)


def _weather(code: int):
    return lambda *a, **k: {"weather_code": code}


class TestThreatClassification:
    def test_severe_code_is_a_panic(self, monkeypatch):
        monkeypatch.setattr(ups_mod, "_query_open_meteo", _weather(95))  # thunderstorm
        status = UPS(latitude=1.0, longitude=1.0).check(force=True)
        assert status.weather_severe is True
        assert status.panic is True
        assert status.active is True

    def test_elevated_code_is_active_but_not_panic(self, monkeypatch):
        monkeypatch.setattr(ups_mod, "_query_open_meteo", _weather(61))  # light rain
        status = UPS(latitude=1.0, longitude=1.0).check(force=True)
        assert status.weather_elevated is True
        assert status.active is True
        assert status.panic is False

    def test_clear_weather_is_quiet(self, monkeypatch):
        monkeypatch.setattr(ups_mod, "_query_open_meteo", _weather(0))
        status = UPS(latitude=1.0, longitude=1.0).check(force=True)
        assert status.active is False
        assert status.panic is False

    def test_a_scheduled_outage_is_a_panic_on_its_own(self):
        status = UPS(offline=True, outage_check_fn=lambda _addr: True).check(force=True)
        assert status.scheduled_outage is True
        assert status.panic is True

    def test_a_raising_outage_callback_does_not_propagate(self):
        """User-supplied code failing must not take down the training
        loop that called check()."""
        def boom(_addr):
            raise RuntimeError("utility website is down")

        status = UPS(offline=True, outage_check_fn=boom).check(force=True)
        assert status.scheduled_outage is False

    def test_offline_skips_the_weather_call(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            ups_mod, "_query_open_meteo",
            lambda *a, **k: called.append(1) or {"weather_code": 95},
        )
        status = UPS(latitude=1.0, longitude=1.0, offline=True).check(force=True)
        assert called == []
        assert status.panic is False


class TestCadence:
    def test_panic_tightens_the_save_cadence(self):
        guard = UPS(offline=True, outage_check_fn=lambda _a: True, cadence_multiplier=3)
        assert guard.adjusted_save_every(600) == 200

    def test_quiet_leaves_it_alone(self):
        guard = UPS(offline=True, cadence_multiplier=3)
        assert guard.adjusted_save_every(600) == 600

    def test_cadence_floors_at_one(self):
        guard = UPS(offline=True, outage_check_fn=lambda _a: True, cadence_multiplier=10)
        assert guard.adjusted_save_every(2) == 1

    def test_multiplier_must_be_at_least_one(self):
        with pytest.raises(ValueError):
            UPS(cadence_multiplier=0)


class TestSnapshotHook:
    def test_fires_once_on_the_transition_into_panic(self):
        fired = threading.Event()
        guard = UPS(
            offline=True,
            outage_check_fn=lambda _a: True,
            snapshot_fn=fired.set,
            check_interval_seconds=0,
        )
        guard.check(force=True)
        assert fired.wait(timeout=5.0), "snapshot_fn was never called"

    def test_does_not_refire_while_the_panic_persists(self):
        calls: list[int] = []
        guard = UPS(
            offline=True,
            outage_check_fn=lambda _a: True,
            snapshot_fn=lambda: calls.append(1),
            check_interval_seconds=0,
        )
        for _ in range(3):
            guard.check(force=True)
        time.sleep(0.3)
        assert len(calls) == 1, f"snapshot fired {len(calls)} times for one panic"

    def test_force_snapshot_is_manual(self):
        calls: list[int] = []
        UPS(offline=True, snapshot_fn=lambda: calls.append(1)).force_snapshot()
        assert calls == [1]

    def test_no_snapshot_fn_is_not_an_error(self):
        UPS(offline=True, outage_check_fn=lambda _a: True).check(force=True)


class TestCachingAndLifecycle:
    def test_repeat_checks_inside_the_interval_reuse_the_result(self, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(
            ups_mod, "_query_open_meteo",
            lambda *a, **k: (calls.append(1), {"weather_code": 0})[1],
        )
        guard = UPS(latitude=1.0, longitude=1.0, check_interval_seconds=3600)
        guard.check(force=True)
        # Measure the delta, not the absolute count: the forced check and
        # the background poller's own first pass are both expected. What
        # matters is that further checks inside the interval add nothing.
        after_first = len(calls)
        for _ in range(5):
            guard.check()
        guard.stop()
        assert len(calls) == after_first

    def test_stop_ends_the_background_poller(self):
        """`_stop_event` existed and `_bg_loop` honoured it, but nothing
        ever set it — there was no way to stop a UPS."""
        guard = UPS(latitude=1.0, longitude=1.0, check_interval_seconds=0.05)
        guard.check(force=True)
        assert guard._bg_thread is not None
        thread = guard._bg_thread
        guard.stop()
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    def test_context_manager_stops_it(self):
        with UPS(latitude=1.0, longitude=1.0, check_interval_seconds=0.05) as guard:
            guard.check(force=True)
            thread = guard._bg_thread
        assert thread is not None
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    def test_threat_now_leaves_no_poller_behind(self):
        """A helper documented as one-shot used to leave a thread polling
        open-meteo every five minutes for the life of the process."""
        before = threading.active_count()
        ups_mod.threat_now(latitude=1.0, longitude=1.0)
        time.sleep(0.2)
        assert threading.active_count() <= before

    def test_the_network_call_is_not_made_under_the_lock(self, monkeypatch):
        """A five-second HTTP timeout held against every other caller
        would serialize a training loop on the weather.

        The probe runs on the same thread as ``check()``. ``_lock`` is a
        plain ``Lock``, not an ``RLock``, so a successful non-blocking
        acquire here can only mean this call is *not* holding it — which is
        the property under test.

        That reasoning only holds while no other thread can touch the lock,
        and ``check()`` starts the background poller, which calls ``check()``
        itself and takes the same lock. Leaving it running turned this into
        a measurement of "was any thread holding the lock at that instant",
        which is a race — it passed locally and failed on CI. The poller is
        irrelevant to the property, so it simply doesn't run here.
        """
        holder: dict[str, bool] = {}

        def probing_query(*_a, **_k):
            holder["lock_free"] = guard._lock.acquire(blocking=False)
            if holder["lock_free"]:
                guard._lock.release()
            return {"weather_code": 0}

        monkeypatch.setattr(ups_mod, "_query_open_meteo", probing_query)
        monkeypatch.setattr(UPS, "_start_bg_thread", lambda self: None)
        guard = UPS(latitude=1.0, longitude=1.0, offline=False)
        guard.check(force=True)
        guard.stop()
        assert holder.get("lock_free") is True, "the weather call ran while holding the lock"
        assert guard._bg_thread is None, "the poller stub did not take effect"

    def test_history_is_bounded(self):
        guard = UPS(offline=True, outage_check_fn=lambda _a: True, check_interval_seconds=0)
        for i in range(ups_mod.MAX_HISTORY + 50):
            guard.check(force=True)
            # Distinct timestamps, otherwise the de-dup check swallows them.
            guard.last_status.timestamp = float(i)
        assert len(guard.history) <= ups_mod.MAX_HISTORY


class TestCLI:
    def test_help(self, capsys):
        assert ups_mod.cli_main(["--help"]) == 0
        assert "usage: ups" in capsys.readouterr().out

    def test_quiet_status_exits_zero(self, capsys):
        assert ups_mod.cli_main(["--offline"]) == 0
        assert "ok" in capsys.readouterr().out

    def test_panic_exits_two(self, monkeypatch, capsys):
        """Distinct exit codes let a script react proportionately."""
        monkeypatch.setattr(ups_mod, "_query_open_meteo", _weather(95))
        assert ups_mod.cli_main(["status", "--lat", "1", "--lon", "1"]) == 2
        assert "PANIC" in capsys.readouterr().out

    def test_elevated_exits_one(self, monkeypatch, capsys):
        monkeypatch.setattr(ups_mod, "_query_open_meteo", _weather(61))
        assert ups_mod.cli_main(["status", "--lat", "1", "--lon", "1"]) == 1
        assert "ALERT" in capsys.readouterr().out

    def test_json_output(self, capsys):
        import json

        assert ups_mod.cli_main(["--offline", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["panic"] is False
        assert "summary" in payload

    def test_lat_without_lon_is_rejected(self, capsys):
        assert ups_mod.cli_main(["--lat", "47.6"]) == 64
        assert "together" in capsys.readouterr().err

    def test_unknown_command(self, capsys):
        assert ups_mod.cli_main(["explode"]) == 64

    def test_non_numeric_coordinate(self, capsys):
        assert ups_mod.cli_main(["--lat", "north", "--lon", "1"]) == 64

    def test_missing_option_value(self, capsys):
        assert ups_mod.cli_main(["--offline", "--interval"]) == 64


class TestPackaging:
    def test_console_script_is_registered(self):
        """The module was complete and unreachable — no entry point."""
        from pathlib import Path

        root = Path(ups_mod.__file__).resolve().parents[3]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        assert 'ups = "hypernix.system.ups:cli_main"' in pyproject

    def test_has_a_main_guard(self):
        from pathlib import Path

        source = Path(ups_mod.__file__).read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in source
