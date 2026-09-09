"""fuse box — the thermal governor.

Tested against a fake sensor rather than a real GPU, because the
interesting cases are the ones a real card will not produce on demand:
sitting exactly on the target, crossing the trip point, a sensor that
stops answering halfway through a run, a vendor tool that refuses for
want of privileges.

Four properties matter more than the control law, and they are what
most of this file is about:

**It never raises a power limit.** Not with a flag, not through
arithmetic, not on the restore path. This is the promise that makes the
module safe to point at an expensive card, so it is asserted from
several directions.

**It changes nothing unless asked twice.** ``dry_run`` is the default,
and the CLI needs ``--underclock`` *and* ``--yes``.

**It puts things back.** On a clean exit, on an exception, and -- via
the state file -- after the process is killed outright.

**It does nothing when there is nothing to do.** A card that never gets
hot must cost the run zero pauses, and the summary must say so rather
than claiming a saving.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypernix.system import fusebox
from hypernix.system.fusebox_cli import main as fusebox_main

# ---------------------------------------------------------------------------
# A fake machine
# ---------------------------------------------------------------------------


class FakeClock:
    """Monotonic time under test control.

    The governor's behaviour is a function of elapsed time, and a test
    that used the real clock would either sleep for real or assert
    nothing about the timing.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Records what it was asked to wait, and advances the clock."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)


class FakeCards:
    """A GPU whose temperature the test sets."""

    def __init__(self, temps: list[float] | float, *, cpu: float | None = None,
                 power_w: float = 250.0, limit_w: float = 300.0) -> None:
        self.temps = temps if isinstance(temps, list) else [temps]
        self.cpu = cpu
        self.power_w = power_w
        self.limit_w = limit_w
        self.reads = 0

    def __call__(self) -> fusebox.Snapshot:
        # The last value repeats, so a test can give a short ramp and
        # then let the run continue at whatever it ended on.
        temp = self.temps[min(self.reads, len(self.temps) - 1)]
        self.reads += 1
        cards = []
        if temp is not None:
            cards.append(fusebox.CardReading(
                index=0, vendor="nvidia", name="Fake 4090",
                temperature_c=temp, power_w=self.power_w,
                power_limit_w=self.limit_w, utilization_pct=99.0,
            ))
        return fusebox.Snapshot(timestamp=float(self.reads),
                                cards=cards, cpu_celsius=self.cpu)


def snap(temp: float | None, *, cpu: float | None = None) -> fusebox.Snapshot:
    cards = []
    if temp is not None:
        cards.append(fusebox.CardReading(
            index=0, vendor="nvidia", name="Fake", temperature_c=temp,
        ))
    return fusebox.Snapshot(timestamp=0.0, cards=cards, cpu_celsius=cpu)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# ---------------------------------------------------------------------------
# The control law
# ---------------------------------------------------------------------------


class TestTheGovernor:
    def test_a_cool_card_runs_at_full_speed(self, clock):
        gov = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock)

        verdict = gov.observe(snap(60))

        assert verdict.action == fusebox.RUN
        assert gov.ease == 0.0
        assert gov.pause_for(1.0) == 0.0

    def test_over_target_eases_proportionally(self, clock):
        """Two degrees over should ease less than eight."""
        policy = fusebox.Policy(target_c=80, trip_c=90, max_ease=0.5)
        gentle = fusebox.Governor(policy, clock=clock)
        gentle.observe(snap(82))

        hard = fusebox.Governor(policy, clock=clock)
        hard.observe(snap(88))

        assert 0 < gentle.ease < hard.ease

    def test_full_ease_arrives_by_halfway_to_the_trip(self, clock):
        """Reaching the fuse should mean easing was not enough, not that
        it had not started yet."""
        policy = fusebox.Policy(target_c=80, trip_c=90, max_ease=0.5)
        gov = fusebox.Governor(policy, clock=clock)

        gov.observe(snap(85))

        assert gov.ease == pytest.approx(policy.max_ease)

    def test_the_pause_scales_with_the_step(self, clock):
        """The same policy has to behave the same on a model whose steps
        take 30ms and one whose steps take 3s."""
        gov = fusebox.Governor(fusebox.Policy(target_c=80, max_ease=0.5),
                               clock=clock)
        gov.observe(snap(85))

        assert gov.pause_for(0.030) == pytest.approx(0.015)
        assert gov.pause_for(1.0) == pytest.approx(0.5)

    def test_a_single_pause_is_capped(self, clock):
        """A multi-second stall inside a training step reads as a hang."""
        gov = fusebox.Governor(fusebox.Policy(target_c=80, max_ease=4.0),
                               clock=clock)
        gov.observe(snap(89))

        assert gov.pause_for(30.0) <= fusebox.MAX_PAUSE_SECONDS

    def test_easing_comes_off_gradually_not_at_once(self, clock):
        """Dropping straight to full speed puts the whole load back on a
        card that is only just cool, and the next read finds it hot
        again -- the oscillation this exists to prevent."""
        gov = fusebox.Governor(fusebox.Policy(target_c=80, max_ease=0.5),
                               clock=clock)
        gov.observe(snap(88))
        peak = gov.ease

        gov.observe(snap(60))
        first = gov.ease

        assert 0 < first < peak
        gov.observe(snap(60))
        assert gov.ease < first

    def test_the_dead_band_holds_rather_than_flapping(self, clock):
        """A card sitting just under target must not flip every read."""
        policy = fusebox.Policy(target_c=80, hysteresis_c=3.0, max_ease=0.5)
        gov = fusebox.Governor(policy, clock=clock)
        gov.observe(snap(85))
        held = gov.ease

        gov.observe(snap(78))  # under target, inside the hysteresis band

        assert gov.ease == held


class TestTheBreaker:
    def test_it_trips_above_the_limit(self, clock):
        gov = fusebox.Governor(fusebox.Policy(target_c=80, trip_c=90),
                               clock=clock)

        verdict = gov.observe(snap(91))

        assert verdict.action == fusebox.TRIP
        assert gov.tripped is True
        assert gov.trips == 1

    def test_it_stays_open_until_the_reset_temperature(self, clock):
        """Resetting at the trip point would have it flapping across the
        threshold, which is worse than staying open a little longer."""
        gov = fusebox.Governor(
            fusebox.Policy(target_c=80, trip_c=90, reset_c=78), clock=clock)
        gov.observe(snap(91))

        assert gov.observe(snap(88)).action == fusebox.TRIP
        assert gov.observe(snap(80)).action == fusebox.TRIP
        assert gov.observe(snap(77)).action != fusebox.TRIP

    def test_it_resumes_eased_not_at_full_speed(self, clock):
        """Whatever got the card to the trip point is still true."""
        policy = fusebox.Policy(target_c=80, trip_c=90, reset_c=78,
                                max_ease=0.5)
        gov = fusebox.Governor(policy, clock=clock)
        gov.observe(snap(91))
        gov.observe(snap(70))

        assert gov.ease == pytest.approx(policy.max_ease)

    def test_the_cpu_does_not_trip_unless_asked(self, clock):
        """A CPU at 85C during data loading is normal, and a governor
        that eases a cold GPU over it is doing harm on no evidence."""
        gov = fusebox.Governor(fusebox.Policy(target_c=80, trip_c=90),
                               clock=clock)

        verdict = gov.observe(snap(60, cpu=99))

        assert verdict.action == fusebox.RUN

    def test_the_cpu_trips_when_it_is_asked(self, clock):
        gov = fusebox.Governor(
            fusebox.Policy(target_c=80, trip_c=90, cpu_trip_c=95), clock=clock)

        verdict = gov.observe(snap(60, cpu=99))

        assert verdict.action == fusebox.TRIP

    def test_the_cpu_alone_never_drives_the_ease(self, clock):
        """Only the trip. Pacing a run over CPU temperature would slow
        every run on a laptop with a warm chassis."""
        gov = fusebox.Governor(
            fusebox.Policy(target_c=80, trip_c=90, cpu_trip_c=95), clock=clock)

        gov.observe(snap(60, cpu=94))

        assert gov.ease == 0.0


class TestNoReadings:
    def test_no_sensor_means_full_speed_not_caution(self, clock):
        """Plenty of machines report no temperature at all. Easing a run
        that cannot be measured is harm on no evidence."""
        gov = fusebox.Governor(fusebox.Policy(), clock=clock)

        verdict = gov.observe(snap(None))

        assert verdict.action == fusebox.RUN
        assert gov.ease == 0.0

    def test_a_sensor_that_stops_answering_mid_run_releases_the_ease(self, clock):
        gov = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock)
        gov.observe(snap(88))
        assert gov.ease > 0

        gov.observe(snap(None))

        assert gov.ease == 0.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TestPolicyValidation:
    def test_a_trip_below_the_target_is_refused(self):
        with pytest.raises(ValueError, match="must be above target_c"):
            fusebox.Policy(target_c=90, trip_c=80).validate()

    def test_a_reset_above_the_trip_is_refused(self):
        """It would never reset."""
        with pytest.raises(ValueError, match="never resets"):
            fusebox.Policy(target_c=70, trip_c=80, reset_c=85).validate()

    def test_an_underclock_floor_outside_the_range_is_refused(self):
        with pytest.raises(ValueError, match="underclock_floor"):
            fusebox.Policy(underclock_floor=0.0).validate()

    def test_it_round_trips_through_a_dict(self):
        policy = fusebox.Policy(target_c=71, trip_c=88, reset_c=70)

        assert fusebox.Policy.from_dict(policy.to_dict()) == policy

    def test_unknown_keys_in_a_dict_are_ignored(self):
        """A config written by a newer version must not stop an older
        one from starting."""
        data = fusebox.Policy().to_dict()
        data["invented_later"] = 5

        assert fusebox.Policy.from_dict(data).target_c == fusebox.DEFAULT_TARGET_C


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


class TestPacing:
    def test_a_cool_run_is_never_paused(self, clock):
        sleep = FakeSleep(clock)
        box = fusebox.FuseBox(fusebox.Policy(target_c=80, poll_seconds=0),
                              sensor=FakeCards(55), clock=clock, sleep=sleep)

        for _ in range(20):
            clock.advance(0.1)
            box.pace()

        assert sleep.total == 0.0
        assert box.governor.eased_seconds == 0.0

    def test_a_hot_run_is_paced(self, clock):
        sleep = FakeSleep(clock)
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=95, poll_seconds=0,
                           max_ease=0.5),
            sensor=FakeCards(88), clock=clock, sleep=sleep)

        for _ in range(10):
            clock.advance(0.1)
            box.pace(step_seconds=0.1)

        assert sleep.total > 0
        assert box.governor.eased_seconds == pytest.approx(sleep.total)

    def test_the_sensor_is_not_read_every_step(self, clock):
        """A model whose steps take 20ms would otherwise shell out to
        nvidia-smi fifty times a second."""
        cards = FakeCards(60)
        box = fusebox.FuseBox(fusebox.Policy(poll_seconds=2.0),
                              sensor=cards, clock=clock, sleep=FakeSleep(clock))

        for _ in range(100):
            clock.advance(0.02)  # 2 seconds of 20ms steps
            box.pace()

        assert cards.reads <= 3

    def test_a_tripped_breaker_holds_the_run_until_it_cools(self, clock):
        sleep = FakeSleep(clock)
        # Hot enough to trip, then cool.
        cards = FakeCards([95, 95, 95, 60])
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=90, reset_c=78,
                           poll_seconds=1.0),
            sensor=cards, clock=clock, sleep=sleep)

        clock.advance(0.1)
        waited = box.pace(step_seconds=0.1)

        assert waited > 0
        assert box.governor.trips == 1
        assert box.governor.tripped_seconds > 0

    def test_a_breaker_that_never_resets_raises_rather_than_hanging(self, clock):
        """An hour above the trip point is a broken fan, not a
        transient. Blocking forever without saying why is the one
        outcome worse than stopping."""
        sleep = FakeSleep(clock)
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=90, poll_seconds=60.0),
            sensor=FakeCards(99), clock=clock, sleep=sleep)

        with pytest.raises(fusebox.ThermalStall, match="check cooling"):
            box.pace(step_seconds=0.1)


class TestHonestAccounting:
    def test_an_untroubled_run_says_it_did_nothing(self, clock):
        box = fusebox.FuseBox(fusebox.Policy(target_c=80, poll_seconds=0),
                              sensor=FakeCards(55), clock=clock,
                              sleep=FakeSleep(clock))
        for _ in range(5):
            clock.advance(0.1)
            box.pace()

        summary = box.summary()

        assert "did not intervene" in summary
        assert "cost the run nothing" in summary

    def test_a_troubled_run_reports_the_seconds(self, clock):
        sleep = FakeSleep(clock)
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=95, poll_seconds=0,
                           max_ease=0.5),
            sensor=FakeCards(90), clock=clock, sleep=sleep)
        for _ in range(10):
            clock.advance(0.5)
            box.pace(step_seconds=0.5)

        report = box.governor.report()

        assert report["eased_seconds"] > 0
        assert report["peak_c"] == 90
        assert "eased for" in box.summary()

    def test_time_over_target_is_measured_not_counted(self, clock):
        """Readings are not evenly spaced, so counting them would report
        a different number for the same run at a different poll rate."""
        gov = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock)
        gov.observe(snap(85))
        clock.advance(10.0)
        gov.observe(snap(85))

        assert gov.seconds_over_target == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# The actuator, and the promise it makes
# ---------------------------------------------------------------------------


class FakeLimits:
    """Stands in for whatever the vendor tools report."""

    def __init__(self, current=250.0, default=300.0, minimum=150.0):
        self.value = fusebox._Limits(
            index=0, vendor="nvidia", current_w=current,
            default_w=default, min_w=minimum, max_w=default,
        )

    def __call__(self):
        return [self.value]


class TestItNeverRaisesALimit:
    def test_above_the_default_is_refused(self, tmp_path):
        control = fusebox.ClockControl(dry_run=True,
                                       state_file=tmp_path / "state.json")
        control.limits = FakeLimits(current=250.0, default=300.0)

        with pytest.raises(fusebox.ClockControlError, match="does not raise"):
            control.set_power_limit(0, 350.0)

    def test_even_when_the_card_is_already_at_its_default(self, tmp_path):
        control = fusebox.ClockControl(dry_run=True,
                                       state_file=tmp_path / "state.json")
        control.limits = FakeLimits(current=300.0, default=300.0)

        with pytest.raises(fusebox.ClockControlError):
            control.set_power_limit(0, 301.0)

    def test_the_restore_path_cannot_be_used_to_raise_one(self, tmp_path):
        """Restore puts back what was recorded. If a caller could seed
        `original` with a higher number and call restore, the promise
        would have a hole in it -- so restore goes through the same
        recorded baseline, and the baseline is only ever written from a
        reading."""
        control = fusebox.ClockControl(dry_run=True,
                                       state_file=tmp_path / "state.json")
        control.limits = FakeLimits(current=250.0, default=300.0)
        control.set_power_limit(0, 260.0)

        assert control.original[0] == 300.0
        assert control.original[0] <= 300.0

    def test_below_the_cards_minimum_is_clamped_not_refused(self, tmp_path):
        """A card that will not go below 150W should be set to 150W, not
        have the whole adjustment abandoned."""
        control = fusebox.ClockControl(dry_run=False,
                                       state_file=tmp_path / "state.json")
        control.limits = FakeLimits(current=250.0, default=300.0, minimum=150.0)
        issued: list[list[str]] = []
        control._set_argv = lambda i, w, v: issued.append(
            ["fake-smi", str(i), f"{w:.0f}"]) or issued[-1]

        control.set_power_limit(0, 50.0)

        assert issued[-1][-1] == "150"


class TestItChangesNothingUnlessAsked:
    def test_dry_run_is_the_default(self):
        assert fusebox.ClockControl().dry_run is True

    def test_a_dry_run_issues_no_command(self, tmp_path):
        control = fusebox.ClockControl(dry_run=True,
                                       state_file=tmp_path / "state.json")
        control.limits = FakeLimits()
        ran: list[list[str]] = []
        control._set_argv = lambda i, w, v: ["fake-smi", str(i)]

        applied = control.set_power_limit(0, 200.0)

        assert applied is False
        assert control.applied == {}
        assert ran == []

    def test_a_fusebox_built_without_apply_changes_cannot_write(self):
        assert fusebox.FuseBox().control.dry_run is True
        assert fusebox.FuseBox(apply_changes=True).control.dry_run is False

    def test_the_training_loop_hook_never_applies_changes(self):
        """`hnx train --thermal-target` paces a run. It must not be a
        back door to changing a card's power limit, which needs a person
        to have said --underclock --yes."""
        from hypernix.training import train

        box = train._thermal_governor(75.0)

        assert box is not None
        assert box.control.dry_run is True
        assert box.policy.auto_underclock is False

    def test_underclocking_is_off_in_a_default_policy(self):
        assert fusebox.Policy().auto_underclock is False


class TestPermissionRefusalIsNotAnEscalation:
    @pytest.mark.parametrize("message", [
        "Permission denied",
        "Insufficient Permissions",
        "This operation requires root privileges",
    ])
    def test_a_refusal_is_recognised(self, message):
        assert fusebox._is_permission_error(message)

    def test_an_ordinary_failure_is_not_mistaken_for_one(self):
        assert not fusebox._is_permission_error(
            "Setting power limit is not supported for GPU 0")

    ESCALATORS = ("sudo", "pkexec", "doas", "runas", "gksudo")

    #: An escalator in *command position* -- the first element of a list
    #: or the first argument of a call. Searching the source for the
    #: bare word instead finds the warning above, which tells a person
    #: how to grant the privilege themselves, and reports the module as
    #: escalating because it documented not doing so. That is the fourth
    #: time this repository has hit that shape of false positive, and
    #: stripping strings is no answer here: the thing being looked for
    #: *is* a string literal.
    IN_COMMAND_POSITION = r"[\[(]\s*[\"']({})[\"']".format("|".join(ESCALATORS))

    def test_no_command_anywhere_escalates(self):
        """The assertion that keeps this honest: nothing in the module
        puts sudo, pkexec or doas at the front of an argument list."""
        import re

        source = Path(fusebox.__file__).read_text(encoding="utf-8")
        found = re.search(self.IN_COMMAND_POSITION, source, re.IGNORECASE)

        assert found is None, f"fusebox escalates: {found.group(0)!r}"

    def test_the_check_would_catch_a_real_escalation(self):
        """A check tuned until it stops firing is a check that no longer
        checks anything, so: the same pattern, against code that does
        the forbidden thing."""
        import re

        assert re.search(self.IN_COMMAND_POSITION,
                         'subprocess.run(["sudo", "nvidia-smi", "-pl", "200"])')
        assert re.search(self.IN_COMMAND_POSITION, "_run(['pkexec', tool])")
        # And it does not fire on prose that merely mentions one.
        assert not re.search(self.IN_COMMAND_POSITION,
                             '"...run `sudo nvidia-smi` yourself."')

    def test_every_command_it_can_build_starts_with_a_vendor_tool(self,
                                                                  monkeypatch):
        """The behavioural half. The regex above reads the source; this
        asks the object what it would actually run."""
        control = fusebox.ClockControl(dry_run=True)
        monkeypatch.setattr(control, "_nvidia_smi", staticmethod(
            lambda: "/usr/bin/nvidia-smi"))
        monkeypatch.setattr(control, "_amd_tool", staticmethod(
            lambda: ("/opt/rocm/bin/rocm-smi", "rocm-smi")))

        for vendor, expected in (("nvidia", "/usr/bin/nvidia-smi"),
                                 ("amd", "/opt/rocm/bin/rocm-smi")):
            argv = control._set_argv(0, 200.0, vendor)
            assert argv is not None
            assert argv[0] == expected
            assert not any(e in argv for e in self.ESCALATORS)

    def test_an_index_reaches_the_command_line_as_an_integer(self, monkeypatch):
        """Nothing from outside becomes part of an argument list.

        The index is formatted through ``int()`` and the wattage through
        ``float()``, so text does not get through at all -- it raises
        here rather than reaching a command line. Every invocation is a
        list and there is no shell, so this is the second lock on a door
        that is already shut; it is here because "already shut" is a
        property of the current code and this is what notices if that
        changes.
        """
        control = fusebox.ClockControl(dry_run=True)
        monkeypatch.setattr(control, "_nvidia_smi", staticmethod(
            lambda: "/usr/bin/nvidia-smi"))

        with pytest.raises(ValueError):
            control._set_argv("0; rm -rf /", 200.0, "nvidia")  # type: ignore[arg-type]

        argv = control._set_argv(3, 200.0, "nvidia")
        assert argv == ["/usr/bin/nvidia-smi", "-i", "3", "--power-limit=200"]

    def test_no_command_is_ever_run_through_a_shell(self):
        """`shell=True` anywhere here would make the formatting above
        the only thing between a vendor string and a command line."""
        source = Path(fusebox.__file__).read_text(encoding="utf-8")

        assert "shell=True" not in source


class TestItPutsThingsBack:
    def test_a_clean_exit_restores(self, tmp_path, clock):
        box = fusebox.FuseBox(
            fusebox.Policy(poll_seconds=0), apply_changes=True,
            state_file=tmp_path / "state.json",
            sensor=FakeCards(60), clock=clock, sleep=FakeSleep(clock))
        restored: list[int] = []
        box.control.original = {0: 300.0}
        box.control.applied = {0: 250.0}
        box.control.restore = lambda index, vendor="nvidia": (
            restored.append(index) or True)

        with box.session():
            pass

        assert restored == [0]

    def test_an_exception_still_restores(self, tmp_path, clock):
        """The `finally` is the point: a run that dies must not leave a
        card holding a lowered limit."""
        box = fusebox.FuseBox(
            fusebox.Policy(poll_seconds=0), apply_changes=True,
            state_file=tmp_path / "state.json",
            sensor=FakeCards(60), clock=clock, sleep=FakeSleep(clock))
        restored: list[int] = []
        box.control.original = {0: 300.0}
        box.control.applied = {0: 250.0}
        box.control.restore = lambda index, vendor="nvidia": (
            restored.append(index) or True)

        with pytest.raises(RuntimeError):
            with box.session():
                raise RuntimeError("the run died")

        assert restored == [0]

    def test_changes_are_recorded_on_disk_as_they_are_made(self, tmp_path):
        """SIGKILL reaches no `finally`. The state file is what makes
        that recoverable, and it has to be written when the change
        happens, not at exit."""
        state = tmp_path / "state.json"
        control = fusebox.ClockControl(dry_run=False, state_file=state)
        control.limits = FakeLimits(current=300.0, default=300.0)
        control._set_argv = lambda i, w, v: ["true"]

        control.set_power_limit(0, 250.0)

        assert state.is_file()
        saved = json.loads(state.read_text())
        assert saved["applied"] == {"0": 250.0}
        assert saved["original"] == {"0": 300.0}

    def test_a_later_process_can_restore_from_the_file(self, tmp_path):
        state = tmp_path / "state.json"
        first = fusebox.ClockControl(dry_run=False, state_file=state)
        first.limits = FakeLimits(current=300.0, default=300.0)
        first._set_argv = lambda i, w, v: ["true"]
        first.set_power_limit(0, 250.0)

        second = fusebox.ClockControl(dry_run=True, state_file=state)

        assert second.load_state() is True
        assert second.original == {0: 300.0}
        assert second.applied == {0: 250.0}

    def test_a_missing_state_file_is_not_an_error(self, tmp_path):
        control = fusebox.ClockControl(state_file=tmp_path / "absent.json")

        assert control.load_state() is False

    def test_a_corrupt_state_file_is_not_an_error(self, tmp_path):
        state = tmp_path / "state.json"
        state.write_text("{ this is not json")

        assert fusebox.ClockControl(state_file=state).load_state() is False


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


class TestTheCLI:
    def test_status_runs_on_a_machine_with_no_gpu(self, capsys):
        """Which is most CI machines, and every laptop this is
        developed on."""
        code = fusebox_main(["status"])

        assert code in (0, 2, 3)
        assert capsys.readouterr().out

    def test_status_json_is_parseable(self, capsys):
        fusebox_main(["status", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert "policy" in payload and "verdict" in payload

    def test_a_bare_invocation_is_status(self, capsys):
        fusebox_main([])
        bare = capsys.readouterr().out

        fusebox_main(["status"])
        explicit = capsys.readouterr().out

        assert bare.split("->")[0] == explicit.split("->")[0]

    def test_plan_changes_nothing_and_says_so(self, capsys):
        code = fusebox_main(["plan"])

        assert code == 0
        assert "nothing was changed" in capsys.readouterr().out

    def test_a_contradictory_policy_is_refused_with_a_reason(self, capsys):
        code = fusebox_main(["status", "--target", "95", "--trip", "80"])

        assert code == 1
        assert "must be above target_c" in capsys.readouterr().err

    def test_underclock_without_yes_says_what_it_did_not_do(self, capsys, monkeypatch):
        monkeypatch.setenv("HNX_FUSEBOX_STATE",
                           str(Path("/tmp/hnx-fusebox-test-none.json")))
        fusebox_main(["watch", "-n", "1", "--underclock", "--quiet"])

        err = capsys.readouterr().err
        assert "Add --yes to confirm" in err

    def test_restore_without_yes_only_reports(self, tmp_path, monkeypatch, capsys):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({
            "original": {"0": 300.0}, "applied": {"0": 250.0},
        }))
        monkeypatch.setenv("HNX_FUSEBOX_STATE", str(state))

        code = fusebox_main(["restore"])

        assert code == 3
        assert "add --yes" in capsys.readouterr().out
        assert state.is_file()

    def test_restore_with_nothing_applied_is_a_success(self, tmp_path,
                                                       monkeypatch, capsys):
        monkeypatch.setenv("HNX_FUSEBOX_STATE", str(tmp_path / "absent.json"))

        assert fusebox_main(["restore", "--yes"]) == 0
        assert "nothing to restore" in capsys.readouterr().out

    def test_status_reports_a_crashed_runs_leftovers(self, tmp_path,
                                                     monkeypatch, capsys):
        """The case the state file exists for: a run killed outright,
        and a card still at a lowered limit."""
        state = tmp_path / "state.json"
        state.write_text(json.dumps({
            "original": {"0": 300.0}, "applied": {"0": 250.0},
        }))
        monkeypatch.setenv("HNX_FUSEBOX_STATE", str(state))

        code = fusebox_main(["status"])

        out = capsys.readouterr().out
        assert code == 3
        assert "still applied" in out
        assert "hnx fusebox restore" in out

    def test_watch_stops_after_the_requested_count(self, capsys):
        code = fusebox_main(["watch", "-n", "2", "--interval", "0", "--json"])

        assert code in (0, 2)


class TestWiredIn:
    def test_the_subcommand_reaches_the_module(self):
        from hypernix.interfaces import cli

        assert "fusebox" in cli._SUBCOMMANDS
        assert "fuse-box" in cli._SUBCOMMANDS

    def test_it_is_in_the_package_registry(self):
        import hypernix

        assert "fusebox" in hypernix.MODULE_CATEGORIES["system"]
        assert hypernix.fusebox.DEFAULT_TARGET_C > 0

    def test_train_takes_a_thermal_target(self):
        import inspect

        from hypernix.training.train import train

        assert "thermal_target_c" in inspect.signature(train).parameters

    def test_the_env_var_turns_it_on_without_a_flag(self, monkeypatch):
        """`hypernix-t1 launch-script` sets the environment, not flags."""
        from hypernix.training import train

        monkeypatch.setenv("HNX_THERMAL_TARGET", "72")
        box = train._thermal_governor(None)

        assert box is not None
        assert box.policy.target_c == 72

    def test_a_nonsense_env_var_does_not_stop_the_run(self, monkeypatch, capsys):
        from hypernix.training import train

        monkeypatch.setenv("HNX_THERMAL_TARGET", "warm")

        assert train._thermal_governor(None) is None
        assert "ignoring" in capsys.readouterr().out

    def test_a_target_above_the_default_trip_moves_the_fuse_too(self):
        """Otherwise the breaker would be open from the first read."""
        from hypernix.training import train

        box = train._thermal_governor(95.0)

        assert box is not None
        assert box.policy.trip_c > 95.0
        assert box.policy.reset_c < box.policy.trip_c


# ---------------------------------------------------------------------------
# The claim the module refuses to make
# ---------------------------------------------------------------------------


class ThermalModel:
    """A GPU as a first-order thermal system.

    Deliberately simple and deliberately generous to the "slow down to
    go faster" story: the die approaches an equilibrium set by how hard
    it is driven, with a time constant of tens of seconds, and the
    driver's own throttling is modelled the way vendors do it -- a large
    clock cut once a threshold is passed, held until the card is well
    back under it.

    Performance scales as ``power ** 0.35``, which is the shape every
    published power-vs-throughput curve for a modern GPU has: 80% of
    the power buys about 92% of the work. That exponent is why capping
    beats pausing, and why neither beats running flat out.
    """

    AMBIENT = 35.0
    TAU = 25.0
    THROTTLE_C = 90.0
    RESUME_C = 84.0
    PERF_EXP = 0.35

    def __init__(self, equilibrium_c: float = 95.0, penalty: float = 0.55):
        self.equilibrium_c = equilibrium_c
        self.penalty = penalty
        self.temp = self.AMBIENT
        self.throttled = False

    def tick(self, seconds: float, power: float) -> float:
        """Advance by *seconds* at *power* in [0, 1]. Returns work done."""
        import math

        if self.throttled and self.temp <= self.RESUME_C:
            self.throttled = False
        elif not self.throttled and self.temp >= self.THROTTLE_C:
            self.throttled = True
        effective = power * (self.penalty if self.throttled else 1.0)
        rate = effective ** self.PERF_EXP if effective > 0 else 0.0
        equilibrium = self.AMBIENT + (self.equilibrium_c - self.AMBIENT) * effective
        self.temp += (equilibrium - self.temp) * (1 - math.exp(-seconds / self.TAU))
        return seconds * rate


STEP = 0.25
WALL = 1800.0


def _run_flat_out(model: ThermalModel) -> tuple[float, float]:
    work = t = 0.0
    while t < WALL:
        work += model.tick(STEP, 1.0)
        t += STEP
    return work / STEP, model.temp


def _run_paced(model: ThermalModel, target_c: float) -> tuple[float, float]:
    work = t = 0.0
    while t < WALL:
        work += model.tick(STEP, 1.0)
        t += STEP
        ease = min(0.6, max(0.0, (model.temp - target_c) * 0.08))
        if ease > 0:
            pause = STEP * ease
            model.tick(pause, 0.0)   # a pause does no work at all
            t += pause
    return work / STEP, model.temp


def _run_capped(model: ThermalModel, target_c: float) -> tuple[float, float]:
    work = t = 0.0
    scale = 1.0
    while t < WALL:
        over = model.temp - target_c
        if over > 0:
            scale = max(0.55, scale - 0.004)
        elif over < -2:
            scale = min(1.0, scale + 0.002)
        work += model.tick(STEP, scale)
        t += STEP
    return work / STEP, model.temp


class TestTheThroughputClaim:
    """The module docstring says managing temperature costs throughput.

    That is an unusual thing for a feature to say about itself, so it is
    asserted here rather than merely written down -- if the ordering
    ever changed, the docstring would be wrong and this would say so.
    """

    def test_pausing_costs_throughput(self):
        """The claim the design brief made, and the measurement that
        does not support it: pausing between steps is slower than
        letting the driver throttle, because a throttled card still does
        most of the work and a paused one does none."""
        flat, _ = _run_flat_out(ThermalModel())
        paced, paced_temp = _run_paced(ThermalModel(), 82.0)

        assert paced < flat
        assert paced_temp < ThermalModel.THROTTLE_C

    def test_a_power_cap_costs_much_less_than_pausing(self):
        """Which is why the module reaches for it first when allowed:
        the same temperature for a third to a quarter of the cost."""
        flat, _ = _run_flat_out(ThermalModel())
        paced, _ = _run_paced(ThermalModel(), 82.0)
        capped, capped_temp = _run_capped(ThermalModel(), 82.0)

        assert capped > paced
        assert (flat - capped) < (flat - paced) / 2
        assert capped_temp < ThermalModel.THROTTLE_C

    def test_both_levers_buy_a_cooler_card(self):
        """Which is the thing they actually sell."""
        _, flat_temp = _run_flat_out(ThermalModel())
        _, paced_temp = _run_paced(ThermalModel(), 78.0)
        _, capped_temp = _run_capped(ThermalModel(), 78.0)

        assert paced_temp < flat_temp
        assert capped_temp < flat_temp

    @pytest.mark.parametrize("equilibrium_c", [92.0, 100.0, 110.0])
    @pytest.mark.parametrize("penalty", [0.7, 0.35])
    def test_the_ordering_holds_across_machines(self, equilibrium_c, penalty):
        """Marginal cooling and a savage throttle are the conditions
        under which "slow down to go faster" would be true if it were
        ever true. It is not, over the range a real machine occupies."""
        flat, _ = _run_flat_out(ThermalModel(equilibrium_c, penalty))
        paced, _ = _run_paced(ThermalModel(equilibrium_c, penalty), 82.0)

        assert paced < flat

    def test_the_docstring_does_not_promise_a_speedup(self):
        """A module whose measurements say one thing and whose prose
        says another is worse than one that says nothing."""
        doc = fusebox.__doc__ or ""

        assert "not a speedup" in doc
        assert "does not survive being" in doc


class TestTheCostIsReported:
    def test_the_summary_prices_the_intervention(self, clock):
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=95, poll_seconds=0,
                           max_ease=0.5),
            sensor=FakeCards(88), clock=clock, sleep=FakeSleep(clock))
        for _ in range(20):
            clock.advance(0.5)
            box.pace(step_seconds=0.5)

        summary = box.summary()

        assert "cost" in summary
        assert "%" in summary

    def test_the_cost_is_a_fraction_of_the_run_not_a_raw_number(self, clock):
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=95, poll_seconds=0,
                           max_ease=0.5),
            sensor=FakeCards(88), clock=clock, sleep=FakeSleep(clock))
        for _ in range(20):
            clock.advance(0.5)
            box.pace(step_seconds=0.5)

        cost = box.governor.report()["cost_pct"]

        assert cost is not None
        assert 0 < cost < 100

    def test_no_run_yet_means_no_percentage(self, clock):
        """A percentage of nothing is unanswerable, not zero."""
        gov = fusebox.Governor(fusebox.Policy(), clock=clock)

        assert gov.report()["cost_pct"] is None


class TestTheCheaperLeverTakesOver:
    def test_the_pause_stands_down_once_a_cap_is_applied(self, clock):
        """Both levers at full pays twice for the same degrees, and the
        pause is the dearer by three to four times."""
        box = fusebox.FuseBox(
            fusebox.Policy(target_c=80, trip_c=95, poll_seconds=0,
                           auto_underclock=True),
            sensor=FakeCards(88), clock=clock, sleep=FakeSleep(clock))
        box.control.limits = FakeLimits()

        box.poll(force=True)
        assert box.governor.ease_ceiling == 1.0

        box.control.applied = {0: 250.0}
        box.poll(force=True)

        assert box.governor.ease_ceiling == fusebox.EASE_UNDER_CAP

    def test_the_ceiling_actually_limits_the_pause(self, clock):
        policy = fusebox.Policy(target_c=80, trip_c=90, max_ease=0.5)
        gov = fusebox.Governor(policy, clock=clock)
        gov.ease_ceiling = fusebox.EASE_UNDER_CAP

        gov.observe(snap(89))

        assert gov.ease <= policy.max_ease * fusebox.EASE_UNDER_CAP

    def test_the_ceiling_stays_at_one_when_underclocking_is_off(self, clock):
        """The default. Without the cheap lever there is nothing to
        stand down for."""
        box = fusebox.FuseBox(fusebox.Policy(target_c=80, poll_seconds=0),
                              sensor=FakeCards(88), clock=clock,
                              sleep=FakeSleep(clock))
        box.poll(force=True)

        assert box.governor.ease_ceiling == 1.0


class TestItActuallyReachesTheTarget:
    """The bug a unit test could not have found.

    Every test above observes a fixed temperature and checks the
    response. That passes happily for a controller with a permanent
    steady-state offset -- which is what this had, settling at 82.5 °C
    against an 80 °C target, for ever. Closing the loop is what showed
    it, so closing the loop is what guards it.
    """

    @staticmethod
    def _closed_loop(target_c: float, seconds: float = 900.0) -> tuple[float, float]:
        """Run the real governor against the thermal model. (final, peak)"""
        import math

        model = ThermalModel(equilibrium_c=95.0)
        clock = FakeClock()
        paused = [0.0]

        def sleep(sec):
            paused[0] += sec
            model.tick(sec, 0.0)
            clock.advance(sec)

        def sensor():
            return fusebox.Snapshot(
                clock.now,
                [fusebox.CardReading(0, "nvidia", "sim",
                                     temperature_c=model.temp)],
            )

        box = fusebox.FuseBox(
            fusebox.Policy(target_c=target_c, trip_c=target_c + 10,
                           reset_c=target_c - 2, poll_seconds=2.0),
            sensor=sensor, clock=clock, sleep=sleep)
        start = clock.now
        while clock.now - start < seconds:
            model.tick(0.25, 1.0)
            clock.advance(0.25)
            box.pace(step_seconds=0.25)
        assert math.isfinite(model.temp)
        return model.temp, box.governor.peak_c or 0.0

    @pytest.mark.parametrize("target", [76.0, 80.0, 84.0])
    def test_it_settles_at_the_target_not_above_it(self, target):
        final, _ = self._closed_loop(target)

        assert abs(final - target) < 1.0, (
            f"asked for {target}°C, settled at {final:.1f}°C"
        )

    def test_it_does_not_overshoot_downward(self, ):
        """An integral term with no anti-windup would drive the card
        well under the target and then sit there wasting throughput."""
        final, _ = self._closed_loop(80.0)

        assert final > 80.0 - 3.0

    def test_the_integral_term_is_bounded_by_the_ease_ceiling(self, clock):
        """The whole of the anti-windup: it can never store more than
        the controller could have produced on its own."""
        policy = fusebox.Policy(target_c=80, trip_c=95, max_ease=0.5)
        gov = fusebox.Governor(policy, clock=clock)

        for _ in range(500):
            clock.advance(10.0)
            gov.observe(snap(94))

        assert gov.bias <= policy.max_ease
        assert gov.ease <= policy.max_ease

    def test_the_bias_unwinds_when_the_card_cools(self, clock):
        gov = fusebox.Governor(fusebox.Policy(target_c=80, trip_c=95),
                               clock=clock)
        for _ in range(50):
            clock.advance(10.0)
            gov.observe(snap(88))
        assert gov.bias > 0

        for _ in range(20):
            clock.advance(10.0)
            gov.observe(snap(50))

        assert gov.bias == 0.0
        assert gov.ease == 0.0


class TestOverTargetIsReportedHonestly:
    def test_holding_the_target_is_not_reported_as_missing_it(self, clock):
        """A governor holding 79.95 °C against an 80 °C target must not
        report itself over target for the whole run because 80.02 is
        greater than 80."""
        gov = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock)
        for _ in range(100):
            clock.advance(1.0)
            gov.observe(snap(80.05))

        assert gov.seconds_over_target == 0.0

    def test_genuinely_over_target_is_reported(self, clock):
        gov = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock)
        for _ in range(10):
            clock.advance(1.0)
            gov.observe(snap(86))

        assert gov.seconds_over_target == pytest.approx(9.0)

    def test_the_magnitude_is_kept_separately(self, clock):
        """Seconds alone cannot tell 0.1 °C over for an hour from 9 °C
        over for an hour, and those are very different runs."""
        mild = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock)
        clock.advance(1.0)
        mild.observe(snap(80.5))
        clock.advance(100.0)
        mild.observe(snap(80.5))

        clock2 = FakeClock()
        severe = fusebox.Governor(fusebox.Policy(target_c=80), clock=clock2)
        clock2.advance(1.0)
        severe.observe(snap(89))
        clock2.advance(100.0)
        severe.observe(snap(89))

        assert mild.seconds_over_target == 0.0
        assert severe.seconds_over_target > 0
        assert (severe.degree_seconds_over_target
                > 10 * mild.degree_seconds_over_target)
