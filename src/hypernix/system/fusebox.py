"""fusebox — the breaker panel for a training run.

Manages the temperature of a long run: watches every card, paces the
loop or lowers a power limit to hold a target, and trips like a fuse if
a card passes a hard limit anyway.

What this does not do, measured
-------------------------------
The obvious pitch for a module like this is "slow down now to go faster
later" -- ease off before the driver throttles, and win back more than
you gave up. That was the design brief, and it does not survive being
simulated. ``tests/test_fusebox.py::TestTheThroughputClaim`` runs a
first-order thermal model against three strategies and the ordering
does not change across the useful range of cooling quality and throttle
severity:

===================  ===========================================
running flat out     fastest, in almost every regime
a lower power limit  2-7% slower, and much cooler
pausing between      12-24% slower, and much cooler
steps
===================  ===========================================

The reason is not subtle once you see it: a driver's thermal throttle
still does *most* of the work -- clocks at 55% are 55% of a card, not
zero -- while a pause does none. And performance scales sublinearly with
power (roughly ``power ** 0.35`` over the useful range), so 80% of the
power buys about 92% of the throughput while pausing 20% of the time
buys 80%. Nothing you do above the throttle point recovers more than the
throttle costs.

So this module is not a speedup, and it is not sold as one. It sells a
different thing:

**A temperature you chose, at a cost you can see.** "Hold this card at
78 °C" is a legitimate thing to want -- a shared machine, a laptop on a
desk, a room someone sleeps in, a card you would like to still own in
three years, a power bill. This delivers exactly that, and
:meth:`Governor.report` says in seconds what it cost, so the trade is
visible rather than assumed.

**A fuse.** A card at 95 °C with a failing fan should stop, and "the
driver will handle it" is not a plan when the driver's next move is a
shutdown in the middle of a checkpoint write. The breaker pauses the
run, waits for the card to cool, and resumes -- and if it is still hot
an hour later it says the fan is broken instead of blocking forever.

Because the power limit is the cheaper lever by a factor of three or
four, that is the one this reaches for first when it is allowed to; the
pause is the fallback that needs no privileges and can always be used.

How it works
------------
* It **watches** every card, whoever made it, through
  :mod:`hypernix.system.gpus`, plus the CPU through
  :mod:`hypernix.monitoring.thermometer`.
* It **eases** — inserts small pauses between training steps — when a
  card climbs past the target, and takes them straight back off when it
  cools. Free, instant, reversible, and the expensive lever.
* It **underclocks**, if you ask it to, by lowering a card's power
  limit — and puts the limit back when the run ends. Cheap, but it
  needs privileges and outlives the process.
* It **trips**, like a fuse, if a card passes the hard limit anyway.

What it will not do
-------------------
**It will not raise a power limit above the card's factory default.**
Not with a flag, not on request. Lowering a limit and restoring it is
thermal management; going past the default is overclocking, and a
training run that quietly overvolts someone's card is not a feature.

**It will not touch a card's settings unless you ask.** ``ease`` and
``trip`` need no permissions and change nothing outside this process, so
they are on by default. Underclocking shells out to a vendor tool, needs
privileges, and persists past the run — so it is off until
``auto_underclock=True``, and every change is recorded so
``hnx fusebox restore`` can undo it after a crash.

**It will not run anything it was handed.** The vendor commands are
built from a fixed table in this file. Nothing from a config file, an
environment variable or a network response reaches a command line.

**It will not sudo.** If a vendor tool refuses for want of privileges,
that is reported once and the governor falls back to pausing, which
needs none. A tool that escalates on its own is a tool nobody can audit.

**It will not intervene when it has nothing to do.** On a card that
never approaches the target this is one cheap sensor read per interval
and nothing else — no pauses, no changes, no cost.
:meth:`Governor.summary` says so in those words rather than claiming a
saving it did not make.

Using it
--------
From a training loop::

    from hypernix.system import fusebox

    box = fusebox.FuseBox()
    with box.session():
        for step in range(steps):
            train_one_step()
            box.pace()          # returns the seconds it paused, if any

From the shell::

    hnx fusebox status
    hnx fusebox watch --target 78
    hnx fusebox watch --target 78 --underclock --yes
    hnx fusebox restore
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Policy",
    "CardReading",
    "Snapshot",
    "Verdict",
    "Governor",
    "ClockControl",
    "FuseBox",
    "read_snapshot",
    "state_path",
    "restore_all",
    "EASE",
    "RUN",
    "TRIP",
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Where a card should sit. 80 °C is under the point at which every
#: current consumer and datacentre part begins its own throttling, with
#: room for the sensor lag between the die and what nvidia-smi reports.
DEFAULT_TARGET_C = 80.0

#: The fuse. Above this the run stops until the card cools, because at
#: this point the driver is throttling anyway and the pause at least
#: happens somewhere we can account for it.
DEFAULT_TRIP_C = 90.0

#: And the point it may start again. Well below the trip, so a card that
#: is genuinely at its limit does not oscillate across the threshold.
DEFAULT_RESET_C = 78.0

#: How much of a step's own duration may be spent pausing. 0.5 means a
#: step can take half again as long; past that a run is barely moving
#: and stopping to fix the cooling is the better answer.
DEFAULT_MAX_EASE = 0.5

#: Never pause longer than this in one go, whatever the arithmetic says.
#: A single multi-second pause inside a training step looks like a hang.
MAX_PAUSE_SECONDS = 2.0

#: Minimum seconds between sensor reads. Every read shells out to a
#: vendor tool; doing that per step on a fast model would cost more than
#: the pauses save.
DEFAULT_POLL_SECONDS = 2.0

#: A card is only reported as "over target" once it is this far over.
#: Without a margin, a governor doing its job perfectly -- holding the
#: card at 79.95 °C against an 80 °C target -- reports itself as over
#: target for the entire run, because 80.02 is greater than 80. The
#: number a person reads should mean "it got away from me", not "the
#: float compared the way floats do".
OVER_TARGET_MARGIN_C = 1.0

#: Integral gain, in ease per degree-second. Sized against the plant:
#: a card sitting 2.5 °C over target closes that offset in roughly a
#: minute, which is fast next to a thermal time constant of tens of
#: seconds and slow next to the 2-second poll. Larger and it hunts;
#: smaller and the offset it exists to remove outlives the run.
INTEGRAL_GAIN = 0.0015

#: How far below target a card must fall before the governor eases off.
#: Without a gap, a card sitting exactly on target flips every read.
HYSTERESIS_C = 3.0

#: Smallest power-limit step, as a fraction of the card's default. Small
#: enough to find the right level, large enough that it is reached in a
#: few minutes rather than an hour.
UNDERCLOCK_STEP = 0.05

#: A power limit is never taken below this fraction of default. Under
#: it, most cards lose more throughput than the heat is worth, and some
#: refuse the setting outright.
UNDERCLOCK_FLOOR = 0.60

#: Consecutive easing decisions before the underclocker is asked to act.
#: Short, when underclocking is allowed at all: the measurements in the
#: module docstring put a power cap at a third to a quarter of pausing's
#: cost for the same temperature, so the cheap lever should take over
#: from the expensive one quickly. Not zero, because a limit change
#: outlives the process and a two-second transient is not worth one.
UNDERCLOCK_AFTER = 3

#: Once a power limit is actually holding the temperature down, the
#: pause is paying for the same degrees twice. Ease is capped at this
#: fraction of ``max_ease`` while a limit is applied.
EASE_UNDER_CAP = 0.4

#: Consecutive cool readings before a step of power limit is given back.
RESTORE_AFTER = 20

#: What the governor decided.
RUN = "run"
EASE = "ease"
TRIP = "trip"

_TOOL_TIMEOUT = 10.0


def state_path() -> Path:
    """Where applied hardware changes are recorded.

    On disk rather than in memory because the whole point is to survive
    the process: a run killed with SIGKILL leaves a card underclocked,
    and something has to know what it was before.
    """
    root = os.environ.get("HNX_FUSEBOX_STATE")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".hypernix" / "fusebox-state.json"


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------


@dataclass
class CardReading:
    """One card at one moment. Every measurement may be missing."""

    index: int
    vendor: str
    name: str
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    utilization_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Snapshot:
    """Every sensor, once."""

    timestamp: float = 0.0
    cards: list[CardReading] = field(default_factory=list)
    cpu_celsius: float | None = None

    @property
    def hottest_c(self) -> float | None:
        """The hottest *GPU*, or None if no card reported a temperature.

        The CPU is deliberately not in here. It is reported, and it can
        trip the breaker through :attr:`cpu_celsius`, but it must not
        drive the governor: a CPU at 85 °C during data loading is
        normal and would have the governor easing a GPU that is cold.
        """
        temps = [c.temperature_c for c in self.cards if c.temperature_c is not None]
        return max(temps) if temps else None

    @property
    def hottest_card(self) -> CardReading | None:
        hot = None
        for card in self.cards:
            if card.temperature_c is None:
                continue
            if hot is None or card.temperature_c > (hot.temperature_c or -1):
                hot = card
        return hot

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cards": [c.to_dict() for c in self.cards],
            "cpu_celsius": self.cpu_celsius,
            "hottest_c": self.hottest_c,
        }


def read_snapshot() -> Snapshot:
    """Read every sensor once. Never raises.

    A monitoring layer that can throw is a monitoring layer that takes
    the run down with it, which is precisely backwards.
    """
    cards: list[CardReading] = []
    try:
        from . import gpus

        for card in gpus.detect():
            cards.append(CardReading(
                index=card.index,
                vendor=str(card.vendor),
                name=card.name,
                temperature_c=card.temperature_c,
                power_w=card.power_w,
                power_limit_w=card.power_limit_w,
                utilization_pct=card.utilization_pct,
            ))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("fusebox: GPU read failed: %s", exc)

    cpu: float | None = None
    try:
        from ..monitoring import thermometer

        cpu = thermometer.read_cpu_temp()[0]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("fusebox: CPU read failed: %s", exc)

    return Snapshot(timestamp=time.time(), cards=cards, cpu_celsius=cpu)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    """The numbers. Everything the governor does comes from here."""

    target_c: float = DEFAULT_TARGET_C
    trip_c: float = DEFAULT_TRIP_C
    reset_c: float = DEFAULT_RESET_C
    #: A CPU above this trips the breaker too. Off by default: the CPU
    #: being hot during data loading is normal and is not the run's
    #: problem to solve.
    cpu_trip_c: float | None = None
    max_ease: float = DEFAULT_MAX_EASE
    poll_seconds: float = DEFAULT_POLL_SECONDS
    hysteresis_c: float = HYSTERESIS_C
    #: Lower power limits under sustained heat. Off by default: it needs
    #: privileges and it outlives the process.
    auto_underclock: bool = False
    underclock_floor: float = UNDERCLOCK_FLOOR
    underclock_step: float = UNDERCLOCK_STEP
    #: Cards to manage: "all", or indices like "0,2".
    cards: str = "all"
    #: With torch, cap the process's own CPU threads while easing. Job
    #: scoped and reversible, unlike anything that touches the governor
    #: or the machine's frequency scaling.
    limit_cpu_threads: bool = False

    def validate(self) -> None:
        if self.trip_c <= self.target_c:
            raise ValueError(
                f"trip_c ({self.trip_c}) must be above target_c "
                f"({self.target_c}): the trip is the emergency, the "
                f"target is where the run should sit."
            )
        if self.reset_c >= self.trip_c:
            raise ValueError(
                f"reset_c ({self.reset_c}) must be below trip_c "
                f"({self.trip_c}), or a tripped breaker never resets."
            )
        if not 0.0 < self.max_ease <= 4.0:
            raise ValueError(
                f"max_ease ({self.max_ease}) is a fraction of a step's "
                f"duration and must be in (0, 4]."
            )
        if not 0.1 <= self.underclock_floor <= 1.0:
            raise ValueError(
                f"underclock_floor ({self.underclock_floor}) is a fraction "
                f"of the card's default power limit and must be in [0.1, 1.0]."
            )
        if self.poll_seconds < 0:
            raise ValueError("poll_seconds cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        known = {f for f in cls.__dataclass_fields__}
        policy = cls(**{k: v for k, v in data.items() if k in known})
        policy.validate()
        return policy


# ---------------------------------------------------------------------------
# The governor
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    """What to do about the last snapshot."""

    action: str = RUN
    pause_seconds: float = 0.0
    reason: str = ""
    hottest_c: float | None = None
    ease: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Governor:
    """Decides, from temperatures, how hard the run may push.

    Proportional plus a slow, clamped integral term.

    The proportional half is the whole of the response to a change and
    is what makes it quick. It is also, on its own, wrong: a pure
    proportional controller settles wherever its output happens to
    balance the error, which for a 80 °C target with these gains is
    about 82.5 °C -- for ever. That is not a subtlety, it is the
    difference between the temperature someone asked for and the one
    they get, and it showed up the first time this was run against a
    thermal model rather than a unit test.

    So there is an integral term: :attr:`bias`, accumulated at
    :data:`INTEGRAL_GAIN` per degree-second and clamped to the same
    ceiling as the ease itself. The clamp is the whole of the anti-windup
    story and it is why a slow plant is safe here -- the term cannot
    grow past a value the controller could have produced anyway, so
    there is nothing stored up to unwind when the card finally cools.
    """

    def __init__(self, policy: Policy | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy or Policy()
        self.policy.validate()
        self._clock = clock
        #: 0.0 = full speed; 0.5 = pause for half of each step's duration.
        self.ease = 0.0
        #: The integral term. Clamped to the same ceiling as `ease`, so
        #: it can never store up more than the controller could have
        #: produced on its own -- which is the anti-windup.
        self.bias = 0.0
        #: Ceiling on :attr:`ease`, as a fraction of ``policy.max_ease``.
        #: Lowered to :data:`EASE_UNDER_CAP` once a power limit is
        #: holding the temperature, because pausing on top of a cap pays
        #: for the same degrees twice -- and pausing is the dearer of
        #: the two by three to four times.
        self.ease_ceiling = 1.0
        self.tripped = False
        self._hot_streak = 0
        self._cool_streak = 0
        # Accounting, so the claim can be checked.
        self.steps = 0
        self.eased_seconds = 0.0
        self.tripped_seconds = 0.0
        self.trips = 0
        self.readings = 0
        self.seconds_over_target = 0.0
        self.degree_seconds_over_target = 0.0
        self.peak_c: float | None = None
        self._last_reading_at: float | None = None
        self._started_at: float | None = None

    @property
    def _max_ease(self) -> float:
        return self.policy.max_ease * self.ease_ceiling

    # -- the decision ----------------------------------------------------

    def observe(self, snapshot: Snapshot) -> Verdict:
        """Fold a snapshot into the controller and say what to do."""
        self.readings += 1
        hot = snapshot.hottest_c
        now = self._clock()
        if self._started_at is None:
            self._started_at = now
        elapsed = 0.0 if self._last_reading_at is None else now - self._last_reading_at
        self._last_reading_at = now

        if hot is not None:
            self.peak_c = hot if self.peak_c is None else max(self.peak_c, hot)
            over_now = hot - self.policy.target_c
            if over_now > OVER_TARGET_MARGIN_C:
                self.seconds_over_target += elapsed
            if over_now > 0:
                # The magnitude, kept separately. Seconds alone cannot
                # tell 0.1 °C over for an hour from 9 °C over for an
                # hour, and those are very different runs.
                self.degree_seconds_over_target += over_now * elapsed

        cpu_trip = (
            self.policy.cpu_trip_c is not None
            and snapshot.cpu_celsius is not None
            and snapshot.cpu_celsius >= self.policy.cpu_trip_c
        )

        if hot is None and not cpu_trip:
            # No temperature from anything. Not an error and not a
            # reason to throttle: plenty of machines have no sensor the
            # vendor tools will report, and a governor that eases a run
            # it cannot measure is doing harm on no evidence.
            self.ease = 0.0
            self.bias = 0.0
            self.tripped = False
            return Verdict(RUN, 0.0, "no temperature reading", None, 0.0)

        # The fuse, checked before anything else.
        if self.tripped:
            still_hot = hot is not None and hot > self.policy.reset_c
            if still_hot or cpu_trip:
                where = "GPU" if still_hot else "CPU"
                return Verdict(
                    TRIP, self.policy.poll_seconds,
                    f"breaker open: {where} at "
                    f"{(hot if still_hot else snapshot.cpu_celsius):.0f}°C, "
                    f"waiting for {self.policy.reset_c:.0f}°C",
                    hot, self.ease,
                )
            self.tripped = False
            # Not straight back to full speed: whatever got the card to
            # the trip point is still true, so the run resumes eased and
            # works its way down from there.
            self.ease = self._max_ease
            return Verdict(
                EASE, 0.0,
                f"breaker reset at {hot:.0f}°C, resuming eased" if hot is not None
                else "breaker reset, resuming eased",
                hot, self.ease,
            )

        if (hot is not None and hot >= self.policy.trip_c) or cpu_trip:
            self.tripped = True
            self.trips += 1
            source = f"GPU {hot:.0f}°C" if (hot is not None and hot >= self.policy.trip_c) \
                else f"CPU {snapshot.cpu_celsius:.0f}°C"
            return Verdict(
                TRIP, self.policy.poll_seconds,
                f"breaker tripped: {source} at or above "
                f"{self.policy.trip_c:.0f}°C",
                hot, self.ease,
            )

        # Below the fuse: proportional easing around the target.
        assert hot is not None  # cpu_trip alone is handled above
        over = hot - self.policy.target_c
        if over > 0:
            self._hot_streak += 1
            self._cool_streak = 0
            # Proportional: full ease by the time the card is halfway
            # from target to trip. Reaching the trip point should mean
            # the easing was not enough, not that it had not started.
            span = max(1.0, (self.policy.trip_c - self.policy.target_c) / 2.0)
            proportional = self._max_ease * over / span
            # Integral: what closes the last couple of degrees. Without
            # it the card settles above target and stays there.
            self.bias = min(self._max_ease,
                            self.bias + INTEGRAL_GAIN * over * elapsed)
            self.ease = min(self._max_ease,
                            max(self.ease, proportional + self.bias))
            return Verdict(
                EASE, 0.0,
                f"{hot:.0f}°C is {over:.0f}° over target",
                hot, self.ease,
            )

        if hot < self.policy.target_c - self.policy.hysteresis_c:
            self._cool_streak += 1
            self._hot_streak = 0
            if self.ease > 0:
                # Back off gradually. Dropping straight to zero puts the
                # full load back on a card that is only just cool, and
                # the next read finds it hot again -- the oscillation
                # this is meant to prevent.
                self.ease = max(0.0, self.ease - self.policy.max_ease / 4.0)
                # The integral term unwinds with it. It is bounded, so
                # there is never much to unwind, but a bias left behind
                # would put the ease straight back up on the next read
                # that is even slightly over target.
                self.bias = min(self.bias, self.ease)
                return Verdict(
                    EASE if self.ease > 0 else RUN, 0.0,
                    f"{hot:.0f}°C, easing off", hot, self.ease,
                )
            return Verdict(RUN, 0.0, f"{hot:.0f}°C, full speed", hot, 0.0)

        # In the dead band: hold whatever we are doing.
        return Verdict(
            EASE if self.ease > 0 else RUN, 0.0,
            f"{hot:.0f}°C, holding", hot, self.ease,
        )

    # -- what the training loop calls ------------------------------------

    def pause_for(self, step_seconds: float) -> float:
        """Seconds to pause after a step that took *step_seconds*.

        Proportional to the step, so the same policy behaves the same on
        a model whose steps take 30 ms and one whose steps take 3 s.
        """
        self.steps += 1
        if self.ease <= 0 or step_seconds <= 0:
            return 0.0
        return min(MAX_PAUSE_SECONDS, step_seconds * self.ease)

    # -- underclocking hints ---------------------------------------------

    def wants_underclock(self) -> bool:
        """Whether the heat has been sustained long enough to act on."""
        return (
            self.policy.auto_underclock
            and self.ease >= self._max_ease
            and self._hot_streak >= UNDERCLOCK_AFTER
        )

    def wants_restore(self) -> bool:
        """Whether it has been cool long enough to give a step back."""
        return (
            self.policy.auto_underclock
            and self.ease <= 0.0
            and self._cool_streak >= RESTORE_AFTER
        )

    def note_underclock(self) -> None:
        self._hot_streak = 0

    def note_restore(self) -> None:
        self._cool_streak = 0

    # -- accounting -------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """What it actually did, in seconds.

        Written to be checkable rather than flattering. On a
        well-cooled card every number here is zero, which is the correct
        outcome and should look like one.
        """
        return {
            "steps": self.steps,
            "readings": self.readings,
            "eased_seconds": round(self.eased_seconds, 2),
            "tripped_seconds": round(self.tripped_seconds, 2),
            "trips": self.trips,
            # Seconds more than OVER_TARGET_MARGIN_C over; the
            # degree-seconds alongside it carry the magnitude.
            "seconds_over_target": round(self.seconds_over_target, 2),
            "degree_seconds_over_target": round(
                self.degree_seconds_over_target, 1),
            "peak_c": self.peak_c,
            "ease": round(self.ease, 3),
            "bias": round(self.bias, 3),
            "ease_ceiling": round(self.ease_ceiling, 3),
            "target_c": self.policy.target_c,
            "trip_c": self.policy.trip_c,
            # The number to judge this by. Pausing is the dearest lever
            # here -- see the module docstring -- so what it cost should
            # be a headline figure, not something to derive.
            "cost_pct": self._cost_pct(),
        }

    def _cost_pct(self) -> float | None:
        """Pauses as a percentage of the run so far, or None.

        None rather than 0 before there is a run to measure: a
        percentage of nothing is not zero, it is unanswerable.
        """
        if self._last_reading_at is None or self._started_at is None:
            return None
        elapsed = self._last_reading_at - self._started_at
        if elapsed <= 0:
            return None
        lost = self.eased_seconds + self.tripped_seconds
        return round(100.0 * lost / elapsed, 2)

    def summary(self) -> str:
        """One sentence, honest about having done nothing."""
        if self.trips == 0 and self.eased_seconds <= 0:
            peak = f"peaked at {self.peak_c:.0f}°C" if self.peak_c is not None \
                else "no temperature readings"
            return (
                f"fusebox did not intervene: {peak}, target "
                f"{self.policy.target_c:.0f}°C. It cost the run nothing."
            )
        parts = []
        if self.eased_seconds > 0:
            parts.append(f"eased for {self.eased_seconds:.0f}s")
        if self.trips:
            parts.append(
                f"tripped {self.trips}× ({self.tripped_seconds:.0f}s stopped)"
            )
        peak = f", peak {self.peak_c:.0f}°C" if self.peak_c is not None else ""
        cost = self._cost_pct()
        # Stated, not implied. Pausing costs throughput -- that is what
        # it is for -- and a summary that reported only the temperature
        # would be selling the half of the trade that flatters it.
        priced = "" if cost is None else f" That cost {cost:.1f}% of the run."
        return f"fusebox {', '.join(parts)}{peak}.{priced}"


# ---------------------------------------------------------------------------
# The actuator
# ---------------------------------------------------------------------------


class ClockControlError(RuntimeError):
    """A vendor tool refused. Carries what it said."""


@dataclass
class _Limits:
    """A card's power limits, in watts."""

    index: int
    vendor: str
    current_w: float | None = None
    default_w: float | None = None
    min_w: float | None = None
    max_w: float | None = None


class ClockControl:
    """Lowers and restores power limits, and remembers what it changed.

    Every command this can issue is a formatted vendor invocation from
    the table below. There is no path by which a string from a config
    file, an environment variable or a network response becomes part of
    an argument list: the index is an int, the wattage is a float, and
    both are formatted, never interpolated from text.

    ``dry_run`` is the default. Constructing this object reads; it takes
    an explicit ``dry_run=False`` before it writes anything, which is
    what ``--underclock --yes`` on the command line supplies.
    """

    def __init__(self, *, dry_run: bool = True,
                 state_file: Path | None = None) -> None:
        self.dry_run = dry_run
        self.state_file = state_file or state_path()
        #: index -> original watts, for restoring.
        self.original: dict[int, float] = {}
        #: index -> what we last set it to.
        self.applied: dict[int, float] = {}
        self.attempted: list[str] = []
        self._denied_reported = False

    # -- reading ----------------------------------------------------------

    def available(self) -> bool:
        """Whether any vendor tool that can set a limit is present."""
        return bool(self._nvidia_smi() or self._amd_tool()[0])

    def limits(self) -> list[_Limits]:
        """Per-card power limits, from whichever vendor tools answer."""
        out: list[_Limits] = []
        out.extend(self._nvidia_limits())
        out.extend(self._amd_limits())
        return out

    def _nvidia_limits(self) -> list[_Limits]:
        tool = self._nvidia_smi()
        if not tool:
            return []
        text = _run([
            tool,
            "--query-gpu=index,power.limit,power.default_limit,"
            "power.min_limit,power.max_limit",
            "--format=csv,noheader,nounits",
        ])
        found: list[_Limits] = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                index = int(float(parts[0]))
            except ValueError:
                continue
            found.append(_Limits(
                index=index,
                vendor="nvidia",
                current_w=_watts(parts[1]),
                default_w=_watts(parts[2]),
                min_w=_watts(parts[3]),
                max_w=_watts(parts[4]),
            ))
        return found

    def _amd_limits(self) -> list[_Limits]:
        """AMD's, from whatever the installed tool reports.

        AMD does not expose a "default limit" the way NVIDIA does, so
        the limit observed the first time a card is touched is treated
        as its default and is what a restore returns it to. That is
        weaker than reading it from the driver and is stated rather than
        hidden: if something else lowered the cap before the run
        started, restoring puts back what was there at the start of the
        run, not the factory value.
        """
        try:
            from . import gpus

            cards = [c for c in gpus.detect() if str(c.vendor) == "amd"]
        except Exception:  # pragma: no cover - defensive
            return []
        found = []
        for card in cards:
            baseline = self.original.get(card.index, card.power_limit_w)
            found.append(_Limits(
                index=card.index,
                vendor="amd",
                current_w=card.power_limit_w,
                default_w=baseline,
                min_w=None if baseline is None else baseline * UNDERCLOCK_FLOOR,
                max_w=baseline,
            ))
        return found

    # -- writing ----------------------------------------------------------

    def set_power_limit(self, index: int, watts: float, *,
                        vendor: str = "nvidia") -> bool:
        """Set one card's limit. Returns whether it was applied.

        Refuses upward of the recorded default. Not as a safety net for
        a caller's arithmetic -- as the module's promise: this never
        overclocks anything, and the check lives at the one place that
        could break it.
        """
        limits = {lim.index: lim for lim in self.limits()}
        lim = limits.get(index)
        ceiling = None
        if lim is not None:
            ceiling = lim.default_w if lim.default_w is not None else lim.max_w
        ceiling = self.original.get(index, ceiling)
        if ceiling is not None and watts > ceiling + 0.5:
            raise ClockControlError(
                f"refusing to set card {index} to {watts:.0f}W: that is above "
                f"its default limit of {ceiling:.0f}W. fusebox lowers limits "
                f"and puts them back; it does not raise them."
            )
        floor = lim.min_w if lim is not None and lim.min_w is not None else None
        if floor is not None and watts < floor:
            watts = floor

        if index not in self.original and lim is not None and lim.current_w is not None:
            self.original[index] = (
                lim.default_w if lim.default_w is not None else lim.current_w
            )

        argv = self._set_argv(index, watts, vendor)
        if argv is None:
            return False
        self.attempted.append(" ".join(argv))
        if self.dry_run:
            logger.info("fusebox (dry run): would run %s", " ".join(argv))
            return False

        ok, message = _run_checked(argv)
        if not ok:
            if _is_permission_error(message):
                if not self._denied_reported:
                    self._denied_reported = True
                    logger.warning(
                        "fusebox: %s refused to set a power limit for want of "
                        "privileges. Underclocking is off for this run; the "
                        "governor will keep pacing the run instead, which "
                        "needs none. fusebox does not escalate on its own. "
                        "Setting a power limit needs root on every current "
                        "driver, so if you want it, run `hnx fusebox watch "
                        "--underclock --yes` as root yourself -- and note "
                        "that it is the trainer, not this warning, that "
                        "decides whether that is worth doing.",
                        argv[0],
                    )
                return False
            logger.warning("fusebox: %s failed: %s", argv[0], message.strip())
            return False

        self.applied[index] = watts
        self._save()
        return True

    def _set_argv(self, index: int, watts: float, vendor: str) -> list[str] | None:
        if vendor == "nvidia":
            tool = self._nvidia_smi()
            if not tool:
                return None
            return [tool, "-i", str(int(index)), f"--power-limit={watts:.0f}"]
        if vendor == "amd":
            tool, kind = self._amd_tool()
            if not tool:
                return None
            if kind == "amd-smi":
                return [tool, "set", "-g", str(int(index)),
                        "--power-cap", f"{watts:.0f}"]
            return [tool, "-d", str(int(index)),
                    "--setpoweroverdrive", f"{watts:.0f}"]
        return None

    def restore(self, index: int, *, vendor: str = "nvidia") -> bool:
        """Put one card's limit back to what it was."""
        original = self.original.get(index)
        if original is None:
            return False
        argv = self._set_argv(index, original, vendor)
        if argv is None:
            return False
        if self.dry_run:
            logger.info("fusebox (dry run): would run %s", " ".join(argv))
            return False
        ok, message = _run_checked(argv)
        if not ok:
            logger.warning(
                "fusebox: could not restore card %s to %.0fW: %s. Run "
                "`hnx fusebox restore` once the tool will accept it.",
                index, original, message.strip(),
            )
            return False
        self.applied.pop(index, None)
        self._save()
        return True

    def restore_all(self) -> int:
        """Put every card back. Returns how many were restored."""
        vendors = {lim.index: lim.vendor for lim in self.limits()}
        done = 0
        for index in list(self.original):
            if self.restore(index, vendor=vendors.get(index, "nvidia")):
                done += 1
        return done

    # -- persistence -------------------------------------------------------

    def _save(self) -> None:
        """Record what is changed, so a crash is recoverable.

        Written every time rather than at exit: the case this exists for
        is the process not reaching its exit.
        """
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "pid": os.getpid(),
                "written_at": time.time(),
                "original": {str(k): v for k, v in self.original.items()},
                "applied": {str(k): v for k, v in self.applied.items()},
            }
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug("fusebox: could not write state: %s", exc)

    def load_state(self) -> bool:
        """Read a previous run's changes. True if there were any."""
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        original = data.get("original") or {}
        applied = data.get("applied") or {}
        try:
            self.original = {int(k): float(v) for k, v in original.items()}
            self.applied = {int(k): float(v) for k, v in applied.items()}
        except (TypeError, ValueError):
            return False
        return bool(self.applied)

    def clear_state(self) -> None:
        try:
            self.state_file.unlink()
        except OSError:
            pass

    # -- tools -------------------------------------------------------------

    @staticmethod
    def _nvidia_smi() -> str:
        from . import gpus

        return gpus._NVIDIA.resolve()

    @staticmethod
    def _amd_tool() -> tuple[str, str]:
        from . import gpus

        found = gpus._AMD_SMI.resolve()
        if found:
            return found, "amd-smi"
        return gpus._ROCM_SMI.resolve(), "rocm-smi"


def restore_all(state_file: Path | None = None) -> int:
    """Undo whatever a previous run left applied. Returns the count.

    This is what ``hnx fusebox restore`` calls, and the reason the state
    file exists: a run killed outright leaves a card at a lower limit
    and nothing in the process to put it back.
    """
    control = ClockControl(dry_run=False, state_file=state_file)
    if not control.load_state():
        return 0
    done = control.restore_all()
    if done:
        control.clear_state()
    return done


def _watts(text: str) -> float | None:
    cleaned = text.strip().rstrip("W").strip()
    if not cleaned or cleaned.lower() in ("n/a", "[n/a]", "not supported", "unknown"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _run(argv: list[str], *, timeout: float = _TOOL_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("fusebox: %s failed: %s", argv[0], exc)
        return ""
    return result.stdout if result.returncode == 0 else ""


def _run_checked(argv: list[str], *, timeout: float = _TOOL_TIMEOUT) -> tuple[bool, str]:
    """Run a vendor tool; return (ok, combined output).

    Unlike :func:`_run`, the failure text is kept: "permission denied"
    and "unsupported on this card" want different responses, and both
    arrive as a non-zero exit.
    """
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def _is_permission_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in (
        "permission denied", "insufficient permission", "not permitted",
        "requires root", "must be run as", "access denied", "eperm",
    ))


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


class FuseBox:
    """Sensors, governor and actuator, wired together.

    The object a training loop holds. It reads no more often than
    ``policy.poll_seconds`` however often :meth:`pace` is called, so a
    model whose steps take 20 ms does not shell out to ``nvidia-smi``
    fifty times a second.
    """

    def __init__(self, policy: Policy | None = None, *,
                 apply_changes: bool = False,
                 state_file: Path | None = None,
                 sensor: Callable[[], Snapshot] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.policy = policy or Policy()
        self.policy.validate()
        self.governor = Governor(self.policy, clock=clock)
        self.control = ClockControl(
            dry_run=not apply_changes, state_file=state_file,
        )
        self._sensor = sensor or read_snapshot
        self._clock = clock
        self._sleep = sleep
        self.snapshot = Snapshot()
        self.verdict = Verdict()
        self._last_poll: float | None = None
        self._last_step_at: float | None = None
        self._lock = threading.Lock()
        self._original_threads: int | None = None
        self._threads_limited = False

    # -- the training-loop hook -------------------------------------------

    def pace(self, step_seconds: float | None = None) -> float:
        """Call once per training step. Returns the seconds it paused.

        *step_seconds* is how long the step took; left out, it is
        measured from the previous call, which is what a loop that just
        calls ``box.pace()`` at the bottom of its body gets.
        """
        now = self._clock()
        if step_seconds is None:
            step_seconds = 0.0 if self._last_step_at is None else now - self._last_step_at
        self._last_step_at = now

        self.poll()

        if self.verdict.action == TRIP:
            return self._hold_until_cool()

        pause = self.governor.pause_for(step_seconds)
        if pause > 0:
            self.governor.eased_seconds += pause
            self._sleep(pause)
            self._last_step_at = self._clock()
        return pause

    def poll(self, *, force: bool = False) -> Verdict:
        """Read the sensors if it is time, and decide.

        Rate-limited: the sensors cost a subprocess each and the thing
        being measured has a time constant of tens of seconds, so
        reading faster buys nothing and costs real time.
        """
        with self._lock:
            now = self._clock()
            due = (
                force
                or self._last_poll is None
                or (now - self._last_poll) >= self.policy.poll_seconds
            )
            if not due:
                return self.verdict
            self._last_poll = now
            self.snapshot = self._sensor()
            self.verdict = self.governor.observe(self.snapshot)
            self._maybe_adjust_clocks()
            self._maybe_adjust_threads()
            return self.verdict

    def _hold_until_cool(self) -> float:
        """Sit out a tripped breaker, re-reading until it resets.

        A loop rather than one long sleep: a card that cools quickly
        should get the run back quickly, and a single sleep sized for
        the worst case is the worst case every time.
        """
        waited = 0.0
        while True:
            self._sleep(max(0.1, self.policy.poll_seconds))
            waited += max(0.1, self.policy.poll_seconds)
            self.governor.tripped_seconds += max(0.1, self.policy.poll_seconds)
            verdict = self.poll(force=True)
            if verdict.action != TRIP:
                break
            if waited > 3600:
                # An hour above the trip point is not a thermal
                # transient, it is a broken fan. Handing the run back is
                # wrong, and so is blocking forever without saying why.
                raise ThermalStall(
                    f"fusebox: still above {self.policy.trip_c:.0f}°C after an "
                    f"hour paused (now {verdict.hottest_c}). This is not a "
                    f"transient -- check cooling."
                )
        self._last_step_at = self._clock()
        return waited

    # -- actuation ---------------------------------------------------------

    def _maybe_adjust_clocks(self) -> None:
        if not self.policy.auto_underclock:
            return
        # A power limit that is actually applied is holding some of the
        # temperature down, so the pause should stand down to match.
        # Both levers at full is paying twice for the same degrees, and
        # the pause is the dearer by three to four times.
        self.governor.ease_ceiling = (
            EASE_UNDER_CAP if self.control.applied else 1.0
        )
        if self.governor.wants_underclock():
            self._underclock_step()
            self.governor.note_underclock()
        elif self.governor.wants_restore():
            self._restore_step()
            self.governor.note_restore()

    def _underclock_step(self) -> None:
        card = self.snapshot.hottest_card
        if card is None:
            return
        limits = {lim.index: lim for lim in self.control.limits()}
        lim = limits.get(card.index)
        if lim is None or lim.current_w is None:
            return
        baseline = lim.default_w if lim.default_w is not None else lim.current_w
        if baseline is None:
            return
        floor = baseline * self.policy.underclock_floor
        target = max(floor, lim.current_w - baseline * self.policy.underclock_step)
        if target >= lim.current_w - 0.5:
            return  # already at the floor
        try:
            applied = self.control.set_power_limit(
                card.index, target, vendor=card.vendor,
            )
        except ClockControlError as exc:  # pragma: no cover - guarded above
            logger.warning("fusebox: %s", exc)
            return
        if applied:
            logger.info(
                "fusebox: card %s (%s) power limit %.0fW -> %.0fW, sustained "
                "%.0f°C over target",
                card.index, card.name, lim.current_w, target,
                (card.temperature_c or 0) - self.policy.target_c,
            )

    def _restore_step(self) -> None:
        for index, original in list(self.control.original.items()):
            current = self.control.applied.get(index)
            if current is None or current >= original - 0.5:
                continue
            step = original * self.policy.underclock_step
            target = min(original, current + step)
            vendor = next(
                (c.vendor for c in self.snapshot.cards if c.index == index),
                "nvidia",
            )
            try:
                if self.control.set_power_limit(index, target, vendor=vendor):
                    logger.info(
                        "fusebox: card %s cool, power limit %.0fW -> %.0fW",
                        index, current, target,
                    )
            except ClockControlError as exc:  # pragma: no cover
                logger.warning("fusebox: %s", exc)
            break  # one card per cool period, so it walks back gently

    def _maybe_adjust_threads(self) -> None:
        """Cap this process's CPU threads while easing, if asked.

        The one CPU-side lever here, and it is job-scoped: it changes
        how many threads *this* process uses, not the machine's
        frequency scaling or governor. Turning a laptop's clocks down
        because a training run is hot would slow down everything else
        the person is doing, and they did not ask for that.
        """
        if not self.policy.limit_cpu_threads:
            return
        try:
            import torch
        except Exception:
            return
        easing = self.verdict.action in (EASE, TRIP) and self.governor.ease > 0
        if easing and not self._threads_limited:
            self._original_threads = torch.get_num_threads()
            reduced = max(1, self._original_threads // 2)
            torch.set_num_threads(reduced)
            self._threads_limited = True
            logger.info("fusebox: CPU threads %d -> %d while easing",
                        self._original_threads, reduced)
        elif not easing and self._threads_limited and self._original_threads:
            torch.set_num_threads(self._original_threads)
            self._threads_limited = False
            logger.info("fusebox: CPU threads restored to %d",
                        self._original_threads)

    # -- lifecycle ---------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator[FuseBox]:
        """Manage a run, and put everything back afterwards.

        The ``finally`` is the point. A run that raises, or is
        interrupted, must not leave a card underclocked -- and because
        SIGKILL reaches no ``finally`` at all, the state file exists
        too, and ``hnx fusebox restore`` reads it.
        """
        self.poll(force=True)
        try:
            yield self
        finally:
            self.close()

    def close(self) -> None:
        restored = 0
        if not self.control.dry_run:
            restored = self.control.restore_all()
            if restored and not self.control.applied:
                self.control.clear_state()
        if self._threads_limited and self._original_threads:
            try:
                import torch

                torch.set_num_threads(self._original_threads)
            except Exception:  # pragma: no cover - defensive
                pass
            self._threads_limited = False
        if restored:
            logger.info("fusebox: restored %d card(s) to their original limits",
                        restored)

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "verdict": self.verdict.to_dict(),
            "governor": self.governor.report(),
            "underclock": {
                "enabled": self.policy.auto_underclock,
                "dry_run": self.control.dry_run,
                "available": self.control.available(),
                "original_w": dict(self.control.original),
                "applied_w": dict(self.control.applied),
            },
        }

    def summary(self) -> str:
        return self.governor.summary()


class ThermalStall(RuntimeError):
    """The breaker stayed open long enough that it is not a transient."""


def watch(policy: Policy | None = None, *, apply_changes: bool = False,
          iterations: int | None = None,
          on_tick: Callable[[FuseBox], None] | None = None,
          sensor: Callable[[], Snapshot] | None = None,
          sleep: Callable[[float], None] = time.sleep) -> FuseBox:
    """Run the panel on its own, with no training loop to pace.

    What ``hnx fusebox watch`` calls. Useful when the training is in
    another process (or is llama.cpp, or is not Python at all): this
    cannot pause someone else's steps, so it manages temperature with
    the power limit alone and reports the rest.
    """
    box = FuseBox(policy, apply_changes=apply_changes, sensor=sensor, sleep=sleep)
    count = 0
    try:
        while iterations is None or count < iterations:
            box.poll(force=True)
            if on_tick is not None:
                on_tick(box)
            count += 1
            if iterations is None or count < iterations:
                sleep(max(0.1, box.policy.poll_seconds))
    except KeyboardInterrupt:
        pass
    finally:
        box.close()
    return box
