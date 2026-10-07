"""timer — countdown / interval / pomodoro helpers.

Four tiers, all sharing the same ``BaseTimer`` interface
(:meth:`start` / :meth:`elapsed` / :meth:`expired` /
:meth:`should_fire` / :meth:`reset`):

* :class:`KitchenTimer`  — t1.  Plain countdown.  ``expired()`` flips
                                  to True after ``duration`` seconds.
* :class:`EggTimer`      — t2.  KitchenTimer + an explicit ``rang``
                                  flag and ``on_ring`` callback so a
                                  caller can register "what to do
                                  when the timer goes off".
* :class:`IntervalTimer` — t3.  Fires every ``interval_seconds`` —
                                  used to throttle log lines, save
                                  cadence, eval cadence, etc.
                                  ``should_fire()`` advances the
                                  internal next-fire deadline.
* :class:`PomodoroTimer` — t4.  Alternates between work / rest
                                  blocks.  ``state`` returns
                                  ``"work" | "rest"`` and the timer
                                  cycles automatically.

All four work on a monotonic clock so they're robust against
wall-time jumps: ``time.perf_counter``, not ``time.monotonic``. Before
Python 3.13 Windows' monotonic clock ticks every ~15.6 ms, so a short
interval could not be seen to elapse for up to a tick; perf_counter is
monotonic too, and high-resolution everywhere.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BaseTimer:
    duration: float = 60.0
    started_at: float | None = field(default=None, init=False)

    def start(self) -> BaseTimer:
        self.started_at = time.perf_counter()
        return self

    def reset(self) -> BaseTimer:
        self.started_at = None
        return self

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.perf_counter() - self.started_at

    def expired(self) -> bool:
        return self.started_at is not None and self.elapsed() >= self.duration

    def remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed())


# ---------------------------------------------------------------------------
# Tier 1 — KitchenTimer
# ---------------------------------------------------------------------------

@dataclass
class KitchenTimer(BaseTimer):
    """Plain countdown timer."""

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("duration must be >= 0")


# ---------------------------------------------------------------------------
# Tier 2 — EggTimer
# ---------------------------------------------------------------------------

@dataclass
class EggTimer(KitchenTimer):
    """Countdown with an explicit ``rang`` flag plus an ``on_ring``
    callback fired exactly once when the timer first expires."""

    on_ring: Callable[[], None] | None = None
    rang: bool = field(default=False, init=False)

    def check(self) -> bool:
        """Poll the timer.  When it first crosses ``duration`` seconds,
        fire ``on_ring`` (if set) and flip ``rang``.  Returns the
        new value of ``rang``."""
        if not self.rang and self.expired():
            self.rang = True
            if self.on_ring is not None:
                self.on_ring()
        return self.rang

    def reset(self) -> EggTimer:
        super().reset()
        self.rang = False
        return self


# ---------------------------------------------------------------------------
# Tier 3 — IntervalTimer
# ---------------------------------------------------------------------------

@dataclass
class IntervalTimer(BaseTimer):
    """Fires every ``interval_seconds``.  Use ``should_fire()`` inside
    a training loop to throttle expensive operations (log emit,
    checkpoint save, eval) without rolling your own time math."""

    interval_seconds: float = 1.0
    next_fire: float | None = field(default=None, init=False)
    fire_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

    def start(self) -> IntervalTimer:
        super().start()
        self.next_fire = self.started_at + self.interval_seconds  # type: ignore[operator]
        return self

    def reset(self) -> IntervalTimer:
        super().reset()
        self.next_fire = None
        self.fire_count = 0
        return self

    def should_fire(self) -> bool:
        """Advance the deadline + return True when the interval has
        elapsed since the last fire.  Idempotent within a single
        interval — calling it 1000x in a tight loop fires once per
        ``interval_seconds`` only."""
        if self.next_fire is None:
            self.start()
            return True
        now = time.perf_counter()
        if now >= self.next_fire:
            # Re-anchor to *now* rather than incrementing the stale
            # deadline. Incrementing (``next_fire += interval_seconds``)
            # keeps a fixed cadence, but if the caller is delayed (e.g. a
            # slow CI runner, GC pause, or a busy training loop) by more
            # than one interval between polls, the incremented deadline
            # can still be <= the time of the *next* poll, causing two
            # fires back-to-back even though only one interval's worth
            # of work should happen. Anchoring to ``now`` guarantees a
            # full fresh interval must elapse before the next fire.
            self.next_fire = now + self.interval_seconds
            self.fire_count += 1
            return True
        return False


# ---------------------------------------------------------------------------
# Tier 4 — PomodoroTimer
# ---------------------------------------------------------------------------

@dataclass
class PomodoroTimer(BaseTimer):
    """Alternates between work and rest blocks.  Default: 25 min work,
    5 min rest (the classic Pomodoro)."""

    work_seconds: float = 25 * 60
    rest_seconds: float = 5 * 60
    cycles: int = field(default=0, init=False)
    _phase_start: float | None = field(default=None, init=False, repr=False)
    _state: str = field(default="work", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.work_seconds <= 0 or self.rest_seconds <= 0:
            raise ValueError("work_seconds and rest_seconds must be > 0")
        # Match BaseTimer's interface; ``duration`` is the current phase.
        self.duration = self.work_seconds

    def start(self) -> PomodoroTimer:
        now = time.perf_counter()
        self.started_at = now
        self._phase_start = now
        self._state = "work"
        self.duration = self.work_seconds
        self.cycles = 0
        return self

    def reset(self) -> PomodoroTimer:
        self.started_at = None
        self._phase_start = None
        self._state = "work"
        self.duration = self.work_seconds
        self.cycles = 0
        return self

    @property
    def state(self) -> str:
        return self._state

    def tick(self) -> str:
        """Advance the timer state if the current phase has elapsed
        and return the (possibly newly switched-to) state."""
        if self._phase_start is None:
            self.start()
            return self._state
        phase_dur = self.work_seconds if self._state == "work" else self.rest_seconds
        if time.perf_counter() - self._phase_start >= phase_dur:
            if self._state == "work":
                self._state = "rest"
                self.duration = self.rest_seconds
            else:
                self._state = "work"
                self.duration = self.work_seconds
                self.cycles += 1
            self._phase_start = time.perf_counter()
        return self._state


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

TIERS: dict[str, type[BaseTimer]] = {
    "kitchen": KitchenTimer,
    "egg": EggTimer,
    "interval": IntervalTimer,
    "pomodoro": PomodoroTimer,
}


def timer(kind: str = "kitchen", **kw) -> BaseTimer:
    if kind not in TIERS:
        raise ValueError(f"unknown timer kind {kind!r}; valid: {sorted(TIERS)}")
    return TIERS[kind](**kw).start()


__all__ = [
    "BaseTimer",
    "EggTimer",
    "IntervalTimer",
    "KitchenTimer",
    "PomodoroTimer",
    "TIERS",
    "timer",
]
