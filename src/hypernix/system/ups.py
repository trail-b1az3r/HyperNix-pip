"""ups — uninterruptible-power-supply mode for training.

Watches two real-world signals and triggers a "checkpoint
panic" when either fires:

1.  **Weather** — heavy rain, severe thunderstorms, or any other
    WMO weather code in :data:`SEVERE_WEATHER_CODES`.  Queries
    open-meteo.com (free, no API key) at the configured
    coordinates.
2.  **Scheduled power outage** — pluggable callback.  Most US /
    European utilities don't expose a public API, so the user
    supplies an ``outage_check_fn(address) -> bool`` that returns
    ``True`` when the utility website lists a scheduled outage
    covering the configured address / window.

When either signal is positive, the UPS:

* **forces** an immediate snapshot via the user-supplied
  ``snapshot_fn``,
* **multiplies** the trainer's checkpoint cadence (default ``3×``)
  via :meth:`adjusted_save_every`,
* **records** every threat in :attr:`history` so a downstream
  log can show the user "we sped up checkpoint cadence at
  17:42:11 because of incoming thunderstorms".

Quick use::

    from hypernix.ups import UPS

    ups = UPS(latitude=47.61, longitude=-122.33)  # Seattle

    save_every = ups.adjusted_save_every(base_save_every=500)
    if ups.threat_active():
        print("UPS:", ups.last_status.summary)

The threat check itself is HTTP — set ``offline=True`` on
construction (or HYPERNIX_UPS_OFFLINE=1) to skip the network
call and only consult the user-supplied ``outage_check_fn``.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

#: WMO weather codes considered "force checkpoint now" severe.
#:
#: 65 = heavy rain                    95 = thunderstorm
#: 66, 67 = freezing rain (heavy)     96 = thunderstorm + slight hail
#: 75 = heavy snow                    99 = thunderstorm + heavy hail
#: 82 = violent rain showers
SEVERE_WEATHER_CODES: frozenset[int] = frozenset({
    65, 66, 67,
    75, 77,
    82,
    95, 96, 99,
})

#: Cap on :attr:`UPS.history`. Enough to see a storm pass through at the
#: default five-minute cadence (~40 hours) without growing without bound.
MAX_HISTORY: int = 500

#: WMO codes treated as "elevated risk" (forces checkpoint but
#: doesn't 3× the cadence).
ELEVATED_WEATHER_CODES: frozenset[int] = frozenset({
    61, 63,    # rain (light, moderate)
    71, 73,    # snow
    80, 81,    # rain showers
})


@dataclass
class ThreatStatus:
    timestamp: float
    weather_code: int | None = None
    weather_severe: bool = False
    weather_elevated: bool = False
    scheduled_outage: bool = False
    summary: str = ""

    @property
    def active(self) -> bool:
        return self.weather_severe or self.weather_elevated or self.scheduled_outage

    @property
    def panic(self) -> bool:
        """Severe weather or a scheduled outage — force an immediate
        checkpoint and 3× the cadence."""
        return self.weather_severe or self.scheduled_outage


# ---------------------------------------------------------------------------
# Weather backend (open-meteo, no key)
# ---------------------------------------------------------------------------

def _query_open_meteo(
    latitude: float, longitude: float, *, timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Return open-meteo's ``current`` block, or ``None`` on any
    error.  Pure stdlib HTTP."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,weather_code,precipitation,wind_speed_10m"
    )
    try:
        with safe_urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    return data.get("current") if isinstance(data, dict) else None


def _autodetect_coords(timeout: float = 5.0) -> tuple[float, float] | None:
    """Best-effort IP-geolocation via ipapi.co (free, no key)."""
    try:
        with safe_urlopen(
            "https://ipapi.co/json/", timeout=timeout,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    lat = data.get("latitude")
    lon = data.get("longitude")
    if isinstance(lat, int | float) and isinstance(lon, int | float):
        return float(lat), float(lon)
    return None


# ---------------------------------------------------------------------------
# UPS
# ---------------------------------------------------------------------------

@dataclass
class UPS:
    """Uninterruptible-power-supply guard.

    Args:
        latitude / longitude: Coordinates used for the weather
            query.  When ``None``, :class:`UPS` tries IP-based
            geolocation via ipapi.co; if that also fails the
            weather check becomes a no-op.
        outage_check_fn: Optional ``(address: str | None) -> bool``
            returning ``True`` when the user's utility lists a
            scheduled outage covering the configured ``address``
            / current time.  When ``None``, scheduled-outage
            detection is disabled.
        address: Free-form address string passed to
            ``outage_check_fn``.
        offline: Skip every network call (or set
            ``HYPERNIX_UPS_OFFLINE=1`` in the env).
        cadence_multiplier: Multiplier applied to the trainer's
            ``save_every`` when a panic is active.  Default 3 →
            "save 3× more often".
        check_interval_seconds: Minimum seconds between weather
            HTTP calls (so a tight training loop doesn't hammer
            open-meteo).  Default 300 (5 minutes).
        snapshot_fn: Optional callable invoked once the moment a
            new panic begins, so the trainer can dump a snapshot
            before the lights go out.
    """

    latitude: float | None = None
    longitude: float | None = None
    outage_check_fn: Callable[[str | None], bool] | None = None
    address: str | None = None
    offline: bool = False
    cadence_multiplier: int = 3
    check_interval_seconds: float = 300.0
    snapshot_fn: Callable[[], None] | None = None

    last_status: ThreatStatus | None = field(default=None, init=False)
    history: list[ThreatStatus] = field(default_factory=list, init=False, repr=False)
    _last_check_at: float = field(default=0.0, init=False, repr=False)

    _coords_resolved: bool = field(default=False, init=False, repr=False)
    _bg_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cached_status: ThreatStatus | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cadence_multiplier < 1:
            raise ValueError("cadence_multiplier must be >= 1")
        if os.environ.get("HYPERNIX_UPS_OFFLINE") == "1":
            self.offline = True
        # Patch (0.61.1): IP-geolocation is *lazy* — defer until the
        # first :meth:`check` so constructing a UPS doesn't block on
        # a 5s HTTPS round-trip.  Tests / quick scripts that pass
        # ``offline=True`` or explicit coords pay no network cost.

    def _ensure_coords(self) -> None:
        if self._coords_resolved:
            return
        self._coords_resolved = True
        if self.offline:
            return
        if self.latitude is not None and self.longitude is not None:
            return
        coords = _autodetect_coords()
        if coords is not None:
            self.latitude, self.longitude = coords

    # ------------------------------------------------------------------
    # Threat polling
    # ------------------------------------------------------------------

    def check(self, *, force: bool = False) -> ThreatStatus:
        """Sample both signals (or return cached status).

        The network call deliberately happens *outside* the lock. It used
        to run inside it, which meant a five-second HTTP timeout was held
        against every other caller — including the background poller,
        which needs the same lock to publish its result. A training loop
        calling :meth:`adjusted_save_every` every step would serialize on
        that. Locking only around the state updates keeps the critical
        section to a few field assignments.
        """
        now = time.time()

        with self._lock:
            if (
                not force
                and self.last_status is not None
                and now - self._last_check_at < self.check_interval_seconds
            ):
                return self.last_status
            cached = self._cached_status

        if not self.offline and not self._background:
            self._start_bg_thread()

        if cached is not None and not force:
            status = cached
        else:
            # Outside the lock: this is the part that can block.
            status = self._perform_sync_check(now)

        with self._lock:
            was_panic = self.last_status is not None and self.last_status.panic
            self.last_status = status
            self._last_check_at = now
            if status.active and (
                not self.history or self.history[-1].timestamp != status.timestamp
            ):
                self.history.append(status)
                # Bounded: a long training run checking every five
                # minutes through a stormy week would otherwise keep every
                # record forever.
                if len(self.history) > MAX_HISTORY:
                    del self.history[: len(self.history) - MAX_HISTORY]
            fire_snapshot = status.panic and not was_panic and self.snapshot_fn is not None

        # Also outside the lock — snapshot_fn is user code that writes
        # checkpoints, and holding a lock across it would be worse than
        # anything it could do on its own.
        if fire_snapshot:
            try:
                threading.Thread(
                    target=self.snapshot_fn, daemon=True, name="hypernix-ups-snapshot",
                ).start()
            except Exception as exc:  # noqa: BLE001
                status.summary += f" (snapshot_fn async trigger failed: {exc})"
        return status

    @property
    def _background(self) -> bool:
        with self._lock:
            return self._bg_thread is not None

    def _start_bg_thread(self) -> None:
        with self._lock:
            if self._bg_thread is not None:
                return
            self._stop_event.clear()
            self._bg_thread = threading.Thread(
                target=self._bg_loop, daemon=True, name="hypernix-ups-poll",
            )
            self._bg_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background poller.

        ``_stop_event`` existed and ``_bg_loop`` honoured it, but nothing
        ever set it — there was no way to stop a UPS once it had started
        polling. The thread is a daemon so a process could still exit,
        but a long-lived one (a server, a notebook) leaked a thread and
        an HTTP request every five minutes per UPS ever constructed.
        """
        self._stop_event.set()
        with self._lock:
            thread = self._bg_thread
            self._bg_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def __enter__(self) -> UPS:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _bg_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            status = self._perform_sync_check(now)
            with self._lock:
                self._cached_status = status
            self._stop_event.wait(self.check_interval_seconds)

    def _perform_sync_check(self, now: float) -> ThreatStatus:
        status = ThreatStatus(timestamp=now)
        # 0.61.1: lazy IP-geolocation on the first check.
        self._ensure_coords()
        # 1. Weather
        if (
            not self.offline
            and self.latitude is not None
            and self.longitude is not None
        ):
            current = _query_open_meteo(self.latitude, self.longitude)
            if current is not None:
                wc = current.get("weather_code")
                if isinstance(wc, int):
                    status.weather_code = wc
                    status.weather_severe = wc in SEVERE_WEATHER_CODES
                    status.weather_elevated = wc in ELEVATED_WEATHER_CODES
        # 2. Scheduled outage hook
        if self.outage_check_fn is not None:
            try:
                status.scheduled_outage = bool(self.outage_check_fn(self.address))
            except Exception:  # noqa: BLE001
                status.scheduled_outage = False

        status.summary = self._summarise(status)
        return status

    def _summarise(self, s: ThreatStatus) -> str:
        bits: list[str] = []
        if s.weather_severe:
            bits.append(f"SEVERE WEATHER (wmo={s.weather_code})")
        elif s.weather_elevated:
            bits.append(f"elevated weather (wmo={s.weather_code})")
        elif s.weather_code is not None:
            bits.append(f"weather=clear (wmo={s.weather_code})")
        if s.scheduled_outage:
            bits.append("SCHEDULED OUTAGE")
        if not bits:
            bits.append("no signal")
        return " ; ".join(bits)

    # ------------------------------------------------------------------
    # Trainer-facing helpers
    # ------------------------------------------------------------------

    def threat_active(self) -> bool:
        s = self.check()
        return s.active

    def panic(self) -> bool:
        s = self.check()
        return s.panic

    def adjusted_save_every(self, base_save_every: int) -> int:
        """Return the effective ``save_every`` given the current
        threat level.  Under panic, divides ``base_save_every`` by
        :attr:`cadence_multiplier` (so the trainer saves N× more
        often).  Floors at 1."""
        if base_save_every <= 0:
            return base_save_every
        if self.panic():
            return max(1, base_save_every // self.cadence_multiplier)
        return base_save_every

    def force_snapshot(self) -> None:
        """Manually fire the configured ``snapshot_fn`` (no-op if
        unset).  Useful for "I'm shutting the laptop now" buttons."""
        if self.snapshot_fn is not None:
            self.snapshot_fn()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def ups(
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    outage_check_fn: Callable[[str | None], bool] | None = None,
    address: str | None = None,
    offline: bool = False,
    cadence_multiplier: int = 3,
) -> UPS:
    return UPS(
        latitude=latitude, longitude=longitude,
        outage_check_fn=outage_check_fn, address=address,
        offline=offline, cadence_multiplier=cadence_multiplier,
    )


def threat_now(
    latitude: float | None = None, longitude: float | None = None,
) -> ThreatStatus:
    """One-shot weather check; offline-fallback returns an inactive status.

    Stops the poller before returning. Without that, a helper documented
    as "one-shot" left a thread behind polling open-meteo every five
    minutes for the life of the process.
    """
    guard = UPS(latitude=latitude, longitude=longitude)
    try:
        return guard.check(force=True)
    finally:
        guard.stop()


# ---------------------------------------------------------------------------
# CLI entry point — installed as ``ups``
#
# The module had no entry point at all: no console script, no cli_main,
# and nothing in the package imported it. Everything below existed and
# worked, and there was no way to run any of it.
# ---------------------------------------------------------------------------

_USAGE = """\
usage: ups [status|watch] [--lat N --lon N] [--offline] [--interval SECONDS]

  ups                 check once and print the current threat status
  ups status          the same, explicitly
  ups watch           keep checking, printing only when the status changes

Options:
  --lat N --lon N     coordinates for the weather check. Omitted, ups asks
                      ipapi.co where this machine is; if that fails too,
                      the weather signal is skipped rather than guessed.
  --address TEXT      passed to a scheduled-outage callback (library use)
  --offline           skip every network call
  --interval SECONDS  seconds between checks in watch mode (default 300)
  --json              machine-readable output

Exit status: 0 when nothing is happening, 1 under an elevated signal, and
2 under a panic signal (severe weather or a scheduled outage) — so a
training script can gate on it:

    ups --offline || echo "storm incoming, checkpoint now"
"""


def _status_dict(status: ThreatStatus) -> dict[str, Any]:
    return {
        "timestamp": status.timestamp,
        "weather_code": status.weather_code,
        "weather_severe": status.weather_severe,
        "weather_elevated": status.weather_elevated,
        "scheduled_outage": status.scheduled_outage,
        "active": status.active,
        "panic": status.panic,
        "summary": status.summary,
    }


def _exit_code(status: ThreatStatus) -> int:
    """0 quiet, 1 elevated, 2 panic — distinct so a caller can react
    proportionately instead of only knowing "something or nothing"."""
    if status.panic:
        return 2
    if status.active:
        return 1
    return 0


def _render(status: ThreatStatus, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_status_dict(status), indent=2))
        return
    marker = "PANIC" if status.panic else ("ALERT" if status.active else "ok")
    stamp = time.strftime("%H:%M:%S", time.localtime(status.timestamp))
    print(f"[{stamp}] {marker:<5} {status.summary}")


def cli_main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0

    command = "status"
    if args and not args[0].startswith("-"):
        command = args.pop(0).lower()
    if command not in ("status", "watch"):
        print(f"ups: unknown command {command!r}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 64

    def _opt(name: str) -> str | None:
        if name not in args:
            return None
        i = args.index(name)
        if i + 1 >= len(args):
            raise _CLIError(f"{name} requires a value")
        return args[i + 1]

    try:
        lat_s, lon_s = _opt("--lat"), _opt("--lon")
        interval_s = _opt("--interval")
        address = _opt("--address")
        latitude = float(lat_s) if lat_s is not None else None
        longitude = float(lon_s) if lon_s is not None else None
        interval = float(interval_s) if interval_s is not None else 300.0
    except _CLIError as exc:
        print(f"ups: {exc}", file=sys.stderr)
        return 64
    except ValueError as exc:
        print(f"ups: expected a number ({exc})", file=sys.stderr)
        return 64

    if (latitude is None) != (longitude is None):
        print("ups: --lat and --lon must be given together", file=sys.stderr)
        return 64

    as_json = "--json" in args
    guard = UPS(
        latitude=latitude,
        longitude=longitude,
        address=address,
        offline="--offline" in args,
        check_interval_seconds=interval,
    )

    try:
        if command == "status":
            status = guard.check(force=True)
            _render(status, as_json=as_json)
            return _exit_code(status)

        # watch: print on change only, so leaving it running overnight
        # produces a readable log rather than one line every interval.
        last_summary: str | None = None
        while True:
            status = guard.check(force=True)
            if status.summary != last_summary:
                _render(status, as_json=as_json)
                last_summary = status.summary
            time.sleep(max(1.0, interval))
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        guard.stop()


class _CLIError(ValueError):
    """Argument-parsing failure; message is user-facing."""


__all__ = [
    "ELEVATED_WEATHER_CODES",
    "MAX_HISTORY",
    "SEVERE_WEATHER_CODES",
    "ThreatStatus",
    "UPS",
    "cli_main",
    "threat_now",
    "ups",
]


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(cli_main())
