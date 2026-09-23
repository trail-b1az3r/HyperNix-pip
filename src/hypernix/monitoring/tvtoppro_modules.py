"""hypernix.monitoring.tvtoppro_modules — extra panels for tvtoppro.

tvtoppro draws five fixed boxes: cpu, mem, gpu, training, proc. They are
the five that matter on a training box and they are not the only five
anybody wants. Disk I/O matters when the dataloader is the bottleneck;
network matters on a multi-node run; thermal zones matter on a laptop
that is about to throttle. Adding each of those to
:func:`~hypernix.monitoring.tvtoppro.TvTopPro.render` would be a
thousand-line method and still not cover the next one.

So a module is a thing that returns numbers, and tvtoppro draws them.

Writing one
-----------
Subclass :class:`Module` and implement :meth:`Module.poll`::

    from hypernix.monitoring.tvtoppro_modules import Module, Panel, Reading

    class Batteries(Module):
        name = "battery"
        title = "battery"

        def available(self) -> bool:
            return Path("/sys/class/power_supply/BAT0").exists()

        def poll(self) -> Panel:
            charge = _read_charge()
            return Panel(title="battery", readings=[
                Reading("charge", f"{charge:.0f}%", fraction=charge / 100, ramp="free"),
            ])

Then drop it in ``~/.config/hypernix/tvtoppro/modules/battery.py`` with a
module-level ``MODULE = Batteries()``, or ship it in a package with a
``hypernix.tvtoppro_modules`` entry point. ``tvtoppro --list-modules``
shows what was found and, for anything that failed to load, why.

What a module does *not* do
----------------------------
It does not draw. It returns :class:`Reading` objects — a label, a
formatted value, optionally a 0..1 fraction to meter and a 0..1 history
to graph — and tvtoppro renders them with the active theme's ramps and
box characters. A module that emitted Rich markup would look wrong under
every theme but the author's, and would have to be rewritten each time
the presentation changed.

It also does not get to hang the dashboard. :func:`poll_all` gives each
module a wall-clock budget and turns an overrun or an exception into a
panel that says so, because a monitoring tool whose third-party plugin
can freeze the whole screen is one nobody can safely install a plugin
for.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Reading",
    "Panel",
    "Module",
    "ModuleError",
    "BUILTIN",
    "discover",
    "load_modules",
    "poll_all",
    "user_module_dir",
    "ENTRY_POINT_GROUP",
]

#: Packages advertise modules here. ``[project.entry-points."hypernix.tvtoppro_modules"]``
ENTRY_POINT_GROUP = "hypernix.tvtoppro_modules"

#: How long one module gets to produce a panel before it is reported as
#: slow. Not a hard kill -- :func:`poll_all` cannot interrupt a blocked
#: syscall -- but it does mean a module that reads a spun-down disk gets
#: named rather than the dashboard just feeling broken.
POLL_BUDGET_SECONDS = 0.25


class ModuleError(Exception):
    """A module could not be loaded or polled."""


@dataclass
class Reading:
    """One line in a panel.

    *fraction* and *history* are both optional and both in 0..1. A
    reading with neither is a plain label/value line; with a fraction it
    gets a gradient meter; with a history it gets a braille graph under
    it. Values are pre-formatted strings because only the module knows
    whether 1024 is a kilobyte, a millisecond or a queue depth.
    """

    label: str
    value: str = ""
    fraction: float | None = None
    #: Which of the theme's colour ramps to draw the meter and graph in.
    #: One of cpu, used, free, available, temp, download, upload. An
    #: unknown name falls back to the theme's default rather than
    #: raising: a module should not be able to crash the dashboard by
    #: naming a colour.
    ramp: str = "cpu"
    history: list[float] | None = None

    def __post_init__(self) -> None:
        if self.fraction is not None:
            self.fraction = max(0.0, min(1.0, float(self.fraction)))
        if self.history is not None:
            self.history = [max(0.0, min(1.0, float(v))) for v in self.history]


@dataclass
class Panel:
    """What one module contributes: a titled box of readings."""

    title: str
    readings: list[Reading] = field(default_factory=list)
    #: Shown instead of the readings when there are none. Say what is
    #: missing and how to get it, the way the gpu box says "no
    #: nvidia-smi here" rather than drawing zeroes.
    note: str = ""
    #: Set by :func:`poll_all` when the module raised or ran long. The
    #: panel still renders; it says what went wrong instead of numbers.
    error: str = ""


class Module(ABC):
    """A source of extra statistics for tvtoppro.

    Subclasses set :attr:`name` (what ``--modules`` matches) and
    :attr:`title` (what goes in the box border), and implement
    :meth:`poll`.
    """

    #: Matched by ``--modules``. Lower-case, no spaces.
    name: str = ""
    #: Drawn in the box's border rule.
    title: str = ""
    #: One line for ``--list-modules``.
    description: str = ""

    def available(self) -> bool:
        """Whether this machine has what the module reads.

        Checked once, at load. A module that is not available is listed
        with the reason rather than silently skipped, because "why is
        there no disk box" is otherwise unanswerable.
        """
        return True

    def unavailable_reason(self) -> str:
        """Why :meth:`available` said no. Shown by ``--list-modules``."""
        return "not available on this machine"

    @abstractmethod
    def poll(self) -> Panel:
        """Collect one frame's numbers."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r}>"


# ---------------------------------------------------------------------------
# Rate helpers
# ---------------------------------------------------------------------------


class _Rate:
    """Counter deltas over wall-clock time, with the first read discarded.

    Every counter here is monotonic-since-boot, so the first sample has
    nothing to subtract from and would read as "everything that has ever
    happened, in one second". Returning zero for it is not a lie worth
    worrying about and a 40 GB/s disk read on the first frame is.
    """

    def __init__(self) -> None:
        self._previous: dict[str, float] = {}
        self._when: float = 0.0

    def update(self, counters: dict[str, float]) -> dict[str, float]:
        now = time.monotonic()
        elapsed = now - self._when
        rates: dict[str, float] = {}
        if self._when and elapsed > 0:
            for key, value in counters.items():
                before = self._previous.get(key)
                if before is not None and value >= before:
                    rates[key] = (value - before) / elapsed
                else:
                    # A counter that went backwards is one that wrapped or
                    # a device that was re-enumerated. Either way the
                    # delta is meaningless, so it is skipped rather than
                    # reported as a negative rate.
                    rates[key] = 0.0
        else:
            rates = dict.fromkeys(counters, 0.0)
        self._previous = dict(counters)
        self._when = now
        return rates


def _fmt_rate(value: float) -> str:
    for unit, scale in (("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value:.0f} B/s"


def _fmt_bytes(value: float | int | None) -> str:
    if value is None:
        return "?"
    number = float(value)
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(number) >= scale:
            return f"{number / scale:.1f} {unit}"
    return f"{number:.0f} B"


#: Samples kept for the little graphs. Matches tvtop++'s history length so
#: a module's graph spans the same wall-clock window as the cpu graph
#: beside it -- two graphs on one screen covering different periods is a
#: comparison nobody can make.
HISTORY = 120


# ---------------------------------------------------------------------------
# Built-in modules
# ---------------------------------------------------------------------------


class DiskModule(Module):
    """Filesystem usage and read/write throughput.

    The one people want most on a training box, because "the GPU is at
    30%" and "the dataloader is reading 900 MB/s off a spinning disk" are
    the same problem and only one of them is visible in the cpu box.
    """

    name = "disk"
    title = "disk"
    description = "Filesystem usage and read/write throughput"

    def __init__(self, paths: list[str] | None = None) -> None:
        self.paths = paths or ["/"]
        self._io = _Rate()
        self._read_history: deque[float] = deque(maxlen=HISTORY)
        self._write_history: deque[float] = deque(maxlen=HISTORY)
        self._peak = 1.0

    def available(self) -> bool:
        # shutil.disk_usage works everywhere; os.statvfs is POSIX-only, and
        # testing for it hid the module on Windows until psutil was
        # installed — and then it was shown and crashed on its first poll.
        return True

    def unavailable_reason(self) -> str:
        return ""

    def poll(self) -> Panel:
        readings: list[Reading] = []
        for path in self.paths:
            try:
                usage = shutil.disk_usage(path)
            except OSError as exc:
                readings.append(Reading(path, f"unreadable: {exc.strerror}"))
                continue
            # used + free, not usage.total: what a non-root user can
            # actually fill, which is what statvfs's f_bavail gave before.
            total = usage.used + usage.free
            used = usage.used
            readings.append(Reading(
                label=path,
                value=f"{_fmt_bytes(used)}/{_fmt_bytes(total)}",
                fraction=(used / total) if total else 0.0,
                ramp="used",
            ))

        psutil = _psutil()
        if psutil is not None:
            try:
                counters = psutil.disk_io_counters()
            except Exception:  # noqa: BLE001 - psutil raises plenty
                counters = None
            if counters is not None:
                rates = self._io.update({
                    "read": float(counters.read_bytes),
                    "write": float(counters.write_bytes),
                })
                # One shared scale for both graphs, so "reads are twice
                # writes" is visible as one line being twice as tall.
                # Per-graph autoscaling would draw them the same height.
                self._peak = max(self._peak, rates["read"], rates["write"])
                self._read_history.append(rates["read"] / self._peak)
                self._write_history.append(rates["write"] / self._peak)
                readings.append(Reading(
                    label="read", value=_fmt_rate(rates["read"]),
                    ramp="download", history=list(self._read_history),
                ))
                readings.append(Reading(
                    label="write", value=_fmt_rate(rates["write"]),
                    ramp="upload", history=list(self._write_history),
                ))
        return Panel(title=self.title, readings=readings)


class NetworkModule(Module):
    """Per-interface receive and transmit rates, from /proc/net/dev."""

    name = "net"
    title = "net"
    description = "Network receive/transmit rates"

    #: Interfaces nobody wants a box for. Loopback traffic on a training
    #: box is the dataloader talking to itself and it dwarfs the real
    #: link, so a chart including it shows nothing else.
    SKIP = ("lo", "docker", "veth", "br-", "virbr")

    def __init__(self) -> None:
        self._rate = _Rate()
        self._rx: deque[float] = deque(maxlen=HISTORY)
        self._tx: deque[float] = deque(maxlen=HISTORY)
        self._peak = 1.0

    def available(self) -> bool:
        return Path("/proc/net/dev").exists()

    def unavailable_reason(self) -> str:
        return "no /proc/net/dev (not Linux)"

    def _counters(self) -> dict[str, float]:
        totals = {"rx": 0.0, "tx": 0.0}
        try:
            lines = Path("/proc/net/dev").read_text().splitlines()[2:]
        except OSError:
            return totals
        for line in lines:
            name, _, rest = line.partition(":")
            name = name.strip()
            if not rest or any(name.startswith(skip) for skip in self.SKIP):
                continue
            fields = rest.split()
            if len(fields) < 9:
                continue
            try:
                totals["rx"] += float(fields[0])
                totals["tx"] += float(fields[8])
            except ValueError:
                continue
        return totals

    def poll(self) -> Panel:
        rates = self._rate.update(self._counters())
        self._peak = max(self._peak, rates["rx"], rates["tx"])
        self._rx.append(rates["rx"] / self._peak)
        self._tx.append(rates["tx"] / self._peak)
        return Panel(title=self.title, readings=[
            Reading("down", _fmt_rate(rates["rx"]), ramp="download",
                    history=list(self._rx)),
            Reading("up", _fmt_rate(rates["tx"]), ramp="upload",
                    history=list(self._tx)),
        ])


class TemperatureModule(Module):
    """Thermal zones, from /sys/class/thermal.

    On a laptop this is the box that explains a run that got slower for
    no visible reason: the cpu box shows 100% either way, and the clock
    it is running at does not appear anywhere else.
    """

    name = "temps"
    title = "temps"
    description = "Thermal zones (/sys/class/thermal)"

    ROOT = Path("/sys/class/thermal")

    def available(self) -> bool:
        return self.ROOT.is_dir() and any(self.ROOT.glob("thermal_zone*"))

    def unavailable_reason(self) -> str:
        return "no thermal zones under /sys/class/thermal"

    def poll(self) -> Panel:
        readings: list[Reading] = []
        for zone in sorted(self.ROOT.glob("thermal_zone*")):
            try:
                millidegrees = int((zone / "temp").read_text().strip())
            except (OSError, ValueError):
                continue
            try:
                label = (zone / "type").read_text().strip()
            except OSError:
                label = zone.name
            celsius = millidegrees / 1000.0
            readings.append(Reading(
                label=label[:14],
                value=f"{celsius:.0f}°C",
                # 100 C as full scale: every thermal limit worth watching
                # is under it, so the meter is comparable between zones
                # rather than each one being scaled to its own maximum.
                fraction=min(celsius, 100.0) / 100.0,
                ramp="temp",
            ))
        if not readings:
            return Panel(title=self.title, note="thermal zones present but unreadable")
        return Panel(title=self.title, readings=readings)


class LoadModule(Module):
    """Load average and process counts."""

    name = "load"
    title = "load"
    description = "Load average and process/thread counts"

    def available(self) -> bool:
        return hasattr(os, "getloadavg")

    def unavailable_reason(self) -> str:
        return "no os.getloadavg on this platform"

    def poll(self) -> Panel:
        try:
            one, five, fifteen = os.getloadavg()
        except OSError:
            return Panel(title=self.title, note="load average unreadable")
        cpus = os.cpu_count() or 1
        readings = [
            # Scaled by core count, because a load of 8 is idle on a
            # 64-core box and a fire on a 4-core one, and an unscaled
            # meter says the same thing about both.
            Reading("1 min", f"{one:.2f}", fraction=one / cpus, ramp="cpu"),
            Reading("5 min", f"{five:.2f}", fraction=five / cpus, ramp="cpu"),
            Reading("15 min", f"{fifteen:.2f}", fraction=fifteen / cpus, ramp="cpu"),
        ]
        psutil = _psutil()
        if psutil is not None:
            try:
                readings.append(Reading(
                    "processes", f"{len(psutil.pids())}", ramp="free",
                ))
            except Exception:  # noqa: BLE001
                pass
        return Panel(title=self.title, readings=readings)


class SwapModule(Module):
    """Swap usage. Its own box because swapping during training is fatal.

    A run that starts swapping does not slow down by 20%, it slows down
    by two orders of magnitude, and the mem box shows RAM at a
    comfortable 85% throughout. This is the number that says why.
    """

    name = "swap"
    title = "swap"
    description = "Swap usage — the number that explains a run that fell off a cliff"

    def available(self) -> bool:
        return Path("/proc/meminfo").exists()

    def unavailable_reason(self) -> str:
        return "no /proc/meminfo (not Linux)"

    def poll(self) -> Panel:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                if key in ("SwapTotal", "SwapFree", "SwapCached"):
                    values[key] = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return Panel(title=self.title, note="/proc/meminfo unreadable")
        total = values.get("SwapTotal", 0)
        if not total:
            return Panel(title=self.title, note="no swap configured")
        used = total - values.get("SwapFree", 0)
        return Panel(title=self.title, readings=[
            Reading("used", f"{_fmt_bytes(used)}/{_fmt_bytes(total)}",
                    fraction=used / total, ramp="used"),
            Reading("cached", _fmt_bytes(values.get("SwapCached", 0)),
                    ramp="available"),
        ])


def _psutil() -> Any:
    """psutil if it is installed, else ``None``.

    Imported lazily and never required: everything here that can be done
    with /proc is, and psutil only adds the counters that cannot.
    """
    try:
        import psutil
    except ImportError:
        return None
    return psutil


#: What ships. Instantiated on demand by :func:`load_modules` so that a
#: module holding rate state gets a fresh one per dashboard.
BUILTIN: dict[str, type[Module]] = {
    "disk": DiskModule,
    "net": NetworkModule,
    "temps": TemperatureModule,
    "load": LoadModule,
    "swap": SwapModule,
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def user_module_dir() -> Path:
    """Where a user drops a ``.py`` file to add a panel."""
    root = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(root) / "hypernix" / "tvtoppro" / "modules"


def _from_entry_points() -> dict[str, Module | str]:
    """Modules advertised by installed packages.

    A value that is a string is a load failure, kept rather than dropped:
    ``--list-modules`` shows it, because a plugin that silently does not
    appear is one the author cannot debug.
    """
    found: dict[str, Module | str] = {}
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - 3.8 and older
        return found
    try:
        points = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - older selectable API
        points = entry_points().get(ENTRY_POINT_GROUP, [])
    for point in points:
        try:
            loaded = point.load()
            instance = loaded() if isinstance(loaded, type) else loaded
            if not isinstance(instance, Module):
                found[point.name] = (
                    f"{point.value} is not a Module subclass or instance"
                )
                continue
            found[instance.name or point.name] = instance
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not stop the rest
            logger.debug("tvtoppro: entry point %s failed", point.name, exc_info=True)
            found[point.name] = f"{type(exc).__name__}: {exc}"
    return found


def _from_user_dir(directory: Path | None = None) -> dict[str, Module | str]:
    """Modules from ``~/.config/hypernix/tvtoppro/modules/*.py``.

    Each file may define ``MODULE`` (an instance), ``MODULES`` (a list),
    or a ``Module`` subclass, which is instantiated with no arguments.
    """
    directory = directory or user_module_dir()
    found: dict[str, Module | str] = {}
    if not directory.is_dir():
        return found
    import importlib.util

    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"hypernix_tvtoppro_user_{path.stem}", path
            )
            if spec is None or spec.loader is None:
                found[path.stem] = "not importable as a Python module"
                continue
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
        except Exception as exc:  # noqa: BLE001 - a bad file must not stop the rest
            logger.debug("tvtoppro: user module %s failed", path, exc_info=True)
            found[path.stem] = f"{type(exc).__name__}: {exc}"
            continue

        instances: list[Module] = []
        candidates = list(getattr(loaded, "MODULES", []) or [])
        single = getattr(loaded, "MODULE", None)
        if single is not None:
            candidates.append(single)
        if not candidates:
            candidates = [
                value for value in vars(loaded).values()
                if isinstance(value, type)
                and issubclass(value, Module)
                and value is not Module
            ]
        for candidate in candidates:
            try:
                instance = candidate() if isinstance(candidate, type) else candidate
            except Exception as exc:  # noqa: BLE001
                found[path.stem] = f"{type(exc).__name__}: {exc}"
                continue
            if isinstance(instance, Module):
                instances.append(instance)
        if not instances:
            found[path.stem] = "defines no Module subclass, MODULE or MODULES"
            continue
        for instance in instances:
            found[instance.name or path.stem] = instance
    return found


def discover(*, user_dir: Path | None = None) -> dict[str, Module | str]:
    """Every module this machine offers, by name.

    Built-ins first, then entry points, then the user directory — so a
    user's own file wins over a package's, and a package's over a
    built-in. That order is deliberate: the more local the source, the
    more it is a deliberate override.

    A string value instead of a :class:`Module` is a load failure kept
    for ``--list-modules`` to report.
    """
    found: dict[str, Module | str] = {}
    for name, factory in BUILTIN.items():
        try:
            found[name] = factory()
        except Exception as exc:  # noqa: BLE001 - should not happen; say so if it does
            found[name] = f"{type(exc).__name__}: {exc}"
    found.update(_from_entry_points())
    found.update(_from_user_dir(user_dir))
    return found


def load_modules(
    names: list[str] | str | None, *, user_dir: Path | None = None
) -> tuple[list[Module], list[tuple[str, str]]]:
    """Resolve *names* to modules. Returns ``(loaded, problems)``.

    *names* may be a list, a comma-separated string, or ``"all"`` for
    every available one. Unknown names and unavailable modules come back
    in *problems* as ``(name, reason)`` rather than raising: one bad name
    in ``--modules disk,nte,swap`` should cost the typo, not the
    dashboard.
    """
    if isinstance(names, str):
        names = [part.strip() for part in names.replace(" ", ",").split(",")]
    wanted = [name for name in (names or []) if name]
    available = discover(user_dir=user_dir)

    if any(name.lower() == "all" for name in wanted):
        wanted = [
            name for name, module in available.items()
            if isinstance(module, Module) and module.available()
        ]

    loaded: list[Module] = []
    problems: list[tuple[str, str]] = []
    for name in wanted:
        module = available.get(name)
        if module is None:
            close = [
                other for other in available
                if other.startswith(name[:2]) and other != name
            ]
            hint = f"; did you mean {', '.join(sorted(close)[:3])}?" if close else ""
            problems.append((name, f"no such module{hint}"))
            continue
        if isinstance(module, str):
            problems.append((name, module))
            continue
        try:
            usable = module.available()
        except Exception as exc:  # noqa: BLE001
            problems.append((name, f"available() raised {type(exc).__name__}: {exc}"))
            continue
        if not usable:
            problems.append((name, module.unavailable_reason()))
            continue
        loaded.append(module)
    return loaded, problems


def poll_all(modules: list[Module]) -> list[Panel]:
    """Poll each module, turning a failure into a panel that says so.

    A module that raises gets a box with the exception in it rather than
    taking the dashboard down, and one that runs over
    :data:`POLL_BUDGET_SECONDS` gets the number said out loud. Neither
    can be enforced harder than this without threads, and a monitoring
    tool that spawns a thread per plugin per frame has a worse problem
    than a slow plugin.
    """
    panels: list[Panel] = []
    for module in modules:
        started = time.monotonic()
        try:
            panel = module.poll()
        except Exception as exc:  # noqa: BLE001 - the whole point
            logger.debug("tvtoppro: module %s raised", module.name, exc_info=True)
            panels.append(Panel(
                title=module.title or module.name,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue
        elapsed = time.monotonic() - started
        if not isinstance(panel, Panel):
            panels.append(Panel(
                title=module.title or module.name,
                error=f"poll() returned {type(panel).__name__}, not a Panel",
            ))
            continue
        if elapsed > POLL_BUDGET_SECONDS:
            panel.error = (
                f"took {elapsed * 1000:.0f} ms (budget "
                f"{POLL_BUDGET_SECONDS * 1000:.0f} ms) — the dashboard "
                f"waits for this"
            )
        panels.append(panel)
    return panels
