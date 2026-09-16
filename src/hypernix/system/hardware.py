"""hypernix.system.hardware — what this machine is doing, as data.

The dashboards (``tvtoppro``, ``cctvtop``, ``hnx-map``) all sample the
same hardware and all shape it for a terminal. HyperLink needs the same
numbers over a socket, and a phone six hundred miles away is exactly the
situation where "is the server actually busy, or is my model just slow?"
cannot be answered any other way.

So this is the sampling with no presentation attached: one snapshot,
JSON-shaped, cheap enough to call on a timer.

Everything is optional
----------------------
Every field can be ``None``, and that is the design rather than a
weakness. psutil may not be installed. ``/proc`` is not there on macOS.
A container sees the host's CPU count and its own cgroup limit and they
disagree. No GPU vendor tool is guaranteed present. A snapshot that
invented a zero for any of those would be a dashboard confidently
reporting that a machine under load is idle — so a number that could not
be read is absent, and the reader says "unknown" rather than "0%".

Nothing here shells out to anything that can block for long: the GPU
query is :mod:`hypernix.system.gpus`, which already caps its own
subprocess timeouts because it runs on a monitoring tick.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CPUReading",
    "MemoryReading",
    "DiskReading",
    "GPUReading",
    "HardwareSnapshot",
    "snapshot",
    "uptime_seconds",
    "process_uptime_seconds",
]


def _psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Uptime
# ---------------------------------------------------------------------------


def uptime_seconds() -> float | None:
    """How long the *machine* has been up.

    ``/proc/uptime`` first because it is exact and free; psutil's boot
    time second; nothing third. A server that has been up eleven minutes
    is a server that rebooted, which is usually the answer to whatever
    prompted somebody to look.
    """
    try:
        with open("/proc/uptime", encoding="ascii") as handle:
            return float(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    psutil = _psutil()
    if psutil is not None:
        try:
            return max(0.0, time.time() - psutil.boot_time())
        except Exception:  # noqa: BLE001 - a missing reading is not an error
            return None
    return None


#: When this process started. Module import time is close enough — the
#: API server imports this during startup — and it is exact where psutil
#: is available.
_STARTED_AT = time.time()


def process_uptime_seconds() -> float:
    """How long *this server* has been running.

    Distinct from machine uptime and usually the more interesting of the
    two: "the API restarted an hour ago" explains a dropped session in a
    way "the box has been up for nine days" does not.
    """
    psutil = _psutil()
    if psutil is not None:
        try:
            return max(0.0, time.time() - psutil.Process(os.getpid()).create_time())
        except Exception:  # noqa: BLE001
            pass
    return max(0.0, time.time() - _STARTED_AT)


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------


@dataclass
class CPUReading:
    percent: float | None = None
    cores_physical: int | None = None
    cores_logical: int | None = None
    #: 1/5/15-minute load average. Absent on Windows, which has no such
    #: concept rather than a zero one.
    load_average: list[float] = field(default_factory=list)
    frequency_mhz: float | None = None
    temperature_c: float | None = None
    model: str = ""


@dataclass
class MemoryReading:
    total_bytes: int | None = None
    used_bytes: int | None = None
    available_bytes: int | None = None
    percent: float | None = None


@dataclass
class DiskReading:
    mount: str = "/"
    total_bytes: int | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None
    percent: float | None = None


@dataclass
class GPUReading:
    index: int = 0
    vendor: str = ""
    name: str = ""
    memory_total_bytes: int | None = None
    memory_used_bytes: int | None = None
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None


@dataclass
class HardwareSnapshot:
    """One sample. Every field optional; see the module docstring."""

    sampled_at: float
    hostname: str = ""
    platform: str = ""
    #: Machine uptime, or None where it could not be read.
    uptime_seconds: float | None = None
    #: This server process's uptime. Always available.
    process_uptime_seconds: float = 0.0
    cpu: CPUReading = field(default_factory=CPUReading)
    memory: MemoryReading = field(default_factory=MemoryReading)
    swap: MemoryReading = field(default_factory=MemoryReading)
    disks: list[DiskReading] = field(default_factory=list)
    gpus: list[GPUReading] = field(default_factory=list)
    #: What could not be sampled, and why. Present so a reader can say
    #: "psutil is not installed" instead of showing an empty dashboard.
    unavailable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _cpu(psutil, unavailable: list[str]) -> CPUReading:
    reading = CPUReading(model=platform.processor() or "")
    try:
        reading.load_average = [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):
        # Windows. Not a failure — there is no load average there.
        pass

    if psutil is None:
        unavailable.append("cpu: psutil is not installed")
        return reading
    try:
        # interval=None is the non-blocking form: it reports the average
        # since the previous call. The first call after import returns
        # 0.0, which is why callers that want a real first number sample
        # twice — but blocking the request for a whole interval to get
        # one is worse, and a dashboard polls anyway.
        reading.percent = psutil.cpu_percent(interval=None)
        reading.cores_physical = psutil.cpu_count(logical=False)
        reading.cores_logical = psutil.cpu_count(logical=True)
    except Exception as exc:  # noqa: BLE001
        unavailable.append(f"cpu: {exc}")
    try:
        frequency = psutil.cpu_freq()
        if frequency is not None:
            reading.frequency_mhz = round(float(frequency.current), 1)
    except Exception:  # noqa: BLE001 - absent in many containers
        pass
    reading.temperature_c = _cpu_temperature(psutil)
    return reading


def _cpu_temperature(psutil) -> float | None:
    """The package temperature, where the platform reports one.

    Names vary by driver — coretemp on Intel, k10temp on AMD, cpu_thermal
    on a Pi — so the known ones are tried in order and anything with a
    current reading is taken rather than insisting on a label.
    """
    try:
        temperatures = psutil.sensors_temperatures()
    except (AttributeError, Exception):  # noqa: BLE001
        return None
    if not temperatures:
        return None
    for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz", "zenpower"):
        for entry in temperatures.get(key, []):
            if entry.current:
                return round(float(entry.current), 1)
    for entries in temperatures.values():
        for entry in entries:
            if entry.current:
                return round(float(entry.current), 1)
    return None


def _memory(psutil, unavailable: list[str]) -> tuple[MemoryReading, MemoryReading]:
    if psutil is None:
        unavailable.append("memory: psutil is not installed")
        return MemoryReading(), MemoryReading()
    memory, swap = MemoryReading(), MemoryReading()
    try:
        virtual = psutil.virtual_memory()
        memory = MemoryReading(
            total_bytes=int(virtual.total),
            used_bytes=int(virtual.used),
            available_bytes=int(virtual.available),
            percent=round(float(virtual.percent), 1),
        )
    except Exception as exc:  # noqa: BLE001
        unavailable.append(f"memory: {exc}")
    try:
        paging = psutil.swap_memory()
        swap = MemoryReading(
            total_bytes=int(paging.total),
            used_bytes=int(paging.used),
            available_bytes=int(paging.free),
            percent=round(float(paging.percent), 1),
        )
    except Exception as exc:  # noqa: BLE001
        unavailable.append(f"swap: {exc}")
    return memory, swap


def _disks(paths: list[str], unavailable: list[str]) -> list[DiskReading]:
    """Usage for the mounts that matter, not every mount there is.

    A container has dozens and a phone screen has room for two. The
    caller names them; the default is the root and the models directory,
    because "am I out of space for another model" is the disk question
    this API gets asked.
    """
    readings: list[DiskReading] = []
    seen: set[str] = set()
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            unavailable.append(f"disk {path}: {exc}")
            continue
        # Two paths on one filesystem report identical numbers; listing
        # them twice is noise.
        signature = f"{usage.total}:{usage.free}"
        if signature in seen:
            continue
        seen.add(signature)
        readings.append(DiskReading(
            mount=path,
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            percent=round(usage.used / usage.total * 100, 1) if usage.total else None,
        ))
    return readings


def _gpus(unavailable: list[str]) -> list[GPUReading]:
    try:
        from . import gpus as gpu_module

        cards = gpu_module.detect()
    except Exception as exc:  # noqa: BLE001 - a vendor tool is never guaranteed
        unavailable.append(f"gpu: {exc}")
        return []

    readings = []
    for card in cards:
        readings.append(GPUReading(
            index=int(getattr(card, "index", 0)),
            vendor=str(getattr(card, "vendor", "") or ""),
            name=str(getattr(card, "name", "") or ""),
            memory_total_bytes=_mb(getattr(card, "memory_total_mb", None)),
            memory_used_bytes=_mb(getattr(card, "memory_used_mb", None)),
            utilization_percent=getattr(card, "utilization_pct", None),
            temperature_c=getattr(card, "temperature_c", None),
            power_w=getattr(card, "power_w", None),
            power_limit_w=getattr(card, "power_limit_w", None),
        ))
    return readings


def _mb(value: int | None) -> int | None:
    """Megabytes to bytes, keeping None as None.

    Bytes on the wire throughout: a client that formats one field in MB
    and another in bytes gets it wrong eventually, and the server is the
    place to be consistent.
    """
    return None if value is None else int(value) * 1024 * 1024


def snapshot(*, disk_paths: list[str] | None = None) -> HardwareSnapshot:
    """One sample of everything, with whatever could not be read named."""
    unavailable: list[str] = []
    psutil = _psutil()

    if disk_paths is None:
        from pathlib import Path

        candidates = ["/", str(Path.home() / ".hypernix")]
        disk_paths = [p for p in candidates if os.path.exists(p)]

    memory, swap = _memory(psutil, unavailable)
    return HardwareSnapshot(
        sampled_at=time.time(),
        hostname=platform.node(),
        platform=f"{platform.system()} {platform.release()}".strip(),
        uptime_seconds=uptime_seconds(),
        process_uptime_seconds=round(process_uptime_seconds(), 2),
        cpu=_cpu(psutil, unavailable),
        memory=memory,
        swap=swap,
        disks=_disks(disk_paths, unavailable),
        gpus=_gpus(unavailable),
        unavailable=unavailable,
    )
