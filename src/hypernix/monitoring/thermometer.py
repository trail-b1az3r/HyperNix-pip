"""thermometer — sample CPU / GPU temperatures during training.

Four tiers, all returning :class:`Reading` snapshots:

* :class:`InstantThermometer`   — t1.  One-shot read, no history.
* :class:`ProbeThermometer`     — t2.  Stores a rolling window so
                                       you can ask for the recent
                                       max / mean / min.
* :class:`InfraredThermometer`  — t3.  Tracks per-source max
                                       temperatures over the run +
                                       a configurable warning threshold.
* :class:`DigitalThermometer`   — t4.  Logs every reading to a file
                                       for post-mortem inspection.

Read sources, tried in order:

* CPU: ``psutil.sensors_temperatures`` (when psutil is available),
  Linux ``/sys/class/thermal/thermal_zone*/temp``.
* GPU: ``nvidia-smi --query-gpu=temperature.gpu``.
* Falls back to ``None`` for sources that aren't available.

Zero hard deps; psutil is optional.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Reading:
    timestamp: float
    cpu_celsius: float | None = None
    gpu_celsius: float | None = None
    sources: dict[str, float] = field(default_factory=dict)

    def hottest(self) -> float | None:
        vals = [v for v in (self.cpu_celsius, self.gpu_celsius) if v is not None]
        if not vals:
            return None
        return max(vals)


# ---------------------------------------------------------------------------
# Reading helpers (no hard deps)
# ---------------------------------------------------------------------------

def _read_cpu_temp_psutil() -> tuple[float | None, dict[str, float]]:
    try:
        import psutil  # type: ignore
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:  # noqa: BLE001
        return None, {}
    if not temps:
        return None, {}
    by_src: dict[str, float] = {}
    for chip, entries in temps.items():
        for e in entries:
            if e.current is None:
                continue
            label = e.label or chip
            by_src[f"{chip}:{label}"] = float(e.current)
    if not by_src:
        return None, {}
    return max(by_src.values()), by_src


def _read_cpu_temp_sysfs() -> tuple[float | None, dict[str, float]]:
    base = Path("/sys/class/thermal")
    if not base.exists():
        return None, {}
    by_src: dict[str, float] = {}
    for zone in base.glob("thermal_zone*"):
        try:
            t = int((zone / "temp").read_text().strip()) / 1000.0
        except Exception:  # noqa: BLE001
            continue
        try:
            tname = (zone / "type").read_text().strip()
        except Exception:  # noqa: BLE001
            tname = zone.name
        by_src[f"{zone.name}:{tname}"] = t
    if not by_src:
        return None, {}
    return max(by_src.values()), by_src


def read_cpu_temp() -> tuple[float | None, dict[str, float]]:
    val, by = _read_cpu_temp_psutil()
    if val is None:
        val, by = _read_cpu_temp_sysfs()
    return val, by


def read_gpu_temp() -> float | None:
    """The hottest GPU on the machine, whoever made it.

    Was ``nvidia-smi`` directly, which meant an AMD card had no
    temperature at all -- the thermometer reported the CPU and a blank
    where the GPU should be, on the hardware where a temperature reading
    matters most. :mod:`hypernix.system.gpus` asks whichever vendor tool
    is present and reports the same shape either way.

    The *hottest* card and not card 0: this feeds a single number, and
    the only useful single number from a multi-GPU box is the one closest
    to throttling.
    """
    from ..system import gpus as _gpus

    readings = [
        card.temperature_c for card in _gpus.detect()
        if card.temperature_c is not None
    ]
    return max(readings) if readings else None


def take_reading() -> Reading:
    cpu, by_src = read_cpu_temp()
    gpu = read_gpu_temp()
    sources = dict(by_src)
    if gpu is not None:
        # Named for the abstraction rather than for nvidia-smi, because
        # the reading may well have come from rocm-smi. A source label
        # that names the wrong tool is worse than a vague one when
        # somebody is working out why a number looks wrong.
        sources["gpu"] = gpu
    return Reading(timestamp=time.time(), cpu_celsius=cpu, gpu_celsius=gpu, sources=sources)


# ---------------------------------------------------------------------------
# Tier 1 — InstantThermometer
# ---------------------------------------------------------------------------

@dataclass
class InstantThermometer:
    """One-shot read, no history."""

    name: str = "InstantThermometer"

    def read(self) -> Reading:
        return take_reading()


# ---------------------------------------------------------------------------
# Tier 2 — ProbeThermometer
# ---------------------------------------------------------------------------

@dataclass
class ProbeThermometer:
    """Keeps a rolling window of recent readings."""

    name: str = "ProbeThermometer"
    history_size: int = 60
    history: deque[Reading] = field(default_factory=lambda: deque(maxlen=60), init=False)

    def __post_init__(self) -> None:
        if self.history_size != 60:
            self.history = deque(maxlen=self.history_size)

    def read(self) -> Reading:
        r = take_reading()
        self.history.append(r)
        return r

    def recent_max(self) -> float | None:
        vals = [r.hottest() for r in self.history if r.hottest() is not None]
        return max(vals) if vals else None

    def recent_mean(self) -> float | None:
        vals = [r.hottest() for r in self.history if r.hottest() is not None]
        return sum(vals) / len(vals) if vals else None

    def recent_min(self) -> float | None:
        vals = [r.hottest() for r in self.history if r.hottest() is not None]
        return min(vals) if vals else None


# ---------------------------------------------------------------------------
# Tier 3 — InfraredThermometer
# ---------------------------------------------------------------------------

@dataclass
class InfraredThermometer:
    """Tracks per-source max temperatures plus a warning threshold."""

    name: str = "InfraredThermometer"
    warn_celsius: float = 85.0
    critical_celsius: float = 95.0
    peaks: dict[str, float] = field(default_factory=dict, init=False)
    warnings_emitted: int = field(default=0, init=False)

    def read(self) -> Reading:
        r = take_reading()
        for src, val in r.sources.items():
            cur = self.peaks.get(src, float("-inf"))
            if val > cur:
                self.peaks[src] = val
        if r.hottest() is not None and r.hottest() >= self.warn_celsius:
            self.warnings_emitted += 1
        return r

    def is_critical(self) -> bool:
        peak = max(self.peaks.values()) if self.peaks else float("-inf")
        return peak >= self.critical_celsius

    def status(self) -> str:
        peak = max(self.peaks.values()) if self.peaks else None
        if peak is None:
            return "no readings"
        if peak >= self.critical_celsius:
            return f"CRITICAL: peak {peak:.1f}°C >= {self.critical_celsius}°C"
        if peak >= self.warn_celsius:
            return f"WARN: peak {peak:.1f}°C >= {self.warn_celsius}°C"
        return f"OK: peak {peak:.1f}°C"


# ---------------------------------------------------------------------------
# Tier 4 — DigitalThermometer
# ---------------------------------------------------------------------------

@dataclass
class DigitalThermometer:
    """Logs every reading to a JSONL file for later analysis."""

    name: str = "DigitalThermometer"
    log_path: Path | str = "thermometer.jsonl"
    _fh: object = field(default=None, init=False, repr=False)

    def open(self) -> DigitalThermometer:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = Path(self.log_path).open("a", encoding="utf-8")  # type: ignore[attr-defined]
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()  # type: ignore[union-attr]
            self._fh = None

    def __enter__(self) -> DigitalThermometer:
        return self.open()

    def __exit__(self, *exc) -> bool:  # noqa: ANN002
        self.close()
        return False

    def read(self) -> Reading:
        r = take_reading()
        if self._fh is None:
            self.open()
        line = json.dumps({
            "timestamp": r.timestamp,
            "cpu_celsius": r.cpu_celsius,
            "gpu_celsius": r.gpu_celsius,
            "sources": r.sources,
        }, ensure_ascii=False)
        self._fh.write(line + "\n")  # type: ignore[union-attr]
        self._fh.flush()  # type: ignore[union-attr]
        return r


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

TIERS = {
    "instant": InstantThermometer,
    "probe": ProbeThermometer,
    "infrared": InfraredThermometer,
    "digital": DigitalThermometer,
}


def thermometer(kind: str = "instant", **kw):
    if kind not in TIERS:
        raise ValueError(f"unknown thermometer kind {kind!r}; valid: {sorted(TIERS)}")
    return TIERS[kind](**kw)


__all__ = [
    "DigitalThermometer",
    "InfraredThermometer",
    "InstantThermometer",
    "ProbeThermometer",
    "Reading",
    "TIERS",
    "read_cpu_temp",
    "read_gpu_temp",
    "take_reading",
    "thermometer",
]
