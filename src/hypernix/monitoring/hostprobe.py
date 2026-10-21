"""What the machine is doing, measured rather than estimated.

tvtop-max's machine panels read from here (through
:mod:`.tvtop_max_bridge`), and tvtop-pro's memory figures too, so the
two agree.

``memory()``
    RAM as the kernel accounts for it. *Used* is memory that cannot be
    handed back on demand -- total minus ``MemAvailable`` on Linux; app,
    wired and compressed memory on macOS, as Activity Monitor counts it --
    and the percentage is that over the total, so the bar and the numbers
    beside it are one measurement. Inside a container with a memory
    limit, the limit is the total and the cgroup's own accounting is the
    use, because the host's ``/proc/meminfo`` describes a machine the
    container cannot have.
``cpu_temperatures(cores)``
    Per-core temperatures where the driver reports them (Intel
    ``coretemp``), mapped onto logical CPUs through the CPU topology;
    the package or per-CCD temperatures where it does not (AMD
    ``k10temp``), with a note saying so.
``audio()``
    The default output's volume as the sound server reports it -- in
    decibels from PipeWire/PulseAudio or ALSA, never worked out from a
    percentage -- whether something is playing, the level of what is
    playing (sampled from the output's monitor), and a warning when it
    is loud.
``disks()`` / ``network()`` / ``system()``
    Space, throughput, load and uptime.

Everything here is best-effort and never raises: a source that is not
there comes back as ``None`` with a note, not as a guess.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

__all__ = [
    "LOUD_DB",
    "UNSAFE_DB",
    "audio",
    "cpu_temperatures",
    "disks",
    "memory",
    "memory_users",
    "network",
    "system",
]

_MIB = 1024 * 1024
_UNLIMITED = 1 << 60

#: Output gain at or above which audio that is playing is called loud.
#: Not a sound level -- that depends on the speakers or headphones and
#: cannot be measured from here -- but the gain at which most headphones
#: reach the levels hearing guidance warns about (around 85 dB SPL) when
#: what is playing is mastered near full scale. Override with
#: ``TVTOP_MAX_LOUD_DB``.
LOUD_DB = -6.0
#: Above 0 dB the sound server is amplifying past 100%: louder than the
#: device's own maximum, and distorting.
UNSAFE_DB = 0.0


def _run(argv: list[str], timeout: float = 1.5) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603 - fixed argv
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


# -- memory --------------------------------------------------------------------


def _meminfo(text: str) -> dict[str, int]:
    """``/proc/meminfo`` in bytes."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return out


def _from_meminfo(info: dict[str, int]) -> dict[str, Any] | None:
    total = info.get("MemTotal", 0)
    if not total:
        return None
    available = info.get("MemAvailable")
    if available is None:  # kernels before 3.14
        available = info.get("MemFree", 0) + info.get("Buffers", 0) + info.get("Cached", 0) + info.get("SReclaimable", 0)
    used = max(0, total - available)
    cache = max(0, info.get("Buffers", 0) + info.get("Cached", 0) + info.get("SReclaimable", 0) - info.get("Shmem", 0))
    swap_total = info.get("SwapTotal", 0)
    return _memory_dict(total, used, available, info.get("MemFree", 0), cache, info.get("Shmem", 0),
                        swap_total - info.get("SwapFree", swap_total), swap_total, "meminfo")


def _memory_dict(total: int, used: int, available: int, free: int, cache: int, shared: int,
                 swap_used: int, swap_total: int, source: str) -> dict[str, Any]:
    return {
        "total_mib": total // _MIB,
        "used_mib": used // _MIB,
        "available_mib": available // _MIB,
        # Before 0.72.8 this key held the *available* figure; it is now
        # what it says, and the panels show both.
        "free_mib": free // _MIB,
        "cached_mib": cache // _MIB,
        "shared_mib": shared // _MIB,
        "swap_used_mib": max(0, swap_used) // _MIB,
        "swap_total_mib": swap_total // _MIB,
        "percent": round(100.0 * used / total, 1) if total else None,
        "source": source,
    }


def _cgroup_dir(root: Path = Path("/sys/fs/cgroup"), proc_cgroup: str | None = None) -> Path | None:
    """This process's cgroup v2 directory, if it has a memory controller."""
    text = proc_cgroup if proc_cgroup is not None else _read("/proc/self/cgroup")
    for line in (text or "").splitlines():
        if line.startswith("0::"):
            rel = line[3:].strip().lstrip("/")
            for candidate in (root / rel, root):
                if (candidate / "memory.max").exists():
                    return candidate
    return None


def _from_cgroup(host_total: int, root: Path = Path("/sys/fs/cgroup"),
                 proc_cgroup: str | None = None) -> dict[str, Any] | None:
    """A container's memory, when it is limited below the host's."""
    v2 = _cgroup_dir(root, proc_cgroup)
    if v2 is not None:
        limit_text = _read(v2 / "memory.max") or "max"
        current_text = _read(v2 / "memory.current")
        stat_text = _read(v2 / "memory.stat") or ""
        swap_max, swap_now = _read(v2 / "memory.swap.max"), _read(v2 / "memory.swap.current")
        inactive_key, file_key = "inactive_file", "file"
    else:
        v1 = root / "memory"
        limit_text = _read(v1 / "memory.limit_in_bytes")
        current_text = _read(v1 / "memory.usage_in_bytes")
        stat_text = _read(v1 / "memory.stat") or ""
        swap_max = swap_now = None
        inactive_key, file_key = "total_inactive_file", "total_cache"
    if not limit_text or not current_text or limit_text == "max" or not limit_text.isdigit():
        return None
    limit = int(limit_text)
    if limit >= _UNLIMITED or (host_total and limit >= host_total):
        return None
    stat = {k: int(v) for k, v in (line.split()[:2] for line in stat_text.splitlines() if len(line.split()) >= 2)
            if v.isdigit()}
    # What `docker stats` and the kernel's OOM killer count: usage less
    # the page cache that can be dropped.
    used = max(0, int(current_text) - stat.get(inactive_key, 0))
    swap_total = int(swap_max) if swap_max and swap_max.isdigit() else 0
    return _memory_dict(limit, used, max(0, limit - used), max(0, limit - int(current_text)),
                        stat.get(file_key, 0), stat.get("shmem", 0),
                        int(swap_now) if swap_now and swap_now.isdigit() else 0, swap_total, "cgroup")


def _from_vm_stat(vm_stat: str, memsize: int, swapusage: str = "") -> dict[str, Any] | None:
    """macOS, as Activity Monitor counts it: app + wired + compressed."""
    page = int(m.group(1)) if (m := re.search(r"page size of (\d+) bytes", vm_stat)) else 4096
    pages = {k.strip().strip('"'): int(v.strip().rstrip(".")) for k, v in
             (line.split(":", 1) for line in vm_stat.splitlines()[1:] if ":" in line)
             if v.strip().rstrip(".").isdigit()}
    if not memsize:
        return None
    app = pages.get("Anonymous pages", 0) - pages.get("Pages purgeable", 0)
    wired = pages.get("Pages wired down", 0)
    compressed = pages.get("Pages occupied by compressor", 0)
    used = max(0, (app + wired + compressed) * page)
    cache = (pages.get("File-backed pages", 0) + pages.get("Pages purgeable", 0)) * page
    free = pages.get("Pages free", 0) * page
    swap_used = swap_total = 0
    if m := re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", swapusage):
        swap_total, swap_used = int(float(m.group(1)) * _MIB), int(float(m.group(2)) * _MIB)
    return _memory_dict(memsize, used, max(0, memsize - used), free, cache, 0, swap_used, swap_total, "vm_stat")


def _from_psutil() -> dict[str, Any] | None:
    try:
        import psutil

        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    except Exception:  # noqa: BLE001
        return None
    used = max(0, vm.total - vm.available)
    return _memory_dict(vm.total, used, vm.available, getattr(vm, "free", 0), getattr(vm, "cached", 0),
                        getattr(vm, "shared", 0), sw.used, sw.total, "psutil")


def memory() -> dict[str, Any] | None:
    """RAM and swap, in MiB, with ``percent`` = used / total."""
    system_name = platform.system()
    if system_name == "Linux":
        text = _read("/proc/meminfo")
        host = _from_meminfo(_meminfo(text)) if text else None
        if host is not None:
            return _from_cgroup(host["total_mib"] * _MIB) or host
    if system_name == "Darwin":
        vm_stat = _run(["vm_stat"])
        memsize = _run(["sysctl", "-n", "hw.memsize"])
        if vm_stat and memsize and memsize.strip().isdigit():
            found = _from_vm_stat(vm_stat, int(memsize.strip()), _run(["sysctl", "-n", "vm.swapusage"]) or "")
            if found:
                return found
    return _from_psutil()


# -- temperatures -----------------------------------------------------------------


def _topology(cpu: int, base: Path) -> tuple[int, int] | None:
    package = _read(base / f"cpu{cpu}" / "topology" / "physical_package_id")
    core = _read(base / f"cpu{cpu}" / "topology" / "core_id")
    if package is None or core is None or not package.lstrip("-").isdigit() or not core.isdigit():
        return None
    return int(package), int(core)


def cpu_temperatures(cores: int, *, hwmon: Path = Path("/sys/class/hwmon"),
                     cpus: Path = Path("/sys/devices/system/cpu")) -> dict[str, Any]:
    """``per_core``: one reading (°C, or None) per logical CPU;
    ``package_c``: the package, or the hottest die; ``dies``: named
    readings such as AMD's per-CCD ``Tccd1``; ``note``: what is missing
    and why."""
    result: dict[str, Any] = {"per_core": [None] * max(0, cores), "package_c": None,
                              "dies": {}, "source": "", "note": ""}
    by_core: dict[tuple[int, int], float] = {}
    packages: dict[int, float] = {}
    amd = False
    for sensor in sorted(hwmon.glob("hwmon*")) if hwmon.exists() else []:
        name = _read(sensor / "name") or ""
        if name not in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "soc_thermal"):
            continue
        package_id = None
        readings: list[tuple[str, float]] = []
        for label_file in sorted(sensor.glob("temp*_input")):
            raw = _read(label_file)
            if raw is None or not raw.lstrip("-").isdigit():
                continue
            label = _read(label_file.with_name(label_file.name.replace("_input", "_label"))) or name
            readings.append((label, int(raw) / 1000.0))
        for label, celsius in readings:
            if m := re.match(r"Package id (\d+)", label):
                package_id = int(m.group(1))
                packages[package_id] = celsius
        for label, celsius in readings:
            if m := re.match(r"Core (\d+)", label):
                by_core[(package_id or 0, int(m.group(1)))] = celsius
            elif name in ("k10temp", "zenpower"):
                amd = True
                result["dies"][label] = celsius
            elif name not in ("coretemp",):
                result["dies"][label] = celsius
        result["source"] = name
    if by_core:
        for cpu in range(cores):
            where = _topology(cpu, cpus)
            if where is not None and where in by_core:
                result["per_core"][cpu] = by_core[where]
    if packages:
        result["package_c"] = max(packages.values())
    elif result["dies"]:
        preferred = [v for k, v in result["dies"].items() if k in ("Tctl", "Tdie")]
        result["package_c"] = max(preferred or result["dies"].values())
    if not by_core:
        if amd:
            result["note"] = ("AMD's k10temp driver reports the package (Tctl) and each CCD "
                              "(Tccd), not each core")
        elif platform.system() == "Darwin":
            result["note"] = "macOS gives core temperatures only to privileged tools (powermetrics)"
        elif not result["source"]:
            result["note"] = "no CPU temperature sensor is exposed here"
        else:
            result["note"] = f"the {result['source']} sensor reports no per-core readings"
    return result


# -- audio ----------------------------------------------------------------------


_AUDIO_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_AUDIO_LOCK = threading.Lock()
_AUDIO_TTL = 2.0


def _loud_db() -> float:
    try:
        return float(os.environ.get("TVTOP_MAX_LOUD_DB", LOUD_DB))
    except ValueError:
        return LOUD_DB


def _db(text: str) -> float | None:
    text = text.strip().lower()
    if "-inf" in text:
        return -math.inf
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*db", text)
    return float(m.group(1)) if m else None


def _pulse() -> dict[str, Any] | None:
    default = (_run(["pactl", "get-default-sink"]) or "").strip()
    raw = _run(["pactl", "--format=json", "list", "sinks"])
    if not default or raw is None:
        return None
    try:
        sinks = json.loads(raw)
    except ValueError:
        return None
    sink = next((s for s in sinks if s.get("name") == default), None)
    if sink is None:
        return None
    channels = (sink.get("volume") or {}).values()
    dbs = [d for d in (_db(str(c.get("db", ""))) for c in channels) if d is not None]
    percents = [int(str(c.get("value_percent", "0")).rstrip("%") or 0) for c in channels]
    port = str(sink.get("active_port") or "")
    server = "pipewire" if "pipewire" in json.dumps(sink.get("properties") or {}).lower() else "pulseaudio"
    return {
        "backend": server,
        "device": str(sink.get("description") or default),
        "port": port,
        "headphones": "headphone" in port.lower() or "headset" in port.lower()
                      or "bluez" in default.lower(),
        "volume_db": max(dbs) if dbs else None,
        "volume_percent": max(percents) if percents else None,
        "muted": bool(sink.get("mute")),
        "playing": str(sink.get("state", "")).upper() == "RUNNING",
        "monitor": str(sink.get("monitor_source") or f"{default}.monitor"),
    }


def _alsa() -> dict[str, Any] | None:
    out = _run(["amixer", "sget", "Master"])
    if out is None:
        return None
    dbs = [_db(m) for m in re.findall(r"\[(-?[\d.]+dB|-inf dB)\]", out)]
    percents = [int(p) for p in re.findall(r"\[(\d+)%\]", out)]
    playing = any("state: RUNNING" in (_read(p) or "")
                  for p in Path("/proc/asound").glob("card*/pcm*p/sub*/status"))
    return {
        "backend": "alsa", "device": "Master", "port": "", "headphones": False,
        "volume_db": max([d for d in dbs if d is not None], default=None),
        "volume_percent": max(percents, default=None),
        "muted": "[off]" in out, "playing": playing, "monitor": "",
    }


def _macos() -> dict[str, Any] | None:
    out = _run(["osascript", "-e", "get volume settings"])
    if out is None:
        return None
    m = re.search(r"output volume:(\d+)", out)
    assertions = _run(["pmset", "-g", "assertions"]) or ""
    return {
        "backend": "coreaudio", "device": "default output", "port": "", "headphones": False,
        "volume_db": None,
        "volume_percent": int(m.group(1)) if m else None,
        "muted": "output muted:true" in out,
        "playing": "com.apple.audio" in assertions and "Output" in assertions,
        "monitor": "",
    }


def _signal_level(monitor: str, seconds: float = 0.25) -> tuple[float | None, float | None]:
    """Peak and RMS of what is playing, in dBFS, read from the output's monitor."""
    if not monitor or shutil.which("parec") is None:
        return None, None
    rate = 16000
    want = int(rate * seconds) * 2
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv
            [shutil.which("parec") or "parec", f"--device={monitor}", "--raw", "--format=s16le", "--channels=1",
             f"--rate={rate}", "--latency-msec=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None, None
    data = b""
    deadline = time.monotonic() + seconds + 1.0
    try:
        while len(data) < want and time.monotonic() < deadline and proc.stdout is not None:
            chunk = proc.stdout.read1(want - len(data)) if hasattr(proc.stdout, "read1") else proc.stdout.read(want - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        proc.kill()
        proc.wait(timeout=2)
    samples = struct.unpack(f"<{len(data) // 2}h", data[: len(data) // 2 * 2]) if len(data) >= 2 else ()
    if not samples:
        return None, None
    peak = max(abs(s) for s in samples) / 32768.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
    to_db = lambda v: round(20 * math.log10(v), 1) if v > 0 else -math.inf  # noqa: E731
    return to_db(peak), to_db(rms)


def _judge(state: dict[str, Any]) -> tuple[str, str]:
    """("ok" | "loud" | "unsafe", warning text)."""
    if not state.get("playing") or state.get("muted"):
        return "ok", ""
    db = state.get("volume_db")
    where = "headphones" if state.get("headphones") else "this output"
    if db is not None and db > UNSAFE_DB:
        return "unsafe", (f"UNSAFE: amplified to {db:+.1f} dB (over 100%) while playing — "
                          f"distorts and can damage hearing and speakers")
    loud = _loud_db()
    peak = state.get("signal_peak_dbfs")
    if db is not None and db >= loud and (peak is None or peak >= -12.0):
        return "loud", (f"loud: playing at {db:+.1f} dB on {where} — at or above {loud:+.0f} dB, "
                        f"long listening risks hearing damage")
    if db is None and (state.get("volume_percent") or 0) >= 80:
        return "loud", (f"loud: playing at {state['volume_percent']}% — this system does not "
                        f"report decibels, so this is the volume slider")
    return "ok", ""


def audio(*, sample_signal: bool = True) -> dict[str, Any]:
    """The default output, as the sound server reports it. Cached for two seconds."""
    with _AUDIO_LOCK:
        if _AUDIO_CACHE["value"] is not None and time.monotonic() - _AUDIO_CACHE["at"] < _AUDIO_TTL:
            return _AUDIO_CACHE["value"]
    system_name = platform.system()
    state = None
    if system_name == "Linux":
        state = _pulse() or _alsa()
    elif system_name == "Darwin":
        state = _macos()
    if state is None:
        value: dict[str, Any] = {"available": False, "note": "no sound server answered (pactl, amixer or osascript)"}
    else:
        peak = rms = None
        if sample_signal and state.get("playing") and not state.get("muted"):
            peak, rms = _signal_level(state.get("monitor", ""))
        state.update(available=True, signal_peak_dbfs=peak, signal_rms_dbfs=rms)
        state["level"], state["warning"] = _judge(state)
        state["note"] = ("dB is the output gain the sound server applies (0 dB = 100%); the "
                         "sound level at your ears also depends on the speakers or headphones")
        if state.get("volume_db") is None:
            state["note"] = "this system reports the volume as a percentage only, not in dB"
        state.pop("monitor", None)
        for key in ("volume_db", "signal_peak_dbfs", "signal_rms_dbfs"):
            if state.get(key) == -math.inf:
                state[key] = None
                state[f"{key}_silent"] = True
        value = state
    with _AUDIO_LOCK:
        _AUDIO_CACHE.update(at=time.monotonic(), value=value)
    return value


# -- disks, network, system -------------------------------------------------------


_IO_PREV: dict[str, tuple[float, dict[str, tuple[int, int]]]] = {}


def _rates(kind: str, counters: dict[str, tuple[int, int]]) -> dict[str, tuple[float, float]]:
    now = time.monotonic()
    before = _IO_PREV.get(kind)
    _IO_PREV[kind] = (now, counters)
    if before is None:
        return {}
    elapsed = max(1e-6, now - before[0])
    return {name: (max(0, a - before[1][name][0]) / elapsed, max(0, b - before[1][name][1]) / elapsed)
            for name, (a, b) in counters.items() if name in before[1]}


def disks() -> list[dict[str, Any]]:
    """Mounted filesystems with their space, and read/write speeds per device."""
    try:
        import psutil
    except ImportError:
        return []
    try:
        io = {k: (v.read_bytes, v.write_bytes) for k, v in (psutil.disk_io_counters(perdisk=True) or {}).items()}
    except Exception:  # noqa: BLE001
        io = {}
    speeds = _rates("disk", io)
    out = []
    seen: set[str] = set()
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint in seen or part.fstype in ("squashfs", "tmpfs", "devtmpfs", "overlay"):
            continue
        seen.add(part.mountpoint)
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        device = Path(part.device).name
        read, write = speeds.get(device, (None, None))
        out.append({"mount": part.mountpoint, "device": device, "fstype": part.fstype,
                    "total_bytes": usage.total, "used_bytes": usage.used, "percent": usage.percent,
                    "read_bps": read, "write_bps": write})
    return out


def network() -> list[dict[str, Any]]:
    """Interfaces that are up, with receive/send speeds."""
    try:
        import psutil
    except ImportError:
        return []
    try:
        counters = {k: (v.bytes_recv, v.bytes_sent) for k, v in psutil.net_io_counters(pernic=True).items()}
        stats = psutil.net_if_stats()
    except Exception:  # noqa: BLE001
        return []
    speeds = _rates("net", counters)
    out = []
    for name, (recv, sent) in counters.items():
        if name == "lo" or (name in stats and not stats[name].isup):
            continue
        down, up = speeds.get(name, (None, None))
        out.append({"name": name, "recv_bytes": recv, "sent_bytes": sent, "down_bps": down, "up_bps": up,
                    "link_mbps": stats[name].speed if name in stats else None})
    return out


def system() -> dict[str, Any]:
    """Load, uptime, and the battery when there is one."""
    out: dict[str, Any] = {"hostname": platform.node(), "os": f"{platform.system()} {platform.release()}"}
    try:
        out["load"] = list(os.getloadavg())
    except (OSError, AttributeError):
        out["load"] = None
    try:
        import psutil

        out["uptime_seconds"] = max(0.0, time.time() - psutil.boot_time())
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        out["battery"] = ({"percent": battery.percent, "plugged": battery.power_plugged,
                           "seconds_left": battery.secsleft if battery.secsleft and battery.secsleft > 0 else None}
                          if battery else None)
        out["processes"] = len(psutil.pids())
    except Exception:  # noqa: BLE001
        uptime = _read("/proc/uptime")
        out["uptime_seconds"] = float(uptime.split()[0]) if uptime else None
        out["battery"] = None
    out["cores"] = os.cpu_count()
    return out


_RSS_PREV: dict[int, tuple[float, int]] = {}


def memory_users(limit: int = 15) -> list[dict[str, Any]]:
    """The processes holding the most memory, machine-wide, with how fast
    each is growing (bytes per second since the last call; negative when
    it is giving memory back)."""
    try:
        import psutil
    except ImportError:
        return []
    now = time.monotonic()
    rows = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        info = proc.info
        mem = info.get("memory_info")
        if mem is None:
            continue
        rows.append((info["pid"], info.get("name") or "", " ".join(info.get("cmdline") or []), mem.rss))
    seen = set()
    out = []
    for pid, name, command, rss in sorted(rows, key=lambda r: r[3], reverse=True):
        seen.add(pid)
        before = _RSS_PREV.get(pid)
        growth = (rss - before[1]) / max(1e-6, now - before[0]) if before else None
        _RSS_PREV[pid] = (now, rss)
        if len(out) < limit:
            out.append({"pid": pid, "name": name, "command": command, "rss_bytes": rss,
                        "growth_bps": growth})
    for pid in [p for p in _RSS_PREV if p not in seen]:
        del _RSS_PREV[pid]
    return out
