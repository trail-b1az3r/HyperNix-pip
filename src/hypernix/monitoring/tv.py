"""tv — the kitchen TV.  A btop++-style training dashboard.

Run with the ``tvtop`` console script (registered in
``pyproject.toml``)::

    tvtop                       # auto-detect a hypernix training log
    tvtop --log path/to.log     # explicit log path
    tvtop --no-color            # no ANSI colour
    tvtop --ascii               # ASCII bars + sparkline (no Unicode)

Or programmatically::

    from hypernix.tv import TVTop
    TVTop(log_path="train.log").run()    # blocks until Ctrl-C

What it shows (when actual hypernix training data is present):

* Current training step / total / percent complete with progress bar.
* Loss + LR + throughput row, plus an inline sparkline of the most
  recent loss values.
* Live elapsed wall time + ETA.
* Hardware vitals: CPU% / RAM% / GPU util% / VRAM (via
  ``nvidia-smi`` when available, throttled to once every 3s).
* Recent log tail (last 8 lines, binary chars sanitised).

When a log doesn't contain any parsable training lines yet, the
dashboard renders a clean "waiting for training data…" state
instead of misleading zeros.

Zero hard dependencies — pure stdlib + ANSI escape codes, no
curses, runs in any TTY.  On a non-TTY stdout (CI, redirect to
file) the renderer gracefully degrades to a one-line-per-frame
text mode.

Patches in 0.61.0b1:

* Auto-detect skips logs that have no ``step … loss=…`` lines
  (avoids accidentally tailing a Konsole / browser dev log).
* Log lines are sanitised — non-printable bytes are replaced with
  ``?`` so binary garbage can't corrupt the render.
* Empty-state header when no training data has been parsed yet.
* Two-column hardware bar with mini gauges instead of cramped
  single-line text.
* ``nvidia-smi`` cached for 3 seconds so a 1-second refresh
  doesn't shell out 60×/min.
* Frame-diff render: only writes lines that changed, with cursor-
  home + per-line clear instead of a full screen flush each tick.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

CSI = "\x1b["
CLEAR_SCREEN = f"{CSI}2J{CSI}H"
CLEAR_LINE = f"{CSI}2K"
CURSOR_HOME = f"{CSI}H"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"


def _color(code: int, text: str, *, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{CSI}{code}m{text}{CSI}0m"


def _bold(text: str, *, enabled: bool = True) -> str:
    return _color(1, text, enabled=enabled) if enabled else text


# ---------------------------------------------------------------------------
# Log parsing — generous regex and iterative parser.
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s*(?:/\s*(?P<total>\d+))?\s+loss\s*=\s*(?P<loss>[-\d.eE+]+)"
    r"(?:\s+lr\s*=\s*(?P<lr>[-\d.eE+]+))?"
    r"(?:\s+throughput\s*=\s*(?P<tput>[-\d.eE+]+))?",
    re.IGNORECASE,
)


def _looks_like_training_log(path: Path, *, peek_bytes: int = 16384) -> bool:
    """Cheap classifier — read the first peek_bytes and look for at
    least one ``step ... loss=...`` or ``loss=...`` match.  Used by autodetect to
    skip Konsole / browser / system logs."""
    try:
        with path.open("rb") as fh:
            buf = fh.read(peek_bytes)
    except OSError:
        return False
    try:
        text = buf.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return False
    return bool(
        _STEP_RE.search(text) or
        re.search(r"(?i)loss\s*[:=]", text) or
        re.search(r"(?i)step\s*[:=]", text) or
        re.search(r"\b\d+/\d+\b", text)
    )


# Patch (0.61.1): exempt \r (0x0D) so CRLF-terminated Windows logs
# don't have every line ending sanitised away.  We still strip
# every other C0 / C1 control char.
_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitise(line: str, *, width: int = 200) -> str:
    """Replace non-printable bytes with ``?`` and trim to ``width``."""
    if not line:
        return ""
    line = _PRINTABLE.sub("?", line)
    if len(line) > width:
        line = line[: width - 1] + "…"
    return line


# ---------------------------------------------------------------------------
# Frame model
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    step: int = 0
    total_steps: int | None = None
    loss: float | None = None
    lr: float | None = None
    throughput: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    cpu_percent: float | None = None
    ram_percent: float | None = None
    gpu_util_percent: float | None = None
    gpu_mem_used_mib: int | None = None
    gpu_mem_total_mib: int | None = None
    # 0.61.2 extended hardware fields.
    cpu_per_core: list[float] = field(default_factory=list)
    cpu_history: list[float] = field(default_factory=list)
    ram_history: list[float] = field(default_factory=list)
    gpu_util_history: list[float] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None
    gpu_power_limit_w: float | None = None
    gpu_name: str | None = None
    recent_losses: list[float] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)
    has_training_data: bool = False

    @property
    def progress(self) -> float:
        if self.total_steps and self.total_steps > 0:
            return min(1.0, self.step / self.total_steps)
        return 0.0


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

_SPARK_BARS: tuple[str, ...] = ("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
_ASCII_BARS: tuple[str, ...] = (".", ":", "-", "=", "+", "*", "#", "@")


def sparkline(values: Iterable[float], *, ascii_only: bool = False) -> str:
    vs = [float(v) for v in values]
    if not vs:
        return ""
    bars = _ASCII_BARS if ascii_only else _SPARK_BARS
    lo, hi = min(vs), max(vs)
    if hi == lo:
        return bars[len(bars) // 2] * len(vs)
    out: list[str] = []
    for v in vs:
        idx = int((v - lo) / (hi - lo) * (len(bars) - 1))
        out.append(bars[idx])
    return "".join(out)


def multi_row_graph(
    values: Iterable[float],
    *,
    width: int,
    height: int,
    ascii_only: bool = False,
) -> list[str]:
    """Render a btop++-style multi-row block graph.

    Each column maps to one value; column heights are quantised to
    ``height * 8`` sub-pixels using the Unicode lower-block ladder
    (``▁ ▂ ▃ ▄ ▅ ▆ ▇ █``).  Returns a list of exactly ``height``
    strings, each ``width`` chars wide.

    With ``ascii_only=True`` the block ladder degrades to ``# +``
    fills so non-UTF terminals still get a readable graph.
    """
    if width <= 0 or height <= 0:
        return [""] * max(0, height)
    vs = [float(v) for v in values]
    if not vs:
        return [" " * width] * height

    # Decimate / pad to width.
    if len(vs) > width:
        stride = len(vs) / width
        vs = [vs[min(len(vs) - 1, int(i * stride))] for i in range(width)]
    elif len(vs) < width:
        # Right-align the data: pad with the floor on the left.
        vs = [vs[0]] * (width - len(vs)) + vs

    lo, hi = min(vs), max(vs)
    rng = max(1e-9, hi - lo)
    if ascii_only:
        bars = (" ", ".", ":", "-", "=", "+", "*", "#", "#")
        full = "#"
    else:
        bars = (" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
        full = "█"

    # For each value compute a "tower" of total sub-pixels.  Then
    # render row by row, top-down.
    sub_total = height * 8
    towers = [
        max(0, min(sub_total, int(round((v - lo) / rng * sub_total))))
        for v in vs
    ]

    rows: list[str] = []
    for r in range(height):  # 0 = top, height-1 = bottom
        bottom_subs = (height - 1 - r) * 8           # this row's floor
        top_subs = bottom_subs + 8                    # this row's ceiling
        line: list[str] = []
        for t in towers:
            if t >= top_subs:
                line.append(full)
            elif t <= bottom_subs:
                line.append(" ")
            else:
                line.append(bars[t - bottom_subs])
        rows.append("".join(line))
    return rows


def _block_history_bar(history: list[float], width: int, color_enabled: bool) -> str:
    """Render a single-row block-density history bar from historic percentages.

    Using Unicode density blocks: ' ', '░', '▒', '▓', '█'.
    """
    if not history or width <= 0:
        return " " * width
    # Decimate / pad to width
    vs = list(history)
    if len(vs) > width:
        stride = len(vs) / width
        vs = [vs[min(len(vs) - 1, int(i * stride))] for i in range(width)]
    elif len(vs) < width:
        # Pad with 0.0 on the left
        vs = [0.0] * (width - len(vs)) + vs

    chars = (" ", "░", "▒", "▓", "█")
    out = []
    for v in vs:
        # Map 0-100% to 0-4
        idx = min(4, max(0, int(v / 20.0)))
        ch = chars[idx]
        if color_enabled:
            # Color by intensity: green < 50%, yellow < 80%, red >= 80%
            if v < 50.0:
                color_code = 32 # green
            elif v < 80.0:
                color_code = 33 # yellow
            else:
                color_code = 31 # red
            out.append(f"{CSI}{color_code}m{ch}{CSI}0m")
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Hardware probes (no hard deps)
# ---------------------------------------------------------------------------

def _safe_psutil_percent() -> tuple[float | None, float | None]:
    try:
        import psutil  # type: ignore
        return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
    except Exception:  # noqa: BLE001
        return None, None


def _safe_psutil_per_core() -> list[float] | None:
    """Per-core CPU percentages (one float per logical CPU)."""
    try:
        import psutil  # type: ignore
        return list(psutil.cpu_percent(interval=None, percpu=True))
    except Exception:  # noqa: BLE001
        return None


def _read_proc_stat_per_core() -> list[float] | None:
    """Linux fallback when psutil is not installed.  Reads /proc/stat
    once and computes per-core percentages from the delta against the
    previous call.  First call returns None; subsequent calls return
    a list of percentages, one per core."""
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    cores: list[tuple[int, int]] = []
    for line in lines:
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        parts = line.split()
        nums = [int(x) for x in parts[1:8]]
        cores.append((nums[3], sum(nums)))
    if not cores:
        return None
    prev = getattr(_read_proc_stat_per_core, "_prev", None)
    _read_proc_stat_per_core._prev = cores  # type: ignore[attr-defined]
    if prev is None or len(prev) != len(cores):
        return None
    out: list[float] = []
    for (p_idle, p_total), (idle, total) in zip(prev, cores, strict=False):
        d_idle = idle - p_idle
        d_total = total - p_total
        out.append(max(0.0, 100.0 * (1.0 - d_idle / d_total)) if d_total > 0 else 0.0)
    return out


def _read_memory_breakdown() -> dict[str, float | int] | None:
    """Return a dict with ``total_mib`` / ``used_mib`` / ``free_mib`` /
    ``cached_mib`` / ``swap_used_mib`` / ``swap_total_mib`` /
    ``percent`` — populated from psutil on every platform, /proc/meminfo
    on Linux, or ``None`` when neither works."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return {
            "total_mib": vm.total // (1024 * 1024),
            "used_mib": vm.used // (1024 * 1024),
            "free_mib": vm.available // (1024 * 1024),
            "cached_mib": getattr(vm, "cached", 0) // (1024 * 1024),
            "swap_used_mib": sw.used // (1024 * 1024),
            "swap_total_mib": sw.total // (1024 * 1024),
            "percent": float(vm.percent),
        }
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            data = {
                k.strip(): int(v.split()[0])
                for k, v in (line.split(":", 1) for line in fh if ":" in line)
            }
    except OSError:
        return None
    total = data.get("MemTotal", 0)
    avail = data.get("MemAvailable", 0)
    cached = data.get("Cached", 0)
    swap_total = data.get("SwapTotal", 0)
    swap_free = data.get("SwapFree", 0)
    if not total:
        return None
    used = total - avail
    return {
        "total_mib": total // 1024,
        "used_mib": used // 1024,
        "free_mib": avail // 1024,
        "cached_mib": cached // 1024,
        "swap_used_mib": (swap_total - swap_free) // 1024,
        "swap_total_mib": swap_total // 1024,
        "percent": 100.0 * (1.0 - avail / total),
    }


def _read_proc_stat_cpu() -> float | None:
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            line = fh.readline()
        nums = [int(x) for x in line.split()[1:8]]
        idle, total = nums[3], sum(nums)
        prev = getattr(_read_proc_stat_cpu, "_prev", None)
        _read_proc_stat_cpu._prev = (idle, total)  # type: ignore[attr-defined]
        if prev is None:
            return None
        d_idle, d_total = idle - prev[0], total - prev[1]
        if d_total <= 0:
            return None
        return max(0.0, 100.0 * (1.0 - d_idle / d_total))
    except Exception:  # noqa: BLE001
        return None


def _read_meminfo_percent() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            data = {
                k.strip(): int(v.split()[0])
                for k, v in (line.split(":", 1) for line in fh if ":" in line)
            }
        total = data.get("MemTotal", 0)
        avail = data.get("MemAvailable", 0)
        if not total:
            return None
        return max(0.0, 100.0 * (1.0 - avail / total))
    except Exception:  # noqa: BLE001
        return None


# Throttled nvidia-smi cache so we don't shell out every refresh.
_NVIDIA_CACHE: dict[str, object] = {"at": 0.0, "value": (None, None, None)}
_NVIDIA_TTL_SECONDS = 3.0


def _query_nvidia_smi() -> tuple[int | None, int | None, float | None]:
    """Throttled to once per 3 seconds.  Returns ``(mem_used_mib,
    mem_total_mib, util_percent)`` for backwards-compat."""
    full = _query_nvidia_smi_full()
    return (full["mem_used_mib"], full["mem_total_mib"], full["util_percent"])


def _query_nvidia_smi_full() -> dict[str, Any]:
    """Throttled extended GPU query — adds temperature, power, name.
    Cached for ``_NVIDIA_TTL_SECONDS``."""
    now = time.time()
    if now - float(_NVIDIA_CACHE["at"]) < _NVIDIA_TTL_SECONDS:
        cached = _NVIDIA_CACHE.get("full")
        if isinstance(cached, dict):
            return dict(cached)
    empty: dict[str, Any] = {
        "mem_used_mib": None, "mem_total_mib": None, "util_percent": None,
        "temp_c": None, "power_w": None, "power_limit_w": None, "name": None,
    }
    # Through hypernix.system.gpus rather than nvidia-smi directly, so an
    # AMD card fills this panel too. It was NVIDIA-only, which on a Radeon
    # box meant every GPU field in the dashboard read blank -- and the
    # dashboard is the thing people open *because* they want those
    # numbers.
    try:
        from ..system import gpus as _gpus

        cards = _gpus.detect()
        if not cards:
            full = empty
        else:
            # Card 0. tv's GPU panel shows one card; `tvtop` is the view
            # for a multi-GPU machine.
            card = cards[0]
            full = {
                "mem_used_mib": card.memory_used_mb,
                "mem_total_mib": card.memory_total_mb,
                "util_percent": card.utilization_pct,
                "temp_c": card.temperature_c,
                "power_w": card.power_w,
                "power_limit_w": card.power_limit_w,
                "name": card.name or None,
            }
    except Exception:  # noqa: BLE001
        full = empty
    _NVIDIA_CACHE["at"] = now
    _NVIDIA_CACHE["full"] = full
    _NVIDIA_CACHE["value"] = (full["mem_used_mib"], full["mem_total_mib"], full["util_percent"])
    return dict(full)


def _maybe_int(s: str) -> int | None:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _maybe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Log tail reader — tracks file-position and the latest parsed step
# ---------------------------------------------------------------------------

@dataclass
class LogTail:
    path: Path
    history_size: int = 8
    _last_size: int = field(default=0, init=False)
    _buf: deque[str] = field(default=None, init=False)  # type: ignore[assignment]
    latest_match: re.Match[str] | None = field(default=None, init=False)
    losses: deque[float] = field(default=None, init=False)  # type: ignore[assignment]

    # Parsed training fields
    step: int | None = field(default=None, init=False)
    total_steps: int | None = field(default=None, init=False)
    loss: float | None = field(default=None, init=False)
    lr: float | None = field(default=None, init=False)
    throughput: float | None = field(default=None, init=False)
    has_training_data: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=self.history_size)
        self.losses = deque(maxlen=120)

    def poll(self) -> list[str]:
        if not self.path.exists():
            return []
        new_size = self.path.stat().st_size
        if new_size < self._last_size:
            self._last_size = 0  # rotated / truncated
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._last_size)
                chunk = fh.read()
                self._last_size = fh.tell()
        except OSError:
            return []
        new_lines: list[str] = []
        for raw in chunk.splitlines():
            line = _sanitise(raw)
            if not line.strip():
                continue
            self._buf.append(line)
            new_lines.append(line)

            # Try to match the legacy format if possible
            m = _STEP_RE.search(line)
            if m:
                self.latest_match = m
                if m.group("loss"):
                    try:
                        val = float(m.group("loss"))
                        self.loss = val
                        self.losses.append(val)
                        self.has_training_data = True
                    except ValueError:
                        pass
                if m.group("step"):
                    self.step = int(m.group("step"))
                    self.has_training_data = True
                if m.group("total"):
                    self.total_steps = int(m.group("total"))
                if m.group("lr"):
                    try:
                        self.lr = float(m.group("lr"))
                    except ValueError:
                        pass
                if m.group("tput"):
                    try:
                        self.throughput = float(m.group("tput"))
                    except ValueError:
                        pass
                continue

            # Otherwise, use resilient iterative matching:
            matched_any = False

            # 1. Loss matching: loss=... or loss:... or 'loss':...
            loss_match = re.search(r"(?i)\bloss\s*[:=]\s*(?P<loss>[-\d.eE+]+|[+-]?inf|[+-]?nan)", line)
            if loss_match:
                try:
                    val = float(loss_match.group("loss"))
                    self.loss = val
                    self.losses.append(val)
                    self.has_training_data = True
                    matched_any = True
                except ValueError:
                    pass

            # 2. Step and Total steps matching: step=value/value or epoch=value/value or [value/value] or step/total
            step_total_match = re.search(
                r"(?i)(?:step\s*[:=]?\s*|epoch\s*[:=]?\s*|\[|\b)(?P<step>\d+)\s*/\s*(?P<total>\d+)\b", line
            )
            if step_total_match:
                self.step = int(step_total_match.group("step"))
                self.total_steps = int(step_total_match.group("total"))
                self.has_training_data = True
                matched_any = True
            else:
                # Try single step
                step_match = re.search(r"(?i)\bstep\s*[:=]\s*(?P<step>\d+)\b", line)
                if not step_match:
                     step_match = re.search(r"(?i)\bstep\s+(?P<step>\d+)\b", line)
                if step_match:
                     self.step = int(step_match.group("step"))
                     self.has_training_data = True
                     matched_any = True

            # 3. Percentage matching for step simulation: e.g. 50%
            pct_match = re.search(r"\b(?P<pct>\d+(?:\.\d+)?)%", line)
            if pct_match:
                pct = float(pct_match.group("pct"))
                if self.total_steps and self.total_steps > 0:
                    self.step = int((pct / 100.0) * self.total_steps)
                self.has_training_data = True
                matched_any = True

            # 4. Learning rate matching: lr=... or lr:...
            lr_match = re.search(r"(?i)\blr\s*[:=]\s*(?P<lr>[-\d.eE+]+)", line)
            if lr_match:
                try:
                    self.lr = float(lr_match.group("lr"))
                    self.has_training_data = True
                    matched_any = True
                except ValueError:
                    pass

            # 5. Throughput matching: tput=... or throughput=... or suffix it/s, steps/s
            tput_match = re.search(r"(?i)\b(?:tput|throughput)\s*[:=]\s*(?P<tput>[-\d.eE+]+)", line)
            if tput_match:
                try:
                    self.throughput = float(tput_match.group("tput"))
                    self.has_training_data = True
                    matched_any = True
                except ValueError:
                    pass
            else:
                tput_unit_match = re.search(
                    r"\b(?P<tput>[-\d.eE+]+)\s*(?:it/s|s/s|steps/s|step/s|samples/s|items/s)", line
                )
                if tput_unit_match:
                    try:
                        self.throughput = float(tput_unit_match.group("tput"))
                        self.has_training_data = True
                        matched_any = True
                    except ValueError:
                        pass

            if matched_any:
                self.has_training_data = True

        return new_lines

    @property
    def tail(self) -> list[str]:
        return list(self._buf)


# ---------------------------------------------------------------------------
# TVTop
# ---------------------------------------------------------------------------

@dataclass
class TVTop:
    log_path: Path | str | None = None
    refresh_seconds: float = 1.0
    color: bool = True
    ascii_only: bool = False
    width: int | None = None
    small_mode: bool = False
    started_at: float = field(default_factory=time.time)
    log_tail: LogTail | None = field(default=None, init=False)
    _prev_render: str = field(default="", init=False, repr=False)
    # 0.61.2: rolling hardware history for the multi-row time graphs.
    _cpu_history: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False, repr=False)
    _ram_history: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False, repr=False)
    _gpu_util_history: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_tail = LogTail(Path(self.log_path), history_size=8)

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def latest_frame(self) -> Frame:
        f = Frame()
        if self.log_tail is not None:
            self.log_tail.poll()
            if self.log_tail.has_training_data:
                f.has_training_data = True
                f.step = self.log_tail.step or 0
                f.total_steps = self.log_tail.total_steps
                f.loss = self.log_tail.loss
                f.lr = self.log_tail.lr
                f.throughput = self.log_tail.throughput
            elif self.log_tail.latest_match is not None:
                m = self.log_tail.latest_match
                f.has_training_data = True
                f.step = int(m.group("step"))
                if m.group("total"):
                    f.total_steps = int(m.group("total"))
                if m.group("loss"):
                    f.loss = float(m.group("loss"))
                if m.group("lr"):
                    f.lr = float(m.group("lr"))
                if m.group("tput"):
                    f.throughput = float(m.group("tput"))
            f.recent_losses = list(self.log_tail.losses)
            f.log_tail = list(self.log_tail.tail)
        f.elapsed_seconds = time.time() - self.started_at
        if f.has_training_data and f.throughput is None and f.step > 0 and f.elapsed_seconds > 0:
            f.throughput = f.step / f.elapsed_seconds
        if f.has_training_data and f.total_steps and f.throughput and f.throughput > 0:
            f.eta_seconds = max(0.0, (f.total_steps - f.step) / f.throughput)
        cpu, ram = _safe_psutil_percent()
        if cpu is None:
            cpu = _read_proc_stat_cpu()
        # Per-core: psutil first, then a /proc/stat fallback.
        per_core = _safe_psutil_per_core()
        if per_core is None:
            per_core = _read_proc_stat_per_core() or []
        f.cpu_per_core = list(per_core)
        # Memory breakdown.
        mem = _read_memory_breakdown() or {}
        f.memory = mem
        if ram is None and mem.get("percent") is not None:
            ram = float(mem["percent"])
        f.cpu_percent = cpu
        f.ram_percent = ram
        # Extended GPU stats — also keep the legacy 3-tuple in sync.
        gpu = _query_nvidia_smi_full()
        f.gpu_mem_used_mib = gpu["mem_used_mib"]
        f.gpu_mem_total_mib = gpu["mem_total_mib"]
        f.gpu_util_percent = gpu["util_percent"]
        f.gpu_temp_c = gpu["temp_c"]
        f.gpu_power_w = gpu["power_w"]
        f.gpu_power_limit_w = gpu["power_limit_w"]
        f.gpu_name = gpu["name"]
        # Push into rolling histories so render() has time-series data.
        if cpu is not None:
            self._cpu_history.append(cpu)
        if ram is not None:
            self._ram_history.append(ram)
        if gpu["util_percent"] is not None:
            self._gpu_util_history.append(gpu["util_percent"])
        f.cpu_history = list(self._cpu_history)
        f.ram_history = list(self._ram_history)
        f.gpu_util_history = list(self._gpu_util_history)
        return f

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _term_width(self) -> int:
        if self.width is not None:
            return self.width
        try:
            return shutil.get_terminal_size((100, 24)).columns
        except Exception:  # noqa: BLE001
            return 100

    def _bar(self, frac: float, width: int, *, fill: str | None = None) -> str:
        frac = max(0.0, min(1.0, frac))
        filled = int(round(frac * width))
        empty = width - filled
        if self.ascii_only:
            ch_full = fill or "#"
            ch_empty = "."
        else:
            ch_full = fill or "█"
            ch_empty = "░"
        return ch_full * filled + ch_empty * empty

    def _gauge(self, label: str, pct: float | None, width: int) -> str:
        c = self.color and not self.ascii_only
        if pct is None:
            bar = self._bar(0.0, width)
            return f"{label:<5} {bar} {' --':>6}"
        # Colour by load: green / yellow / red.
        if c:
            code = 32 if pct < 60 else (33 if pct < 85 else 31)
            return f"{label:<5} {_color(code, self._bar(pct / 100.0, width), enabled=c)} {pct:>5.1f}%"
        return f"{label:<5} {self._bar(pct / 100.0, width)} {pct:>5.1f}%"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, f: Frame) -> str:
        return _render_dashboard(self, f)

    # ------------------------------------------------------------------
    # Run loop — minimised flicker via cursor-home + per-line clear
    # ------------------------------------------------------------------

    def is_tty(self) -> bool:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())

    def run(self, *, max_frames: int | None = None) -> None:
        """Run the training dashboard.

        On a real TTY, uses ``rich.live.Live`` for a flicker-free
        btop++-style display when ``rich`` is available, falling back
        to the classic ANSI cursor-home loop otherwise.  On a non-TTY
        (CI / piped output) it prints one line per frame.
        """
        tty = self.is_tty()
        # Try rich first when on a TTY.
        if tty:
            try:
                self._run_rich(max_frames=max_frames)
                return
            except ImportError:
                pass  # rich not installed — fall through to classic
        # Classic ANSI loop (fallback or non-TTY).
        try:
            if tty:
                sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
                sys.stdout.flush()
            n = 0
            while True:
                frame = self.latest_frame()
                if tty:
                    body = self.render(frame)
                    if body != self._prev_render:
                        out = CURSOR_HOME
                        for line in body.splitlines():
                            out += line + CLEAR_LINE + "\n"
                        prev_count = len(self._prev_render.splitlines())
                        cur_count = len(body.splitlines())
                        for _ in range(max(0, prev_count - cur_count)):
                            out += CLEAR_LINE + "\n"
                        sys.stdout.write(out)
                        sys.stdout.flush()
                        self._prev_render = body
                else:
                    sys.stdout.write(self.render_one_line(frame) + "\n")
                    sys.stdout.flush()
                n += 1
                if max_frames is not None and n >= max_frames:
                    return
                time.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            return
        finally:
            if tty:
                sys.stdout.write(SHOW_CURSOR + "\n")
                sys.stdout.flush()

    def _run_rich(self, *, max_frames: int | None = None) -> None:
        """Rich-powered live dashboard (btop++ style)."""
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        from rich.table import Table
        from rich.text import Text

        console = Console()

        def _build_layout(f: Frame) -> Layout:

            # ── Training progress ──────────────────────────────────────
            prog = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}[/]"),
                BarColumn(bar_width=None, complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
                TextColumn("\u2022 {task.completed}/{task.total} steps"),
                expand=True,
            )
            total = f.total_steps or 1
            desc = "Training" if f.has_training_data else "Waiting for data..."
            prog.add_task(desc, total=total, completed=f.step)
            prog_panel = Panel(prog, title="[bold cyan]Progress[/]", border_style="cyan")

            # ── Metrics table ──────────────────────────────────────────
            metrics = Table(show_header=False, box=None, expand=True, padding=(0, 1))
            metrics.add_column(style="dim")
            metrics.add_column(style="bold white")
            if f.has_training_data:
                metrics.add_row("Loss",     f"{f.loss:.4f}" if f.loss is not None else "—")
                metrics.add_row("LR",       f"{f.lr:.2e}" if f.lr is not None else "—")
                metrics.add_row("Tput",     f"{f.throughput:.2f}/s" if f.throughput else "—")
                metrics.add_row("Elapsed",  _fmt_duration(f.elapsed_seconds))
                metrics.add_row("ETA",      _fmt_duration(f.eta_seconds) if f.eta_seconds is not None else "—")
            else:
                metrics.add_row("Status", "[yellow]Waiting for training data...[/]")
                metrics.add_row("Log",    str(self.log_path or "(none)"))
            metrics_panel = Panel(metrics, title="[bold yellow]Metrics[/]", border_style="yellow")

            # ── Sparkline / loss curve ─────────────────────────────────
            if f.recent_losses:
                spark = sparkline(f.recent_losses, ascii_only=self.ascii_only)
                curve_body = Text(spark, style="cyan")
            else:
                curve_body = Text("(no loss data yet)", style="dim")
            curve_panel = Panel(curve_body, title="[bold]Loss Curve[/]", border_style="blue")

            # ── Hardware ───────────────────────────────────────────────
            hw = Table(show_header=False, box=None, expand=True, padding=(0, 1))
            hw.add_column(style="dim")
            hw.add_column(style="bold white")
            hw.add_row("CPU",   f"{f.cpu_percent:.1f}%" if f.cpu_percent is not None else "—")
            hw.add_row("RAM",   f"{f.ram_percent:.1f}%" if f.ram_percent is not None else "—")
            hw.add_row("GPU",   f"{f.gpu_util_percent:.1f}%" if f.gpu_util_percent is not None else "—")
            if f.gpu_mem_used_mib and f.gpu_mem_total_mib:
                hw.add_row("VRAM", f"{f.gpu_mem_used_mib}/{f.gpu_mem_total_mib} MiB")
            if f.gpu_temp_c is not None:
                hw.add_row("Temp", f"{f.gpu_temp_c:.1f}\u00b0C")
            if f.gpu_name:
                hw.add_row("GPU Name", f.gpu_name)
            hw_panel = Panel(hw, title="[bold green]Hardware[/]", border_style="green")

            # ── Log tail ───────────────────────────────────────────────
            tail_lines = (f.log_tail or [])[-6:]
            tail_text = Text("\n".join(tail_lines) if tail_lines else "(no log lines yet)", style="dim")
            log_panel = Panel(tail_text, title="[bold]Log Tail[/]", border_style="white")

            # ── Assemble layout ────────────────────────────────────────
            layout = Layout()
            if self.small_mode:
                # Compact: just progress + metrics stacked
                layout.split_column(
                    Layout(prog_panel, size=4),
                    Layout(metrics_panel),
                )
            else:
                layout.split_column(
                    Layout(name="top"),
                    Layout(curve_panel, size=7),
                    Layout(log_panel, size=8),
                )
                layout["top"].split_row(
                    Layout(name="top_left", ratio=3),
                    Layout(hw_panel, ratio=2),
                )
                layout["top_left"].split_column(
                    Layout(prog_panel, size=4),
                    Layout(metrics_panel),
                )
            return layout

        n = 0
        try:
            frame = self.latest_frame()
            with Live(
                _build_layout(frame),
                console=console,
                refresh_per_second=max(1, int(1.0 / max(0.1, self.refresh_seconds))),
                screen=True,
            ) as live:
                while True:
                    frame = self.latest_frame()
                    live.update(_build_layout(frame))
                    n += 1
                    if max_frames is not None and n >= max_frames:
                        return
                    time.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            return

    def render_one_line(self, f: Frame) -> str:
        if not f.has_training_data:
            return f"[tvtop] waiting for training data... ({_fmt_duration(f.elapsed_seconds)} elapsed)"
        return (
            f"step={f.step}/{f.total_steps or '-'}  loss="
            f"{f.loss if f.loss is not None else '-'}"
            f"  tput={f.throughput if f.throughput else '-'}  "
            f"eta={_fmt_duration(f.eta_seconds) if f.eta_seconds else '-'}"
        )


# ---------------------------------------------------------------------------
# btop++-style panel rendering (v0.61.0b1)
# ---------------------------------------------------------------------------

def _panel_chars(ascii_only: bool) -> dict[str, str]:
    if ascii_only:
        return {
            "tl": "+", "tr": "+", "bl": "+", "br": "+",
            "h": "-", "v": "|",
        }
    return {
        # Rounded corners — a touch nicer than btop++'s sharp ones.
        "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
        "h": "─", "v": "│",
    }


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)


def _visible_len(s: str) -> int:
    return len(_strip_ansi(s))


def _pad(s: str, width: int) -> str:
    """Pad-or-truncate a string to ``width`` visible characters
    (ignoring ANSI escapes)."""
    vis = _visible_len(s)
    if vis == width:
        return s
    if vis < width:
        return s + " " * (width - vis)
    # Truncate visible content — drop ANSI for a safe cut.  Re-applying
    # colour at the cut boundary is ugly; better to render plain when
    # we have to clip.
    plain = _strip_ansi(s)
    return plain[: max(0, width - 1)] + ("…" if width > 0 else "")


def _frame_panel(
    title: str,
    body: list[str],
    *,
    width: int,
    ascii_only: bool,
    color_enabled: bool,
    title_color: int = 36,
) -> list[str]:
    """Wrap ``body`` (list of pre-padded interior lines) in a framed
    panel ``width`` cells wide.  Interior is ``width - 2``."""
    ch = _panel_chars(ascii_only)
    inner_w = max(1, width - 2)
    title_text = f" {title} "
    title_render = _color(title_color, _bold(title_text, enabled=color_enabled), enabled=color_enabled)
    title_vis = len(title_text)
    fill_after = max(0, inner_w - 2 - title_vis)
    top = ch["tl"] + ch["h"] + title_render + ch["h"] * fill_after + ch["tr"]
    bot = ch["bl"] + ch["h"] * inner_w + ch["br"]
    rows = [top]
    for ln in body:
        rows.append(ch["v"] + _pad(ln, inner_w) + ch["v"])
    rows.append(bot)
    return rows


def _hcat(left: list[str], right: list[str], *, gap: int = 1) -> list[str]:
    """Stack two equal-length panel row lists side by side."""
    n = max(len(left), len(right))
    while len(left) < n:
        left.append("")
    while len(right) < n:
        right.append("")
    return [a + " " * gap + b for a, b in zip(left, right, strict=False)]


def _bar_str(frac: float, width: int, *, ascii_only: bool, color_enabled: bool) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    empty = width - filled
    if ascii_only:
        s = "#" * filled + "." * empty
    else:
        s = "█" * filled + "░" * empty
    if color_enabled:
        # Green → yellow → red as the bar fills.
        code = 32 if frac < 0.6 else (33 if frac < 0.85 else 31)
        return _color(code, s, enabled=True)
    return s


def _gauge_line(label: str, pct: float | None, bar_w: int, *, ascii_only: bool, color_enabled: bool) -> str:
    if pct is None:
        bar = _bar_str(0.0, bar_w, ascii_only=ascii_only, color_enabled=False)
        return f"{label:<5} {bar}    --"
    bar = _bar_str(pct / 100.0, bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
    return f"{label:<5} {bar} {pct:5.1f}%"


def _render_dashboard(self_tv: TVTop, f: Frame) -> str:
    w = max(72, self_tv._term_width())
    c = self_tv.color and not self_tv.ascii_only
    aa = self_tv.ascii_only

    # Title bar
    title_text = " hypernix tvtop · training dashboard "
    title = _bold(title_text, enabled=c)
    title_line = _color(96, title, enabled=c) + " " * max(0, w - _visible_len(title))

    # Two columns; CPU panel on the left is wide so the per-core grid
    # has room.  Memory + GPU sit below them.
    left_w = max(36, int(w * 0.50))
    right_w = w - left_w - 1
    if right_w < 30:
        right_w = 30
        left_w = max(36, w - right_w - 1)

    cpu_panel = _build_cpu_panel(f, width=left_w, ascii_only=aa, color_enabled=c)
    tr_panel = _build_training_panel(self_tv, f, width=right_w, ascii_only=aa, color_enabled=c)
    top = _hcat(cpu_panel, tr_panel, gap=1)

    mem_panel = _build_memory_panel(f, width=left_w, ascii_only=aa, color_enabled=c)
    gpu_panel = _build_gpu_panel(f, width=right_w, ascii_only=aa, color_enabled=c)
    middle = _hcat(mem_panel, gpu_panel, gap=1)

    # Loss curve (full width)
    graph_inner = w - 4
    loss_title = "loss curve"
    if f.recent_losses:
        # Generate an estimated future curve to append
        if len(f.recent_losses) >= 5:
            recent_ma = sum(f.recent_losses[-3:]) / 3.0
            prev_ma = sum(f.recent_losses[-6:-3]) / 3.0 if len(f.recent_losses) >= 6 else f.recent_losses[0]
            slope = (recent_ma - prev_ma) / 3.0

            future_losses = []
            cur = f.recent_losses[-1]
            temp_slope = slope
            for _ in range(15):  # Predict next 15 steps
                cur += temp_slope
                cur = max(0.0, cur)
                future_losses.append(cur)
                temp_slope *= 0.85  # Dampen the slope (exponential decay simulation)

            combined = f.recent_losses + future_losses
            est_val = future_losses[-1]
            loss_title = (
                f"loss curve (min: {min(f.recent_losses):.4f} · max: {max(f.recent_losses):.4f} · "
                f"current: {f.loss:.4f} · est: {est_val:.4f})"
            )
        else:
            combined = f.recent_losses
            curr_val = f.loss if f.loss is not None else f.recent_losses[-1]
            loss_title = f"loss curve (min: {min(f.recent_losses):.4f} · max: {max(f.recent_losses):.4f} · current: {curr_val:.4f})"

        graph_rows = multi_row_graph(
            combined, width=graph_inner, height=5, ascii_only=aa,
        )

        # Color real data cyan, predicted data dim/magenta
        if c:
            colored_rows = []
            for r in graph_rows:
                if len(combined) > 15:
                    pred_chars = int((15 / len(combined)) * graph_inner)
                    real_chars = max(0, graph_inner - pred_chars)
                    real_str = r[:real_chars]
                    pred_str = r[real_chars:]
                    colored_rows.append(_color(36, real_str, enabled=c) + _color(35, pred_str, enabled=c))
                else:
                    colored_rows.append(_color(36, r, enabled=c))
            graph_rows = colored_rows
    else:
        graph_rows = ["(no loss values yet — graph fills in once `loss=…` lines arrive)"]
        graph_rows += [""] * 4
    loss_panel = _frame_panel(
        loss_title, graph_rows, width=w,
        ascii_only=aa, color_enabled=c, title_color=33,
    )

    # Recent log (full width)
    log_lines = (f.log_tail or [])[-6:]
    log_body = [raw[: w - 6] for raw in log_lines] or ["(no log lines yet)"]
    while len(log_body) < 6:
        log_body.append("")
    log_panel = _frame_panel(
        "recent log", log_body, width=w,
        ascii_only=aa, color_enabled=c, title_color=35,
    )

    footer = _color(
        90,
        f" press Ctrl-C to quit · refresh {self_tv.refresh_seconds:.1f}s · "
        f"width={w} · {len(f.cpu_per_core)} cores · gpu={f.gpu_name or '—'}",
        enabled=c,
    )

    return "\n".join([title_line, *top, *middle, *loss_panel, *log_panel, footer])


# ---------------------------------------------------------------------------
# Panel builders (0.61.2)
# ---------------------------------------------------------------------------

def _build_cpu_panel(
    f: Frame, *, width: int, ascii_only: bool, color_enabled: bool,
) -> list[str]:
    inner = max(8, width - 2)
    body: list[str] = []
    bar_w = max(8, inner - 16)
    body.append(_gauge_line(
        "TOTAL", f.cpu_percent, bar_w,
        ascii_only=ascii_only, color_enabled=color_enabled,
    ))
    body.append("")
    if f.cpu_per_core:
        # Two-column grid of mini bars.  Each cell renders as
        # ``cN <bar> NN.N%`` — cell width = bar_w + 12 chars.
        cell_total = max(20, (inner - 1) // 2)
        cell_bar_w = max(4, cell_total - 12)
        cells: list[str] = []
        for i, pct in enumerate(f.cpu_per_core):
            bar = _bar_str(
                pct / 100.0, cell_bar_w,
                ascii_only=ascii_only, color_enabled=color_enabled,
            )
            cells.append(f"c{i:>2} {bar} {pct:5.1f}%")
        for i in range(0, len(cells), 2):
            left = cells[i]
            right = cells[i + 1] if i + 1 < len(cells) else ""
            body.append(_pad(left, cell_total) + " " + right)
        body.append("")
    if f.cpu_history:
        block_bar = _block_history_bar(f.cpu_history, width=max(1, inner - 14), color_enabled=color_enabled)
        body.append(_color(36, f"block hist: [{block_bar}]", enabled=color_enabled))
        body.append(_color(36, "history (last ~2 min):", enabled=color_enabled))
        graph = multi_row_graph(
            f.cpu_history, width=inner, height=3, ascii_only=ascii_only,
        )
        body.extend(_color(36, r, enabled=color_enabled) for r in graph)
    return _frame_panel(
        "cpu", body, width=width,
        ascii_only=ascii_only, color_enabled=color_enabled, title_color=32,
    )


def _build_memory_panel(
    f: Frame, *, width: int, ascii_only: bool, color_enabled: bool,
) -> list[str]:
    inner = max(8, width - 2)
    bar_w = max(8, inner - 24)
    mem = f.memory or {}
    body: list[str] = []
    if mem.get("total_mib"):
        used = int(mem.get("used_mib", 0))
        total = int(mem["total_mib"])
        free = int(mem.get("free_mib", 0))
        cached = int(mem.get("cached_mib", 0))
        used_pct = 100.0 * used / max(1, total)
        bar = _bar_str(used_pct / 100.0, bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
        body.append(f"USED  {bar} {used:>6}/{total:<6} MiB")
        if cached:
            cb = _bar_str(cached / max(1, total), bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
            body.append(f"CACHE {cb} {cached:>6} MiB")
        fb = _bar_str(free / max(1, total), bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
        body.append(f"FREE  {fb} {free:>6} MiB")
        sw_total = int(mem.get("swap_total_mib", 0) or 0)
        if sw_total:
            sw_used = int(mem.get("swap_used_mib", 0) or 0)
            sw_pct = 100.0 * sw_used / max(1, sw_total)
            sb = _bar_str(sw_pct / 100.0, bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
            body.append(f"SWAP  {sb} {sw_used:>6}/{sw_total:<6} MiB")
    elif f.ram_percent is not None:
        body.append(f"RAM {f.ram_percent:.1f}%")
    else:
        body.append("(no memory data)")
    if f.ram_history:
        body.append("")
        block_bar = _block_history_bar(f.ram_history, width=max(1, inner - 14), color_enabled=color_enabled)
        body.append(_color(35, f"block hist: [{block_bar}]", enabled=color_enabled))
        body.append(_color(35, "history:", enabled=color_enabled))
        graph = multi_row_graph(f.ram_history, width=inner, height=2, ascii_only=ascii_only)
        body.extend(_color(35, r, enabled=color_enabled) for r in graph)
    return _frame_panel(
        "memory", body, width=width,
        ascii_only=ascii_only, color_enabled=color_enabled, title_color=35,
    )


def _build_gpu_panel(
    f: Frame, *, width: int, ascii_only: bool, color_enabled: bool,
) -> list[str]:
    inner = max(8, width - 2)
    bar_w = max(8, inner - 18)
    body: list[str] = []
    if f.gpu_util_percent is None and f.gpu_mem_total_mib is None:
        body.append("(no GPU detected — install nvidia-smi to populate)")
        body.extend([""] * 5)
        return _frame_panel(
            "gpu", body, width=width,
            ascii_only=ascii_only, color_enabled=color_enabled, title_color=92,
        )
    if f.gpu_name:
        body.append(_color(96, f.gpu_name[: inner], enabled=color_enabled))
    body.append(_gauge_line(
        "UTIL", f.gpu_util_percent, bar_w,
        ascii_only=ascii_only, color_enabled=color_enabled,
    ))
    if f.gpu_mem_used_mib is not None and f.gpu_mem_total_mib:
        mem_pct = 100.0 * f.gpu_mem_used_mib / max(1, f.gpu_mem_total_mib)
        body.append(_gauge_line(
            "VRAM", mem_pct, bar_w,
            ascii_only=ascii_only, color_enabled=color_enabled,
        ))
        body.append(f"      {f.gpu_mem_used_mib:>6}/{f.gpu_mem_total_mib:<6} MiB")
    if f.gpu_temp_c is not None:
        # Map 30→100°C as the bar range so a hot GPU stands out.
        temp_norm = max(0.0, min(1.0, (f.gpu_temp_c - 30) / 70.0))
        bar = _bar_str(temp_norm, bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
        body.append(f"TEMP  {bar} {f.gpu_temp_c:5.1f}°C")
    if f.gpu_power_w is not None and f.gpu_power_limit_w:
        pwr_pct = 100.0 * f.gpu_power_w / max(1.0, f.gpu_power_limit_w)
        bar = _bar_str(pwr_pct / 100.0, bar_w, ascii_only=ascii_only, color_enabled=color_enabled)
        body.append(f"PWR   {bar} {f.gpu_power_w:5.1f}/{f.gpu_power_limit_w:<5.0f} W")
    if f.gpu_util_history:
        body.append("")
        block_bar = _block_history_bar(f.gpu_util_history, width=max(1, inner - 14), color_enabled=color_enabled)
        body.append(_color(92, f"block hist: [{block_bar}]", enabled=color_enabled))
        body.append(_color(92, "util history:", enabled=color_enabled))
        graph = multi_row_graph(f.gpu_util_history, width=inner, height=2, ascii_only=ascii_only)
        body.extend(_color(92, r, enabled=color_enabled) for r in graph)
    return _frame_panel(
        "gpu", body, width=width,
        ascii_only=ascii_only, color_enabled=color_enabled, title_color=92,
    )


def _build_training_panel(
    self_tv: TVTop, f: Frame, *,
    width: int, ascii_only: bool, color_enabled: bool,
) -> list[str]:
    inner = max(10, width - 2)
    body: list[str] = []
    if f.has_training_data:
        prog_w = max(10, inner - 28)
        pct = f.progress * 100.0
        step_field = f"step {f.step:>6}"
        if f.total_steps:
            step_field += f" / {f.total_steps}"
        prog_bar = _bar_str(f.progress, prog_w, ascii_only=ascii_only, color_enabled=color_enabled)
        body.append(_color(36, f"{step_field:<18} {prog_bar} {pct:5.1f}%", enabled=color_enabled))
        body.append("")
        loss_s = f"loss  {f.loss:.4f}" if f.loss is not None else "loss  —"
        lr_s = f"lr  {f.lr:.2e}" if f.lr is not None else "lr  —"
        tput_s = f"tput  {f.throughput:.2f}/s" if f.throughput else "tput  —"
        body.append(_color(33, f"{loss_s:<22}{lr_s}", enabled=color_enabled))
        body.append(_color(33, tput_s, enabled=color_enabled))
        elapsed = _fmt_duration(f.elapsed_seconds)
        eta = _fmt_duration(f.eta_seconds) if f.eta_seconds is not None else "—"
        body.append(_color(35, f"elapsed  {elapsed:<10}   ETA  {eta}", enabled=color_enabled))
    else:
        wait_label = "⏳ waiting for training data…" if not ascii_only else "[waiting for training data...]"
        body.append(_color(33, wait_label, enabled=color_enabled))
        body.append("")
        body.append(f"log: {self_tv.log_path or '(none)'}")
        body.append("")
        body.append("(point --log at a file with `step N/M loss=…` lines)")
    return _frame_panel(
        "training", body, width=width,
        ascii_only=ascii_only, color_enabled=color_enabled, title_color=36,
    )


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Auto-detect — only pick logs that look like training logs
# ---------------------------------------------------------------------------

def _autodetect_log(start: Path = Path(".")) -> Path | None:
    """Pick the newest training-shaped log under ``start``.  We
    short-list candidates by mtime, then rank by:

    1. Is the filename ``train*.log`` / ``*training*.log``?
    2. Does the file contain a ``step ... loss=...`` match in the
       first 16 KiB?

    Logs that match neither are still acceptable as a fallback,
    but only if no shaped log exists.  This stops the dashboard
    from latching onto random Konsole / browser / system logs.

    Before doing any of that, ``~/checkpoints/train.log`` is checked
    first and returned immediately if present -- it's the conventional
    location hypernix training runs write to, so tvtop finds it without
    needing a full glob scan of the current directory.
    """
    home_default = Path.home() / "checkpoints" / "train.log"
    if home_default.exists():
        return home_default

    candidates: list[tuple[float, Path]] = []
    for pattern in ("**/train*.log", "**/*training*.log", "**/*.log"):
        for p in start.glob(pattern):
            try:
                name = p.name.lower()
                if "chromium" in name or "chrome" in name:
                    continue
                candidates.append((p.stat().st_mtime, p))
            except OSError:
                continue
    seen: set[Path] = set()
    uniq: list[tuple[float, Path]] = []
    for mtime, p in sorted(candidates, key=lambda x: -x[0]):
        if p in seen:
            continue
        seen.add(p)
        uniq.append((mtime, p))
    # Prefer paths that look like training logs.
    shaped = [p for _t, p in uniq if _looks_like_training_log(p)]
    if shaped:
        return shaped[0]
    name_pref = [p for _t, p in uniq if "train" in p.name.lower()]
    if name_pref:
        return name_pref[0]
    # Do not fall back to arbitrary logs (prevents tailing binary chromium logs like 000003.log)
    return None


# ---------------------------------------------------------------------------
# CLI entry point — installed as ``tvtop``
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    log: Path | None = None
    color = True
    ascii_only = False
    refresh = 1.0
    small_mode = False
    if "--no-color" in args:
        color = False
        args.remove("--no-color")
    if "--ascii" in args:
        ascii_only = True
        args.remove("--ascii")
    if "-s" in args or "--small" in args:
        small_mode = True
        args = [a for a in args if a not in ("-s", "--small")]
    if "--refresh" in args:
        i = args.index("--refresh")
        if i + 1 < len(args):
            refresh = float(args[i + 1])
            del args[i : i + 2]
    if "--log" in args:
        i = args.index("--log")
        if i + 1 < len(args):
            log = Path(args[i + 1])
            del args[i : i + 2]
    if "--help" in args or "-h" in args:
        print(
            "usage: tvtop [--log path] [--no-color] [--ascii] "
            "[--refresh SECONDS] [-s|--small]\n"
            "Auto-detects newest *.log under cwd that contains "
            "`step N/M loss=X` lines (so a stray Konsole / browser\n"
            "log won't be picked).  Pass --log <path> for explicit "
            "selection.\n"
            "  -s, --small   compact mode for smaller terminals",
        )
        return 0

    # --- Startup animation -----------------------------------------------
    _is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if color and _is_tty:
        try:
            from hypernix.timing.spinner import Spinner, anime_print
            anime_print("tvtop", style="glitch", delay=0.04)
        except Exception:
            pass

    if log is None:
        if color and _is_tty:
            try:
                from hypernix.timing.spinner import Spinner
                with Spinner("Detecting training log...", style="dots"):
                    log = _autodetect_log()
            except Exception:
                log = _autodetect_log()
        else:
            log = _autodetect_log()
    if log is None:
        print(
            "tvtop: no training log found.  Pass --log <path>, or run "
            "from a directory containing a *.log written by hypernix "
            "training (looking for lines like `step 100/2000 loss=2.3`).",
            file=sys.stderr,
        )
        return 2
    if not _looks_like_training_log(log):
        print(
            f"[tvtop] warning: {log} doesn't contain any "
            f"`step N/M loss=...` lines yet — dashboard will show a "
            f"waiting-state until training data appears.",
            file=sys.stderr,
        )
    print(f"[tvtop] tailing {log}")
    TVTop(
        log_path=log,
        color=color,
        ascii_only=ascii_only,
        refresh_seconds=refresh,
        small_mode=small_mode,
    ).run()
    return 0


__all__ = [
    "Frame",
    "LogTail",
    "TVTop",
    "cli_main",
    "sparkline",
]
