"""ethanol — turn the GPU clock up.

⚠️  **Real overclocking voids warranties, can crash your machine,
and on consumer cards may permanently damage hardware.**  This
module wraps the *standard* vendor tools (``nvidia-smi`` /
``nvidia-settings`` / ``rocm-smi`` / ``intel_gpu_frequency``) and
maps a single integer "level" 0…30 to bounded clock + memory
offsets.  Level 0 resets to defaults; level 30 is the maximum
offset we'll ever apply (and it's still well below typical
manual-overclocker limits).  The helpers refuse to run unless
``confirm=True`` is passed (or the ``HYPERNIX_ETHANOL_CONFIRM=1``
env var is set), so a mistyped script can't accidentally crank
your GPU.

Quick use::

    from hypernix.ethanol import Ethanol
    Ethanol(level=5).apply(confirm=True)

CLI (registered in ``pyproject.toml``)::

    eth 0      # reset to stock
    eth 5      # mild bump
    eth 30     # max-supported offset

The CLI requires the ``HYPERNIX_ETHANOL_CONFIRM=1`` env var to
actually apply; without it, it prints what *would* happen and
exits 0.

The returned :class:`OverclockResult` records what was attempted,
what succeeded, and whatever stderr came back from the vendor
tool — so you have a record even when an offset gets clamped by
the driver.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

#: Hard ceilings — we will NEVER apply more than these regardless
#: of the level requested.  Tuned to stay inside the bounds the
#: stock driver / msi-afterburner stable bands typically allow.
MAX_CORE_OFFSET_MHZ: int = 200      # core clock offset
MAX_MEM_OFFSET_MHZ: int = 1500      # memory clock offset
MAX_POWER_LIMIT_PCT: int = 115      # power limit % of stock

#: Number of valid levels, inclusive.  Level 0 resets to stock.
MAX_LEVEL: int = 30

#: Above this die temperature, :meth:`Ethanol.apply` refuses to raise
#: clocks any further (levels above 10).
THERMAL_ABORT_C: float = 85.0

#: ``(temperature_below, level)``, coldest first — the bands
#: :meth:`Ethanol.auto_level` picks from. A GPU already running warm gets
#: a smaller bump, and one running hot gets none.
_AUTO_LEVEL_BANDS: tuple[tuple[float, int], ...] = (
    (50.0, 20),
    (65.0, 15),
    (75.0, 10),
    (THERMAL_ABORT_C, 5),
)


@dataclass
class OverclockResult:
    level: int
    core_offset_mhz: int
    mem_offset_mhz: int
    power_limit_pct: int
    backend: str
    applied: bool
    notes: str = ""
    stderr: str = ""


def _level_to_offsets(level: int) -> tuple[int, int, int]:
    """Map ``level`` 0..30 to ``(core_mhz, mem_mhz, power_pct)``.

    Linear ramp; level 0 is full stock, level 30 hits the hard
    ceilings declared at module level.  Levels above 30 are
    clamped to 30 (rather than rejected) to keep the helpers
    forgiving — but the CLI rejects out-of-range input.
    """
    if level < 0:
        raise ValueError("level must be >= 0")
    eff = min(level, MAX_LEVEL)
    if eff == 0:
        return (0, 0, 100)
    f = eff / MAX_LEVEL
    core = int(round(MAX_CORE_OFFSET_MHZ * f))
    mem = int(round(MAX_MEM_OFFSET_MHZ * f))
    # Power scales 100 → 115 across the range.
    power = int(round(100 + (MAX_POWER_LIMIT_PCT - 100) * f))
    return (core, mem, power)


def _has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _detect_backend() -> str:
    """Pick the best vendor tool available.  ``"none"`` when no
    overclocker is installed.

    Tool *presence* rather than card detection, and that is right for
    this module: everything here writes, so the question is which tool
    can write, not which card exists. A read-only abstraction cannot
    answer it.

    But presence alone was wrong in one case worth closing. A driver
    package can leave ``nvidia-smi`` on a machine whose card is a
    Radeon -- a dual-GPU box, or a box where the NVIDIA card was
    removed -- and this then picked "nvidia" and wrote clock offsets at
    a device index that is not the AMD card. So when
    :mod:`hypernix.system.gpus` can see the cards, the tool has to match
    one of them. When it cannot see any, presence decides as before:
    detection failing is not a reason to refuse to overclock a card the
    operator knows is there.
    """
    vendors = _visible_vendors()

    def usable(vendor: str) -> bool:
        return not vendors or vendor in vendors

    if usable("nvidia") and _has_binary("nvidia-smi") and _has_binary("nvidia-settings"):
        return "nvidia"
    if usable("nvidia") and _has_binary("nvidia-smi"):
        return "nvidia-smi-only"
    if usable("amd") and _has_binary("rocm-smi"):
        return "rocm"
    if usable("intel") and _has_binary("intel_gpu_frequency"):
        return "intel"
    return "none"


def _visible_vendors() -> set[str]:
    """Vendors with a card the abstraction can actually see.

    Empty when detection found nothing, which reads as "no opinion" at
    every call site rather than as "no GPUs" -- a container with the
    devices passed through but no vendor tool visible is a real case,
    and refusing to overclock there would be worse than trusting the
    operator.
    """
    try:
        from . import gpus as _gpus

        return {card.vendor.value for card in _gpus.detect()}
    except Exception:  # noqa: BLE001 - detection is advisory here
        # Swallowed without a log line on purpose. This module has no
        # logger, an empty set already means "no opinion" at every call
        # site, and adding logging configuration for one advisory
        # message would be the larger change.
        return set()


def _confirmed(confirm: bool) -> bool:
    if confirm:
        return True
    return os.environ.get("HYPERNIX_ETHANOL_CONFIRM") == "1"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@dataclass
class Ethanol:
    """Overclock helper.  Construct with a level, call :meth:`apply`."""

    level: int = 0
    backend: str | None = None
    gpu_index: int = 0
    extra_notes: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = _detect_backend()

    def offsets(self) -> tuple[int, int, int]:
        return _level_to_offsets(self.level)

    def plan(self) -> OverclockResult:
        core, mem, power = self.offsets()
        return OverclockResult(
            level=self.level,
            core_offset_mhz=core,
            mem_offset_mhz=mem,
            power_limit_pct=power,
            backend=self.backend or "none",
            applied=False,
            notes="(plan only)",
        )

    def read_temperature(self) -> float | None:
        """Current GPU temperature in °C, or None if it can't be read.

        None means "unknown", which is deliberately distinct from a
        reading of 0: :meth:`auto_level` refuses to pick a level from an
        unknown temperature, and returning 0.0 for "couldn't read" would
        have made an unreadable GPU look like the coldest possible one.
        """
        if self.backend in ("nvidia", "nvidia-smi-only"):
            res = _run([
                "nvidia-smi", "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits", "-i", str(self.gpu_index),
            ])
            if res.returncode != 0:
                return None
            try:
                return float(res.stdout.strip().splitlines()[0])
            except (ValueError, IndexError):
                return None
        if self.backend == "rocm":
            res = _run(["rocm-smi", "--showtemp", "--csv"])
            if res.returncode != 0:
                return None
            for line in res.stdout.splitlines():
                for cell in line.split(","):
                    try:
                        value = float(cell.strip())
                    except ValueError:
                        continue
                    # Plausible die temperature; skips the card index and
                    # any other small integers in the row.
                    if 10.0 <= value <= 130.0:
                        return value
            return None
        return None

    def auto_level(self) -> tuple[int, str]:
        """Pick a level from the *measured* temperature.

        Returns ``(level, reason)``. This used to be the literal constant
        15 with a comment claiming it auto-detected — the CLI advertised
        "auto-pick safe level based on temperature" and never read a
        temperature at all. With no reading available it returns 0
        (stock), because the honest response to "I can't tell how hot
        this GPU is" is not to overclock it.
        """
        temp = self.read_temperature()
        if temp is None:
            return 0, (
                "could not read GPU temperature — staying at stock. "
                "Pass an explicit level to override."
            )
        for ceiling, level in _AUTO_LEVEL_BANDS:
            if temp < ceiling:
                return level, f"GPU at {temp:.0f}°C -> level {level}"
        return 0, f"GPU at {temp:.0f}°C is too hot to overclock — staying at stock"

    def _check_temperature_safe(self) -> tuple[bool, float, str]:
        """Whether it is safe to apply the *current* level right now."""
        temp = self.read_temperature()
        if temp is None:
            return True, 0.0, "could not read temperature"
        if temp > THERMAL_ABORT_C and self.level > 10:
            return False, temp, (
                f"GPU too hot ({temp:.0f}°C > {THERMAL_ABORT_C:.0f}°C) for level {self.level}"
            )
        return True, temp, f"temperature OK ({temp:.0f}°C)"

    def apply(self, *, confirm: bool = False, auto_throttle: bool = True) -> OverclockResult:
        """Apply the offsets via the detected vendor tool.

        Without ``confirm=True`` (or ``HYPERNIX_ETHANOL_CONFIRM=1``)
        this returns a planned result without touching the GPU.
        """
        plan = self.plan()
        # Report "there is nothing here to drive" *before* the confirm
        # gate. The other order told a user on a machine with no GPU tools
        # to set HYPERNIX_ETHANOL_CONFIRM=1 — advice that cannot help,
        # since confirming only gets them to the same dead end one step
        # later.
        if self.backend == "none":
            plan.notes = (
                "ethanol: no supported overclocking tool found on this machine "
                "(looked for nvidia-settings, nvidia-smi, rocm-smi, "
                "intel_gpu_frequency). Nothing to apply."
            )
            return plan
        if not _confirmed(confirm):
            plan.notes = (
                "ethanol: refusing to apply without confirm=True or "
                "HYPERNIX_ETHANOL_CONFIRM=1; returning plan."
            )
            return plan

        if auto_throttle:
            is_safe, temp, msg = self._check_temperature_safe()
            if not is_safe:
                plan.notes = f"ethanol safety: {msg}. Aborting apply."
                plan.applied = False
                return plan

        if self.backend == "nvidia":
            return self._apply_nvidia(plan)
        if self.backend == "nvidia-smi-only":
            return self._apply_nvidia_smi_only(plan)
        if self.backend == "rocm":
            return self._apply_rocm(plan)
        if self.backend == "intel":
            return self._apply_intel(plan)
        plan.notes = (
            f"ethanol: no supported overclocker found "
            f"(backend={self.backend!r}); install nvidia-settings, "
            "rocm-smi, or intel_gpu_frequency."
        )
        return plan

    def reset(self, *, confirm: bool = False) -> OverclockResult:
        """Convenience: same as ``Ethanol(level=0).apply(...)``."""
        self.level = 0
        return self.apply(confirm=confirm)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _apply_nvidia(self, plan: OverclockResult) -> OverclockResult:
        cmds: list[list[str]] = [
            ["nvidia-settings", "-a",
             f"[gpu:{self.gpu_index}]/GPUGraphicsClockOffsetAllPerformanceLevels={plan.core_offset_mhz}"],
            ["nvidia-settings", "-a",
             f"[gpu:{self.gpu_index}]/GPUMemoryTransferRateOffsetAllPerformanceLevels={plan.mem_offset_mhz}"],
            ["nvidia-smi", "-i", str(self.gpu_index),
             "-pl", str(self._stock_power_watts_or_default())],
        ]
        return self._run_cmds(cmds, plan)

    def _apply_nvidia_smi_only(self, plan: OverclockResult) -> OverclockResult:
        # Without nvidia-settings we can only set power limit cleanly.
        cmds = [[
            "nvidia-smi", "-i", str(self.gpu_index),
            "-pl", str(self._stock_power_watts_or_default()),
        ]]
        return self._run_cmds(cmds, plan)

    def _apply_rocm(self, plan: OverclockResult) -> OverclockResult:
        # Level 0 is documented as "reset to stock", and it has to
        # actually reset. The previous version ran `--setperflevel manual`
        # + `--setsclk 7` unconditionally, so `eth 0` on an AMD card
        # pinned it to manual mode at the *highest* clock index — the
        # exact opposite of the documented behaviour, in the dangerous
        # direction, on the one command a user reaches for when something
        # has gone wrong.
        if plan.level == 0:
            cmds = [
                ["rocm-smi", "--resetclocks"],
                ["rocm-smi", "--setperflevel", "auto"],
                ["rocm-smi", "--setpoweroverdrive", "0"],
            ]
            return self._run_cmds(cmds, plan)
        cmds = [
            ["rocm-smi", "--setperflevel", "manual"],
            ["rocm-smi", "--setsclk", "7"],
            ["rocm-smi", "--setpoweroverdrive", str(plan.power_limit_pct - 100)],
        ]
        return self._run_cmds(cmds, plan)

    def _apply_intel(self, plan: OverclockResult) -> OverclockResult:
        # Same reset problem: `-s +0` is a no-op, not a reset, so level 0
        # left whatever the previous level had set still in place.
        if plan.level == 0:
            cmds = [["intel_gpu_frequency", "-d"]]
            return self._run_cmds(cmds, plan)
        cmds = [["intel_gpu_frequency", "-s", f"+{plan.core_offset_mhz}"]]
        return self._run_cmds(cmds, plan)

    def _run_cmds(
        self, cmds: list[list[str]], plan: OverclockResult,
    ) -> OverclockResult:
        out_notes: list[str] = []
        out_err: list[str] = []
        any_failed = False
        permission_denied = False
        for cmd in cmds:
            res = _run(cmd)
            out_notes.append(f"$ {' '.join(cmd)} -> rc={res.returncode}")
            if res.stderr:
                out_err.append(res.stderr.strip())
                # Detect permission errors to give helpful guidance
                if "permission" in res.stderr.lower() or "not permitted" in res.stderr.lower():
                    permission_denied = True
            if res.returncode != 0:
                any_failed = True
        plan.applied = not any_failed
        plan.notes = "; ".join(out_notes)
        plan.stderr = "\n".join(out_err)
        if permission_denied:
            plan.notes += (
                "\n\nethanol: permission denied — GPU overclocking requires elevated privileges.\n"
                "Try: sudo eth <level> --confirm\n"
                "Or set persistent permissions via nvidia-settings config file."
            )
        return plan

    def _stock_power_watts_or_default(self) -> int:
        """Read default power limit via nvidia-smi; if that fails,
        fall back to 250W (4080-class default).  Then bump by the
        target percent."""
        try:
            res = _run([
                "nvidia-smi", "--query-gpu=power.default_limit",
                "--format=csv,noheader,nounits", "-i", str(self.gpu_index),
            ])
            stock = float(res.stdout.strip())
        except Exception:  # noqa: BLE001
            stock = 250.0
        _core, _mem, power_pct = self.offsets()
        return int(round(stock * power_pct / 100.0))


    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Current GPU state: clocks, power, temperature, utilisation.

        Read-only and unguarded — nothing here changes a setting, so it
        needs no confirmation. It exists because every other entry point
        in this module either plans or applies an overclock, which left
        no way to answer "what is my GPU doing right now", the question
        you actually want before and after touching anything.

        Missing values come back as None rather than being omitted, so a
        caller can tell "this backend does not report it" from "this key
        does not exist".
        """
        info: dict[str, Any] = {
            "backend": self.backend,
            "gpu_index": self.gpu_index,
            "name": None,
            "temperature_c": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "clock_graphics_mhz": None,
            "clock_memory_mhz": None,
            "utilization_pct": None,
        }
        if self.backend in ("nvidia", "nvidia-smi-only"):
            fields = [
                "name", "temperature.gpu", "power.draw", "power.limit",
                "clocks.current.graphics", "clocks.current.memory",
                "utilization.gpu",
            ]
            res = _run([
                "nvidia-smi", f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits", "-i", str(self.gpu_index),
            ])
            if res.returncode == 0 and res.stdout.strip():
                cells = [c.strip() for c in res.stdout.strip().splitlines()[0].split(",")]
                keys = [
                    "name", "temperature_c", "power_draw_w", "power_limit_w",
                    "clock_graphics_mhz", "clock_memory_mhz", "utilization_pct",
                ]
                for key, cell in zip(keys, cells, strict=False):
                    if cell in ("", "[N/A]", "N/A"):
                        continue
                    info[key] = cell if key == "name" else _maybe_float(cell)
            return info
        if self.backend == "rocm":
            info["temperature_c"] = self.read_temperature()
            return info
        return info


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def _maybe_float(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def ethanol(level: int = 0, **kw: Any) -> Ethanol:
    return Ethanol(level=level, **kw)


def overclock(level: int, *, confirm: bool = False, gpu_index: int = 0) -> OverclockResult:
    """One-shot helper.  Equivalent to
    ``Ethanol(level=level, gpu_index=gpu_index).apply(confirm=...)``."""
    return Ethanol(level=level, gpu_index=gpu_index).apply(confirm=confirm)


# ---------------------------------------------------------------------------
# CLI entry point — installed as ``eth``
# ---------------------------------------------------------------------------

_USAGE = """\
usage: eth <level 0..30 | auto | status | reset> [--confirm] [--gpu N]

  eth status     show the GPU's current clocks, power, and temperature
  eth 0          reset to stock
  eth reset      same as `eth 0`
  eth 5          mild bump
  eth 30         max-supported offset
  eth auto       pick a level from the GPU's measured temperature

Applying requires --confirm or HYPERNIX_ETHANOL_CONFIRM=1; without one,
the plan is printed and nothing is touched.

Overclocking can crash your machine and may damage hardware. Level 0 is
always a reset, on every supported backend.
"""


def _print_status(eth: Ethanol) -> int:
    info = eth.status()
    if info["backend"] == "none":
        print(
            "eth: no supported GPU tool found (looked for nvidia-smi, "
            "rocm-smi, intel_gpu_frequency).",
            file=sys.stderr,
        )
        return 1
    print(f"backend  {info['backend']}  (gpu {info['gpu_index']})")
    if info["name"]:
        print(f"gpu      {info['name']}")
    rows = (
        ("temp", info["temperature_c"], "°C"),
        ("power", info["power_draw_w"], "W"),
        ("limit", info["power_limit_w"], "W"),
        ("core", info["clock_graphics_mhz"], "MHz"),
        ("mem", info["clock_memory_mhz"], "MHz"),
        ("util", info["utilization_pct"], "%"),
    )
    for label, value, unit in rows:
        # "not reported" rather than a blank: a missing reading is
        # information, and printing nothing looks like a rendering bug.
        shown = f"{value:g}{unit}" if isinstance(value, (int, float)) else "not reported"
        print(f"{label:<8} {shown}")
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0

    confirm = "--confirm" in args
    gpu = 0
    if "--gpu" in args:
        i = args.index("--gpu")
        if i + 1 >= len(args):
            print("eth: --gpu requires an index", file=sys.stderr)
            return 2
        try:
            gpu = int(args[i + 1])
        except ValueError:
            print(f"eth: --gpu takes an integer, got {args[i + 1]!r}", file=sys.stderr)
            return 2

    command = args[0].lower()

    if command == "status":
        return _print_status(Ethanol(level=0, gpu_index=gpu))

    if command == "auto":
        level, reason = Ethanol(level=0, gpu_index=gpu).auto_level()
        print(f"eth auto: {reason}")
    elif command == "reset":
        level = 0
    else:
        try:
            level = int(args[0])
        except ValueError:
            print(
                f"eth: expected a level 0..{MAX_LEVEL}, 'auto', 'status' or "
                f"'reset', got {args[0]!r}",
                file=sys.stderr,
            )
            return 2
        if level < 0 or level > MAX_LEVEL:
            print(f"eth: level must be in 0..{MAX_LEVEL}", file=sys.stderr)
            return 2

    res = Ethanol(level=level, gpu_index=gpu).apply(confirm=confirm)
    print(
        f"ethanol level={res.level} core+{res.core_offset_mhz} MHz "
        f"mem+{res.mem_offset_mhz} MHz power={res.power_limit_pct}% "
        f"backend={res.backend} applied={res.applied}",
    )
    if res.notes:
        print(res.notes)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    # No backend is a real failure of the requested operation, not a
    # successful no-op — it used to exit 0 and look like it worked.
    if res.backend == "none":
        return 1
    return 0 if res.applied or not _confirmed(confirm) else 1


__all__ = [
    "Ethanol",
    "THERMAL_ABORT_C",
    "MAX_CORE_OFFSET_MHZ",
    "MAX_LEVEL",
    "MAX_MEM_OFFSET_MHZ",
    "MAX_POWER_LIMIT_PCT",
    "OverclockResult",
    "cli_main",
    "ethanol",
    "overclock",
]


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(cli_main())
