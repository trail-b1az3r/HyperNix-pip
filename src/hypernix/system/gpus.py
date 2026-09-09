"""One way to ask about a GPU, whoever made it.

Before this there were 42 places that shelled out to ``nvidia-smi`` and
12 that knew about ``rocm-smi``, spread over seven modules. The count is
the whole problem: AMD support was not missing so much as *unevenly
present*, and every new panel or check had to reimplement the same
parsing and get the same edge cases wrong again.

So this is the layer everything else asks. It reports what is there,
from whichever vendor tool answers, in one shape:

    from hypernix.system import gpus
    for card in gpus.detect():
        print(card.vendor, card.name, card.memory_total_mb)

What it promises
----------------
**It never raises for want of hardware.** No GPU, no driver, no vendor
tool, a tool that errors, a tool that prints something unexpected — all
of them mean "no cards found", because a monitoring panel that crashes
on a laptop is worse than one that says the laptop has no GPU.

**Every field is optional.** ``rocm-smi`` reports power on some cards
and not others; ``nvidia-smi`` reports ``[N/A]`` for several fields on
consumer cards under WSL. A missing reading is ``None``, never 0 — a
temperature of zero and an unknown temperature are different facts and
a dashboard that conflates them tells you the card is freezing.

**Vendor tools are asked in a fixed order** and every one that answers
contributes, so a machine with an NVIDIA card and an AMD card lists
both rather than whichever was looked for first.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

__all__ = [
    "GPU",
    "GPUProcess",
    "Vendor",
    "detect",
    "backend",
    "processes",
    "select",
    "describe",
]

#: How long a vendor tool gets. These shell out on a monitoring tick, so
#: a hung driver must not hang the dashboard.
_TIMEOUT = 6.0


class Vendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"

    @property
    def framework(self) -> str:
        """The compute stack this vendor's cards use here."""
        return {
            "nvidia": "cuda",
            "amd": "rocm",
            "intel": "xpu",
            "apple": "mps",
        }.get(self.value, "cpu")


@dataclass
class GPU:
    """One card. Every measurement is optional; see the module docstring."""

    index: int
    vendor: Vendor
    name: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    utilization_pct: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    driver: str = ""
    uuid: str = ""
    #: Compute capability for NVIDIA, gfx target for AMD.
    compute: str = ""

    @property
    def memory_free_mb(self) -> int | None:
        if self.memory_total_mb is None or self.memory_used_mb is None:
            return None
        return max(0, self.memory_total_mb - self.memory_used_mb)

    @property
    def framework(self) -> str:
        return self.vendor.framework

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["vendor"] = self.vendor.value
        payload["framework"] = self.framework
        payload["memory_free_mb"] = self.memory_free_mb
        return payload


@dataclass
class GPUProcess:
    pid: int
    name: str
    gpu_index: int
    memory_mb: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Tool:
    """A vendor command and what it is for."""

    name: str
    vendor: Vendor
    paths: tuple[str, ...] = field(default_factory=tuple)

    def resolve(self) -> str:
        found = shutil.which(self.name)
        if found:
            return found
        for candidate in self.paths:
            if os.path.exists(candidate):
                return candidate
        return ""


_NVIDIA = _Tool("nvidia-smi", Vendor.NVIDIA, (
    "/usr/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "C:/Windows/System32/nvidia-smi.exe",
))
#: amd-smi first: it supersedes rocm-smi on ROCm 6+, and where both are
#: present the newer one is the one AMD keeps accurate.
_AMD_SMI = _Tool("amd-smi", Vendor.AMD, ("/opt/rocm/bin/amd-smi",))
_ROCM_SMI = _Tool("rocm-smi", Vendor.AMD, (
    "/opt/rocm/bin/rocm-smi", "/usr/bin/rocm-smi",
))


def _run(argv: list[str], *, timeout: float = _TIMEOUT) -> str:
    """Run a vendor tool and return stdout, or "" for any failure.

    Every way this can go wrong means the same thing to a caller — no
    readings — so they are collapsed here rather than in five places.
    """
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("gpus: %s failed: %s", argv[0], exc)
        return ""
    if result.returncode != 0:
        logger.debug("gpus: %s exited %s", argv[0], result.returncode)
        return ""
    return result.stdout


def _number(text: str) -> float | None:
    """A reading, or None for every way a vendor spells "no value".

    ``[N/A]``, ``N/A``, ``Not Supported``, ``unknown`` and an empty cell
    all appear in real output, and each has to become None rather than
    0: a dashboard that reports an unknown temperature as zero says the
    card is freezing.
    """
    cleaned = text.strip().strip("%").replace("[", "").replace("]", "").strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in ("n/a", "na", "unknown", "not supported", "none", "-"):
        return None
    # Strip a trailing unit: "35.0C", "120.5W", "8192 MiB".
    for suffix in ("mib", "mb", "gib", "gb", "c", "w", "%"):
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break
    try:
        return float(cleaned)
    except ValueError:
        return None


def _int(text: str) -> int | None:
    value = _number(text)
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------

_NVIDIA_FIELDS = (
    "index", "name", "memory.total", "memory.used", "utilization.gpu",
    "temperature.gpu", "power.draw", "power.limit", "driver_version",
    "uuid", "compute_cap",
)


def parse_nvidia(text: str) -> list[GPU]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``."""
    cards: list[GPU] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        # Pad rather than refuse: an older driver answers a shorter row
        # for the same query, and the fields it does return are still
        # worth having.
        parts += [""] * (len(_NVIDIA_FIELDS) - len(parts))
        index = _int(parts[0])
        cards.append(GPU(
            index=index if index is not None else len(cards),
            vendor=Vendor.NVIDIA,
            name=parts[1] or "NVIDIA GPU",
            memory_total_mb=_int(parts[2]),
            memory_used_mb=_int(parts[3]),
            utilization_pct=_number(parts[4]),
            temperature_c=_number(parts[5]),
            power_w=_number(parts[6]),
            power_limit_w=_number(parts[7]),
            driver=parts[8],
            uuid=parts[9],
            compute=parts[10],
        ))
    return cards


def _detect_nvidia() -> list[GPU]:
    binary = _NVIDIA.resolve()
    if not binary:
        return []
    return parse_nvidia(_run([
        binary, f"--query-gpu={','.join(_NVIDIA_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]))


# ---------------------------------------------------------------------------
# AMD
# ---------------------------------------------------------------------------


def parse_rocm_json(text: str) -> list[GPU]:
    """Parse ``rocm-smi --json`` output.

    The shape is ``{"card0": {"Field": "value", ...}, ...}`` and the
    field names have changed across ROCm releases, so each value is
    looked up through a list of the spellings that have been used rather
    than one exact key. A renamed field should cost that reading, not
    the whole card.
    """
    try:
        payload = json.loads(text)
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []

    def pick(entry: dict, *names: str) -> str:
        for name in names:
            for key, value in entry.items():
                if key.lower().replace(" ", "") == name.lower().replace(" ", ""):
                    return str(value)
        return ""

    cards: list[GPU] = []
    for key in sorted(payload):
        if not key.lower().startswith("card"):
            continue
        entry = payload[key]
        if not isinstance(entry, dict):
            continue
        digits = "".join(c for c in key if c.isdigit())
        total = _number(pick(entry, "VRAM Total Memory (B)", "VRAM Total Memory"))
        used = _number(pick(entry, "VRAM Total Used Memory (B)", "VRAM Total Used Memory"))
        cards.append(GPU(
            index=int(digits) if digits else len(cards),
            vendor=Vendor.AMD,
            name=pick(entry, "Card Series", "Card Model", "Device Name",
                      "Card SKU") or "AMD GPU",
            # rocm-smi reports VRAM in bytes; every other source here is
            # megabytes, and a dashboard mixing the two is unreadable.
            memory_total_mb=int(total / 1024 / 1024) if total else None,
            memory_used_mb=int(used / 1024 / 1024) if used is not None else None,
            utilization_pct=_number(pick(entry, "GPU use (%)", "GPU Utilization (%)")),
            temperature_c=_number(pick(
                entry, "Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)",
                "Temperature (Sensor memory) (C)",
            )),
            power_w=_number(pick(
                entry, "Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)",
            )),
            power_limit_w=_number(pick(entry, "Max Graphics Package Power (W)")),
            driver=pick(entry, "Driver version"),
            uuid=pick(entry, "Unique ID", "GUID"),
            compute=pick(entry, "GFX Version", "Card gfx"),
        ))
    return cards


def _detect_amd() -> list[GPU]:
    for tool in (_AMD_SMI, _ROCM_SMI):
        binary = tool.resolve()
        if not binary:
            continue
        flags = (
            ["--json", "--showallinfo"] if tool is _ROCM_SMI
            else ["static", "--json"]
        )
        cards = parse_rocm_json(_run([binary, *flags]))
        if cards:
            return cards
        # amd-smi's own shape, when its static output did not parse.
        if tool is _AMD_SMI:
            cards = parse_amd_smi(_run([binary, "metric", "--json"]))
            if cards:
                return cards
    return []


def parse_amd_smi(text: str) -> list[GPU]:
    """Parse ``amd-smi ... --json``, which is a list of GPU objects."""
    try:
        payload = json.loads(text)
    except ValueError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("gpus") or payload.get("gpu") or []
    if not isinstance(payload, list):
        return []

    cards: list[GPU] = []
    for position, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue

        def dig(*path, entry=entry):
            node = entry
            for step in path:
                if not isinstance(node, dict) or step not in node:
                    return ""
                node = node[step]
            if isinstance(node, dict):
                node = node.get("value", "")
            return str(node)

        index = _int(dig("gpu")) if dig("gpu") else position
        cards.append(GPU(
            index=index if index is not None else position,
            vendor=Vendor.AMD,
            name=dig("asic", "market_name") or dig("market_name") or "AMD GPU",
            memory_total_mb=_int(dig("mem_usage", "total_vram")),
            memory_used_mb=_int(dig("mem_usage", "used_vram")),
            utilization_pct=_number(dig("usage", "gfx_activity")),
            temperature_c=_number(dig("temperature", "edge")),
            power_w=_number(dig("power", "socket_power")),
            power_limit_w=_number(dig("power", "power_limit")),
            uuid=dig("asic", "asic_serial"),
            compute=dig("asic", "target_graphics_version"),
        ))
    return cards


# ---------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------


def detect() -> list[GPU]:
    """Every GPU this machine can see, from every vendor that answers."""
    cards: list[GPU] = []
    for probe in (_detect_nvidia, _detect_amd):
        try:
            cards.extend(probe())
        except Exception as exc:  # noqa: BLE001 - detection must not raise
            logger.debug("gpus: %s raised %s", probe.__name__, exc)
    return cards


def backend() -> Vendor:
    """The vendor whose stack this machine would compute on.

    ``Vendor.UNKNOWN`` means CPU-only, which is a real answer rather than
    a failure — most machines running this are CPU-only.
    """
    cards = detect()
    for vendor in (Vendor.NVIDIA, Vendor.AMD, Vendor.INTEL, Vendor.APPLE):
        if any(card.vendor is vendor for card in cards):
            return vendor
    return Vendor.UNKNOWN


def parse_nvidia_processes(text: str) -> list[GPUProcess]:
    found: list[GPUProcess] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid = _int(parts[0])
        if pid is None:
            continue
        found.append(GPUProcess(
            pid=pid, name=parts[1], gpu_index=_int(parts[2]) or 0,
            memory_mb=_int(parts[3]) if len(parts) > 3 else None,
        ))
    return found


def processes() -> list[GPUProcess]:
    """What is currently using a GPU. Empty when nothing can tell us."""
    binary = _NVIDIA.resolve()
    if binary:
        text = _run([
            binary,
            "--query-compute-apps=pid,process_name,gpu_bus_id,used_memory",
            "--format=csv,noheader,nounits",
        ])
        found = parse_nvidia_processes(text)
        if found:
            return found
    return []


def select(spec: str, cards: list[GPU] | None = None) -> list[GPU]:
    """The cards named by *spec*: ``""``/``auto``/``all``, or ``"0,2"``.

    An index that is not present is an error rather than a silent empty
    result: asking for GPU 3 on a two-card machine is a mistake worth
    hearing about, and returning nothing would look like "no GPUs".
    """
    available = detect() if cards is None else cards
    text = (spec or "").strip().lower()
    if text in ("", "auto", "all"):
        return list(available)
    wanted: list[GPU] = []
    for chunk in text.replace(";", ",").split(","):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            index = int(piece)
        except ValueError:
            raise ValueError(
                f"{piece!r} is not a GPU index. Use numbers, or 'all'."
            ) from None
        match = next((c for c in available if c.index == index), None)
        if match is None:
            listing = ", ".join(str(c.index) for c in available) or "none"
            raise ValueError(
                f"No GPU with index {index}. Available: {listing}."
            )
        wanted.append(match)
    return wanted


def describe(cards: list[GPU] | None = None) -> str:
    """A short human report, for `waiter` and the CLI."""
    found = detect() if cards is None else cards
    if not found:
        return (
            "No GPU detected. Nothing here needs one — this is the CPU-only "
            "path, not a failure."
        )
    lines = []
    for card in found:
        bits = [f"{card.vendor.value}:{card.index}", card.name]
        if card.memory_total_mb:
            used = card.memory_used_mb
            bits.append(
                f"{used}/{card.memory_total_mb} MB" if used is not None
                else f"{card.memory_total_mb} MB"
            )
        if card.utilization_pct is not None:
            bits.append(f"{card.utilization_pct:.0f}%")
        if card.temperature_c is not None:
            bits.append(f"{card.temperature_c:.0f}C")
        if card.power_w is not None:
            bits.append(f"{card.power_w:.0f}W")
        lines.append("  " + "  ".join(bits))
    return "\n".join(lines)
