"""strife.placement — what this machine has, and where the model will run.

The plan states every number separately and never rounds a partial
placement up to "on the GPU":

* host RAM and how much the model's weights, the decode cache and the
  KV caches can take of it;
* each GPU found, its VRAM and compute capability;
* how many layers run where, and *why* (a request the hardware or this
  build cannot honour is reported as a fallback reason, not ignored).

**This alpha computes on the CPU.** GPU offload is the next step of
Strife; until it lands, ``gpu_layers`` above 0 is accepted and answered
with ``gpu_layers_used: 0`` and the reason. Platform scope is stated by
:func:`platform_support`: Linux x86_64 and Windows are the alpha's
targets; macOS is experimental on Apple M5 or newer only.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess  # sysctl, for the Apple chip name
from typing import Any

from .inspect import ModelReport

__all__ = ["detect", "plan", "platform_support"]


def _ram() -> dict[str, int]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_bytes": int(vm.total), "available_bytes": int(vm.available)}
    except Exception:
        try:
            pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES")
            return {"total_bytes": pages * size, "available_bytes": avail * size}
        except (ValueError, OSError, AttributeError):
            return {"total_bytes": 0, "available_bytes": 0}


def _gpus() -> list[dict[str, Any]]:
    """Every GPU :mod:`hypernix.system.gpus` finds, NVIDIA and AMD alike."""
    try:
        from ..system import gpus

        cards = gpus.detect()
    except Exception:
        return []
    return [{"index": c.index, "name": c.name, "vendor": str(getattr(c.vendor, "value", c.vendor)),
             "vram_bytes": int(c.memory_total_mb or 0) * 1024 * 1024,
             "compute_capability": c.compute} for c in cards]


def _apple_chip() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def platform_support(system: str | None = None, machine: str | None = None,
                     chip: str | None = None) -> dict[str, Any]:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return {"tier": "alpha", "note": "Linux x86_64: an alpha target."}
    if system == "Windows":
        return {"tier": "alpha", "note": "Windows: an alpha target."}
    if system == "Darwin":
        chip = chip if chip is not None else _apple_chip()
        match = re.search(r"Apple M(\d+)", chip or "")
        if match and int(match.group(1)) >= 5:
            return {"tier": "experimental", "note": f"{chip}: experimental in this alpha."}
        return {"tier": "unsupported",
                "note": f"{chip or 'This Mac'}: Strife alpha supports Apple M5 or newer only "
                        "(experimentally). It may run on the CPU path, but it is not tested or "
                        "supported here."}
    return {"tier": "untested", "note": f"{system} {machine}: not an alpha target; the CPU "
                                        "path may work but is untested."}


def detect() -> dict[str, Any]:
    return {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "python": platform.python_version(), "cpu_count": os.cpu_count() or 1,
            "cpu": platform.processor() or platform.machine(), "ram": _ram(),
            "gpus": _gpus(), "support": platform_support()}


def plan(report: ModelReport, *, n_ctx: int, gpu_layers: int = 0, max_sessions: int = 1,
         hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    hw = hardware or detect()
    from .model import Hyper, KVCache

    hp = Hyper.from_report(report)
    kv_per_session = KVCache.nbytes_for(hp, n_ctx)
    reasons: list[str] = []
    if gpu_layers:
        if not hw["gpus"]:
            reasons.append("no GPU was found, so every layer runs on the CPU")
        else:
            reasons.append("GPU execution is not in Strife alpha 1 yet; every layer runs on the "
                           "CPU (the llama.cpp runner offloads today)")
    avail = int(hw["ram"].get("available_bytes") or 0)
    if avail and report.weight_bytes > avail:
        reasons.append(f"the weights ({report.weight_bytes / 2**30:.1f} GiB) are larger than "
                       f"free RAM ({avail / 2**30:.1f} GiB); they stay memory-mapped and are "
                       "read from disk as needed, which is slow")
    return {
        "backend": "cpu-numpy",
        "layers_total": hp.n_layers, "gpu_layers_requested": int(gpu_layers),
        "gpu_layers_used": 0, "cpu_layers": hp.n_layers,
        "weights_mapped_bytes": report.weight_bytes,
        "kv_cache_bytes_per_session": kv_per_session,
        "kv_cache_bytes_max": kv_per_session * max(1, max_sessions),
        "peak_vram_bytes": 0,
        "fallback_reasons": reasons, "hardware": hw,
    }
