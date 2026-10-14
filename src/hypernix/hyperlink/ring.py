"""The two-layer ring beside the chat box (0.72.8).

The inner ring is the conversation's context: how much of the loaded
model's window the next turn will use. The outer ring is the person's
usage allowance when their key and this server both have one: the
model is registered with a token limit and usage is metered for the key.
Otherwise it is what the machine is doing: the GPU's load when the
server has a GPU, and the CPU's when the model runs on the CPU.

Every number here is a fraction from 0 to 1, with a label the app shows
under the ring, so the clients draw and never compute.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["Layer", "context_layer", "hardware_layer", "usage_layer"]

_HW_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_HW_LOCK = threading.Lock()
#: How long one hardware reading is reused. The app polls every few
#: seconds, and asking every GPU on every poll is what makes a busy
#: server busier.
HW_TTL = 3.0


@dataclass(frozen=True)
class Layer:
    kind: str           # context | usage | gpu | cpu
    fraction: float     # 0..1
    label: str
    used: float = 0.0
    limit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "fraction": round(min(1.0, max(0.0, self.fraction)), 4),
                "label": self.label, "used": self.used, "limit": self.limit}


def _short(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{int(n)}"


def context_layer(used_tokens: int, context_length: int, *, from_model: bool) -> Layer:
    limit = max(1, int(context_length))
    where = "of the model's context" if from_model else "of your context budget"
    return Layer("context", used_tokens / limit,
                 f"{_short(used_tokens)} / {_short(limit)} tokens {where}", used_tokens, limit)


def usage_layer(snapshot: Any) -> Layer | None:
    """The allowance used, or None when the key or the server has none."""
    if snapshot is None:
        return None
    pairs = [(snapshot.input_tokens_used, snapshot.input_token_limit),
             (snapshot.output_tokens_used, snapshot.output_token_limit)]
    pairs = [(used, limit) for used, limit in pairs if limit and limit > 0]
    if not pairs:
        return None
    used, limit = max(pairs, key=lambda p: p[0] / p[1])
    return Layer("usage", used / limit, f"{_short(used)} / {_short(limit)} tokens of your allowance",
                 used, limit)


def _snapshot() -> Any:
    with _HW_LOCK:
        if _HW_CACHE["value"] is not None and time.monotonic() - _HW_CACHE["at"] < HW_TTL:
            return _HW_CACHE["value"]
    from ..system.hardware import snapshot

    value = snapshot(disk_paths=[])
    with _HW_LOCK:
        _HW_CACHE.update(at=time.monotonic(), value=value)
    return value


def hardware_layer(*, on_gpu: bool | None = None, reading: Any = None) -> Layer:
    """GPU load when there is a GPU the model can use, else CPU load.

    *on_gpu* False (the runner put no layers on a GPU) means the CPU is
    what is working, whatever cards the machine has.
    """
    reading = reading if reading is not None else _snapshot()
    gpus = [g for g in (getattr(reading, "gpus", None) or [])
            if getattr(g, "utilization_percent", None) is not None]
    if gpus and on_gpu is not False:
        busiest = max(gpus, key=lambda g: g.utilization_percent)
        name = getattr(busiest, "name", "") or "GPU"
        return Layer("gpu", busiest.utilization_percent / 100.0,
                     f"{name}: {busiest.utilization_percent:.0f}% busy", busiest.utilization_percent, 100)
    cpu = getattr(reading, "cpu", None)
    percent = float(getattr(cpu, "percent", 0.0) or 0.0)
    return Layer("cpu", percent / 100.0, f"CPU: {percent:.0f}% busy", percent, 100)
