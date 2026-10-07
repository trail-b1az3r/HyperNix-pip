"""What the runner has loaded, so ``hnx-t1 runner auto`` can load it again.

Every successful load is appended to ``<T1 config dir>/runner-history.json``:
the model, the backend asked for and the one it landed on, and the
settings it was loaded with. :func:`choose` reads that back:

* **the model** -- the last one loaded, if it is still on this server;
  otherwise (or with ``prefer="most"``) the one loaded most often that
  still is;
* **the backend** -- the one asked for most often, across every load,
  with the most recent winning a tie: the machine's habit, not one
  load's accident;
* **the settings** -- GPU layers, context length and layer count from
  the last time *that model* was loaded, so a context someone tuned is
  kept.

Only what was asked for is replayed. A ``gpu_layers`` of ``None`` meant
"work it out", and stays that: replaying the number the planner chose
last time would pin a split computed for whatever else was in VRAM then.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["AutoChoice", "LoadRecord", "MAX_RECORDS", "choose", "history_path", "read", "record"]

HISTORY_FILE = "runner-history.json"
#: Enough to know a habit; small enough to read on every `auto`.
MAX_RECORDS = 200

_lock = threading.Lock()


@dataclass
class LoadRecord:
    model_id: str
    backend: str = "auto"
    #: Where it actually ran (the placement's backend), for the record.
    resolved_backend: str = ""
    gpu_layers: int | None = None
    context_length: int | None = None
    total_layers: int | None = None
    at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoadRecord:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class AutoChoice:
    model_id: str
    backend: str
    gpu_layers: int | None
    context_length: int | None
    total_layers: int | None
    why: str

    def payload(self) -> dict[str, Any]:
        """The ``/runner/load`` body this choice amounts to."""
        return {"model_id": self.model_id, "backend": self.backend,
                "gpu_layers": self.gpu_layers, "context_length": self.context_length,
                "total_layers": self.total_layers}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def history_path(config_dir: str | Path | None = None) -> Path:
    base = (config_dir or os.environ.get("T1_CONFIG_DIR")
            or Path.home() / ".hypernix" / "t1api")
    return Path(base).expanduser() / HISTORY_FILE


def read(path: str | Path | None = None) -> list[LoadRecord]:
    """Every remembered load, oldest first. A damaged file is no history."""
    target = Path(path) if path else history_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = data.get("loads") if isinstance(data, dict) else None
    out = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("model_id"):
            try:
                out.append(LoadRecord.from_dict(row))
            except TypeError:
                continue
    return out


def record(entry: LoadRecord, path: str | Path | None = None) -> None:
    """Append *entry*, keeping the newest :data:`MAX_RECORDS`.

    Written to a temporary file and renamed into place, so a server
    killed mid-write leaves the old history, not half of a new one.
    """
    target = Path(path) if path else history_path()
    with _lock:
        rows = read(target)
        rows.append(entry)
        rows = rows[-MAX_RECORDS:]
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".runner-history-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"loads": [asdict(r) for r in rows]}, handle, indent=1)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def choose(records: list[LoadRecord], *, loadable: set[str] | None = None,
           prefer: str = "last") -> AutoChoice | None:
    """What ``auto`` loads, or ``None`` when there is nothing to go on.

    *loadable* is the model ids this server can load now; a model that
    has since been deleted or unlinked is passed over rather than tried.
    """
    if prefer not in ("last", "most"):
        raise ValueError("prefer is 'last' or 'most'")
    usable = [r for r in records if loadable is None or r.model_id in loadable]
    if not usable:
        return None

    last = usable[-1]
    counts = Counter(r.model_id for r in usable)
    # Most loaded, the more recent winning a tie.
    order = {r.model_id: i for i, r in enumerate(usable)}
    most = max(counts, key=lambda m: (counts[m], order[m]))
    if prefer == "most":
        model_id = most
        why = f"{model_id}: loaded {counts[model_id]} time(s), more than any other"
    else:
        model_id = last.model_id
        why = f"{model_id}: the model loaded last"
        if loadable is not None and records and records[-1].model_id != model_id:
            why += f" that is still here ({records[-1].model_id} is not)"

    backends = Counter(r.backend or "auto" for r in records)
    border = {r.backend or "auto": i for i, r in enumerate(records)}
    backend = max(backends, key=lambda b: (backends[b], border[b]))

    settings = next(r for r in reversed(usable) if r.model_id == model_id)
    why += (f"; backend {backend} (used for {backends[backend]} of {len(records)} loads); "
            f"settings from its last load")
    return AutoChoice(model_id=model_id, backend=backend, gpu_layers=settings.gpu_layers,
                      context_length=settings.context_length,
                      total_layers=settings.total_layers, why=why)
