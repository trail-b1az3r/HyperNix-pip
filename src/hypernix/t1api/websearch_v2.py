"""Web search v2 (T1 v1.1.26.10.1.0): the grammar, the five admin
settings, and keeping what a search found.

    /web/v2/I?=query&q?=depth&mx?=maxitems&sq?=#&&kpd?=true

``q`` is the depth (as in v1), ``mx`` the most results to return, ``sq``
how many of the top-ranked sites to read and summarise (``sq?=1`` reads
only the first), and ``kpd`` whether to keep the search and the sites'
text. A bare value with no name is the query. ``&&`` separates the
parameters written in the ``?=`` form from a flag after them, and is
read as one separator; the conventional spelling (``?q=cats&depth=2&
max=10&read=1&keep=true``) works too.

The five settings (``hypernix-t1 config web``, or ``/web/v1/ai/config``
for the less risky ones):

``store_enabled``  keep searches at all (AI search storage); off by default
``store_format``   ``jsonl`` or ``parquet``
``max_items``      the most results ``mx`` may ask for
``max_read_sites`` the most sites ``sq`` may read
``ai_config``      who may let a model change settings: ``off``,
                   ``partial`` (partial admins and admins) or ``admin``
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .websearch import DEFAULT_DEPTH, MAX_DEPTH, _clean

__all__ = [
    "AI_SAFE",
    "SearchRequest",
    "WebAdminSettings",
    "keep",
    "load_settings",
    "parse_v2_query",
    "save_settings",
    "settings_path",
]

FORMATS = ("jsonl", "parquet")
AI_CONFIG = ("off", "partial", "admin")
#: What a model may change through /web/v1/ai/config with partial admin.
#: Turning storage on, and who may configure, need a full admin.
AI_SAFE = ("store_format", "max_items", "max_read_sites")


@dataclass
class WebAdminSettings:
    store_enabled: bool = False
    store_format: str = "jsonl"
    max_items: int = 20
    max_read_sites: int = 5
    ai_config: str = "partial"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def set(self, name: str, raw: Any) -> None:
        """Set one setting from text or a value, checked. Raises ValueError."""
        known = {f.name: f for f in fields(self)}
        if name not in known:
            raise ValueError(f"no web setting {name!r}; there are: {', '.join(known)}")
        value = str(raw).strip().lower() if not isinstance(raw, (bool, int)) else raw
        if name == "store_enabled":
            if isinstance(value, bool):
                pass
            elif value in ("1", "true", "yes", "on"):
                value = True
            elif value in ("0", "false", "no", "off"):
                value = False
            else:
                raise ValueError("store_enabled is true or false")
        elif name == "store_format":
            if value not in FORMATS:
                raise ValueError(f"store_format is one of {', '.join(FORMATS)}")
        elif name in ("max_items", "max_read_sites"):
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} is a whole number") from None
            ceiling = 100 if name == "max_items" else 20
            if not 1 <= value <= ceiling:
                raise ValueError(f"{name} is between 1 and {ceiling}")
        elif name == "ai_config" and value not in AI_CONFIG:
            raise ValueError(f"ai_config is one of {', '.join(AI_CONFIG)}")
        setattr(self, name, value)


def settings_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir) if config_dir else Path(
        os.environ.get("T1_CONFIG_DIR") or Path.home() / ".hypernix" / "t1api")
    return base / "web.json"


def load_settings(config_dir: str | Path | None = None) -> WebAdminSettings:
    """The settings on disk, or the defaults. A damaged file is the defaults."""
    try:
        data = json.loads(settings_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return WebAdminSettings()
    current = WebAdminSettings()
    for name, value in (data or {}).items():
        try:
            current.set(name, value)
        except ValueError:
            continue
    return current


def save_settings(settings: WebAdminSettings, config_dir: str | Path | None = None) -> Path:
    path = settings_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    scratch.replace(path)
    return path


@dataclass(frozen=True)
class SearchRequest:
    query: str
    depth: int = DEFAULT_DEPTH
    max_items: int = 10
    read_sites: int = 0
    keep: bool = False


def _truthy(text: str) -> bool:
    return text.strip().lower().strip("[]") in ("1", "true", "yes", "on")


def parse_v2_query(raw: str, settings: WebAdminSettings | None = None) -> SearchRequest:
    """The v2 grammar, capped by the settings."""
    settings = settings or WebAdminSettings()
    query, depth, max_items, read_sites, keep_it = "", DEFAULT_DEPTH, 10, 0, False
    for part in (raw or "").split("&"):
        part = part.strip().rstrip("/")
        if not part:
            continue  # `&&` is one separator
        name, _, value = part.partition("=")
        name = name.strip().rstrip("?").lower()
        value = _clean(value)
        if not name or name in ("i", "query", "search"):
            query = query or value
        elif name in ("q", "depth"):
            if value.isdigit():
                depth = int(value)
            elif not query:
                query = value
        elif name in ("mx", "max", "max_items") and value.isdigit():
            max_items = int(value)
        elif name in ("sq", "read", "read_sites") and value.isdigit():
            read_sites = int(value)
        elif name in ("kpd", "keep"):
            keep_it = _truthy(value)
    return SearchRequest(
        query=query,
        depth=max(1, min(MAX_DEPTH, depth)),
        max_items=max(1, min(settings.max_items, max_items)),
        read_sites=max(0, min(settings.max_read_sites, read_sites)),
        keep=keep_it,
    )


def keep(record: dict[str, Any], *, settings: WebAdminSettings,
         config_dir: str | Path | None = None) -> dict[str, Any]:
    """Append *record* to the day's file in the chosen format.

    Returns where it went, or why it did not. Parquet needs pyarrow; a
    server without it keeps JSONL and says so rather than losing the
    record.
    """
    if not settings.store_enabled:
        return {"kept": False, "why": "search storage is off (hypernix-t1 config web --set store_enabled=true)"}
    folder = settings_path(config_dir).parent / "web-store"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"id": uuid.uuid4().hex, "kept_at": time.time(), **record}
    day = time.strftime("%Y-%m-%d")
    fmt = settings.store_format
    note = ""
    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            fmt, note = "jsonl", "pyarrow is not installed, so this was kept as JSONL"
        else:
            path = folder / f"search-{day}-{record['id'][:8]}.parquet"
            flat = {k: [json.dumps(v) if isinstance(v, (dict, list)) else v] for k, v in record.items()}
            pq.write_table(pa.table(flat), path)
            return {"kept": True, "id": record["id"], "path": str(path), "format": "parquet"}
    path = folder / f"search-{day}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"kept": True, "id": record["id"], "path": str(path), "format": "jsonl", **({"note": note} if note else {})}
