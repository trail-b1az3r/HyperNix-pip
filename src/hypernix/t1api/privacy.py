"""t1api.privacy — conceal mode and 36-hour retention (0.72.6).

A key can ask this server to *conceal* it (``waiter serv -c``, or
``POST /privacy/conceal``). Two things follow, for as long as it is on:

**Its address is not kept.** Every audit record the key produces stores
a masked address (``203.0.113.0/24``, ``2001:db8:1::/48``) instead of
the real one. Security records — a request refused by the block list, a
bad key — keep the full address, because they are what the operator
needs to block an abuser, and a concealed key cannot be a way to hide
from that. Nothing else this server stores or shows carries a client's
address, so the audit log is the one place to fix.

What conceal cannot do, said plainly: every server sees the address a
connection comes from. Conceal stops this server *keeping* or *showing*
it. It is not a proxy and does not hide anything from the network.

**Its data is kept for 36 hours.** A sweep deletes what the key made
more than :data:`RETENTION_HOURS` ago: HyperLink chat messages and the
sessions they empty, uploaded files, finished jobs, and non-security
audit records. It does not delete:

* **memories** and **preferences** — "the profile": what the key asked
  the server to remember is not chat history, and deleting it would be
  the server forgetting something it was told to keep;
* **usage counts** — token numbers with no content, and the quota is
  computed from them. Deleting them would reset the key's allowance
  every 36 hours;
* **security audit records**, for the reason above.

Conceal needs an access level of 3 or more: a T2 or T2C key at level 3+,
or a T1 key (which predates levels and clears every level check — see
``AuthContext.meets_access_level``).
"""
from __future__ import annotations

import ipaddress
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from .db import SQLBackend

logger = logging.getLogger(__name__)

__all__ = [
    "CONCEAL_MIN_ACCESS_LEVEL",
    "RETENTION_HOURS",
    "ConcealStore",
    "SweepResult",
    "RetentionSweeper",
    "mask_address",
]

#: How long a concealed key's data is kept.
RETENTION_HOURS = 36

#: The access level a key needs to turn conceal on.
CONCEAL_MIN_ACCESS_LEVEL = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS t1_conceal (
    key_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    since REAL NOT NULL,
    last_swept REAL NOT NULL DEFAULT 0
);
"""


def mask_address(address: str) -> str:
    """The network an address is in, not the address: /24 or /48."""
    try:
        ip = ipaddress.ip_address((address or "").strip())
    except ValueError:
        return "concealed" if address else ""
    prefix = 24 if ip.version == 4 else 48
    return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))


class ConcealStore:
    """Which keys asked to be concealed. One row per key."""

    def __init__(self, backend: SQLBackend) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._cache: dict[str, bool] = {}
        self.backend.executescript(_SCHEMA)

    def set(self, key_id: str, enabled: bool) -> dict[str, Any]:
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            if enabled:
                existing = conn.execute(
                    "SELECT since, enabled FROM t1_conceal WHERE key_id = ?", (key_id,)
                ).fetchone()
                since = float(existing["since"]) if existing and existing["enabled"] else now
                conn.execute("DELETE FROM t1_conceal WHERE key_id = ?", (key_id,))
                conn.execute(
                    "INSERT INTO t1_conceal (key_id, enabled, since, last_swept) VALUES (?, 1, ?, 0)",
                    (key_id, since),
                )
            else:
                conn.execute("DELETE FROM t1_conceal WHERE key_id = ?", (key_id,))
            self._cache[key_id] = enabled
        return self.status(key_id)

    def is_concealed(self, key_id: str) -> bool:
        if not key_id:
            return False
        cached = self._cache.get(key_id)
        if cached is not None:
            return cached
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT enabled FROM t1_conceal WHERE key_id = ?", (key_id,)
            ).fetchone()
        value = bool(row and row["enabled"])
        self._cache[key_id] = value
        return value

    def status(self, key_id: str) -> dict[str, Any]:
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT since, last_swept FROM t1_conceal WHERE key_id = ? AND enabled = 1",
                (key_id,),
            ).fetchone()
        if row is None:
            return {"concealed": False, "retention_hours": None, "since": None, "last_swept": None}
        return {
            "concealed": True,
            "retention_hours": RETENTION_HOURS,
            "since": float(row["since"]),
            "last_swept": float(row["last_swept"]) or None,
        }

    def concealed_keys(self) -> list[str]:
        with self.backend.connect() as conn:
            rows = conn.execute("SELECT key_id FROM t1_conceal WHERE enabled = 1").fetchall()
        return [str(r["key_id"]) for r in rows]

    def mark_swept(self, key_id: str, when: float) -> None:
        with self._lock, self.backend.connect() as conn:
            conn.execute("UPDATE t1_conceal SET last_swept = ? WHERE key_id = ?", (when, key_id))


@dataclass
class SweepResult:
    key_id: str
    messages: int = 0
    sessions: int = 0
    files: int = 0
    jobs: int = 0
    audit_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages, "sessions": self.sessions, "files": self.files,
            "jobs": self.jobs, "audit_records": self.audit_records,
        }


class RetentionSweeper:
    """Deletes what concealed keys made more than 36 hours ago."""

    #: How often the background thread sweeps.
    INTERVAL_SECONDS = 600.0

    def __init__(self, backend: SQLBackend, conceal: ConcealStore, *, attachments: Any = None) -> None:
        self.backend = backend
        self.conceal = conceal
        self.attachments = attachments
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sweep_key(self, key_id: str, *, now: float | None = None) -> SweepResult:
        now = time.time() if now is None else now
        cutoff = now - RETENTION_HOURS * 3600
        result = SweepResult(key_id=key_id)

        # Files first, through the store, so a blob no record needs any
        # more is removed from disk too.
        if self.attachments is not None:
            with self.backend.connect() as conn:
                rows = conn.execute(
                    "SELECT file_id FROM hyperlink_files WHERE owner = ? AND created_at < ?",
                    (key_id, cutoff),
                ).fetchall()
            for row in rows:
                try:
                    self.attachments.delete(str(row["file_id"]), owner=key_id)
                    result.files += 1
                except Exception:  # noqa: BLE001 - one bad record must not stop the sweep
                    logger.warning("privacy: could not delete file %s", row["file_id"], exc_info=True)

        from .audit import mask_identifier

        with self.backend.connect() as conn:
            cursor = conn.execute(
                """DELETE FROM hyperlink_messages
                   WHERE created_at < ? AND session_id IN
                     (SELECT session_id FROM hyperlink_sessions WHERE owner = ?)""",
                (cutoff, key_id),
            )
            result.messages = max(cursor.rowcount or 0, 0)
            cursor = conn.execute(
                """DELETE FROM hyperlink_sessions
                   WHERE owner = ? AND updated_at < ? AND session_id NOT IN
                     (SELECT DISTINCT session_id FROM hyperlink_messages)""",
                (key_id, cutoff),
            )
            result.sessions = max(cursor.rowcount or 0, 0)
            cursor = conn.execute(
                """DELETE FROM jobs WHERE created_by = ? AND created_at < ?
                   AND status IN ('succeeded', 'failed', 'cancelled')""",
                (key_id, cutoff),
            )
            result.jobs = max(cursor.rowcount or 0, 0)
            # Audit records hold the key id masked to its first eight
            # characters (audit.mask_identifier), so that is what matches.
            cursor = conn.execute(
                """DELETE FROM audit_events WHERE actor_key_id = ?
                   AND ts < ? AND category != 'security'""",
                (mask_identifier(key_id), cutoff),
            )
            result.audit_records = max(cursor.rowcount or 0, 0)
        self.conceal.mark_swept(key_id, now)
        return result

    def sweep(self, *, now: float | None = None) -> list[SweepResult]:
        results = []
        for key_id in self.conceal.concealed_keys():
            try:
                results.append(self.sweep_key(key_id, now=now))
            except Exception:  # noqa: BLE001 - keep sweeping the others
                logger.warning("privacy: sweep failed for a concealed key", exc_info=True)
        return results

    # -- background ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="t1-retention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep()
            except Exception:  # noqa: BLE001 - never let the thread die
                logger.warning("privacy: retention sweep failed", exc_info=True)
            self._stop.wait(self.INTERVAL_SECONDS)
