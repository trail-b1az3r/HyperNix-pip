"""t1api.authhistory — undo and redo for authentication changes.

``POST /t1/auth/undo`` exists because the operations it reverses are the
ones people get wrong at 2am: rotating a key that a running service still
holds, promoting the wrong key to admin, revoking the key you are
currently authenticated with. Every one of those is recoverable in
principle — the previous state was known a second ago — and irrecoverable
in practice unless something wrote it down.

This writes it down.

What is and is not undoable
---------------------------
An entry records the *inverse* operation, not a snapshot. Rotation is
undoable because the old key material is still known at the moment of
rotation; a scope change is undoable because the old scope set is;
deletion of a key is undoable only if the key material was captured, and
:meth:`AuthHistory.record` refuses an entry it could not actually
reverse rather than accepting one and failing later. An undo stack that
lies about what it can restore is worse than no undo stack.

Redo is the same mechanism run forwards. Recording a *new* operation
clears the redo stack — the standard editor semantics, and for the
standard reason: once history diverges, replaying the old future
produces a state nobody asked for.

Bounded by construction
-----------------------
Auth history is security-sensitive: it contains, by necessity, the key
material needed to reverse a rotation. So it is bounded in both size
(``max_entries``) and age (``ttl_seconds``), encrypted at rest through
Keymaster's Fernet when the ``security`` extra is present, and never
returned by any endpoint that lists it — :meth:`AuthHistory.describe`
returns what happened, never the material.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .db import SQLiteBackend
from .errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

__all__ = ["AuthOp", "AuthHistoryEntry", "AuthHistory"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS t1_auth_history (
    entry_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    op TEXT NOT NULL,
    actor TEXT NOT NULL,
    target_key_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    undone_at REAL,
    payload TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_t1_auth_history_seq ON t1_auth_history (seq);
"""


class AuthOp(StrEnum):
    """The authentication operations that can be reversed."""

    ROTATE = "rotate"                 # inverse: restore the previous key material
    PROMOTE = "promote"               # inverse: restore the previous key type
    SCOPE_CHANGE = "scope_change"     # inverse: restore the previous scope set
    REVOKE = "revoke"                 # inverse: un-revoke
    ASSIGN_SSPKID = "assign_sspkid"   # inverse: restore the previous SSPKID (or none)
    PLAN_CHANGE = "plan_change"       # inverse: restore the previous plan


#: Which fields an entry must carry to be reversible. An entry missing
#: one of these is refused at record time — see the module docstring.
_REQUIRED_PAYLOAD: dict[AuthOp, tuple[str, ...]] = {
    AuthOp.ROTATE: ("previous_key", "new_key"),
    AuthOp.PROMOTE: ("previous_type", "new_type"),
    AuthOp.SCOPE_CHANGE: ("previous_scopes", "new_scopes"),
    AuthOp.REVOKE: ("previous_revoked",),
    AuthOp.ASSIGN_SSPKID: ("previous_sspkid", "new_sspkid"),
    AuthOp.PLAN_CHANGE: ("previous_plan", "new_plan"),
}

#: Payload keys that are key material. Never returned by describe().
_SECRET_KEYS = frozenset({"previous_key", "new_key"})


@dataclass
class AuthHistoryEntry:
    entry_id: str
    seq: int
    op: AuthOp
    actor: str
    target_key_id: str
    created_at: float
    summary: str
    payload: dict[str, Any] = field(default_factory=dict, repr=False)
    undone_at: float | None = None

    @property
    def undone(self) -> bool:
        return self.undone_at is not None

    def describe(self) -> dict[str, Any]:
        """What happened, with the key material removed.

        Not "redacted with a placeholder" — the keys are simply absent,
        because a field named ``previous_key`` whose value is ``"***"``
        still tells a reader that a rotation is reversible from here.
        """
        return {
            "entry_id": self.entry_id,
            "seq": self.seq,
            "op": self.op.value,
            "actor": self.actor,
            "target_key_id": self.target_key_id,
            "created_at": self.created_at,
            "undone": self.undone,
            "undone_at": self.undone_at,
            "summary": self.summary,
            "details": {k: v for k, v in self.payload.items() if k not in _SECRET_KEYS},
        }


class AuthHistory:
    """A bounded, reversible log of authentication changes."""

    def __init__(
        self,
        backend: SQLiteBackend | None = None,
        *,
        max_entries: int = 200,
        ttl_seconds: float = 7 * 24 * 3600,
        fernet: Any = None,
    ) -> None:
        self.backend = backend or SQLiteBackend()
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        #: Fernet instance from Keymaster when the ``security`` extra is
        #: installed. Without it payloads are stored as plain JSON and a
        #: warning is logged once — the history is still useful, it is
        #: just only as protected as the database file.
        self._fernet = fernet
        if fernet is None:
            logger.warning(
                "t1api.authhistory: no encryption backend; undo payloads (which include key "
                "material for rotations) are stored as plain JSON. Install hypernix[security]."
            )
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # -- writing ------------------------------------------------------

    def record(
        self,
        op: AuthOp | str,
        *,
        actor: str,
        target_key_id: str,
        payload: dict[str, Any],
        summary: str = "",
    ) -> AuthHistoryEntry:
        """Record a reversible operation, clearing the redo stack.

        Refuses a payload that does not carry what the inverse needs.
        The alternative — accept it now, fail at undo time — turns a
        programming mistake into a 2am surprise, which is precisely the
        situation this module exists for.
        """
        op = AuthOp(op)
        required = _REQUIRED_PAYLOAD[op]
        missing = [k for k in required if k not in payload]
        if missing:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Cannot record a reversible {op.value}: payload is missing "
                f"{', '.join(missing)}. An entry that cannot be undone must not be recorded "
                "as though it can.",
            )

        now = time.time()
        with self._lock, self.backend.connect() as conn:
            # A new operation invalidates the redo stack: once history
            # diverges, replaying the old future produces a state nobody
            # asked for.
            conn.execute("DELETE FROM t1_auth_history WHERE undone_at IS NOT NULL")
            row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS top FROM t1_auth_history").fetchone()
            seq = int(row["top"] or 0) + 1
            entry = AuthHistoryEntry(
                entry_id=uuid.uuid4().hex,
                seq=seq,
                op=op,
                actor=actor,
                target_key_id=target_key_id,
                created_at=now,
                summary=summary or f"{op.value} on {target_key_id}",
                payload=dict(payload),
            )
            conn.execute(
                """INSERT INTO t1_auth_history
                   (entry_id, seq, op, actor, target_key_id, created_at, undone_at, payload, summary)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    entry.entry_id, entry.seq, entry.op.value, entry.actor,
                    entry.target_key_id, entry.created_at,
                    self._encrypt(entry.payload), entry.summary,
                ),
            )
            self._prune(conn, now)
        return entry

    def _prune(self, conn: Any, now: float) -> None:
        """Drop entries past the age or count limit.

        Both limits, not either: a quiet server would keep key material
        for a year under a count limit alone, and a busy one would blow
        past any reasonable size under an age limit alone.
        """
        conn.execute("DELETE FROM t1_auth_history WHERE created_at < ?", (now - self.ttl_seconds,))
        row = conn.execute("SELECT COUNT(*) AS n FROM t1_auth_history").fetchone()
        excess = int(row["n"] or 0) - self.max_entries
        if excess > 0:
            conn.execute(
                "DELETE FROM t1_auth_history WHERE entry_id IN "
                "(SELECT entry_id FROM t1_auth_history ORDER BY seq ASC LIMIT ?)",
                (excess,),
            )

    # -- undo / redo --------------------------------------------------

    def peek_undo(self) -> AuthHistoryEntry | None:
        """The next entry :meth:`undo` would reverse."""
        return self._latest(undone=False, order="DESC")

    def peek_redo(self) -> AuthHistoryEntry | None:
        """The next entry :meth:`redo` would replay."""
        return self._latest(undone=True, order="ASC")

    def _latest(self, *, undone: bool, order: str) -> AuthHistoryEntry | None:
        if order not in ("ASC", "DESC"):
            raise ValueError(f"order must be ASC or DESC, not {order!r}")
        clause = "undone_at IS NOT NULL" if undone else "undone_at IS NULL"
        with self.backend.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM t1_auth_history WHERE {clause} ORDER BY seq {order} LIMIT 1"  # nosec B608 - values are bound parameters; interpolated SQL is fixed or allowlisted
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def undo(self, applier: Any) -> AuthHistoryEntry:
        """Reverse the most recent operation.

        *applier* is called as ``applier(entry, direction="undo")`` and is
        what actually mutates the key store — this class owns the history
        and the ordering, not the key semantics. If the applier raises,
        the entry stays on the undo stack: a failed undo must not look
        like a successful one.
        """
        entry = self.peek_undo()
        if entry is None:
            raise T1APIError(
                T1ErrorCode.NOT_FOUND,
                "Nothing to undo: no reversible authentication change on record.",
            )
        applier(entry, direction="undo")
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE t1_auth_history SET undone_at = ? WHERE entry_id = ?",
                (time.time(), entry.entry_id),
            )
        entry.undone_at = time.time()
        return entry

    def redo(self, applier: Any) -> AuthHistoryEntry:
        """Replay the oldest undone operation."""
        entry = self.peek_redo()
        if entry is None:
            raise T1APIError(
                T1ErrorCode.NOT_FOUND,
                "Nothing to redo: no undone authentication change on record.",
            )
        applier(entry, direction="redo")
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE t1_auth_history SET undone_at = NULL WHERE entry_id = ?",
                (entry.entry_id,),
            )
        entry.undone_at = None
        return entry

    # -- reading ------------------------------------------------------

    def list_entries(self, *, limit: int = 50, include_undone: bool = True) -> list[AuthHistoryEntry]:
        clause = "" if include_undone else "WHERE undone_at IS NULL"
        with self.backend.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM t1_auth_history {clause} ORDER BY seq DESC LIMIT ?", (int(limit),)  # nosec B608 - values are bound parameters; interpolated SQL is fixed or allowlisted
            ).fetchall()
        return [self._from_row(r) for r in rows]

    def describe(self, *, limit: int = 50) -> dict[str, Any]:
        """The stack as an endpoint should return it — no key material."""
        entries = self.list_entries(limit=limit)
        undo_next = self.peek_undo()
        redo_next = self.peek_redo()
        return {
            "entries": [e.describe() for e in entries],
            "count": len(entries),
            "can_undo": undo_next is not None,
            "can_redo": redo_next is not None,
            "next_undo": undo_next.describe() if undo_next else None,
            "next_redo": redo_next.describe() if redo_next else None,
        }

    # -- payload encryption -------------------------------------------

    def _encrypt(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload)
        if self._fernet is None:
            return raw
        return self._fernet.encrypt(raw.encode("utf-8")).decode("ascii")

    def _decrypt(self, stored: str) -> dict[str, Any]:
        if self._fernet is None:
            return json.loads(stored)
        try:
            return json.loads(self._fernet.decrypt(stored.encode("ascii")).decode("utf-8"))
        except Exception:
            # A payload written before encryption was configured, or with
            # a different master key. Readable-as-JSON is the only other
            # possibility; anything else is genuinely lost.
            try:
                return json.loads(stored)
            except json.JSONDecodeError:
                logger.error("t1api.authhistory: an entry's payload could not be decrypted")
                return {}

    def _from_row(self, row: Any) -> AuthHistoryEntry:
        return AuthHistoryEntry(
            entry_id=row["entry_id"],
            seq=int(row["seq"]),
            op=AuthOp(row["op"]),
            actor=row["actor"],
            target_key_id=row["target_key_id"],
            created_at=row["created_at"],
            undone_at=row["undone_at"],
            payload=self._decrypt(row["payload"]),
            summary=row["summary"],
        )
