"""hyperlink.sync — catching up, and not sending the same thing twice.

A phone is not a desktop client. It loses the route mid-request, it is
suspended by the OS in the middle of a POST, it comes back after four
hours on a different network, and it retries whatever it thinks did not
land. Two failures follow from that, and neither is fixable on the
client alone.

**A retry sends the message twice.** The phone POSTs a turn, the
connection drops before the response arrives, and the phone cannot tell
"the server never saw it" from "the server saw it and the reply was
lost". Retrying is the only safe-looking option, and it produces two
identical user messages and two model replies — one of which cost real
tokens for nothing.

The fix is an idempotency key chosen by the *client*: a
``client_msg_id`` minted before the first attempt and reused on every
retry. :meth:`SyncStore.claim` records it and says whether this is the
first claim. A second claim returns what the first one produced, so the
retry gets the original answer instead of causing a second one. Keys are
scoped per device, so two phones cannot collide, and expire after
:data:`CLAIM_TTL_SECONDS` because an idempotency table that only grows
is a slow leak.

**A phone that was away does not know what it missed.** Polling
``GET /sessions`` tells it the current state of everything, which on a
cellular connection means downloading conversations it already has, and
still does not reveal that a session was *deleted* — an absence is
invisible when you are comparing against a list you no longer trust.

The fix is a change log. Every mutation appends a row with a monotonic
sequence number, deletions included, and a client asks "what has changed
since 412?". It gets a bounded page of changes and a cursor. Deletions
are real rows — tombstones — so the phone can drop a session it still
has, and tombstones are kept for :data:`TOMBSTONE_TTL_SECONDS` so a
device that has been away longer than that is told to resynchronise
rather than silently keeping a session the server has forgotten.

Sequence numbers come from a single counter row, taken inside the same
transaction as the change it labels. That matters: two concurrent writers
allocating sequence numbers separately can commit out of order, and a
client that reads up to 500 while 499 is still uncommitted will never see
499 again. Holding the counter in the transaction makes the log's order
the commit order.

What this module deliberately does not do
-----------------------------------------
It does not merge. There is exactly one writer of record for a
conversation — the machine — and the phone is a client of it, so
"resolve a conflict between two divergent histories" is a problem this
shape does not have. Message content is append-only; session metadata is
last-write-wins with the writer's sequence number available to detect a
stale update. Anything cleverer would be a distributed-systems
liability in exchange for a case that cannot arise.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

__all__ = [
    "CLAIM_TTL_SECONDS",
    "Change",
    "ChangeKind",
    "ClaimResult",
    "MAX_PAGE",
    "SyncPage",
    "SyncStore",
    "TOMBSTONE_TTL_SECONDS",
]

# How long a client's idempotency key is remembered. A retry loop on a
# phone gives up long before this; the window only needs to outlive the
# longest plausible "app suspended mid-POST, resumed later" gap.
CLAIM_TTL_SECONDS = 24 * 3600

# How long a deletion stays visible in the log. A device away for longer
# is told to resynchronise from scratch, because the alternative is
# keeping a tombstone table forever to serve a client that may never
# return.
TOMBSTONE_TTL_SECONDS = 30 * 24 * 3600

# A page cap the caller cannot raise. The point of the change feed is a
# bounded response on a cellular link; an unbounded `limit` would let a
# client ask for the entire history in one request and undo that.
MAX_PAGE = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_change_log (
    seq INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    owner TEXT NOT NULL,
    device_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS hyperlink_change_counter (
    id INTEGER PRIMARY KEY,
    next_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS hyperlink_claims (
    device_id TEXT NOT NULL,
    client_msg_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT '{}',
    settled INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (device_id, client_msg_id)
);
CREATE INDEX IF NOT EXISTS idx_hyperlink_change_owner
    ON hyperlink_change_log (owner, seq);
CREATE INDEX IF NOT EXISTS idx_hyperlink_change_session
    ON hyperlink_change_log (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_hyperlink_claims_age
    ON hyperlink_claims (created_at);
"""


class ChangeKind:
    """What happened to an entity.

    Strings rather than an enum because they cross a JSON boundary to a
    Swift client and back, and a value the client does not recognise has
    to survive the round trip rather than raise.
    """

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"

    ALL = frozenset({CREATED, UPDATED, DELETED})


# The entities a phone mirrors. Anything else has no business in a feed
# whose job is to keep one app's local copy honest.
VALID_ENTITIES = frozenset({"session", "message", "device", "attachment"})


@dataclass
class Change:
    seq: int
    kind: str
    entity: str
    entity_id: str
    owner: str
    created_at: float
    session_id: str = ""
    device_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seq": self.seq,
            "kind": self.kind,
            "entity": self.entity,
            "entity_id": self.entity_id,
            "created_at": self.created_at,
        }
        # Emitted only when set, so a message change does not carry an
        # empty device_id and a device change does not carry an empty
        # session_id. The client's decoder reads absence as "not
        # applicable", which is what it is.
        if self.session_id:
            out["session_id"] = self.session_id
        if self.device_id:
            out["device_id"] = self.device_id
        if self.payload:
            out["payload"] = self.payload
        return out


@dataclass
class SyncPage:
    """One bounded answer to "what changed since N?"."""

    changes: list[Change]
    cursor: int
    more: bool
    resync_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "cursor": self.cursor,
            "more": self.more,
            "resync_required": self.resync_required,
        }


@dataclass
class ClaimResult:
    """The outcome of claiming an idempotency key.

    ``fresh`` is the only bit the caller must branch on: True means do
    the work, False means the work was already done (or is in flight)
    and ``result`` holds whatever the first claim recorded.
    """

    fresh: bool
    client_msg_id: str
    result: dict[str, Any] = field(default_factory=dict)
    settled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fresh": self.fresh,
            "client_msg_id": self.client_msg_id,
            "settled": self.settled,
            "result": self.result,
        }


class SyncStore:
    """The change log and the idempotency table."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # -- the change log -----------------------------------------------

    def record(
        self,
        *,
        kind: str,
        entity: str,
        entity_id: str,
        owner: str,
        session_id: str = "",
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> Change:
        """Append one change and return it, sequence number included."""
        if kind not in ChangeKind.ALL:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Unknown change kind {kind!r}; expected one of "
                f"{sorted(ChangeKind.ALL)}",
            )
        if entity not in VALID_ENTITIES:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Unknown entity {entity!r}; expected one of "
                f"{sorted(VALID_ENTITIES)}",
            )
        if not entity_id:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "entity_id is required")
        if not owner:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "owner is required")

        now = time.time()
        with self._lock, self.backend.connect() as conn:
            seq = self._next_seq(conn)
            conn.execute(
                """INSERT INTO hyperlink_change_log
                   (seq, kind, entity, entity_id, session_id, owner, device_id,
                    created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    seq, kind, entity, entity_id, session_id, owner, device_id,
                    now, json.dumps(payload or {}),
                ),
            )
        return Change(
            seq=seq, kind=kind, entity=entity, entity_id=entity_id, owner=owner,
            created_at=now, session_id=session_id, device_id=device_id,
            payload=payload or {},
        )

    def record_many(self, changes: Iterable[dict[str, Any]]) -> list[Change]:
        """Append several changes, each with its own sequence number.

        One transaction, so a burst of messages written together cannot
        be observed half-applied by a client polling in between.
        """
        pending = list(changes)
        for entry in pending:
            if entry.get("kind") not in ChangeKind.ALL:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    f"Unknown change kind {entry.get('kind')!r}",
                )
            if entry.get("entity") not in VALID_ENTITIES:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    f"Unknown entity {entry.get('entity')!r}",
                )
        now = time.time()
        out: list[Change] = []
        with self._lock, self.backend.connect() as conn:
            for entry in pending:
                seq = self._next_seq(conn)
                change = Change(
                    seq=seq,
                    kind=entry["kind"],
                    entity=entry["entity"],
                    entity_id=entry["entity_id"],
                    owner=entry["owner"],
                    created_at=now,
                    session_id=entry.get("session_id", ""),
                    device_id=entry.get("device_id", ""),
                    payload=entry.get("payload") or {},
                )
                conn.execute(
                    """INSERT INTO hyperlink_change_log
                       (seq, kind, entity, entity_id, session_id, owner, device_id,
                        created_at, payload)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        change.seq, change.kind, change.entity, change.entity_id,
                        change.session_id, change.owner, change.device_id,
                        change.created_at, json.dumps(change.payload),
                    ),
                )
                out.append(change)
        return out

    def since(
        self,
        cursor: int,
        *,
        owner: str,
        limit: int = 100,
        session_id: str | None = None,
        entities: Iterable[str] | None = None,
    ) -> SyncPage:
        """Changes after *cursor*, for one owner, as a bounded page.

        The returned cursor is the last sequence number in the page, so
        a client loops until ``more`` is False. It is never the caller's
        cursor advanced by the page size: a filtered feed skips rows,
        and guessing would skip real changes with them.
        """
        if cursor < 0:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "cursor cannot be negative")
        if not owner:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "owner is required")
        capped = max(1, min(int(limit), MAX_PAGE))

        wanted = set(entities) if entities is not None else None
        if wanted is not None:
            unknown = wanted - VALID_ENTITIES
            if unknown:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    f"Unknown entities {sorted(unknown)}",
                )

        resync = self._resync_required(cursor, owner=owner)

        sql = [
            "SELECT * FROM hyperlink_change_log WHERE owner = ? AND seq > ?"
        ]
        params: list[Any] = [owner, cursor]
        if session_id:
            sql.append("AND session_id = ?")
            params.append(session_id)
        if wanted:
            placeholders = ",".join("?" for _ in wanted)
            sql.append(f"AND entity IN ({placeholders})")
            params.extend(sorted(wanted))
        sql.append("ORDER BY seq ASC LIMIT ?")
        # One more than asked for, so `more` is a fact rather than a
        # guess from a full page -- a page that happens to be exactly
        # `limit` long with nothing after it would otherwise send the
        # client round again for an empty answer.
        params.append(capped + 1)

        with self.backend.connect() as conn:
            rows = conn.execute(" ".join(sql), tuple(params)).fetchall()

        more = len(rows) > capped
        rows = rows[:capped]
        changes = [_change_from_row(r) for r in rows]
        next_cursor = changes[-1].seq if changes else cursor
        return SyncPage(
            changes=changes, cursor=next_cursor, more=more, resync_required=resync,
        )

    def head(self, *, owner: str | None = None) -> int:
        """The highest sequence number issued, or 0 for an empty log.

        A client that has never synchronised uses this to skip the
        history: fetch current state, then start the feed at head. The
        alternative -- starting at 0 -- replays every change ever made
        to build a state it already has.
        """
        sql = "SELECT MAX(seq) AS top FROM hyperlink_change_log"
        params: tuple[Any, ...] = ()
        if owner:
            sql += " WHERE owner = ?"
            params = (owner,)
        with self.backend.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["top"] or 0)

    def _resync_required(self, cursor: int, *, owner: str) -> bool:
        """Has the client's cursor fallen off the back of the log?

        Tombstones expire, so a device whose cursor predates the oldest
        surviving row cannot be brought up to date by replaying changes
        — the deletions it needs to hear about are gone. Saying so is
        the only honest answer; the client refetches state and restarts
        the feed at head.
        """
        if cursor == 0:
            return False
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT MIN(seq) AS oldest FROM hyperlink_change_log WHERE owner = ?",
                (owner,),
            ).fetchone()
        oldest = int(row["oldest"] or 0)
        # An empty log is not a gap: nothing has been forgotten, there
        # was simply never anything to forget.
        if oldest == 0:
            return False
        return cursor < oldest - 1

    def prune(self, *, now: float | None = None) -> dict[str, int]:
        """Drop expired tombstones and settled claims.

        Returns what it removed, so a caller can log or report it. Only
        `deleted` rows expire: a create or update is superseded by
        later state and is safe to keep, while dropping a create would
        make its entity look like it never existed.
        """
        moment = time.time() if now is None else now
        with self._lock, self.backend.connect() as conn:
            cur = conn.execute(
                "DELETE FROM hyperlink_change_log WHERE kind = ? AND created_at < ?",
                (ChangeKind.DELETED, moment - TOMBSTONE_TTL_SECONDS),
            )
            tombstones = int(cur.rowcount or 0)
            cur = conn.execute(
                "DELETE FROM hyperlink_claims WHERE created_at < ?",
                (moment - CLAIM_TTL_SECONDS,),
            )
            claims = int(cur.rowcount or 0)
        return {"tombstones": tombstones, "claims": claims}

    # -- idempotency --------------------------------------------------

    def claim(
        self,
        *,
        device_id: str,
        client_msg_id: str,
        owner: str,
    ) -> ClaimResult:
        """Claim an idempotency key for one device.

        ``fresh=True`` means this caller won the claim and should do the
        work. ``fresh=False`` means an earlier attempt already did, and
        ``result`` carries what it recorded — empty if that attempt is
        still in flight, in which case ``settled`` is False and the
        caller should tell the client to retry rather than duplicate
        the work.
        """
        if not device_id:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "device_id is required")
        if not client_msg_id:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "client_msg_id is required")
        if len(client_msg_id) > 128:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                "client_msg_id must be 128 characters or fewer",
            )
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                """SELECT result, settled, owner, created_at FROM hyperlink_claims
                   WHERE device_id = ? AND client_msg_id = ?""",
                (device_id, client_msg_id),
            ).fetchone()
            if row is not None:
                # A key past its TTL is treated as never seen. The row is
                # replaced rather than left to be pruned later, so a
                # client reusing an old id gets one clear answer instead
                # of depending on when the pruner last ran.
                if now - float(row["created_at"]) > CLAIM_TTL_SECONDS:
                    conn.execute(
                        """DELETE FROM hyperlink_claims
                           WHERE device_id = ? AND client_msg_id = ?""",
                        (device_id, client_msg_id),
                    )
                else:
                    # Keys are scoped per device, and a device belongs to
                    # one owner. A mismatch means the device changed
                    # hands or the caller is guessing; either way this is
                    # not the claim it thinks it is.
                    if row["owner"] != owner:
                        raise T1APIError(
                            T1ErrorCode.CONFLICT,
                            "That client_msg_id belongs to a different owner",
                        )
                    return ClaimResult(
                        fresh=False,
                        client_msg_id=client_msg_id,
                        result=json.loads(row["result"] or "{}"),
                        settled=bool(row["settled"]),
                    )
            conn.execute(
                """INSERT INTO hyperlink_claims
                   (device_id, client_msg_id, owner, result, settled, created_at)
                   VALUES (?, ?, ?, '{}', 0, ?)""",
                (device_id, client_msg_id, owner, now),
            )
        return ClaimResult(fresh=True, client_msg_id=client_msg_id)

    def settle(
        self,
        *,
        device_id: str,
        client_msg_id: str,
        result: dict[str, Any],
    ) -> None:
        """Record what a claim produced, so a retry can be answered."""
        with self._lock, self.backend.connect() as conn:
            cur = conn.execute(
                """UPDATE hyperlink_claims SET result = ?, settled = 1
                   WHERE device_id = ? AND client_msg_id = ?""",
                (json.dumps(result), device_id, client_msg_id),
            )
            if not cur.rowcount:
                raise T1APIError(
                    T1ErrorCode.NOT_FOUND,
                    f"No claim {client_msg_id!r} for device {device_id!r}",
                )

    def release(self, *, device_id: str, client_msg_id: str) -> None:
        """Give up a claim that failed, so the client may try again.

        Without this a request that errored would hold its key for the
        full TTL and every retry would be answered "already done" with
        an empty result — the work permanently unfinished and the
        client unable to ask for it again.
        """
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """DELETE FROM hyperlink_claims
                   WHERE device_id = ? AND client_msg_id = ? AND settled = 0""",
                (device_id, client_msg_id),
            )

    def new_client_msg_id(self) -> str:
        """A key, for a caller that has no reason to mint its own."""
        return "cmsg_" + uuid.uuid4().hex[:24]

    # -- internals ----------------------------------------------------

    def _next_seq(self, conn: Any) -> int:
        """Allocate the next sequence number inside *conn*'s transaction.

        Deliberately not `MAX(seq) + 1`: two writers reading the same
        maximum both pick the same number, and the second insert fails
        on the primary key -- or worse, on a backend without that
        constraint, succeeds and produces two rows a client can only
        see one of. A counter row read and written in the same
        transaction serialises the allocation with the write it labels.
        """
        row = conn.execute(
            "SELECT next_seq FROM hyperlink_change_counter WHERE id = 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO hyperlink_change_counter (id, next_seq) VALUES (1, 2)"
            )
            return 1
        seq = int(row["next_seq"])
        conn.execute(
            "UPDATE hyperlink_change_counter SET next_seq = ? WHERE id = 1",
            (seq + 1,),
        )
        return seq


def _change_from_row(row: Any) -> Change:
    return Change(
        seq=int(row["seq"]),
        kind=row["kind"],
        entity=row["entity"],
        entity_id=row["entity_id"],
        owner=row["owner"],
        created_at=float(row["created_at"]),
        session_id=row["session_id"] or "",
        device_id=row["device_id"] or "",
        payload=json.loads(row["payload"] or "{}"),
    )
