"""t1api.audit — the audit log.

The spec asks for this three separate times: "Add audit logs for all
administrator actions", "Record security-relevant events in the audit
log", and "Provide audit logging" as a hard requirement. Beta 1/2 logged
some of it to the Python logger; Beta 3 makes it a **queryable, durable
record** with a stable schema, because an audit trail you can't query
after the fact isn't an audit trail.

What gets recorded
------------------
Three categories, all through :class:`AuditLog.record`:

1. **Administrator actions** — anything gated by ``require_admin``:
   key rotation/promotion, server trust changes, registry mutation,
   payment-token minting, direct balance credits, network-policy edits,
   forced limits, module deletion.
2. **Security-relevant events** — authentication failures, scope
   denials, rate-limit trips, blocked IPs, SSRF/path-traversal
   rejections, mTLS failures.
3. **State-changing writes** — module create/upload/sync, server
   register/delete, job submit/cancel, redemptions.

What never gets recorded
------------------------
Secrets. Not raw T1 keys, not scoped tokens, not payment tokens, not
passwords or DSNs. :func:`mask_identifier` is the only way an identifier
reaches a record, and it keeps a short prefix and nothing else. The
``details`` dict is passed through :func:`scrub_details`, which drops
any key whose name looks secret-ish (``key``, ``token``, ``secret``,
``password``, ``authorization``, ``dsn``, ``credential``) regardless of
what the caller passed — a defence against a future call site
accidentally handing us the thing we must never store. The scrub is an
allowlist-of-shape, deny-by-name check applied at write time, so it also
protects records written by code that hasn't been reviewed against this
rule.

Failure policy
--------------
An audit write must never take down the request it is describing:
:meth:`record` catches and logs storage failures rather than raising.
The one exception is :meth:`record_strict`, used where losing the record
would be worse than failing the operation (admin credential changes) —
callers there want the write to be part of the transaction's success
condition.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .db import SQLBackend, SQLiteBackend

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    audit_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    category TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    actor_key_id TEXT NOT NULL DEFAULT '',
    actor_is_admin INTEGER NOT NULL DEFAULT 0,
    resource_type TEXT NOT NULL DEFAULT '',
    resource_id TEXT NOT NULL DEFAULT '',
    client_ip TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events(actor_key_id);
CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_events(category);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_events(action);
"""

# Any details key containing one of these substrings is dropped before
# storage. Substring match, case-insensitive, so 't1_token_secret',
# 'api_key' and 'Authorization' are all caught by one rule each.
_SECRET_KEY_MARKERS = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "credential",
    "dsn",
    "database_url",
    "cookie",
)

# ...except these, which are identifiers rather than secrets. A key_id is
# not a key; a payment_token_id is not a payment token. Without this
# carve-out the marker list above would strip exactly the fields that
# make an audit record useful.
_SECRET_KEY_EXCEPTIONS = frozenset(
    {
        "key_id",
        "target_key_id",
        "actor_key_id",
        "payment_token_id",
        "token_id",
        "key_type",
        "key_count",
        "keys",
        "scopes",
        "token_expires_at",
    }
)

_REDACTED = "[redacted]"


class AuditCategory(StrEnum):
    ADMIN = "admin"  # an administrator action
    SECURITY = "security"  # auth failure, blocked IP, rate limit, SSRF...
    WRITE = "write"  # ordinary state-changing operation
    BILLING = "billing"  # anything touching balances or payment tokens


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    DENIED = "denied"
    FAILURE = "failure"


def mask_identifier(value: str | None, keep: int = 8) -> str:
    """``'abcdef0123456789'`` → ``'abcdef01…'``.

    Used for every identifier that reaches a record. Even key *ids* are
    masked: they're not secrets, but a full one is enough to make an
    audit dump a useful target list, and the prefix is enough to
    correlate records.
    """
    if not value:
        return ""
    return value[:keep] + "…" if len(value) > keep else value


def _looks_secret(name: str) -> bool:
    lowered = name.lower()
    if lowered in _SECRET_KEY_EXCEPTIONS:
        return False
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def scrub_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Drop anything secret-shaped from a details dict, recursively.

    Deny-by-name rather than trusting call sites: if a future caller
    passes ``{"key": "T1-live-..."}`` it is replaced with
    ``{"key": "[redacted]"}`` here rather than being written to disk.
    Values are also length-capped so an audit record can't become a
    dumping ground for a request body.
    """
    if not details:
        return {}
    out: dict[str, Any] = {}
    for name, value in details.items():
        if _looks_secret(str(name)):
            out[str(name)] = _REDACTED
            continue
        if isinstance(value, dict):
            out[str(name)] = scrub_details(value)
        elif isinstance(value, (list, tuple)):
            out[str(name)] = [
                scrub_details(v) if isinstance(v, dict) else _cap(v) for v in list(value)[:20]
            ]
        else:
            out[str(name)] = _cap(value)
    return out


def _cap(value: Any, limit: int = 500) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


@dataclass
class AuditRecord:
    audit_id: str
    ts: float
    category: AuditCategory
    action: str
    outcome: AuditOutcome
    actor_key_id: str
    actor_is_admin: bool
    resource_type: str
    resource_id: str
    client_ip: str
    request_id: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "ts": self.ts,
            "category": self.category.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "actor_key_id": self.actor_key_id,
            "actor_is_admin": self.actor_is_admin,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "client_ip": self.client_ip,
            "request_id": self.request_id,
            "details": self.details,
        }


class AuditLog:
    """Durable, queryable audit trail backed by the shared SQL backend."""

    def __init__(self, backend: SQLBackend | None = None, *, enabled: bool = True) -> None:
        self.backend = backend or SQLiteBackend()
        self.enabled = enabled
        self._lock = threading.Lock()
        # The schema is created even when auditing is disabled. `enabled`
        # gates *writing*, not the table's existence: a disabled log must
        # still be readable (returning nothing) so that toggling
        # T1_AUDIT_ENABLED, or querying GET /audit on a deployment that
        # has auditing off, produces an empty result rather than a
        # database error from somewhere deep in the stack.
        self.backend.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(
        self,
        action: str,
        *,
        category: AuditCategory = AuditCategory.WRITE,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        actor_key_id: str = "",
        actor_is_admin: bool = False,
        resource_type: str = "",
        resource_id: str = "",
        client_ip: str = "",
        request_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditRecord | None:
        """Write one audit record. Never raises — a failed audit write is
        logged and swallowed so it can't turn a successful operation into
        a 500. Use :meth:`record_strict` where that trade-off is wrong."""
        try:
            return self._write(
                action,
                category=category,
                outcome=outcome,
                actor_key_id=actor_key_id,
                actor_is_admin=actor_is_admin,
                resource_type=resource_type,
                resource_id=resource_id,
                client_ip=client_ip,
                request_id=request_id,
                details=details,
            )
        except Exception:  # noqa: BLE001 - see docstring
            logger.exception("t1api.audit: failed to write audit record for action=%s", action)
            return None

    def record_strict(self, action: str, **kwargs: Any) -> AuditRecord | None:
        """Like :meth:`record` but propagates storage failures. For the
        handful of actions (admin credential changes) where an
        unrecorded success is worse than a failed request."""
        return self._write(action, **kwargs)

    def _write(
        self,
        action: str,
        *,
        category: AuditCategory = AuditCategory.WRITE,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        actor_key_id: str = "",
        actor_is_admin: bool = False,
        resource_type: str = "",
        resource_id: str = "",
        client_ip: str = "",
        request_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditRecord | None:
        if not self.enabled:
            return None
        # A concealed key's address is kept as its network, except on
        # security records — see hypernix.t1api.privacy.
        conceal = getattr(self, "conceal", None)
        if (client_ip and conceal is not None and category is not AuditCategory.SECURITY
                and conceal.is_concealed(actor_key_id)):
            from .privacy import mask_address

            client_ip = mask_address(client_ip)
        record = AuditRecord(
            audit_id=uuid.uuid4().hex,
            ts=time.time(),
            category=category,
            action=action,
            outcome=outcome,
            actor_key_id=mask_identifier(actor_key_id),
            actor_is_admin=bool(actor_is_admin),
            resource_type=resource_type,
            resource_id=mask_identifier(resource_id, keep=12),
            client_ip=client_ip,
            request_id=request_id,
            details=scrub_details(details),
        )
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """INSERT INTO audit_events
                   (audit_id, ts, category, action, outcome, actor_key_id, actor_is_admin,
                    resource_type, resource_id, client_ip, request_id, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.audit_id,
                    record.ts,
                    record.category.value,
                    record.action,
                    record.outcome.value,
                    record.actor_key_id,
                    1 if record.actor_is_admin else 0,
                    record.resource_type,
                    record.resource_id,
                    record.client_ip,
                    record.request_id,
                    json.dumps(record.details, default=str),
                ),
            )
        return record

    # ------------------------------------------------------------------
    # Reads (admin-only at the HTTP layer)
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        category: str | None = None,
        action: str | None = None,
        outcome: str | None = None,
        actor_key_id: str | None = None,
        resource_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        # Column names here are literals in this function, never caller
        # input — the caller only supplies bound values.
        for column, value in (
            ("category", category),
            ("action", action),
            ("outcome", outcome),
            ("resource_type", resource_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if actor_key_id is not None:
            # Callers naturally have the full key_id; records store the
            # mask. Match on the mask so the obvious query works.
            clauses.append("actor_key_id = ?")
            params.append(mask_identifier(actor_key_id))
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("ts < ?")
            params.append(until)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"SELECT * FROM audit_events{where} ORDER BY ts DESC LIMIT ? OFFSET ?"
        with self._lock, self.backend.connect() as conn:
            rows = conn.execute(query, [*params, int(limit), int(offset)]).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self) -> int:
        with self._lock, self.backend.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM audit_events").fetchone()
        return int(row["n"])

    def purge_before(self, ts: float) -> int:
        """Delete records older than *ts*, returning how many went.

        Retention is a deployment policy (and in some jurisdictions a
        legal one), so it's an explicit operation rather than something
        that happens on its own schedule.
        """
        with self._lock, self.backend.connect() as conn:
            before = conn.execute("SELECT COUNT(*) AS n FROM audit_events WHERE ts < ?", (ts,)).fetchone()
            conn.execute("DELETE FROM audit_events WHERE ts < ?", (ts,))
        return int(before["n"])


def _row_to_record(row: Any) -> AuditRecord:
    try:
        details = json.loads(row["details"])
    except (TypeError, ValueError):
        details = {}
    return AuditRecord(
        audit_id=row["audit_id"],
        ts=row["ts"],
        category=AuditCategory(row["category"]),
        action=row["action"],
        outcome=AuditOutcome(row["outcome"]),
        actor_key_id=row["actor_key_id"],
        actor_is_admin=bool(row["actor_is_admin"]),
        resource_type=row["resource_type"],
        resource_id=row["resource_id"],
        client_ip=row["client_ip"],
        request_id=row["request_id"],
        details=details,
    )


__all__ = [
    "AuditCategory",
    "AuditLog",
    "AuditOutcome",
    "AuditRecord",
    "mask_identifier",
    "scrub_details",
]
