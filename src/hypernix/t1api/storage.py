"""t1api.storage — usage-event persistence.

Beta 1 shipped this as SQLite-only. Beta 3 keeps every query exactly as
it was and simply hands them to a :class:`t1api.db.SQLBackend`, so the
same code runs on "SQLite for development, PostgreSQL for production"
(the spec's own words) with no branching here. ``UsageStore(db_path)``
still works verbatim — a path builds a :class:`SQLiteBackend` internally
— so nothing constructed the Beta 1 way breaks.

Beta 3 also widens what a usage event records and what can be asked of
it, because the spec's usage/cost endpoints need slices Beta 1 didn't
store:

* ``user_id`` / ``account_id`` columns, so usage can be reported
  "per-user" and "per-account" and not only per-key. They're added by
  online migration (:meth:`_ensure_columns`), so an existing Beta 1/2
  database upgrades in place rather than needing a dump/reload.
* :meth:`history` — raw events with date-range + dimension filtering.
* :meth:`aggregate` — one grouped-totals query that every "usage by X"
  endpoint (model/key/server/module/user/account/day) is built from,
  instead of a bespoke query per dimension.

Stdlib-only for SQLite; PostgreSQL pulls psycopg in lazily via
``t1api.db``. Either way this module imports fine without FastAPI.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .db import DIALECT_POSTGRES, SQLBackend, SQLiteBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 1,
    endpoint TEXT NOT NULL DEFAULT '',
    server_id TEXT NOT NULL DEFAULT '',
    module_id TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_key_id ON usage_events(key_id);
CREATE INDEX IF NOT EXISTS idx_usage_model_id ON usage_events(model_id);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts);
"""

# Columns added after Beta 1. Applied by online migration so an existing
# database upgrades in place. (name, DDL type + default)
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("user_id", "TEXT NOT NULL DEFAULT ''"),
    ("account_id", "TEXT NOT NULL DEFAULT ''"),
)

# Dimensions callers may group or filter by. An allowlist, not a
# free-form column name — these values reach SQL as identifiers, so they
# can never come from a request unchecked.
GROUPABLE = ("model_id", "key_id", "server_id", "module_id", "user_id", "account_id", "endpoint")
FILTERABLE = GROUPABLE


class UsageStore:
    """Thread-safe store for per-request usage events.

    Connections are short-lived (opened per call): SQLite handles aren't
    safe to share across threads without care, and the T1 API runs under
    an async server with a thread pool. PostgreSQL deployments get
    connection pooling transparently from ``t1api.db``.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        backend: SQLBackend | None = None,
    ) -> None:
        self.backend = backend or SQLiteBackend(db_path)
        # Kept for backwards compatibility with Beta 1/2 callers and tests
        # that read ``store.db_path``; None when running on PostgreSQL.
        self.db_path = getattr(self.backend, "db_path", None)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self.backend.executescript(_SCHEMA)
            self._ensure_columns()

    def _existing_columns(self, conn: Any) -> set[str]:
        if self.backend.dialect == DIALECT_POSTGRES:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                ("usage_events",),
            ).fetchall()
            return {r["column_name"] for r in rows}
        rows = conn.execute("PRAGMA table_info(usage_events)").fetchall()
        return {r["name"] for r in rows}

    def _ensure_columns(self) -> None:
        """Add post-Beta-1 columns to an existing table. Idempotent —
        safe to run on every startup, including concurrent ones (a lost
        race just means the other process already added the column)."""
        with self.backend.connect() as conn:
            existing = self._existing_columns(conn)
            for name, ddl in _ADDED_COLUMNS:
                if name in existing:
                    continue
                conn.execute(f"ALTER TABLE usage_events ADD COLUMN {name} {ddl}")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        key_id: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 1,
        endpoint: str = "",
        server_id: str = "",
        module_id: str = "",
        user_id: str = "",
        account_id: str = "",
        ts: float | None = None,
    ) -> None:
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """INSERT INTO usage_events
                   (key_id, model_id, input_tokens, output_tokens, requests,
                    endpoint, server_id, module_id, user_id, account_id, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    key_id,
                    model_id,
                    int(input_tokens),
                    int(output_tokens),
                    int(requests),
                    endpoint,
                    server_id,
                    module_id,
                    user_id,
                    account_id,
                    ts if ts is not None else time.time(),
                ),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def _where(
        filters: dict[str, str | None],
        *,
        since: float | None,
        until: float | None,
    ) -> tuple[str, list[Any]]:
        """Build a WHERE clause from an allowlisted filter dict.

        Filter *keys* are validated against :data:`FILTERABLE` before
        being interpolated as column names; filter *values* are always
        bound parameters. An unknown key is a programming error, not a
        user error — it raises rather than being silently dropped, so a
        typo can't quietly widen a query's result set.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in filters.items():
            if value is None:
                continue
            if column not in FILTERABLE:
                raise ValueError(f"{column!r} is not a filterable usage dimension")
            clauses.append(f"{column} = ?")
            params.append(value)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("ts < ?")
            params.append(until)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    def totals_for_key(
        self, key_id: str, *, since: float | None = None, model_id: str | None = None
    ) -> dict[str, Any]:
        """Aggregate (requests, input_tokens, output_tokens) for a key,
        optionally filtered by a start timestamp and/or model_id."""
        return self.totals(key_id=key_id, model_id=model_id, since=since)

    def totals(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        **filters: str | None,
    ) -> dict[str, Any]:
        where, params = self._where(filters, since=since, until=until)
        query = (
            "SELECT COALESCE(SUM(requests),0) AS requests, "
            "COALESCE(SUM(input_tokens),0) AS input_tokens, "
            "COALESCE(SUM(output_tokens),0) AS output_tokens "
            "FROM usage_events" + where
        )
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return _totals_row(row)

    def aggregate(
        self,
        group_by: str,
        *,
        since: float | None = None,
        until: float | None = None,
        limit: int = 500,
        **filters: str | None,
    ) -> list[dict[str, Any]]:
        """Grouped totals along one dimension — the single query behind
        every "usage by model/key/server/module/user/account" report."""
        if group_by not in GROUPABLE:
            raise ValueError(f"{group_by!r} is not a groupable usage dimension")
        groupable_columns = {name: name for name in GROUPABLE}
        group_column = groupable_columns[group_by]
        where, params = self._where(filters, since=since, until=until)
        query = (
            f"SELECT {group_column} AS group_key, "
            "COALESCE(SUM(requests),0) AS requests, "
            "COALESCE(SUM(input_tokens),0) AS input_tokens, "
            "COALESCE(SUM(output_tokens),0) AS output_tokens "
            f"FROM usage_events{where} GROUP BY {group_column} "
            "ORDER BY (SUM(input_tokens) + SUM(output_tokens)) DESC LIMIT ?"
        )
        with self._lock, self.backend.connect() as conn:
            rows = conn.execute(query, [*params, int(limit)]).fetchall()
        return [{group_by: r["group_key"], **_totals_row(r)} for r in rows]

    def breakdown_by_model(self, key_id: str, *, since: float | None = None) -> list[dict[str, Any]]:
        return self.aggregate("model_id", key_id=key_id, since=since)

    def history(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
        **filters: str | None,
    ) -> list[dict[str, Any]]:
        """Raw usage events, newest first — backs GET /usage/history."""
        where, params = self._where(filters, since=since, until=until)
        query = (
            "SELECT key_id, model_id, input_tokens, output_tokens, requests, endpoint, "
            "server_id, module_id, user_id, account_id, ts FROM usage_events"
            f"{where} ORDER BY ts DESC LIMIT ? OFFSET ?"
        )
        with self._lock, self.backend.connect() as conn:
            rows = conn.execute(query, [*params, int(limit), int(offset)]).fetchall()
        return [
            {
                "key_id": r["key_id"],
                "model_id": r["model_id"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "total_tokens": r["input_tokens"] + r["output_tokens"],
                "requests": r["requests"],
                "endpoint": r["endpoint"],
                "server_id": r["server_id"],
                "module_id": r["module_id"],
                "user_id": r["user_id"],
                "account_id": r["account_id"],
                "ts": r["ts"],
            }
            for r in rows
        ]

    def first_event_ts(self, **filters: str | None) -> float | None:
        """Timestamp of the earliest matching event, or None. Used by the
        cost forecaster to know how long a spend rate has been observed
        for — forecasting from a 10-minute-old account is not the same as
        forecasting from a 30-day-old one."""
        where, params = self._where(filters, since=None, until=None)
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(f"SELECT MIN(ts) AS first_ts FROM usage_events{where}", params).fetchone()
        return row["first_ts"] if row and row["first_ts"] is not None else None

    def event_count(self) -> int:
        with self._lock, self.backend.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()
        return int(row["n"])


def _totals_row(row: Any) -> dict[str, Any]:
    input_tokens = int(row["input_tokens"] or 0)
    output_tokens = int(row["output_tokens"] or 0)
    return {
        "requests": int(row["requests"] or 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


__all__ = ["UsageStore", "GROUPABLE", "FILTERABLE"]
