"""t1api.jobs — the asynchronous job system.

Every long-running or cross-system operation (module sync, remote fetch,
future HyperNix-pip capabilities like conversion/quantization/training —
see the spec's "HYPERNIX-PIP INTEGRATION" section) runs as a job rather
than blocking a request. States match the spec exactly: queued, running,
succeeded, failed, cancelled.

**Plugin-like extension system**: job *kinds* are handlers registered by
name (:meth:`JobQueue.register_handler`) rather than a hard-coded
``if kind == ...`` chain. Submitting a job whose kind has no registered
handler raises ``NOT_SUPPORTED`` — same "don't invent a capability that
doesn't exist" rule the spec applies to HyperNix-pip integration, applied
here to jobs generally. Beta 2 registers exactly one real handler
(``module_sync``, wired in ``t1api.app`` since it needs both
``ModuleRegistry`` and ``ServerRegistry``); everything else (real
HyperNix-pip conversion/quantization/training jobs) is future work — the
registry mechanism is what ships now, not fake implementations of those
kinds.

Execution model: a small in-process thread pool
(``concurrent.futures.ThreadPoolExecutor``), not a separate task queue
(Celery/RQ/etc.) — consistent with "SQLite for development" scale. Job
*records* persist in SQLite so ``GET /jobs/{id}`` survives a request
crossing threads; the executor just drives state transitions on top of
that.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .db import SQLiteBackend
from .errors import T1APIError, T1ErrorCode
from .events import EventBus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    error TEXT,
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_by ON jobs(created_by);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}

JobHandler = Callable[[dict[str, Any], threading.Event], dict[str, Any]]
"""A job handler receives the job's payload and a cooperative cancellation
``Event`` (handlers that loop should check ``event.is_set()``) and returns
a JSON-serializable result dict, or raises to fail the job."""


@dataclass
class JobEntry:
    job_id: str
    kind: str
    payload: dict[str, Any]
    status: JobStatus
    result: dict[str, Any] | None
    error: str | None
    created_by: str
    created_at: float
    started_at: float | None
    finished_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "payload": self.payload,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def _from_row(cls, row: Any) -> JobEntry:
        return cls(
            job_id=row["job_id"],
            kind=row["kind"],
            payload=json.loads(row["payload"]),
            status=JobStatus(row["status"]),
            result=json.loads(row["result"]) if row["result"] is not None else None,
            error=row["error"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


class JobQueue:
    """SQLite-persisted job records + an in-process thread-pool executor.

    Args:
        backend: shared SQLite backend (same file as servers/modules is
            fine — different table).
        max_workers: thread pool size.
        synchronous: if True, :meth:`submit` runs the handler inline
            before returning instead of dispatching to the pool. Used by
            tests that want deterministic ordering without sleeping/
            polling for a background thread to finish; production code
            should leave this False.
        event_bus: optional :class:`t1api.events.EventBus`. When set,
            every state transition (queued/running/succeeded/failed/
            cancelled) is published automatically as a ``job.<status>``
            event — callers of :meth:`submit` don't need to know or care
            that an event bus exists; this is what makes ``GET /events``
            show job activity for every job kind, including ones nobody
            wrote bus-publishing code for.
    """

    def __init__(
        self,
        backend: SQLiteBackend | None = None,
        *,
        max_workers: int = 4,
        synchronous: bool = False,
        event_bus: EventBus | None = None,
    ) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)
        self._handlers: dict[str, JobHandler] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._synchronous = synchronous
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="t1api-job")
        self._event_bus = event_bus

    def _publish(self, job_id: str, kind: str, status: JobStatus, **extra: Any) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(
                f"job.{status.value}", {"job_id": job_id, "kind": kind, **extra}, source="jobs"
            )
        except Exception:  # noqa: BLE001 - event publishing must never break job execution
            logger.warning("t1api.jobs: failed to publish event for job %s", job_id[:8], exc_info=True)

    def register_handler(self, kind: str, handler: JobHandler) -> None:
        self._handlers[kind] = handler
        logger.info("t1api.jobs: registered handler for kind=%s", kind)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    # ------------------------------------------------------------------

    def submit(self, kind: str, payload: dict[str, Any], *, created_by: str) -> JobEntry:
        if kind not in self._handlers:
            raise T1APIError(
                T1ErrorCode.NOT_SUPPORTED,
                f"No job handler registered for kind '{kind}'.",
                details={"kind": kind, "available_kinds": sorted(self._handlers)},
                http_status=501,
            )
        now = time.time()
        entry = JobEntry(
            job_id=uuid.uuid4().hex,
            kind=kind,
            payload=payload,
            status=JobStatus.QUEUED,
            result=None,
            error=None,
            created_by=created_by,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        self._insert(entry)
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[entry.job_id] = cancel_event
        logger.info("t1api.jobs: submitted %s kind=%s", entry.job_id[:8], kind)
        self._publish(entry.job_id, kind, JobStatus.QUEUED)

        if self._synchronous:
            self._run(entry.job_id, kind, payload, cancel_event)
        else:
            self._executor.submit(self._run, entry.job_id, kind, payload, cancel_event)
        return self.require(entry.job_id)

    def _run(self, job_id: str, kind: str, payload: dict[str, Any], cancel_event: threading.Event) -> None:
        try:
            self._run_inner(job_id, kind, payload, cancel_event)
        finally:
            # The event is only useful while the job can still be waited
            # on or cancelled; every path through _run_inner ends in a
            # terminal status, so this is safe unconditionally. Without
            # it, a long-running server accumulates one Event per job
            # ever submitted, forever.
            with self._lock:
                self._cancel_events.pop(job_id, None)

    def _run_inner(self, job_id: str, kind: str, payload: dict[str, Any], cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            self._set_status(job_id, JobStatus.CANCELLED, finished=True)
            self._publish(job_id, kind, JobStatus.CANCELLED)
            return
        self._set_status(job_id, JobStatus.RUNNING, started=True)
        self._publish(job_id, kind, JobStatus.RUNNING)
        handler = self._handlers[kind]
        try:
            result = handler(payload, cancel_event)
            if cancel_event.is_set():
                self._set_status(job_id, JobStatus.CANCELLED, finished=True)
                self._publish(job_id, kind, JobStatus.CANCELLED)
            else:
                self._set_result(job_id, result)
                self._publish(job_id, kind, JobStatus.SUCCEEDED)
        except Exception as exc:  # noqa: BLE001 - job failures are data, not a crash
            logger.warning("t1api.jobs: job %s failed: %s", job_id[:8], exc)
            self._set_error(job_id, f"{exc.__class__.__name__}: {exc}")
            self._publish(job_id, kind, JobStatus.FAILED, error=f"{exc.__class__.__name__}: {exc}")
            logger.debug("t1api.jobs: %s", traceback.format_exc())

    def get(self, job_id: str) -> JobEntry | None:
        with self._lock, self.backend.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return JobEntry._from_row(row) if row else None

    def require(self, job_id: str) -> JobEntry:
        entry = self.get(job_id)
        if entry is None:
            raise T1APIError(
                T1ErrorCode.JOB_NOT_FOUND,
                f"Job '{job_id}' not found.",
                details={"job_id": job_id},
                http_status=404,
            )
        return entry

    def list(self, *, status: JobStatus | None = None, created_by: str | None = None) -> list[JobEntry]:
        query = "SELECT * FROM jobs"
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if created_by is not None:
            clauses.append("created_by = ?")
            params.append(created_by)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._lock, self.backend.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [JobEntry._from_row(r) for r in rows]

    def cancel(self, job_id: str) -> JobEntry:
        entry = self.require(job_id)
        if entry.status in _TERMINAL_STATUSES:
            raise T1APIError(
                T1ErrorCode.JOB_NOT_CANCELLABLE,
                f"Job '{job_id}' is already {entry.status.value} and cannot be cancelled.",
                details={"job_id": job_id, "status": entry.status.value},
                http_status=409,
            )
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        # Conditional on status still being 'queued' at write time, not on
        # the *entry* fetched above: between that read and here, the
        # executor may have already picked the job up and moved it to
        # 'running'. Without the WHERE clause, this would force-write
        # CANCELLED over a job that is actually mid-execution, and the
        # worker thread's own terminal write would then race it.
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, finished_at=? WHERE job_id=? AND status=?",
                (JobStatus.CANCELLED.value, now, job_id, JobStatus.QUEUED.value),
            )
            cancelled = cur.rowcount == 1
        if cancelled:
            self._publish(job_id, entry.kind, JobStatus.CANCELLED)
        return self.require(job_id)

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------

    def _set_status(self, job_id: str, status: JobStatus, *, started: bool = False, finished: bool = False) -> None:
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            if started:
                conn.execute("UPDATE jobs SET status=?, started_at=? WHERE job_id=?", (status.value, now, job_id))
            elif finished:
                conn.execute("UPDATE jobs SET status=?, finished_at=? WHERE job_id=?", (status.value, now, job_id))
            else:
                conn.execute("UPDATE jobs SET status=? WHERE job_id=?", (status.value, job_id))

    def _set_result(self, job_id: str, result: dict[str, Any]) -> None:
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, result=?, finished_at=? WHERE job_id=?",
                (JobStatus.SUCCEEDED.value, json.dumps(result), now, job_id),
            )

    def _set_error(self, job_id: str, error: str) -> None:
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE job_id=?",
                (JobStatus.FAILED.value, error, now, job_id),
            )

    def _insert(self, entry: JobEntry) -> None:
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (job_id, kind, payload, status, result, error, created_by,
                    created_at, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.job_id, entry.kind, json.dumps(entry.payload), entry.status.value,
                    None, None, entry.created_by, entry.created_at, None, None,
                ),
            )


__all__ = ["JobStatus", "JobEntry", "JobQueue", "JobHandler"]
