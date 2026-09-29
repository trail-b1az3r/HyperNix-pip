"""Tests for hypernix.t1api.jobs — the async job system.

Most tests use ``synchronous=True`` for deterministic ordering (no
sleeping/polling for a background thread). A dedicated
``TestRealAsyncExecution`` class exercises the actual thread-pool path,
including cancelling a job that's genuinely mid-flight in another thread.
"""
from __future__ import annotations

import time

import pytest

from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.jobs import JobQueue, JobStatus


@pytest.fixture
def queue(tmp_path) -> JobQueue:
    q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=True)
    yield q
    q.shutdown()


def _echo(payload, cancel_event):
    return {"echo": payload}


def _fail(payload, cancel_event):
    raise ValueError("boom")


class TestSubmitSynchronous:
    def test_successful_job_transitions_to_succeeded(self, queue: JobQueue):
        queue.register_handler("echo", _echo)
        job = queue.submit("echo", {"x": 1}, created_by="user1")
        assert job.status == JobStatus.SUCCEEDED
        assert job.result == {"echo": {"x": 1}}
        assert job.finished_at is not None

    def test_failing_handler_transitions_to_failed_with_error_recorded(self, queue: JobQueue):
        queue.register_handler("fail", _fail)
        job = queue.submit("fail", {}, created_by="user1")
        assert job.status == JobStatus.FAILED
        assert "boom" in job.error

    def test_unregistered_kind_raises_not_supported(self, queue: JobQueue):
        with pytest.raises(T1APIError) as exc_info:
            queue.submit("no-such-kind", {}, created_by="user1")
        assert exc_info.value.code == T1ErrorCode.NOT_SUPPORTED

    def test_job_records_creator(self, queue: JobQueue):
        queue.register_handler("echo", _echo)
        job = queue.submit("echo", {}, created_by="user42")
        assert job.created_by == "user42"


class TestLookup:
    def test_get_unknown_returns_none(self, queue: JobQueue):
        assert queue.get("nope") is None

    def test_require_unknown_raises_job_not_found(self, queue: JobQueue):
        with pytest.raises(T1APIError) as exc_info:
            queue.require("nope")
        assert exc_info.value.code == T1ErrorCode.JOB_NOT_FOUND


class TestList:
    def test_list_filters_by_created_by(self, queue: JobQueue):
        queue.register_handler("echo", _echo)
        queue.submit("echo", {}, created_by="alice")
        queue.submit("echo", {}, created_by="bob")
        alice_jobs = queue.list(created_by="alice")
        assert len(alice_jobs) == 1

    def test_list_filters_by_status(self, queue: JobQueue):
        queue.register_handler("echo", _echo)
        queue.register_handler("fail", _fail)
        queue.submit("echo", {}, created_by="alice")
        queue.submit("fail", {}, created_by="alice")
        succeeded = queue.list(status=JobStatus.SUCCEEDED)
        failed = queue.list(status=JobStatus.FAILED)
        assert len(succeeded) == 1
        assert len(failed) == 1


class TestCancel:
    def test_cancel_terminal_job_raises_job_not_cancellable(self, queue: JobQueue):
        queue.register_handler("echo", _echo)
        job = queue.submit("echo", {}, created_by="alice")
        with pytest.raises(T1APIError) as exc_info:
            queue.cancel(job.job_id)
        assert exc_info.value.code == T1ErrorCode.JOB_NOT_CANCELLABLE

    def test_cancel_unknown_job_raises_job_not_found(self, queue: JobQueue):
        with pytest.raises(T1APIError) as exc_info:
            queue.cancel("nope")
        assert exc_info.value.code == T1ErrorCode.JOB_NOT_FOUND


class TestRealAsyncExecution:
    """Exercises the actual ThreadPoolExecutor path (synchronous=False),
    including a genuine race: cancelling a job while it's running in
    another thread."""

    def test_job_runs_in_background_and_completes(self, tmp_path):
        q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=False, max_workers=2)
        try:
            q.register_handler("echo", _echo)
            job = q.submit("echo", {"x": 1}, created_by="alice")
            # Should not block submit() waiting for completion.
            deadline = time.time() + 2.0
            while q.require(job.job_id).status not in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                assert time.time() < deadline, "job never completed"
                time.sleep(0.01)
            final = q.require(job.job_id)
            assert final.status == JobStatus.SUCCEEDED
            assert final.result == {"echo": {"x": 1}}
        finally:
            q.shutdown()

    def test_cancelling_a_running_job_stops_it(self, tmp_path):
        def slow_handler(payload, cancel_event):
            for i in range(200):
                if cancel_event.is_set():
                    return {"stopped_at": i}
                time.sleep(0.01)
            return {"done": True}

        q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=False, max_workers=2)
        try:
            q.register_handler("slow", slow_handler)
            job = q.submit("slow", {}, created_by="alice")

            deadline = time.time() + 2.0
            while q.require(job.job_id).status != JobStatus.RUNNING:
                assert time.time() < deadline, "job never started running"
                time.sleep(0.01)

            cancelled = q.cancel(job.job_id)
            assert cancelled.status in (JobStatus.RUNNING, JobStatus.CANCELLED)

            deadline = time.time() + 2.0
            while q.require(job.job_id).status == JobStatus.RUNNING:
                assert time.time() < deadline, "job never finished cancelling"
                time.sleep(0.01)

            final = q.require(job.job_id)
            assert final.status == JobStatus.CANCELLED
        finally:
            q.shutdown()

    def test_submit_does_not_block_caller(self, tmp_path):
        """A slow handler must not make submit() itself slow — it should
        dispatch to the pool and return immediately."""
        def slow_handler(payload, cancel_event):
            time.sleep(1.0)
            return {}

        q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=False, max_workers=2)
        try:
            q.register_handler("slow", slow_handler)
            start = time.time()
            job = q.submit("slow", {}, created_by="alice")
            elapsed = time.time() - start
            assert elapsed < 0.5
            assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        finally:
            q.shutdown()


class TestEventBusIntegration:
    """JobQueue optionally publishes job.<status> events onto a shared
    EventBus for ANY job kind, with no per-handler code required."""

    def test_successful_job_publishes_queued_running_succeeded(self, tmp_path):
        from hypernix.t1api.events import EventBus

        bus = EventBus()
        q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=True, event_bus=bus)
        try:
            q.register_handler("echo", _echo)
            sub = bus.subscribe()
            job = q.submit("echo", {"x": 1}, created_by="alice")
            events = list(sub.iter(timeout=0.1))
            assert [e.type for e in events] == ["job.queued", "job.running", "job.succeeded"]
            assert all(e.data["job_id"] == job.job_id for e in events)
        finally:
            q.shutdown()

    def test_failed_job_publishes_job_failed_with_error(self, tmp_path):
        from hypernix.t1api.events import EventBus

        bus = EventBus()
        q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=True, event_bus=bus)
        try:
            q.register_handler("fail", _fail)
            sub = bus.subscribe()
            q.submit("fail", {}, created_by="alice")
            events = list(sub.iter(timeout=0.1))
            assert events[-1].type == "job.failed"
            assert "boom" in events[-1].data["error"]
        finally:
            q.shutdown()

    def test_jobs_work_fine_without_an_event_bus(self, tmp_path):
        """Backward compatibility: event_bus is optional and defaults to
        None — every earlier test in this file relies on this."""
        q = JobQueue(SQLiteBackend(tmp_path / "t1api.sqlite3"), synchronous=True)
        try:
            q.register_handler("echo", _echo)
            job = q.submit("echo", {}, created_by="alice")
            assert job.status == JobStatus.SUCCEEDED
        finally:
            q.shutdown()
