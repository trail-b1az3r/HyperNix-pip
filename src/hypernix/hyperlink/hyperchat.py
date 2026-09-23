"""hyperchat — several prompts in flight, or a queue when they cannot be.

The problem. :class:`~hypernix.hyperlink.managed.ManagedRunner` owns one
llama.cpp process, so a second prompt arriving while the first is being
answered has two possible fates and both used to happen by accident: it
contends with the first inside one server and both get slower, or it is
dropped. HyperLink shows neither — it shows a spinner that does not move.

Two answers, one shape
----------------------
When the operator has turned multiple instances on, N copies of the
model run side by side and N prompts are answered at once. When they
have not, prompts wait in a queue and are answered in the order they
arrived. Callers do not branch on which: both are :class:`Hyperchat`,
both return a :class:`Ticket`, and the queue is simply the pool with
one worker in it. Anything else means every caller — HyperLink, the API,
the CLI — implementing a fallback, and each of them getting it slightly
different.

One core is never allocated
---------------------------
Instances get two cores each and the machine keeps one for itself. Not
a rounding-down convenience: llama.cpp will happily take every core it
is given, and with all of them spoken for, the process that has to
accept the *next* HTTP request, answer a health check, or notice a
child has died does not get scheduled. The symptom is a server that
stops responding under exactly the load this feature was added for.
So the budget is ``(cores - 1) // 2``, and a machine too small for two
instances runs the queue instead — which is a real answer, not a
degraded one.

Ordering
--------
The queue is a queue, not a lock. A mutex hands the next prompt to
whichever thread the scheduler happens to wake, so a burst arrives out
of order and an unlucky request can wait behind every prompt sent after
it. A phone that has been waiting the longest should be served next,
and it can only be told its position if there is a line to have a
position in.
"""
from __future__ import annotations

import itertools
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CORES_PER_INSTANCE",
    "RESERVED_CORES",
    "MAX_INSTANCES",
    "DEFAULT_MAX_QUEUED",
    "CoreBudget",
    "Ticket",
    "Hyperchat",
    "plan_cores",
]

#: Cores handed to each model instance.
CORES_PER_INSTANCE = 2

#: Cores never handed to any instance. See the module docstring — this
#: is the difference between a busy server and an unresponsive one.
RESERVED_CORES = 1

#: However many cores a machine has, this many instances is plenty; past
#: it the instances are competing for memory bandwidth rather than
#: adding throughput.
MAX_INSTANCES = 8

#: Prompts allowed to pile up before new ones are refused. An unbounded
#: queue is a memory leak with a waiting list attached, and a prompt
#: accepted now and answered in an hour is worse than one refused now.
DEFAULT_MAX_QUEUED = 256


@dataclass(frozen=True)
class CoreBudget:
    """How many instances this machine can run, and why that many."""

    total_cores: int
    reserved: int
    cores_per_instance: int
    instances: int
    note: str = ""

    @property
    def cores_used(self) -> int:
        return self.instances * self.cores_per_instance

    @property
    def cores_free(self) -> int:
        return max(0, self.total_cores - self.cores_used)

    @property
    def pooling(self) -> bool:
        """True when this is a pool rather than a queue."""
        return self.instances > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cores": self.total_cores,
            "reserved": self.reserved,
            "cores_per_instance": self.cores_per_instance,
            "instances": self.instances,
            "cores_used": self.cores_used,
            "cores_free": self.cores_free,
            "mode": "pool" if self.pooling else "queue",
            "note": self.note,
        }


def plan_cores(
    total_cores: int | None = None,
    *,
    cores_per_instance: int = CORES_PER_INSTANCE,
    reserved: int = RESERVED_CORES,
    ceiling: int = MAX_INSTANCES,
    wanted: int = 0,
) -> CoreBudget:
    """What this machine can run.

    *wanted* lowers the answer, never raises it: an operator asking for
    six instances on a four-core box is asking for the machine to stop
    responding, and the honest reply is the number it can actually have.
    """
    cores = total_cores if total_cores is not None else (os.cpu_count() or 1)
    cores = max(1, int(cores))
    cores_per_instance = max(1, int(cores_per_instance))
    reserved = max(0, int(reserved))

    allocatable = max(0, cores - reserved)
    possible = allocatable // cores_per_instance

    if possible < 1:
        return CoreBudget(
            cores, reserved, cores_per_instance, 1,
            f"{cores} core(s) with {reserved} held back leaves "
            f"{allocatable} to share, which is under the "
            f"{cores_per_instance} an instance needs — running the queue "
            f"instead. It answers one prompt at a time, in order.",
        )

    instances = min(possible, max(1, int(ceiling)))
    if wanted > 0 and wanted < instances:
        instances = wanted
        note = f"{instances} instance(s), as asked for"
    elif wanted > instances:
        note = (
            f"{wanted} instance(s) asked for; {instances} is what "
            f"{cores} core(s) allow at {cores_per_instance} each with "
            f"{reserved} held back for the server itself"
        )
    else:
        note = (f"{instances} instance(s) — {cores} core(s), "
                f"{reserved} held back")
    return CoreBudget(cores, reserved, cores_per_instance, instances, note)


# ---------------------------------------------------------------------------


class Ticket:
    """One submitted prompt, and the answer when there is one."""

    __slots__ = ("id", "prompt", "kwargs", "submitted_at", "started_at",
                 "finished_at", "_state", "_result", "_error", "_done",
                 "_lock")

    def __init__(self, ticket_id: int, prompt: str, kwargs: dict[str, Any]):
        self.id = ticket_id
        self.prompt = prompt
        self.kwargs = kwargs
        self.submitted_at = time.time()
        self.started_at = 0.0
        self.finished_at = 0.0
        self._state = "queued"
        self._result = ""
        self._error = ""
        self._done = threading.Event()
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """queued | running | done | failed | cancelled"""
        return self._state

    @property
    def waited(self) -> float:
        end = self.started_at or time.time()
        return max(0.0, end - self.submitted_at)

    def cancel(self) -> bool:
        """Drop it if it has not started. ``True`` if it was dropped.

        A prompt already being generated is not cancelled here: the
        token loop is inside llama.cpp and stopping it half way leaves
        the instance in a state this module cannot reason about. What
        this prevents is the common case — somebody backing out of a
        screen while three prompts sit in front of theirs.
        """
        with self._lock:
            if self._state != "queued":
                return False
            self._state = "cancelled"
            self._error = "cancelled before it started"
        self._done.set()
        return True

    def wait(self, timeout: float | None = None) -> str:
        """Block for the answer. Raises on failure or cancellation."""
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"no answer after {timeout}s; the prompt is still "
                f"{self._state}"
            )
        if self._state == "done":
            return self._result
        raise RuntimeError(self._error or f"prompt {self._state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self._state,
            "waited_seconds": round(self.waited, 2),
            "submitted_at": self.submitted_at,
            "error": self._error,
        }

    # -- used by the workers --------------------------------------------

    def _claim(self) -> bool:
        with self._lock:
            if self._state != "queued":
                return False
            self._state = "running"
            self.started_at = time.time()
        return True

    def _finish(self, result: str) -> None:
        with self._lock:
            self._state = "done"
            self._result = result
            self.finished_at = time.time()
        self._done.set()

    def _fail(self, error: str) -> None:
        with self._lock:
            self._state = "failed"
            self._error = error
            self.finished_at = time.time()
        self._done.set()


class QueueFull(RuntimeError):
    """Too many prompts already waiting. Says how many and for how long."""


class Hyperchat:
    """N instances answering prompts, or one — the same object either way.

    *factory* is called once per worker and returns that worker's
    generator: ``(prompt, **kwargs) -> str``. Called in the worker
    thread and lazily, so N instances cost nothing until N prompts
    actually arrive at once — which on a server that is usually idle is
    the difference between holding the model N times and holding it once.
    """

    def __init__(
        self,
        factory: Callable[[], Callable[..., str]],
        *,
        budget: CoreBudget | None = None,
        instances: int = 0,
        max_queued: int = DEFAULT_MAX_QUEUED,
    ) -> None:
        self.budget = budget or plan_cores(wanted=instances)
        self.factory = factory
        self.max_queued = max(1, int(max_queued))
        self._queue: queue.Queue[Ticket | None] = queue.Queue()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._waiting = 0
        self._running = 0
        self._served = 0
        self._failed = 0
        self._closed = False
        self._workers: list[threading.Thread] = []
        for index in range(self.budget.instances):
            worker = threading.Thread(
                target=self._work, args=(index,),
                name=f"hyperchat-{index}", daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    # -- submitting ------------------------------------------------------

    def submit(self, prompt: str, **kwargs: Any) -> Ticket:
        if self._closed:
            raise RuntimeError("this hyperchat has been closed")
        with self._lock:
            if self._waiting >= self.max_queued:
                raise QueueFull(
                    f"{self._waiting} prompt(s) already waiting for "
                    f"{self.budget.instances} instance(s). Try again in a "
                    f"moment, or raise the instance count if the machine "
                    f"has the cores for it."
                )
            self._waiting += 1
        ticket = Ticket(next(self._ids), prompt, kwargs)
        self._queue.put(ticket)
        return ticket

    def ask(self, prompt: str, *, timeout: float | None = None,
            **kwargs: Any) -> str:
        """Submit and wait. The simple case, for callers with no UI."""
        return self.submit(prompt, **kwargs).wait(timeout)

    def position(self, ticket: Ticket) -> int:
        """How many prompts are ahead of this one. 0 means next or running.

        Approximate by construction — a worker may pick one up while
        this is counting — and it is what HyperLink shows instead of a
        spinner that does not move, which is worth far more than being
        exact.
        """
        if ticket.state != "queued":
            return 0
        ahead = 0
        for other in list(self._queue.queue):
            if other is None or other is ticket:
                break
            if other.state == "queued":
                ahead += 1
        return ahead

    # -- the workers -----------------------------------------------------

    def _work(self, index: int) -> None:
        generate: Callable[..., str] | None = None
        while True:
            ticket = self._queue.get()
            if ticket is None:
                self._queue.task_done()
                return
            with self._lock:
                self._waiting = max(0, self._waiting - 1)
            if not ticket._claim():
                # Cancelled while it was waiting. Nothing to do, and no
                # instance was built for it.
                self._queue.task_done()
                continue

            with self._lock:
                self._running += 1
            try:
                if generate is None:
                    # Built on first use, in this thread: N idle workers
                    # must not mean N copies of the model in memory.
                    generate = self.factory()
                answer = generate(ticket.prompt, **ticket.kwargs)
                ticket._finish(str(answer))
                with self._lock:
                    self._served += 1
            except Exception as exc:  # noqa: BLE001 - one prompt, not the pool
                logger.warning("hyperchat: worker %d failed: %s", index, exc)
                ticket._fail(f"{type(exc).__name__}: {exc}")
                with self._lock:
                    self._failed += 1
                # The instance is discarded rather than reused. A
                # generator that raised may have a half-consumed context
                # in it, and the next prompt would inherit it.
                generate = None
            finally:
                with self._lock:
                    self._running = max(0, self._running - 1)
                self._queue.task_done()

    # -- state -----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "pool" if self.budget.pooling else "queue",
                "instances": self.budget.instances,
                "waiting": self._waiting,
                "running": self._running,
                "served": self._served,
                "failed": self._failed,
                "max_queued": self.max_queued,
                "cores": self.budget.to_dict(),
            }

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop the workers. Prompts already running are waited for."""
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=timeout)

    def __enter__(self) -> Hyperchat:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
