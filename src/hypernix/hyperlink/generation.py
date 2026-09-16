"""hypernix.hyperlink.generation — stopping a generation that is running.

Why Stop did not stop
---------------------
The app's Stop button cancelled the *client's* task. Nothing told the
server, and nothing told LM Studio, so the model carried on producing
tokens into a socket nobody was reading — burning the GPU for a minute
and a half, and on a metered deployment billing for output the person had
explicitly asked not to have.

Closing the connection does not fix it either, which is the part worth
being precise about. ``LMStudioBridge.chat_stream`` closes its upstream
response in a ``finally``, and that ``finally`` runs when the generator
is closed. But a **sync** generator handed to ``StreamingResponse`` is
iterated with ``iterate_in_threadpool``: on client disconnect Starlette
cancels the task that was awaiting the thread, and the thread itself
keeps running ``next()`` to completion because a Python thread blocked in
a socket read cannot be interrupted. The cleanup is correct and
unreachable.

So cancellation has to be cooperative: something the generator *checks*,
between chunks, on the thread it is already running on. That is all this
module is — a registry of in-flight generations, each with an event, and
the small amount of bookkeeping needed to find the right one to set.

Who may cancel what
-------------------
Owner-scoped, always. ``cancel`` takes an owner and will not touch a
generation belonging to anyone else, because the alternative is one
phone able to stop another person's answer by guessing a session id.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ActiveGeneration",
    "GenerationRegistry",
    "CancelledByClient",
]


class CancelledByClient(Exception):
    """Raised inside a stream when the owner asked it to stop.

    Not an error: the partial reply is still persisted and still worth
    having. It exists so the two ways a stream can end early — the model
    stopped, the person stopped it — are distinguishable at the point
    that decides what to write down.
    """


@dataclass
class ActiveGeneration:
    """One generation in flight, and the switch that ends it."""

    generation_id: str
    session_id: str
    owner: str
    started_at: float
    #: Set by :meth:`GenerationRegistry.cancel`. The streaming loop reads
    #: it between chunks — an Event rather than a bool because the reader
    #: and the writer are different threads.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    #: Filled in when something asks for this to stop, for the audit line
    #: and for the frame the app receives.
    cancelled_by: str = ""

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "age_seconds": round(self.age_seconds, 2),
            "cancelled": self.cancelled,
        }


class GenerationRegistry:
    """Every generation this process is currently running.

    In-process on purpose. A generation lives inside one worker's
    streaming response, so the worker holding it is the only one that can
    stop it; putting this in the database would let a second worker
    record a cancellation that nothing acts on, which is the current bug
    with extra steps. A deployment behind several workers needs sticky
    routing for the stream, which it already needs for the stream itself.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, ActiveGeneration] = {}

    def begin(self, session_id: str, owner: str) -> ActiveGeneration:
        record = ActiveGeneration(
            generation_id=uuid.uuid4().hex,
            session_id=session_id,
            owner=owner,
            started_at=time.time(),
        )
        with self._lock:
            self._active[record.generation_id] = record
        return record

    def finish(self, generation_id: str) -> None:
        """Called from the stream's ``finally``, whatever ended it."""
        with self._lock:
            self._active.pop(generation_id, None)

    def get(self, generation_id: str, *, owner: str) -> ActiveGeneration | None:
        with self._lock:
            record = self._active.get(generation_id)
        if record is None or record.owner != owner:
            return None
        return record

    def active(self, *, owner: str, session_id: str | None = None) -> list[ActiveGeneration]:
        with self._lock:
            records = list(self._active.values())
        return [
            r for r in records
            if r.owner == owner and (session_id is None or r.session_id == session_id)
        ]

    def cancel(
        self,
        *,
        owner: str,
        session_id: str | None = None,
        generation_id: str | None = None,
        by: str = "client",
    ) -> list[str]:
        """Stop matching generations. Returns the ids actually stopped.

        An empty list is the ordinary answer, not a failure: the model
        finished a quarter-second before the Stop reached the server, and
        the person got what they wanted either way. Raising there would
        turn a successful stop into an error toast.

        Never matches on session alone across owners — see the module
        docstring. A caller that supplies neither a session nor a
        generation cancels everything *of their own*, which is what the
        app does when it goes to the background.
        """
        stopped: list[str] = []
        with self._lock:
            for record in self._active.values():
                if record.owner != owner:
                    continue
                if generation_id is not None and record.generation_id != generation_id:
                    continue
                if session_id is not None and record.session_id != session_id:
                    continue
                if not record.cancel_event.is_set():
                    record.cancel_event.set()
                    record.cancelled_by = by
                    stopped.append(record.generation_id)
        if stopped:
            logger.info(
                "hyperlink.generation: %d generation(s) cancelled by %s", len(stopped), by
            )
        return stopped

    def cancel_all(self, *, by: str = "shutdown") -> int:
        """Stop everything. For shutdown, where ownership is not the point."""
        with self._lock:
            records = list(self._active.values())
        count = 0
        for record in records:
            if not record.cancel_event.is_set():
                record.cancel_event.set()
                record.cancelled_by = by
                count += 1
        return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._active)
