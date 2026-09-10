"""hyperlink.notify — telling a phone something happened while it slept.

The gap: every interesting event on a HyperNix machine happens on a
timescale a phone is not awake for. A fine-tune runs for six hours. A
70B download takes forty minutes on a domestic line. A chat turn against
a large local model takes ninety seconds — long enough for iOS to
suspend the app. Today all of those complete into an app that is not
running, and the operator finds out by opening it and looking.

What this module is
-------------------
The parts of push notification that belong on the machine: which devices
want to hear about what, a durable queue of things to send, retry with
backoff, and the exact payload APNs expects. Delivery itself sits behind
:class:`Transport`, because the machine cannot send an APNs push without
an Apple team key, a signing key, and a route to
``api.push.apple.com`` — none of which ships with an open-source
package, and none of which a self-hosted server necessarily has.

So this is honest about its boundary: everything up to the HTTP/2 POST
is here, tested, and works offline. The POST is an operator-supplied
transport. :class:`RecordingTransport` makes the whole path testable
without Apple, and is also a perfectly good production choice for
someone who wants notifications in a log rather than on a lock screen.

Device tokens are credentials
-----------------------------
An APNs device token lets whoever holds it send notifications to that
device. It is therefore treated the way keys are treated everywhere else
in this codebase: stored because delivery needs it, and never returned
by an API, never written to a log, never included in a repr. Everything
outward-facing carries :attr:`Registration.fingerprint` instead — eight
hex characters of a SHA-256, enough for an operator to tell two devices
apart and useless for sending anything.

Rules, not a firehose
---------------------
A device subscribes to event kinds. The default set is deliberately
small — the things worth waking a screen for — because a notification
for every token generated is a notification nobody reads. An event with
no subscriber is dropped at enqueue time rather than queued and
discarded later, so the queue length means something.

The 4 KB wall
-------------
APNs rejects a payload over :data:`MAX_PAYLOAD_BYTES`, and a model reply
is frequently longer than that on its own. :func:`build_apns_payload`
trims the body to fit by *measuring the encoded payload*, not by
subtracting an estimate from the limit — see :func:`encode_payload` for
why the first attempt at this shipped payloads at twice their intended
size.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

__all__ = [
    "DEFAULT_EVENTS",
    "EventKind",
    "MAX_PAYLOAD_BYTES",
    "encode_payload",
    "NotificationStore",
    "Notifier",
    "PendingNotification",
    "RecordingTransport",
    "Registration",
    "Transport",
    "build_apns_payload",
    "fingerprint_token",
    "truncate_to_bytes",
]

# APNs' documented limit for a regular (non-VoIP) notification.
MAX_PAYLOAD_BYTES = 4096

# Attempts before a notification is given up on. APNs failures are
# mostly permanent (bad token, unregistered device) or brief; five
# attempts over the backoff schedule below spans about half an hour,
# which covers a transient outage without queueing forever.
MAX_ATTEMPTS = 5

# Seconds to wait before attempt N. Exponential, capped, and jitter-free
# on purpose: a single machine notifying its own owner's phones is not a
# thundering herd, and a deterministic schedule is one a test can assert.
BACKOFF_SECONDS = (0, 30, 120, 480, 1200)


class EventKind:
    """Things worth waking a screen for."""

    CHAT_REPLY = "chat.reply"
    TRAINING_DONE = "training.done"
    TRAINING_FAILED = "training.failed"
    DOWNLOAD_DONE = "download.done"
    DOWNLOAD_FAILED = "download.failed"
    QUANTIZE_DONE = "quantize.done"
    THERMAL_ALERT = "thermal.alert"
    SERVER_OFFLINE = "server.offline"
    DEVICE_PAIRED = "device.paired"
    JOB_DONE = "job.done"

    ALL = frozenset({
        CHAT_REPLY, TRAINING_DONE, TRAINING_FAILED, DOWNLOAD_DONE,
        DOWNLOAD_FAILED, QUANTIZE_DONE, THERMAL_ALERT, SERVER_OFFLINE,
        DEVICE_PAIRED, JOB_DONE,
    })


# What a device gets if it registers without saying. Long-running work
# and failures, not routine chatter: a reply that arrives while the app
# is open needs no notification, and one that arrives while it is closed
# is the case CHAT_REPLY exists for -- so it is in, but a phone that
# finds it noisy can drop it.
DEFAULT_EVENTS = frozenset({
    EventKind.CHAT_REPLY,
    EventKind.TRAINING_DONE,
    EventKind.TRAINING_FAILED,
    EventKind.DOWNLOAD_DONE,
    EventKind.DOWNLOAD_FAILED,
    EventKind.QUANTIZE_DONE,
    EventKind.THERMAL_ALERT,
})

# APNs tokens are hex. Rejecting anything else at the boundary keeps a
# pasted placeholder or a truncated copy out of the queue, where it
# would fail five times before anyone noticed.
_TOKEN_RE = re.compile(r"\A[0-9a-fA-F]{64,200}\Z")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_push_registrations (
    registration_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    token TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'ios',
    bundle_id TEXT NOT NULL DEFAULT '',
    environment TEXT NOT NULL DEFAULT 'production',
    events TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_delivery_at REAL NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS hyperlink_push_queue (
    notification_id TEXT PRIMARY KEY,
    registration_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    event TEXT NOT NULL,
    payload TEXT NOT NULL,
    collapse_id TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL,
    created_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_hyperlink_push_token
    ON hyperlink_push_registrations (device_id, fingerprint);
CREATE INDEX IF NOT EXISTS idx_hyperlink_push_owner
    ON hyperlink_push_registrations (owner, enabled);
CREATE INDEX IF NOT EXISTS idx_hyperlink_push_due
    ON hyperlink_push_queue (state, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_hyperlink_push_collapse
    ON hyperlink_push_queue (registration_id, collapse_id, state);
"""


def fingerprint_token(token: str) -> str:
    """Eight hex characters identifying a token without revealing it.

    Enough for an operator to tell two of their own phones apart in a
    device list, and no use at all for sending a notification — which
    is the whole point, since the list is served over an API.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def truncate_to_bytes(text: str, limit: int) -> str:
    """Trim *text* so its UTF-8 encoding fits in *limit* bytes.

    Character counts are the wrong unit for this: the APNs limit is on
    encoded bytes, so "こんにちは" costs 15 and "hello" costs 5. Cutting
    by characters overshoots on one and wastes room on the other.

    Cuts on a character boundary — never mid-sequence, which would
    produce invalid UTF-8 — and appends an ellipsis when it trims,
    counting the ellipsis against the limit rather than adding it after
    the measurement.
    """
    if limit <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    ellipsis = "…"
    room = limit - len(ellipsis.encode("utf-8"))
    if room <= 0:
        # No space even for the marker; return whatever whole
        # characters fit, which may be nothing.
        return encoded[:limit].decode("utf-8", errors="ignore")
    return encoded[:room].decode("utf-8", errors="ignore") + ellipsis


@dataclass
class Registration:
    """One device's willingness to be notified.

    ``token`` is present because delivery needs it and is excluded from
    :meth:`to_dict`. Read it only to hand to a transport.
    """

    registration_id: str
    device_id: str
    owner: str
    token: str
    fingerprint: str
    platform: str = "ios"
    bundle_id: str = ""
    environment: str = "production"
    events: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_EVENTS))
    enabled: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0
    last_delivery_at: float = 0.0
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Everything except the token."""
        return {
            "registration_id": self.registration_id,
            "device_id": self.device_id,
            "fingerprint": self.fingerprint,
            "platform": self.platform,
            "bundle_id": self.bundle_id,
            "environment": self.environment,
            "events": sorted(self.events),
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_delivery_at": self.last_delivery_at,
            "failure_count": self.failure_count,
        }

    def __repr__(self) -> str:
        """Deliberately token-free.

        A dataclass repr would print the token into any traceback,
        debugger session or log line that touched a Registration.
        """
        return (
            f"Registration(registration_id={self.registration_id!r}, "
            f"device_id={self.device_id!r}, fingerprint={self.fingerprint!r}, "
            f"platform={self.platform!r}, enabled={self.enabled!r})"
        )

    def wants(self, event: str) -> bool:
        return self.enabled and event in self.events


@dataclass
class PendingNotification:
    notification_id: str
    registration_id: str
    owner: str
    event: str
    payload: dict[str, Any]
    collapse_id: str = ""
    attempts: int = 0
    next_attempt_at: float = 0.0
    created_at: float = 0.0
    state: str = "pending"
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "registration_id": self.registration_id,
            "event": self.event,
            "payload": self.payload,
            "collapse_id": self.collapse_id,
            "attempts": self.attempts,
            "next_attempt_at": self.next_attempt_at,
            "created_at": self.created_at,
            "state": self.state,
            "last_error": self.last_error,
        }


def encode_payload(payload: dict[str, Any]) -> bytes:
    """The canonical wire form of a notification payload.

    Every size decision in this module measures *this*, and a transport
    must send this, or the two disagree and APNs rejects what the queue
    thought was fine.

    ``ensure_ascii=False`` is the load-bearing argument. The default,
    ``True``, escapes every non-ASCII character to ``\\uXXXX`` — six
    ASCII bytes for something UTF-8 stores in three. A body trimmed to
    4036 UTF-8 bytes of Japanese therefore serialised to about 8000 and
    was refused, which is exactly what happened the first time this was
    run against real text. Sending UTF-8 also gets roughly twice as
    much of a non-Latin reply onto the lock screen.
    """
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_apns_payload(
    *,
    title: str,
    body: str,
    event: str,
    data: dict[str, Any] | None = None,
    badge: int | None = None,
    sound: str | None = "default",
    thread_id: str = "",
    interruption_level: str | None = None,
) -> dict[str, Any]:
    """An APNs payload that fits, with the body trimmed if it must.

    The fit is *measured*, not estimated. An earlier version subtracted
    the skeleton's length from the limit and trimmed the body to the
    difference, which is wrong twice over: JSON escaping inflates
    quotes, backslashes and control characters, and — before
    :func:`encode_payload` existed — non-ASCII doubled. So the body is
    trimmed, the whole payload encoded, and the trim tightened until
    the real bytes fit. It converges in two or three passes and is
    exact for any input, including a body made entirely of quotes.

    A title that is itself over the limit cannot be trimmed away by the
    body, so it is trimmed first, to a quarter of the budget: a
    notification is a glance, and a 1 KB title is not one.
    """
    safe_title = truncate_to_bytes(title, MAX_PAYLOAD_BYTES // 4)
    aps: dict[str, Any] = {"alert": {"title": safe_title, "body": ""}}
    if badge is not None:
        aps["badge"] = int(badge)
    if sound:
        aps["sound"] = sound
    if thread_id:
        aps["thread-id"] = thread_id
    if interruption_level:
        # "passive", "active", "time-sensitive", "critical" -- critical
        # additionally needs an Apple entitlement, which is the
        # operator's problem and not validated here.
        aps["interruption-level"] = interruption_level

    payload: dict[str, Any] = {"aps": aps, "hnx": {"event": event}}
    if data:
        payload["hnx"]["data"] = data

    # Binary search the body budget for the largest that still fits.
    #
    # The obvious loop -- trim, measure, subtract the overflow, repeat --
    # over-corrects badly. A body of 8000 double quotes escapes to two
    # bytes each, so the first overflow is about as large as the budget
    # itself, the subtraction drives the budget to zero, and a reply
    # full of code arrives with no body at all. That is what this did
    # before being measured against a body of quotes: 143 bytes of
    # payload and nothing to read.
    #
    # Searching costs about a dozen encodes of a 4 KB dict and gives
    # the exact maximum for any escaping behaviour, quotes, backslashes,
    # control characters and astral-plane emoji included.
    def fits(budget: int) -> bool:
        aps["alert"]["body"] = truncate_to_bytes(body, budget)
        return len(encode_payload(payload)) <= MAX_PAYLOAD_BYTES

    if fits(len(body.encode("utf-8"))):
        return payload
    low, high = 0, MAX_PAYLOAD_BYTES
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if fits(mid):
            best, low = mid, mid + 1
        else:
            high = mid - 1
    # `fits` left the body at whatever the last probe was, which may be
    # a budget that does not fit. Set it back to the best found.
    aps["alert"]["body"] = truncate_to_bytes(body, best)
    return payload


class Transport(Protocol):
    """Where a built notification actually goes.

    Return normally to mean delivered. Raise to mean "try again", except
    for :class:`PermanentDeliveryError`, which means the token is dead
    and retrying is pointless.
    """

    def send(self, *, token: str, payload: dict[str, Any], collapse_id: str) -> None:
        ...


class PermanentDeliveryError(Exception):
    """The token will never work again — stop trying.

    APNs says this with 410 Unregistered or 400 BadDeviceToken. Treating
    it as retryable burns five attempts and leaves a dead registration
    in place collecting failures.
    """


class RecordingTransport:
    """Keeps what it was asked to send, instead of sending it.

    The test double, and a usable production choice for an operator who
    wants notifications in a log rather than on a lock screen. Records
    the fingerprint rather than the token, so a test fixture cannot
    become a place tokens are written down.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail_next: int = 0
        self.permanent_failures: set[str] = set()

    def send(self, *, token: str, payload: dict[str, Any], collapse_id: str) -> None:
        if token in self.permanent_failures:
            raise PermanentDeliveryError(f"token {fingerprint_token(token)} unregistered")
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("transport temporarily unavailable")
        self.sent.append({
            "fingerprint": fingerprint_token(token),
            "payload": payload,
            "collapse_id": collapse_id,
        })


class NotificationStore:
    """Registrations and the outbound queue."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # -- registrations ------------------------------------------------

    def register(
        self,
        *,
        device_id: str,
        owner: str,
        token: str,
        platform: str = "ios",
        bundle_id: str = "",
        environment: str = "production",
        events: Iterable[str] | None = None,
    ) -> Registration:
        """Register (or re-register) a device for notifications.

        Re-registering the same token for the same device updates it
        rather than adding a second row: iOS hands the app a token on
        every launch, and an app that launches daily would otherwise
        accumulate a registration a day and send every notification
        that many times.
        """
        if not device_id:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "device_id is required")
        if not owner:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "owner is required")
        if not _TOKEN_RE.match(token or ""):
            # The message says nothing about the value -- an error
            # string is a place tokens leak.
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                "token must be 64-200 hexadecimal characters",
            )
        if environment not in ("production", "sandbox"):
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                "environment must be 'production' or 'sandbox'",
            )
        wanted = frozenset(events) if events is not None else frozenset(DEFAULT_EVENTS)
        unknown = wanted - EventKind.ALL
        if unknown:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Unknown event kinds {sorted(unknown)}; expected a subset of "
                f"{sorted(EventKind.ALL)}",
            )

        now = time.time()
        finger = fingerprint_token(token)
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                """SELECT registration_id, created_at FROM hyperlink_push_registrations
                   WHERE device_id = ? AND fingerprint = ?""",
                (device_id, finger),
            ).fetchone()
            if row is not None:
                reg_id = row["registration_id"]
                created = float(row["created_at"])
                conn.execute(
                    """UPDATE hyperlink_push_registrations
                       SET token = ?, platform = ?, bundle_id = ?, environment = ?,
                           events = ?, enabled = 1, updated_at = ?, failure_count = 0
                       WHERE registration_id = ?""",
                    (
                        token, platform, bundle_id, environment,
                        json.dumps(sorted(wanted)), now, reg_id,
                    ),
                )
            else:
                reg_id = "push_" + uuid.uuid4().hex[:20]
                created = now
                conn.execute(
                    """INSERT INTO hyperlink_push_registrations
                       (registration_id, device_id, owner, token, fingerprint,
                        platform, bundle_id, environment, events, enabled,
                        created_at, updated_at, last_delivery_at, failure_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 0, 0)""",
                    (
                        reg_id, device_id, owner, token, finger, platform,
                        bundle_id, environment, json.dumps(sorted(wanted)),
                        created, now,
                    ),
                )
        return Registration(
            registration_id=reg_id, device_id=device_id, owner=owner, token=token,
            fingerprint=finger, platform=platform, bundle_id=bundle_id,
            environment=environment, events=wanted, enabled=True,
            created_at=created, updated_at=now,
        )

    def list_registrations(
        self, *, owner: str, include_disabled: bool = False
    ) -> list[Registration]:
        sql = "SELECT * FROM hyperlink_push_registrations WHERE owner = ?"
        params: list[Any] = [owner]
        if not include_disabled:
            sql += " AND enabled = 1"
        sql += " ORDER BY created_at ASC"
        with self.backend.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_registration_from_row(r) for r in rows]

    def get_registration(self, registration_id: str) -> Registration:
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_push_registrations WHERE registration_id = ?",
                (registration_id,),
            ).fetchone()
        if row is None:
            raise T1APIError(
                T1ErrorCode.NOT_FOUND, f"No push registration {registration_id!r}"
            )
        return _registration_from_row(row)

    def set_events(self, registration_id: str, events: Iterable[str]) -> Registration:
        wanted = frozenset(events)
        unknown = wanted - EventKind.ALL
        if unknown:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR, f"Unknown event kinds {sorted(unknown)}"
            )
        with self._lock, self.backend.connect() as conn:
            cur = conn.execute(
                """UPDATE hyperlink_push_registrations
                   SET events = ?, updated_at = ? WHERE registration_id = ?""",
                (json.dumps(sorted(wanted)), time.time(), registration_id),
            )
            if not cur.rowcount:
                raise T1APIError(
                    T1ErrorCode.NOT_FOUND,
                    f"No push registration {registration_id!r}",
                )
        return self.get_registration(registration_id)

    def unregister(self, registration_id: str) -> None:
        """Remove a registration and anything still queued for it.

        Leaving the queue behind would keep trying to reach a device
        whose owner has just said to stop.
        """
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "DELETE FROM hyperlink_push_queue WHERE registration_id = ?",
                (registration_id,),
            )
            cur = conn.execute(
                "DELETE FROM hyperlink_push_registrations WHERE registration_id = ?",
                (registration_id,),
            )
            if not cur.rowcount:
                raise T1APIError(
                    T1ErrorCode.NOT_FOUND,
                    f"No push registration {registration_id!r}",
                )

    def disable(self, registration_id: str, *, reason: str = "") -> None:
        """Stop using a registration without forgetting it.

        What a permanent APNs failure leads to: the token is dead, but
        the row is worth keeping so an operator can see *why* their
        phone went quiet instead of finding an empty list.
        """
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """UPDATE hyperlink_push_registrations
                   SET enabled = 0, updated_at = ? WHERE registration_id = ?""",
                (time.time(), registration_id),
            )
            conn.execute(
                """UPDATE hyperlink_push_queue SET state = 'dropped', last_error = ?
                   WHERE registration_id = ? AND state = 'pending'""",
                (reason or "registration disabled", registration_id),
            )

    # -- the queue ----------------------------------------------------

    def enqueue(
        self,
        *,
        owner: str,
        event: str,
        payload: dict[str, Any],
        collapse_id: str = "",
        registration_ids: Iterable[str] | None = None,
        now: float | None = None,
    ) -> list[PendingNotification]:
        """Queue one event for every registration that wants it.

        Returns what was queued, which is empty when nobody subscribed
        — dropped here rather than queued and discarded at send time,
        so a queue length of zero means "nothing to say" and not
        "plenty to say, all of it unwanted".
        """
        if event not in EventKind.ALL:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Unknown event kind {event!r}; expected one of "
                f"{sorted(EventKind.ALL)}",
            )
        moment = time.time() if now is None else now
        targets = [
            r for r in self.list_registrations(owner=owner)
            if r.wants(event)
            and (registration_ids is None or r.registration_id in set(registration_ids))
        ]
        if not targets:
            return []

        wire = encode_payload(payload)
        encoded = wire.decode("utf-8")
        size = len(wire)
        if size > MAX_PAYLOAD_BYTES:
            raise T1APIError(
                T1ErrorCode.PAYLOAD_TOO_LARGE,
                f"Notification payload is {size} bytes; APNs allows "
                f"{MAX_PAYLOAD_BYTES}. Build it with build_apns_payload(), "
                f"which trims the body to fit.",
            )

        out: list[PendingNotification] = []
        with self._lock, self.backend.connect() as conn:
            for reg in targets:
                # A collapse id replaces an undelivered notification for
                # the same thing rather than stacking another on top:
                # download progress at 40%, 60% and 80% should be one
                # notification the phone sees once, not three.
                if collapse_id:
                    conn.execute(
                        """UPDATE hyperlink_push_queue SET state = 'superseded'
                           WHERE registration_id = ? AND collapse_id = ?
                             AND state = 'pending'""",
                        (reg.registration_id, collapse_id),
                    )
                note = PendingNotification(
                    notification_id="note_" + uuid.uuid4().hex[:20],
                    registration_id=reg.registration_id,
                    owner=owner,
                    event=event,
                    payload=payload,
                    collapse_id=collapse_id,
                    next_attempt_at=moment,
                    created_at=moment,
                )
                conn.execute(
                    """INSERT INTO hyperlink_push_queue
                       (notification_id, registration_id, owner, event, payload,
                        collapse_id, attempts, next_attempt_at, created_at,
                        state, last_error)
                       VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 'pending', '')""",
                    (
                        note.notification_id, note.registration_id, owner, event,
                        encoded, collapse_id, note.next_attempt_at, note.created_at,
                    ),
                )
                out.append(note)
        return out

    def due(self, *, limit: int = 50, now: float | None = None) -> list[PendingNotification]:
        moment = time.time() if now is None else now
        with self.backend.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM hyperlink_push_queue
                   WHERE state = 'pending' AND next_attempt_at <= ?
                   ORDER BY next_attempt_at ASC, created_at ASC LIMIT ?""",
                (moment, max(1, int(limit))),
            ).fetchall()
        return [_notification_from_row(r) for r in rows]

    def pending_count(self, *, owner: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM hyperlink_push_queue WHERE state = 'pending'"
        params: tuple[Any, ...] = ()
        if owner:
            sql += " AND owner = ?"
            params = (owner,)
        with self.backend.connect() as conn:
            return int(conn.execute(sql, params).fetchone()["n"])

    def mark_sent(self, notification_id: str, *, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """UPDATE hyperlink_push_queue SET state = 'sent', attempts = attempts + 1
                   WHERE notification_id = ?""",
                (notification_id,),
            )
            conn.execute(
                """UPDATE hyperlink_push_registrations
                   SET last_delivery_at = ?, failure_count = 0
                   WHERE registration_id = (
                       SELECT registration_id FROM hyperlink_push_queue
                       WHERE notification_id = ?)""",
                (moment, notification_id),
            )

    def mark_failed(
        self,
        notification_id: str,
        *,
        error: str,
        permanent: bool = False,
        now: float | None = None,
    ) -> str:
        """Record a failed attempt and return the resulting state.

        ``'pending'`` means it will be retried; ``'failed'`` means it
        will not, either because the error was permanent or because the
        attempts ran out.
        """
        moment = time.time() if now is None else now
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT attempts, registration_id FROM hyperlink_push_queue "
                "WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
            if row is None:
                raise T1APIError(
                    T1ErrorCode.NOT_FOUND, f"No queued notification {notification_id!r}"
                )
            attempts = int(row["attempts"]) + 1
            exhausted = permanent or attempts >= MAX_ATTEMPTS
            state = "failed" if exhausted else "pending"
            delay = BACKOFF_SECONDS[min(attempts, len(BACKOFF_SECONDS) - 1)]
            conn.execute(
                """UPDATE hyperlink_push_queue
                   SET attempts = ?, state = ?, last_error = ?, next_attempt_at = ?
                   WHERE notification_id = ?""",
                (attempts, state, error[:500], moment + delay, notification_id),
            )
            conn.execute(
                """UPDATE hyperlink_push_registrations
                   SET failure_count = failure_count + 1
                   WHERE registration_id = ?""",
                (row["registration_id"],),
            )
        return state

    def purge(self, *, older_than: float = 7 * 24 * 3600, now: float | None = None) -> int:
        """Drop finished queue rows. Returns how many went."""
        moment = time.time() if now is None else now
        with self._lock, self.backend.connect() as conn:
            cur = conn.execute(
                """DELETE FROM hyperlink_push_queue
                   WHERE state != 'pending' AND created_at < ?""",
                (moment - older_than,),
            )
            return int(cur.rowcount or 0)


class Notifier:
    """Drains the queue through a transport.

    Kept separate from the store so the queue can be inspected, tested
    and reasoned about without a transport in the picture, and so an
    operator can run the drain wherever they like — a thread, a cron
    job, a systemd timer.
    """

    def __init__(self, store: NotificationStore, transport: Transport) -> None:
        self.store = store
        self.transport = transport

    def drain(self, *, limit: int = 50, now: float | None = None) -> dict[str, int]:
        """Send everything due. Returns a count per outcome."""
        moment = time.time() if now is None else now
        result = {"sent": 0, "retry": 0, "failed": 0, "dropped": 0}
        for note in self.store.due(limit=limit, now=moment):
            try:
                reg = self.store.get_registration(note.registration_id)
            except T1APIError:
                # The registration went away between enqueue and drain.
                self.store.mark_failed(
                    note.notification_id, error="registration gone",
                    permanent=True, now=moment,
                )
                result["dropped"] += 1
                continue
            if not reg.enabled:
                self.store.mark_failed(
                    note.notification_id, error="registration disabled",
                    permanent=True, now=moment,
                )
                result["dropped"] += 1
                continue
            try:
                self.transport.send(
                    token=reg.token, payload=note.payload,
                    collapse_id=note.collapse_id,
                )
            except PermanentDeliveryError as exc:
                self.store.mark_failed(
                    note.notification_id, error=str(exc), permanent=True, now=moment,
                )
                # A dead token stays dead; disabling it stops the next
                # hundred notifications from queueing against it.
                self.store.disable(reg.registration_id, reason=str(exc))
                result["failed"] += 1
            except Exception as exc:  # noqa: BLE001 - any transport error is retryable
                state = self.store.mark_failed(
                    note.notification_id, error=str(exc), now=moment,
                )
                result["retry" if state == "pending" else "failed"] += 1
            else:
                self.store.mark_sent(note.notification_id, now=moment)
                result["sent"] += 1
        return result


def _registration_from_row(row: Any) -> Registration:
    return Registration(
        registration_id=row["registration_id"],
        device_id=row["device_id"],
        owner=row["owner"],
        token=row["token"],
        fingerprint=row["fingerprint"],
        platform=row["platform"],
        bundle_id=row["bundle_id"] or "",
        environment=row["environment"],
        events=frozenset(json.loads(row["events"] or "[]")),
        enabled=bool(row["enabled"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_delivery_at=float(row["last_delivery_at"] or 0.0),
        failure_count=int(row["failure_count"] or 0),
    )


def _notification_from_row(row: Any) -> PendingNotification:
    return PendingNotification(
        notification_id=row["notification_id"],
        registration_id=row["registration_id"],
        owner=row["owner"],
        event=row["event"],
        payload=json.loads(row["payload"] or "{}"),
        collapse_id=row["collapse_id"] or "",
        attempts=int(row["attempts"] or 0),
        next_attempt_at=float(row["next_attempt_at"] or 0.0),
        created_at=float(row["created_at"]),
        state=row["state"],
        last_error=row["last_error"] or "",
    )
