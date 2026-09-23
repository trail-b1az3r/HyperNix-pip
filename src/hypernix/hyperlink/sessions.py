"""hyperlink.sessions — conversations that outlive the app that showed them.

A chat app on a phone that keeps its history in the phone has one
conversation. A chat app whose history lives on the PC that runs the
model has *the* conversation: start it on the laptop at a desk, carry
on from the phone on a train, and it is the same thread with the same
context. That is the difference this module exists to make, and it is
why history is stored server-side rather than synced between clients.

Design notes worth stating
--------------------------
**Messages are append-only.** Editing history in place is how a client
bug turns into a corrupted conversation with no way back. Deleting a
whole session is supported; rewriting a message is not.

**Attachments are referenced, never inlined.** A message holds
``attachment_ids``; the bytes live in :mod:`hyperlink.files`. A
conversation with four screenshots in it stays a few kilobytes of JSON,
which is what makes listing sessions fast and syncing cheap on cellular.

**The model that answered is recorded per message, not per session.**
People switch models mid-conversation — start on the small local one,
escalate to the 70B when it matters — and "which model said this" is
the first question asked when re-reading a thread later.

**Trimming is by token budget, not message count.** :meth:`ChatSessionStore.context_for`
walks backwards from the newest message until the budget is spent, and
always keeps the system prompt. A fixed "last 20 messages" either
overflows a small context window or wastes a large one.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

__all__ = ["ChatMessage", "ChatSession", "ChatSessionStore", "estimate_tokens"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    owner TEXT NOT NULL,
    device_id TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    touched_at REAL NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS hyperlink_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model_id TEXT NOT NULL DEFAULT '',
    attachments TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_hyperlink_messages_session
    ON hyperlink_messages (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_hyperlink_sessions_owner
    ON hyperlink_sessions (owner, updated_at);
"""

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


def estimate_tokens(text: str) -> int:
    """A cheap token estimate: ~4 characters per token, floor of 1.

    Deliberately not a real tokeniser. Loading one costs a dependency
    and tens of megabytes on a server whose job here is to decide how
    many old messages to include, and being 15% wrong about that
    changes nothing — the budget already carries a margin. The real
    token counts, once inference has run, are stored per message and
    are what usage accounting uses.
    """
    return max(1, (len(text) + 3) // 4)


@dataclass
class ChatMessage:
    message_id: str
    session_id: str
    seq: int
    role: str
    content: str
    created_at: float
    model_id: str = ""
    attachment_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "role": self.role,
            "content": self.content,
            "model_id": self.model_id,
            "attachment_ids": list(self.attachment_ids),
            "created_at": self.created_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass
class ChatSession:
    session_id: str
    title: str
    owner: str
    created_at: float
    #: When this conversation last *said* anything — a prompt sent or a
    #: reply received. Not when its title or model changed.
    updated_at: float
    #: When anything about the session last changed, metadata included.
    #: Kept apart from ``updated_at`` because the phone shows the latter
    #: as "updated 2m ago", and a rename is not the conversation moving.
    #: Renaming eleven old chats used to send all eleven to the top of
    #: the list looking like they had just replied.
    touched_at: float = 0.0
    device_id: str = ""
    model_id: str = ""
    backend: str = ""
    system_prompt: str = ""
    archived: bool = False
    message_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "owner": self.owner,
            "device_id": self.device_id,
            "model_id": self.model_id,
            "backend": self.backend,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "touched_at": self.touched_at or self.updated_at,
            "archived": self.archived,
            "message_count": self.message_count,
            "metadata": dict(self.metadata),
        }


class ChatSessionStore:
    """Sessions and their messages, in the T1 API's database."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)
        self._add_touched_at()

    def _add_touched_at(self) -> None:
        """Add ``touched_at`` to a database written before it existed.

        ``CREATE TABLE IF NOT EXISTS`` does not alter an existing table,
        so the column in ``_SCHEMA`` only reaches new databases. Seeded
        from ``updated_at``, which before this change carried both
        meanings — the best available answer for a row that predates the
        distinction, and never zero, which would sort every old session
        to the bottom of a list ordered by it.
        """
        with self._lock, self.backend.connect() as conn:
            columns = {
                row[1] for row in
                conn.execute("PRAGMA table_info(hyperlink_sessions)").fetchall()
            }
            if "touched_at" in columns:
                return
            conn.execute(
                "ALTER TABLE hyperlink_sessions ADD COLUMN "
                "touched_at REAL NOT NULL DEFAULT 0"
            )
            conn.execute(
                "UPDATE hyperlink_sessions SET touched_at = updated_at"
            )

    # -- sessions -----------------------------------------------------

    def create(
        self,
        *,
        owner: str,
        title: str = "",
        model_id: str = "",
        backend: str = "",
        system_prompt: str = "",
        device_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        now = time.time()
        session = ChatSession(
            session_id="chat_" + uuid.uuid4().hex[:20],
            title=title.strip() or "New chat",
            owner=owner,
            created_at=now,
            updated_at=now,
            touched_at=now,
            device_id=device_id,
            model_id=model_id,
            backend=backend,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """INSERT INTO hyperlink_sessions
                   (session_id, title, owner, device_id, model_id, backend, system_prompt,
                    created_at, updated_at, touched_at, archived, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    session.session_id, session.title, session.owner, session.device_id,
                    session.model_id, session.backend, session.system_prompt,
                    session.created_at, session.updated_at, session.touched_at,
                    json.dumps(session.metadata),
                ),
            )
        if system_prompt:
            self.append(
                session.session_id, role="system", content=system_prompt, owner=owner
            )
        return session

    def get(self, session_id: str, *, owner: str | None = None) -> ChatSession:
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise T1APIError(T1ErrorCode.NOT_FOUND, f"No chat session {session_id!r}")
            session = _session_from_row(row)
            if owner is not None and session.owner != owner:
                raise T1APIError(T1ErrorCode.NOT_FOUND, f"No chat session {session_id!r}")
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM hyperlink_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        session.message_count = int(count["n"]) if count is not None else 0
        return session

    def list_sessions(
        self,
        *,
        owner: str | None = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatSession]:
        clauses, params = [], []
        if owner is not None:
            clauses.append("s.owner = ?")
            params.append(owner)
        if not include_archived:
            clauses.append("s.archived = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([int(limit), int(offset)])
        # One query with a correlated count, not N+1: the session list is
        # the app's home screen and it is fetched on every foreground.
        with self.backend.connect() as conn:
            rows = conn.execute(
                f"""SELECT s.*,
                           (SELECT COUNT(*) FROM hyperlink_messages m
                             WHERE m.session_id = s.session_id) AS message_count
                    FROM hyperlink_sessions s
                    {where}
                    ORDER BY s.updated_at DESC
                    LIMIT ? OFFSET ?""",
                tuple(params),
            ).fetchall()
        sessions = []
        for row in rows:
            session = _session_from_row(row)
            session.message_count = int(row["message_count"] or 0)
            sessions.append(session)
        return sessions

    def update(
        self,
        session_id: str,
        *,
        owner: str | None = None,
        title: str | None = None,
        model_id: str | None = None,
        backend: str | None = None,
        system_prompt: str | None = None,
        archived: bool | None = None,
    ) -> ChatSession:
        session = self.get(session_id, owner=owner)
        fields, params = [], []
        if title is not None:
            fields.append("title = ?")
            params.append(title.strip() or session.title)
        if model_id is not None:
            fields.append("model_id = ?")
            params.append(model_id)
        if backend is not None:
            fields.append("backend = ?")
            params.append(backend)
        if system_prompt is not None:
            fields.append("system_prompt = ?")
            params.append(system_prompt)
        if archived is not None:
            fields.append("archived = ?")
            params.append(1 if archived else 0)
        if not fields:
            return session
        # `touched_at`, not `updated_at`. A rename, a model switch or an
        # archive is not the conversation saying anything, and the phone
        # shows `updated_at` as "updated 2m ago" — renaming eleven old
        # chats used to send all eleven to the top of the list looking
        # like they had all just replied.
        fields.append("touched_at = ?")
        params.extend([time.time(), session_id])
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                f"UPDATE hyperlink_sessions SET {', '.join(fields)} WHERE session_id = ?",
                tuple(params),
            )
        return self.get(session_id, owner=owner)

    def delete(self, session_id: str, *, owner: str | None = None) -> bool:
        self.get(session_id, owner=owner)     # ownership check, or raise
        with self._lock, self.backend.connect() as conn:
            conn.execute("DELETE FROM hyperlink_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM hyperlink_sessions WHERE session_id = ?", (session_id,))
        return True

    # -- messages -----------------------------------------------------

    def append(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        owner: str | None = None,
        model_id: str = "",
        attachment_ids: list[str] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Append one message and bump the session's ``updated_at``.

        The sequence number is allocated inside the same transaction as
        the insert. Two devices posting at once would otherwise both
        read ``MAX(seq)`` as 7 and both write 8, and the conversation
        would render in an order that depends on the reader.
        """
        if role not in VALID_ROLES:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"role must be one of {', '.join(sorted(VALID_ROLES))}; got {role!r}",
            )
        session = self.get(session_id, owner=owner)
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS top FROM hyperlink_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row["top"] or 0) + 1
            message = ChatMessage(
                message_id="msg_" + uuid.uuid4().hex[:20],
                session_id=session_id,
                seq=seq,
                role=role,
                content=content,
                created_at=now,
                model_id=model_id or session.model_id,
                attachment_ids=list(attachment_ids or []),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=metadata or {},
            )
            conn.execute(
                """INSERT INTO hyperlink_messages
                   (message_id, session_id, seq, role, content, model_id, attachments,
                    created_at, input_tokens, output_tokens, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.message_id, message.session_id, message.seq, message.role,
                    message.content, message.model_id, json.dumps(message.attachment_ids),
                    message.created_at, message.input_tokens, message.output_tokens,
                    json.dumps(message.metadata),
                ),
            )
            conn.execute(
                "UPDATE hyperlink_sessions SET updated_at = ?, touched_at = ? "
                "WHERE session_id = ?",
                (now, now, session_id),
            )
        return message

    def messages(
        self, session_id: str, *, owner: str | None = None, limit: int = 0, after_seq: int = 0
    ) -> list[ChatMessage]:
        """Messages in order. ``after_seq`` is the incremental-sync hook.

        With ``limit`` set, the *newest* messages are returned (still in
        ascending order) — a client opening a long thread wants the end
        of it, not the beginning.
        """
        self.get(session_id, owner=owner)
        with self.backend.connect() as conn:
            if limit:
                rows = conn.execute(
                    """SELECT * FROM (
                           SELECT * FROM hyperlink_messages
                            WHERE session_id = ? AND seq > ?
                            ORDER BY seq DESC LIMIT ?
                       ) ORDER BY seq ASC""",
                    (session_id, int(after_seq), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM hyperlink_messages WHERE session_id = ? AND seq > ? ORDER BY seq ASC",
                    (session_id, int(after_seq)),
                ).fetchall()
        return [_message_from_row(r) for r in rows]

    def edit_message(
        self,
        session_id: str,
        message_id: str,
        *,
        content: str,
        owner: str | None = None,
        truncate: bool = True,
    ) -> list[ChatMessage]:
        """Rewrite one message, and drop what came after it.

        The truncation is the point, not a side effect. Everything below
        an edited message was written *in reply to the old text*: leave
        it and the conversation reads as the model answering a question
        nobody asked, and — worse — that transcript is what gets sent as
        context on the next turn, so the model is being told it said
        things it did not.

        ``truncate=False`` exists for fixing a typo in the last message
        with nothing after it, where there is nothing to drop and
        re-running would be wasteful.

        Returns the messages that were removed, so a caller can say how
        much the edit cost rather than silently deleting eleven turns.
        """
        if not content.strip():
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                "An edited message cannot be empty. Delete it instead.",
            )
        self.get(session_id, owner=owner)
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_messages WHERE message_id = ? AND session_id = ?",
                (message_id, session_id),
            ).fetchone()
            if row is None:
                raise T1APIError(
                    T1ErrorCode.NOT_FOUND,
                    f"No message {message_id!r} in this conversation.",
                    http_status=404,
                )
            original = _message_from_row(row)
            if original.role != "user":
                # Editing what the model said turns the transcript into
                # fiction: the next turn would be built from words it
                # never produced, and every later reply would be
                # reasoning about them.
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    "Only your own messages can be edited. Editing the "
                    "assistant's would make the transcript a record of "
                    "something that did not happen.",
                )

            removed: list[ChatMessage] = []
            if truncate:
                later = conn.execute(
                    "SELECT * FROM hyperlink_messages WHERE session_id = ? AND seq > ? "
                    "ORDER BY seq ASC",
                    (session_id, original.seq),
                ).fetchall()
                removed = [_message_from_row(r) for r in later]
                conn.execute(
                    "DELETE FROM hyperlink_messages WHERE session_id = ? AND seq > ?",
                    (session_id, original.seq),
                )

            metadata = dict(original.metadata)
            # Kept rather than overwritten: somebody reading the
            # transcript later can see that this line is not what was
            # originally sent, which a silent rewrite would hide.
            metadata.setdefault("edited_from", original.content)
            metadata["edited_at"] = now
            conn.execute(
                "UPDATE hyperlink_messages SET content = ?, metadata = ? "
                "WHERE message_id = ?",
                (content, json.dumps(metadata), message_id),
            )
            conn.execute(
                "UPDATE hyperlink_sessions SET updated_at = ?, touched_at = ? "
                "WHERE session_id = ?",
                (now, now, session_id),
            )
        return removed

    def delete_message(
        self, session_id: str, message_id: str, *, owner: str | None = None
    ) -> bool:
        """Remove one message. Returns whether there was one.

        Deliberately does *not* truncate: deleting a message is usually
        about removing something that should not be stored — a pasted
        key, a name — and taking the rest of the conversation with it
        would make people keep the secret rather than lose the thread.
        """
        self.get(session_id, owner=owner)
        with self._lock, self.backend.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM hyperlink_messages WHERE message_id = ? AND session_id = ?",
                (message_id, session_id),
            )
            removed = cursor.rowcount > 0
            if removed:
                # `touched_at` only. Removing a pasted key from a chat
                # from last Tuesday is housekeeping, not the chat
                # replying, and sending it to the top of the list under
                # "updated just now" is the opposite of discreet.
                conn.execute(
                    "UPDATE hyperlink_sessions SET touched_at = ? "
                    "WHERE session_id = ?",
                    (time.time(), session_id),
                )
        return removed

    def mark_compacted(
        self,
        message_ids: list[str],
        *,
        session_id: str,
        owner: str | None = None,
        summary_id: str,
    ) -> int:
        """Record that these messages are now represented by a summary.

        A flag, not a delete. `messages()` keeps returning them so the
        person's transcript is unchanged; `context_for` skips them so the
        model sees the short version. Returns how many were marked.

        Idempotent: marking an already-marked message again is a no-op
        rather than an error, because a retried compaction is a retry and
        not a fault.
        """
        if not message_ids:
            return 0
        self.get(session_id, owner=owner)
        marked = 0
        with self._lock, self.backend.connect() as conn:
            for message_id in message_ids:
                row = conn.execute(
                    "SELECT metadata FROM hyperlink_messages "
                    "WHERE message_id = ? AND session_id = ?",
                    (message_id, session_id),
                ).fetchone()
                if row is None:
                    continue
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except (ValueError, TypeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                if metadata.get("compacted_by"):
                    continue
                metadata["compacted_by"] = summary_id
                conn.execute(
                    "UPDATE hyperlink_messages SET metadata = ? WHERE message_id = ?",
                    (json.dumps(metadata), message_id),
                )
                marked += 1
        return marked

    def context_for(
        self,
        session_id: str,
        *,
        owner: str | None = None,
        token_budget: int = 8000,
    ) -> list[ChatMessage]:
        """The tail of the conversation that fits in *token_budget*.

        The system message is always kept and its cost is charged first,
        so a long system prompt eats into history rather than being
        dropped — dropping it changes the assistant's behaviour
        mid-thread, which is worse than a shorter memory.
        """
        history = self.messages(session_id, owner=owner)
        # A compacted message has been replaced by a summary that is
        # itself in this history. Including both would mean paying for
        # the long version and the short one, which is the opposite of
        # what compaction was asked to do.
        #
        # Skipped here and not in `messages()`: the transcript a person
        # scrolls is still the one they had, and only the model sees the
        # shortened version. Deleting instead would make compaction
        # destructive over data the person may care about more than the
        # model does.
        history = [
            m for m in history
            if not (isinstance(m.metadata, dict) and m.metadata.get("compacted_by"))
        ]
        system = [m for m in history if m.role == "system"]
        rest = [m for m in history if m.role != "system"]

        spent = sum(estimate_tokens(m.content) for m in system)
        kept: list[ChatMessage] = []
        for message in reversed(rest):
            cost = estimate_tokens(message.content) + 4     # per-message envelope
            if spent + cost > token_budget and kept:
                break
            spent += cost
            kept.append(message)
        kept.reverse()
        return system + kept

    def autotitle(self, session_id: str, *, owner: str | None = None) -> ChatSession:
        """Name an untitled session after its first user message.

        Runs after the first exchange, not before: titling a session at
        creation means every session in the list is called "New chat"
        until someone renames it, which nobody does.
        """
        session = self.get(session_id, owner=owner)
        if session.title not in ("", "New chat"):
            return session
        for message in self.messages(session_id, owner=owner):
            if message.role != "user" or not message.content.strip():
                continue
            first_line = message.content.strip().splitlines()[0]
            title = first_line[:60].rstrip()
            if len(first_line) > 60:
                title += "…"
            return self.update(session_id, owner=owner, title=title)
        return session

    def stats(self, *, owner: str | None = None) -> dict[str, Any]:
        where = "WHERE owner = ?" if owner is not None else ""
        params = (owner,) if owner is not None else ()
        with self.backend.connect() as conn:
            sessions = conn.execute(
                f"SELECT COUNT(*) AS n FROM hyperlink_sessions {where}", params
            ).fetchone()
            if owner is not None:
                messages = conn.execute(
                    """SELECT COUNT(*) AS n,
                              COALESCE(SUM(input_tokens), 0) AS inp,
                              COALESCE(SUM(output_tokens), 0) AS outp
                         FROM hyperlink_messages
                        WHERE session_id IN (SELECT session_id FROM hyperlink_sessions WHERE owner = ?)""",
                    (owner,),
                ).fetchone()
            else:
                messages = conn.execute(
                    """SELECT COUNT(*) AS n,
                              COALESCE(SUM(input_tokens), 0) AS inp,
                              COALESCE(SUM(output_tokens), 0) AS outp
                         FROM hyperlink_messages"""
                ).fetchone()
        return {
            "sessions": int(sessions["n"]) if sessions else 0,
            "messages": int(messages["n"]) if messages else 0,
            "input_tokens": int(messages["inp"]) if messages else 0,
            "output_tokens": int(messages["outp"]) if messages else 0,
        }


def _column(row: Any, name: str, default: Any = None) -> Any:
    """One column, or *default* when the row does not have it.

    A row read through a connection opened before the migration ran —
    or from a query written against the older shape — has no
    ``touched_at``, and ``row["touched_at"]`` raises rather than
    returning None.
    """
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _session_from_row(row: Any) -> ChatSession:
    return ChatSession(
        session_id=row["session_id"],
        title=row["title"],
        owner=row["owner"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        touched_at=_column(row, "touched_at") or row["updated_at"],
        device_id=row["device_id"] or "",
        model_id=row["model_id"] or "",
        backend=row["backend"] or "",
        system_prompt=row["system_prompt"] or "",
        archived=bool(row["archived"]),
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _message_from_row(row: Any) -> ChatMessage:
    return ChatMessage(
        message_id=row["message_id"],
        session_id=row["session_id"],
        seq=int(row["seq"]),
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        model_id=row["model_id"] or "",
        attachment_ids=json.loads(row["attachments"] or "[]"),
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
        metadata=json.loads(row["metadata"] or "{}"),
    )
