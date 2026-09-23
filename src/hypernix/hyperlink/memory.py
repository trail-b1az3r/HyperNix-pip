"""hyperlink.memory — what the assistant is supposed to remember.

A chat session is a transcript, and a transcript is the wrong shape for
"I use metric", "my GPU is a 1080", "call me Mason". Those are true
across every conversation and they fall out of the context window the
moment the session gets long — which is exactly when a model most needs
them.

So memories live outside sessions: small, durable, owner-scoped facts
that get injected into the prompt rather than carried in the history.

Written by two different things
-------------------------------
A person types one. Or the assistant notices one and writes it itself,
which is the interesting half and the one that needs guard rails:

* **Every memory records its source.** ``manual`` or ``auto``, and
  ``auto`` also records the session it was learned in. A person deleting
  "you dislike Python" needs to be able to see where the model got that
  idea.
* **Auto-memory has a budget.** :data:`AUTO_LIMIT` per owner, oldest
  evicted first. Without a cap a model that writes a memory per turn
  produces a prompt prefix that is longer than the conversation, and the
  cost lands on the person who never asked for any of it.
* **A near-duplicate updates rather than appends.** "User's name is
  Mason" written forty times is forty memories that say one thing and
  cost forty times as much to carry.

What is deliberately simple
---------------------------
There is no embedding search here and no relevance model. Memories are
short, there are at most a few dozen, and the whole set fits in a prompt
prefix — so "which memories are relevant" is a question this shape does
not have to answer, and answering it badly would be worse than not
answering it. Retrieval is by owner, ordered by pinned-then-recent.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

__all__ = [
    "Memory",
    "MemoryStore",
    "AUTO_LIMIT",
    "MAX_CONTENT",
    "SOURCES",
]

#: How many model-written memories one owner may accumulate. Oldest
#: unpinned goes first. A person's own memories are not counted here —
#: they asked for those.
AUTO_LIMIT = 64

#: Longest single memory. A memory is a fact, not a document; anything
#: past this is a conversation that wanted to be a session.
MAX_CONTENT = 2000

SOURCES = ("manual", "auto")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_memories (
    memory_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'manual',
    session_id TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_hyperlink_memories_owner
    ON hyperlink_memories (owner, pinned DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_hyperlink_memories_source
    ON hyperlink_memories (owner, source, updated_at);
"""


@dataclass
class Memory:
    memory_id: str
    owner: str
    content: str
    category: str = ""
    #: "manual" (a person wrote it) or "auto" (the model did).
    source: str = "manual"
    #: Where an auto memory was learned. Empty for a manual one. What
    #: lets somebody find out why the assistant believes something.
    session_id: str = ""
    #: Pinned memories are never evicted by the auto budget and sort
    #: first into the prompt.
    pinned: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "category": self.category,
            "source": self.source,
            "session_id": self.session_id,
            "pinned": self.pinned,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


def _normalise(text: str) -> str:
    """What counts as "the same memory said again".

    Case and punctuation dropped: a model writing "User's name is Mason."
    on Monday and "user's name is mason" on Tuesday has not learned
    anything new, and storing both costs twice as much prompt for ever.
    """
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


class MemoryStore:
    """Durable per-owner facts, outside any session."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # -- writing --------------------------------------------------------

    def create(
        self,
        *,
        owner: str,
        content: str,
        category: str = "",
        source: str = "manual",
        session_id: str = "",
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Remember something. A near-duplicate updates instead.

        Returning the existing memory rather than raising on a duplicate
        is deliberate: the caller asked for this fact to be remembered,
        and after this call it is. An error would make every auto-memory
        write a thing the model has to handle.
        """
        content = (content or "").strip()
        if not content:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "A memory needs content.")
        if len(content) > MAX_CONTENT:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"A memory is a fact, not a document: {len(content)} characters, "
                f"limit {MAX_CONTENT}.",
            )
        if source not in SOURCES:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Unknown memory source {source!r}; expected one of {list(SOURCES)}.",
            )
        if not owner:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "owner is required")

        now = time.time()
        with self._lock:
            existing = self._find_duplicate(owner, content)
            if existing is not None:
                return self._update(
                    existing.memory_id,
                    owner=owner,
                    content=content,
                    category=category or existing.category,
                    pinned=pinned or existing.pinned,
                    metadata=metadata,
                    now=now,
                )

            record = Memory(
                memory_id=f"mem_{uuid.uuid4().hex[:20]}",
                owner=owner,
                content=content,
                category=category,
                source=source,
                session_id=session_id,
                pinned=pinned,
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}),
            )
            with self.backend.connect() as conn:
                conn.execute(
                    "INSERT INTO hyperlink_memories (memory_id, owner, content, "
                    "category, source, session_id, pinned, created_at, updated_at, "
                    "metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.memory_id, owner, content, category, source,
                        session_id, 1 if pinned else 0, now, now,
                        json.dumps(record.metadata),
                    ),
                )
            if source == "auto":
                self._evict_over_budget(owner)
            return record

    def _find_duplicate(self, owner: str, content: str) -> Memory | None:
        wanted = _normalise(content)
        if not wanted:
            return None
        for record in self._all(owner):
            if _normalise(record.content) == wanted:
                return record
        return None

    def edit(
        self,
        memory_id: str,
        *,
        owner: str,
        content: str | None = None,
        category: str | None = None,
        pinned: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Change one. Owner-scoped: a guessed id reaches nothing."""
        if content is not None:
            content = content.strip()
            if not content:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    "A memory cannot be edited to nothing. Delete it instead.",
                )
            if len(content) > MAX_CONTENT:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    f"Too long: {len(content)} characters, limit {MAX_CONTENT}.",
                )
        with self._lock:
            return self._update(
                memory_id, owner=owner, content=content, category=category,
                pinned=pinned, metadata=metadata, now=time.time(),
            )

    def _update(
        self,
        memory_id: str,
        *,
        owner: str,
        content: str | None,
        category: str | None,
        pinned: bool | None,
        metadata: dict[str, Any] | None,
        now: float,
    ) -> Memory:
        record = self._get(memory_id, owner)
        if content is not None:
            record.content = content
        if category is not None:
            record.category = category
        if pinned is not None:
            record.pinned = pinned
        if metadata is not None:
            record.metadata = dict(metadata)
        record.updated_at = now
        with self.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_memories SET content = ?, category = ?, "
                "pinned = ?, updated_at = ?, metadata = ? "
                "WHERE memory_id = ? AND owner = ?",
                (
                    record.content, record.category, 1 if record.pinned else 0,
                    record.updated_at, json.dumps(record.metadata),
                    memory_id, owner,
                ),
            )
        return record

    def delete(self, memory_id: str, *, owner: str) -> bool:
        with self._lock, self.backend.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM hyperlink_memories WHERE memory_id = ? AND owner = ?",
                (memory_id, owner),
            )
            return bool(cursor.rowcount)

    # -- reading --------------------------------------------------------

    def get(self, memory_id: str, *, owner: str) -> Memory:
        with self._lock:
            return self._get(memory_id, owner)

    def _get(self, memory_id: str, owner: str) -> Memory:
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_memories WHERE memory_id = ? AND owner = ?",
                (memory_id, owner),
            ).fetchone()
        if row is None:
            # Same answer whether it does not exist or belongs to
            # somebody else: distinguishing them would let a caller
            # enumerate other people's memory ids.
            raise T1APIError(
                T1ErrorCode.NOT_FOUND, f"No memory {memory_id!r}.", http_status=404
            )
        return _from_row(row)

    def list(
        self,
        *,
        owner: str,
        category: str = "",
        source: str = "",
        limit: int = 200,
    ) -> list[Memory]:
        with self._lock:
            records = self._all(owner)
        if category:
            records = [r for r in records if r.category == category]
        if source:
            records = [r for r in records if r.source == source]
        return records[:limit]

    def _all(self, owner: str) -> list[Memory]:
        with self.backend.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hyperlink_memories WHERE owner = ? "
                "ORDER BY pinned DESC, updated_at DESC",
                (owner,),
            ).fetchall()
        return [_from_row(row) for row in rows]

    def prompt_block(self, *, owner: str, limit: int = 40) -> str:
        """The memories, as a block to put in front of a conversation.

        Pinned first, then most recently touched — so when the budget
        does bite, what is lost is what nobody has confirmed lately
        rather than what somebody pinned on purpose.

        Empty string when there is nothing, so a caller can concatenate
        without having to check.
        """
        records = self.list(owner=owner, limit=limit)
        if not records:
            return ""
        lines = ["What you know about this person:"]
        for record in records:
            prefix = f"[{record.category}] " if record.category else ""
            lines.append(f"- {prefix}{record.content}")
        return "\n".join(lines)

    # -- the budget -----------------------------------------------------

    def _evict_over_budget(self, owner: str) -> int:
        """Keep model-written memories under :data:`AUTO_LIMIT`.

        Pinned memories are exempt and manual ones are never counted: a
        person who wrote two hundred memories meant to, and evicting
        their notes to make room for the model's guesses is precisely
        backwards.
        """
        with self.backend.connect() as conn:
            rows = conn.execute(
                "SELECT memory_id FROM hyperlink_memories "
                "WHERE owner = ? AND source = 'auto' AND pinned = 0 "
                "ORDER BY updated_at DESC",
                (owner,),
            ).fetchall()
            doomed = [row["memory_id"] for row in rows[AUTO_LIMIT:]]
            for memory_id in doomed:
                conn.execute(
                    "DELETE FROM hyperlink_memories WHERE memory_id = ?", (memory_id,)
                )
        if doomed:
            logger.info(
                "hyperlink.memory: evicted %d auto memories for %s (limit %d)",
                len(doomed), owner, AUTO_LIMIT,
            )
        return len(doomed)

    def count(self, *, owner: str, source: str = "") -> int:
        return len(self.list(owner=owner, source=source, limit=10_000))


def _from_row(row: Any) -> Memory:
    try:
        metadata = json.loads(row["metadata"] or "{}")
    except (ValueError, TypeError):
        metadata = {}
    return Memory(
        memory_id=row["memory_id"],
        owner=row["owner"],
        content=row["content"],
        category=row["category"] or "",
        source=row["source"] or "manual",
        session_id=row["session_id"] or "",
        pinned=bool(row["pinned"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


# ---------------------------------------------------------------------------
# Organising (0.72.5.post16)
#
# The model's memory tool used to file every fact under its own key, so
# the Memories screen grew one "category" per fact — "favourite editor",
# "home city", "dog's name" — which is a list, not an organisation. Facts
# are filed under a small set of topics instead, the key kept alongside.
# ---------------------------------------------------------------------------

#: Topic -> words that suggest it. First topic with the most hits wins;
#: order breaks ties, so the more personal topics come first.
TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("About you", ("name", "age", "birthday", "born", "pronoun", "live", "lives", "from",
                   "family", "married", "partner", "wife", "husband", "kid", "child",
                   "pet", "dog", "cat", "language", "nationality")),
    ("Preferences", ("prefer", "preference", "favourite", "favorite", "like", "likes",
                     "love", "hate", "dislike", "style", "tone", "format", "always",
                     "never", "theme", "diet", "vegetarian", "vegan")),
    ("Work", ("work", "job", "role", "company", "employer", "team", "manager",
              "colleague", "office", "career", "title", "client", "meeting")),
    ("Projects", ("project", "repo", "repository", "building", "app", "codebase",
                  "startup", "side project", "deadline", "launch", "roadmap")),
    ("Tech", ("editor", "ide", "language", "python", "rust", "swift", "javascript",
              "typescript", "linux", "mac", "windows", "gpu", "cuda", "server", "os",
              "shell", "vim", "helix", "emacs", "docker", "model", "framework")),
    ("Health", ("health", "allergy", "allergic", "medication", "doctor", "exercise",
                "sleep", "condition")),
    ("Places", ("city", "country", "home", "address", "travel", "trip", "timezone",
                "time zone", "moved")),
    ("Schedule", ("schedule", "every", "weekday", "weekend", "morning", "evening",
                  "routine", "remind")),
)

#: Where a fact with no topic words goes.
DEFAULT_TOPIC = "General"


def suggest_category(text: str) -> str:
    """The topic a fact belongs under, from the words in it."""
    haystack = " " + " ".join(re.findall(r"[a-z]+", (text or "").lower())) + " "
    best, best_hits = DEFAULT_TOPIC, 0
    for topic, cues in TOPICS:
        hits = sum(1 for cue in cues if f" {cue} " in haystack)
        if hits > best_hits:
            best, best_hits = topic, hits
    return best


def _is_topic(category: str) -> bool:
    return category in {t for t, _ in TOPICS} or category == DEFAULT_TOPIC


def categories(store: MemoryStore, *, owner: str) -> list[dict[str, Any]]:
    """Each category with how many memories it holds, largest first."""
    counts: dict[str, dict[str, Any]] = {}
    for memory in store.list(owner=owner, limit=100_000):
        name = memory.category or DEFAULT_TOPIC
        entry = counts.setdefault(name, {"name": name, "count": 0, "pinned": 0, "auto": 0})
        entry["count"] += 1
        entry["pinned"] += int(memory.pinned)
        entry["auto"] += int(memory.source == "auto")
    return sorted(counts.values(), key=lambda c: (-c["count"], c["name"].lower()))


def rename_category(store: MemoryStore, *, owner: str, old: str, new: str) -> int:
    """Move every memory in *old* to *new*. Merges when *new* exists."""
    new = (new or "").strip()
    if not new:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "A category needs a name.")
    if len(new) > 60:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "A category name is at most 60 characters.")
    moved = 0
    for memory in store.list(owner=owner, limit=100_000):
        if (memory.category or DEFAULT_TOPIC) == old and memory.category != new:
            store.edit(memory.memory_id, owner=owner, category=new)
            moved += 1
    return moved


def organise(store: MemoryStore, *, owner: str, dry_run: bool = False) -> list[dict[str, str]]:
    """File memories that are not under a topic under one.

    What it touches: a memory with no category, and a model-written one
    whose category is really a per-fact key (the old filing). A category
    a person chose is theirs and is left exactly where it is.
    """
    changes = []
    for memory in store.list(owner=owner, limit=100_000):
        if _is_topic(memory.category):
            continue
        if memory.category and memory.source != "auto":
            continue
        key = memory.category or str(memory.metadata.get("key", ""))
        topic = suggest_category(f"{key} {memory.content}")
        changes.append({"memory_id": memory.memory_id, "from": memory.category, "to": topic})
        if not dry_run:
            metadata = dict(memory.metadata)
            if key and "key" not in metadata:
                metadata["key"] = key
            content = memory.content
            if key and memory.source == "auto" and not content.lower().startswith(key.lower()):
                content = f"{key}: {content}"
            store.edit(memory.memory_id, owner=owner, category=topic, content=content,
                       metadata=metadata)
    return changes


class ToolMemoryBackend:
    """The model's memory tools, writing into a person's real memories.

    A model in a HyperLink chat calls `update_memory(key, value)`. Before
    this, that went to a JSON file in the tool workspace which nothing
    reads — so a person asked the assistant to remember something, it
    said it had, and the Memories screen never showed it. This puts it
    in the same store the screen lists, for the same owner, marked
    `auto` so "where did it get that idea" still has an answer.

    The fact is filed under a topic (see :func:`suggest_category`) with
    its key kept in the memory's metadata, and the content reads as the
    fact: "favourite editor: Helix", under Tech.
    """

    def __init__(self, store: MemoryStore, owner: str, session_id: str = "") -> None:
        self.store = store
        self.owner = owner
        self.session_id = session_id

    def _mine(self, key: str) -> list[Memory]:
        key = key.strip()
        return [
            m for m in self.store.list(owner=self.owner, limit=1000)
            if m.source == "auto"
            # The key in metadata, or — for a memory filed before topics —
            # the key as its category.
            and (m.metadata.get("key") == key or m.category == key)
        ]

    def load(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for memory in self.store.list(owner=self.owner, limit=1000):
            key = str(memory.metadata.get("key") or "") or memory.category or memory.memory_id
            value = memory.content
            if value.lower().startswith(f"{key.lower()}: "):
                value = value[len(key) + 2:]
            out[key] = {"value": value, "updated_at": memory.updated_at,
                        "source": memory.source, "category": memory.category}
        return out

    def set(self, key: str, value: Any) -> Memory:
        key = key.strip()
        value_text = value if isinstance(value, str) else json.dumps(value, default=str)
        content = f"{key}: {value_text}" if key else value_text
        topic = suggest_category(f"{key} {value_text}")
        existing = self._mine(key)
        if existing:
            # Updated in place rather than added: a model that learns you
            # moved from Vim to Helix should replace the fact, not keep both.
            current = existing[0]
            category = current.category if _is_topic(current.category) or current.source != "auto" else topic
            return self.store.edit(current.memory_id, owner=self.owner, content=content,
                                   category=category, metadata={**current.metadata, "key": key})
        return self.store.create(owner=self.owner, content=content, category=topic,
                                 source="auto", session_id=self.session_id,
                                 metadata={"key": key})

    def forget(self, key: str) -> bool:
        # Only what the model wrote. A memory the person typed themselves
        # is theirs to delete, not the model's.
        removed = False
        for memory in self._mine(key):
            removed = self.store.delete(memory.memory_id, owner=self.owner) or removed
        return removed

