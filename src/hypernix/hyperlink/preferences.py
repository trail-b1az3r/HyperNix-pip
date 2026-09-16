"""Per-person settings: who they are, and how they want to be answered.

Everything in the settings screen that is not a server lives here —
profile, bio, the custom system prompt, effort level, context bounds,
the backup model, the backend, and whether tools and auto-memory are on.

Why the server and not the phone
--------------------------------
Two reasons, and the second is the one that decides it.

A person with a phone and a tablet is one person. Preferences kept on
the device mean two different answers to "how do you want to be
answered" depending on which thing they picked up.

And these are *inputs to generation*. The system prompt, the effort
level and the context bounds all have to be in the process that builds
the request — which is the server. Keeping them on the phone would mean
sending them on every turn and trusting whatever arrives, which is both
more work and a worse story: a stale client would silently answer with
last month's settings.

Bounds, not preferences
-----------------------
Several of these are clamped rather than accepted. `context_maximum` of
four million is not a preference, it is a number that makes every reply
fail with an out-of-memory two minutes in — and the failure arrives
somewhere unrelated, so it reads as the model being broken. The clamps
are named constants below and :meth:`PreferenceStore.save` reports what
it changed rather than quietly fixing it.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

#: How hard the model should think, in the order people mean them.
#:
#: These are *this project's* names, mapped onto whatever the backend
#: understands in `sampling_for` below. A model that has a real
#: reasoning-effort control gets it; one that does not gets the
#: temperature and token budget that approximate it, which is honest
#: about being an approximation rather than silently ignoring the
#: setting.
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "maximum")
DEFAULT_EFFORT = "medium"

#: The smallest context worth serving. Below this a system prompt plus
#: two memories is the whole window and the conversation has nowhere to
#: go, which reads as the model forgetting everything instantly.
CONTEXT_FLOOR = 1024

#: The largest this will accept. Not a hardware limit — it is the point
#: past which "it didn't work" stops being about the setting and starts
#: being about VRAM, in an error that arrives two minutes later and
#: somewhere else.
CONTEXT_CEILING = 1_048_576

#: Longest custom system prompt. The request asked for a "massive" one
#: and this is that: roughly 8,000 words, which is more instruction than
#: most models can follow and comfortably more than anybody writes by
#: hand. Past it the prompt is eating the context it is meant to be
#: steering.
MAX_SYSTEM_PROMPT = 32_000

#: Longest bio. A paragraph about yourself, prepended to every
#: conversation — it is charged for on every single turn, which is the
#: argument for a limit that a system prompt does not have.
MAX_BIO = 2_000

MAX_DISPLAY_NAME = 120

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_preferences (
    owner TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    bio TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    effort TEXT NOT NULL DEFAULT 'medium',
    context_minimum INTEGER NOT NULL DEFAULT 0,
    context_maximum INTEGER NOT NULL DEFAULT 0,
    backup_model TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT '',
    tools_enabled INTEGER NOT NULL DEFAULT 0,
    auto_memory INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


@dataclass
class Preferences:
    """One person's settings. Every field has a working default."""

    owner: str = ""
    display_name: str = ""
    bio: str = ""
    #: Prepended to every conversation, before the session's own prompt.
    system_prompt: str = ""
    effort: str = DEFAULT_EFFORT
    #: 0 means "no opinion" for both, which is the right default: a
    #: number here overrides what the model itself says it can do, and
    #: most people should not be doing that.
    context_minimum: int = 0
    context_maximum: int = 0
    #: Tried when the main model fails. Empty means "fail honestly",
    #: which is better than silently answering as somebody else.
    backup_model: str = ""
    backend: str = ""
    tools_enabled: bool = False
    auto_memory: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "display_name": self.display_name,
            "bio": self.bio,
            "system_prompt": self.system_prompt,
            "effort": self.effort,
            "context_minimum": self.context_minimum,
            "context_maximum": self.context_maximum,
            "backup_model": self.backup_model,
            "backend": self.backend,
            "tools_enabled": self.tools_enabled,
            "auto_memory": self.auto_memory,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


def sampling_for(effort: str) -> dict[str, Any]:
    """Sampling settings for an effort level.

    Two knobs, because two is what every backend has. A model with a
    real reasoning-effort parameter is given the level by name as well
    — see ``reasoning_effort`` — and one without gets an approximation
    that is at least in the right direction: lower effort is cooler and
    shorter, higher effort is warmer and allowed to run on.

    Calling this an approximation in the docstring rather than
    pretending it is a dial into the model's cognition is the whole
    point. It is scheduling, not thinking.
    """
    table = {
        "minimal": {"temperature": 0.0, "max_tokens": 512},
        "low": {"temperature": 0.3, "max_tokens": 1024},
        "medium": {"temperature": 0.7, "max_tokens": 2048},
        "high": {"temperature": 0.8, "max_tokens": 4096},
        "maximum": {"temperature": 0.9, "max_tokens": 8192},
    }
    known = effort if effort in table else DEFAULT_EFFORT
    settings = dict(table[known])
    # The *normalised* level, not what was passed in. An unknown name
    # reaching a backend that understands this field would be rejected
    # there — turning a stale row written by an older build into a
    # failure on every reply, which is exactly what falling back is
    # supposed to prevent.
    settings["reasoning_effort"] = known
    return settings


class PreferenceStore:
    """One row per owner, created on first read."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    def get(self, *, owner: str) -> Preferences:
        """This owner's settings, defaulted if they have never set any.

        Deliberately does not write on read. A GET that creates a row
        means a server whose database grows every time an unauthorised
        caller guesses an owner name.
        """
        if not owner:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR, "Preferences need an owner."
            )
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_preferences WHERE owner = ?", (owner,)
            ).fetchone()
        return _from_row(row) if row is not None else Preferences(owner=owner)

    def save(self, *, owner: str, **changes: Any) -> tuple[Preferences, list[str]]:
        """Apply *changes*. Returns the result and what it had to adjust.

        Only the keys given are touched, so a client that knows about
        six settings cannot blank the four it has never heard of by
        sending them as defaults — which is what a whole-object PUT
        would do the first time the app and the server disagreed about
        the field list.

        The second element is every clamp and correction applied, in
        words. Silently storing something other than what somebody typed
        is how a settings screen becomes untrustworthy.
        """
        current = self.get(owner=owner)
        notes: list[str] = []

        def text(name: str, limit: int, label: str) -> None:
            if name not in changes:
                return
            value = str(changes[name] or "")
            if len(value) > limit:
                notes.append(
                    f"{label} was {len(value)} characters and has been cut to "
                    f"{limit}."
                )
                value = value[:limit]
            setattr(current, name, value)

        text("display_name", MAX_DISPLAY_NAME, "The display name")
        text("bio", MAX_BIO, "The bio")
        text("system_prompt", MAX_SYSTEM_PROMPT, "The system prompt")

        if "effort" in changes:
            wanted = str(changes["effort"] or "").strip().lower()
            if wanted not in EFFORT_LEVELS:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR,
                    f"effort is one of {', '.join(EFFORT_LEVELS)}; got "
                    f"{changes['effort']!r}",
                )
            current.effort = wanted

        for name, label in (
            ("context_minimum", "The context minimum"),
            ("context_maximum", "The context maximum"),
        ):
            if name not in changes:
                continue
            try:
                value = int(changes[name] or 0)
            except (TypeError, ValueError) as exc:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR, f"{name} must be a number."
                ) from exc
            if value < 0:
                raise T1APIError(
                    T1ErrorCode.VALIDATION_ERROR, f"{name} cannot be negative."
                )
            if value and value < CONTEXT_FLOOR:
                notes.append(
                    f"{label} was {value}, which leaves no room for a system "
                    f"prompt — raised to {CONTEXT_FLOOR}."
                )
                value = CONTEXT_FLOOR
            if value > CONTEXT_CEILING:
                notes.append(
                    f"{label} was {value}, past what any of this can serve — "
                    f"lowered to {CONTEXT_CEILING}."
                )
                value = CONTEXT_CEILING
            setattr(current, name, value)

        if current.context_minimum and current.context_maximum:
            if current.context_minimum > current.context_maximum:
                # Swapped rather than refused: somebody who typed them
                # the wrong way round meant the range, and a 422 here is
                # a settings screen that will not save.
                notes.append(
                    "The minimum was above the maximum, so they have been "
                    "swapped."
                )
                current.context_minimum, current.context_maximum = (
                    current.context_maximum, current.context_minimum,
                )

        if "backup_model" in changes:
            current.backup_model = str(changes["backup_model"] or "")
        if "backend" in changes:
            current.backend = str(changes["backend"] or "")
        if "tools_enabled" in changes:
            current.tools_enabled = bool(changes["tools_enabled"])
        if "auto_memory" in changes:
            current.auto_memory = bool(changes["auto_memory"])
        if "metadata" in changes and isinstance(changes["metadata"], dict):
            current.metadata = dict(changes["metadata"])

        now = time.time()
        current.owner = owner
        current.updated_at = now
        current.created_at = current.created_at or now

        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """INSERT INTO hyperlink_preferences
                   (owner, display_name, bio, system_prompt, effort,
                    context_minimum, context_maximum, backup_model, backend,
                    tools_enabled, auto_memory, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(owner) DO UPDATE SET
                     display_name = excluded.display_name,
                     bio = excluded.bio,
                     system_prompt = excluded.system_prompt,
                     effort = excluded.effort,
                     context_minimum = excluded.context_minimum,
                     context_maximum = excluded.context_maximum,
                     backup_model = excluded.backup_model,
                     backend = excluded.backend,
                     tools_enabled = excluded.tools_enabled,
                     auto_memory = excluded.auto_memory,
                     updated_at = excluded.updated_at,
                     metadata = excluded.metadata""",
                (
                    owner, current.display_name, current.bio, current.system_prompt,
                    current.effort, current.context_minimum, current.context_maximum,
                    current.backup_model, current.backend,
                    int(current.tools_enabled), int(current.auto_memory),
                    current.created_at, current.updated_at,
                    json.dumps(current.metadata),
                ),
            )
        return current, notes

    def reset(self, *, owner: str) -> None:
        """Back to the defaults. Deletes the row rather than blanking it,
        so "never set anything" and "set everything back" are the same
        state — which is what somebody pressing Reset means."""
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "DELETE FROM hyperlink_preferences WHERE owner = ?", (owner,)
            )


def system_prompt_for(
    preferences: Preferences,
    *,
    session_prompt: str = "",
    memory_block: str = "",
) -> str:
    """The system prompt a turn actually gets.

    Order is deliberate and it is the order of *scope*: who the person
    is, then how they want to be answered generally, then what this
    particular conversation is for, then what is known about them.

    The session's own prompt comes after the global one so a
    conversation can override the default rather than fight it — the
    later instruction is the one a model follows when two conflict.
    """
    parts: list[str] = []
    if preferences.display_name or preferences.bio:
        who = preferences.display_name or "The person you are talking to"
        parts.append(
            f"About {who}:\n{preferences.bio}" if preferences.bio
            else f"You are talking to {who}."
        )
    if preferences.system_prompt:
        parts.append(preferences.system_prompt)
    if session_prompt:
        parts.append(session_prompt)
    if memory_block:
        parts.append(memory_block)
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _from_row(row: Any) -> Preferences:
    try:
        metadata = json.loads(row["metadata"] or "{}")
    except (TypeError, ValueError):
        metadata = {}
    return Preferences(
        owner=row["owner"],
        display_name=row["display_name"],
        bio=row["bio"],
        system_prompt=row["system_prompt"],
        effort=row["effort"] or DEFAULT_EFFORT,
        context_minimum=int(row["context_minimum"] or 0),
        context_maximum=int(row["context_maximum"] or 0),
        backup_model=row["backup_model"],
        backend=row["backend"],
        tools_enabled=bool(row["tools_enabled"]),
        auto_memory=bool(row["auto_memory"]),
        created_at=float(row["created_at"] or 0.0),
        updated_at=float(row["updated_at"] or 0.0),
        metadata=metadata if isinstance(metadata, dict) else {},
    )
