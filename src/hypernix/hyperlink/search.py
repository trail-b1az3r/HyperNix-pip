"""hyperlink.search — finding the conversation you half remember.

A phone accumulates months of conversations and offers a list sorted by
date to find them in. The thing people actually want is "the one where I
worked out the CUDA thing", and scrolling is the only tool for it.

Why not FTS5
------------
SQLite's full-text extension is the obvious answer and the wrong one
here. The T1 API runs on SQLite *or* PostgreSQL — that is what
``hypernix.t1api.db`` exists for — and FTS5 has no PostgreSQL
counterpart that shares its syntax. Writing to an FTS5 virtual table
would make search a SQLite-only feature and the schema unportable, and
maintaining two query paths for a personal server's chat history buys
speed nobody can perceive.

So: SQL narrows, Python matches. The narrowing is by owner, archive
state and date range — all indexed — and the matching runs over the
candidates in Python, which buys three things SQL LIKE cannot give:

* **Unicode-correct case folding.** SQLite's ``LIKE`` is
  case-insensitive for ASCII only. ``LIKE '%STRASSE%'`` does not match
  "straße", and ``LIKE '%İSTANBUL%'`` does not match "istanbul".
  :meth:`str.casefold` handles both.
* **No wildcard injection.** A query containing ``%`` is a query for a
  percent sign, not a request to match everything. In a ``LIKE`` it is
  the latter unless every call escapes it, and the call that forgets is
  the one that returns the whole database.
* **Ranking that can see the whole match.** Term proximity and field
  weighting need the positions, which a boolean ``LIKE`` has thrown
  away by the time it answers.

The cost is a bounded scan, and it is bounded on purpose:
:data:`MAX_SCAN` rows. Past that the result says :attr:`Results.capped`
so the client can say "showing the most recent N" rather than implying
it searched everything. A silent partial answer is the failure mode
worth avoiding — someone concludes the conversation is gone.

Ranking
-------
Deliberately simple and explainable, because a chat search that returns
a surprising order is worse than one that returns an obvious one:
matches in a title outweigh matches in a body, all-terms-present
outweighs some, more occurrences outweigh fewer, and recent outweighs
old. The weights are module constants so they can be argued with.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

__all__ = [
    "MAX_SCAN",
    "MAX_TERMS",
    "Hit",
    "Results",
    "SearchIndex",
    "Snippet",
    "normalise",
    "tokenize",
]

# The most rows one search will look at. A personal machine's history
# fits comfortably; a shared one gets a capped, honest answer instead of
# an unbounded scan holding a connection open.
MAX_SCAN = 20_000

# More terms than this is a paste, not a query, and each one costs a
# pass over every candidate.
MAX_TERMS = 24

# Snippet width, in characters either side of the match.
SNIPPET_CONTEXT = 60

# Ranking weights. A title match is worth more than a body match because
# a title is a deliberate summary; recency breaks ties because the
# conversation you half remember is usually a recent one.
WEIGHT_TITLE = 8.0
WEIGHT_BODY = 1.0
WEIGHT_ALL_TERMS = 12.0
WEIGHT_WHOLE_PHRASE = 6.0
RECENCY_HALF_LIFE_DAYS = 30.0

# Roles whose content is searched. `system` is excluded: a system prompt
# is configuration that appears in every session, so matching it returns
# everything and ranks it all identically.
SEARCHABLE_ROLES = ("user", "assistant", "tool")

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    """Case-folded, NFKC-normalised text, for comparison only.

    NFKC first so that characters with more than one encoding compare
    equal — "ﬁ" as one codepoint and "fi" as two, or a full-width "Ａ"
    and an ASCII "A". A phone keyboard and a desktop keyboard routinely
    produce different encodings of what the user considers one string.

    Then :meth:`str.casefold`, which is not ``lower()``: folding maps
    "ß" to "ss" and handles the Turkish dotted/dotless I, so a search
    for "STRASSE" finds "straße".
    """
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(query: str) -> list[str]:
    """Split a query into search terms, longest first.

    Longest first so the highest-signal term decides the snippet: a
    search for "the cuda thing" should show the line with *cuda* in it,
    not the first "the".
    """
    terms = [t for t in _WORD_RE.findall(normalise(query)) if t]
    # Deduplicated, because repeating a term should not multiply its
    # weight -- "cuda cuda cuda" is one intent, not three.
    seen: set[str] = set()
    unique = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    unique.sort(key=len, reverse=True)
    return unique[:MAX_TERMS]


@dataclass
class Snippet:
    """A window of text around a match, with the match located.

    Offsets rather than markup: the client decides whether a match is
    bold, highlighted or underlined, and a server that returns HTML has
    made that decision for a SwiftUI view that cannot use it.
    """

    text: str
    ranges: list[tuple[int, int]] = field(default_factory=list)
    truncated_start: bool = False
    truncated_end: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "ranges": [list(r) for r in self.ranges],
            "truncated_start": self.truncated_start,
            "truncated_end": self.truncated_end,
        }


@dataclass
class Hit:
    session_id: str
    title: str
    score: float
    updated_at: float
    message_id: str = ""
    role: str = ""
    created_at: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    snippet: Snippet | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "session_id": self.session_id,
            "title": self.title,
            "score": round(self.score, 4),
            "updated_at": self.updated_at,
            "matched_terms": self.matched_terms,
        }
        if self.message_id:
            out["message_id"] = self.message_id
            out["role"] = self.role
            out["created_at"] = self.created_at
        if self.snippet is not None:
            out["snippet"] = self.snippet.to_dict()
        return out


@dataclass
class Results:
    hits: list[Hit]
    scanned: int
    capped: bool
    terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "scanned": self.scanned,
            "capped": self.capped,
            "terms": self.terms,
        }


def _match_ranges(haystack_folded: str, terms: Sequence[str]) -> list[tuple[int, int]]:
    """Where each term occurs, as (start, end) offsets, merged.

    Offsets are into the *folded* string. Case folding can change
    length — "ß" folds to "ss" — so these are not always valid offsets
    into the original, which is why the snippet is cut from the folded
    text and the original is not used for display where a match is
    highlighted. Losing the original casing in a snippet is a smaller
    problem than a highlight landing on the wrong characters.
    """
    spans: list[tuple[int, int]] = []
    for term in terms:
        start = 0
        while True:
            found = haystack_folded.find(term, start)
            if found < 0:
                break
            spans.append((found, found + len(term)))
            start = found + 1
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for lo, hi in spans[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _build_snippet(folded: str, spans: Sequence[tuple[int, int]]) -> Snippet:
    """A window around the first match, with offsets rebased into it."""
    if not spans:
        head = folded[: SNIPPET_CONTEXT * 2]
        return Snippet(text=head, truncated_end=len(folded) > len(head))
    first_lo, _ = spans[0]
    start = max(0, first_lo - SNIPPET_CONTEXT)
    end = min(len(folded), spans[0][1] + SNIPPET_CONTEXT * 2)
    window = folded[start:end]
    rebased = [
        (lo - start, hi - start)
        for lo, hi in spans
        if lo >= start and hi <= end
    ]
    return Snippet(
        text=window,
        ranges=rebased,
        truncated_start=start > 0,
        truncated_end=end < len(folded),
    )


def _recency_boost(when: float, *, now: float) -> float:
    """A gentle decay, never negative and never zero.

    Halving every :data:`RECENCY_HALF_LIFE_DAYS` means a month-old
    conversation ranks below an identical one from today, without a
    year-old exact match being buried under today's near-misses.
    """
    age_days = max(0.0, (now - when) / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


class SearchIndex:
    """Search over one owner's sessions and messages.

    No index is built or maintained: it reads the same tables
    :class:`~hypernix.hyperlink.sessions.ChatSessionStore` writes, so
    there is nothing to keep in step and nothing to rebuild after a
    restore. That is the trade — a bounded scan instead of an index
    that can be stale or corrupt.
    """

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()

    def search(
        self,
        query: str,
        *,
        owner: str,
        limit: int = 25,
        include_archived: bool = False,
        session_id: str | None = None,
        roles: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        now: float | None = None,
    ) -> Results:
        """Find messages and sessions matching *query*.

        Terms are ANDed for ranking but not for inclusion: a message
        matching two of three terms is still a hit, ranked below one
        matching all three. Requiring every term turns a slightly
        misremembered query into no results at all, which is the
        outcome someone reads as "it is gone".
        """
        if not owner:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "owner is required")
        terms = tokenize(query)
        if not terms:
            return Results(hits=[], scanned=0, capped=False, terms=[])

        wanted_roles = tuple(roles) if roles is not None else SEARCHABLE_ROLES
        unknown = set(wanted_roles) - set(SEARCHABLE_ROLES)
        if unknown:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"Cannot search roles {sorted(unknown)}; searchable roles are "
                f"{list(SEARCHABLE_ROLES)}",
            )
        capped_limit = max(1, min(int(limit), 200))
        moment = time.time() if now is None else now
        phrase = normalise(query).strip()

        sessions = self._candidate_sessions(
            owner=owner, include_archived=include_archived, session_id=session_id
        )
        if not sessions:
            return Results(hits=[], scanned=0, capped=False, terms=terms)

        hits: list[Hit] = []
        for row in sessions:
            folded_title = normalise(row["title"] or "")
            spans = _match_ranges(folded_title, terms)
            if spans:
                matched = [t for t in terms if t in folded_title]
                hits.append(
                    Hit(
                        session_id=row["session_id"],
                        title=row["title"],
                        updated_at=float(row["updated_at"]),
                        matched_terms=matched,
                        snippet=_build_snippet(folded_title, spans),
                        score=self._score(
                            spans=spans, matched=matched, terms=terms,
                            weight=WEIGHT_TITLE, when=float(row["updated_at"]),
                            haystack=folded_title, phrase=phrase, now=moment,
                        ),
                    )
                )

        titles = {r["session_id"]: (r["title"], float(r["updated_at"])) for r in sessions}
        scanned, capped = self._scan_messages(
            hits, titles=titles, terms=terms, phrase=phrase, roles=wanted_roles,
            since=since, until=until, now=moment,
        )

        hits.sort(key=lambda h: (-h.score, -h.created_at, -h.updated_at))
        return Results(
            hits=hits[:capped_limit], scanned=scanned, capped=capped, terms=terms
        )

    # -- internals ----------------------------------------------------

    def _candidate_sessions(
        self, *, owner: str, include_archived: bool, session_id: str | None
    ) -> list[Any]:
        sql = "SELECT session_id, title, updated_at FROM hyperlink_sessions WHERE owner = ?"
        params: list[Any] = [owner]
        if not include_archived:
            sql += " AND archived = 0"
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(MAX_SCAN)
        with self.backend.connect() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def _scan_messages(
        self,
        hits: list[Hit],
        *,
        titles: dict[str, tuple[str, float]],
        terms: Sequence[str],
        phrase: str,
        roles: Sequence[str],
        since: float | None,
        until: float | None,
        now: float,
    ) -> tuple[int, bool]:
        if not titles or not roles:
            return 0, False
        session_ph = ",".join("?" for _ in titles)
        role_ph = ",".join("?" for _ in roles)
        sql = (
            f"SELECT message_id, session_id, role, content, created_at "
            f"FROM hyperlink_messages "
            f"WHERE session_id IN ({session_ph}) AND role IN ({role_ph})"
        )
        params: list[Any] = [*titles.keys(), *roles]
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        if until is not None:
            sql += " AND created_at <= ?"
            params.append(until)
        # Newest first, so a capped scan keeps the recent history rather
        # than an arbitrary slice of the oldest.
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(MAX_SCAN + 1)

        with self.backend.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        capped = len(rows) > MAX_SCAN
        rows = rows[:MAX_SCAN]

        for row in rows:
            folded = normalise(row["content"] or "")
            spans = _match_ranges(folded, terms)
            if not spans:
                continue
            matched = [t for t in terms if t in folded]
            title, updated = titles[row["session_id"]]
            created = float(row["created_at"])
            hits.append(
                Hit(
                    session_id=row["session_id"],
                    title=title,
                    updated_at=updated,
                    message_id=row["message_id"],
                    role=row["role"],
                    created_at=created,
                    matched_terms=matched,
                    snippet=_build_snippet(folded, spans),
                    score=self._score(
                        spans=spans, matched=matched, terms=terms,
                        weight=WEIGHT_BODY, when=created, haystack=folded,
                        phrase=phrase, now=now,
                    ),
                )
            )
        return len(rows), capped

    def _score(
        self,
        *,
        spans: Sequence[tuple[int, int]],
        matched: Sequence[str],
        terms: Sequence[str],
        weight: float,
        when: float,
        haystack: str,
        phrase: str,
        now: float,
    ) -> float:
        """Occurrences, coverage, exact phrase, then recency.

        Occurrences are damped by a square root: a message mentioning a
        term forty times is more relevant than one mentioning it twice,
        but not twenty times more, and without damping a long document
        wins every search by being long.
        """
        base = weight * (len(spans) ** 0.5)
        if len(matched) == len(terms) and terms:
            base += WEIGHT_ALL_TERMS
        # An exact phrase is a much stronger signal than the same words
        # scattered, and it is the thing someone typing a remembered
        # sentence is actually asking for.
        if phrase and len(phrase) > 3 and phrase in haystack:
            base += WEIGHT_WHOLE_PHRASE
        return base * (1.0 + _recency_boost(when, now=now))
