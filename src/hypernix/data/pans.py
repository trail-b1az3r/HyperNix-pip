"""pans — five progressive tiers of training-data preprocessing.

A *pan* reads text and yields processed strings.  Like cookware, each
pan applies more heat (transformation) than the one before it.  Chain
pans together with :class:`hypernix.sink.Sink` to write the output to
disk for training, or iterate them directly for use inline.

Tiers, lightest to heaviest:

1. :class:`FryingPan` — verbatim lines, lightly trimmed.
2. :class:`SaucePan`  — + whitespace normalization + empty-line drop.
3. :class:`Skillet`   — + chat/instruction formatting (role tags).
4. :class:`GrillPan`  — + hash-based deduplication + min-length filter.
5. :class:`Wok`       — + in-memory shuffle (and optional line-reversal
                          augmentation).

Every pan supports the same iterator protocol so callers can swap
tiers by name.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

_WS_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Pan:
    """Abstract pan.  Subclasses override :meth:`cook` (one line in,
    zero-or-one line out) or :meth:`iter` (full iterator override).

    ``name`` is deliberately a :class:`ClassVar` rather than a
    ``@dataclass`` field — it's a human-readable label used in
    ``__repr__`` / logs, not state.  Making it a field exposed it as
    the *second positional argument* of every subclass, which meant
    ``Skillet("file.txt", "instruct")`` silently set ``name="instruct"``
    instead of ``mode="instruct"`` and left the caller confused.

    Every pan accepts two *keyword-only* length caps:

    * ``max_chars`` — hard per-line character cap.  Longer lines are
      truncated (not split).
    * ``context_length`` — convenience for training-script callers.
      Treated as an approximate token budget and converted to
      ``max_chars = context_length * 4`` (a reasonable English-BPE
      heuristic).  Takes precedence over ``max_chars`` when both are
      set.  For precise chunking by tokens / chars use
      :mod:`hypernix.food_processor` (``SliceBlade`` / ``ShredBlade``)
      instead.
    """

    source: Path | str | Iterable[str]
    max_chars: int | None = field(default=None, kw_only=True)
    context_length: int | None = field(default=None, kw_only=True)
    name: ClassVar[str] = "Pan"

    # Rough chars-per-token ratio for English BPE tokenizers.  Only
    # used when ``context_length`` is set and ``max_chars`` is not.
    _CHARS_PER_TOKEN: ClassVar[int] = 4

    def _effective_max_chars(self) -> int | None:
        if self.context_length is not None:
            return self.context_length * self._CHARS_PER_TOKEN
        return self.max_chars

    def _source_lines(self) -> Iterator[str]:
        if isinstance(self.source, str | Path):
            p = Path(self.source)
            with p.open(encoding="utf-8") as f:
                for line in f:
                    yield line.rstrip("\n")
        else:
            yield from self.source

    def cook(self, line: str) -> str | None:
        return line

    def iter(self) -> Iterator[str]:
        cap = self._effective_max_chars()
        for raw in self._source_lines():
            out = self.cook(raw)
            if out is None:
                continue
            if cap is not None and len(out) > cap:
                out = out[:cap]
            yield out

    def __iter__(self) -> Iterator[str]:
        return self.iter()


# ---------------------------------------------------------------------------
# Tier 1 — FryingPan
# ---------------------------------------------------------------------------

@dataclass
class FryingPan(Pan):
    """Lightest tier.  Strip trailing whitespace, pass everything else
    through.  Useful when the input is already clean (already-formatted
    corpora, pre-tokenized chunks)."""

    name: ClassVar[str] = "FryingPan"

    def cook(self, line: str) -> str | None:
        return line.rstrip()


# ---------------------------------------------------------------------------
# Tier 2 — SaucePan
# ---------------------------------------------------------------------------

@dataclass
class SaucePan(Pan):
    """Reduce.  Collapse runs of internal whitespace, drop empty lines,
    strip leading/trailing whitespace.  Standard mild cleaning for
    scraped or OCR'd text."""

    name: ClassVar[str] = "SaucePan"

    def cook(self, line: str) -> str | None:
        line = _WS_RE.sub(" ", line.strip())
        return line or None


# ---------------------------------------------------------------------------
# Tier 3 — Skillet
# ---------------------------------------------------------------------------

@dataclass
class Skillet(Pan):
    """Versatile.  Applies chat / instruction formatting by wrapping
    each line with role tags.  ``mode='chat'`` yields
    ``<USER>line\\n<ASSISTANT>`` templates; ``mode='instruct'`` yields
    ``### Instruction: line\\n### Response:`` — use one when every input
    line is a user turn and the dataset is paired with assistant
    replies on the following line (or in a separate corpus).
    """

    mode: str = "chat"            # "chat" | "instruct"
    user_tag: str = "<USER>"
    assistant_tag: str = "<ASSISTANT>"
    name: ClassVar[str] = "Skillet"

    def cook(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        if self.mode == "instruct":
            return f"### Instruction: {line}\n### Response:"
        return f"{self.user_tag} {line}\n{self.assistant_tag}"


# ---------------------------------------------------------------------------
# Tier 4 — GrillPan
# ---------------------------------------------------------------------------

@dataclass
class GrillPan(Pan):
    """High direct heat.  SaucePan-style cleaning plus deduplication
    (SHA1 hash set) and a minimum length filter.  Safe on arbitrary
    web-scraped text: collapses boilerplate, drops single-word lines.

    ``_seen`` is internal dedupe state and is not part of the public
    ``__init__`` signature.
    """

    min_chars: int = 8
    name: ClassVar[str] = "GrillPan"
    _seen: set[str] = field(default_factory=set, repr=False, init=False)

    def cook(self, line: str) -> str | None:
        line = _WS_RE.sub(" ", line.strip())
        if len(line) < self.min_chars:
            return None
        # Dedupe only — not a security digest. The flag says so, and is
        # what keeps this working on a FIPS-enabled host, where an
        # unflagged SHA-1 raises rather than returning a hash.
        h = hashlib.sha1(line.encode("utf-8"), usedforsecurity=False).hexdigest()
        if h in self._seen:
            return None
        self._seen.add(h)
        return line


# ---------------------------------------------------------------------------
# Tier 5 — Wok
# ---------------------------------------------------------------------------

@dataclass
class Wok(Pan):
    """Heaviest.  Buffers the whole source in memory, shuffles it,
    and optionally injects line-reversal augmentation (``reverse_ratio``
    fraction of lines emitted in reverse word order).  Use this for
    small corpora where order-randomization matters and you can afford
    the memory."""

    seed: int = 0
    reverse_ratio: float = 0.0
    name: ClassVar[str] = "Wok"

    def iter(self) -> Iterator[str]:
        # Pre-clean with SaucePan semantics.
        rng = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
        cap = self._effective_max_chars()
        buffer: list[str] = []
        for raw in self._source_lines():
            line = _WS_RE.sub(" ", raw.strip())
            if line:
                buffer.append(line)
        rng.shuffle(buffer)
        for line in buffer:
            out = (
                " ".join(reversed(line.split()))
                if self.reverse_ratio > 0 and rng.random() < self.reverse_ratio
                else line
            )
            if cap is not None and len(out) > cap:
                out = out[:cap]
            yield out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TIERS: dict[str, type[Pan]] = {
    "frying-pan": FryingPan,
    "sauce-pan": SaucePan,
    "skillet": Skillet,
    "grill-pan": GrillPan,
    "wok": Wok,
}


def pick_pan(tier: str, source: Path | str | Iterable[str], **kwargs: Any) -> Pan:
    """Return a pan instance by tier name (``"frying-pan"``..``"wok"``).

    Raises :class:`ValueError` if ``tier`` is unknown (with the list of
    valid tiers in the message) or if ``**kwargs`` contains a keyword
    the selected pan doesn't accept (with the list of valid kwargs).
    """
    key = tier.lower().replace("_", "-")
    if key not in TIERS:
        raise ValueError(
            f"unknown pan tier {tier!r}; valid tiers are: {sorted(TIERS)}"
        )
    cls = TIERS[key]
    try:
        return cls(source=source, **kwargs)
    except TypeError as exc:
        # Re-raise with a message that lists the keywords the tier
        # actually accepts — the raw dataclass message ("unexpected
        # keyword argument 'X'") doesn't say what *is* accepted.
        import inspect
        valid = [p for p in inspect.signature(cls).parameters if p != "source"]
        raise ValueError(
            f"{cls.__name__} rejected an argument ({exc}). "
            f"Valid keyword arguments for {cls.__name__} (besides "
            f"'source'): {valid}"
        ) from exc
