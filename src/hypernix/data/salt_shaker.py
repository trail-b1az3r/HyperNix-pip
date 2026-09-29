"""salt_shaker — gentle noise / data augmentation.

Salt is subtle.  In the kitchen idiom, a salt shaker adds flavor
without overwhelming the dish; here it lightly perturbs training
examples to improve robustness without corrupting meaning.

Three tiers, coarsest to finest:

* :class:`FromTheBag`   — t1.  Straight from the bag: uniform
                            character-level noise.  Replace a fraction
                            of characters with random characters from
                            the same printable set.
* :class:`HandCrusher`  — t2.  Coarse grind: token-level swaps.
                            Splits on whitespace and swaps adjacent
                            tokens at ``rate``.
* :class:`PoshSaltDish` — t3.  Precise dosing: word-aware drop /
                            insert / swap at configurable rates,
                            preserves punctuation attached to
                            neighbouring tokens.

All three share a ``Shaker`` base, a ``season(line) -> str`` per-line
hook, and plug into :class:`hypernix.sink.Sink.pour` like the pans.
"""
from __future__ import annotations

import random
import string
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

_PRINTABLE = string.ascii_letters + string.digits + " "


@dataclass
class Shaker:
    """Abstract seasoning applicator.  Subclasses override :meth:`season`."""

    source: Path | str | Iterable[str]
    rate: float = 0.1
    seed: int = 0
    name: ClassVar[str] = "Shaker"
    _rng: random.Random = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self._rng = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret

    def _source_lines(self) -> Iterator[str]:
        if isinstance(self.source, str | Path):
            with Path(self.source).open(encoding="utf-8") as f:
                for line in f:
                    yield line.rstrip("\n")
        else:
            yield from self.source

    def season(self, line: str) -> str:
        return line

    def __iter__(self) -> Iterator[str]:
        for line in self._source_lines():
            yield self.season(line)


# ---------------------------------------------------------------------------
# t1 — FromTheBag
# ---------------------------------------------------------------------------

@dataclass
class FromTheBag(Shaker):
    """Cheapest tier.  Per-character substitution at ``rate`` with a
    random printable ASCII character.  Preserves line length."""

    name: ClassVar[str] = "FromTheBag"

    def season(self, line: str) -> str:
        if not line:
            return line
        out = []
        for ch in line:
            if ch != "\n" and self._rng.random() < self.rate:
                out.append(self._rng.choice(_PRINTABLE))
            else:
                out.append(ch)
        return "".join(out)


# ---------------------------------------------------------------------------
# t2 — HandCrusher
# ---------------------------------------------------------------------------

@dataclass
class HandCrusher(Shaker):
    """Middle tier.  Token-level adjacent-swap at ``rate``.  Produces
    lines that are still mostly readable but structurally shuffled."""

    name: ClassVar[str] = "HandCrusher"

    def season(self, line: str) -> str:
        tokens = line.split()
        if len(tokens) < 2:
            return line
        i = 0
        while i < len(tokens) - 1:
            if self._rng.random() < self.rate:
                tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
                i += 2
            else:
                i += 1
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# t3 — PoshSaltDish
# ---------------------------------------------------------------------------

@dataclass
class PoshSaltDish(Shaker):
    """Finest tier.  Word-aware drop / duplicate / swap at independent
    rates.  Keeps attached punctuation with its neighbour so commas
    and periods don't end up orphaned."""

    drop_rate: float = 0.03
    duplicate_rate: float = 0.02
    swap_rate: float = 0.03
    name: ClassVar[str] = "PoshSaltDish"

    def __post_init__(self) -> None:
        super().__post_init__()
        for r in (self.drop_rate, self.duplicate_rate, self.swap_rate):
            if not 0.0 <= r <= 1.0:
                raise ValueError("posh rates must be in [0, 1]")

    def season(self, line: str) -> str:
        tokens = line.split()
        if not tokens:
            return line
        out: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            r = self._rng.random()
            if r < self.drop_rate:
                i += 1
                continue
            if r < self.drop_rate + self.duplicate_rate:
                out.append(tok)
                out.append(tok)
                i += 1
                continue
            if (
                i + 1 < len(tokens)
                and r < self.drop_rate + self.duplicate_rate + self.swap_rate
            ):
                out.append(tokens[i + 1])
                out.append(tok)
                i += 2
                continue
            out.append(tok)
            i += 1
        return " ".join(out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TIERS: dict[str, type[Shaker]] = {
    "from-the-bag": FromTheBag,
    "hand-crusher": HandCrusher,
    "posh-salt-dish": PoshSaltDish,
}


def salt_shaker(tier: str, source, **kw):
    """Construct a salt-shaker tier by short name."""
    key = tier.lower().replace("_", "-")
    if key not in TIERS:
        raise ValueError(
            f"unknown salt tier {tier!r}; valid: {sorted(TIERS)}"
        )
    return TIERS[key](source=source, **kw)
