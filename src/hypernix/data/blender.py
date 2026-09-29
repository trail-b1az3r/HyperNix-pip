"""blender — mix multiple data streams into one corpus.

Given a set of input streams (file paths or iterables of strings),
a blender yields lines from all of them at some blend ratio.  Four
tiers, from a hand whisk up to a commercial blender:

* :class:`HandBlender`       — straight concatenation, no interleave.
* :class:`PersonalBlender`   — round-robin interleave.
* :class:`CountertopBlender` — weighted random sampling (keeps the
                                specified ratio in expectation).
* :class:`HighPowerBlender`  — reservoir-shuffled output of all
                                sources.  Near-uniform; buffers in
                                RAM.

All four yield strings and plug into :class:`hypernix.sink.Sink.pour`.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


def _open_stream(source: Path | str | Iterable[str]) -> Iterator[str]:
    if isinstance(source, str | Path):
        with Path(source).open(encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
    else:
        yield from source


# ---------------------------------------------------------------------------
# Tier 1 — HandBlender
# ---------------------------------------------------------------------------

@dataclass
class HandBlender:
    """Straight concatenation.  Source A, then source B, then C, …"""

    sources: list[Path | str | Iterable[str]]
    name: str = "HandBlender"

    def __iter__(self) -> Iterator[str]:
        for s in self.sources:
            yield from _open_stream(s)


# ---------------------------------------------------------------------------
# Tier 2 — PersonalBlender (round-robin)
# ---------------------------------------------------------------------------

@dataclass
class PersonalBlender:
    """Round-robin interleave.  Pulls one line from each source in
    order, repeating until every source is exhausted."""

    sources: list[Path | str | Iterable[str]]
    name: str = "PersonalBlender"

    def __iter__(self) -> Iterator[str]:
        iters = [iter(_open_stream(s)) for s in self.sources]
        alive = list(range(len(iters)))
        while alive:
            next_alive = []
            for i in alive:
                try:
                    yield next(iters[i])
                    next_alive.append(i)
                except StopIteration:
                    pass
            alive = next_alive


# ---------------------------------------------------------------------------
# Tier 3 — CountertopBlender (weighted sampling)
# ---------------------------------------------------------------------------

@dataclass
class CountertopBlender:
    """Weighted random sampling with replacement.

    ``weights`` is a list of nonnegative floats — probability of
    drawing from each source per step.  Useful when you want "70%
    scraped, 30% curated" in the final mix.
    """

    sources: list[Path | str | Iterable[str]]
    weights: list[float] | None = None
    seed: int = 0
    name: str = "CountertopBlender"

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
        iters = [iter(_open_stream(s)) for s in self.sources]
        weights = list(self.weights) if self.weights else [1.0] * len(iters)
        if len(weights) != len(iters):
            raise ValueError("len(weights) != len(sources)")
        alive = [i for i in range(len(iters))]
        while alive:
            idx = rng.choices(alive, weights=[weights[i] for i in alive])[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                alive.remove(idx)


# ---------------------------------------------------------------------------
# Tier 4 — HighPowerBlender (buffered shuffle)
# ---------------------------------------------------------------------------

@dataclass
class HighPowerBlender:
    """Buffers every source into RAM, concatenates, then shuffles.
    Near-uniform output distribution — the strongest homogenization
    available.  Memory footprint scales with the total input size,
    so for multi-GB corpora prefer :class:`CountertopBlender`."""

    sources: list[Path | str | Iterable[str]]
    seed: int = 0
    name: str = "HighPowerBlender"
    _buffered: list[str] = field(default_factory=list, repr=False)

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
        pool: list[str] = []
        for s in self.sources:
            pool.extend(_open_stream(s))
        rng.shuffle(pool)
        self._buffered = pool
        yield from pool


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TIERS: dict[str, type] = {
    "hand-blender": HandBlender,
    "personal-blender": PersonalBlender,
    "countertop-blender": CountertopBlender,
    "high-power-blender": HighPowerBlender,
}


def blender(tier: str, sources: list[Path | str | Iterable[str]], **kw):
    """Construct a blender tier by name."""
    cls = TIERS[tier.lower().replace("_", "-")]
    return cls(sources=sources, **kw)
