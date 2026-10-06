"""Flattening for Python 3.12-3.14 (see :mod:`hypernix._compat`)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import chain


def flatten[T](iterables: Iterable[Iterable[T]]) -> list[T]:
    """Every item of every iterable, in order, as one list."""
    return list(chain.from_iterable(iterables))


def union[T](iterables: Iterable[Iterable[T]]) -> set[T]:
    """Every item of every iterable, as one set."""
    return set(chain.from_iterable(iterables))


def merge[K, V](mappings: Iterable[Mapping[K, V]]) -> dict[K, V]:
    """The mappings combined left to right; a later key wins."""
    out: dict[K, V] = {}
    for mapping in mappings:
        out.update(mapping)
    return out
