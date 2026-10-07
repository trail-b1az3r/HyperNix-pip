"""Flattening in Python 3.15 syntax: PEP 798, unpacking in comprehensions.

Imported only on 3.15+; 3.12-3.14 cannot parse this file (see
:mod:`hypernix._compat`). Same API and results as ``_iter_legacy``: one
comprehension instead of ``itertools.chain`` and a manual update loop.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def flatten[T](iterables: Iterable[Iterable[T]]) -> list[T]:
    """Every item of every iterable, in order, as one list."""
    return [*items for items in iterables]


def union[T](iterables: Iterable[Iterable[T]]) -> set[T]:
    """Every item of every iterable, as one set."""
    return {*items for items in iterables}


def merge[K, V](mappings: Iterable[Mapping[K, V]]) -> dict[K, V]:
    """The mappings combined left to right; a later key wins."""
    return {**mapping for mapping in mappings}
