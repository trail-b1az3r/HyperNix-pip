"""cutting_board — train / val / test splitting.

A cutting board is where you portion ingredients before they go to
the pan.  Same idea here: take a single corpus file (or any
iterable of strings / dicts), split it deterministically into
train / val / test slices according to ratio, write each slice to
its own file or hand back as in-memory lists.

Two modes:

* :class:`CuttingBoard`     — deterministic, seed-based split.
                              ``slice("corpus.txt")`` returns
                              ``{"train": [...], "val": [...],
                              "test": [...]}``.
* :class:`StratifiedBoard`  — stratified by a label column so each
                              split keeps the original class
                              distribution.  Use on labelled judge
                              corpora (where every record has a
                              ``label`` field with values in
                              ``{"GOOD", "BAD"}``).

Edge cases:

* ratios are renormalised if they don't sum to 1.0,
* ``test_ratio`` may be 0.0 — you'll get train + val and an empty
  test slice,
* tiny inputs (< 3 records) still split: every slice gets at most
  one record, and empty slices land empty.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _open_stream(source: Path | str | Iterable[str]) -> Iterable[str]:
    if isinstance(source, str | Path):
        return Path(source).read_text(encoding="utf-8").splitlines()
    return list(source)


@dataclass
class CuttingBoard:
    """Deterministic random split."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        for r in (self.train_ratio, self.val_ratio, self.test_ratio):
            if r < 0.0:
                raise ValueError("ratios must be >= 0")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total <= 0:
            raise ValueError("at least one of train/val/test ratio must be > 0")

    def _normalise(self) -> tuple[float, float, float]:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        return (
            self.train_ratio / total,
            self.val_ratio / total,
            self.test_ratio / total,
        )

    def slice(
        self, source: Path | str | Iterable[str],
    ) -> dict[str, list[str]]:
        rows = list(_open_stream(source))
        if self.shuffle:
            rng = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
            rng.shuffle(rows)
        n = len(rows)
        train_r, val_r, _ = self._normalise()
        i_train = int(n * train_r)
        i_val = i_train + int(n * val_r)
        return {
            "train": rows[:i_train],
            "val": rows[i_train:i_val],
            "test": rows[i_val:],
        }

    def slice_to_files(
        self,
        source: Path | str | Iterable[str],
        out_dir: Path | str,
        *,
        suffix: str = ".txt",
    ) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        slices = self.slice(source)
        paths: dict[str, Path] = {}
        for split, rows in slices.items():
            p = out / f"{split}{suffix}"
            p.write_text(
                "\n".join(rows) + ("\n" if rows else ""),
                encoding="utf-8",
            )
            paths[split] = p
        return paths


# ---------------------------------------------------------------------------
# Stratified
# ---------------------------------------------------------------------------

@dataclass
class StratifiedBoard:
    """Split a list of dict records while preserving the
    distribution of ``label_key``.

    Each unique label gets independently shuffled and split, then
    the per-class slices are concatenated and shuffled once more
    so the output isn't grouped by class.
    """

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0
    label_key: str = "label"
    fallback: str = "__missing__"

    def slice(
        self, records: Iterable[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            lbl = r.get(self.label_key, self.fallback)
            groups.setdefault(lbl, []).append(r)

        rng = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
        train: list[dict] = []
        val: list[dict] = []
        test: list[dict] = []
        total = self.train_ratio + self.val_ratio + self.test_ratio
        train_r = self.train_ratio / total
        val_r = self.val_ratio / total
        for _, group in sorted(groups.items()):
            rng.shuffle(group)
            n = len(group)
            i_train = int(n * train_r)
            i_val = i_train + int(n * val_r)
            train.extend(group[:i_train])
            val.extend(group[i_train:i_val])
            test.extend(group[i_val:])
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def cutting_board(
    source: Path | str | Iterable[str] | None = None,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    shuffle: bool = True,
) -> CuttingBoard | dict[str, list[str]]:
    """If ``source`` is given, return the slice dict directly; else
    return a configured :class:`CuttingBoard` for repeated use."""
    board = CuttingBoard(
        train_ratio=train, val_ratio=val, test_ratio=test,
        seed=seed, shuffle=shuffle,
    )
    if source is None:
        return board
    return board.slice(source)


def stratified_split(
    records: Iterable[dict[str, Any]],
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    label_key: str = "label",
) -> dict[str, list[dict[str, Any]]]:
    return StratifiedBoard(
        train_ratio=train, val_ratio=val, test_ratio=test,
        seed=seed, label_key=label_key,
    ).slice(records)
