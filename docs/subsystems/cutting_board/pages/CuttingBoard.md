# Cutting Board — `hypernix.cutting_board`

A cutting board is where you portion ingredients before they go to the
pan. `cutting_board` takes a corpus (file or iterable) and splits it
deterministically into train/val/test slices, either as in-memory lists
or written to per-split files.

## `CuttingBoard(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0, shuffle=True)`

Deterministic random split of a flat list of strings.

```python
from hypernix.cutting_board import CuttingBoard

board = CuttingBoard(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
splits = board.slice("corpus.txt")   # {"train": [...], "val": [...], "test": [...]}
paths = board.slice_to_files("corpus.txt", "out/")  # writes out/train.txt, out/val.txt, out/test.txt
```

| Method | Signature | Notes |
|---|---|---|
| `.slice(source)` | `(Path\|str\|Iterable[str]) -> dict[str, list[str]]` | Reads lines, shuffles (if `shuffle=True`) via `random.Random(seed)`, then cuts at `int(n*train_r)` and `int(n*train_r) + int(n*val_r)`. |
| `.slice_to_files(source, out_dir, *, suffix=".txt")` | `-> dict[str, Path]` | Calls `.slice()` then writes each split to `out_dir/{split}{suffix}`, one line per row. |

Ratios are validated in `__post_init__`: each must be `>= 0`
(`ValueError` otherwise) and at least one must be `> 0`. They don't need
to sum to `1.0` — `_normalise()` renormalizes proportionally at split
time. `test_ratio=0.0` is valid (gives train+val, empty test slice).
Very small inputs (< 3 records) still split without error; some slices
may end up empty.

## `StratifiedBoard(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0, label_key="label", fallback="__missing__")`

Splits a list of **dict records** while preserving the distribution of
`label_key` across train/val/test — use on labelled judge corpora where
every record has e.g. `{"label": "GOOD"}` / `{"label": "BAD"}`.

```python
from hypernix.cutting_board import StratifiedBoard

splits = StratifiedBoard(label_key="label").slice(judge_records)
```

Groups records by `record.get(label_key, fallback)`, shuffles and splits
**each group independently** at the same ratios, concatenates all
groups' train/val/test pieces, then shuffles each combined split once
more (so the output isn't grouped by class). Groups are iterated in
sorted-key order for deterministic behavior given the same seed.

**Note:** unlike `CuttingBoard`, `StratifiedBoard` has no `__post_init__`
validation — passing negative or all-zero ratios won't raise until the
division happens, and may silently produce empty or nonsensical splits.

## Module-level shortcuts

```python
from hypernix.cutting_board import cutting_board, stratified_split

splits = cutting_board("corpus.txt", train=0.8, val=0.1, test=0.1, seed=1)
board = cutting_board(train=0.9, val=0.05, test=0.05)   # no source -> returns a reusable CuttingBoard

strat_splits = stratified_split(records, train=0.8, val=0.1, test=0.1, label_key="label")
```

`cutting_board(source=None, *, train=0.8, val=0.1, test=0.1, seed=0, shuffle=True)`
— if `source` is given, returns the slice dict directly; if omitted,
returns a configured `CuttingBoard` instance for repeated use.

`stratified_split(records, *, train=0.8, val=0.1, test=0.1, seed=0, label_key="label")`
— one-shot wrapper around `StratifiedBoard(...).slice(records)`.

### Required modules

Standard library only — `random`, `dataclasses`, `pathlib`, `collections.abc`, `typing`.


> **Canonical source:** `src/hypernix/data/cutting_board.py`. The `hypernix.cutting_board` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. The split is deterministic for the same inputs, ratios, and seed; it is a dataset splitting utility, not a tokenizer-aware sequence packer.

---

## See also

- [Pans](Pans.md) — typical upstream cleaning step before splitting
- `hypernix.mediocre_fridge` — judge-corpus generation, the typical source of `StratifiedBoard`'s labelled records
