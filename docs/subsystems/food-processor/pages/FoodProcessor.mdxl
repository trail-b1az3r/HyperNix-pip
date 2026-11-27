# Food Processor — `hypernix.food_processor`

A food processor turns bulky ingredients into ready-to-use pieces.
`food_processor` is the bulk-text counterpart to the per-line
[toaster](Toaster.md) and per-file [pan](Pans.md): it takes one large
text blob and chunks it into training-ready pieces. Four blade tiers:

| Tier | Class | Strategy |
|---|---|---|
| 1 | `ChopBlade` | Splits on a separator (blank line by default). Fastest; preserves document order. |
| 2 | `SliceBlade` | Fixed-length **character** slicing, with optional overlap. |
| 3 | `ShredBlade` | Whitespace-tokenized sliding window, with configurable stride. |
| 4 | `PureeBlade` | The whole file as a single output, with internal whitespace collapsed. |

Each blade yields strings and pairs naturally with
[`Sink.pour()`](Pans.md) for on-disk output.

## `ChopBlade(source, separator="\n\n")`

Splits the file text on `separator`, strips each piece, yields
non-empty pieces only.

## `SliceBlade(source, slice_chars=1024, overlap_chars=0)`

```python
from hypernix.food_processor import SliceBlade
chunks = list(SliceBlade("book.txt", slice_chars=2048, overlap_chars=256))
```

Fixed-length character slicing — good for turning one huge document
into training chunks of a known size. Validates:
- `slice_chars > 0` (else `ValueError`)
- `overlap_chars >= 0` (else `ValueError`)
- `overlap_chars < slice_chars` (else `ValueError` — this specific check
  was added because `overlap_chars >= slice_chars` would otherwise make
  the window step by as little as 1 character, silently emitting
  near-duplicate windows without making forward progress)

Steps by `slice_chars - overlap_chars`; the final partial window at the
end of the text is still emitted if non-blank.

## `ShredBlade(source, window_tokens=256, stride_tokens=128)`

Whitespace-tokenized sliding window (`text.split()`, not a real
tokenizer — a cheap proxy). Validates `window_tokens > 0` (else
`ValueError`); `stride_tokens` is floored to at least 1 internally
(`step = max(1, stride_tokens)`) so a `stride_tokens=0` can't cause an
infinite loop. Yields space-joined windows; stops once the final window
would run past the end of the token list.

## `PureeBlade(source)`

The whole file as one output, with all internal whitespace runs
collapsed to a single space (`re.sub(r"\s+", " ", text).strip()`). Use
when you want a training example that's a single cleaned blob rather
than multiple pieces. Yields nothing if the cleaned result is empty.

## Factory

```python
from hypernix.food_processor import food_processor
pieces = food_processor("slice", "book.txt", slice_chars=2048)
```

`food_processor(tier, source, **kw)` — `tier` case-insensitive, `_`
normalized to `-`. Valid tiers (also in `TIERS: dict[str, type]`):
`"chop"`, `"slice"`, `"shred"`, `"puree"`. Unknown tier raises a plain
`KeyError`; bad kwargs raise the raw dataclass `TypeError`.

### Required modules

Standard library only — `re`, `dataclasses`, `pathlib`, `collections.abc`.


> **Canonical source:** `src/hypernix/data/food_processor.py`. The `hypernix.food_processor` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. `ShredBlade` windows are whitespace-split word windows, not model-tokenizer windows; use the target model's tokenizer separately when exact token counts matter.

---

## See also

- [Pans](Pans.md) — where blade output typically gets written via `Sink.pour()`
- [Toaster](Toaster.md) — per-line formatting instead of bulk chunking
- `hypernix.pans.Pan.max_chars` / `context_length` — a cheaper truncate-only alternative when you don't need precise chunking
