# Pans & Sink — `hypernix.pans`, `hypernix.sink`

A *pan* reads text and yields processed strings; like cookware, each pan
applies more heat (transformation) than the one before it. A *sink* is
the terminal node of the pipeline — it takes an iterable of strings (or
dicts) and appends them to a file. Chain the two together for the
standard "clean raw text → training file" pipeline.

```python
from hypernix.pans import FryingPan
from hypernix.sink import Sink

Sink("clean.txt").pour(FryingPan("raw.txt"))
```


> **Canonical source:** `src/hypernix/data/pans.py and src/hypernix/data/sink.py`. The `hypernix.pans / hypernix.sink` import spelling used in examples is the compatibility-facing API; the source path above is the implementation to check when verifying behavior. The sink writes each item immediately; rotation and deduplication are per-process/in-memory helpers, not a distributed deduplication service.

---

## `hypernix.pans` — five progressive tiers

| Tier | Class | What it adds |
|---|---|---|
| 1 (lightest) | `FryingPan` | Strips trailing whitespace only. Use when input is already clean. |
| 2 | `SaucePan` | + collapses internal whitespace runs, strips lead/trail, drops empty lines. Standard mild cleaning for scraped/OCR'd text. |
| 3 | `Skillet` | + chat/instruction formatting — wraps each line in role tags. |
| 4 | `GrillPan` | + SHA1-hash deduplication + minimum-length filter. Safe on arbitrary web-scraped text. |
| 5 (heaviest) | `Wok` | + buffers the whole source in memory, shuffles it, and optionally injects line-reversal augmentation. |

Every pan supports the same iterator protocol (`__iter__`), so callers
can swap tiers by name without changing calling code.

### `Pan` (base class)

```python
@dataclass
class Pan:
    source: Path | str | Iterable[str]
    max_chars: int | None = None       # keyword-only
    context_length: int | None = None  # keyword-only
```

| Field | Notes |
|---|---|
| `source` | A file path or any iterable of strings. |
| `max_chars` | Hard per-line character cap; longer lines are **truncated**, not split. |
| `context_length` | Convenience for training-script callers — treated as an approximate token budget, converted to `max_chars = context_length * 4` (English-BPE heuristic). Takes precedence over `max_chars` when both are set. For precise chunking by tokens/chars, use `hypernix.food_processor` (`SliceBlade` / `ShredBlade`) instead. |

`name` is deliberately a `ClassVar`, not a dataclass field — making it a
field would expose it as the *second positional argument* of every
subclass (`Skillet("file.txt", "instruct")` would silently set
`name="instruct"` instead of `mode="instruct"`).

Subclasses override either `.cook(line) -> str | None` (one line in,
zero-or-one line out — return `None` to drop a line) or `.iter()`
directly for full control (only `Wok` does this, since shuffling needs
to see the whole buffer at once).

### Tier details

**`FryingPan`** — `cook()` is just `line.rstrip()`.

**`SaucePan`** — `cook()` collapses `[ \t]+` runs to a single space,
strips the line, returns `None` if the result is empty.

**`Skillet(mode="chat", user_tag="<USER>", assistant_tag="<ASSISTANT>")`**
— `mode="chat"` yields `"{user_tag} {line}\n{assistant_tag}"`;
`mode="instruct"` yields `"### Instruction: {line}\n### Response:"`. Use
when every input line is a user turn paired with an assistant reply
elsewhere (following line or separate corpus).

**`GrillPan(min_chars=8)`** — SaucePan-style cleaning, then drops lines
shorter than `min_chars`, then drops lines whose SHA1 hash has already
been seen this run (`_seen` is internal state, not part of `__init__`).

**`Wok(seed=0, reverse_ratio=0.0)`** — overrides `.iter()` entirely:
pre-cleans with SaucePan semantics into an in-memory buffer, shuffles
with `random.Random(seed)`, then for each line, with probability
`reverse_ratio`, emits it with word order reversed (`" ".join(reversed(line.split()))`).

### Factory

```python
from hypernix.pans import pick_pan

pan = pick_pan("grill-pan", "raw.txt", min_chars=12)
```

`pick_pan(tier, source, **kwargs)` — `tier` is case-insensitive with `_`
normalized to `-` (`"grill_pan"` and `"grill-pan"` both work). Valid
tiers: `"frying-pan"`, `"sauce-pan"`, `"skillet"`, `"grill-pan"`,
`"wok"` (also in `TIERS: dict[str, type[Pan]]`). Raises `ValueError`
listing valid tiers if `tier` is unknown, or listing the tier's valid
kwargs (introspected via `inspect.signature`) if `**kwargs` passes
something the selected tier doesn't accept.

### Required modules

Standard library only — `hashlib`, `random`, `re`, `dataclasses`,
`pathlib`, `collections.abc`. No HyperNix internal dependencies.

---

## `hypernix.sink` — the terminal write step

### `Sink` (dataclass)

```python
@dataclass
class Sink:
    path: Path | str
    rotate_bytes: int | None = None
    dedupe: bool = False
```

| Field | Notes |
|---|---|
| `path` | Base output path. |
| `rotate_bytes` | If set > 0, rolls over to `path.1`, `path.2`, … once the current file passes this many written bytes — useful for a 24/7 scraper that would otherwise fill a disk. |
| `dedupe` | If `True`, keeps a running in-memory SHA1 set and skips lines already written this run. |

### Methods

| Method | Signature | Notes |
|---|---|---|
| `.write(line)` | `(str) -> bool` | Appends `line` (newline added if missing). Opens/closes the file per write for crash-safety. Returns `False` only when `dedupe=True` suppressed a duplicate, else `True`. Advances rotation once `rotate_bytes` is crossed. |
| `.pour(iterable)` | `(Iterable[str]) -> Path` | Writes every item (via `str(item)`) and returns the **current** path at call time — doesn't reflect further rotation that happens mid-iteration. |
| `.write_json(obj)` | `(dict) -> bool` | Writes `json.dumps(obj, ensure_ascii=False, separators=(",", ":"))` via `.write()`. |
| `.drain()` | `() -> None` | No-op — present so callers don't have to think about whether the sink is buffered (it isn't; every write is immediately flushed to disk). |
| `.close()` | `() -> None` | Calls `.drain()`. |

`Sink` supports the context-manager protocol (`__enter__`/`__exit__`),
though since there's no buffering to flush, using it as a `with` block
is purely for symmetry/readability, not correctness.

### Required modules

Standard library only — `hashlib`, `json`, `dataclasses`, `pathlib`.

---

## See also

- `hypernix.food_processor` — for precise token/char chunking instead of `Pan.max_chars`' truncate-only behavior
- `hypernix.qa` — `QAProcessor` output can be piped straight into a `Sink`
- [Kitchen](Kitchen.md) — general data-prep subsystem overview
