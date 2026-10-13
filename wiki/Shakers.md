# Salt Shaker & Pepper Shaker — `hypernix.salt_shaker`, `hypernix.pepper_shaker`

Salt is subtle; pepper bites. Both modules perturb training examples for
augmentation, sharing one `Shaker` base class (`pepper_shaker` literally
imports and subclasses `salt_shaker.Shaker`). Salt shakers keep meaning
intact (robustness without corruption); pepper shakers deliberately
change meaning or add difficulty (hard-negative mining, MLM-style
masking, negation training).

Both plug into [`Sink.pour()`](Pans.md) like `pans`.


> **Canonical source:** `src/hypernix/data/salt_shaker.py and src/hypernix/data/pepper_shaker.py`. The `hypernix.salt_shaker / hypernix.pepper_shaker` import spelling used in examples is the compatibility-facing API; the source path above is the implementation to check when verifying behavior. These are text-level augmentation helpers, not tokenizer-aware or semantics-preserving transformations; inspect generated samples before adding them to a training corpus.

---

## `Shaker` (base class, in `salt_shaker.py`)

```python
@dataclass
class Shaker:
    source: Path | str | Iterable[str]
    rate: float = 0.1
    seed: int = 0
```

`rate` must be in `[0, 1]` (`ValueError` in `__post_init__` otherwise).
`__post_init__` also seeds `self._rng = random.Random(seed)`. Subclasses
override `.season(line) -> str`; `Shaker.__iter__` reads each source
line and yields `self.season(line)`.

---

## `hypernix.salt_shaker` — three gentle tiers

| Tier | Class | Mechanism |
|---|---|---|
| 1 (coarsest) | `FromTheBag` | Per-character substitution at `rate` with a random printable ASCII char (`string.ascii_letters + digits + " "`). Preserves line length. |
| 2 | `HandCrusher` | Token-level adjacent swap at `rate` — splits on whitespace, walks left-to-right, and with probability `rate` swaps a token with its right neighbor, then skips past the swapped pair (so no token is swapped twice in one pass). |
| 3 (finest) | `PoshSaltDish` | Word-aware drop / duplicate / swap at three independent rates. |

### `FromTheBag(rate=0.1)`

```python
from hypernix.salt_shaker import FromTheBag
list(FromTheBag(["hello world"], rate=0.1, seed=1))
```

### `HandCrusher(rate=0.1)`

Structurally shuffles while staying mostly readable. Lines with fewer
than 2 tokens pass through unchanged.

### `PoshSaltDish(drop_rate=0.03, duplicate_rate=0.02, swap_rate=0.03)`

For each token, draws one random number `r` and takes the **first**
matching bucket in order — drop, then duplicate, then swap (with the
next token) — falling through to "keep unchanged" if none hit. Because
the buckets are checked as cumulative thresholds (`r < drop_rate`, then
`r < drop_rate + duplicate_rate`, then `r < drop_rate + duplicate_rate +
swap_rate`), the three rates are **not fully independent per-token
probabilities** — effectively at most one of the three operations can
apply to a given token per pass. All three rates individually validated
to be in `[0, 1]` in `__post_init__`.

### Factory

```python
from hypernix.salt_shaker import salt_shaker
s = salt_shaker("hand-crusher", "raw.txt", rate=0.15, seed=7)
```

`salt_shaker(tier, source, **kw)` — case-insensitive, `_` → `-`. Valid
tiers (also `TIERS: dict[str, type[Shaker]]`): `"from-the-bag"`,
`"hand-crusher"`, `"posh-salt-dish"`. Unknown tier raises `ValueError`
listing valid tiers.

---

## `hypernix.pepper_shaker` — three sharp tiers

| Tier | Class | Mechanism |
|---|---|---|
| 1 (coarsest) | `SmallShaker` | Random whole-word masking — replaces tokens with `mask_token` at `rate`. |
| 2 | `Dish` | Typo injection — drop or duplicate one internal character per affected word. |
| 3 (finest) | `TallHandmade` | Negation injection — prepends `negator` before a random token at `rate`. |

### `SmallShaker(mask_token="[MASK]", rate=0.1)`

Each whitespace-split token independently replaced with `mask_token`
with probability `rate`.

### `Dish(rate=0.1)`

```python
from hypernix.pepper_shaker import Dish
list(Dish(["the quick brown fox"], rate=0.3, seed=1))
```

For each affected word (`rate` chance): if `len(word) < 3`, left
unchanged (too short to typo safely). Otherwise picks `drop` or
`duplicate` at random, picks an internal index (never the first or last
character — so both ends stay recognizable, the classic "jumbled
letters still readable" effect), and either removes that character or
duplicates it in place.

### `TallHandmade(negator="NOT", rate=0.1)`

For each token, with probability `rate`, inserts `negator` immediately
before it. Useful for hard-negative mining and negation-robustness
training (judges, entailment models).

### Factory

```python
from hypernix.pepper_shaker import pepper_shaker
p = pepper_shaker("dish", "raw.txt", rate=0.2)
```

`pepper_shaker(tier, source, **kw) -> Shaker` — case-insensitive, `_` →
`-`. Valid tiers (also `TIERS: dict[str, type[Shaker]]`):
`"small-shaker"`, `"dish"`, `"tall-handmade"`. Unknown tier raises
`ValueError` listing valid tiers.

There's also a `_iter_source(_src)` helper re-exported for API symmetry
with `salt_shaker.Shaker._source_lines` (which is private) — it's not
used internally by any `Shaker` subclass in this file and exists purely
for callers who want to iterate a raw source without constructing a
`Shaker` at all.

### Required modules

Both modules: standard library only — `random` (`salt_shaker` only),
`string` (`salt_shaker` only), `dataclasses`, `collections.abc`,
`pathlib` (`salt_shaker` only). `pepper_shaker` imports `Shaker` from
`salt_shaker` — the two modules are not independently importable if you
want `pepper_shaker`'s classes (they'll pull in `salt_shaker` as a
dependency automatically).

---

## See also

- [Pans](Pans.md) — the `Sink.pour()` these shakers typically feed into
- `hypernix.qa.QAProcessor` — accepts both shaker types via `salt_shaker=`/`pepper_shaker=`, applying pepper before salt
- `hypernix.deep_fryer` — weight-side perturbation, the model-side analog of these data-side shakers
