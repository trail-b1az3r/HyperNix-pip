# Smoker — `hypernix.smoker`

A smoker is the opposite of a microwave: give it time and ingredients
and it rewards patience with depth of flavor. `smoker` wraps
the oven object's `.train()` method with progressively more
sophisticated training recipes. Every tier exposes
`smoke(dataset, out_dir) -> Path` with the same signature, so a smoker
drops in anywhere you were calling the supplied oven's `.train()` method directly.

## Four tiers, ascending quality (and cost)

| Tier | Class | Adds |
|---|---|---|
| 1 | `UseableSmoker` | Nothing — straight pass-through to `oven.train(**kwargs)`. No scheduler, no EMA, no validation. |
| 2 | `GoodSmoker` | A warmup → plateau → cooldown LR shape (approximated as a single effective LR — see caveat below). |
| 3 | `CommercialSmoker` (subclasses `GoodSmoker`) | + EMA (exponential moving average) of weights, blended post-hoc. |
| 4 | `HighQualitySmoker` (subclasses `CommercialSmoker`) | + progressive context-length curriculum across sub-runs. |

## `Smoker` (base class)

| Field | Type | Default |
|---|---|---|
| `oven` | `Any` | required |
| `steps` | `int` | `500` |
| `batch_size` | `int` | `1` |
| `context_length` | `int` | `512` |
| `lr` | `float` | `3e-4` |
| `weight_decay` | `float` | `0.1` |
| `grad_clip` | `float` | `1.0` |
| `log_every` | `int` | `10` |
| `save_every` | `int` | `0` |
| `seed` | `int \| None` | `None` |
| `quiet` | `bool` | `True` |
| `history` | `list[dict]` | `[]` |

`_common_kwargs()` bundles all of the above (except `oven`/`history`)
into a dict forwarded to `oven.train()`.

## Tier 1 — `UseableSmoker`

```python
from hypernix.smoker import useable_smoker
path = useable_smoker(oven, steps=1000).smoke("data.jsonl", "out/")
```

`.smoke()` is a direct call to `oven.train(dataset, out_dir,
**self._common_kwargs())`. Use this when you want `smoker` semantics
(one call → trained snapshot) with zero extra machinery.

## Tier 2 — `GoodSmoker(warmup_frac=0.1, cooldown_frac=0.2)`

**Caveat, stated plainly in the source itself:** this tier doesn't
actually implement a real step-by-step LR scheduler — it computes a
single "effective LR" heuristic (a weighted average across the warmup
/ plateau / cooldown phase lengths, with cooldown weighted at 0.5x
peak) and passes that one flat value to `oven.train()`. The comment in
the code explains this is a workaround because the underlying trainer
doesn't expose a scheduler hook — real per-step scheduling would
require hand-rolling the training loop with `PressureCooker` directly.
`history` records `{warmup, plateau, cooldown, effective_lr, out}` per
run.

## Tier 3 — `CommercialSmoker(ema_decay=0.95, validation_steps=0)`

Runs the parent (`GoodSmoker`) training, then blends pre- and
post-training state dicts as a **one-shot approximation of EMA**:
`ema = pre * ema_decay + post * (1 - ema_decay)`. This is explicitly
described in the source as an approximation of a proper step-by-step
EMA, "well enough for our small models" — not a running per-step
average. The blended weights are re-saved to `out_dir` via
`hypernix.train.save_snapshot`. `validation_steps=0` means validation
is skipped (the field exists but periodic validation isn't wired into
`.smoke()` in this tier).

## Tier 4 — `HighQualitySmoker(base_context_length=128, patience=2)`

```python
from hypernix.smoker import high_quality_smoker
path = high_quality_smoker(oven, steps=1500, context_length=2048).smoke("data.jsonl", "out/")
```

Trains in ascending context-length phases: `sorted({base_context_length,
2*base_context_length, context_length})`, splitting `steps` evenly
across however many distinct phase values result (fewer than 3 phases
if the schedule values collapse via the `set()` dedup, e.g. when
`context_length <= 2 * base_context_length`). Each phase runs as a
`CommercialSmoker` internally (via `copy.copy(self)` + reassigning
`__class__` to bypass infinite recursion into `HighQualitySmoker.smoke`
itself) into its own `out_dir/phase_N_ctxM/` subdirectory, then the
final phase's weights are re-saved directly under the outer `out_dir`.

**Caveat:** the docstring describes early-stopping "if validation loss
hasn't improved in `patience` sub-runs," but the current `.smoke()`
implementation does not check validation loss or `patience` anywhere —
all phases always run to completion. Treat `patience` as currently
inert; the field exists but has no effect yet.

## Factory and shortcuts

```python
from hypernix.smoker import smoker
s = smoker("commercial", oven, ema_decay=0.9)
```

`smoker(tier, oven, **kw)` — `tier` case-insensitive, `_` normalized to
`-`. Valid tiers (also in `TIERS: dict[str, type[Smoker]]`):
`"useable"`, `"good"`, `"commercial"`, `"high-quality"`. Unknown tier
raises a plain `KeyError`.

Dedicated shortcuts: `useable_smoker(oven, **kw)`, `good_smoker(oven,
**kw)`, `commercial_smoker(oven, **kw)`, `high_quality_smoker(oven, **kw)`.

### Required modules

- `torch` — only imported inside `CommercialSmoker.smoke()` (lazy, for the EMA blend)
- `hypernix.train.save_snapshot` (internal) — imported lazily inside `CommercialSmoker.smoke()` / `HighQualitySmoker.smoke()`
- Standard library: `copy`, `dataclasses`, `pathlib`


> **Canonical source:** `src/hypernix/training/smoker.py`. The `hypernix.smoker` import spelling used in examples is the compatibility-facing API; the source path above is the implementation to check when verifying behavior. Treat the described effective-LR and post-hoc EMA behavior as the checked-in implementation: neither is a true per-step LR schedule or per-step EMA. `patience` is documented as inert until the code implements early stopping.

---

## See also

- [Training](Training.md) — `oven.train()` / `hypernix.train` internals this module wraps
- `hypernix.pressure_cooker` — the real optimizer with genuine step-by-step scheduling, for when `GoodSmoker`'s heuristic LR isn't precise enough
- `hypernix.cake_pan` — training-time NaN/Inf and stability guard, complementary to smoker's quality tiers
