# Deep Fryer — `hypernix.deep_fryer`

A deep fryer drops parameters into hot oil: some come out crispy, some
come out ruined. `deep_fryer` perturbs a fraction of a model's weights
with Gaussian noise, in place on a `torch.nn.Module` — useful for
**regularisation** (light frying, small σ, small fraction) and
**robustness testing** (heavy frying, bigger σ, larger fraction, plus
random zeroing).

## Two tiers

| Tier | Class | Defaults | Use case |
|---|---|---|---|
| 1 | `LightFry` | `fraction=0.02`, `noise_std=0.1` | Regulariser during training, or between epochs to knock the model off a local minimum. |
| 2 | `HeavyFry` | `fraction=0.3`, `noise_std=0.5`, `zero_rate=0.1` | Robustness testing; generating deliberately "bad model" negatives to train a judge against; input to a [`salt_shaker`](Kitchen.md) chain. |

## Snapshot / restore

```python
from hypernix.deep_fryer import LightFry

fryer = LightFry(model=my_model, fraction=0.03, noise_std=0.15, seed=42)
fryer.save_pristine()   # snapshot untouched weights
fryer.fry()              # perturb in place
# ... evaluate / use the perturbed model ...
fryer.un_fry()           # restore the pristine snapshot
```

| Method | Signature | Notes |
|---|---|---|
| `.save_pristine()` | `() -> int` | Snapshots `model.state_dict()` (detached clones), returns count of tensors snapshotted. |
| `.un_fry()` | `() -> int` | Restores the last snapshot in place; returns count restored, `0` if none exists. |

## `Fryer` (base class) fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | `torch.nn.Module` | required | |
| `fraction` | `float` | `0.02` | Fraction of each matching parameter's elements to perturb. Must be in `[0, 1]` (`ValueError` otherwise, checked in `__post_init__`). |
| `noise_std` | `float` | `0.1` | Gaussian std-dev of additive noise, **relative to the parameter's own std** (not absolute). Must be `>= 0`. |
| `patterns` | `tuple[str, ...]` | `()` | Name substring filters — if non-empty, only parameters whose name contains at least one pattern are fried. Empty means every parameter is a candidate. |
| `seed` | `int` | `0` | Drives both which elements get picked (`random.Random(seed)`, shared across all parameters) and the noise values themselves (a **per-parameter** `torch.Generator` seeded as `seed + sum(map(ord, param_name))`, on the parameter's own device — reproducible on CPU and CUDA alike). |

## `.fry()` — the perturbation algorithm

```python
@torch.no_grad()
def fry(self) -> dict[str, int]
```

For each parameter matching `patterns` (and matching `_should_fry_frozen()`
— see below): picks `k = max(1, int(numel * fraction))` random element
indices, computes the parameter's own std, and adds
`N(0, 1) * noise_std * param_std` noise at those indices — **in place**,
no gradient tracking. All-zero-std tensors are skipped (a degenerate
Gaussian). Returns `{param_name: n_elements_perturbed}` for provenance.

`_should_fry_frozen()` gates whether `requires_grad=False` parameters
are eligible: `LightFry` (base default) skips frozen params;
`HeavyFry` overrides it to fry them too.

## `LightFry` vs `HeavyFry` specifics

**`LightFry(fraction=0.02, noise_std=0.1)`** — no extra behavior beyond
the base `fry()`.

**`HeavyFry(fraction=0.3, noise_std=0.5, zero_rate=0.1)`** — after
adding noise, an additional `_apply_extra()` hook zeroes out
`int(k * zero_rate)` of the just-perturbed indices exactly to `0.0`
(sampled from among the already-noised elements, not a separate draw).

## Factory

```python
from hypernix.deep_fryer import deep_fryer

fryer = deep_fryer("heavy-fry", model, fraction=0.25, patterns=("attn",), seed=7)
```

`deep_fryer(tier, model, *, fraction=None, noise_std=None,
patterns=(), seed=0, **extra)` — `tier` case-insensitive, `_`
normalized to `-`. Valid tiers (also in `TIERS: dict[str, type[Fryer]]`):
`"light-fry"`, `"heavy-fry"`. `fraction`/`noise_std` only override the
tier's default if explicitly passed (not `None`); any `**extra` kwargs
(e.g. `zero_rate` for heavy-fry) pass straight through to the
constructor.

### Required modules

- `torch`, `torch.nn` (hard dependency — all perturbation is tensor ops)
- Standard library: `random`, `dataclasses`, `collections.abc`, `typing`


> **Canonical source:** `src/hypernix/training/deep_fryer.py`. The `hypernix.deep_fryer` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. Weight perturbation happens in place; save the pristine snapshot first if you need to restore the original weights.

---

## See also

- `hypernix.salt_shaker` / `hypernix.pepper_shaker` — data-side augmentation, a common pairing with weight-side perturbation
- `hypernix.cake_pan` — training-time NaN/Inf guard; useful to run alongside `HeavyFry` since large perturbations can destabilize a run
