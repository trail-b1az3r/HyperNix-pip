# Espresso Maker — `hypernix.espresso_maker`

Espresso is the fast, concentrated, pressure-brewed cousin of drip
coffee. `espresso_maker` is the fast, concentrated counterpart to
[`CoffeeMaker`](CoffeeMaker.md): run a model against a small prompt
battery and get scores back quickly. No warmup, no schedule, no
retries — just a pull.

## Four tiers (espresso drink sizes)

| Tier | Class | Samples/prompt | Tokens | Temp | Use case |
|---|---|---|---|---|---|
| 1 (shortest) | `Ristretto` | 1 | 16 | 0.0 (greedy) | Deterministic spot-checks. |
| 2 | `SingleShot` | 1 | 64 | 0.2 | Standard pull. |
| 3 | `DoubleShot` | 2 | 96 | 0.4 | Two samples; scorer (if given) picks the winner — without a scorer, the first sample always wins. |
| 4 (longest) | `Lungo` | 4 | 256 | 0.8 | Diverse outputs for eyeballing creativity/coverage; pair with a scorer to pick a winner. |

All four subclass `EspressoMaker` and share `.pull()`.

## `Shot` (frozen dataclass — one generation record)

```python
@dataclass(frozen=True)
class Shot:
    prompt: str
    output: str
    score: float | None = None
    reference: str | None = None
```

## `EspressoMaker` (base class)

| Field | Type | Default | Notes |
|---|---|---|---|
| `oven` | `Any` | required | Anything exposing `.complete(prompt, **kwargs) -> str`. |
| `max_new_tokens` | `int` | `32` | Overridden per tier. |
| `temperature` | `float` | `0.0` | Overridden per tier. |
| `top_k` | `int` | `1` | Overridden per tier. |
| `top_p` | `float` | `1.0` | Overridden per tier. |
| `samples_per_prompt` | `int` | `1` | Overridden per tier. |
| `scorer` | `Callable[[str, str, str \| None], float] \| None` | `None` | Called as `scorer(prompt, output, reference)`. Without one, `.pull()` always keeps the first candidate per prompt. |
| `history` | `list[Shot]` | `[]` | Accumulates the winning `Shot` from every `.pull()` call. |

### Methods

| Method | Signature | Notes |
|---|---|---|
| `.pull(prompts, references=None)` | `(Sequence[str], Sequence[str]\|None) -> list[Shot]` | For each prompt, generates `samples_per_prompt` candidates via `oven.complete()`, scores each (if `scorer` set), keeps the best-scoring one (`max()` on score, ties break on the first candidate), appends it to `history`, and returns the list of winners (one per prompt). `references`, if given, must match `prompts` in length or raises `ValueError`. |
| `.mean_score` | `property -> float \| None` | Mean of all scored (non-`None`) shots across `history`; `None` if nothing has been scored yet. |

## Constructing a tier

```python
from hypernix.espresso_maker import espresso_maker, ristretto, lungo

e = espresso_maker("double-shot", oven, scorer=my_scorer)
results = e.pull(["What is 2+2?", "Capital of France?"], references=["4", "Paris"])
print(e.mean_score)

# Or use a dedicated shortcut:
r = ristretto(oven)
l = lungo(oven, scorer=my_scorer)
```

`espresso_maker(tier, oven, *, scorer=None, **kwargs)` — `tier`
case-insensitive, `_` normalized to `-`. Valid tiers (also in
`TIERS: dict[str, type[EspressoMaker]]`): `"ristretto"`,
`"single-shot"`, `"double-shot"`, `"lungo"`.

Dedicated shortcuts, one per tier: `ristretto(oven, **kw)`,
`single_shot(oven, **kw)`, `double_shot(oven, **kw)`, `lungo(oven, **kw)`
— each just constructs its class directly.

### Required modules

Standard library only — `dataclasses`, `collections.abc`, `typing`. No
HyperNix internal dependencies beyond whatever `oven` object you pass in
(structurally typed — no import of `old_oven` required).


> **Canonical source:** `src/hypernix/evaluation/espresso_maker.py`. The `hypernix.espresso_maker` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. Without a scorer, the first generated candidate wins even when a tier generates several candidates per prompt.

---

## See also

- [Coffee Maker](CoffeeMaker.md) — the scheduled-training counterpart to this scored-evaluation module
- `hypernix.old_oven.CodeOven` — the typical `oven` you'd pass in
