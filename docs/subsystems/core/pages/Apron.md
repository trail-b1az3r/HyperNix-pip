# Apron — `hypernix.apron`

An apron protects what's underneath while you cook. `apron` captures
every random-number source hypernix (or your script) might touch —
Python's `random`, NumPy (if installed), PyTorch CPU, and every CUDA
device — and restores it on exit, so a step that perturbs the global
RNG never leaks that perturbation back to the caller.

## As a context manager (recommended)

```python
from hypernix.apron import apron

with apron(seed=0):
    # everything inside is deterministic; nothing leaks out
    random.shuffle(my_list)
    torch.randn(10)
# global RNG state here is exactly what it was before the `with` block
```

`apron(*, seed=None)` — snapshots every RNG, optionally seeds them all,
runs the body, restores the original states on exit (even on exception,
via `try`/`finally`). Yields the `Apron` snapshot object if you want to
read the captured pre-seed states inside the block.

## Object form (finer control, e.g. across Jupyter cells)

```python
from hypernix.apron import Apron

a = Apron.snapshot(seed=0)
...
a.restore()
```

## `Apron` (dataclass)

| Field | Type | Notes |
|---|---|---|
| `py_state` | `tuple` | `random.getstate()` |
| `numpy_state` | `Any` | `numpy.random.get_state()`, or `None` if NumPy isn't installed |
| `torch_state` | `torch.Tensor \| None` | `torch.get_rng_state()` |
| `cuda_states` | `list[torch.Tensor]` | One entry per CUDA device (empty if no CUDA) |

| Method | Notes |
|---|---|
| `Apron.snapshot(*, seed=None)` | Captures the **pre-seed** state first, then (if `seed` is given) seeds `random`, `torch.manual_seed`, NumPy (`seed % 2**32`, since NumPy's legacy seeding requires a 32-bit value), and `torch.cuda.manual_seed_all`. Because the snapshot is taken *before* seeding, `.restore()` puts you back to whatever you were doing before the seed — not back to the seeded starting point — which is what `with apron(seed=42):` callers want. |
| `.restore()` | Restores every captured RNG that was actually present at snapshot time (each field is checked for truthiness/non-`None` before restoring). |

### Required modules

- `torch` (hard dependency)
- `numpy` — optional; gracefully skipped via `ImportError` handling if not installed
- Standard library: `contextlib`, `random`, `dataclasses`, `collections.abc`, `typing`


> **Canonical source:** `src/hypernix/training/apron.py`. The flat `hypernix.apron` import spelling used in examples is a compatibility-facing API; the path above is the implementation to check when verifying behavior. `numpy` support is optional; `torch` is required. The context manager restores captured RNG state even when the body raises.

---

## See also

- `hypernix.espresso_maker` — evaluation runs are a common place to wrap generation calls in `apron` so eval-time sampling doesn't perturb a training loop's RNG state
