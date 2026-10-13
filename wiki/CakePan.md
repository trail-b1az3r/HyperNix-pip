# Cake Pan — `hypernix.cake_pan`

A cake pan is what keeps the cake together while it bakes. `CakePan`
wraps a single training step with hybrid CPU/GPU safety features:
NaN/Inf detection with rollback, a memory watchdog, a wall-time
watchdog, and periodic on-disk snapshots. `bake()` runs exactly one
step; the training loop itself lives in the caller (or use `.oven()`
for a built-in loop with retries).

```python
from hypernix.cake_pan import CakePan

pan = CakePan(
    model, optimizer,
    gpu_device="cuda",
    cpu_offload_patterns=("embed_tokens", "lm_head"),
    free_gb_trip=0.5, step_timeout_s=120.0,
    snapshot_every=100, snapshot_path="run/ckpt.pt",
)
pan.save_pristine()
try:
    for batch in loader:
        loss = pan.bake(lambda: one_training_step(batch))
except pan.BakeOff as exc:
    print("rolled back:", exc)
```


> **Canonical source:** `src/hypernix/training/cake_pan.py`. The flat `hypernix.cakepan` import spelling used in examples is a compatibility-facing API; the path above is the implementation to check when verifying behavior. The timeout guard uses `SIGALRM` where available and is skipped on platforms such as Windows. Treat any on-disk PyTorch checkpoint as trusted input.

---

## `CakePan` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | `nn.Module` | required | |
| `optimizer` | `torch.optim.Optimizer \| None` | `None` | If set, its per-parameter state (e.g. Adam moments) is zeroed on rollback — NaN-contaminated moments would otherwise re-propagate corruption on the next step. |
| `gpu_device` | `str` | `"cuda"` | |
| `cpu_device` | `str` | `"cpu"` | |
| `cpu_offload_patterns` | `tuple[str, ...]` | `()` | Module-name substrings; matching modules get offloaded to `cpu_device` when the memory watchdog trips. |
| `free_gb_trip` | `float` | `0.5` | Free-GPU-memory threshold (GB) below which the memory watchdog activates. |
| `step_timeout_s` | `float` | `120.0` | Max wall-clock seconds a single step may take before it's assumed stuck. |
| `snapshot_every` | `int` | `0` | Pickle the state dict to `snapshot_path` every N steps. `0` disables. |
| `snapshot_path` | `Path \| str \| None` | `None` | Required if `snapshot_every > 0`. |
| `check_grads` | `bool` | `True` | Also NaN/Inf-scan every gradient tensor after each step, not just the loss. Disable to save wall time on very large models. |
| `step_count` | `int` | `0` (init=False) | Incremented at the start of every `.bake()` call. |
| `pristine_state` | `dict[str, Tensor] \| None` | `None` (init=False) | Set by `.save_pristine()`. |

`BakeOff` is also re-exported as an **instance attribute**
(`pan.BakeOff`), so `except pan.BakeOff:` works without a separate
top-level import.

## `BakeOff` exception

```python
class BakeOff(RuntimeError):
    def __init__(self, reason: str, step: int) -> None: ...
```

Raised on any corruption-class event: NaN/Inf loss, NaN/Inf gradient
(if `check_grads=True`), a step exceeding `step_timeout_s`, or a CUDA
OOM. Carries `.reason` (str) and `.step` (int, the step count at
detection). The message is formatted as `"BakeOff @ step {step}: {reason}"`.

## Pristine snapshots (rollback point)

| Method | Signature | Notes |
|---|---|---|
| `.save_pristine()` | `() -> None` | Captures the current state dict **on CPU** (detached clones). Call again after a clean stretch to advance the rollback point. |
| `.roll_back()` | `() -> bool` | Restores the last pristine snapshot in place, moving tensors back to their original device. Also zeroes optimizer state tensors (if `optimizer` is set) to prevent re-propagating NaN-contaminated moments. Returns `False` if no snapshot exists yet. |

These are distinct from the on-disk `snapshot_every` snapshots — pristine
state lives in memory (CPU RAM) and is the rollback target; on-disk
snapshots are for crash recovery across process restarts.

## `.memory_guard()` — the memory watchdog

```python
def memory_guard(self) -> bool
```

Uses `hypernix.freezer.probe_vram()`. Returns `False` immediately on a
CPU-only setup (`total == 0`) or if free memory is still above
`free_gb_trip`. Otherwise: calls `torch.cuda.empty_cache()`, then — if
`cpu_offload_patterns` is set — moves any module whose name contains a
matching pattern to `cpu_device`. Returns `True` only if at least one
module was actually moved.

## `.bake(step_fn)` — the guarded step

```python
def bake(self, step_fn: Callable[[], torch.Tensor | Any]) -> Any
```

Runs one step under all the guards, in this order:

1. Installs a `SIGALRM`-based wall-time watchdog (Linux/macOS only —
   silently skipped on platforms without `signal.SIGALRM`, e.g.
   Windows). If the alarm fires mid-step, the handler rolls back
   *before* raising `BakeOff`, so the caller never has to remember to.
2. Calls `.memory_guard()`, then `step_fn()`.
3. Catches `torch.cuda.OutOfMemoryError` → rolls back, raises `BakeOff("OOM: ...")`.
4. If the result is a `torch.Tensor`, NaN/Inf-checks it as `"loss"`.
5. If `check_grads=True`, NaN/Inf-checks every parameter's `.grad`.
6. Any NaN/Inf hit → rolls back, raises `BakeOff("{what} contains NaN/Inf")`.
7. Writes an on-disk snapshot if `snapshot_every` divides `step_count`.
8. In a `finally` block: cancels the alarm, restores the previous
   `SIGALRM` handler, and records `self.last_step_seconds`.

**Caveat:** `last_step_seconds` is set as a plain instance attribute
inside `bake()`'s `finally` block, not declared as a dataclass field —
it doesn't exist until after the *first* `.bake()` call completes.
Accessing `pan.last_step_seconds` before that raises `AttributeError`.

## `.oven(batches, step_fn, *, on_bake_off=None, max_retries_per_batch=2)`

Convenience loop: calls `step_fn(batch)` under `.bake()` guard for every
item in `batches`. On `BakeOff`, optionally invokes `on_bake_off(exc)`,
then retries the same batch up to `max_retries_per_batch` times before
giving up on it and moving to the next. Returns the count of
successfully completed steps.

## Module-level constructor

```python
from hypernix.cake_pan import cake_pan
pan = cake_pan(model, optimizer, free_gb_trip=1.0)
```

`cake_pan(model, optimizer=None, **kwargs) -> CakePan` — thin wrapper
around the `CakePan` constructor.

### Required modules

- `torch`, `torch.nn` (hard dependency)
- `hypernix.freezer.probe_vram` (internal, for the memory watchdog)
- Standard library: `signal`, `time`, `dataclasses`, `pathlib`, `collections.abc`

---

## See also

- [Freezer](Freezer.md) — `probe_vram()`, used by the memory watchdog
- `hypernix.smoker` — training-quality tiers; pairs naturally with `CakePan` for stability during longer/heavier runs
- `hypernix.deep_fryer` — weight perturbation; running it alongside `CakePan` is a good idea since large perturbations can destabilize a run
