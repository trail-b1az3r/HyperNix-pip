# Freezer — VRAM manager

`hypernix.freezer` is the small subsystem that makes sure a training
or inference run fits on whatever GPU (or CPU) you happen to have.
Three concrete classes behind one `Freezer` base:

| Class | Target | Batch | Ctx | dtype | Cache chill / step | Hint cap |
|---|---|---|---|---|---|---|
| `OldFreezer` | 8 – 10 GB | 1 | 512 | Pascal-safe (fp16 / fp32 on CPU) | ✓ | ✓ |
| `NewFreezer` | 11 GB + | 8 | 2048 | fp32 (bf16 on Ampere+) | ✗ | ✗ |
| `FlashFreezer` | adaptive | inherits | inherits | inherits | inherits | inherits |

## Public API

```python
from hypernix import freezer

# Pick the right one based on detected VRAM.
fz = freezer.auto_freezer(threshold_gb=11.0)

# Or explicit.
fz = freezer.old_freezer()
fz = freezer.new_freezer()

# Wrap any freezer in FlashFreezer for OOM safety.
safe = freezer.flash_freezer(base=fz, max_retries=5, backoff_s=2.0, slow=True)

# Consult it.
bs   = fz.suggest_batch_size(hint=32)   # 1 on Old, 32 on New
ctx  = fz.suggest_context_length(hint=4096)
dtype = fz.preferred_dtype
print(fz.budget())    # VRAMBudget(device="cuda:0", total=10.0, free=8.3)

# Run something OOM-safe.
result = safe.guard(lambda: model(batch))
```

## VRAM probing

```python
from hypernix.freezer import probe_vram, VRAMBudget

b = probe_vram()
# VRAMBudget(device="cuda:0", total=12884901888, free=9000000000)
b.total_gb, b.free_gb, b.used_gb
```

On CPU-only hosts `probe_vram()` returns a zeroed budget
(`device="cpu"`, `total=0`, `free=0`) so callers can `if b.total:` as
a feature gate.

## FlashFreezer — OOM retry loop

```python
for step in range(n_steps):
    try:
        safe.guard(lambda: one_training_step(model, batch))
    except torch.cuda.OutOfMemoryError:
        # Only reraised after max_retries attempts.
        logger.exception("persistent OOM at step %d", step)
        break
```

On each `guard(fn)` call:

1. Run `fn`.
2. If a `torch.cuda.OutOfMemoryError` fires:
   a. `torch.cuda.empty_cache()`
   b. Exponential backoff — sleep `min(backoff_s * 2^attempt, 60s)`
   c. `wait_for(min_free_gb)` — block until VRAM recovers (or timeout).
   d. If `slow=True`, halve `current_batch_size` (floored at 1) so a
      caller consulting it sees a smaller batch on the retry.
3. Retry, up to `max_retries` times.
4. If still OOM, re-raise.

### "Slow" mode

```python
safe = freezer.flash_freezer(base=freezer.new_freezer(), slow=True)
# safe.current_batch_size starts at 8 (NewFreezer default).
# After 2 OOMs it becomes 2; after 4 it becomes 1.

for step in range(n_steps):
    bs = safe.current_batch_size     # <- consult here, not a fixed constant
    batch = make_batch(bs)
    safe.guard(lambda: model(batch))
```

Your loop must consult `safe.current_batch_size` for `slow=True` to
have any effect — FlashFreezer doesn't have a handle on your batch
builder.

## Pascal (CUDA 6.1) support

`OldFreezer.preferred_dtype` routes through `pascal_safe_dtype()`:

| Device | dtype chosen |
|---|---|
| CPU (no CUDA) | `torch.float32` |
| Pascal (sm_6x) / Volta (sm_70) / Turing (sm_75) | `torch.float16` |
| Ampere (sm_80+) with `is_bf16_supported()` | `torch.bfloat16` |

See [Pascal.md](Pascal.md) for the full Pascal training playbook.

## Examples

### Detect hardware, pick defaults

```python
from hypernix import freezer

fz = freezer.auto_freezer()
print(fz)
# <OldFreezer device=cuda:0 total=8.0GB free=6.3GB bs=1 ctx=512>

# -> pass fz.preferred_dtype, fz.base_batch_size, fz.base_context_length
#    into your training loop.
```

### Combine with an oven

```python
from hypernix import old_oven, freezer

fz = freezer.flash_freezer(base=freezer.auto_freezer(), slow=True)

oven = old_oven.preheat(
    repo_id="nix2.5",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=str(fz.preferred_dtype).split(".")[-1],  # e.g. "float16"
)

# Wrap the training call.
fz.guard(lambda: oven.train("corpus.txt", "./trained",
                            steps=1000,
                            batch_size=fz.suggest_batch_size(hint=4)))
```

## Testing

Tests mock `torch.cuda.is_available` / `get_device_capability` /
`mem_get_info` so the whole thing exercises on CPU-only CI:

```bash
pytest tests/system/test_freezer.py -v
```

26 tests cover: probe on CPU vs CUDA, OldFreezer hint capping,
NewFreezer hint pass-through, `auto_freezer` threshold logic, OOM retry
count, batch-size halving, re-raise after max_retries, `wait_for` CPU
short-circuit, `repr` formatting, Pascal / non-Pascal detection, dtype
selection across sm_6x / sm_7x / sm_8x.
