# Smoke alarms — training-step planner & monitor

A **smoke alarm** plans a training run before it starts and watches it
while it's going. Inputs are time budget + hardware + model details;
outputs are a recommended step count, a per-step time estimate, and an
on-pace check during the run.

Four tiers, named in the smoke-detector idiom:

| Tier | Class | Key inputs | When to use |
|---|---|---|---|
| Lightest (Rads) | `RadsAlarm` | model size, time budget | one-shot run, scripts, tests |
| Mid | `GasAlarm` | + CPU and/or GPU preset | known hardware without a warmup |
| Highest | `ModernAlarm` | + warmup `step_fn` | most accurate; measures real time |
| Auto | `AutoAlarm` | any of the above | picks the most detailed alarm whose inputs are present |

All four expose the same surface so they're swappable in and out:

```python
alarm.estimate_step_seconds()    # -> float
alarm.recommended_steps()        # -> int
alarm.budget()                   # -> TrainingBudget(time, sps, margin, steps, notes)
alarm.check(elapsed_s, steps)    # -> AlarmStatus(on_pace, expected, eta, message)
alarm.storage_warning(save_every, snapshot_size_gb)  # -> "" | "[Alarm] storage warning: ..."
```

## RadsAlarm — radioactive / "Rads" alarm (lightest)

Pure constants. Scales a 1 s/step baseline (100 M params, ctx=1024,
batch=1) linearly by params, context, and batch. No hardware
introspection, no warmup.

```python
from hypernix import smoke_alarm

a = smoke_alarm.rads_alarm(
    time_budget_seconds=3600.0,    # train for 1h
    model_params=92_100_000,       # HyperNix 1.5
    context_length=1024,
    batch_size=1,
    safety_margin=0.10,            # reserve 10% of the time budget
)
b = a.budget()
# TrainingBudget(time_seconds=3600, estimated_step_seconds=0.92,
#                safety_margin=0.10, recommended_steps=3521, notes='RadsAlarm: ...')
```

Use this when you literally just need a number. Aliases:
`smoke_alarm.rad_alarm`, `smoke_alarm.radioactive_alarm`.

## GasAlarm — preset-aware (mid)

Looks up a CPU and/or GPU preset and scales the baseline by their
throughput. GPU bandwidth is the dominant factor; if no GPU is given,
falls back to CPU GFLOPS plus a 20× CPU-vs-GPU penalty.

```python
gas = smoke_alarm.gas_alarm(
    time_budget_seconds=3600.0,
    model_params=92_100_000,
    gpu_name="rtx-3080-ti",        # GPU_PRESETS lookup
    cpu_name="i7-13700k",          # CPU_PRESETS lookup
    available_vram_gb=11.5,
)
gas.estimate_step_seconds()        # -> ~0.7s, scaled by 912 GB/s vs. 700 baseline
```

See [Freezer.md](Freezer.md) for the full
[CPU_PRESETS](#cpu-presets) and [GPU_PRESETS](#gpu-presets) tables;
known unknowns fall back to the generic baseline silently.

## ModernAlarm — warmup-measured (highest accuracy)

Runs `warmup_steps` real training steps against a caller-supplied
closure, takes the median wall-clock time, and uses that as the
estimate. Every other alarm method behaves the same; only the
per-step number changes.

```python
def one_step():
    out = oven.model(batch[:, :-1], labels=batch[:, 1:])
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

modern = smoke_alarm.modern_alarm(
    time_budget_seconds=3600.0,
    step_fn=one_step,
    warmup_steps=5,
)
modern.recommended_steps()
```

The warmup itself runs synchronously inside the constructor —
budget for it explicitly when the per-step time is large.

## AutoAlarm — selector

Picks the most detailed alarm whose inputs are satisfied:

| You supply… | You get |
|---|---|
| `warmup_step_fn=...` | `ModernAlarm` (after running the warmup) |
| `cpu_name=...` or `gpu_name=...` | `GasAlarm` |
| nothing else | `RadsAlarm` |

```python
alarm = smoke_alarm.auto_alarm(
    time_budget_seconds=3600.0,
    model_params=92_100_000,
    # ↓ All optional; auto_alarm consults torch.cuda + /proc/cpuinfo
    #   when detect_hardware=True (default).
    gpu_name=None,
    cpu_name=None,
    warmup_step_fn=None,
)
print(alarm.budget())
```

Hardware detection (`detect_hardware=True`, the default) tries to
match `torch.cuda.get_device_name(0)` against `GPU_PRESETS` and
`/proc/cpuinfo` against `CPU_PRESETS`. CPU detection is Linux-only;
on macOS / Windows it returns None and AutoAlarm falls back to
`RadsAlarm` unless you provide `cpu_name=` / `gpu_name=` explicitly.

## Mid-run monitoring

Once training is underway, call `check` on each log boundary:

```python
import time

start = time.monotonic()
for step in range(alarm.recommended_steps()):
    one_step()
    if step % 100 == 0:
        status = alarm.check(time.monotonic() - start, step)
        print(status.message)
        if not status.on_pace:
            # downscale batch, halve save_every, raise a CI alert, …
            ...
```

`AlarmStatus` carries `on_pace`, `completed_steps`, `expected_steps`,
`elapsed_seconds`, `eta_seconds`, and a pre-formatted `message`
string.

## Storage warning

Optional — pass `available_storage_gb=` and call `storage_warning`:

```python
alarm = smoke_alarm.rads_alarm(
    time_budget_seconds=10000.0,
    model_params=92_000_000,
    available_storage_gb=20.0,
)
warning = alarm.storage_warning(save_every=500, snapshot_size_gb=0.4)
if warning:
    print(warning)
# "[RadsAlarm] storage warning: 18 snapshots × 0.40 GB ≈ 7.2 GB needed
#  but only 20.0 GB available."   <- example output, varies with budget
```

Returns "" when the budget would fit. Useful as a precondition gate
before starting a long run on a constrained box.

## CPU presets

Looked up via `hypernix.freezer.cpu_preset(name)`. Names are
case-insensitive and treat `-` / `_` / spaces as equivalent.

| Family | Names |
|---|---|
| **Intel 7th gen (Kaby Lake)** | `i5-7200u`, `i5-7300hq`, `i5-7400`, `i5-7600k`, `i7-7660u`, `i7-7700hq`, `i7-7700k`, `i9-7900x`, `i9-7980xe` |
| **Intel 11th gen (Rocket Lake / Tiger Lake-H)** | `i5-11400`, `i5-11600k`, `i5-11320h`, `i7-11700k`, `i7-11800h`, `i9-11900k` |
| **Intel 12th gen (Alder Lake)** | `i5-12400`, `i5-12500`, `i5-12600k`, `i7-12700k`, `i7-12700h`, `i9-12900k`, `i9-12900hx` |
| **Intel 13th gen (Raptor Lake)** | `i5-13400`, `i5-13500`, `i5-13600k`, `i7-13700k`, `i7-13700h`, `i9-13900k`, `i9-13900hx` |
| **Intel 14th gen (Raptor Lake-R)** | `i5-14400`, `i5-14500`, `i5-14600k`, `i7-14700k`, `i7-14700hx`, `i9-14900k`, `i9-14900ks`, `i9-14900hx` |
| **Core Ultra Series 1** (Meteor / Lunar Lake) | `core-ultra-5-125h`, `core-ultra-5-135h`, `core-ultra-5-228v`, `core-ultra-7-155h`, `core-ultra-7-165h`, `core-ultra-7-258v`, `core-ultra-9-185h` |
| **Core Ultra Series 2** (Arrow Lake, AVX10) | `core-ultra-5-225k`, `core-ultra-5-235k`, `core-ultra-7-265k`, `core-ultra-9-285k` |
| **Ryzen 5000 series** (Zen 3, AM4, AVX2 only) | `ryzen-5-5600x`, `ryzen-7-5800x`, `ryzen-9-5900x`, `ryzen-9-5950x` |
| **Ryzen 7000 series** (Zen 4, AM5, AVX-512) | `ryzen-5-7600x`, `ryzen-7-7700x`, `ryzen-9-7900x`, `ryzen-9-7950x` |
| **Ryzen 9000 series** (Zen 5, AM5, AVX-512) | `ryzen-5-9600x`, `ryzen-7-9700x`, `ryzen-9-9900x`, `ryzen-9-9950x` |

Generational-family aliases resolve to a representative SKU when you
don't need the exact model — e.g. `i7-12th-gen`, `ultra-9`,
`ryzen-9000`, `ryzen-7-7000`.

Each preset carries: `cores`, `threads`, `base_clock_ghz`, `avx_levels`
tuple (e.g. `("AVX2", "AVX-VNNI", "AVX10")`), `recommended_threads`
for BLAS / OpenMP, and `gflops_per_thread` for the smoke alarms.

## GPU presets

Looked up via `hypernix.freezer.gpu_preset(name)`.

| Tier | Names | dtype | Freezer class |
|---|---|---|---|
| **Hopper data-center** | `h100`, `h100-94`, `h200` | bf16 | New |
| **Ampere workstation** (sm_86) | `rtx-a4500`, `rtx-a5000`, `rtx-a5500`, `rtx-a6000` | bf16 | New |
| **RTX PRO Ada** (sm_89) | `rtx-pro-4000-ada`, `rtx-pro-5000-ada`, `rtx-pro-6000-ada` | bf16 | New |
| **RTX PRO Blackwell** (sm_120) | `rtx-pro-6000-blackwell` | bf16 | New |
| **Ada consumer** (sm_89) | `rtx-4070-ti-super`, `rtx-4080-super` | bf16 | New |
| **Ampere consumer** (sm_86) | `rtx-3080-ti` | bf16 | New |
| **Turing consumer** (sm_75) | `gtx-1660-ti`, `rtx-2080`, `rtx-2080-super`, `rtx-2080-ti` | fp16 | Old / New (2080 Ti only) |
| **Pascal** (sm_61) | `gtx-1080`, `gtx-1080-ti` | fp16 | Old / New (1080 Ti only) |

Each preset carries: `name`, `vram_gb`, `compute_capability`,
`preferred_dtype`, `bandwidth_gb_s` (rough memory bandwidth — the
dominant signal for transformer training throughput), and
`freezer_class` (`"Old"` or `"New"`).

## Example: planning a HyperNix 1.5 run on a GTX 1080

```python
from hypernix import smoke_alarm

# 4 hours of training time, HyperNix 1.5 (92.1M), GTX 1080.
alarm = smoke_alarm.gas_alarm(
    time_budget_seconds=4 * 3600,
    model_params=92_100_000,
    gpu_name="gtx-1080",
    cpu_name="i7-7700hq",       # the 1080 commonly ships with a 7700HQ in 2017 laptops
    available_vram_gb=8.0,
    available_storage_gb=50.0,
)

print(alarm.budget())
# TrainingBudget(time_seconds=14400.0, estimated_step_seconds=2.01,
#                safety_margin=0.1, recommended_steps=6437,
#                notes='GasAlarm: cpu=Intel Core i7-7700HQ, gpu=NVIDIA GeForce GTX 1080')

print(alarm.storage_warning(save_every=500, snapshot_size_gb=0.18))
# ""  (~12 snapshots × 0.18 GB = 2.2 GB, plenty of headroom)
```

## Putting it next to the freezer

The alarm tells you *how many steps*; the freezer tells you *how to
actually run them*. They share the GPU preset registry and compose
naturally:

```python
from hypernix import freezer, smoke_alarm

fz = freezer.flash_freezer(base=freezer.auto_freezer(), slow=True)
alarm = smoke_alarm.auto_alarm(time_budget_seconds=4*3600,
                               model_params=92_100_000)

bs   = fz.suggest_batch_size(hint=4)
ctx  = fz.suggest_context_length(hint=1024)
n    = alarm.recommended_steps()

# … pass bs, ctx, n into oven.train wrapped in fz.guard
```
