# Optimizers — `pressure_cooker`, `optimizer_framework`, `pressure_cooker_v4`

Three related modules covering HyperNix's custom AdamW optimizer family,
from the original device-tiered `PressureCooker` through to the
`OptimizerBase`-powered V4 line. V4 is not the newest generation — it's
superseded for new training by the memory-first V5/V5S line and the
speed-first V6 line, both documented on their own pages:
[Pressure Cooker V5 / V5+ / V5S](Pressure-Cooker-V5.md) and
[Pressure Cooker V6 / V6V](Pressure-Cooker-V6.md). (V3 is also documented
separately: [Pressure Cooker V3](Pressure-Cooker-V3.md).)

**Deprecated since 0.72.6:** the original `PressureCooker` (v1, below)
and `PressureCookerV3`. Constructing either warns with a `FutureWarning`
naming V4 (kept) or V5/V6 as the replacement; importing does not warn.

**Accuracy note:** `pressure_cooker.py`'s module docstring used to list
`PressureCookerV2` / `PressureCookerV2Plus` and a "WORKSHOP INTEGRATION"
/ "TTS/ASR PIPELINES" section (`WorkshopFramework`, `TTSEngine`,
`ASREngine`, etc.) as if they lived in this file. They never did —
`WorkshopFramework`/`TTSEngine`/`ASREngine` actually live in
`hypernix.workshop` (see [Workshop](Workshop.md)), and no
`PressureCookerV2`/`V2Plus` class has ever existed anywhere in the
codebase. The docstring has since been corrected to match reality; the
tables below reflect what's actually implemented.

---

## `hypernix.pressure_cooker` — the original device-tiered optimizer

### `PressureCooker(Optimizer)` — base class

```python
from hypernix.pressure_cooker import PressureCooker

opt = PressureCooker(
    model.parameters(),
    peak_lr=3e-4, warmup_steps=200, plateau_steps=1000, cooldown_steps=200,
    betas=(0.9, 0.95), weight_decay=0.1,
    lookahead_k=5, lookahead_alpha=0.5,
    grad_accum_steps=4,
)
```

A pure-Python AdamW implementation with a built-in linear-warmup →
plateau → cosine-cooldown LR schedule, optional lookahead, gradient
accumulation, and mixed-precision/fused-kernel support.

| Constructor arg | Type | Default | Notes |
|---|---|---|---|
| `params` | iterable of params/dicts | required | |
| `peak_lr` | `float` | `3e-4` | Must be `> 0`. |
| `warmup_steps` / `plateau_steps` / `cooldown_steps` | `int` | `200` / `1000` / `200` | All must be `>= 0`. |
| `betas` | `tuple[float, float]` | `(0.9, 0.95)` | |
| `eps` | `float` | `1e-8` | |
| `weight_decay` | `float` | `0.1` | Decoupled (AdamW-style). |
| `lookahead_k` | `int` | `0` | `0` disables lookahead. |
| `lookahead_alpha` | `float` | `0.5` | Must be in `[0, 1]`. |
| `grad_scaler` | `torch.cuda.amp.GradScaler \| None` | `None` | If set, `.step()` unscales and skips the update cleanly on an inf grad rather than corrupting state. |
| `grad_accum_steps` | `int` | `1` | Must be `>= 1`. Only every Nth `.step()` call actually updates parameters. |
| `foreach` | `bool \| None` | `None` | Use the vectorized multi-tensor AdamW kernel. |
| `fused` | `bool \| None` | `None` | Use torch's fused AdamW kernel (torch ≥ 2.0 only). |
| `amsgrad` | `bool` | `False` | |

| Method | Notes |
|---|---|
| `.scheduled_lr(step=None)` | Linear warmup, flat plateau, then cosine decay to `0.0` over `cooldown_steps`. |
| `.step(closure=None)` | Handles grad accumulation, GradScaler unscale/inf-skip, dispatches to the scalar or multitensor AdamW path, applies lookahead if enabled. |
| `.phase(step=None)` | Returns `"warmup"` / `"plateau"` / `"cooldown"` / `"done"`. |
| `.describe()` | Dict snapshot of every config field plus `torch_version`. |

### Device tiers (all subclass `PressureCooker`)

| Tier | Class | Defaults forced |
|---|---|---|
| CPU 1 | `StovetopCooker` | `foreach=False`, `fused=False`, `grad_scaler=None` — minimum-memory path, no multi-tensor/GPU kernels. |
| CPU 2 | `ElectricCooker` | `foreach=True` (if torch ≥ 1.12, else auto-falls back), `fused=False`, `grad_scaler=None` — vectorized multi-tensor updates. |
| GPU 1 | `InductionCooker` | `foreach=True`, `fused=True` (if torch ≥ 2.0) — first-class `GradScaler` integration for fp16. |
| GPU 2 | `ProCooker(InductionCooker)` | Adds CUDA graph capture: `.warmup_graph(step_fn)` records a graph from a representative step (raises `RuntimeError` off-CUDA); `.replay_graph()` replays it for a speedup on small models / repetitive shapes. Skip on dynamic-shape/control-flow models. |

### `UniversalCooker.select()` / `universal_cooker()` — auto device selection

```python
from hypernix.pressure_cooker import universal_cooker
opt = universal_cooker(model.parameters(), prefer_speed=True)
```

**Patch (0.71.1):** the default now prioritizes the V5 / V5S optimizer
family ([Pressure-Cooker-V5](Pressure-Cooker-V5.md) — fully custom
optimizers, not restricted to CUDA) instead of the legacy device tiers
below. Pass `variant=` to choose:

| `variant=` | Returns |
|---|---|
| `"v5s"` (default) | `PressureCookerV5S` — newest/most specialised tier (3D oscillation resistance, pressure diffusion, low-power mode). |
| `"v5"` | `PressureCookerV5` (ORCP core). |
| `"v5-plus"` | `PressureCookerV5Plus` (ORCP-Ultra core; entropy scaling, resonance detection, QAT/MTP defaults). |
| `"legacy"` | The pre-0.71.1 behavior described below (`ProCooker`/`InductionCooker`/`ElectricCooker`/`StovetopCooker`). |

Whichever V5-family variant is picked, a CUDA device detected as
pre-Volta (Pascal, sm_61/6.2 — e.g. GTX 1080) automatically gets the
matching `Aged*` tier instead (`Agedcookerv5`, `ULTRAagedcookerv5`, or
`Agedcookerv5s`), which enforces `fused=False` and other Pascal-safe
defaults.

#### `variant="legacy"`

Picks the tier matching the first parameter's device: `ElectricCooker`/
`StovetopCooker` on CPU (by `prefer_speed`), `ProCooker`/`InductionCooker`
on CUDA (by `prefer_speed`) — **except** on pre-Volta CUDA devices
(compute capability < 7.0, i.e. Pascal / GTX 1080 and earlier), where it
always forces `InductionCooker` with `fused=False` regardless of
`prefer_speed`, since fused AdamW and CUDA graphs require sm_70+ and
would otherwise crash with `RuntimeError: fused=True requires CUDA
capability >= 7.0`.

### Factory / shortcuts

```python
from hypernix.pressure_cooker import pressure_cooker
opt = pressure_cooker(model.parameters(), tier="induction", peak_lr=5e-4)
```

`pressure_cooker(params, **kwargs)` — pass `tier=` (case-insensitive,
`_`→`-`) to pick a variant; valid: `"pressure-cooker"`, `"stovetop"`,
`"electric"`, `"induction"`, `"pro"` (also in `TIERS`). Dedicated
shortcuts: `stovetop_cooker`, `electric_cooker`, `induction_cooker`,
`pro_cooker`, `universal_cooker`.

### Required modules

`torch` (hard dependency). Standard library: `math`, `collections.abc`, `typing`.

---

## `hypernix.optimizer_framework` — the shared base for newer optimizers

Added in v0.70.4b2. Composable, schedule-aware base class plus profiling
helpers, used by `PressureCookerV4` (below).

### `ScheduleConfig` (dataclass)

```python
from hypernix.optimizer_framework import ScheduleConfig
sched = ScheduleConfig(lr=3e-4, warmup_steps=200, plateau_steps=1000, cooldown_steps=200, min_lr=1e-6).validate()
```

Same linear-warmup → plateau → cosine-cooldown shape as `PressureCooker`,
but cools down **to `min_lr`** rather than to `0.0`. `.validate()`
raises `ValueError` on any negative field or non-positive `lr`, and
returns `self` so it chains. `.phase_at_step(step)` / `.lr_at_step(step)`
are the pure functions driving the schedule.

### `OptimizerBase(torch.optim.Optimizer)`

Subclasses must implement `.step()` — the base handles LR scheduling,
gradient clipping, and optional profiling for you.

| Constructor arg | Default | Notes |
|---|---|---|
| `params`, `defaults` | required | Standard `torch.optim.Optimizer` args. |
| `schedule` | `ScheduleConfig()` | |
| `grad_clip` | `None` | Clip threshold. |
| `grad_clip_mode` | `"norm"` | `"norm"` or `"value"`. |
| `enable_profiling` | `False` | Attaches an `OptimizerProfiler`. |

| Method | Notes |
|---|---|
| `.gradient_clip()` | Returns `GradStats(total_norm, clipped, clip_threshold)`. `"norm"` mode uses `torch.nn.utils.clip_grad_norm_`; `"value"` mode clamps each grad element-wise and reports whether the pre-clamp max abs exceeded the threshold. |
| `.profile_start()` / `.profile_end(tokens=None)` | No-ops if `enable_profiling=False`; otherwise records timing via `OptimizerProfiler`. |
| `.scheduled_lr(step=None)` / `.phase(step=None)` | Delegate to the attached `ScheduleConfig`. |
| `.describe()` / `__repr__` | Config + live state snapshot. |

`OptimizerProfiler(window=50)` keeps a rolling `deque` of `StepProfile`
(`step`, `elapsed_ms`, `tokens`, `.tokens_per_sec` property) and exposes
`.mean_step_ms` / `.mean_tokens_per_sec`.

`fused_adamw_step(params, grads, exp_avgs, exp_avg_sqs, *, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, step)`
— a standalone, CPU-safe, in-place AdamW update function (Loshchilov &
Hutter 2019 decoupled weight decay), usable outside any optimizer class.

### Required modules

`torch`. Standard library: `dataclasses`, `math`, `time`, `collections`, `collections.abc`, `typing`.

---

## `hypernix.pressure_cooker_v4` — next-gen, built on `OptimizerBase`

v0.70.4b14. `PressureCookerV4` layers several new features onto
`OptimizerBase`, with device-specific and QAT-focused subclasses.

### `PressureCookerV4(OptimizerBase)`

```python
from hypernix.pressure_cooker_v4 import PressureCookerV4
from hypernix.optimizer_framework import ScheduleConfig

opt = PressureCookerV4(
    model.parameters(),
    schedule=ScheduleConfig(lr=3e-4),
    use_ema=True, ema_beta=0.999,
    lars_adaptation=True,
    mpt_support=True,
)
```

| Constructor arg | Default | Actually implemented? |
|---|---|---|
| `schedule` | `None` (→ default `ScheduleConfig`) | ✅ |
| `betas`, `eps`, `weight_decay` | `(0.9, 0.95)`, `1e-8`, `0.1` | ✅ standard AdamW terms |
| `grad_clip` | `None` | ✅ delegates to `OptimizerBase.gradient_clip()` |
| `use_ema` / `ema_beta` | `False` / `0.999` | ✅ — per-parameter EMA shadow, updated every step |
| `distributed_ema` | `False` | ✅ — `dist.all_reduce(..., op=AVG)` on the EMA tensor when `torch.distributed` is initialized |
| `sophia_clipping` | `False` | ⚠️ **stored but never used.** The code comment literally says "Custom Sophia clipping could augment this, but we use base for now" — the flag has no effect on `.step()`. |
| `stochastic_rounding` | `False` | ✅ — adds small uniform noise (`±5e-5`) to the update before applying it, but **only** when the param dtype is `float16`/`bfloat16`. |
| `lars_adaptation` | `False` | ✅ — LARS-style trust-ratio scaling: `local_lr = lr * (‖p‖ / ‖g‖)`. |
| `mpt_support` | `True` | ✅ but described in-source as a "very naive MPT heuristic hook": any parameter whose first dim is divisible by 3 (a proxy for MPT's fused `Wqkv` tensor) has its gradient scaled by `0.95` to reduce explosion risk. |
| `fused` | `None` | Forced to `False` automatically if running on CUDA compute capability ≤ 6.1/6.2 (Pascal). |

`.step()` order: apply LR schedule → optional gradient clip → AdamW
update (with the MPT/LARS/stochastic-rounding hooks above baked in) →
optional EMA update → increment global step.

### Subclasses

| Class | Forces |
|---|---|
| `StovetopV4Cooker` | `fused=False` — CPU/safe port of `StovetopV3Cooker` onto `OptimizerBase`. |
| `StovetopV4CookerPlus` | `fused=False`, `use_ema=True`, `grad_clip=1.0` by default. |
| `Agedcookerv4` | `fused=False`, `stochastic_rounding=False` (not well supported on Pascal). **Note:** the class docstring says it "warns" when hardware isn't CUDA 6.1/6.2-compatible, but the actual `__init__` body is `if not self.cuda_61_compatible: pass` — no warning is currently emitted; this is a placeholder. |
| `Ultracookerv4` | Accepts `qat_mode="iq4"` (also `iq1`/`iq2xxs`/`iq3s`/`iq4xl`/`iq4xs`/`q3-x`), forces `stochastic_rounding=True`. **Note:** `_adamw_step()` is overridden but just calls `super()._adamw_step()` — the comment says "Hooks for specialized iq-quantization scaling could go here," meaning `qat_mode` is currently stored but not yet wired into the actual update math. |
| `ULTRAagedcookerv4(Ultracookerv4)` | Adds `fused=False` on top of `Ultracookerv4`. |
| `CookerLite` | `fused=False`, `use_ema=False`, `mpt_support=False` — fastest/plainest CPU-only V4 variant. |

### Required modules

- `torch`, `torch.distributed` (hard dependencies)
- `hypernix.optimizer_framework` (`OptimizerBase`, `ScheduleConfig`) — internal
- `hypernix.pressure_cooker_v3` (`_flatten_optimizer_params`, `_is_cuda_61_or_older`, `_params_cuda_capability`) — internal, reused private helpers
- Standard library: `math`, `collections.abc`, `typing`

---

## `hypernix.pressure_cooker_v6` / `hypernix.pressure_cooker_v6v` — speed-first generation

Full writeup: [Pressure Cooker V6 / V6V](Pressure-Cooker-V6.md). Briefly,
for cross-reference from this page: `PressureCookerV6` (`OptimizerBase`
subclass, like V4/V5) drops V5's whole feature set (no oscillation
tracking, no curvature estimate, no quantized momentum) down to a single
fused momentum buffer updated via `torch._foreach_*` multi-tensor ops,
trading V5's adaptive-per-parameter learning rate for raw step throughput.
`PressureCookerV6V` adds CUDA graph capture (`warmup_graph`/`replay_graph`,
same contract as `pressure_cooker.ProCooker`) and optional `torch.compile`
on top, requires at least one CUDA parameter, and never touches the SSTM
update rule itself. No Pascal-specific `Agedcookerv6` tier exists — V6
never calls torch's `fused=True` AdamW kernel (the thing that actually
requires sm_70+), so there's nothing to work around. It also carries
`InductionCooker`'s real-training features — `grad_accum_steps` and
`GradScaler` integration (`grad_scaler=`) — with a batched, single-sync
non-finite check rather than a per-parameter one; see the dedicated page
for the CUDA-graph-capture caveat that comes with combining those and
`PressureCookerV6V`.

## See also

- [Pressure Cooker V3](Pressure-Cooker-V3.md) — the ZeRO/FP8-focused generation between the base tier and V4
- [Pressure Cooker V5 / V5+ / V5S](Pressure-Cooker-V5.md) — the memory-first generation after V4
- [Pressure Cooker V6 / V6V](Pressure-Cooker-V6.md) — the speed-first generation, contrasted with V5 above
- [Frameworks](Frameworks.md) — `compute_framework` (hardware abstraction) and `workshop`, a distinct module family from `optimizer_framework` despite the similar name
- `hypernix.smoker.GoodSmoker` — a much simpler heuristic LR shaping approach, contrasted with the real per-step scheduling here
