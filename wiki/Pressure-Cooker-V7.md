# Pressure Cooker V7 and Optimizer Framework V2

New in 0.73.0a2. Modules: `hypernix.optimizers.pressure_cooker_v7`,
`pressure_cooker_v7v`, `v7m_features`, `optimizer_framework_v2`, and the
`hnx optimizer` command.

V7 is one update rule with a family of configurations around it, and a
framework that creates any HyperNix optimizer (AdamW and V1 to V7) through
one interface and can explain which one it would pick. Nothing older was
changed: V1 to V6V keep their classes, arguments, update rules and
checkpoints.

**Read this first.** The figures on this page were measured on one CPU
machine by `hnx optimizer benchmark`, on small synthetic workloads. They
are not a claim about your model. CUDA was not available where this was
written, so every GPU path (V7V's compiled step and CUDA graphs, the
automatic choice of compressed state on a GPU, V6V in the comparison) is
implemented and **unmeasured**. Run the benchmark on your hardware before
choosing on speed.

## Quick start

```python
from hypernix.optimizers.pressure_cooker_v7 import PressureCookerV7

optimizer = PressureCookerV7(model.parameters(), lr=3e-3)
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.report_loss(loss.item())   # optional: feeds Adaptive Pressure
```

Or let the framework choose and say why:

```python
from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2

choice = OptimizerFrameworkV2.auto_select(model.parameters(), objective="balanced")
print(choice.key, choice.reason_code, choice.reason)
optimizer = OptimizerFrameworkV2.build(choice, model.parameters(), lr=3e-3)
```

The default `lr` is 3e-4. In the measurements below V7 Balanced did best
at the same rate AdamW did (3e-3 on most of those small tasks); V7Q, like
V6, wanted a larger one. Sweep it for your model.

## What the update does

For every tensor, each step:

1. **Gradient norm**, one pass over all gradients. It serves the clip
   (`grad_clip`, default 1.0), the non-finite check and spike damping, and
   is read to the host once per step for the whole model. A step whose
   gradients are not finite is skipped; parameters and state are left
   exactly as they were.
2. **Factored statistics** (matrices with both sides at least
   `factored_min`): running row and column second moments, as in Adafactor,
   scale the gradient by broadcasting. There is no full second-moment
   tensor. This is what makes Balanced adapt per row and column rather than
   per tensor.
3. **Momentum**, one buffer per parameter, stored as `fp32`, `bf16`, `fp16`
   or `int8` (`state_dtype`). `"auto"`, the default, keeps `fp32` on a CPU
   (compressed momentum measured 1.6 to 5 times slower there) and picks a
   compressed type on CUDA or when free memory is short; `describe()` says
   which it chose and why.
4. **Stability score**, every `stat_interval` steps: the cosine between
   gradient and momentum per tensor, averaged at three timescales, combined
   into one score. Positive means the gradient keeps agreeing with itself.
5. **Adaptive Pressure**: a per-tensor multiplier from the stability score
   and from the loss trend you report with `report_loss`, clamped to
   `pressure_clip` (0.25 to 2.0).
6. **Trust ratio** for tensors without factored statistics: the step is
   scaled by `‖w‖ / ‖update‖`, clamped to `trust_clip` (0.05 to 5.0).
7. **Pressure diffusion** (optional, light): a three-tap blend along the
   gradient, with its strength scaled by the measured oscillation, and
   skipped when there is none.
8. **Weight decay**, decoupled.

The statistics live in small vectors on the parameters' device, one entry
per tensor, so nothing per-tensor is read to the host.

## Presets

`PressureCookerV7(params, preset=...)`. Any argument you pass overrides the
preset.

| Preset | Momentum | Statistics every | Factored | Diffusion | For |
|---|---|---|---|---|---|
| `balanced` (default) | auto | 4 steps | yes | 0.05 | most training |
| `fast` | fp32 | 8 steps | no | off | throughput, with V7's safety checks kept |
| `memory` | int8 | 4 steps | yes | 0.05 | the smallest state |
| `stable` | auto | 2 steps | yes | 0.15 | noisy or unstable training |

## The family

All of these are `PressureCookerV7` with different defaults, so they share
its arguments, its checkpoint format and its tests.

| Class | What it is |
|---|---|
| `PressureCookerV7` | Balanced. The default. |
| `PressureCookerV7V` | V7 with `torch.compile` and CUDA-graph replay where available. Falls back to the same foreach step everywhere else and records why in `describe()["fallback_reasons"]`. Never required. |
| `PressureCookerV7Mini` | For models with more than 39,000,000 and fewer than 750,000,000 trainable parameters: statistics every 8 steps, no diffusion, lazy weight decay. It runs outside that range; `describe()["band_note"]` says its advantage is not established there. |
| `PressureCookerV7Q` | Throughput first: fused momentum update, statistics every 16 steps, fp32 state, and by default **no gradient clipping and no non-finite check**, so no host read. It has no stability tracking and no Adaptive Pressure. |
| `PressureCookerV7S` | The V5S lineage: int8 momentum, factored curvature, three-timescale oscillation tracking with V5S's weights and decays, pressure diffusion with V5S's kernel. |
| `PressureCookerV7M` | The V5 / V5+ lineage: int8 momentum, factored curvature, trust region, optional EMA weights. QAT, MTP, sensitivity analysis and mixed precision are explicit calls in `v7m_features` and are off unless you call them. |

### What V7S and V7M are not

V7S is **not** a reproduction of V5S. V5S's power update, its element-wise
clip and its gradient-age freezing are not carried over; V7S follows a
different trajectory from the same start. V7M does not read V5 or V5+
checkpoints. Both refuse another optimizer's state with an error rather
than reading it as their own. If you need V5S's or V5's exact behaviour,
those classes are still there, unchanged.

### V7M's training features

```python
from hypernix.optimizers import v7m_features as f

handle = f.attach_qat(model)                 # fake-quantise linear and conv weights
sens = f.quantization_sensitivity(model, loss_on_fixed_batch, bits=(4, 6, 8))
plan = f.mixed_precision_plan(sens)          # a width per layer
handle.remove()                              # the model is as it was

head = f.make_mtp_head(hidden_dim, vocab_size)
extra = f.mtp_loss(head, hidden_states, targets)

with f.ema_weights(optimizer):               # needs ema_decay=... on the optimizer
    evaluate(model)
```

## V7V

```python
from hypernix.optimizers.pressure_cooker_v7v import PressureCookerV7V

optimizer = PressureCookerV7V(model.parameters(), lr=3e-3)   # compile="auto"
print(optimizer.describe()["kernel"])        # foreach | compiled | cuda-graph
```

`compile="auto"` compiles on CUDA only. A compiled step that fails at run
time warns once and the run continues on the foreach path.

A CUDA graph replays recorded kernels and nothing else, so anything V7
decides in Python per step would be frozen. `warmup_graph` therefore
refuses, with the list, when any of these is set: `grad_accum_steps > 1`, a
`grad_scaler`, `skip_on_nonfinite=True`, `stat_interval > 1`, diffusion.
Capturing a graph means giving up the non-finite check.

## Checkpoints

`state_dict()` carries a `"v7"` entry with the format number, variant,
state dtype, step and per-tensor statistics; resuming continues the same
trajectory. Loading into an optimizer with a different `state_dtype`
converts the momentum and warns, and says when the conversion loses
precision. A checkpoint without the `"v7"` entry (AdamW, V5, V5+, V5S, V6)
is refused.

Like PyTorch's own optimizers, `state_dict()` refers to the live tensors.
Copy it (`copy.deepcopy`) or save it before you take another step.

## Limits

- Sparse gradients are refused; use `torch.optim.SparseAdam` for that
  parameter.
- `PressureCookerV7Q` does not catch a bad batch unless you turn
  `skip_on_nonfinite` or `grad_clip` back on, which costs one norm pass.
- On a CPU, Balanced holds as much state as V6 (4 bytes per parameter).
  The smaller state needs `state_dtype="int8"`/`"bf16"` or the `memory`
  preset, and costs step time on a CPU.

## Optimizer Framework V2

```python
from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2 as F

F.names()                                   # every key
F.create("v5s", model.parameters(), lr=1e-3)    # the existing class, unchanged
F.table()                                   # the comparison matrix as data
F.diagnostics(optimizer)                    # variant, measured state bytes, kernel, ...
```

Keys: `adamw`, `v1`, `v2`, `v3`, `v4`, `v5`, `v5plus`, `v5s`, `v6`, `v6v`,
`v7`, `v7v`, `v7mini`, `v7q`, `v7s`, `v7m`.

### Auto-selection

`auto_select(params=None, *, objective, requested, parameter_count,
parameter_count_verified, device, required_features, measured, ...)`
returns a `Selection` with `key`, a stable `reason_code`, a `reason`
sentence, the `options` to build with, and `considered` (what was passed
over, and why).

| Situation | Choice | `reason_code` |
|---|---|---|
| nothing special | `v7` | `balanced` |
| a **verified** count with 39,000,000 < n < 750,000,000 | `v7mini` | `mini_band` |
| exactly 39,000,000 or 750,000,000, or outside | `v7` | `balanced` |
| count unknown or not verified | `v7` (never Mini) | `balanced` |
| `objective="throughput"`, no measurement | `v7q` | `throughput_unmeasured` |
| `objective="throughput"`, measured | `v7q` or `v6`, whichever was faster | `throughput_measured` or `throughput_v6` |
| `objective="memory"` | `v7` with `preset="memory"` | `memory` |
| `objective="stability"` | `v7s` | `stability` |
| `objective="quantization"` | `v7m` | `v5_lineage_features` |
| `objective="compatibility"` | `adamw` | `compatibility` |

A count is verified when the framework counted it from the parameters you
passed, or you pass `parameter_count_verified=True`. A guessed count never
selects V7Mini.

An explicit choice (`requested="v6v"`) is honoured or refused with the
reason. It is never swapped for something else.

## `hnx optimizer`

```bash
hnx optimizer list                                  # the optimizers and their target ratings
hnx optimizer select --params 120000000 --verified  # what auto-selection picks, and why
hnx optimizer select --objective throughput --measured results/optimizer-bench.json
hnx optimizer benchmark                             # quick set: three workloads
hnx optimizer benchmark --workloads all --out-dir results/
hnx optimizer benchmark --optimizers adamw,v6,v7,v7q --lrs 1e-3,3e-3,1e-2 --steps 300
```

The benchmark runs every optimizer on the same model from the same seed,
the same batches in the same order, in fp32, for the same number of steps,
and sweeps the same learning-rate grid for all of them. It reports:

- **step only**: `step()` on fixed gradients, median ms, and the state in
  bytes counted from the optimizer's tensors;
- **full step**: forward, backward and step together, and the optimizer's
  share of it;
- **quality**: final training loss, loss on held-out batches from the same
  task, loss at a fixed step and at a fixed wall-clock time, divergence,
  and how far the final loss moves across the learning-rate grid.

CPU and GPU utilisation, kernel launches and synchronisation counts are not
measured and are listed as such in the output. An optimizer that cannot run
(V6V without CUDA) is recorded as unavailable with the reason.

`--out-dir` writes `optimizer-bench.json`, `.csv` and `.md`. The JSON's
`measured` block is what `hnx optimizer list --measured` and
`auto_select(measured=...)` take.

## Comparison

The words in this table are **design targets**, not results. The measured
section below is what was observed.

| Optimizer | Speed | Memory | Stability | Adaptivity | Best use |
|---|---|---|---|---|---|
| AdamW | Baseline | High | High | High | General compatibility |
| Pressure Cooker V1 | Legacy | High | Legacy | Low | Legacy |
| Pressure Cooker V2 (device tiers) | Legacy | Medium | Legacy | Medium | Legacy |
| Pressure Cooker V3 | Legacy | Medium | Improved | Medium | Legacy |
| Pressure Cooker V4 | Legacy | Medium | Improved | Medium | Legacy |
| Pressure Cooker V5 | Low/Medium | Very Low | High | Specialized | Memory-constrained training |
| Pressure Cooker V5+ | Medium | Very Low | High | High | Quantization-aware training |
| Pressure Cooker V5S | Medium/Low | Extremely Low | Very High | Specialized | Stability + memory |
| Pressure Cooker V6 | Very High | Medium | Medium | Low | Maximum throughput |
| Pressure Cooker V6V | Extremely High* | Medium | Medium | Low | CUDA graph workloads |
| Pressure Cooker V7 Balanced | High target | Low target | High target | Medium/High target | General default and balanced workloads |
| Pressure Cooker V7V | High target | Low target | High target | Medium/High target | V7 with torch.compile and CUDA graphs where available |
| Pressure Cooker V7Mini | Tuned for small-model throughput | Low overhead target | Benchmark required | Architecture-dependent | Models strictly above 39M and below 750M parameters |
| Pressure Cooker V7Q | Fastest variant target | Low/medium target | Benchmark required | Lower adaptivity is acceptable if documented | Maximum throughput where measured faster |
| Pressure Cooker V7S | Benchmark required | Very low state target | High target | Specialized | Modernized V5S-style stability and memory efficiency |
| Pressure Cooker V7M | Benchmark required | Lower than equivalent V5/V5+ target | High target | High target | Modernized V5/V5+ features, QAT/MTP and sensitivity-aware training |

* Hardware and workload dependent.
Constructing it turns 6-bit QAT on unless told otherwise.
No stability tracking, pressure, clipping or non-finite check by default.
QAT, MTP and model preparation are explicit calls in hypernix.optimizers.v7m_features; none is on by default.
Runs anywhere; the accelerated tiers need CUDA.
The device-tier generation; on CUDA its InductionCooker is used.

<!-- MEASURED -->
### Measured

`hnx optimizer benchmark --workloads all --steps 200` on Linux, 4 CPU threads, torch 2.14.1+cu130, fp32, seed 0. Each optimizer trained for 200 steps at each of 0.0003, 0.001, 0.003, 0.01; the row shows its best rate. V6V needs CUDA and did not run.

**small-transformer** (70,080 parameters)

| Optimizer | Step only, ms | State bytes/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share |
|---|---|---|---|---|---|---|---|
| adamw | 0.82 | 8.00 | 0.01 | 0.8723 | 1.2203 | 4.98 | 24% |
| v5 | 6.15 | 1.23 | 0.01 | 2.5073 | 2.4800 | 8.97 | 66% |
| v5plus | 8.26 | 6.13 | 0.01 | 2.5507 | 2.5209 | 14.27 | 69% |
| v5s | 10.35 | 1.23 | 0.01 | 2.5069 | 2.4796 | 12.80 | 69% |
| v6 | 0.51 | 4.00 | 0.01 | 4.5623 | 4.5530 | 3.18 | 18% |
| v6v | N/A (needs CUDA) | | | | | | |
| v7 | 0.79 | 4.24 | 0.01 | 1.5083 | 1.6199 | 4.42 | 26% |
| v7mini | 0.70 | 4.24 | 0.01 | 1.6386 | 1.7214 | 4.31 | 25% |
| v7q | 0.16 | 4.01 | 0.01 | 4.5621 | 4.5527 | 4.00 | 11% |
| v7s | 2.27 | 1.24 | 0.01 | 1.4060 | 1.5761 | 6.61 | 41% |
| v7m | 1.73 | 1.24 | 0.01 | 1.5129 | 1.6818 | 5.05 | 33% |

**medium-transformer** (387,936 parameters)

| Optimizer | Step only, ms | State bytes/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share |
|---|---|---|---|---|---|---|---|
| adamw | 1.65 | 8.00 | 0.003 | 0.6171 | 1.2598 | 10.92 | 21% |
| v5 | 14.15 | 1.12 | 0.01 | 1.9761 | 2.0253 | 25.56 | 58% |
| v5plus | 17.73 | 6.07 | 0.01 | 2.0559 | 2.0913 | 26.17 | 65% |
| v5s | 19.20 | 1.12 | 0.01 | 1.9740 | 2.0261 | 30.64 | 72% |
| v6 | 2.10 | 4.00 | 0.01 | 5.1106 | 5.0542 | 12.00 | 20% |
| v6v | N/A (needs CUDA) | | | | | | |
| v7 | 1.99 | 4.12 | 0.003 | 1.3572 | 1.6161 | 11.84 | 21% |
| v7mini | 1.26 | 4.12 | 0.003 | 1.4090 | 1.6591 | 9.72 | 19% |
| v7q | 0.38 | 4.00 | 0.01 | 5.1103 | 5.0538 | 7.97 | 9% |
| v7s | 3.36 | 1.12 | 0.003 | 1.2713 | 1.5432 | 12.83 | 37% |
| v7m | 3.08 | 1.12 | 0.003 | 1.3586 | 1.6225 | 14.41 | 29% |

**large-transformer** (1,985,664 parameters)

| Optimizer | Step only, ms | State bytes/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share |
|---|---|---|---|---|---|---|---|
| adamw | 2.98 | 8.00 | 0.003 | 0.2452 | 0.9230 | 35.82 | 11% |
| v5 | 29.87 | 1.06 | 0.01 | 1.4324 | 1.4459 | 59.06 | 46% |
| v5plus | 37.76 | 6.03 | 0.01 | 1.5055 | 1.4928 | 59.61 | 50% |
| v5s | 39.15 | 1.06 | 0.01 | 1.4245 | 1.4343 | 94.29 | 63% |
| v6 | 3.57 | 4.00 | 0.01 | 5.6632 | 5.5434 | 34.10 | 13% |
| v6v | N/A (needs CUDA) | | | | | | |
| v7 | 3.16 | 4.06 | 0.003 | 0.7971 | 1.0663 | 39.65 | 13% |
| v7mini | 4.25 | 4.06 | 0.003 | 0.8787 | 1.1018 | 38.40 | 11% |
| v7q | 0.97 | 4.00 | 0.01 | 5.6627 | 5.5427 | 39.09 | 6% |
| v7s | 13.43 | 1.06 | 0.003 | 0.6994 | 0.9947 | 40.89 | 23% |
| v7m | 5.74 | 1.06 | 0.003 | 0.7970 | 1.0704 | 47.31 | 20% |

**mlp** (86,544 parameters)

| Optimizer | Step only, ms | State bytes/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share |
|---|---|---|---|---|---|---|---|
| adamw | 0.34 | 8.00 | 0.003 | 0.1701 | 0.8938 | 1.23 | 45% |
| v5 | 2.18 | 1.09 | 0.01 | 1.7997 | 1.7804 | 2.67 | 81% |
| v5plus | 2.61 | 6.05 | 0.01 | 1.7910 | 1.7710 | 3.30 | 84% |
| v5s | 3.06 | 1.09 | 0.01 | 1.8005 | 1.7822 | 3.88 | 86% |
| v6 | 0.30 | 4.00 | 0.01 | 2.6832 | 2.6863 | 1.19 | 45% |
| v6v | N/A (needs CUDA) | | | | | | |
| v7 | 0.65 | 4.10 | 0.003 | 0.2630 | 0.6925 | 1.36 | 52% |
| v7mini | 0.36 | 4.10 | 0.003 | 0.3192 | 0.7018 | 1.29 | 49% |
| v7q | 0.09 | 4.00 | 0.01 | 2.6834 | 2.6866 | 0.75 | 28% |
| v7s | 0.90 | 1.10 | 0.003 | 0.2533 | 0.6919 | 1.63 | 64% |
| v7m | 0.64 | 1.10 | 0.003 | 0.2629 | 0.6944 | 1.34 | 59% |

**embedding-heavy** (1,285,200 parameters)

| Optimizer | Step only, ms | State bytes/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share |
|---|---|---|---|---|---|---|---|
| adamw | 1.91 | 8.00 | 0.01 | 0.0086 | 5.0736 | 2.77 | 66% |
| v5 | 24.32 | 1.08 | 0.01 | 2.6779 | 2.7346 | 18.73 | 94% |
| v5plus | 26.26 | 6.05 | 0.01 | 2.6976 | 2.7369 | 23.12 | 94% |
| v5s | 55.56 | 1.08 | 0.01 | 2.6796 | 2.7347 | 57.84 | 98% |
| v6 | 3.05 | 4.00 | 0.01 | 2.7404 | 2.7428 | 3.48 | 73% |
| v6v | N/A (needs CUDA) | | | | | | |
| v7 | 1.56 | 4.13 | 0.003 | 0.5973 | 4.2796 | 2.44 | 69% |
| v7mini | 1.74 | 4.13 | 0.003 | 0.3567 | 5.4862 | 2.69 | 66% |
| v7q | 0.52 | 4.00 | 0.01 | 2.7401 | 2.7428 | 1.48 | 43% |
| v7s | 4.40 | 1.13 | 0.001 | 1.3598 | 3.4100 | 5.14 | 83% |
| v7m | 2.61 | 1.13 | 0.003 | 0.6446 | 3.8487 | 3.02 | 75% |

**mixed-1d-2d** (38,280 parameters)

| Optimizer | Step only, ms | State bytes/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share |
|---|---|---|---|---|---|---|---|
| adamw | 0.37 | 8.00 | 0.003 | 0.3087 | 1.0343 | 1.45 | 35% |
| v5 | 2.92 | 1.13 | 0.01 | 1.3271 | 1.4801 | 3.55 | 75% |
| v5plus | 3.52 | 6.07 | 0.01 | 1.3420 | 1.4846 | 5.23 | 81% |
| v5s | 5.24 | 1.13 | 0.01 | 1.3245 | 1.4787 | 4.78 | 81% |
| v6 | 0.21 | 4.00 | 0.01 | 2.6858 | 2.6831 | 1.22 | 30% |
| v6v | N/A (needs CUDA) | | | | | | |
| v7 | 0.40 | 4.17 | 0.003 | 0.4725 | 0.9898 | 1.47 | 41% |
| v7mini | 0.47 | 4.17 | 0.003 | 0.5088 | 1.0426 | 1.38 | 37% |
| v7q | 0.09 | 4.01 | 0.01 | 2.6857 | 2.6831 | 1.05 | 20% |
| v7s | 0.90 | 1.18 | 0.003 | 0.4630 | 0.9558 | 1.91 | 57% |
| v7m | 0.69 | 1.18 | 0.003 | 0.4727 | 0.9911 | 1.71 | 50% |
### What that run shows, and what it does not

Averaged over the six workloads (200 steps each, CPU):

| Optimizer | Step only, ms | Full step, ms | State bytes/param | Final training loss | Held-out loss |
|---|---|---|---|---|---|
| AdamW | 1.34 | 9.53 | 8.00 | 0.370 | 1.734 |
| V5 | 13.26 | 19.76 | 1.12 | 1.953 | 1.991 |
| V5+ | 16.02 | 21.95 | 6.07 | 1.990 | 2.016 |
| V5S | 22.09 | 34.04 | 1.12 | 1.952 | 1.989 |
| V6 | 1.63 | 9.20 | 4.00 | 3.908 | 3.877 |
| V7 Balanced | 1.43 | 10.20 | 4.14 | 0.833 | 1.711 |
| V7Mini | 1.46 | 9.63 | 4.14 | 0.852 | 1.952 |
| V7Q | 0.37 | 9.06 | 4.01 | 3.907 | 3.877 |
| V7S | 4.21 | 11.50 | 1.14 | 0.909 | 1.529 |
| V7M | 2.42 | 12.14 | 1.14 | 0.841 | 1.651 |

- **AdamW reached the lowest training loss on every workload** in 200
  steps. V7 Balanced did not beat it on training loss anywhere here. Its
  held-out loss was within a few percent of AdamW's on average, better on
  three workloads and worse on three.
- **V7 Balanced's step costs about what AdamW's does** (1.43 against 1.34
  ms) **with about half the state** (4.14 against 8.00 bytes per
  parameter). It is not faster than V6 in the full step.
- **V7Q's step is the cheapest measured** (0.37 ms, against 1.63 for V6),
  and it trains exactly as V6 does on this grid. In a whole training step
  the difference mostly disappears on the transformers, where the optimizer
  is 6 to 13 percent of the step.
- **V7S and V7M hold about 1.1 bytes per parameter**, as V5 and V5S do,
  at a fifth to a third of their step time, and trained to a lower loss
  than V5, V5+ and V5S on every workload on this grid.
- **V6, V7Q, V5, V5+ and V5S did best at the largest rate tried (0.01).**
  The grid stops there, so their best is probably beyond it and these
  rows understate them. Their losses here are not their ceiling.
- **V7Mini was not tested in its band.** The largest workload has 2
  million parameters; its band starts above 39 million. Nothing here shows
  Mini is better than Balanced where it is meant to be used.
- These are short runs on small synthetic tasks, one seed, one machine.
  They say nothing about a GPU or about convergence of a real model.

Reproduce with `hnx optimizer benchmark --workloads all --steps 200`.
<!-- /MEASURED -->

## See also

- [Optimizers](Optimizers.md): the framework and the earlier generations
- [Pressure Cooker V5 / V5+ / V5S](Pressure-Cooker-V5.md)
- [Pressure Cooker V6 / V6V](Pressure-Cooker-V6.md)
