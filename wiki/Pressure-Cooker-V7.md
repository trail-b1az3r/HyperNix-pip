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

\* Hardware and workload dependent.

Notes:

- **V2:** the device-tier generation; on CUDA its `InductionCooker` is used.
- **V5+:** constructing it turns 6-bit QAT on unless told otherwise.
- **V7V:** runs anywhere; the accelerated tiers need CUDA.
- **V7Q:** no stability tracking, pressure, clipping or non-finite check by default.
- **V7M:** QAT, MTP and model preparation are explicit calls in
  `hypernix.optimizers.v7m_features`; none is on by default.

<!-- MEASURED -->
<!-- /MEASURED -->

### What the runs show, and what they do not

<!-- ANALYSIS: written once the 0.73.0a2 results are in -->


## See also

- [Optimizers](Optimizers.md): the framework and the earlier generations
- [Pressure Cooker V5 / V5+ / V5S](Pressure-Cooker-V5.md)
- [Pressure Cooker V6 / V6V](Pressure-Cooker-V6.md)
