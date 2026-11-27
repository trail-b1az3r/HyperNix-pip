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

The default `lr` is 3e-4. The optimizers do not share a learning-rate
scale. In the measurements below, the best rates were:

| Optimizer | Best rate on these tasks |
|---|---|
| AdamW, V7 Balanced, V7Mini, V7S, V7M | 1e-3 to 1e-2, mostly 3e-3 |
| V5, V5+, V5S | 3e-2 to 1e-1 |
| V6, V7Q | 1e-1 to 1.0, and on the transformers still the largest rate tried |

Sweep it for your model; comparing optimizers at one shared rate measures
which of them that rate suits.

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
   (int8 momentum measured 1.6 to 1.8 times slower there in the
   ablations) unless fp32 would take
   more than half the free memory. On CUDA it picks bf16 (or fp16), and
   int8 when fp32 would take more than a quarter of free memory.
   `describe()` says which it chose and why.
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
hnx optimizer benchmark --ablations                 # V7, V7S, V7M with one mechanism off
hnx optimizer benchmark --band                      # step() at 30M, 39M, 39.1M, 150M
hnx optimizer benchmark --band 749000000,750000000  # the band's upper edge (~10 GiB)
hnx optimizer benchmark --workloads band-transformer   # training at 41.6M (opt-in)
```

The benchmark runs every optimizer on the same model from the same seed,
the same batches in the same order, in fp32, for the same number of steps,
and sweeps the same learning-rate grid for all of them. It reports:

- **step only**: `step()` on fixed gradients, median, p10 and p90 ms and
  every raw time; the state in bytes counted from the optimizer's
  tensors; what saving and loading that state costs;
- **full step**: forward, backward and step together, and the optimizer's
  share of it;
- **quality**: the loss at every step, final training loss, loss on
  held-out batches from the same task, loss at a fixed step and at a
  fixed wall-clock time, the gradient norm's median and peak, divergence,
  how far the final loss moves across the learning-rate grid, and whether
  an optimizer's best rate sat at the edge of the grid.

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
### Measured

Every figure in this section is read from `benchmarks/results/optimizers-0.73.0a2/` by `scripts/render_optimizer_results.py`; the commands that produced each file are in that folder's README. One machine, one seed, CPU only.

#### Averaged over the six workloads

Run: HyperNix 0.73.0a2, torch 2.15.0+cpu, Linux, 4 CPU threads, fp32, seed 0, 200 training steps at each of 0.0003, 0.001, 0.003, 0.01, 0.03; 30 optimizer-only steps after 5 warm-up.

| Optimizer | Step only, ms | Full step, ms | State bytes/param | Final training loss | Held-out loss | Workloads |
|---|---|---|---|---|---|---|
| AdamW | 1.58 | 10.07 | 8.00 | 0.370 | 1.734 | 6 |
| V5 | 13.05 | 20.56 | 1.12 | 0.973 | 1.509 | 6 |
| V5+ | 16.43 | 25.47 | 6.07 | 1.142 | 1.486 | 6 |
| V5S | 25.94 | 34.33 | 1.12 | 0.969 | 1.499 | 6 |
| V6 | 1.25 | 10.08 | 4.00 | 3.303 | 3.229 | 6 |
| V7 Balanced | 1.66 | 10.02 | 4.14 | 0.833 | 1.711 | 6 |
| V7Mini | 1.35 | 9.62 | 4.14 | 0.852 | 1.952 | 6 |
| V7Q | 0.40 | 8.50 | 4.00 | 3.299 | 3.223 | 6 |
| V7S | 3.48 | 12.74 | 1.14 | 0.909 | 1.529 | 6 |
| V7M | 2.97 | 11.72 | 1.14 | 0.841 | 1.651 | 6 |
| V6V | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | |

#### Rates above the grid

V5, V5+, V5S, V6 and V7Q did best at the top of the main grid, so they were run again at higher rates.

Run: HyperNix 0.73.0a2, torch 2.15.0+cpu, Linux, 4 CPU threads, fp32, seed 0, 200 training steps at each of 0.03, 0.1, 0.3, 1; 30 optimizer-only steps after 5 warm-up.

Best final training loss over 0.0003 to 1, per workload (rate in brackets; `edge` means the best was still the largest rate tried):

| Workload | V5 | V5+ | V5S | V6 | V7Q |
|---|---|---|---|---|---|
| small-transformer | 1.517 (0.03) | 1.547 (0.03) | 1.496 (0.1) | 1.812 (1, edge) | 1.960 (1, edge) |
| medium-transformer | 1.037 (0.03) | 1.091 (0.03) | 1.023 (0.03) | 1.669 (1, edge) | 1.731 (1, edge) |
| large-transformer | 0.680 (0.03) | 0.716 (0.03) | 0.664 (0.03) | 1.596 (1, edge) | 1.613 (1, edge) |
| mlp | 0.621 (0.03) | 0.620 (0.03) | 0.614 (0.03) | 0.683 (0.3) | 0.832 (0.3) |
| embedding-heavy | 0.093 (0.1) | 0.107 (0.1) | 0.070 (0.1) | 2.685 (0.1) | 2.685 (0.1) |
| mixed-1d-2d | 0.680 (0.03) | 0.692 (0.03) | 0.656 (0.03) | 0.869 (0.1) | 1.210 (0.1) |

#### Per workload

**small-transformer** (70,080 parameters)

| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |
|---|---|---|---|---|---|---|---|---|
| AdamW | 0.96 (0.93–1.09) | 8.00 | 0.01 | 0.8723 | 1.2203 | 4.67 | 24% | 1.27 |
| V5 | 7.28 (7.11–7.47) | 1.23 | 0.03 (edge) | 1.5166 | 1.7172 | 11.67 | 68% | 2.29 |
| V5+ | 10.87 (10.55–11.80) | 6.13 | 0.03 (edge) | 1.5471 | 1.7386 | 16.09 | 74% | 2.40 |
| V5S | 10.82 (10.11–12.07) | 1.23 | 0.03 (edge) | 1.5055 | 1.7185 | 14.65 | 75% | 2.32 |
| V6 | 0.64 (0.58–0.70) | 4.00 | 0.03 (edge) | 3.9171 | 3.8755 | 4.52 | 17% | 1.21 |
| V6V | N/A | | | | | | | |
| V7 Balanced | 1.10 (1.01–2.08) | 4.24 | 0.01 | 1.5083 | 1.6199 | 5.14 | 24% | 1.46 |
| V7Mini | 1.38 (1.03–2.22) | 4.24 | 0.01 | 1.6386 | 1.7214 | 4.99 | 24% | 1.43 |
| V7Q | 0.24 (0.19–0.35) | 4.01 | 0.03 (edge) | 3.9147 | 3.8729 | 4.07 | 8% | 1.21 |
| V7S | 2.81 (2.11–3.24) | 1.24 | 0.01 | 1.4060 | 1.5761 | 6.47 | 46% | 1.32 |
| V7M | 2.08 (1.88–2.96) | 1.24 | 0.01 | 1.5129 | 1.6818 | 6.51 | 36% | 1.42 |

**medium-transformer** (387,936 parameters)

| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |
|---|---|---|---|---|---|---|---|---|
| AdamW | 1.97 (1.85–2.34) | 8.00 | 0.003 | 0.6171 | 1.2598 | 11.07 | 19% | 1.25 |
| V5 | 15.78 (15.28–16.46) | 1.12 | 0.03 (edge) | 1.0368 | 1.4706 | 23.90 | 64% | 1.58 |
| V5+ | 19.79 (19.36–21.11) | 6.07 | 0.03 (edge) | 1.0909 | 1.4624 | 31.52 | 71% | 1.60 |
| V5S | 24.12 (23.11–25.57) | 1.12 | 0.03 (edge) | 1.0231 | 1.4596 | 33.41 | 74% | 1.59 |
| V6 | 1.44 (1.37–1.57) | 4.00 | 0.03 (edge) | 4.1698 | 4.0423 | 10.66 | 16% | 1.23 |
| V6V | N/A | | | | | | | |
| V7 Balanced | 2.04 (1.81–3.92) | 4.12 | 0.003 | 1.3572 | 1.6161 | 10.82 | 19% | 1.31 |
| V7Mini | 1.73 (1.59–3.65) | 4.12 | 0.003 | 1.4090 | 1.6591 | 11.28 | 17% | 1.28 |
| V7Q | 0.51 (0.45–0.64) | 4.00 | 0.03 (edge) | 4.1659 | 4.0383 | 9.16 | 7% | 1.22 |
| V7S | 4.99 (3.40–5.59) | 1.12 | 0.003 | 1.2713 | 1.5432 | 13.04 | 41% | 1.35 |
| V7M | 3.59 (3.41–5.31) | 1.12 | 0.003 | 1.3586 | 1.6225 | 13.05 | 30% | 1.31 |

**large-transformer** (1,985,664 parameters)

| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |
|---|---|---|---|---|---|---|---|---|
| AdamW | 2.90 (2.85–2.99) | 8.00 | 0.003 | 0.2452 | 0.9230 | 39.06 | 11% | 1.83 |
| V5 | 29.61 (27.87–37.95) | 1.06 | 0.03 (edge) | 0.6796 | 1.1643 | 61.02 | 46% | 1.36 |
| V5+ | 36.25 (34.49–39.96) | 6.03 | 0.03 (edge) | 0.7160 | 1.1353 | 69.69 | 53% | 1.39 |
| V5S | 47.98 (45.74–49.83) | 1.06 | 0.03 (edge) | 0.6641 | 1.1585 | 86.49 | 60% | 1.36 |
| V6 | 3.05 (2.88–3.49) | 4.00 | 0.03 (edge) | 4.4558 | 4.1950 | 40.51 | 11% | 1.23 |
| V6V | N/A | | | | | | | |
| V7 Balanced | 4.25 (2.98–8.40) | 4.06 | 0.003 | 0.7971 | 1.0663 | 38.52 | 11% | 1.28 |
| V7Mini | 2.81 (2.67–6.20) | 4.06 | 0.003 | 0.8787 | 1.1018 | 36.62 | 10% | 1.26 |
| V7Q | 1.03 (1.00–1.16) | 4.00 | 0.03 (edge) | 4.4503 | 4.1887 | 34.50 | 4% | 1.23 |
| V7S | 7.70 (5.85–9.71) | 1.06 | 0.003 | 0.6994 | 0.9947 | 47.85 | 22% | 1.24 |
| V7M | 7.94 (6.28–10.89) | 1.06 | 0.003 | 0.7970 | 1.0704 | 43.60 | 17% | 1.29 |

**mlp** (86,544 parameters)

| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |
|---|---|---|---|---|---|---|---|---|
| AdamW | 0.42 (0.34–0.50) | 8.00 | 0.003 | 0.1701 | 0.8938 | 1.42 | 43% | 1.61 |
| V5 | 3.73 (2.73–4.25) | 1.09 | 0.03 (edge) | 0.6208 | 0.7713 | 3.75 | 81% | 3.39 |
| V5+ | 3.64 (3.50–4.31) | 6.05 | 0.03 (edge) | 0.6196 | 0.7620 | 5.95 | 86% | 3.35 |
| V5S | 5.66 (4.32–6.96) | 1.09 | 0.03 (edge) | 0.6140 | 0.7506 | 4.95 | 85% | 3.31 |
| V6 | 0.36 (0.34–0.48) | 4.00 | 0.03 (edge) | 2.4681 | 2.4522 | 1.09 | 41% | 1.42 |
| V6V | N/A | | | | | | | |
| V7 Balanced | 0.60 (0.51–1.03) | 4.10 | 0.003 | 0.2630 | 0.6925 | 1.68 | 51% | 1.72 |
| V7Mini | 0.48 (0.42–0.91) | 4.10 | 0.003 | 0.3192 | 0.7018 | 1.26 | 48% | 1.77 |
| V7Q | 0.14 (0.12–0.17) | 4.00 | 0.03 (edge) | 2.4656 | 2.4493 | 0.85 | 29% | 1.43 |
| V7S | 1.19 (0.78–1.34) | 1.10 | 0.003 | 0.2533 | 0.6919 | 2.01 | 66% | 1.68 |
| V7M | 0.90 (0.80–1.46) | 1.10 | 0.003 | 0.2629 | 0.6944 | 2.09 | 62% | 1.72 |

**embedding-heavy** (1,285,200 parameters)

| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |
|---|---|---|---|---|---|---|---|---|
| AdamW | 2.81 (2.34–3.06) | 8.00 | 0.01 | 0.0086 | 5.0736 | 2.05 | 67% | 2.46 |
| V5 | 17.65 (15.84–21.77) | 1.08 | 0.03 (edge) | 1.3066 | 2.8899 | 18.17 | 95% | 1.96 |
| V5+ | 23.22 (19.24–27.98) | 6.05 | 0.03 (edge) | 2.1868 | 2.7693 | 23.25 | 96% | 1.66 |
| V5S | 61.69 (58.60–74.10) | 1.08 | 0.03 (edge) | 1.3515 | 2.8772 | 60.06 | 98% | 1.98 |
| V6 | 1.59 (1.56–1.94) | 4.00 | 0.03 (edge) | 2.7004 | 2.7431 | 2.12 | 71% | 1.38 |
| V6V | N/A | | | | | | | |
| V7 Balanced | 1.37 (1.24–2.97) | 4.13 | 0.003 | 0.5973 | 4.2796 | 1.89 | 67% | 5.16 |
| V7Mini | 1.18 (1.14–2.60) | 4.13 | 0.003 | 0.3567 | 5.4862 | 1.79 | 65% | 7.01 |
| V7Q | 0.39 (0.36–0.45) | 4.00 | 0.03 (edge) | 2.7003 | 2.7432 | 1.11 | 42% | 1.38 |
| V7S | 2.88 (2.02–3.70) | 1.13 | 0.001 | 1.3598 | 3.4100 | 4.01 | 83% | 4.49 |
| V7M | 2.25 (2.00–3.75) | 1.13 | 0.003 | 0.6446 | 3.8487 | 2.74 | 75% | 5.51 |

**mixed-1d-2d** (38,280 parameters)

| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |
|---|---|---|---|---|---|---|---|---|
| AdamW | 0.44 (0.41–0.51) | 8.00 | 0.003 | 0.3087 | 1.0343 | 2.14 | 32% | 1.53 |
| V5 | 4.25 (3.34–4.64) | 1.13 | 0.03 (edge) | 0.6805 | 1.0402 | 4.84 | 77% | 2.53 |
| V5+ | 4.79 (4.70–5.07) | 6.07 | 0.03 (edge) | 0.6921 | 1.0507 | 6.35 | 82% | 2.54 |
| V5S | 5.39 (5.06–5.87) | 1.13 | 0.03 (edge) | 0.6563 | 1.0297 | 6.45 | 82% | 2.55 |
| V6 | 0.44 (0.37–0.56) | 4.00 | 0.03 (edge) | 2.1073 | 2.0649 | 1.61 | 30% | 3.20 |
| V6V | N/A | | | | | | | |
| V7 Balanced | 0.57 (0.53–0.99) | 4.17 | 0.003 | 0.4725 | 0.9898 | 2.07 | 42% | 1.68 |
| V7Mini | 0.53 (0.48–0.94) | 4.17 | 0.003 | 0.5088 | 1.0426 | 1.78 | 39% | 1.70 |
| V7Q | 0.12 (0.11–0.17) | 4.01 | 0.03 (edge) | 2.0949 | 2.0473 | 1.32 | 18% | 3.23 |
| V7S | 1.34 (0.93–1.58) | 1.18 | 0.003 | 0.4630 | 0.9558 | 3.06 | 54% | 1.63 |
| V7M | 1.04 (0.93–1.44) | 1.18 | 0.003 | 0.4727 | 0.9911 | 2.32 | 50% | 1.67 |

#### Ablations

One mechanism changed at a time, averaged over small-transformer, mlp, mixed-1d-2d.

Run: HyperNix 0.73.0a2, torch 2.15.0+cpu, Linux, 4 CPU threads, fp32, seed 0, 200 training steps at each of 0.0003, 0.001, 0.003, 0.01, 0.03; 30 optimizer-only steps after 5 warm-up.

| Configuration | Step only, ms | Full step, ms | State bytes/param | Final training loss | Held-out loss |
|---|---|---|---|---|---|
| Pressure Cooker V7 Balanced | 0.86 | 2.98 | 4.17 | 0.748 | 1.101 |
| Pressure Cooker V7S | 1.83 | 3.71 | 1.17 | 0.707 | 1.075 |
| Pressure Cooker V7M | 1.50 | 3.43 | 1.17 | 0.750 | 1.122 |
| Pressure Cooker V5 | 4.59 | 6.78 | 1.15 | 0.939 | 1.176 |
| Pressure Cooker V5S | 6.41 | 8.56 | 1.15 | 0.925 | 1.166 |
| Pressure Cooker V7 Balanced with factored=False | 0.49 | 2.40 | 4.01 | 2.729 | 2.695 |
| Pressure Cooker V7 Balanced with stability=False | 0.69 | 2.86 | 4.17 | 0.751 | 1.109 |
| Pressure Cooker V7 Balanced with adaptive_pressure=False | 0.66 | 2.93 | 4.17 | 0.742 | 1.111 |
| Pressure Cooker V7 Balanced with diffusion=False | 0.89 | 2.84 | 4.17 | 0.748 | 1.101 |
| Pressure Cooker V7 Balanced with state_dtype=int8 | 1.34 | 3.26 | 1.17 | 0.750 | 1.122 |
| Pressure Cooker V7S with diffusion=False | 1.60 | 3.75 | 1.17 | 0.707 | 1.075 |
| Pressure Cooker V7S with stability=False | 1.53 | 3.41 | 1.17 | 0.722 | 1.071 |
| Pressure Cooker V7S with factored=False | 1.37 | 3.36 | 1.01 | 2.658 | 2.625 |
| Pressure Cooker V7S with state_dtype=fp32 | 0.99 | 3.18 | 4.17 | 0.709 | 1.072 |
| Pressure Cooker V7M with factored=False | 1.17 | 3.25 | 1.01 | 2.729 | 2.695 |
| Pressure Cooker V7M with state_dtype=fp32 | 0.80 | 2.84 | 4.17 | 0.748 | 1.101 |
| Pressure Cooker V7M with ema_decay=0.999 | 1.59 | 3.87 | 5.17 | 0.750 | 1.122 |

#### Inside V7Mini's band: training

`band-transformer`, 41,636,864 parameters. Run: HyperNix 0.73.0a2, torch 2.15.0+cpu, Linux, 4 CPU threads, fp32, seed 0, 30 training steps at each of 0.001, 0.003, 0.01; 6 optimizer-only steps after 2 warm-up.

| Optimizer | Step only, ms | Full step, ms | Final / held-out at 0.001 | Final / held-out at 0.003 | Final / held-out at 0.01 |
|---|---|---|---|---|---|
| AdamW | 91.8 | 253.5 | 7.472 / 13.678 | 8.586 / 17.347 | 12.872 / 22.340 |
| V6 | 80.0 | 223.5 | 9.854 / 9.833 | 9.849 / 9.832 | 9.831 / 9.832 |
| V7 Balanced | 52.6 | 202.1 | 4.457 / 9.697 | 2.443 / 10.048 | 3.751 / 10.823 |
| V7Mini | 45.1 | 211.6 | 2.259 / 9.710 | 0.416 / 9.908 | 1.609 / 22.349 |
| V7Q | 19.4 | 163.2 | 9.852 / 9.832 | 9.842 / 9.832 | 9.807 / 9.831 |

#### Around V7Mini's band: `step()` alone

Tensors with a transformer's shapes and exactly the parameter count shown, random gradients; median ms per step, p10–p90 in brackets. A dtype after them is the momentum `state_dtype="auto"` chose when it was not fp32. This is the optimizer's cost at that size and says nothing about training quality.

| Parameters | In band | Auto-selects | AdamW, ms | V6, ms | V7 Balanced, ms | V7Mini, ms | V7Q, ms |
|---|---|---|---|---|---|---|---|
| 30,000,000 | no | V7 Balanced | 58.8 (52–61) | 65.5 (58–72) | 36.9 (33–57) | 36.4 (33–39) | 15.4 (13–18) |
| 39,000,000 | no | V7 Balanced | 53.8 (53–55) | 75.0 (71–79) | 53.6 (47–73) | 43.7 (40–49) | 19.9 (17–24) |
| 39,100,000 | yes | V7Mini | 52.3 (50–56) | 88.3 (71–96) | 53.4 (49–63) | 41.4 (41–43) | 23.8 (17–27) |
| 150,000,000 | yes | V7Mini | 226.6 (217–234) | 291.5 (270–299) | 193.4 (178–217) | 173.0 (163–178) | 67.9 (62–77) |
| 749,000,000 | yes | V7Mini | 2075.2 (1815–2237) | 1500.9 (1473–1549) | 1633.7 (1378–1897) | 1420.6 (1142–1704) | 372.9 (359–378) |
| 750,000,000 | no | V7 Balanced | 1942.9 (1520–2118) | 1436.3 (1385–1452) | 1489.1 (1340–1573) | 1414.5 (1202–1567) | 386.2 (339–400) |
<!-- /MEASURED -->

### What the runs show, and what they do not

Each optimizer is compared at its own best rate, across both grids. All
of it is CPU, one seed, short runs on small synthetic tasks.

- **AdamW reached the lowest training loss on every workload**, and the
  lowest held-out loss on all three transformers. Nothing in the V7
  family beat it there.
- **The V7 family had the lowest held-out loss on the MLP and the mixed
  model** (V7S: 0.69 and 0.96, against AdamW's 0.89 and 1.03). On the
  embedding-heavy model every optimizer that fit the training set overfit
  it. The lowest held-out loss there is V6's and V7Q's, 2.76, because
  they learned nothing: ln 16 = 2.77 is a uniform guess over its 16
  classes.
- **V7 Balanced costs what AdamW and V6 cost per training step** (10.02
  ms against 10.07 and 10.08, averaged) **with half of AdamW's state**
  (4.14 against 8.00 bytes per parameter). Its `step()` alone is slower
  than V6's (1.66 against 1.25 ms). It reached a lower training loss than
  V6 on every workload, and a lower held-out loss on five of six.
- **V7Q has the cheapest `step()` on every workload** (0.40 ms averaged,
  against 1.25 for V6) **and the fastest full training step** (8.50 ms
  against 10.08). It pays for it in quality: at each one's best rate its
  final loss was worse than V6's on five workloads and equal on the sixth.
- **V7S against V5S** (each at its best rate): a lower held-out loss on
  five of six workloads, but a lower training loss on only three (V5S was
  lower on the medium and large transformers and the embedding model).
  Its `step()` is about seven times cheaper (3.48 against 25.94 ms) at the
  same state, about 1.1 bytes per parameter.
- **V7M against V5 and V5+**: the same pattern. Lower held-out loss on five
  of six, lower training loss on three, and a step four to six times
  cheaper. Run without its explicit features, V7M's optimizer is V7's
  `memory` preset: the ablations show V7M with fp32 state is exactly V7
  Balanced. Its distinct value is in `v7m_features`, which these runs do
  not use.
- **Ablations: the factored statistics do almost all the work.** Without
  them V7's training loss goes from 0.748 to 2.729, which is V6's
  territory. Turning off stability tracking or Adaptive Pressure moved the
  loss by about 0.01. **Diffusion changed nothing, to four decimals, in
  V7 and in V7S**: its gate opens only on oscillation, and on these tasks
  it never opened. These runs do not test diffusion. int8 momentum gave
  the same loss with a quarter of the state, for about 0.5 ms more per
  step.
- **V7Mini's `step()` was cheaper than Balanced's at every band size
  measured, but also below the band** (30M and 39M). So nothing here shows
  the advantage is specific to the band. At 749M and 750M the p10–p90
  spreads of Balanced and Mini overlap. In the 41.6M training run Mini's
  `step()` was cheaper (45 against 53 ms), but its full step was not
  (212 against 202 ms, median across rates).
- **The 41.6M training run measures memorisation, not learning**: 30
  steps over 8 repeated batches. Every optimizer's held-out loss sat at or
  above ln 16000 = 9.68, a uniform guess. Mini's lower training loss there
  means it memorised those batches faster.
- **On this CPU, V6's `step()` was slower than AdamW's from 30M to 150M
  parameters** and faster at 749M and 750M.
- **Not measured:** anything on a GPU. V6V is unavailable without CUDA,
  and V7V ran its CPU fallback.


## See also

- [Optimizers](Optimizers.md): the framework and the earlier generations
- [Pressure Cooker V5 / V5+ / V5S](Pressure-Cooker-V5.md)
- [Pressure Cooker V6 / V6V](Pressure-Cooker-V6.md)
