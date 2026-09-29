# PressureCooker V5 / V5S: An Oscillation-Resistant, Quantized-State Optimizer Family for Memory-Constrained Training

**Author:** ray0rf1re
**Package:** [`hypernix`](https://pypi.org/project/hypernix/) (`hypernix.pressure_cooker_v5`, `hypernix.pressure_cooker_v5s`)
**Scope of this paper:** `PressureCookerV5`, `PressureCookerV5Plus`, `PressureCookerV5S`, and their Pascal-safe variants (`Agedcookerv5`, `ULTRAagedcookerv5`, `Agedcookerv5s`).

> **On accuracy.** Every number in this paper was either (a) measured by directly instantiating the real optimizer classes and reading their actual state tensors / wall-clock timings, with the exact reproduction commands given in §8, or (b) derived algebraically from the update rule as implemented in `pressure_cooker_v5.py` / `pressure_cooker_v5s.py` at the time of writing. Nothing here is an estimate carried over from a docstring or a marketing claim. Where a measurement is unfavorable to V5/V5S (§5), it is reported anyway. Where the source code's own informal comments turned out to disagree with measurement, that disagreement is called out explicitly (§4.3) rather than silently resolved in either direction.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Related work](#2-related-work)
3. [Architecture](#3-architecture)
4. [Memory efficiency](#4-memory-efficiency)
5. [Compute efficiency (step time)](#5-compute-efficiency-step-time)
6. [Use cases](#6-use-cases)
7. [Verification](#7-verification)
8. [Reproducibility](#8-reproducibility)
9. [Limitations and threats to validity](#9-limitations-and-threats-to-validity)
10. [Acknowledgments](#10-acknowledgments)
11. [References](#11-references)

---

## 1. Introduction

Modern first-order optimizers for deep learning overwhelmingly follow the AdamW template: two full-precision buffers per parameter (a first moment and a second moment), each the same shape and dtype as the parameter itself. For a model trained in fp32, that means every parameter tensor carries **2x its own size again** in optimizer state, on top of the parameter and its gradient — a 3x-4x memory multiplier over the raw weights before activations even enter the picture. On memory-constrained hardware (this package explicitly targets Pascal-generation 8 GB cards such as the GTX 1080 alongside modern GPUs), that multiplier is often the binding constraint on model size or batch size, not compute.

`PressureCookerV5` and `PressureCookerV5S` are two related but independently-implemented optimizers in the `hypernix` package that attack this from the state-representation side: instead of a full-precision second moment, they combine a **quantized momentum buffer**, a **factored (row/column) curvature estimate** in place of an elementwise one, and **sub-byte-per-coordinate bookkeeping** (uint8 age counters) for adaptive coordinate freezing. Both are accompanied by an oscillation-resistance mechanism — cosine similarity between the gradient and momentum direction — that adaptively damps the learning rate and the power-law exponent used in the update itself, in place of AdamW's per-coordinate RMS normalization.

This paper documents, with measured numbers: what the two optimizers actually do (§3), how much memory they actually save and why (§4), how they actually perform on wall-clock step time — including where that performance is currently *worse* than AdamW (§5) — what they're for (§6), how they're tested (§7), and exactly how to reproduce every figure in this document (§8).

## 2. Related work

Both optimizers borrow individual *ideas* from prior work but combine them into an update rule that is not a re-parameterization of any single one of these:

- **AdamW** [Loshchilov & Hutter, 2019] — decoupled weight decay is used identically here, but ORCP has no exponential moving average of squared gradients and no bias-correction term.
- **Adafactor** [Shazeer & Stern, 2018] — the row/column-factored curvature estimate for matrix parameters is structurally the same idea (an outer-product approximation to a full second moment), reducing an `O(rows*cols)` state to `O(rows+cols)`.
- **LARS / LAMB** [You et al., 2017; 2019] — the per-tensor trust ratio (`||param|| / ||update||`, clamped) that scales the effective learning rate is the same layerwise-adaptive-rate idea.
- **Sharpness-Aware Minimization (SAM)** [Foret et al., 2020] — `PressureCookerV5` implements the ascend-then-correct SAM step (`sam_rho > 0`) exactly as in the original paper; `PressureCookerV5S` does not yet implement SAM (its own docstring says so explicitly — see §6).
- **Sophia** [Liu et al., 2023] — the hard clip on the curvature-normalized update (`sophia_clip`) is the same defensive clipping idea Sophia uses against noisy/negative curvature estimates.
- **Lion** [Chen et al., 2023] and sign-based updates — the `sign(g) * |g|^power` update generalizes Lion's pure `sign(g)` update (Lion is the `power -> 0` limit); ORCP's power is adaptive rather than fixed.

The novel combination in both optimizers is: (1) driving the adaptive power exponent and the learning-rate damping from a **cosine-similarity oscillation signal** rather than a gradient-magnitude signal, and (2) quantizing momentum to int8 while keeping curvature factored, so the *entire* optimizer state (not just one buffer) stays small. V5S additionally introduces (3) a three-timescale ("3D") version of the oscillation signal and (4) a gradient-domain spatial smoothing pass ("pressure diffusion") that has no analogue in any of the above.

## 3. Architecture

### 3.1 PressureCookerV5 — Oscillation Resistant Cosine Power (ORCP)

Given a parameter `p` with gradient `g` and dequantized momentum `m` (from the int8 buffer, see §4.1), `PressureCookerV5._orcp_step` computes, per parameter tensor, per step:

```
cos_sim        = (g . m) / (||g|| * ||m||)
slow_cos      <- slow_beta * slow_cos + (1 - slow_beta) * cos_sim          [beta = 0.98]
osc_score      = clip( -(cos_sim + slow_cos) / 2,  -1, 1 )
damp           = 1 / (1 + max(osc_score, 0) * 2)

power_target   = power_max - (power_max - power_min) * max(osc_score, 0)
power         <- 0.9 * power + 0.1 * power_target                          [adaptive exponent, EMA-smoothed]

g_pred         = g + extrapolation * m                                     [one-step look-ahead, default alpha=0.15]
m             <- momentum_beta * m + (1 - momentum_beta) * g               [re-quantized to int8]

curv           = factored_or_elementwise_curvature(g_pred)                 [see 4.1]
update         = sign(g_pred) * |g_pred|^power / sqrt(curv)
update         = clip(update, -sophia_clip, sophia_clip)                   [default 1.0]
update        *= freeze_scale(age, freeze_threshold, freeze_patience, freeze_decay)

trust          = clip(||p|| / ||update||, trust_clip_lo, trust_clip_hi)    [default (0.05, 5.0)]
lr_eff         = lr * damp * trust

p             *= (1 - lr_eff * weight_decay)                               [decoupled weight decay]
p             -= update * lr_eff
```

Negative `cos_sim` means the gradient just reversed direction relative to momentum -- classic oscillation. `osc_score` turns that into a `[-1, 1]` signal that (a) shrinks the effective learning rate (`damp`) and (b) pulls the power exponent toward `power_min` (more conservative, closer to a pure sign-update) while training is unstable, and lets it drift back toward `power_max` (more magnitude-sensitive) once the gradient direction stabilizes. This is the "Oscillation Resistant" and "Cosine Power" halves of the name.

`PressureCookerV5Plus` (`V5+`) adds, on top of the above: a second, ultra-slow cosine EMA (`beta ~= 0.995`) for long-horizon drift detection, a resonance-flip counter that detects repeated sign reversals, entropy-based gradient-scale adjustment (`_extra_scale`), directional (row-wise) trust regions for matrix parameters, a "recovery ramp" that un-freezes coordinates gradually rather than instantly, and QAT auto-`prepare_model()`. It also defaults `ema_decay=0.999` (V5's own default is `0.0`, i.e. off) — see §4.2 for what that costs in memory.

### 3.2 PressureCookerV5S — 3D-ORCP + Pressure Diffusion

`PressureCookerV5S` is **not** a subclass of V5 — it is a separate class (`hypernix.pressure_cooker_v5s.PressureCookerV5S`) that shares only the quantized-momentum helper functions and the optional QAT/MTP infrastructure. Its update rule, per `_v5s_step_one`:

```
gd             = pressure_diffuse(g, diffusion_factor, kernel_width=3, mode="gauss")   [see 3.2.1]

fast_cos      <- fast_beta  * fast_cos  + (1 - fast_beta)  * raw_cos(gd, m)   [beta ~= 0.80,  reacts  ~5 steps]
med_cos       <- med_beta   * med_cos   + (1 - med_beta)   * raw_cos(gd, m)   [beta ~= 0.95,  reacts ~20 steps]
ultra_cos     <- ultra_beta * ultra_cos + (1 - ultra_beta) * raw_cos(gd, m)   [beta ~= 0.999, reacts ~1000 steps]

VOS            = clip( -(0.45*fast_cos + 0.35*med_cos + 0.20*ultra_cos), -1, 1 )   [volumetric oscillation score]
damp           = 1 / (1 + max(VOS, 0) * vos_3d_gain)                                [default gain = 3.0]
power_target   = power_max - (power_max - power_min) * max(VOS, 0)
power         <- 0.9 * power + 0.1 * power_target

g_pred         = gd + extrapolation * m
curv           = factored_or_elementwise_curvature(g_pred)                          [identical structure to V5]
update         = sign(g_pred) * |g_pred|^power / sqrt(curv)
update         = clip(update, -sophia_clip, sophia_clip)
update        *= freeze_scale(age, ...) * extra_scale(...)                          [extra_scale = 1.0 in base V5S]

trust          = clip(||p|| / ||update||, trust_clip_lo, trust_clip_hi)
lr_eff         = lr * damp * trust
p             *= (1 - lr_eff * weight_decay)
p             -= update * lr_eff
m             <- momentum_beta * m + (1 - momentum_beta) * gd                       [note: diffused gradient, re-quantized]
```

The structural skeleton (power update, Sophia clip, freeze scale, LARS-style trust ratio, decoupled weight decay) is the same shape as V5's, which is expected since both are exploring the same underlying "quantized-state, cosine-damped, power-law update" design space — but every signal V5S computes it from (VOS instead of a single `osc_score`, `gd` instead of raw `g`) is genuinely different, which is why it is implemented as an independent class rather than a V5 subclass.

#### 3.2.1 Pressure Diffusion

```
pressure_diffuse(g, factor, kernel_width=3, mode="gauss"):
    flat        = g.flatten()
    kernel      = normalized_1d_kernel(kernel_width, mode)     # sums to 1
    diffused    = conv1d(flat, kernel, padding=kernel_width//2)
    return ((1 - factor) * flat + factor * diffused).reshape(g.shape)
```

This is a 1-D convolution over the *flattened* gradient tensor — i.e., it smooths neighboring coordinates in memory layout order, not in any semantic/spatial sense (there is no assumption that adjacent flattened elements are spatially adjacent in the original tensor, e.g. a conv weight's `[out_ch, in_ch, kh, kw]` layout). It costs `O(n)` time and adds **no persistent per-parameter state** — the diffusion kernel is built once per call and discarded. Default `diffusion_factor=0.12` means the diffused gradient is a 88%/12% blend of the original and the locally-smoothed version.

### 3.3 Pascal-safe variants

`Agedcookerv5`, `ULTRAagedcookerv5` (wrapping V5 / V5Plus), and `Agedcookerv5s` (wrapping V5S) force `fused=False`, warn if run on non-Pascal hardware, and — for the V5S variant specifically — cap the diffusion kernel width at 3 and store curvature buffers in fp16 rather than fp32 (halving that already-small component of the state further). These exist because CUDA compute capability 6.1/6.2 (Pascal, e.g. GTX 10-series) lacks native bf16 and has weak fp16 tensor-core support relative to Ampere+, so the fused/tensor-core code paths some optimizers take are actively counterproductive there; see `hypernix.freezer.pascal_mode_hints()` for the corresponding VRAM-side guidance.

## 4. Memory efficiency

### 4.1 What actually gets allocated

Per parameter tensor `p` with `N` elements (`R` rows x `C` cols for a matrix), `_init_state` allocates:

| Buffer | V5 / V5S (matrix param) | V5 / V5S (vector param, `N <= 65536`) | AdamW |
|---|---|---|---|
| Momentum | `N` bytes (int8) + 1 fp32 scalar scale | `N` bytes (int8) + 1 fp32 scalar scale | `4N` bytes (`exp_avg`, fp32) |
| 2nd-moment / curvature | `4(R+C)` bytes (`row_curv` + `col_curv`, fp32) | `4N` bytes (elementwise, fp32) | `4N` bytes (`exp_avg_sq`, fp32) |
| Freeze / age counters | `R` bytes (uint8, `row_age`) | `N` bytes (uint8, `age`) | none |
| EMA (if `ema_decay > 0`) | `4N` bytes (fp32 clone of `p`) | `4N` bytes | none (not a feature) |

For a matrix parameter, as `N = R*C` grows large relative to `R+C` (true for any non-degenerate weight matrix — e.g. a 2048x2048 linear layer has `N=4.19M` vs `R+C=4096`), the row/column curvature and age terms become negligible and the state converges to **`~N` bytes total**, i.e. **1 byte per parameter** versus AdamW's **8 bytes per parameter** (`exp_avg` + `exp_avg_sq`, both fp32) — an asymptotic **8x** reduction in optimizer-state memory, or equivalently the state shrinks from `2.0x` the parameter tensor's own footprint (AdamW) to `~0.25x` (V5/V5S).

### 4.2 Measured numbers

Measured directly by instantiating each real optimizer, running one backward + `step()` (so all lazily-allocated state actually materializes), and summing `tensor.nelement() * tensor.element_size()` over every tensor in `optimizer.state[param]`. This does **not** depend on the device — an int8 buffer is the same number of bytes on CPU or CUDA — so it is a faithful, hardware-independent measurement of the thing the source docstrings make informal claims about. Script: `scripts/measure_optimizer_memory.py`.

Two-layer `Linear(H,H) -> ReLU -> Linear(H,H)` MLP, fp32, batch 32:

| Optimizer | H=1024 state | H=2048 state | H=4096 state | state / param (H=4096) | state vs. AdamW (H=4096) |
|---|---|---|---|---|---|
| AdamW | 16.02 MB | 64.03 MB | 256.06 MB | 2.000x | 1.000x (baseline) |
| **PressureCookerV5** | 2.03 MB | 8.06 MB | 32.12 MB | 0.251x | **0.125x** |
| **PressureCookerV5S** | 2.03 MB | 8.06 MB | 32.12 MB | 0.251x | **0.125x** |
| PressureCookerV5Plus (default `ema_decay=0.999`) | -- | 48.05 MB (H=2048) | -- | 1.501x | 0.750x |

Total *training* footprint (parameters + optimizer state, relative to parameter size alone): **AdamW = 3.00x**, **V5 / V5S = 1.25x**, **V5Plus (EMA on by default) = 2.50x**. All three match the `4.1` derivation almost exactly (the tiny gap between `0.251x` and the `0.25x` asymptote is exactly the `O(R+C)` row/column-curvature term predicted above, and it shrinks as `H` grows: `0.253x` at H=1024 -> `0.252x` at H=2048 -> `0.251x` at H=4096).

**Independent cross-validation.** The repository already contained a prior benchmark (`02_optimizer_speed_and_memory.pdf`, generated on a 4-layer/256-hidden model, batch 8x128) that measured optimizer state the same way (directly from `.state` tensors after one step) on what its companion chart (`03_gpu_utilization_and_vram.pdf`) identifies as an actual NVIDIA GeForce GTX 1080. That measurement: **PressureCookerV5 = 6.11 MB, AdamW = 47.64 MB** -- a ratio of **12.8%**, matching this paper's independently-derived **12.5%** to within a fraction of a percentage point, on a completely different model shape, run at a different time, on real GPU hardware rather than this paper's CPU sandbox. (`PressureCookerV3`, included in the same chart, measured at 47.64 MB — the same as AdamW; V3 predates the quantized-state design introduced in V5 and is out of scope for this paper.)

### 4.3 A correction to the source's own documentation

Prior to this paper, `wiki/Pressure-Cooker-V5.md` and this package's `README.md` described the momentum buffer as **"6-bit, using stochastic rounding."** Neither claim matches `_quantize_momentum()`:

```python
def _quantize_momentum(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = m.abs().amax().clamp(min=1e-12)
    q = (m / scale * _INT8_MAX).round().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale
```

This is **int8** (8-bit, `torch.int8`, range `[-127, 127]`) with **standard round-to-nearest** (`.round()`), not stochastic rounding. The "~75% memory reduction" figure that accompanied the old claim happens to still be numerically correct (1 byte vs. 4 bytes is a 75% reduction), which is presumably why the bit-width error went unnoticed, but the specific mechanism described was wrong. Both documents have been corrected as part of this change (see the README's "What's fixed in this update" section).

## 5. Compute efficiency (step time)

This is the section where the honest result is *not* flattering, and it is reported in full rather than omitted.

Measured with `scripts/benchmark_v5.py` / `scripts/benchmark_v5s.py`: each optimizer trains its own freshly-initialized, identically-seeded model copy (not a model already mutated by a previous optimizer in the loop -- the original versions of these two scripts had that bug; see the README changelog), for 100 steps, wall-clock timed with `time.perf_counter()`.

**This sandbox has no GPU** (`torch.cuda.is_available() == False`), so all first-party numbers below are **CPU-only** (Intel Xeon @ 2.80GHz, PyTorch 2.13.0+cu130, Python 3.12.3) and are reported as such — not extrapolated to GPU behavior:

| Optimizer | Shape | Mean step time (CPU) | Relative to AdamW |
|---|---|---|---|
| AdamW | `Linear(2048,2048)` x2, batch 128 | ~180 ms | 1.0x |
| PressureCookerV5 | same | ~500-630 ms | **2.8x-3.5x slower** |
| AdamW | `Linear(1024,1024)` x2, batch 128 | ~46-51 ms | 1.0x |
| PressureCookerV5 | same | ~136-147 ms | **2.7x-3.0x slower** |
| PressureCookerV5S | same | ~249-278 ms | **5.0x-5.9x slower** |

**Independent cross-validation (real GTX 1080, 4L/256H model, batch 8x128, from `02_optimizer_speed_and_memory.pdf`):** PressureCookerV5 = 275.6 ms/step vs. AdamW = 211.3 ms/step (**1.30x slower**) — same direction as the CPU measurement above, smaller magnitude. A separate 60-step end-to-end training-time comparison in the repository (`01_ram_and_training_time.pdf`) shows the same pattern: PressureCookerV5 = 16.51 s vs. AdamW = 12.82 s over 60 steps (**1.29x slower**).

**Why.** Every one of the extra ideas in §3 — the cosine-similarity computation(s), the adaptive power EMA, the factored curvature update, the freeze-scale age-counter bookkeeping, the trust-ratio norm computations, and (for V5S specifically) the `conv1d`-based pressure diffusion and *three* cosine EMAs instead of one — is an additional small tensor operation launched per parameter tensor, per step. On CPU, and to a lesser extent on older/smaller GPUs where kernel-launch overhead is a larger fraction of total step time, this Python-and-kernel-dispatch overhead dominates and is not hidden by the memory savings. V5S measuring consistently slower than V5 in this paper's own benchmarks (roughly 1.8x-1.9x V5's time) is consistent with it performing three EMA updates and one convolution instead of V5's one EMA update and zero convolutions.

**The honest summary:** PressureCookerV5/V5S trade wall-clock step time for a large, real, independently-corroborated reduction in optimizer-state memory (§4). Whether that trade is worth it depends entirely on which resource is the actual bottleneck for a given run — on an 8 GB Pascal card where optimizer state genuinely limits batch size or model size, freeing up ~87% of that specific memory line-item can let a run happen at all; on a GPU where VRAM is not the constraint, the step-time cost is a real, measured downside with no offsetting benefit shown here.

## 6. Use cases

Grounded in the classes' own constructors and docstrings, not aspirational:

- **General pretraining, memory-constrained hardware.** Defaults (`lr=3e-4`) on either V5 or V5S. This is the primary intended use case given §4-5: favorable when optimizer-state VRAM is the binding constraint.
- **Fine-tuning.** `PressureCookerV5(..., finetune_mode=True)` (V5Plus) sets `power in [0.25, 0.8]` narrower, tighter `trust_clip`, longer `freeze_patience=64`, and disables SAM. `PressureCookerV5S(..., lr=1e-4, diffusion_factor=0.05, vos_3d_gain=4.0, freeze_patience=64)` is the documented equivalent for V5S.
- **Low-memory edge training.** `diffusion_factor=0.0, ema_decay=0.0` (V5S) minimizes both compute and state further by disabling the two optional-cost features.
- **Pascal / GTX 10-series GPUs.** `Agedcookerv5`, `ULTRAagedcookerv5`, `Agedcookerv5s` — force `fused=False`, cap the diffusion kernel, and (for the V5S variant) store curvature in fp16. Combine with `hypernix.freezer.OldFreezer` / `auto_freezer()` and `pascal_safe_dtype()` (which selects fp16, since Pascal has no native bf16).
- **Quantization-Aware Training.** `qat_config=QATConfig(bits=4|5|6|8, ...)` plus `cooker.attach_qat(model)` (V5) or `cooker.prepare_model(model)` (V5Plus, which also applies mixed-precision skip-listing to `nn.Embedding`/`nn.LayerNorm`). V5S accepts the same `qat_config` but has no `attach_qat` convenience method of its own in the current source — it must be attached manually via the shared `QATFakeQuantize` machinery.
- **Multi-Token Prediction training.** `enable_mtp=True, mtp_config=MTPConfig(num_tokens=4, ...)`, then `cooker.get_mtp_head(hidden_dim, vocab_size)` to obtain the prediction head. Available on both V5 and V5S.
- **Flat-minima seeking via SAM.** `PressureCookerV5(..., sam_rho=0.05)` and pass a `closure` to `step()`. **Not currently available on V5S** — its `step()` docstring states this explicitly ("not yet supported in V5S; use PressureCookerV5 for SAM"), and this paper repeats that constraint rather than eliding it.
- **Live diagnostics.** V5S exposes `get_oscillation_stats()` (mean fast/med/ultra cosine + VOS across all parameters), `get_frozen_fraction()` (fraction of coordinates currently soft-frozen), and `print_summary()` for monitoring training stability without external tooling.

## 7. Verification

Test suite state at the time of writing (full repository, `pytest`): **1475 passed, 1 skipped** (a CUDA-only test, expected to skip on a CPU-only machine), **0 failed**, against this paper's own bug fixes applied (§9 of the accompanying README changelog). `tests/optimizers/test_v0705_all.py` and `tests/optimizers/test_v051_1.py` -- the two files most directly exercising the V5/V5S/V5Plus classes -- pass **77/77**. `ruff check src tests` also passes cleanly: 64 pre-existing lint findings elsewhere in the codebase (import sorting, and `isinstance(x, (A, B))` -> PEP 604 `isinstance(x, A | B)`) were fixed alongside the changes described here -- mechanical, behavior-preserving fixes, re-verified against the full test suite after applying them, and unrelated to `pressure_cooker_v5`/`v5s` specifically.

## 8. Reproducibility

Every number in §4-5 can be regenerated directly:

```bash
pip install -e .            # from the repository root
pip install torch            # any recent version; CPU-only is sufficient for §4, §5's CPU rows

python scripts/measure_optimizer_memory.py   # -> Table in §4.2 (state bytes per parameter tensor)
python scripts/benchmark_v5.py               # -> Table in §5, AdamW vs. V5, H=2048
python scripts/benchmark_v5s.py              # -> Table in §5, AdamW vs. V5 vs. V5S, H=1024

pytest tests/ -q                              # -> §7 (full suite)
pytest tests/optimizers/test_v0705_all.py tests/optimizers/test_v051_1.py -q   # -> §7 (V5-specific subset)
```

Environment this paper's own numbers were measured in: Python 3.12.3, PyTorch 2.13.0+cu130, `hypernix` 0.71.3, Linux (CPU-only sandbox, Intel Xeon @ 2.80GHz). GPU-specific numbers are cited from the repository's own pre-existing `01_ram_and_training_time.pdf` / `02_optimizer_speed_and_memory.pdf` (generated on an NVIDIA GeForce GTX 1080 per `03_gpu_utilization_and_vram.pdf`) rather than re-measured, since no GPU was available while writing this paper — anyone with CUDA hardware, including a Pascal card, can re-run the three scripts above directly to get device-specific numbers.

## 9. Limitations and threats to validity

- **No GPU was available while writing this paper.** All first-party timing/memory numbers are CPU-measured. The GPU cross-validation in §4.2/§5 comes from pre-existing repository artifacts generated at an earlier, unknown point in time, on a different (4L/256H) model shape, and this paper cannot independently re-verify that those PDFs' generation process was itself bug-free.
- **Micro-benchmark, not end-to-end training.** §5's timings are per-step wall clock on synthetic random data with a trivial `Linear-ReLU-Linear` model and an `out.pow(2).mean()` loss — not a real training run with a tokenizer, data loading, or a realistic transformer block. Real-workload step time will differ (likely narrowing the relative gap somewhat, since a larger fraction of real step time is spent in forward/backward compute that all three optimizers share equally).
- **CPU step-time results carry inherent run-to-run variance** from OS scheduling noise on a shared sandbox; the ranges reported in §5 reflect two separate runs each, not statistically rigorous multi-trial confidence intervals.
- **PressureCookerV5Plus and V5S's QAT/MTP paths are not benchmarked here** — §4-5 cover the default (QAT/MTP-disabled) configuration only, since that is what the memory/speed claims in the source docstrings were about.
- **This paper does not evaluate model-quality outcomes** (final loss, downstream task accuracy, convergence speed in *steps* rather than wall-clock) for V5/V5S versus AdamW. It is scoped to computational-efficiency claims only, per the request that produced it.

## 10. Acknowledgments

Coding and writing assistance for the `hypernix` `pressure_cooker_v5`/`v5s` codebase and for this paper was provided, across development, by: Claude Sonnet 5, Claude Sonnet 4.6, Claude Opus 4.7, Google Gemini 3.5 Flash, Google Gemini 3.6 Flash, Google Gemini 3.1 Pro, and Qwen 3 Coder. This specific paper draft, the accompanying bug fixes (§4.3, and the `hyped` TUI / benchmark-script fixes noted in the README), and the measurement scripts in §8 were produced with Claude Sonnet 5.

## 11. References

- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
- Shazeer, N. & Stern, M. (2018). *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* ICML.
- You, Y. et al. (2017). *Large Batch Training of Convolutional Networks* (LARS). arXiv:1708.03888.
- You, Y. et al. (2019). *Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes* (LAMB). ICLR.
- Foret, P. et al. (2020). *Sharpness-Aware Minimization for Efficiently Improving Generalization* (SAM). arXiv:2010.01412.
- Liu, H. et al. (2023). *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training.* arXiv:2305.14342.
- Chen, X. et al. (2023). *Symbolic Discovery of Optimization Algorithms* (Lion). NeurIPS.
