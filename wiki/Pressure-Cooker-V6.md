# Pressure Cooker V6 / V6V

## Overview

Where every prior generation (V3 → V4 → V5 → V5S) kept adding machinery —
factored curvature, quantized momentum, oscillation tracking at multiple
timescales, QAT/MTP hooks — V6 goes the other way. It's **speed-first**,
not memory-first: one momentum buffer, fused multi-tensor
(`torch._foreach_*`) updates, and at most one host↔device synchronization
per step instead of several per parameter tensor. `PressureCookerV6V` is
the same optimizer wired for CUDA graph capture and (optional)
`torch.compile`.

This page documents what's actually implemented in
[`pressure_cooker_v6.py`](../src/hypernix/optimizers/pressure_cooker_v6.py) and
[`pressure_cooker_v6v.py`](../src/hypernix/optimizers/pressure_cooker_v6v.py) — see
those module docstrings for the full derivation.

## Why V6 is faster than AdamW (and V5) — the actual mechanism

This is not a smarter update rule than AdamW's — it's fewer operations and
far fewer stalls:

* **One buffer, not two.** AdamW tracks `exp_avg` *and* `exp_avg_sq` and
  does a bias-corrected `sqrt` + division per element. V6 tracks a single
  momentum buffer and does one fused multiply-add.
* **Batched host syncs.** V5's step calls Python `float()` on 3-4 device
  tensors *per parameter, per step* (cosine similarity, oscillation score,
  trust ratio) — each one a CPU↔GPU synchronization barrier. With hundreds
  of parameter tensors in a real model, that's hundreds of stalls every
  step. V6's only optional per-tensor scalar (the trust ratio) is batched
  into a **single** `.tolist()` call across the whole parameter group —
  one sync per step, not one-or-more per tensor.
- **Real fused kernels, not a hand-rolled substitute.** The momentum
  update, weight decay, and trust-ratio norms all go through
  `torch._foreach_*`, the same multi-tensor-apply CUDA kernels PyTorch's
  own `foreach=True`/`fused=True` optimizers use.

The honest tradeoff: V6 has no per-element adaptive learning rate (no
second moment, no curvature estimate), so it typically needs more
learning-rate tuning than AdamW for a new model — closer in lineage to
LARS/LAMB-style per-tensor trust-ratio scaling on top of plain momentum
than to AdamW. It optimizes for step time and a modest, honest memory
saving, not the aggressive quantized/factored savings V5/V5S go after.

## Measured numbers

Real, measured output — not estimates — from the scripts in this repo.
Re-run them yourself; numbers vary by hardware and both scripts print
the device they ran on:

**Memory** (`scripts/measure_optimizer_memory.py`, exact
`nelement() * element_size()` byte counts of everything actually stored in
`optimizer.state`, CPU, device-independent since state bytes don't depend
on device):

| Optimizer | hidden=2048 state | vs AdamW |
|---|---|---|
| AdamW | 64.03 MB | 1.00x |
| PressureCookerV6 | 32.02 MB | **0.50x** — one fp32 buffer instead of two |
| PressureCookerV5 | 8.06 MB | 0.13x (quantized + factored — a different, more aggressive tradeoff) |

**Speed** (`scripts/benchmark_v6.py`, 3-layer `hidden=2048` MLP,
batch=128, 100 steps, **CPU** — this sandbox has no CUDA device):

| Optimizer | ms/step | vs AdamW |
|---|---|---|
| AdamW | 255.0 ms | 1.00x |
| PressureCookerV6 | 160.2 ms | **1.59x faster** |
| PressureCookerV5 | 780.0 ms | 0.33x (3.0x slower) |

**These are CPU numbers and are reported as such.** V6's whole design bet
— fewer host↔device syncs, fused multi-tensor kernels replacing many
small per-tensor kernel launches — targets overhead that's far more
pronounced on GPU than CPU. Don't read the CPU ratio above as a GPU
prediction; re-run `scripts/benchmark_v6.py` on CUDA hardware for numbers
that actually mean something for real training. The script auto-detects
CUDA and benchmarks `PressureCookerV6V` alongside V6 when one's available.

## Basic usage

```python
from hypernix.pressure_cooker_v6 import PressureCookerV6

opt = PressureCookerV6(
    model.parameters(),
    lr=1e-3,               # V6 has no per-element adaptive LR -- expect to
                            # tune this more than you would for AdamW
    momentum_beta=0.9,
    weight_decay=0.01,
    trust_ratio=True,      # LARS/LAMB-style per-tensor scaling; the only
                            # source of per-tensor (not per-element) adaptivity
    trust_clip=(0.05, 5.0),
    nesterov=False,
)
```

Disable `trust_ratio` and/or `foreach` for the absolute minimum op count
(useful for debugging or a torch build without `torch._foreach_norm` —
V6 warns and falls back to a plain per-tensor loop automatically, it
doesn't error).

## Production features, not just the update rule

V6 carries the same real-training features
`hypernix.pressure_cooker.InductionCooker` has, rather than assuming a
toy training loop with a single dense forward/backward every call:

```python
opt = PressureCookerV6(
    model.parameters(),
    lr=1e-3,
    grad_accum_steps=4,        # only every 4th step() call actually updates
    grad_scaler=scaler,        # torch.cuda.amp.GradScaler -- unscales,
                                # skips cleanly on non-finite instead of
                                # corrupting momentum state
    skip_on_nonfinite=True,    # same non-finite skip, opt-in, without a
                                # GradScaler (off by default -- see below)
)
```

- **`grad_accum_steps`**: identical contract to `InductionCooker` — your
  training loop still calls `loss.backward()` every micro-batch as usual;
  this only gates *when the optimizer applies the accumulated gradient*.
- **`grad_scaler`**: same `unscale_`/`update()` contract as
  `InductionCooker`, but the non-finite check is batched into a single
  `torch._foreach_norm` + one host sync for the whole step instead of a
  per-parameter `.all()` sync — consistent with the rest of V6's "batch
  every host sync" design, this is the one place V6 does *better* than
  the pattern it's borrowing.
- **`skip_on_nonfinite`**: the same batched check without a `GradScaler`
  attached. **Off by default** — this keeps the default configuration's
  op count, and the measured numbers in the table above, exactly as
  documented rather than paying for an always-on safety check most
  stable runs will never trip. Turn it on for extra robustness on
  unstable runs.

Tested on more than a toy MLP: `tests/test_pressure_cooker_v6.py`
includes a small but structurally real transformer block (token
embedding + multi-head attention + LayerNorm + GELU MLP + output head —
1-D, 2-D, and embedding-table parameters all in the same optimizer) to
confirm V6 handles realistic parameter heterogeneity, not just uniform
`nn.Linear` stacks.

## `PressureCookerV6V` — CUDA graphs + `torch.compile`

`PressureCookerV6V` is `PressureCookerV6` with two additive, CUDA-only
mechanisms layered on top. It does **not** change the update rule above —
it changes how the update gets onto the GPU.

```python
from hypernix.pressure_cooker_v6v import PressureCookerV6V

model = model.cuda()
opt = PressureCookerV6V(model.parameters(), lr=1e-3, compile=True)

# CUDA graph capture -- identical API/contract to
# hypernix.pressure_cooker.ProCooker.warmup_graph/replay_graph:
def step_fn():
    opt.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    opt.step()
    return loss

opt.warmup_graph(step_fn)   # call once, on a representative fixed-shape batch
for _ in range(num_steps):
    loss = opt.replay_graph()
```

* **`compile=True` (default):** wraps the internal fused step with
  `torch.compile` so TorchInductor/Triton can fuse the multi-tensor ops
  into fewer kernels. If `torch.compile` is unavailable (torch < 2.0) or
  compilation fails for any reason, V6V warns once and falls back to the
  eager V6 path — it never raises for this.
- **CUDA graph capture** (`warmup_graph`/`replay_graph`) eliminates
  per-launch CPU overhead entirely by recording the whole step's kernel
  sequence once and replaying it. Same shape/no-dynamic-control-flow
  restriction as `ProCooker`.
- Requires at least one CUDA parameter at construction; raises
  `RuntimeError` otherwise (same pattern as torch's own `fused=True`
  requiring sm_70+). Use plain `PressureCookerV6` for CPU/portable
  training.

**CUDA graph capture + grad_accum_steps/grad_scaler/skip_on_nonfinite
don't combine safely — read this before using both.** Graph capture
bakes in whichever Python-level branch ran during `warmup_graph`,
permanently: replay always re-executes the same recorded kernels no
matter what later batches look like. If those features are enabled, the
accumulation-gate / non-finite-skip decision only gets evaluated once,
at capture time, and every `replay_graph()` call afterwards repeats
that same decision regardless of the actual data. `warmup_graph` detects
this configuration and warns; it still isn't safe to rely on dynamic
skipping under CUDA graph replay. Either don't graph-capture when using
those features, or capture only once you're confident the branch taken
at capture time (normally "batch is finite, apply the update") is safe
to repeat unconditionally.

**Verification note, stated plainly:** this package's test suite
(`tests/test_pressure_cooker_v6.py`) exercises everything that doesn't
strictly require CUDA — the constructor's device-check guard, the
inherited SSTM update rule (including that the fused and non-fused code
paths produce identical numbers), and the `torch.compile`
wrap-and-fallback logic. The CUDA-only paths — construction on real CUDA
tensors, graph capture/replay, compiled execution actually running on a
GPU — are marked `skipif(not torch.cuda.is_available())` and were not run
against real CUDA hardware while writing this, since the authoring
environment didn't have a GPU available. The graph-capture code reuses
`ProCooker`'s already-shipped implementation verbatim (same method names,
same contract) rather than inventing a new one, which is the best
available substitute for that verification — but it's not a replacement
for it. If you hit an issue running V6V on real CUDA hardware, please
file it.

## No Pascal-specific tier

Unlike V3/V4/V5's `Aged*`/`ULTRAaged*` classes, there's no
`Agedcookerv6`. Those exist to work around torch's `fused=True` AdamW
kernel requiring CUDA sm_70+; V6 never calls that kernel — `torch._foreach_*`
ops run fine on Pascal — so there's nothing to work around.

## See also

- [Pressure Cooker V5 / V5+ / V5S](Pressure-Cooker-V5.md) — the
  memory-first generation V6 is contrasted with above
- [Optimizers](Optimizers.md) — `pressure_cooker`, `optimizer_framework`,
  and the V3/V4 generations
- `hypernix.pressure_cooker.ProCooker` — the existing CUDA-graph
  implementation V6V's `warmup_graph`/`replay_graph` follows
