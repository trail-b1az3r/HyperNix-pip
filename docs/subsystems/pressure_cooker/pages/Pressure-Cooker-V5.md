# Pressure Cooker V5 / V5+ / V5S

## Overview

Pressure Cooker V5 is HyperNix's flagship optimizer, abandoning traditional AdamW mechanics entirely in favor of an **Oscillation Resistant Cosine Power (ORCP)** architecture, combining quantized momentum, Quantization-Aware Training (QAT), and Multi-Token Prediction (MTP) support. V5+ (`PressureCookerV5Plus`) extends V5 with automatic model transformation, quantization sensitivity analysis, and finer directional trust regions. V5S (`PressureCookerV5S`) is a separate, ground-up 3D-ORCP optimizer: it tracks *three* cosine-similarity EMAs at different time horizons instead of one/two, adds a lightweight "pressure diffusion" spatial smoothing pass over the gradient, and keeps memory below AdamW's by never allocating a full second-moment tensor. See the [V5/V5S efficiency paper](../pressure_cooker_v5_v5s_paper.md) for the full derivation and measured numbers.

## Features

### Quantized Momentum

Momentum buffers are quantized to int8 (8-bit signed integers, `[-127, 127]`) with a per-tensor fp32 scale, using standard round-to-nearest (`torch.round`) -- not stochastic rounding. This is always on; there is no `quantize_momentum` toggle. It reduces the momentum buffer's own memory by 75% versus keeping it in fp32 (1 byte/element vs. 4), and -- combined with factored row/column curvature instead of a full elementwise second moment -- measures at roughly 12-13% of AdamW's total optimizer-state footprint in practice (see the [efficiency paper](../pressure_cooker_v5_v5s_paper.md) for the exact measured numbers and derivation).

```python
from hypernix.pressure_cooker_v5 import PressureCookerV5

cooker = PressureCookerV5(
    model.parameters(),
    lr=2e-4,
)
```

### Quantization-Aware Training (QAT)

QAT simulates low-precision quantization during training so models learn to be robust to quantization error. Supports Q4, Q5, Q6, and Q8 bit widths.

```python
from hypernix.pressure_cooker_v5 import PressureCookerV5, QATConfig

# Basic QAT
qat_cfg = QATConfig(bits=6, per_layer=True)
cooker = PressureCookerV5(
    model.parameters(),
    qat_config=qat_cfg,
)
cooker.attach_qat(model)  # Attach fake quantization hooks

# Advanced QAT with learnable scales
qat_cfg = QATConfig(
    bits=4,
    per_layer=True,
    learnable_scales=True,
    dynamic_range=True,
    mixed_precision=True,  # Keep sensitive layers in fp16
)
```

### Multi-Token Prediction (MTP)

MTP trains models to predict multiple future tokens simultaneously, improving training efficiency by 1.5-3x.

```python
from hypernix.pressure_cooker_v5 import PressureCookerV5, MTPConfig

mtp_cfg = MTPConfig(num_tokens=4, lambda_weight=0.3, sequential=True)
cooker = PressureCookerV5(
    model.parameters(),
    enable_mtp=True,
    mtp_config=mtp_cfg,
)

# Get MTP head for your model
mtp_head = cooker.get_mtp_head(hidden_dim=768, vocab_size=32000)
```

### EMA Weight Shadowing

Track exponential moving averages of weights for evaluation:

```python
cooker = PressureCookerV5(model.parameters(), ema_decay=0.999)

# During training: EMA updates automatically

# For evaluation: swap to EMA weights
cooker.swap_ema_weights(model)
evaluate(model)
cooker.swap_ema_weights(model)  # Swap back
```

### V5S: 3D-ORCP (`PressureCookerV5S`)

V5S is not a subclass of V5 -- it's a separate optimizer (`hypernix.pressure_cooker_v5s.PressureCookerV5S`) built around three co-designed ideas:

1. **3D Cosine Oscillation Resistance (3D-COR)** -- three cosine-similarity EMAs at fast (β≈0.80), medium (β≈0.95), and ultra-slow (β≈0.999) time horizons, combined into a single "volumetric oscillation score" (VOS) that damps the effective learning rate.
2. **Pressure Diffusion (PD)** -- a small 1-D convolution over the flattened gradient that smooths high-frequency noise before it enters the update rule, at O(n) cost and zero extra persistent state.
3. **Low Power Mode (LPM)** -- one quantized int8 momentum buffer plus factored row/column curvature, same as V5, so memory stays close to V5's footprint.

```python
from hypernix.pressure_cooker_v5s import PressureCookerV5S, V5SConfig

# Defaults (general pretraining)
cooker = PressureCookerV5S(model.parameters(), lr=3e-4)

# Fine-tuning starting point (see class docstring for other presets)
cooker = PressureCookerV5S(
    model.parameters(),
    lr=1e-4,
    diffusion_factor=0.05,
    vos_3d_gain=4.0,
    freeze_patience=64,
)

# Or via an explicit config object
cfg = V5SConfig(fast_beta=0.80, med_beta=0.95, ultra_slow_beta=0.999, diffusion_factor=0.12)
cooker = PressureCookerV5S(model.parameters(), v5s_config=cfg, lr=3e-4)

# Diagnostics
cooker.print_summary()                  # human-readable config + live oscillation stats
stats = cooker.get_oscillation_stats()  # {"fast_cos", "med_cos", "ultra_cos", "vos"}
frozen = cooker.get_frozen_fraction()   # fraction of coordinates currently soft-frozen
```

`Agedcookerv5s` is the Pascal-safe (CUDA 6.1/6.2, e.g. GTX 10-series) variant: it caps the diffusion kernel width at 3 and stores curvature buffers in fp16.

## GPU Tiers

| Tier | Class | Use Case | QAT | MTP |
|------|-------|----------|-----|-----|
| CPU T1 | StovetopCooker | Low-memory | Yes | No |
| CPU T2 | ElectricCooker | Multi-core | Yes | Yes |
| GPU T1 | InductionCooker | CUDA + AMP | Yes | Yes |
| GPU T2 | ProCooker | CUDA graphs | Yes | Yes |

## QAT Bit Width Comparison

| Bits | Levels | VRAM Overhead | Use Case |
|------|--------|--------------|----------|
| Q4 | 16 | 1.15x | Extreme compression |
| Q5 | 32 | 1.20x | Mobile/edge |
| Q6 | 64 | 1.25x | Balanced (default) |
| Q8 | 256 | 1.35x | Near-lossless |

## V5 vs V5+

| Feature | V5 | V5+ |
|---------|-----|-----|
| QAT | Manual hook attach | Auto `prepare_model()` |
| Mixed precision | Basic | Per-layer sensitivity |
| Gradient tracking | No | Norm monitoring |
| Sensitivity analysis | No | Built-in |
| Default EMA | 0 (off) | 0.999 |

## CLI

There is currently no `pressure-cooker-v5` CLI subcommand -- `hypernix train run` (see `wiki/Training.md` / `hypernix.cli`) drives the built-in training loop, but optimizer choice, QAT, and MTP are configured in Python, not via CLI flags:

```bash
hypernix train run --model-dir ./snapshot --dataset data.txt --out-dir ./ckpt --lr 3e-4
```

```python
from hypernix.pressure_cooker_v5 import PressureCookerV5, QATConfig
from hypernix.pressure_cooker_v5s import PressureCookerV5S

# V5 with QAT
cooker = PressureCookerV5(model.parameters(), lr=2e-4, qat_config=QATConfig(bits=6))
cooker.attach_qat(model)

# V5S (3D-ORCP core, see the V5S section of this page)
cooker = PressureCookerV5S(model.parameters(), lr=3e-4)
```

## Integration with Freezer

```python
from hypernix import freezer
from hypernix.pressure_cooker_v5 import PressureCookerV5, QATConfig

fz = freezer.auto_freezer()
fz = freezer.flash_freezer(base=fz)

# QAT-aware batch sizing
bs = fz.suggest_qat_batch_size(bits=6, hint=8)

# Prepare model for QAT in freezer context
fz.prepare_for_qat(model, bits=6)
```
