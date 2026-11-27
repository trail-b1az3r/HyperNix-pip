# Pressure Cooker v3 — `hypernix.pressure_cooker_v3`

> **Deprecated since 0.72.6.** Constructing `PressureCookerV3` (or a
> subclass) warns with a `FutureWarning`: use `PressureCookerV4`, which
> is kept, or [V5](Pressure-Cooker-V5.md) / [V6](Pressure-Cooker-V6.md)
> for new training. Importing the module does not warn, because V4 uses
> its helpers. The original `PressureCooker` (v1) is deprecated the same
> way; there has never been a V2 class.

ZeRO-aware optimizer family replacing V2 with FP8/QAT support, Pascal-safe
variants, and a CPU-optimized lite tier.

## Classes

| Class | Role |
|---|---|
| `PressureCookerV3` | Base V3 optimizer — warmup / plateau / cosine cooldown LR, EMA, lookahead, ZeRO stage hooks |
| `PressureCookerV3Plus` | Full quantization-aware training (QAT) with calibration + fake-quant |
| `StovetopV3Cooker` | Pascal (sm_61) safe — disables fused / foreach / amsgrad |
| `StovetopV3CookerPlus` | Pascal-safe V3Plus with EMA + adaptive clipping (v0.70.3) |
| `CookerLite` | CPU-only fast path. Now a `PressureCookerV4` subclass in `hypernix.optimizers.pressure_cooker_v4`; `from hypernix import CookerLite` still works |

## Quantization

```python
from hypernix import PressureCookerV3Plus, QuantConfig, QuantDtype

qc = QuantConfig(dtype=QuantDtype.FP8, enabled=True, fake_quant=True)
opt = PressureCookerV3Plus(model.parameters(), quant_config=qc, lr=3e-4)
```

Supported dtypes: `FP8`, `FP16`, `FP32`, `FP64`, `Q8`, `Q6`, `Q5_5`, `Q4M`.

## Pascal / GTX 10-series

On CUDA compute capability ≤ 6.1, V3 automatically disables fused kernels.
Use `StovetopV3Cooker` or `StovetopV3CookerPlus` explicitly when training on
GTX 1080 / 1070 hardware. See [Pascal.md](Pascal.md).

## LR schedule

`scheduled_lr(step)` runs linear warmup → flat plateau → cosine cooldown
with a `1e-6` floor (prevents collapse at end of training).

## CLI / instant pot

`instant_pot.brew()` accepts `PressureCookerV3` / `StovetopV3CookerPlus` via
the recipe's optimizer hook. Pair with [Abbicus](Abbicus.md) for curriculum
length regulation.

See also: [Kitchen.md](Kitchen.md), [Training.md](Training.md), [Quantization.md](Quantization.md).
