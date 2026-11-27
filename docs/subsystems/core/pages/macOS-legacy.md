# macOS on old Intel Macs (torch 1.13)

This page covers running `hypernix` on an old Intel Mac where
PyTorch 2.x won't install — typically macOS 10.15 Catalina / 11 Big
Sur on a Mid-2015 MBP, a 2018 Mac mini, or an Intel iMac.  The last
PyTorch line that cleanly installs on those machines is **1.13.1**;
`hypernix` since v0.46.0 ships the compat shims required to run its
core training / inference paths on that version.

## TL;DR

```bash
./scripts/install_macos_legacy.sh
source .venv/bin/activate
python -c "import hypernix; print(hypernix.torch_compat.describe())"
```

The installer:

1. Creates `.venv/` if missing.
2. Pins `torch==1.13.1` from the PyPI CPU wheel.
3. Installs `hypernix[legacy-torch]` — as of 0.47.1 the main
   `install_requires` accepts `torch>=1.13`, so pip honours the pin
   from step 2 automatically (no `--no-deps` hack).  The
   `[legacy-torch]` extra pulls the smaller companion deps
   (`numpy<2`, `safetensors>=0.3.1`, etc.) at versions known to
   co-install with torch 1.13.
4. Runs `hypernix.torch_compat.describe()` as a smoke check.

## What the compat shim covers

`hypernix.torch_compat` (v0.46.0+) auto-selects between modern and
fallback implementations at import time, version-gated on
`torch.__version__`.

| API | Needs torch | Fallback for torch 1.13 |
|---|---|---|
| `torch.nn.RMSNorm` | ≥ 2.4 | hand-rolled `nn.Module` matching native semantics |
| `torch.nn.functional.scaled_dot_product_attention` | ≥ 2.0 | explicit softmax(QKᵀ/√d) with causal mask |
| `torch.compile` | ≥ 2.0 | **not supported** — training scripts should branch on `torch_compat.is_legacy_torch()` |

`HyperNixModel`, `NanoNanoModel`, and everything they wrap use the
shim, so you get the same outputs on modern and legacy torch (within
fp rounding).

## What doesn't work on torch 1.13

* **`torch.compile`** — no replacement.  Branch on
  `torch_compat.is_legacy_torch()` in your training script and skip
  the compile step when True.
* **FlashAttention / mem-efficient SDPA** — the fused kernels
  require Ampere+ GPUs and aren't in 1.13.  CPU-only training on
  old Intel Macs doesn't need them anyway.
* **bf16 on CPU** — 1.13's bf16 path is unstable.  Use fp32.
  `freezer.pascal_safe_dtype()` already picks fp32 on
  no-CUDA hosts in v0.41+.
* **`torch.export`, `torch._dynamo.optimize`, TorchDynamo in general**
  — all torch 2.x.  Skip.
* **Intel Mac GGUF quantization** — `llama-quantize` auto-fetch
  pulls an `x86_64-macos` binary from `ggml-org/llama.cpp`.  If
  their latest release dropped your macOS version, prefer
  `brew install llama.cpp` and pass `--llama-quantize` explicitly.

## API surface

```python
from hypernix import torch_compat

torch_compat.TORCH_VERSION        # (1, 13) on an old Mac, (2, 7) elsewhere
torch_compat.is_legacy_torch()    # True on torch < 2.0
torch_compat.has_native_rmsnorm() # True on torch >= 2.4
torch_compat.RMSNorm              # native class or the fallback
torch_compat.scaled_dot_product_attention(q, k, v, is_causal=True)
torch_compat.describe()           # one-shot summary dict
```

Every path in this table is exercised by `tests/data/test_shakers_and_torchcompat.py`.

## Manual install (without the script)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip<24' wheel

# torch 1.13.1 FIRST — pinned so pip doesn't pick a 2.x wheel when
# it resolves hypernix's ``torch>=1.13,<3`` requirement.
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    'torch==1.13.1'

# hypernix + the legacy-torch extra.  No --no-deps needed: the main
# torch pin is now `>=1.13,<3`, so the already-installed torch
# 1.13.1 satisfies the requirement.
python -m pip install 'hypernix[legacy-torch]'

# (Optional) If you prefer not to use the extra, pin the companion
# deps explicitly:
python -m pip install \
    'numpy>=1.21,<2' \
    'safetensors>=0.3.1' \
    'huggingface-hub>=0.16' \
    'gguf>=0.10.0' \
    'tqdm>=4.64' \
    'sentencepiece>=0.1.99'
```

## Recommended training shape on an old Mac

Intel Macs without a dGPU run `hypernix` on CPU.  Use
`freezer.old_freezer()` (default batch=1, ctx=512, fp32 on CPU) and
the `smoke_alarm.GasAlarm(preset="i7-7660u")` or similar for a
realistic step-time estimate:

```python
from hypernix import freezer, smoke_alarm

fz = freezer.old_freezer()
alarm = smoke_alarm.gas_alarm(
    time_budget_seconds=4 * 3600,
    preset="i7-7660u",           # or i7-7700hq on 15" MBPs
    model_params=30_000_000,     # keep it small
)
print(alarm.budget())
# TrainingBudget(recommended_steps=~300 for a 4h CPU run)
```

A 4-hour CPU run on a 2016-era i7 is realistic for **fine-tuning a
30M-param model** or **running inference + evaluation** on any
pre-trained snapshot.  Don't try to pretrain from scratch on an old
Intel Mac.

## Caveats

* **Not actually tested in our CI.**  The GitHub runners use torch
  2.7.  The compat shim's fallback paths are exercised by the
  `test_shakers_and_torchcompat.py` tests, but a real torch-1.13
  install is verified manually.  File an issue if you hit a
  regression.
* **The legacy path ends at hypernix 0.72.6.post4.**  `torch==1.13.1`
  is Python 3.8 – 3.10 only, and from 0.72.7 hyperNix-pip requires
  Python 3.12 or newer (it supports 3.12, 3.13, 3.14 and 3.15).  On
  3.10, `pip install "hypernix[legacy-torch]"` -- what the script runs --
  skips every release whose `Requires-Python` excludes 3.10 and installs
  0.72.6.post4, the last that runs on torch 1.13.  It gets security
  fixes only as far as that release has them.  Its source is the
  `lts/0.72.6` branch (the commit tagged `v0.72.6.post4`), for building
  or patching it yourself.
* **GGUF + large models**: large snapshots may exceed RAM on old
  Macs.  Prefer already-quantized `.q4_k_m.gguf` files if you can
  find them on the Hub and run inference via `llama.cpp` rather
  than through the Python sampler.
