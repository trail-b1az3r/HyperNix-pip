# Devices — CUDA, ROCm, Metal, Intel, Vulkan, CPU

```bash
hypernix devices
hypernix devices --json
hypernix generate --model-dir m.gguf --hnx-device cuda
hypernix chat     --model-dir m.gguf --hnx-device auto --cache-bytes 4G
hypernix hyprslug-headers serve m.gguf --device cuda:1
```

## Start here

```
$ hypernix devices
Devices this runtime can use:
  ok  cuda:0     NVIDIA GeForce RTX 3090 (sm_86, Ampere)  25.4 GB
  no  mps        Apple Metal
        This torch was not built with MPS.
  no  vulkan     Vulkan (via llama.cpp, not torch)
        PyTorch has no usable Vulkan backend for transformer inference.
  ok  cpu        x86_64, 32 threads

  --device auto would pick: cuda:0 (NVIDIA GeForce RTX 3090 …)
```

Unusable backends are listed **with the reason**, deliberately. "CUDA is
not available" and "CUDA is available and this wheel has no kernels for
your card" are different problems with different fixes, and a probe that
only showed what works could not tell them apart.

## The Pascal problem — GTX 1080 and everything sm_61

This is the one worth reading even if nothing else here is.

A GTX 1060/1070/1080, a Titan Xp or a P40 is **compute capability 6.1**.
Recent PyTorch wheels do not build for it:

```python
>>> torch.__version__
'2.13.0+cu130'
>>> torch.cuda.get_arch_list()
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
>>> torch.cuda.is_available()
True                      # ← and this is the trap
```

`is_available()` returns **True**. The driver is fine, the card is
visible, memory reports correctly. Everything looks right until the first
kernel launch:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

which reads like a broken driver and is actually a wheel that was never
built for the card. `hypernix devices` compares the capability against
`get_arch_list()` and says so up front, with the wheel to install:

```
  no  cuda:0     NVIDIA GeForce GTX 1080 (sm_61, Pascal (GTX 1060/1070/1080, …))
        This torch (2.13.0+cu130) has no kernels for sm_61. It was built for
        sm_75, sm_80, sm_86, sm_90, sm_100, sm_120. torch.cuda.is_available()
        is True and the first kernel launch would fail with 'no kernel image
        is available for execution on the device'.
        Pascal and Maxwell need a CUDA 11.8 or 12.1-12.6 wheel; builds from
        about cu128 onward drop sm_50 through sm_61. Install:
            pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Which wheel for which card

| Capability | Cards | Wheel |
|---|---|---|
| 5.0–5.2 | Maxwell (GTX 9xx, M60) | cu118 |
| 6.0–6.1 | Pascal (GTX 10xx, P100, P40) | cu118, or cu121–cu126 |
| 7.0 | Volta (V100) | cu118 or any cu12x |
| 7.5 | Turing (RTX 20xx, T4) | anything current |
| 8.0–8.9 | Ampere, Ada | anything current |
| 9.0+ | Hopper, Blackwell | cu124+ / cu128+ |

CUDA **11.8** is the last toolchain that covers Kepler through Hopper in
one wheel, which is why it is the recommendation for anything older than
Turing. CUDA **12.x** drops sm_35/sm_37 and, from around cu128, sm_50
through sm_61 as well.

### Half precision on Pascal is a trap, not a win

GP102/GP104 run FP16 at **1/64** of their FP32 rate. It is *present*, so a
rule as reasonable-looking as "use half on CUDA, float on CPU" finds it
and makes a GTX 1080 dramatically slower while appearing to optimise it.

`default_dtype()` returns float32 below `sm_70` for that reason, and
`hypernix devices` says so on the card:

```
  ok  cuda:0     NVIDIA GeForce GTX 1080 (sm_61, Pascal …)  8.5 GB
        float32 only: sm_61 runs FP16 at a fraction of FP32 rate, so half
        precision would make this slower, not faster.
```

## What running on a GPU actually buys

The packed bytes go to the card **once**, at load, and the decode happens
there. That is the opposite of the obvious arrangement, and the obvious
one is much worse.

Decoding on the host and pushing the result means every forward pass
moves **expanded float32** across PCIe — for a 0.9-bit tensor that is 34×
the bytes the tensor occupies, every token, to save nothing. The packed
form is the small one; that is the entire premise of the tier.

| 7B at `IQ0.9_L` | |
|---|---|
| packed bytes uploaded, once | ~800 MB |
| dequantised, per forward pass | ~28 GB |
| its float16 weights would need | ~14 GB |

So this is not only a speed question. Holding the packed bytes in VRAM is
what lets a 7B sub-bit model fit on a card that could not hold its float16
weights at all.

The decoder is in `hypernix.models.hnxtorch`, written in ops that exist on
every backend torch supports — shifts, masks, gathers, reshapes. No
custom kernels and nothing CUDA-specific, so the same code runs on CUDA,
ROCm, MPS and XPU. It is asserted **bit-identical** to the numpy decoders,
not merely close: integer unpacking followed by one multiply has no
rounding to hide behind.

## ROCm

A ROCm build of torch reports itself as `cuda` — `torch.version.hip` is
set and `torch.version.cuda` is not. Everything above applies unchanged;
`hypernix devices` shows the HIP version where it would show the CUDA one.

## Apple Metal (MPS) and Intel (XPU)

Both are probed and both work if the torch build has them. MPS needs
Apple Silicon and a standard macOS wheel; XPU needs a torch built for
Intel GPUs (`intel-extension-for-pytorch`, or torch 2.5+ from Intel's
index). Neither is special-cased — the decoder is device-agnostic.

## Vulkan

**There is no useful Vulkan path through PyTorch.** The backend exists in
the source tree, is not built into any released wheel, and implements a
small set of vision ops rather than what a transformer needs. Reporting it
as available because an import succeeded would be a lie with a long
debugging tail, so `--device vulkan` refuses — and answers with the route
that does work:

```
PyTorch has no usable Vulkan backend — it is not built into any released
wheel and implements vision ops, not a transformer. The Vulkan path that
works is llama.cpp's, which is what LM Studio uses on AMD, Intel and older
NVIDIA cards. Convert the model to a type it can read:
    hypernix hyprslug-headers wrap MODEL.gguf -o compat.gguf
and load compat.gguf in LM Studio with the Vulkan runtime selected.
```

That copy is a stock quantisation, not a sub-bit model — see
[HyprSlug-Headers](HyprSlug-Headers.md). To keep the tier, run it on CPU
or CUDA here and reach it over HTTP with `hyprslug-headers serve`.

## Using it from LM Studio and Bionic

They share a model store and list whatever their bundled llama.cpp can
open, which a sub-bit GGUF is not. Two routes:

```bash
# Convert and place it where both of them look.
hypernix hyprslug-headers install-model model.iq09.gguf \
    --to Q4_K_M --publisher HyperNix --name Qwen3.8-2B

# Or keep the tier and let them talk to it over HTTP.
hypernix hyprslug-headers serve model.iq09.gguf --device cuda --port 1234
```

`install-model` writes `<root>/<publisher>/<name>/<name>.gguf`, which is
the layout both scan. An already-upstream GGUF is *copied* rather than
re-quantised — it already opens, and re-encoding would lose a generation
of quality for nothing. Set `LMSTUDIO_HOME` or pass `--root` if the store
is somewhere unusual.

## The bug that proves this needed a device to test on

`--hnx-device` defaults to `auto`, so the accelerator path is the default
path. It was also, until `7a6e68d`, broken on **every** accelerator — and
the whole local suite passed anyway.

`_rope` built its inverse-frequency table with `torch.arange(...)` and no
`device=`. On a CPU run that is right by accident, because the default
device *is* the CPU. Anywhere else the table sits on the CPU, the
positions sit on the GPU, and the first token dies:

```
RuntimeError: Expected all tensors to be on the same device, but found
at least two devices, mps:0 and cpu!
```

Not an MPS quirk — CUDA and XPU would have failed identically on the
first forward pass. The macOS CI runners are simply the only machines in
the matrix with a device, so they were the only jobs that could see it:
twelve failures there, green everywhere else, green locally.

The second half was the same shape. `generate_tokens` seeds a
`torch.Generator(device="cpu")`, and `torch.multinomial` refuses a
generator whose device differs from the tensor's, so seeded sampling
raised rather than sampled. The vector is now moved to the CPU rather
than the generator to the device, which also makes a seed pick the same
draws on every backend.

**What the tests learned from it.** A placement bug is invisible on a
one-device machine, so more tests of the ordinary kind could not have
found it. `tests/models/test_hnx_device_placement.py` uses `device="meta"` —
tensors that allocate nothing but still carry a device identity torch
enforces — which makes "would break on MPS" an ordinary assertion that
fails on a CPU-only box. Alongside it, an audit parses the runtime
modules and requires every `torch` tensor factory to say where its result
goes, because that is the class the one line belonged to. Four of the
eleven fail on the pre-fix source.

## Refusals

`--device auto` falls back to the CPU, which cannot be absent. A **named**
device that is present but unusable raises with the reason and the remedy
rather than being silently downgraded: someone who typed `--device cuda`
wants to know why they did not get it, not a slow run they cannot explain.

## See also

- [HnxRun](HnxRun.md) — the runtime and the folded matmul
- [LowBit](LowBit.md) — the types the on-device decoder handles
- [HyprSlug-Headers](HyprSlug-Headers.md) — wrap, serve, install-model
