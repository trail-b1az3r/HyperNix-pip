# llama-essir — a Hugging Face Hub kernel for Llama models

`llama-essir` is hyperNix-pip's kernel on the
[Hugging Face Kernel Hub](https://huggingface.co/docs/kernels/index):
Triton versions of the three element-wise hot spots of every Llama block,
published as [`ray0rf1re/llama-essir`](https://huggingface.co/ray0rf1re/llama-essir)
and swapped into a model at run time by the `kernels` library. Nothing
is compiled on the user's machine and nothing in the model changes.

| Op | Replaces | In the kernel |
| --- | --- | --- |
| RMSNorm | `LlamaRMSNorm` (Transformers hook `RMSNorm`) | `layers.RMSNorm`, `rms_norm` |
| Rotary embedding | `apply_rotary_pos_emb` (hook `rotary_pos_emb`) | `apply_rotary_transformers`, `rotary` |
| SwiGLU gate | a `SiluAndMul` layer | `layers.SiluAndMul`, `silu_and_mul` |

Every Llama-architecture model in Transformers — Llama itself and the
models built from its blocks — carries the first two hooks.

## Using it

From hyperNix-pip, on a CUDA or ROCm GPU:

```python
import torch
from transformers import AutoModelForCausalLM
from hypernix.hub_kernels import kernelize_llama

model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16).cuda()
kernelize_llama(model)                       # inference
kernelize_llama(model, mode="inference+compile")   # before torch.compile
```

Without hyperNix-pip, with only `kernels` and Transformers, the kernel's
[Hub page](https://huggingface.co/ray0rf1re/llama-essir) shows the
mapping to register. The functions are also usable on their own:

```python
from hypernix.hub_kernels import load

k = load()
y = k.rms_norm(x, weight, 1e-6)
q, kk = k.apply_rotary_transformers(q, kk, cos, sin)
```

`pip install kernels` is required (and Transformers, for a Transformers
model).

### What it does and does not do

- **Precision.** Each op computes in fp32 and rounds once. RMSNorm
  rounds at the same point Transformers does and matches it to rounding
  noise; RoPE and the SwiGLU gate round fewer times than the PyTorch
  originals, so in bf16/fp16 they differ from them by about one rounding
  step — towards the exact value.
- **`torch.compile`.** The ops are Torch custom ops with fake kernels,
  so a compiled model traces through them without a graph break.
- **Training.** There is no backward pass. In training mode
  `kernelize_llama` leaves the model's own layers in place (or raises,
  with `use_fallback=False`).
- **CPU.** `kernels.kernelize` only targets accelerators, so a CPU
  model keeps its layers. The kernel's own functions run on CPU tensors
  through the PyTorch definition.
- **Trust.** `ray0rf1re` is not on the `kernels` trusted-publisher
  list, so loading needs consent. `hypernix.hub_kernels` gives it to
  exactly `ray0rf1re/llama-essir` — not to the publisher's other
  repositories.

## Where it lives

The source is [`native/llama-essir`](../native/llama-essir/README.md): a
kernel-builder project (`build.toml`, `flake.nix`, `torch-ext/`) of the
`torch-noarch` kind — Triton and PyTorch only, one build per backend
(`cpu`, `cuda`, `rocm`) that runs on every Torch and every GPU
generation. The loader is `hypernix.runtime.hub_kernels`
(`hypernix.hub_kernels`).

## How it is published

`.github/workflows/llama-essir-kernel.yml` runs on every change to the
kernel:

1. **test** — the kernel is built in kernel-builder's layout, loaded
   through `kernels` like the Hub copy, and checked against the fp32
   truth on the PyTorch path and, through Triton's interpreter, the
   Triton kernels themselves; then a real Transformers Llama is
   kernelized and its logits compared.
2. **publish** — on `main`, or when the workflow is run by hand: Nix with
   the Hugging Face binary cache, `kernel-builder check-config`, then
   `kernel-builder build-and-upload`, which writes the `v1` branch and the
   Hub page (from `CARD.md`).

Publishing needs two things once:

- kernel-creation access for the `ray0rf1re` account
  (huggingface.co/settings/account, "Request Kernels Creation"), and
- a Hugging Face **write** token in the repository secret `HF_TOKEN`.

Without the secret the tests still run and the upload is skipped with a
notice in the run.

## Versions

`version` in `build.toml` is the kernel's major version and its Hub
branch (`v1`); `hypernix.hub_kernels.VERSION` is pinned to it. Code
written against v1 keeps working for as long as v1 exists, so anything a
caller could notice — a new public function, a changed signature, a new
dtype — is a new major version.
