# llama-easy — make Llama models faster in one line

`llama-easy` is hyperNix-pip's kernel on the
[Hugging Face Kernel Hub](https://huggingface.co/docs/kernels/index): it
makes any Llama-architecture model easier to speed up. Triton versions of
the element-wise hot spots of every Llama block, published as
[`ray0rf1re/llama-easy`](https://huggingface.co/ray0rf1re/llama-easy),
swapped into a model with one call. Nothing is compiled on the user's
machine and the model's weights and code are untouched.

| Op | Replaces | In the kernel |
| --- | --- | --- |
| RMSNorm | `LlamaRMSNorm` (Transformers hook `RMSNorm`) | `layers.RMSNorm`, `rms_norm` |
| Rotary embedding | `apply_rotary_pos_emb` (hook `rotary_pos_emb`) | `layers.ApplyRotary`, `apply_rotary_transformers`, `rotary` |
| SwiGLU gate | a `SiluAndMul` layer | `layers.SiluAndMul`, `silu_and_mul` |

Every Llama-architecture model in Transformers — Llama itself and the
models built from its blocks — carries the first two hooks.

## Using it — easiest first

**1. One call, with only `kernels`:**

```python
from kernels import get_kernel

llama_easy = get_kernel("ray0rf1re/llama-easy", version=1, trust_remote_code=True)
llama_easy.kernelize(model)          # a model already on the GPU
model.llama_easy_kernelized          # {'RMSNorm': 65, 'rotary_pos_emb': 1}
```

The counts say what was swapped, so a model the kernel did not recognise
shows zeros instead of silently running as before.

**2. From hyperNix-pip, while loading:**

```python
from hypernix.hub_kernels import from_pretrained

model = from_pretrained("<any Llama-architecture model>", device_map="cuda", torch_dtype="bfloat16")
```

`kernelize_llama(model)` does the same for a model already loaded.

**3. Transformers' own route:** `from_pretrained(..., kernel_config=...)`
with the `KernelConfig` from `hypernix.hub_kernels.kernel_config()` (the
kernel's [Hub page](https://huggingface.co/ray0rf1re/llama-easy) spells
it out for people without hyperNix-pip).

The functions also work on their own:
`llama_easy.rms_norm(x, weight, 1e-6)`,
`llama_easy.apply_rotary_transformers(q, k, cos, sin)`,
`llama_easy.silu_and_mul(gate_up)`.

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
- **CPU.** The kernel is GPU code. `from_pretrained` and
  `kernelize_llama` leave a CPU model as it is (Transformers and
  `kernels.kernelize` do not target the CPU); the kernel's own
  `kernelize` and functions work there through the PyTorch definition,
  which gains nothing but lets a model move between devices.
- **Only this kernel.** `kernelize_llama` applies this kernel's mapping
  alone, not whatever global mapping is registered, so it never fetches
  kernels for hooks you did not ask to replace.
- **Trust.** `ray0rf1re` is not on the `kernels` trusted-publisher
  list, so loading needs consent. `hypernix.hub_kernels` gives it to
  exactly `ray0rf1re/llama-easy` — not to the publisher's other
  repositories. (`trust_remote_code=True` in `get_kernel` likewise
  covers only the repository being loaded.)

## Where it lives

The source is [`native/llama-easy`](../native/llama-easy/README.md): a
kernel-builder project (`build.toml`, `flake.nix`, `torch-ext/`) of the
`torch-noarch` kind — Triton and PyTorch only, one build per backend
(`cpu`, `cuda`, `rocm`) that runs on every Torch and every GPU
generation. The loader is `hypernix.runtime.hub_kernels`
(`hypernix.hub_kernels`).

## How it is published

`.github/workflows/llama-easy-kernel.yml` runs on every change to the
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
