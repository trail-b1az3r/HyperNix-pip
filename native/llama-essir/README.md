# llama-essir

Triton kernels for Llama-architecture models, published to the Hugging
Face Kernel Hub as [`ray0rf1re/llama-essir`](https://huggingface.co/ray0rf1re/llama-essir).
`CARD.md` becomes the Hub page; this file is for working on the kernel.

| Op | Replaces (Transformers hook) | Kernel symbol |
| --- | --- | --- |
| RMSNorm | `LlamaRMSNorm` (`RMSNorm`) | `layers.RMSNorm`, `rms_norm` |
| Rotary embedding | `apply_rotary_pos_emb` (`rotary_pos_emb`) | `apply_rotary_transformers`, `rotary` |
| SwiGLU gate | `SiluAndMul` | `layers.SiluAndMul`, `silu_and_mul` |

Every Llama-architecture model in Transformers carries the first two
hooks, so `kernelize` swaps them in with no change to the model.

## Layout

```
build.toml                kernel-builder config: torch-noarch, cpu/cuda/rocm
flake.nix                 kernel-builder's Nix entry point
CARD.md                   Hub model card template
torch-ext/llama_essir/
  __init__.py             the public API (__all__)
  op.py                   Torch custom ops, in the build's own namespace
  _triton.py              the Triton kernels
  _reference.py           the same ops in plain PyTorch (the definition)
  layers.py               pure layers for kernelize
tests/                    GPU tests, run by `kernel-builder testshell`
local_build.py            kernel-builder's output layout, without Nix
example.py                kernelize a tiny Llama on a GPU
```

Kernel code may import only the standard library, `torch`, `triton` and
itself (relatively); `tests/runtime/test_hub_kernels.py` checks that.

## Develop and test

Without a GPU, from the repository root:

```bash
pip install kernels transformers triton
pytest tests/runtime/test_hub_kernels.py
```

That builds the kernel with `local_build.py`, loads it through
`kernels` like the Hub copy, compares every op with the fp32 truth on the
PyTorch path and -- with `TRITON_INTERPRET=1` -- the Triton kernels
themselves, and kernelizes a real Transformers Llama.

On a GPU, with kernel-builder (`curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash`):

```bash
cd native/llama-essir
kernel-builder testshell
python -m pytest tests
```

## Publishing

`.github/workflows/llama-essir-kernel.yml` tests the kernel on every
change and, on `main` (or when run by hand), builds it with
`kernel-builder build-and-upload`. It needs, once:

1. Kernel-creation access for the `ray0rf1re` account:
   huggingface.co/settings/account, "Request Kernels Creation".
2. A Hugging Face **write** token stored as the repository secret
   `HF_TOKEN` (Settings, Secrets and variables, Actions).

Without the secret the workflow still tests and skips the upload, saying
so.

The version in `build.toml` is the kernel's major version and its Hub
branch (`v1`). Bump it for any change that code written against the old
builds could notice: a new public symbol, a changed signature, a new
dtype. `hypernix.hub_kernels.VERSION` is pinned to the same number.
