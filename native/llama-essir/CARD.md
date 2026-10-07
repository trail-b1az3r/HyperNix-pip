---
library_name: kernels
license: other
license_name: hypernix
license_link: https://github.com/trail-b1az3r/hypernix-pip/blob/main/LICENSE
tags:
  - kernels
  - llama
  - triton
---

# llama-essir

Triton kernels for Llama-architecture models: **RMSNorm**, **rotary
position embeddings** and the **SwiGLU gate** -- the element-wise hot
spots of every Llama block -- as drop-in layers for Transformers models
through the [`kernels`](https://github.com/huggingface/kernels) library.

Built from [`native/llama-essir`](https://github.com/trail-b1az3r/hypernix-pip/tree/main/native/llama-essir)
in hyperNix-pip and published automatically on every change.

## Use it with a Llama model

```python
# pip install -U kernels transformers
import torch
from kernels import FuncRepository, LayerRepository, Mode, kernelize, use_kernel_mapping
from transformers import AutoModelForCausalLM

REPO = "{{ repo_id }}"
trust = [REPO]  # this repository only; its publisher is not on the trusted list
mapping = {
    "RMSNorm": {"cuda": LayerRepository(REPO, layer_name="RMSNorm", version={{ version }}, trust_remote_code=trust)},
    "rotary_pos_emb": {"cuda": FuncRepository(REPO, func_name="apply_rotary_transformers", version={{ version }}, trust_remote_code=trust)},
}

model = AutoModelForCausalLM.from_pretrained("<a Llama-architecture model>", torch_dtype=torch.bfloat16).cuda()
with use_kernel_mapping(mapping):
    kernelize(model, mode=Mode.INFERENCE)
```

With hyperNix-pip installed, `from hypernix.hub_kernels import kernelize_llama; kernelize_llama(model)`
does the same for CUDA and ROCm.

## Use the functions directly

```python
from kernels import get_kernel

k = get_kernel("{{ repo_id }}", version={{ version }}, trust_remote_code=["{{ repo_id }}"])
y = k.rms_norm(x, weight, eps=1e-6)               # LlamaRMSNorm
h = k.silu_and_mul(gate_up)                       # silu(a) * b over the last dim's halves
q, kk = k.apply_rotary_transformers(q, kk, cos, sin)  # apply_rotary_pos_emb
```

| | |
| --- | --- |
| Backends | CUDA and ROCm through Triton; CPU runs the plain PyTorch definition |
| Dtypes | float32, bfloat16, float16 -- computed in fp32, rounded once |
| `torch.compile` | yes, without graph breaks (custom ops with fake kernels) |
| Training | no backward pass: in training mode `kernelize` keeps the model's own layers |

{% if functions %}
## Functions
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% endif %}
{% if layers %}
## Layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}
