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

# llama-easy

Make any Llama-architecture model faster with one line. Triton kernels
for the element-wise hot spots of every Llama block -- **RMSNorm**,
**rotary position embeddings** and the **SwiGLU gate** -- that drop into
a Transformers model with nothing to configure.

Built from [`native/llama-easy`](https://github.com/trail-b1az3r/hypernix-pip/tree/main/native/llama-easy)
in hyperNix-pip and published automatically on every change.

## The easy way: one call

```python
# pip install -U kernels transformers
import torch
from kernels import get_kernel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("<any Llama-architecture model>",
                                             torch_dtype=torch.bfloat16, device_map="cuda")

llama_easy = get_kernel("{{ repo_id }}", version={{ version }}, trust_remote_code=True)
llama_easy.kernelize(model)
print(model.llama_easy_kernelized)   # {'RMSNorm': ..., 'rotary_pos_emb': 1}
```

`trust_remote_code=True` is needed because this publisher is not on the
`kernels` trusted list; it applies to this one repository.

## Or let Transformers do it while loading

```python
from transformers import AutoModelForCausalLM, KernelConfig

config = KernelConfig({
    "RMSNorm": ("{{ repo_id }}:RMSNorm", {"version": {{ version }}, "trust_remote_code": True}),
    "rotary_pos_emb": ("{{ repo_id }}:ApplyRotary", {"version": {{ version }}, "trust_remote_code": True}),
})
model = AutoModelForCausalLM.from_pretrained("<model>", device_map="cuda", kernel_config=config)
```

With hyperNix-pip installed it is shorter still:
`from hypernix.hub_kernels import from_pretrained; model = from_pretrained("<model>", device_map="cuda")`.

## Use the functions directly

```python
y = llama_easy.rms_norm(x, weight, eps=1e-6)                 # LlamaRMSNorm
h = llama_easy.silu_and_mul(gate_up)                         # silu(a) * b over the last dim's halves
q, k = llama_easy.apply_rotary_transformers(q, k, cos, sin)  # apply_rotary_pos_emb
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
