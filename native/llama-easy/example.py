"""Make a Llama model use llama-easy in one call, and check its logits.

    python example.py            # on a CUDA or ROCm GPU
"""

import torch
from kernels import get_kernel
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(vocab_size=1000, hidden_size=256, intermediate_size=688,
                     num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4)
model = LlamaForCausalLM(config).to("cuda", torch.bfloat16).eval()
ids = torch.randint(0, 1000, (2, 64), device="cuda")

with torch.no_grad():
    before = model(ids).logits

llama_easy = get_kernel("ray0rf1re/llama-easy", version=1, trust_remote_code=True)
llama_easy.kernelize(model)

with torch.no_grad():
    after = model(ids).logits

print("swapped:", model.llama_easy_kernelized)
print("largest logit change:", (after.float() - before.float()).abs().max().item())
