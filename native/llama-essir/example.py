"""Kernelize a tiny random Llama and check that the logits do not move.

    python example.py            # on a CUDA or ROCm GPU
"""

import torch
from kernels import FuncRepository, LayerRepository, Mode, kernelize, use_kernel_mapping
from transformers import LlamaConfig, LlamaForCausalLM

REPO = "ray0rf1re/llama-essir"
TRUST = [REPO]
MAPPING = {
    "RMSNorm": {"cuda": LayerRepository(REPO, layer_name="RMSNorm", version=1, trust_remote_code=TRUST)},
    "rotary_pos_emb": {
        "cuda": FuncRepository(REPO, func_name="apply_rotary_transformers", version=1, trust_remote_code=TRUST)
    },
}

config = LlamaConfig(vocab_size=1000, hidden_size=256, intermediate_size=688,
                     num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4)
model = LlamaForCausalLM(config).to("cuda", torch.bfloat16).eval()
ids = torch.randint(0, 1000, (2, 64), device="cuda")

with torch.no_grad():
    before = model(ids).logits
with use_kernel_mapping(MAPPING):
    kernelize(model, mode=Mode.INFERENCE)
with torch.no_grad():
    after = model(ids).logits

print("RMSNorm now runs", model.model.layers[0].input_layernorm.forward.__func__.__qualname__)
print("largest logit change:", (after.float() - before.float()).abs().max().item())
