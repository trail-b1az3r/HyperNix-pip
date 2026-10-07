"""GPU tests, run against the kernel as it is loaded from the Hub.

`kernel-builder testshell` points LOCAL_KERNELS at the local build, so
the same get_kernel call tests a build before it is uploaded. The
HyperNix test suite (tests/runtime/test_hub_kernels.py) covers the CPU
path and the Triton kernels under Triton's interpreter.
"""

import kernels
import pytest
import torch
import torch.nn.functional as F

REPO_ID = "ray0rf1re/llama-easy"
llama_easy = kernels.get_kernel(REPO_ID, version=1, trust_remote_code=[REPO_ID])

DTYPES = [torch.float32, torch.bfloat16, torch.float16]
EPS = {torch.float32: 1e-6, torch.bfloat16: 2**-7, torch.float16: 2**-10}


def _close(actual, truth, dtype, steps=2.0):
    scale = truth.abs().max().item() or 1.0
    torch.testing.assert_close(actual.float(), truth.float(), rtol=0, atol=steps * EPS[dtype] * scale)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [64, 4096, 5120])
def test_rms_norm(device, dtype, hidden):
    x = torch.randn(33, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, dtype=dtype, device=device)
    h = x.float()
    normed = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype)
    _close(llama_easy.rms_norm(x, w, 1e-6), w.float() * normed.float(), dtype)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul(device, dtype):
    x = torch.randn(17, 2 * 11008, dtype=dtype, device=device)
    truth = F.silu(x.float()[..., :11008]) * x.float()[..., 11008:]
    _close(llama_easy.silu_and_mul(x), truth, dtype)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_rotary_on_a_transposed_q(device, dtype, head_dim):
    q = torch.randn(2, 19, 8, head_dim, dtype=dtype, device=device).transpose(1, 2)
    cos = torch.randn(2, 19, head_dim, dtype=dtype, device=device)
    sin = torch.randn(2, 19, head_dim, dtype=dtype, device=device)
    q_out, _ = llama_easy.apply_rotary_transformers(q, q, cos, sin)
    qf, c, s = q.float(), cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    half = head_dim // 2
    truth = qf * c + torch.cat((-qf[..., half:], qf[..., :half]), -1) * s
    _close(q_out, truth, dtype, steps=4)


@pytest.mark.kernels_ci
def test_rms_norm_layer(device):
    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(256, device=device))
            self.variance_epsilon = 1e-5

    norm = Norm()
    x = torch.randn(4, 256, device=device)
    out = llama_easy.layers.RMSNorm.forward(norm, x)
    torch.testing.assert_close(out, llama_easy.rms_norm(x, norm.weight, 1e-5))


@pytest.mark.kernels_ci
def test_compiles_without_graph_breaks(device):
    x = torch.randn(8, 512, device=device, dtype=torch.bfloat16)
    w = torch.randn(512, device=device, dtype=torch.bfloat16)
    compiled = torch.compile(lambda a, b: llama_easy.rms_norm(a, b, 1e-6), fullgraph=True)
    torch.testing.assert_close(compiled(x, w), llama_easy.rms_norm(x, w, 1e-6))


@pytest.mark.kernels_ci
def test_kernelize_a_llama_in_one_call(device):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=1000, hidden_size=256, intermediate_size=688,
                                      num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4)
    model = transformers.LlamaForCausalLM(config).to(device, torch.bfloat16).eval()
    ids = torch.randint(0, 1000, (2, 64), device=device)
    with torch.no_grad():
        before = model(ids).logits
        llama_easy.kernelize(model)
        after = model(ids).logits
    assert model.llama_easy_kernelized == {"RMSNorm": 5, "rotary_pos_emb": 1}
    # bf16 logits, the kernels rounding once where PyTorch rounds per op.
    torch.testing.assert_close(after.float(), before.float(), rtol=0.05, atol=0.1)
