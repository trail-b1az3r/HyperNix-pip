"""llama-easy, the Hub kernel in native/llama-easy, and its loader.

Everything here loads the kernel the way a user does -- through the
``kernels`` library, from a build laid out exactly as kernel-builder
lays one out (``local_build.py``) -- rather than importing the source.

Two paths are checked. Without a GPU the ops run their PyTorch
definition; ``TRITON_INTERPRET=1`` runs the Triton kernels themselves on
CPU tensors, in a child process because Triton reads it at import. Both
are compared with the fp32 truth, within one rounding step of the dtype.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
kernels = pytest.importorskip("kernels")

from hypernix import hub_kernels  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "native" / "llama-easy"
HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
EPS = {torch.float32: 1e-6, torch.bfloat16: 2**-7, torch.float16: 2**-10}


def _local_build():
    spec = importlib.util.spec_from_file_location("llama_easy_local_build", KERNEL / "local_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory) -> Path:
    """A kernel-builder build when LLAMA_EASY_BUILD names one (the
    directory holding its ``build/``), else local_build.py's."""
    given = os.environ.get("LLAMA_EASY_BUILD")
    if given:
        return Path(given)
    out = tmp_path_factory.mktemp("llama-easy")
    _local_build().build(out)
    return out


@pytest.fixture(scope="module")
def kernel(build_dir):
    return hub_kernels.load(local_path=build_dir)


def _close(actual, truth, dtype, steps: float = 2.0):
    """Within *steps* rounding steps of *dtype*, scaled to the values."""
    scale = truth.abs().max().item() or 1.0
    torch.testing.assert_close(actual.float(), truth.float(), rtol=0, atol=steps * EPS[dtype] * scale)


# -- the build ------------------------------------------------------------


class TestTheBuild:
    def test_one_variant_per_backend_in_build_toml(self, build_dir):
        assert sorted(p.name for p in (build_dir / "build").iterdir()) == [
            "torch-cpu", "torch-cuda", "torch-rocm"]

    def test_metadata_names_the_kernel_and_its_backend(self, build_dir):
        meta = json.loads((build_dir / "build" / "torch-cuda" / "metadata.json").read_text(encoding="utf-8"))
        assert meta["name"] == "llama-easy" and meta["version"] == hub_kernels.VERSION
        assert meta["backend"] == {"type": "cuda"}

    def test_the_hub_repo_and_version_agree_with_build_toml(self):
        import tomllib

        config = tomllib.loads((KERNEL / "build.toml").read_text(encoding="utf-8"))
        assert config["general"]["hub"]["repo-id"] == hub_kernels.REPO_ID
        assert config["general"]["version"] == hub_kernels.VERSION

    def test_it_loads_and_exports_its_api(self, kernel):
        assert set(kernel.__all__) == {
            "apply_rotary_transformers", "kernelize", "layers", "rms_norm", "rotary", "silu_and_mul"}
        assert kernel.layers.RMSNorm and kernel.layers.SiluAndMul and kernel.layers.ApplyRotary

    def test_kernel_python_parses_as_python_3_9(self):
        """The Hub requires kernels to run on Python 3.9 and later."""
        import ast

        for path in (KERNEL / "torch-ext" / "llama_easy").glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))

    def test_kernel_python_imports_only_what_the_hub_allows(self):
        """Hub kernels may import the standard library, torch, triton and
        themselves (relatively) -- nothing else."""
        import ast

        allowed = set(sys.stdlib_module_names) | {"torch", "triton"}
        for path in (KERNEL / "torch-ext" / "llama_easy").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    roots = [(node.module or "").split(".")[0]]
                else:
                    continue
                assert set(roots) <= allowed, f"{path.name} imports {roots}"


# -- the ops on the PyTorch path ----------------------------------------


DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dtype", DTYPES)
class TestTheOps:
    def test_rms_norm_is_llama_rms_norm(self, kernel, dtype):
        x = torch.randn(3, 5, 80, dtype=dtype)
        w = torch.randn(80, dtype=dtype)
        hidden = x.float()
        normed = (hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-5)).to(dtype)
        out = kernel.rms_norm(x, w, 1e-5)
        assert out.dtype == dtype and out.shape == x.shape
        _close(out, w.float() * normed.float(), dtype)

    def test_rms_norm_keeps_an_fp32_weight_in_fp32(self, kernel, dtype):
        out = kernel.rms_norm(torch.randn(2, 64, dtype=dtype), torch.ones(64), 1e-6)
        assert out.dtype == torch.float32

    def test_silu_and_mul(self, kernel, dtype):
        x = torch.randn(4, 2 * 96, dtype=dtype)
        gate, up = x.float()[..., :96], x.float()[..., 96:]
        _close(kernel.silu_and_mul(x), torch.nn.functional.silu(gate) * up, dtype)

    def test_rotary_matches_transformers_on_a_transposed_q(self, kernel, dtype):
        q = torch.randn(2, 6, 4, 64, dtype=dtype).transpose(1, 2)  # [b, h, s, d], strided
        k = torch.randn(2, 6, 2, 64, dtype=dtype).transpose(1, 2)
        cos = torch.randn(2, 6, 64, dtype=dtype)
        sin = torch.randn(2, 6, 64, dtype=dtype)

        def truth(x):
            c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
            xf = x.float()
            half = xf.shape[-1] // 2
            return xf * c + torch.cat((-xf[..., half:], xf[..., :half]), -1) * s

        q_out, k_out = kernel.apply_rotary_transformers(q, k, cos, sin)
        assert q_out.shape == q.shape and k_out.shape == k.shape
        _close(q_out, truth(q), dtype, steps=4)
        _close(k_out, truth(k), dtype, steps=4)


def test_empty_inputs_give_empty_outputs(kernel):
    assert kernel.rms_norm(torch.empty(0, 64), torch.ones(64)).shape == (0, 64)
    assert kernel.silu_and_mul(torch.empty(3, 0, 8)).shape == (3, 0, 4)
    q = torch.empty(1, 2, 0, 64)
    cos = torch.empty(1, 0, 64)
    assert kernel.apply_rotary_transformers(q, q, cos, cos)[0].shape == (1, 2, 0, 64)


# -- the Triton kernels, through Triton's interpreter ------------------


_INTERPRETED = textwrap.dedent("""
    import sys
    from pathlib import Path
    import torch
    from kernels import get_local_kernel

    k = get_local_kernel(Path(sys.argv[1]), backend="cpu")
    torch.manual_seed(0)
    worst = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        eps = {torch.float32: 1e-6, torch.bfloat16: 2**-7, torch.float16: 2**-10}[dtype]
        def check(name, got, truth, steps=2.0):
            err = (got.float() - truth.float()).abs().max().item()
            limit = steps * eps * (truth.abs().max().item() or 1.0)
            assert err <= limit, (name, dtype, err, limit)
            worst[f"{name}-{dtype}"] = err

        # 80 and 96: dims that are not a power of two, so masking is exercised.
        x = torch.randn(7, 80, dtype=dtype); w = torch.randn(80, dtype=dtype)
        h = x.float(); n = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype)
        check("rms_norm", k.rms_norm(x, w, 1e-6), w.float() * n.float())

        g = torch.randn(5, 2 * 1100, dtype=dtype)   # > one 1024 block per row
        check("silu_and_mul", k.silu_and_mul(g),
              torch.nn.functional.silu(g.float()[..., :1100]) * g.float()[..., 1100:])

        q = torch.randn(2, 5, 3, 96, dtype=dtype).transpose(1, 2)
        c = torch.randn(2, 5, 96, dtype=dtype); s = torch.randn(2, 5, 96, dtype=dtype)
        qo, _ = k.apply_rotary_transformers(q, q, c, s)
        qf, cf, sf = q.float(), c.float().unsqueeze(1), s.float().unsqueeze(1)
        check("rotary", qo, qf * cf + torch.cat((-qf[..., 48:], qf[..., :48]), -1) * sf, steps=4)

    assert k.rms_norm(torch.empty(0, 64), torch.ones(64)).shape == (0, 64)
    assert k.silu_and_mul(torch.empty(0, 8)).shape == (0, 4)

    # A whole model, Triton kernels throughout.
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 128, (2, 9))
    with torch.no_grad():
        before = model(ids).logits
        k.kernelize(model)
        assert model.llama_easy_kernelized == {"RMSNorm": 5, "rotary_pos_emb": 1}
        after = model(ids).logits
    worst["model-logits"] = (after - before).abs().max().item()
    assert worst["model-logits"] < 1e-4, worst
    print("ok", worst)
""")


@pytest.mark.skipif(not (HAS_TRITON and HAS_TRANSFORMERS), reason="needs triton and transformers")
def test_the_triton_kernels_themselves(build_dir, tmp_path):
    script = tmp_path / "interpreted.py"
    script.write_text(_INTERPRETED, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(script), str(build_dir)],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False)
    assert done.returncode == 0 and "ok" in done.stdout, done.stdout[-3000:] + done.stderr[-3000:]


# -- the layers and a real model ------------------------------------------


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="needs transformers")
class TestInALlamaModel:
    @pytest.fixture
    def model(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        return LlamaForCausalLM(config).eval()

    @pytest.fixture
    def pretend_gpu(self, monkeypatch):
        """kernelize asks the GPU for its compute capability before it
        picks a kernel. There is no GPU here, so it is told 8.0; the
        layers then run on CPU tensors through their PyTorch path."""
        from kernels.layer import repos

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
        repos._find_capability.cache_clear()
        yield
        repos._find_capability.cache_clear()

    def test_the_layers_pass_kernels_own_checks(self, kernel):
        from kernels.layer.layer import _create_func_module, _validate_layer
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb

        _validate_layer(check_cls=LlamaRMSNorm, cls=kernel.layers.RMSNorm, repo="llama-easy")
        # The rotary hook is a function; kernels checks a layer against the
        # module it wraps the original function in.
        original = getattr(apply_rotary_pos_emb, "forward", apply_rotary_pos_emb)
        _validate_layer(check_cls=_create_func_module(original), cls=kernel.layers.ApplyRotary,
                        repo="llama-easy")

    def test_the_kernels_own_kernelize_is_one_call(self, model, kernel):
        """get_kernel(...).kernelize(model): no mapping, nothing but torch."""
        ids = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            before = model(ids).logits
        assert kernel.kernelize(model) is model
        # Two norms per layer and the final one; one shared rotary function.
        assert model.llama_easy_kernelized == {"RMSNorm": 5, "rotary_pos_emb": 1}
        norm = model.model.layers[1].post_attention_layernorm
        assert norm.forward.__func__ is kernel.layers.RMSNorm.forward
        rotary = model.model.layers[0].self_attn._kernel_funcs["rotary_pos_emb"]
        assert rotary.forward.__func__ is kernel.layers.ApplyRotary.forward
        with torch.no_grad():
            after = model(ids).logits
        torch.testing.assert_close(after, before)

    def test_kernelize_says_when_it_found_nothing(self, kernel):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        kernel.kernelize(model)
        assert model.llama_easy_kernelized == {"RMSNorm": 0, "rotary_pos_emb": 0}

    def test_transformers_kernel_config(self, model, build_dir, pretend_gpu, monkeypatch):
        """from_pretrained(..., kernel_config=kernel_config()) -- here as the
        set_use_kernels call it makes, with Transformers told the model is
        on a GPU, since it refuses a kernel_config for a CPU model."""
        import transformers.integrations.hub_kernels as transformers_hub
        import transformers.utils.kernel_config as transformers_config

        monkeypatch.setattr(transformers_config, "infer_device", lambda model: "cuda")
        monkeypatch.setattr(transformers_hub, "get_device_type", lambda device: "cuda")
        ids = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            before = model(ids).logits

        model.set_use_kernels(True, hub_kernels.kernel_config(local_path=build_dir))

        norm = model.model.layers[0].input_layernorm.forward.__func__
        rotary = model.model.layers[0].self_attn._kernel_funcs["rotary_pos_emb"].forward.__func__
        assert norm.__qualname__ == "RMSNorm.forward" and "llama_easy" in norm.__module__
        assert rotary.__qualname__ == "ApplyRotary.forward" and "llama_easy" in rotary.__module__
        with torch.no_grad():
            after = model(ids).logits
        torch.testing.assert_close(after, before)

    def test_the_hub_kernel_config_names_this_repository(self):
        config = hub_kernels.kernel_config()
        assert config.kernel_mapping == {
            "RMSNorm": ("ray0rf1re/llama-easy:RMSNorm", {"version": 1, "trust_remote_code": True}),
            "rotary_pos_emb": ("ray0rf1re/llama-easy:ApplyRotary", {"version": 1, "trust_remote_code": True}),
        }

    def test_from_pretrained_on_the_cpu_loads_the_model_unchanged(self, model, build_dir, tmp_path):
        model.save_pretrained(tmp_path / "tiny")
        loaded = hub_kernels.from_pretrained(tmp_path / "tiny", local_path=build_dir)
        assert type(loaded).__name__ == "LlamaForCausalLM"
        norm = loaded.model.layers[0].input_layernorm
        assert norm.forward.__func__ is type(norm).forward

    def test_kernelize_swaps_the_llama_layers_and_keeps_the_logits(self, model, build_dir, pretend_gpu):
        ids = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            before = model(ids).logits

        hub_kernels.kernelize_llama(model, device="cuda", local_path=build_dir)

        norm = model.model.layers[0].input_layernorm
        assert norm.forward.__func__.__module__.endswith(".layers"), norm.forward
        assert "llama_easy" in norm.forward.__func__.__module__
        with torch.no_grad():
            after = model(ids).logits
        torch.testing.assert_close(after, before)

    def test_training_keeps_the_models_own_forward(self, model, build_dir, pretend_gpu):
        """No backward pass in the kernel, so a training model is left alone."""
        original = type(model.model.layers[0].input_layernorm).forward
        hub_kernels.kernelize_llama(model, mode="training", device="cuda", local_path=build_dir)
        assert model.model.layers[0].input_layernorm.forward.__func__ is original

    def test_training_without_fallback_says_why(self, model, build_dir, pretend_gpu):
        with pytest.raises(ValueError, match="does not support"):
            hub_kernels.kernelize_llama(model, mode="training", device="cuda",
                                        local_path=build_dir, use_fallback=False)


# -- the loader -----------------------------------------------------------


class TestTheLoader:
    def test_trust_is_given_to_this_repository_only(self):
        mapping = hub_kernels.kernel_mapping()
        assert set(mapping) == {"RMSNorm", "rotary_pos_emb", "SiluAndMul"}
        for per_device in mapping.values():
            assert set(per_device) == set(hub_kernels.DEVICES)
            for repo in per_device.values():
                assert repo._trust_remote_code == [hub_kernels.REPO_ID]

    def test_kernelize_applies_only_this_kernel(self, monkeypatch):
        """Not the global mapping too: Transformers registers its own, and
        inheriting it would download kernels for hooks nobody asked for."""
        seen = {}

        class Recorder:
            def __init__(self, mapping, inherit_mapping=True):
                seen["inherit"] = inherit_mapping

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(kernels, "use_kernel_mapping", Recorder)
        monkeypatch.setattr(kernels, "kernelize", lambda model, **kw: model)
        hub_kernels.kernelize_llama(torch.nn.Linear(2, 2), device="cuda")
        assert seen == {"inherit": False}

    def test_the_hub_mapping_pins_the_major_version(self):
        repo = hub_kernels.kernel_mapping()["RMSNorm"]["cuda"]
        assert repo._version == hub_kernels.VERSION and repo._revision is None

    @pytest.mark.parametrize(("text", "flags"), [
        ("inference", {"INFERENCE"}),
        ("training", {"TRAINING"}),
        ("inference+compile", {"INFERENCE", "TORCH_COMPILE"}),
        ("Training + Compile", {"TRAINING", "TORCH_COMPILE"}),
    ])
    def test_modes(self, text, flags):
        mode = hub_kernels._mode(kernels, text)
        assert {flag for flag in ("INFERENCE", "TRAINING", "TORCH_COMPILE")
                if getattr(kernels.Mode, flag) in mode} == flags

    def test_a_nonsense_mode_is_refused(self):
        with pytest.raises(ValueError, match="inference"):
            hub_kernels._mode(kernels, "fast")

    def test_without_kernels_installed_it_says_what_to_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "kernels", None)
        with pytest.raises(hub_kernels.KernelsUnavailable, match="pip install kernels"):
            hub_kernels.load()
