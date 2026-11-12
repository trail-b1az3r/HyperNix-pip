"""Strife's native kernels: built here, checked against the Python decoders.

Everything in this file compiles ``strife/native/strife_kernels.c`` with the
machine's own compiler into a temporary ``HYPERNIX_HOME``; without a C
compiler the file is skipped, which is also what Strife does (numpy).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from hypernix.quant import llamaquants
from hypernix.strife import Engine, SamplerConfig, blasctl, kernels
from hypernix.strife.errors import StrifeError
from hypernix.strife.fixtures import make_model
from hypernix.strife.quants import decode as reference_decode

pytestmark = pytest.mark.skipif(
    shutil.which(os.environ.get("CC") or "cc") is None and shutil.which("gcc") is None
    and shutil.which("clang") is None, reason="no C compiler")

QUANTS = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]


def _cpu_has(flag: str) -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            return any(flag in line.split() for line in handle if line.startswith("flags"))
    except OSError:
        return False


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    """One kernel build for the whole module, in its own HYPERNIX_HOME."""
    base = tmp_path_factory.mktemp("kernels-home")
    saved = {k: os.environ.get(k) for k in ("HYPERNIX_HOME", kernels.ENV)}
    os.environ["HYPERNIX_HOME"] = str(base)
    os.environ.pop(kernels.ENV, None)
    kernels.reset()
    yield base
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    kernels.reset()


@pytest.fixture(scope="module")
def built(home):
    return kernels.build()


def _encoded(name: str, rows: int, cols: int, seed: int = 0) -> tuple[bytes, int]:
    rng = np.random.default_rng(seed)
    values = (rng.standard_normal(rows * cols) * 0.3).astype(np.float32)
    fmt = llamaquants.FORMATS[name]
    return llamaquants.quantize_array(values, name), fmt.ggml_type


class TestBuild:
    def test_every_type_is_verified(self, built):
        failed = {n: e for n, e in built.manifest["verification"].items() if not e["ok"]}
        assert not failed, failed
        assert {"Q4_K", "Q6_K", "Q8_0", "F16", "F32"} <= set(built.describe()["types"])
        assert built.abi == kernels.ABI

    def test_the_library_is_owner_only_and_reused(self, built):
        mode = built.path.stat().st_mode & 0o777
        assert mode & 0o022 == 0
        again = kernels.build()                      # already built: no compile
        assert again.path == built.path
        assert kernels.load() is not None and kernels.status()["active"]

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
    def test_a_library_others_can_write_is_not_loaded(self, built):
        built.path.chmod(0o777)
        kernels.reset()
        try:
            assert kernels.load() is None
            assert "strife.kernels.unsafe" in kernels.status()["reason"]
        finally:
            built.path.chmod(0o700)
            kernels.reset()
        assert kernels.load() is not None

    def test_off_and_missing_compiler(self, built, monkeypatch):
        monkeypatch.setenv(kernels.ENV, "off")
        kernels.reset()
        assert kernels.load() is None
        monkeypatch.delenv(kernels.ENV)
        kernels.reset()
        monkeypatch.setattr(kernels, "_compiler", lambda: None)
        with pytest.raises(StrifeError) as none:
            kernels.build(force=True)
        assert none.value.code == "strife.kernels.no_compiler"
        kernels.reset()

    @pytest.mark.parametrize("flags", [[], ["-mavx2", "-mfma", "-mf16c"]],
                             ids=["plain-c", "avx2"])
    def test_the_other_code_paths_verify_too(self, built, flags):
        """The build above took this CPU's widest path; these take the narrower ones."""
        if flags and not (_cpu_has("avx2") and _cpu_has("fma")):
            pytest.skip("this CPU has no AVX2")
        other = kernels.build(flags=flags)
        failed = {n: e for n, e in other.manifest["verification"].items() if not e["ok"]}
        assert not failed, failed
        assert other.simd == ("avx2" if flags else other.simd)
        assert other.path != built.path
        kernels.reset()
        assert kernels.load().path == built.path     # a custom build is never the default


class TestAgainstThePythonDecoders:
    @pytest.mark.parametrize("name", QUANTS)
    def test_decode_is_what_the_python_decoder_gives(self, built, name):
        raw, ggml_type = _encoded(name, 24, 512)
        want = reference_decode(raw, ggml_type)
        got = built.decode(raw, ggml_type, 24 * 512)
        np.testing.assert_allclose(got, want, rtol=0, atol=2e-6 * float(np.abs(want).max()))

    @pytest.mark.parametrize("name", QUANTS)
    @pytest.mark.parametrize("n", [1, 3, 11, 40])
    def test_matmul_matches_the_decoded_weight(self, built, name, n):
        rows, cols = 37, 768                      # rows: not a whole tile; cols: three blocks of 256
        raw, ggml_type = _encoded(name, rows, cols, seed=n)
        weight = reference_decode(raw, ggml_type).reshape(rows, cols)
        x = np.random.default_rng(n).standard_normal((n, cols)).astype(np.float32)
        out = built.matmul(raw, ggml_type, rows, cols, x)
        want = x.astype(np.float64) @ weight.astype(np.float64).T
        assert out.shape == (n, rows) and out.dtype == np.float32
        np.testing.assert_allclose(out, want, rtol=0, atol=1e-5 * float(np.abs(want).max()))

    @pytest.mark.parametrize(("ggml_type", "dtype"), [(0, "<f4"), (1, "<f2")])
    @pytest.mark.parametrize("cols", [1, 31, 257, 300])
    def test_plain_types_take_any_row_length(self, built, ggml_type, dtype, cols):
        rng = np.random.default_rng(cols)
        weight = rng.standard_normal((19, cols)).astype(dtype)
        x = rng.standard_normal((9, cols)).astype(np.float32)
        out = built.matmul(weight.tobytes(), ggml_type, 19, cols, x)
        want = x.astype(np.float64) @ weight.astype(np.float64).T
        np.testing.assert_allclose(out, want, rtol=0, atol=1e-5 * float(np.abs(want).max()) + 1e-6)

    def test_threads_do_not_change_the_answer(self, built):
        raw, ggml_type = _encoded("Q4_K", 600, 1024)       # large enough to go parallel
        x = np.random.default_rng(1).standard_normal((2, 1024)).astype(np.float32)
        one = built.matmul(raw, ggml_type, 600, 1024, x, threads=1)
        many = built.matmul(raw, ggml_type, 600, 1024, x, threads=4)
        np.testing.assert_array_equal(one, many)

    def test_bad_requests_are_refused(self, built):
        raw, ggml_type = _encoded("Q4_K", 2, 256)
        with pytest.raises(StrifeError):
            built.decode(raw, ggml_type, 100)              # not a whole block
        with pytest.raises(StrifeError):
            built.decode(raw[:10], ggml_type, 512)         # shorter than claimed
        with pytest.raises(StrifeError):
            built.matmul(raw, ggml_type, 2, 256, np.zeros((1, 128), dtype=np.float32))
        with pytest.raises(StrifeError):
            built.decode(raw, 9999, 256)


class TestInTheEngine:
    @pytest.mark.parametrize("arch", ["qwen2", "qwen35", "lfm3", "glm4moe"])
    def test_same_tokens_and_logits_as_numpy(self, built, tmp_path, arch):
        model = make_model(tmp_path / f"{arch}.gguf", arch=arch, seed=5)
        with Engine(model["path"], n_ctx=96, kernels="off") as slow:
            ids = slow.tokenizer.encode("the quick brown fox jumps over")
            want = slow.generate(prompt_ids=ids, max_tokens=12, sampling=SamplerConfig(temperature=0))
            assert not slow.weights.native_matmul
        with Engine(model["path"], n_ctx=96) as fast:
            assert fast.weights.native_matmul, fast.kernels_note
            assert fast.status()["placement"]["backend"] == "cpu-native"
            got = fast.generate(prompt_ids=ids, max_tokens=12, sampling=SamplerConfig(temperature=0))
            assert fast.weights.native_matmuls > 0
            # Nothing the size of a weight matrix was decoded into the cache.
            assert fast.status()["weights"]["cached_bytes"] < os.path.getsize(model["path"]) // 4
        assert got.tokens == want.tokens

    @pytest.mark.parametrize("quant", ["Q4_K", "Q6_K", "Q8_0", "Q3_K"])
    def test_quantised_models_agree_with_numpy(self, built, tmp_path, quant):
        model = make_model(tmp_path / "q.gguf", arch="qwen2", quant=quant, n_embd=256, n_ff=512,
                           n_layers=2, seed=9)
        ids = list(range(20, 60))
        with Engine(model["path"], n_ctx=96, kernels="off") as slow:
            want = slow.generate(prompt_ids=ids, max_tokens=8, sampling=SamplerConfig(temperature=0))
        with Engine(model["path"], n_ctx=96) as fast:
            assert fast.weights.native_matmul
            got = fast.generate(prompt_ids=ids, max_tokens=8, sampling=SamplerConfig(temperature=0))
        assert got.tokens == want.tokens

    def test_blas_threads_are_held_then_given_back(self, built, tmp_path, monkeypatch):
        if not blasctl.controllable() or blasctl.get_threads() is None:
            pytest.skip("BLAS threads cannot be read here")
        from hypernix.strife import engine as engine_module

        # Engines other tests left open still hold BLAS; count from zero here.
        monkeypatch.setattr(engine_module, "_blas_holders", 0)
        model = make_model(tmp_path / "b.gguf", arch="qwen2", seed=1)
        blasctl.set_threads(2)
        with Engine(model["path"], n_ctx=64) as first:
            assert first.weights.native_matmul and blasctl.get_threads() == 1
            with Engine(model["path"], n_ctx=64):
                assert blasctl.get_threads() == 1
            assert blasctl.get_threads() == 1          # the first engine still needs it
        assert blasctl.get_threads() == 2

    def test_types_without_a_kernel_keep_the_numpy_path(self, built, tmp_path):
        from hypernix.strife.fixtures import QUANT_TYPES

        hnx = next((q for q in QUANT_TYPES if q.startswith("HNX")), None)
        if hnx is None:
            pytest.skip("no HyperNix tensor type in the fixtures")
        model = make_model(tmp_path / "h.gguf", arch="qwen2", quant=hnx, n_embd=256, n_ff=512,
                           n_layers=1, seed=2)
        with Engine(model["path"], n_ctx=64) as engine:
            assert not engine.weights.native_matmul
            assert "decode only" in engine.kernels_note
            result = engine.generate(prompt_ids=list(range(10, 20)), max_tokens=3)
            assert len(result.tokens) > 0


def test_cli_status_reports_the_build(built, capsys):
    import json

    from hypernix.strife.cli import cli_main

    assert cli_main(["kernels", "status", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["active"] and "Q4_K" in info["types"]
    assert cli_main(["kernels", "verify", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
