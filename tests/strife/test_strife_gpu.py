"""Strife's GPU offload: placement decisions, and the device path itself.

There is no GPU where these run. The device path is exercised on PyTorch's
CPU device, which runs the same code; the CUDA decisions are exercised by
standing in for ``torch.cuda``.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from hypernix.strife import Engine, SamplerConfig, gpu  # noqa: E402
from hypernix.strife.cli import cli_main  # noqa: E402
from hypernix.strife.fixtures import make_model  # noqa: E402
from hypernix.strife.inspect import read_gguf  # noqa: E402

GREEDY = SamplerConfig(temperature=0)


def _fake_cuda(monkeypatch, *, capability=(8, 9), arch=("sm_75", "sm_80", "sm_86", "sm_89"),
               free=8 << 30, total=8 << 30, name="Test GPU", count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i=0: (free, total))


class TestPlacementDecisions:
    def test_layers_fit_in_free_memory_less_the_reserve(self):
        per_layer = {0: 100, 1: 100, 2: 100, 3: 100}
        assert gpu.layers_that_fit(per_layer, 4, 350, reserve=0) == (3, 300)
        assert gpu.layers_that_fit(per_layer, 2, 10_000, reserve=0) == (2, 200)
        assert gpu.layers_that_fit(per_layer, gpu.ALL_LAYERS, 10_000, reserve=0) == (4, 400)
        assert gpu.layers_that_fit(per_layer, 4, 350, reserve=300) == (0, 0)
        assert gpu.layers_that_fit(per_layer, 9, 10_000, reserve=0) == (4, 400)

    def test_layer_bytes_counts_matrices_only(self, tmp_path):
        model = make_model(tmp_path / "m.gguf", arch="qwen2", n_layers=3)
        gguf, _ = read_gguf(model["path"])
        sizes = gpu.layer_bytes(gguf)
        assert sorted(sizes) == [0, 1, 2] and len(set(sizes.values())) == 1
        assert gpu.layer_bytes(gguf, bytes_per_weight=4)[0] == 2 * sizes[0]

    def test_no_cuda_is_a_reason_not_an_error(self):
        choice = gpu.choose_device()
        if torch.cuda.is_available():
            pytest.skip("this machine has a GPU")
        assert not choice.usable and "no CUDA device" in choice.reasons[0]

    def test_a_card_pytorch_was_not_built_for_is_refused_by_name(self, monkeypatch):
        _fake_cuda(monkeypatch, capability=(6, 1), name="GeForce GTX 1080")
        choice = gpu.choose_device()
        assert not choice.usable
        assert "GTX 1080" in choice.reasons[0] and "6.1" in choice.reasons[0]

    def test_a_supported_card_reports_free_not_total(self, monkeypatch):
        _fake_cuda(monkeypatch, free=3 << 30, total=8 << 30)
        choice = gpu.choose_device("cuda:0")
        assert choice.usable and choice.free_bytes == 3 << 30 and choice.total_bytes == 8 << 30
        assert not gpu.choose_device("cuda:3").usable


class TestTheDevicePath:
    @pytest.mark.parametrize("arch", ["qwen2", "qwen35", "lfm3", "glm4moe"])
    def test_offloaded_layers_give_the_same_tokens(self, tmp_path, arch):
        model = make_model(tmp_path / f"{arch}.gguf", arch=arch, seed=3)
        ids = list(range(10, 30))
        with Engine(model["path"], n_ctx=96, kernels="off") as cpu:
            want = cpu.generate(prompt_ids=ids, max_tokens=10, sampling=GREEDY).tokens
            layers = cpu.model.plan.n_layers
        with Engine(model["path"], n_ctx=96, kernels="off", gpu_layers=1,
                    gpu_device="cpu") as part:
            got = part.generate(prompt_ids=ids, max_tokens=10, sampling=GREEDY).tokens
            placement = part.placement
            assert part.weights.device.stats()["matmuls"] > 0
        assert got == want
        assert placement["gpu_layers_used"] == 1 and placement["cpu_layers"] == layers - 1
        assert placement["backend"].endswith("+gpu") and placement["gpu_bytes"] > 0
        assert placement["peak_vram_bytes"] == placement["gpu_bytes"]

    def test_all_layers_and_more_than_there_are(self, tmp_path):
        model = make_model(tmp_path / "m.gguf", arch="qwen2", n_layers=3)
        for wanted in (-1, 99):
            with Engine(model["path"], n_ctx=64, gpu_layers=wanted, gpu_device="cpu") as engine:
                assert engine.placement["gpu_layers_used"] == 3
                assert engine.placement["cpu_layers"] == 0
                assert engine.generate(prompt_ids=[5, 6, 7], max_tokens=3).tokens

    def test_a_card_too_small_is_split_and_says_so(self, tmp_path, monkeypatch):
        model = make_model(tmp_path / "m.gguf", arch="qwen2", n_layers=4)
        gguf, _ = read_gguf(model["path"])
        one = gpu.layer_bytes(gguf, bytes_per_weight=4)[0]
        real = gpu.choose_device

        def small(wanted=None):
            choice = real("cpu")
            choice.free_bytes = gpu.RESERVE_BYTES + 2 * one + one // 2    # room for two
            return choice

        monkeypatch.setattr(gpu, "choose_device", small)
        with Engine(model["path"], n_ctx=64, gpu_layers=4, gpu_device="cpu") as engine:
            placement = engine.placement
            assert placement["gpu_layers_used"] == 2 and placement["cpu_layers"] == 2
            assert any("2 of the 4 layers" in r for r in placement["fallback_reasons"])
            assert engine.generate(prompt_ids=[5, 6, 7], max_tokens=3).tokens

    def test_a_request_without_a_gpu_runs_on_the_cpu_and_says_why(self, tmp_path):
        if torch.cuda.is_available():
            pytest.skip("this machine has a GPU")
        model = make_model(tmp_path / "m.gguf", arch="qwen2")
        with Engine(model["path"], n_ctx=64, gpu_layers=2) as engine:
            assert engine.placement["gpu_layers_used"] == 0
            assert engine.weights.device is None
            assert engine.placement["fallback_reasons"]
            assert engine.generate(prompt_ids=[5, 6, 7], max_tokens=3).tokens

    def test_diag_plans_the_split_without_loading(self, tmp_path, capsys):
        model = make_model(tmp_path / "m.gguf", arch="qwen2", n_layers=3)
        assert cli_main(["diag", str(model["path"]), "--gpu-layers", "2", "--gpu-device", "cpu",
                         "--json"]) == 0
        placement = json.loads(capsys.readouterr().out)["placement"]
        assert placement["gpu_layers_used"] == 2 and placement["gpu_layers_requested"] == 2


@pytest.mark.parametrize("command", ["inspect", "run", "chat", "diag", "types", "selftest",
                                     "build", "kernels", "benchmark"])
def test_every_subcommand_has_working_help(command, capsys):
    """A bare % in a help string makes argparse raise when --help is printed."""
    with pytest.raises(SystemExit) as done:
        cli_main([command, "--help"])
    assert done.value.code == 0 and "usage:" in capsys.readouterr().out
