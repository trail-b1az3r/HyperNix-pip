"""CPU offloading for the HyperNix runners (0.72.8rc1).

A native HyperNix model used to sit wholly on one device, so one bigger
than the card ran on the CPU; it is split by layer now, as llama.cpp's
``-ngl`` splits a GGUF. And the llama.cpp runners can keep a
mixture-of-experts model's experts, or the KV cache, in RAM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypernix.hyperlink.managed import Placement, _with_offload, fork_argv
from hypernix.hyperlink.offload import Offload, fit_layers, split_layers

torch = pytest.importorskip("torch")


class TestTheLlamaCppFlags:
    @pytest.mark.parametrize("offload,argv", [
        (Offload(), []),
        (Offload(cpu_moe=-1), ["--cpu-moe"]),
        (Offload(cpu_moe=12), ["--n-cpu-moe", "12"]),
        (Offload(kv_on_cpu=True), ["--no-kv-offload"]),
        (Offload(cpu_moe=-1, kv_on_cpu=True), ["--cpu-moe", "--no-kv-offload"]),
    ])
    def test_each_choice_is_its_flag(self, offload, argv):
        assert offload.llama_argv() == argv

    def test_koboldcpp_refuses_rather_than_ignoring_it(self):
        with pytest.raises(ValueError, match="KoboldCpp"):
            Offload(cpu_moe=-1).llama_argv("kobold")
        assert Offload().llama_argv("kobold") == []

    def test_the_server_command_carries_them(self, tmp_path):
        from hypernix.quant.runtime_bridge import Build, serve_argv

        server = tmp_path / "llama-server"
        server.write_text("")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")
        build = Build(root=tmp_path, bin_dir=tmp_path, server=server)
        argv = serve_argv(build, model, gpu_layers=99, extra=Offload(cpu_moe=-1).llama_argv())
        assert argv[-1] == "--cpu-moe" and "-ngl" in argv
        assert fork_argv("ik", server, model, host="127.0.0.1", port=1, gpu_layers=99, context=0,
                         alias="m", extra=["--no-kv-offload"])[-1] == "--no-kv-offload"

    def test_the_plan_says_what_is_in_ram(self):
        placed = _with_offload(Placement(gpu_layers=32, total_layers=32, reason="fits"),
                               Offload(cpu_moe=-1, kv_on_cpu=True))
        assert placed.offload == "every expert in RAM, the KV cache in RAM"
        assert placed.reason.startswith("fits; ") and placed.to_dict()["offload"]
        assert _with_offload(Placement(), Offload()) == Placement()


def _brewer(n_layers=4):
    from hypernix.training.brewer import BrewerConfig, BrewerModel

    torch.manual_seed(0)
    return BrewerModel(BrewerConfig(vocab_size=64, n_layers=n_layers, n_heads=4, n_kv_heads=2, d_model=32,
                                    d_ff=64, max_seq_len=32, use_sliding_window=False, dropout=0.0)).eval()


class TestTheNativeSplit:
    def test_a_split_model_answers_exactly_as_before(self):
        whole = _brewer()
        split = _brewer()
        ids = torch.randint(0, 64, (2, 9))
        with torch.no_grad():
            expected = whole(ids)
            placed = split_layers(split, 2, device="cpu")
            got = split(ids)
        assert placed["gpu_layers"] == 2 and placed["total_layers"] == 4
        assert torch.allclose(got, expected, atol=1e-6)

    def test_the_blocks_go_where_they_are_told(self):
        model = _brewer()
        placed = split_layers(model, 0, device="cpu")
        assert placed["gpu_layers"] == 0 and placed["device"] == "cpu"
        assert placed["cpu_bytes"] > 0 and placed["gpu_bytes"] == 0
        assert split_layers(_brewer(), 99, device="cpu")["gpu_layers"] == 4  # clamped to the model

    def test_as_many_layers_as_fit(self):
        model = _brewer()
        block = sum(p.numel() * p.element_size() for p in model.blocks[0].parameters())
        fixed = sum(p.numel() * p.element_size() for p in model.embed.parameters()) + \
            sum(p.numel() * p.element_size() for p in model.norm_out.parameters())
        assert fit_layers(model, int((fixed + 2.5 * block) / 0.85)) == 2
        assert fit_layers(model, 10**12) == 4
        assert fit_layers(model, 0) == 0


class TestTheRunnerAsksForIt:
    def test_the_native_server_gets_the_split(self, tmp_path, monkeypatch):
        """The PyTorch server is started with --gpu-layers, and the plan says
        how many layers are on the GPU rather than "all of them"."""
        import json
        import subprocess

        from hypernix.hyperlink import managed

        folder = tmp_path / "brew"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"n_layers": 24, "max_seq_len": 2048, "name": "brew"}))
        started = []

        class _Process:
            pid = 1

            def poll(self):
                return None

        monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: started.append(argv) or _Process())
        runner = managed.ManagedRunner()
        monkeypatch.setattr(runner, "_wait_until_ready", lambda process, timeout: True)
        record = runner._load_brewed(Path(folder), model_id="", backend="auto", timeout=1, gpu_layers=10)
        assert started[0][-2:] == ["--gpu-layers", "10"]
        assert record.placement.gpu_layers == 10 and record.placement.total_layers == 24
        assert "10 of 24 layers on the GPU" in record.placement.reason

    def test_auto_remembers_the_offload(self):
        from hypernix.hyperlink.runner_history import AutoChoice, LoadRecord

        record = LoadRecord.from_dict({"model_id": "m", "cpu_moe": -1, "kv_on_cpu": True})
        assert record.cpu_moe == -1 and record.kv_on_cpu
        choice = AutoChoice(model_id="m", backend="auto", gpu_layers=None, context_length=None,
                            total_layers=None, why="", cpu_moe=-1, kv_on_cpu=True)
        assert choice.payload()["cpu_moe"] == -1 and choice.payload()["kv_on_cpu"] is True

    def test_the_cli_flags(self):
        from hypernix.t1api.runner_cli import _load_payload, build_parser

        args = build_parser().parse_args(["load", "m", "--cpu-moe", "--kv-on-cpu"])
        assert _load_payload(args, "m")["cpu_moe"] == -1 and _load_payload(args, "m")["kv_on_cpu"]
        args = build_parser().parse_args(["start", "m", "--cpu-moe", "8"])
        assert _load_payload(args, "m")["cpu_moe"] == 8
        assert "cpu_moe" not in _load_payload(build_parser().parse_args(["load", "m"]), "m")
