"""The multi-llama runner (0.72.8): a GGUF that needs a llama.cpp fork
is served by that fork, and the forks' types are known by name."""
from __future__ import annotations

from pathlib import Path

import pytest

from hypernix.hyperlink import managed
from hypernix.quant import forktypes
from hypernix.quant.ggufcheck import LLAMA_CPP_REGISTERED_TYPES


class TestTheTypes:
    def test_ik_types_by_name(self):
        assert forktypes.fork_type_name(137) == "IQ2_K" and forktypes.fork_type_name(155) == "IQ4_KT"
        assert forktypes.fork_type_name(12) == ""

    def test_the_ids_both_use_are_exactly_the_old_hnx_ones(self):
        """HyperNix's newer types were placed where ik_llama.cpp has nothing."""
        assert forktypes.AMBIGUOUS == {202, 206, 208, 210, 211}
        assert not {224, 225} & set(forktypes.IK_TYPES)
        assert {224, 225} <= set(LLAMA_CPP_REGISTERED_TYPES)

    @pytest.mark.parametrize("types, metadata, runtime", [
        ({2, 12, 14}, {}, "stock"),
        ({12, 137}, {}, "ik"),
        ({210, 14}, {"hypernix.needs": "ggml-hnx"}, "hnx"),
        ({210, 14}, {}, "hnx"),                     # shared ids alone stay HyperNix's
        ({210, 214}, {}, "ik"),                     # ...unless an ik-only type is there too
        ({204, 14}, {}, "hnx"),                     # 204 is only HyperNix's
        ({12}, {"general.architecture": "nanbeige"}, "nanbeige"),
        ({137, 204}, {}, ""),                       # nothing loads both
        ({500}, {}, ""),
    ])
    def test_which_runtime(self, types, metadata, runtime):
        assert forktypes.which_runtime(types, metadata).runtime == runtime


class TestTheBackends:
    def test_they_are_offered(self):
        assert "multilama" in managed.BACKENDS
        assert {f"multilama-{f}" for f in ("ik", "prismml", "kobold", "nanbeige")} <= set(managed.BACKENDS)

    @pytest.mark.parametrize("backend, runtime, fork", [
        ("multilama-kobold", "stock", "kobold"),
        ("multilama", "stock", "ik"),
        ("multilama", "nanbeige", "nanbeige"),
        ("auto", "ik", "ik"),
        ("auto", "stock", ""),
        ("cuda", "hnx", ""),
    ])
    def test_which_fork(self, backend, runtime, fork):
        assert managed._fork_for(backend, forktypes.Verdict(runtime)) == fork

    def test_a_fixed_backend_cannot_load_a_fork_model(self):
        with pytest.raises(managed.ManagedError, match="multilama"):
            managed._fork_for("cuda", forktypes.Verdict("ik", ("IQ2_K",), "IQ2_K is ik_llama.cpp's"))

    def test_the_command_lines(self, tmp_path):
        model = tmp_path / "m.gguf"
        ik = managed.fork_argv("ik", "/bin/llama-server", model, host="127.0.0.1", port=9,
                               gpu_layers=20, context=4096, alias="m")
        assert ik == ["/bin/llama-server", "-m", str(model), "--host", "127.0.0.1", "--port", "9",
                      "-ngl", "20", "-c", "4096", "--alias", "m"]
        kobold = managed.fork_argv("kobold", "/bin/koboldcpp", model, host="127.0.0.1", port=9,
                                   gpu_layers=20, context=4096, alias="m")
        assert kobold == ["/bin/koboldcpp", "--model", str(model), "--host", "127.0.0.1", "--port", "9",
                          "--gpulayers", "20", "--contextsize", "4096"]


def _gguf(path: Path, ggml_type: int, metadata: dict | None = None) -> Path:
    """A one-tensor GGUF carrying *ggml_type*, which no writer here can
    produce for a fork's type: written as F32, then the id swapped."""
    import struct

    from hypernix.quant.gguf import GGMLType, GGUFWriter

    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    for key, value in (metadata or {}).items():
        writer.set_metadata(key, value)
    name = "blk.0.attn_q.weight"
    writer.add_tensor(name, (64,), int(GGMLType.F32))
    writer.write(lambda tensor: b"\0" * 256)
    raw = bytearray(path.read_bytes())
    at = raw.index(name.encode()) + len(name) + 4 + 8      # past the name, n_dims and one dim
    raw[at:at + 4] = struct.pack("<I", ggml_type)
    path.write_bytes(bytes(raw))
    return path


class TestLoading:
    def test_an_ik_model_goes_to_ik_llama(self, tmp_path, monkeypatch):
        from hypernix.models import multilama

        launched = {}

        class Proc:
            pid = 4242

            def poll(self):
                return None

        monkeypatch.setattr(multilama, "ensure_binary", lambda backend, quiet=False: tmp_path / "llama-server")
        monkeypatch.setattr(managed.subprocess, "Popen", lambda argv, **kw: launched.setdefault("argv", argv) and Proc())
        monkeypatch.setattr(managed.ManagedRunner, "_wait_until_ready", lambda self, p, t: True)
        runner = managed.ManagedRunner(port=18555)
        model = _gguf(tmp_path / "iq2k.gguf", 137)
        record = runner.load(model, model_id="iq2k", gpu_layers=0)
        assert launched["argv"][0] == str(tmp_path / "llama-server") and record.model_id == "iq2k"

    def test_a_missing_fork_says_which_and_why(self, tmp_path, monkeypatch):
        from hypernix.models import multilama

        def missing(backend, quiet=False):
            raise multilama.MultiLlamaError("ML-FETCH-001", "no release asset")

        monkeypatch.setattr(multilama, "ensure_binary", missing)
        runner = managed.ManagedRunner(port=18556)
        with pytest.raises(managed.ManagedError, match="needs the ik llama.cpp fork.*IQ2_K"):
            runner.load(_gguf(tmp_path / "iq2k.gguf", 137), gpu_layers=0)


class TestTheCheckerNamesThem:
    def test_a_fork_type_is_named_and_routed(self, tmp_path):
        from hypernix.quant.ggufcheck import check_gguf

        report = check_gguf(_gguf(tmp_path / "iq2k.gguf", 137))
        assert "IQ2_K (ik_llama.cpp)" in report.types
        assert report.runtime == "ik" and "multilama" in report.describe()
        assert report.as_dict()["runtime"] == "ik"
