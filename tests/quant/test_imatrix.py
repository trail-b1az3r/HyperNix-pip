"""The importance matrix: measured from activations, not from weights.

The temptation here is enormous and it is wrong. An imatrix looks like
something you could derive from a weight tensor — it is one number per
input channel, it lives next to the weights, it is used at quantisation
time — and deriving one would mean nobody has to run a model. But it is
a statistic of the *activations*: two models with identical weights and
different training data want different imatrices, and a weight-derived
number is not an approximation of that, it is a different quantity
wearing its name.

So these tests are mostly about two things. That the number really comes
from the forward pass, and that the file it lands in is the one
llama.cpp writes — because an imatrix that only HyperNix can read is
half an imatrix, and the point of doing what upstream does is that the
numbers mean the same thing in both directions.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hypernix.quant.imatrix import (
    Imatrix,
    ImatrixEntry,
    ImatrixError,
    collect,
    collect_from_pretrained,
    expand_for_tensor,
    gguf_tensor_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Tokenizer:
    """Bytes as tokens. Enough to drive a forward pass deterministically."""

    def encode(self, text: str) -> list[int]:
        return [b % 64 for b in text.encode("utf-8")]


class _TinyModel(torch.nn.Module):
    """Two blocks with the names a Llama checkpoint uses.

    Named to match, because the naming is half of what is being tested:
    an imatrix keyed by torch's own module names looks right, loads
    fine, and matches no tensor in any GGUF.
    """

    def __init__(self, width: int = 32, vocab: int = 64, layers: int = 2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, width)
        self.model.layers = torch.nn.ModuleList()
        for _ in range(layers):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.q_proj = torch.nn.Linear(width, width, bias=False)
            block.self_attn.k_proj = torch.nn.Linear(width, width, bias=False)
            block.self_attn.v_proj = torch.nn.Linear(width, width, bias=False)
            block.self_attn.o_proj = torch.nn.Linear(width, width, bias=False)
            block.mlp = torch.nn.Module()
            block.mlp.gate_proj = torch.nn.Linear(width, width * 2, bias=False)
            block.mlp.up_proj = torch.nn.Linear(width, width * 2, bias=False)
            block.mlp.down_proj = torch.nn.Linear(width * 2, width, bias=False)
            self.model.layers.append(block)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        hidden = self.model.embed_tokens(ids)
        for block in self.model.layers:
            attention = block.self_attn.o_proj(
                block.self_attn.q_proj(hidden)
                + block.self_attn.k_proj(hidden)
                + block.self_attn.v_proj(hidden)
            )
            hidden = hidden + attention
            hidden = hidden + block.mlp.down_proj(
                block.mlp.gate_proj(hidden) * block.mlp.up_proj(hidden)
            )
        return self.lm_head(hidden)


@pytest.fixture(scope="module")
def measured():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = _TinyModel()
    text = "the quick brown fox jumps over the lazy dog. " * 60
    return collect(model, _Tokenizer(), [text], chunk_tokens=64, dataset="fox")


class TestTheNaming:
    @pytest.mark.parametrize(
        ("module", "expected"),
        [
            ("model.layers.0.self_attn.q_proj", "blk.0.attn_q.weight"),
            ("model.layers.7.self_attn.k_proj", "blk.7.attn_k.weight"),
            ("model.layers.7.self_attn.v_proj", "blk.7.attn_v.weight"),
            ("model.layers.7.self_attn.o_proj", "blk.7.attn_output.weight"),
            ("model.layers.3.mlp.gate_proj", "blk.3.ffn_gate.weight"),
            ("model.layers.3.mlp.up_proj", "blk.3.ffn_up.weight"),
            ("model.layers.3.mlp.down_proj", "blk.3.ffn_down.weight"),
            ("layers.11.attention.wq", "blk.11.attn_q.weight"),
            ("layers.11.feed_forward.w2", "blk.11.ffn_down.weight"),
            ("lm_head", "output.weight"),
        ],
    )
    def test_a_torch_module_maps_to_its_gguf_tensor(self, module, expected):
        assert gguf_tensor_name(module) == expected

    @pytest.mark.parametrize(
        "module",
        ["model.embed_tokens", "model.norm", "model.layers.0.input_layernorm",
         "something.entirely.else"],
    )
    def test_a_module_with_no_counterpart_gets_no_name(self, module):
        """A guessed name is worse than a missing one: it silently
        weights a tensor the numbers were not measured on."""
        assert gguf_tensor_name(module) == ""


class TestMeasuring:
    def test_it_finds_every_linear_in_the_model(self, measured):
        for name in (
            "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
            "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
            "blk.1.attn_q.weight", "output.weight",
        ):
            assert name in measured, name

    def test_the_channel_count_is_the_layer_s_input_width(self, measured):
        """One number per input channel. A vector the width of the
        *output* would tile onto the tensor just as neatly and be
        entirely the wrong numbers."""
        assert len(measured.entries["blk.0.attn_q.weight"].sums) == 32
        assert len(measured.entries["blk.0.ffn_down.weight"].sums) == 64

    def test_every_value_is_a_non_negative_finite_number(self, measured):
        for entry in measured.entries.values():
            values = entry.means
            assert all(v >= 0 for v in values), entry.name
            assert all(v == v and v != float("inf") for v in values), entry.name

    def test_it_counts_the_tokens_it_saw(self, measured):
        assert measured.tokens > 0
        assert measured.tokens % 64 == 0

    def test_different_text_gives_different_numbers(self):
        """The test that would fail if this were weight-derived.

        Same weights, different calibration text: an imatrix that is
        really a function of the activations moves, and one that is
        secretly a function of the weights does not.
        """
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = _TinyModel()
        first = collect(model, _Tokenizer(), ["aaaa" * 200], chunk_tokens=64)
        second = collect(model, _Tokenizer(), ["zebra crossing " * 100], chunk_tokens=64)

        a = first.entries["blk.0.attn_q.weight"].means
        b = second.entries["blk.0.attn_q.weight"].means
        assert a != b, "the imatrix did not depend on the calibration text"

    def test_a_model_with_nothing_to_hook_says_so(self):
        model = torch.nn.Module()
        with pytest.raises(ImatrixError, match="no torch.nn.Linear"):
            collect(model, _Tokenizer(), ["text"])

    def test_text_too_short_for_one_chunk_is_an_error_not_an_empty_file(self):
        """An imatrix measured on nothing is not a faster imatrix."""
        torch.set_num_threads(1)
        model = _TinyModel()
        with pytest.raises(ImatrixError, match="fewer than one"):
            collect(model, _Tokenizer(), ["hi"], chunk_tokens=512)

    def test_max_chunks_stops_early(self):
        torch.set_num_threads(1)
        model = _TinyModel()
        matrix = collect(
            model, _Tokenizer(), ["abcdefgh" * 500], chunk_tokens=64, max_chunks=3
        )
        assert matrix.tokens == 3 * 64

    def test_the_model_is_left_as_it_was_found(self):
        """The hooks come off even though the run went through them.
        A model that keeps accumulating after the call would grow a leak
        and quietly wrong numbers on the next one."""
        torch.set_num_threads(1)
        model = _TinyModel()
        collect(model, _Tokenizer(), ["abcdefgh" * 200], chunk_tokens=64)
        linear = model.model.layers[0].self_attn.q_proj
        assert not linear._forward_pre_hooks

    def test_an_unmapped_module_is_reported_not_dropped(self):
        torch.set_num_threads(1)
        model = _TinyModel()
        model.model.mystery = torch.nn.Linear(32, 32, bias=False)
        original_forward = model.forward

        def forward(ids):
            out = original_forward(ids)
            model.model.mystery(model.model.embed_tokens(ids))
            return out

        model.forward = forward
        matrix = collect(model, _Tokenizer(), ["abcdefgh" * 200], chunk_tokens=64)
        assert "model.mystery" in matrix.unmapped
        assert "not mapped" in matrix.describe()


class TestTheFileFormats:
    def test_the_binary_round_trips(self, measured, tmp_path):
        path = measured.save_binary(tmp_path / "m.imatrix")
        again = Imatrix.load(path)
        assert set(again.entries) == set(measured.entries)
        for name, entry in measured.entries.items():
            assert again.entries[name].means == pytest.approx(entry.means, rel=1e-5)

    def test_the_binary_layout_is_the_one_llama_cpp_reads(self, tmp_path):
        """Written by hand and read back by hand, because the whole value
        of this format is that another program parses it."""
        matrix = Imatrix(dataset="wiki")
        matrix.add("blk.0.attn_q.weight", [2.0, 4.0, 6.0], calls=2)
        path = matrix.save_binary(tmp_path / "m.imatrix")

        raw = path.read_bytes()
        cursor = 0

        def _int() -> int:
            nonlocal cursor
            value = struct.unpack_from("<i", raw, cursor)[0]
            cursor += 4
            return value

        def _bytes(length: int) -> bytes:
            nonlocal cursor
            value = raw[cursor:cursor + length]
            cursor += length
            return value

        assert _int() == 1                                  # entry count
        assert _bytes(_int()) == b"blk.0.attn_q.weight"     # length-prefixed name
        assert _int() == 2                                  # ncall
        assert _int() == 3                                  # value count
        # Sums, not means: llama.cpp's loader divides by ncall itself, and
        # a file of means with ncall set would be divided twice.
        assert struct.unpack_from("<3f", raw, cursor) == (2.0, 4.0, 6.0)
        cursor += 12
        assert _int() == 2                                  # m_last_call
        assert _bytes(_int()) == b"wiki"                    # dataset name

    def test_the_json_round_trips_with_its_call_counts(self, measured, tmp_path):
        path = measured.save_json(tmp_path / "m.json")
        again = Imatrix.load(path)
        assert again.dataset == measured.dataset
        assert again.tokens == measured.tokens
        for name, entry in measured.entries.items():
            assert again.entries[name].means == pytest.approx(entry.means, rel=1e-6)

    def test_the_simple_json_is_what_hyprslug_reads(self, measured, tmp_path):
        path = measured.save_json(tmp_path / "simple.json", simple=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert isinstance(data["blk.0.attn_q.weight"], list)
        assert Imatrix.load(path).to_simple_dict().keys() == data.keys()

    def test_the_format_is_decided_by_content_not_by_suffix(self, measured, tmp_path):
        """People rename these files."""
        misnamed = tmp_path / "actually-binary.json"
        measured.save_binary(misnamed)
        assert len(Imatrix.load(misnamed)) == len(measured)

        also = tmp_path / "actually-json.imatrix"
        measured.save_json(also)
        assert len(Imatrix.load(also)) == len(measured)

    def test_a_truncated_binary_says_so(self, measured, tmp_path):
        path = measured.save_binary(tmp_path / "m.imatrix")
        cut = path.read_bytes()[:20]
        (tmp_path / "cut.imatrix").write_bytes(cut)
        with pytest.raises(ImatrixError, match="truncated|does not have|past the end"):
            Imatrix.load(tmp_path / "cut.imatrix")

    def test_something_that_is_neither_is_refused_clearly(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("this is just some prose about quantisation", encoding="utf-8")
        with pytest.raises(ImatrixError, match="not a llama.cpp imatrix"):
            Imatrix.load(path)

    def test_an_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "empty.imatrix"
        path.write_bytes(b"")
        with pytest.raises(ImatrixError, match="empty"):
            Imatrix.load(path)


class TestMerging:
    def test_two_runs_add_up(self):
        matrix = Imatrix()
        matrix.add("blk.0.attn_q.weight", [1.0, 2.0], calls=1)
        matrix.add("blk.0.attn_q.weight", [3.0, 4.0], calls=1)
        entry = matrix.entries["blk.0.attn_q.weight"]
        assert entry.sums == [4.0, 6.0]
        assert entry.calls == 2
        assert entry.means == [2.0, 3.0]

    def test_a_different_width_is_refused(self):
        entry = ImatrixEntry("x", [1.0, 2.0], 1)
        with pytest.raises(ImatrixError, match="different models"):
            entry.merge(ImatrixEntry("x", [1.0], 1))


class TestExpanding:
    def test_a_channel_vector_tiles_across_the_rows(self):
        assert expand_for_tensor([1.0, 2.0], 6) == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

    def test_a_full_length_vector_is_used_as_is(self):
        assert expand_for_tensor([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]

    def test_a_width_that_does_not_divide_is_refused(self):
        """A mismatched imatrix is a different model's. Stretching it to
        fit would weight the wrong positions, which is worse than not
        weighting at all."""
        assert expand_for_tensor([1.0, 2.0, 3.0], 8) is None

    def test_nothing_expands_to_nothing(self):
        assert expand_for_tensor([], 8) is None
        assert expand_for_tensor([1.0], 0) is None


class TestItReachesTheQuantiser:
    def test_hyprslug_reads_the_binary_format(self, tmp_path):
        """The end of the chain: measured here, written in llama.cpp's
        format, and used by the quantiser without a conversion step."""
        import struct as _struct

        from hypernix.quant.gguf import GGMLType, GGUFWriter
        from hypernix.quant.hyprslug import quantize_gguf

        source = tmp_path / "src.gguf"
        writer = GGUFWriter(source)
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", (256, 4), int(GGMLType.F32))
        values = [0.01 * ((i % 17) - 8) for i in range(1024)]
        writer.write(lambda _t: _struct.pack("<1024f", *values))

        matrix = Imatrix()
        # One weight per input channel, not per element: the quantiser
        # tiles it, and this is the shape a real imatrix has.
        matrix.add("blk.0.attn_q.weight", [1.0] * 256, calls=1)
        path = matrix.save_binary(tmp_path / "m.imatrix")

        report = quantize_gguf(source, tmp_path / "out.gguf", "Q4_K", imatrix=path)
        assert report.tensors_quantized == 1

    def test_a_per_channel_imatrix_is_not_reported_as_mismatched(self, tmp_path, caplog):
        import struct as _struct

        from hypernix.quant.gguf import GGMLType, GGUFWriter
        from hypernix.quant.hyprslug import quantize_gguf

        source = tmp_path / "src.gguf"
        writer = GGUFWriter(source)
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", (256, 4), int(GGMLType.F32))
        writer.write(lambda _t: _struct.pack("<1024f", *([0.05] * 1024)))

        with caplog.at_level("WARNING"):
            quantize_gguf(
                source, tmp_path / "out.gguf", "Q4_K",
                imatrix={"blk.0.attn_q.weight": [1.0] * 256},
            )
        assert "ignoring it" not in caplog.text


class TestTheCommandLine:
    def _run(self, *argv):
        return subprocess.run(
            [sys.executable, "-m", "hypernix.quant.imatrix_cli", *argv],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        )

    def test_show_describes_a_file(self, measured, tmp_path):
        path = measured.save_binary(tmp_path / "m.imatrix")
        result = self._run("show", str(path))
        assert result.returncode == 0, result.stderr
        assert "blk.0.attn_q.weight" in result.stdout
        assert "channels" in result.stdout

    def test_convert_goes_binary_to_json_and_back(self, measured, tmp_path):
        binary = measured.save_binary(tmp_path / "m.imatrix")
        assert self._run("convert", str(binary), "-o", str(tmp_path / "m.json")).returncode == 0
        assert (tmp_path / "m.json").exists()
        assert self._run(
            "convert", str(tmp_path / "m.json"), "-o", str(tmp_path / "back.imatrix")
        ).returncode == 0
        assert set(Imatrix.load(tmp_path / "back.imatrix").entries) == set(measured.entries)

    def test_measure_with_no_text_refuses(self, tmp_path):
        result = self._run("measure", "some-model", "-o", str(tmp_path / "o.imatrix"))
        assert result.returncode == 2
        assert "no calibration text" in result.stderr

    def test_no_command_prints_help(self):
        result = self._run()
        assert result.returncode == 2
        assert "measure" in result.stdout


class TestTheHonestLimit:
    def test_a_gguf_is_refused_with_the_reason(self, tmp_path):
        """Measuring means running the model, and running a GGUF means an
        inference engine this package does not carry. Saying so beats
        returning something weight-derived and calling it an imatrix."""
        path = tmp_path / "model.gguf"
        path.write_bytes(b"GGUF")
        with pytest.raises(ImatrixError, match="inference engine"):
            collect_from_pretrained(path, ["text"])
