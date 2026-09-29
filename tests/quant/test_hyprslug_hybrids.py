"""Q8_K, INT3, FP8 and the hnx_Q6_H hybrids, and Brewer models as input.

The hybrids are rules rather than tables: a tensor's format depends on
its role *and* its depth, so these build a model with enough layers for
the depth to matter and check where each format lands. The one rule
that is not about quality is the embedding: it is read with get_rows,
which llama.cpp's CPU backend implements only for its own types, so a
hybrid that put a HyperNix type or Q8_K there would write a file that
loads and aborts on the first token.
"""
from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant import llamaquants as lq
from hypernix.quant import lowbit
from hypernix.quant.gguf import _BLOCK_SHAPE, GGMLType, GGUFFile, GGUFWriter
from hypernix.quant.hyprslug import (
    IQ1_XS,
    RECIPES,
    TIER_TYPES,
    file_type_for,
    is_brewer_source,
    plan_tensors,
    quantize_gguf,
    resolve_target,
    target_spec,
)

LAYERS = 8
HYBRIDS = ("hnx_Q6_H_k", "hnx_Q6_H_4", "hnx_Q6_H_2")


def _gaussian(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.02, count).astype(np.float32)


def _model(path: Path, layers: int = LAYERS) -> Path:
    """A llama-shaped GGUF: every role, ``layers`` deep, rows of 256."""
    shapes = {"token_embd.weight": (256, 4), "output.weight": (256, 4),
              "output_norm.weight": (256,)}
    for layer in range(layers):
        for role in ("attn_q", "attn_k", "attn_v", "attn_output",
                     "ffn_gate", "ffn_up", "ffn_down"):
            shapes[f"blk.{layer}.{role}.weight"] = (256, 2)
        shapes[f"blk.{layer}.attn_norm.weight"] = (256,)
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    payload = {}
    for index, (name, shape) in enumerate(shapes.items()):
        count = int(np.prod(shape))
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = _gaussian(count, index).tobytes()
    writer.write(lambda tensor: payload[tensor.name])
    return path


def _formats(path: Path, target: str) -> dict[str, str]:
    """``tensor -> format`` as the planner decides it."""
    model = GGUFFile.read(path)
    return {plan.name: plan.encoding for plan in plan_tensors(model, target_spec(target))}


class TestQ8K:
    def test_it_is_llamacpps_block(self):
        fmt = lq.FORMATS["Q8_K"]
        assert (fmt.ggml_type, fmt.block, fmt.block_bytes) == (15, 256, 292)
        assert _BLOCK_SHAPE[GGMLType.Q8_K] == (256, 292)

    def test_the_largest_weight_lands_on_minus_127(self):
        """quantize_row_q8_K_ref takes the scale from the *signed* peak,
        so that weight is exactly -127 and the scale carries the sign."""
        x = _gaussian(256, 1)
        x[17] = 0.5
        raw = lq.quantize_array(x, "Q8_K")
        d = struct.unpack_from("<f", raw, 0)[0]
        qs = np.frombuffer(raw, dtype=np.int8, count=256, offset=4)
        assert qs[17] == -127
        assert d < 0
        assert d * qs[17] == pytest.approx(0.5, rel=1e-6)

    def test_bsums_are_the_sixteen_partial_sums(self):
        raw = lq.quantize_array(_gaussian(512, 2), "Q8_K")
        for block in range(2):
            base = block * 292
            qs = np.frombuffer(raw, dtype=np.int8, count=256, offset=base + 4)
            bsums = np.frombuffer(raw, dtype="<i2", count=16, offset=base + 260)
            assert list(bsums) == [int(qs[i * 16:(i + 1) * 16].sum()) for i in range(16)]

    def test_it_round_trips_to_eight_bits(self):
        x = _gaussian(256 * 8, 3)
        back = lq.dequantize_array(lq.quantize_array(x, "Q8_K"), "Q8_K")
        assert np.sqrt(np.mean((back - x) ** 2)) / x.std() < 0.01

    def test_a_zero_block_stays_zero(self):
        raw = lq.quantize_array(np.zeros(256), "Q8_K")
        assert not any(raw)


class TestInt3AndFp8:
    @pytest.mark.parametrize("name,ggml_type,block_bytes", [
        ("INT3", GGMLType.HNX_INT3, 98), ("FP8", GGMLType.HNX_FP8, 258),
    ])
    def test_the_block_matches_the_gguf_table(self, name, ggml_type, block_bytes):
        assert lowbit.CODECS[name].block_bytes == block_bytes
        assert _BLOCK_SHAPE[ggml_type] == (256, block_bytes)
        assert TIER_TYPES[name] == (int(ggml_type), name)

    def test_int3_sits_between_int2_and_int4(self):
        x = _gaussian(256 * 16, 4)
        error = {}
        for name in ("INT2", "INT3", "INT4"):
            back = lowbit.dequantize_array(lowbit.quantize_array(x, name), name)
            error[name] = float(np.sqrt(np.mean((back - x) ** 2)) / x.std())
        assert error["INT4"] < error["INT3"] < error["INT2"]

    def test_fp8_has_every_finite_e4m3_value(self):
        codec = lowbit.CODECS["FP8"]
        assert len(codec.levels) == 253           # 256 - 2 NaN - one of +/-0
        assert codec.peak == 448.0
        assert lowbit.e4m3_value(0x7E) == 448.0
        assert lowbit.e4m3_value(0x01) == 2.0 ** -9
        assert lowbit.e4m3_value(0x7F) != lowbit.e4m3_value(0x7F)   # NaN

    def test_fp8_codes_are_the_e4m3_bytes(self):
        """Each stored byte, read as E4M3 and multiplied by the scale, is
        the decoded weight -- so an FP8 kernel can use the codes as is."""
        x = _gaussian(256, 5)
        raw = lowbit.quantize_array(x, "FP8")
        scale = float(np.frombuffer(raw[:2], dtype=np.float16)[0])
        decoded = lowbit.dequantize_array(raw, "FP8")
        by_byte = np.array([lowbit.e4m3_value(b) * scale for b in raw[2:]], dtype=np.float32)
        np.testing.assert_array_equal(decoded, by_byte)

    def test_fp8_never_writes_a_nan_code_or_negative_zero(self):
        x = np.concatenate([_gaussian(256 * 4, 6), np.zeros(256)])
        raw = lowbit.quantize_array(x, "FP8")
        codes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 258)[:, 2:]
        assert not np.isin(codes, [0x7F, 0xFF, 0x80]).any()

    def test_fp8_is_close(self):
        x = _gaussian(256 * 16, 7)
        back = lowbit.dequantize_array(lowbit.quantize_array(x, "FP8"), "FP8")
        assert np.sqrt(np.mean((back - x) ** 2)) / x.std() < 0.05


class TestTheNames:
    @pytest.mark.parametrize("spelling,expected", [
        ("q6h", "hnx_Q6_H_k"), ("hnx_Q6_H_k", "hnx_Q6_H_k"), ("HNX-Q6-H-K", "hnx_Q6_H_k"),
        ("q6h4", "hnx_Q6_H_4"), ("hnxq6h4", "hnx_Q6_H_4"),
        ("q6h2", "hnx_Q6_H_2"), ("hnx_q6_h_2", "hnx_Q6_H_2"),
        ("q8k", "Q8_K"), ("Q8_K", "Q8_K"),
    ])
    def test_recipes(self, spelling, expected):
        assert resolve_target(spelling) == ("recipe", expected)

    @pytest.mark.parametrize("spelling,expected", [
        ("int3", "INT3"), ("i3", "INT3"), ("fp8", "FP8"), ("e4m3", "FP8"),
        ("iq1_xs", "HNX_1375BIT"), ("IQ1XS", "HNX_1375BIT"),
    ])
    def test_tiers(self, spelling, expected):
        assert resolve_target(spelling) == ("tier", expected)

    def test_iq1_xs_is_the_hyperNix_one_bit_type(self):
        """llama.cpp has IQ1_S and IQ1_M and no IQ1_XS; the name means
        the HyperNix type below IQ1_S, and says so in one place."""
        assert IQ1_XS == "HNX_1375BIT"


class TestTheHybrids:
    @pytest.fixture(scope="class")
    def model(self, tmp_path_factory):
        return _model(tmp_path_factory.mktemp("hybrid") / "m.gguf")

    @pytest.mark.parametrize("name", HYBRIDS)
    def test_the_embedding_is_always_a_stock_k_quant(self, model, name):
        fmt = _formats(model, name)["token_embd.weight"]
        assert fmt in lq.FORMATS and fmt != "Q8_K"

    @pytest.mark.parametrize("name", HYBRIDS)
    def test_norms_are_still_left_alone(self, model, name):
        assert _formats(model, name)["blk.0.attn_norm.weight"] == ""

    @pytest.mark.parametrize("name", HYBRIDS)
    def test_every_format_is_one_the_recipe_declares(self, model, name):
        declared = {("HNX_1375BIT" if f == "IQ1_XS" else f) for f in RECIPES[name].ingredients}
        used = {fmt for fmt in _formats(model, name).values() if fmt}
        assert used <= declared
        assert len(used) >= 3, "a hybrid that uses one format is not a hybrid"

    def test_q6h_k_mixes_q8_k_q6_k_and_q3_k(self, model):
        formats = _formats(model, "hnx_Q6_H_k")
        assert formats["output.weight"] == "Q8_K"
        assert formats["blk.0.attn_v.weight"] == "Q8_K"       # an edge layer
        assert formats["blk.0.ffn_up.weight"] == "Q6_K"
        assert formats["blk.4.ffn_up.weight"] == "Q3_K"       # the middle
        assert formats["blk.4.attn_q.weight"] == "Q6_K"
        assert set(formats.values()) - {""} == {"Q8_K", "Q6_K", "Q3_K"}

    def test_q6h_4_puts_int3_and_q2_k_in_the_middle_only(self, model):
        formats = _formats(model, "hnx_Q6_H_4")
        middle = {formats[f"blk.{i}.ffn_gate.weight"] for i in (2, 3, 4, 5)}
        assert middle == {"Q2_K"}
        assert {formats[f"blk.{i}.ffn_up.weight"] for i in (2, 3, 4, 5)} == {"INT3"}
        assert formats["blk.0.ffn_gate.weight"] == "Q5_K"
        assert formats["blk.7.ffn_up.weight"] == "Q5_K"
        assert formats["blk.4.attn_q.weight"] == "Q4_K"
        assert formats["output.weight"] == "Q6_K"
        assert set(formats.values()) - {""} == {"Q6_K", "Q5_K", "Q4_K", "INT3", "Q2_K"}

    def test_q6h_2_is_q6h_4_without_int3_and_q2_k(self, model):
        formats = _formats(model, "hnx_Q6_H_2")
        used = set(formats.values()) - {""}
        assert "INT3" not in used and "Q2_K" not in used and "Q4_K" not in used
        assert formats["blk.4.ffn_gate.weight"] == "HNX_1375BIT"   # IQ1_XS
        assert formats["blk.4.ffn_up.weight"] == "Q3_K"
        assert formats["blk.4.attn_q.weight"] == "Q3_K"            # Q3_K_L's base
        assert formats["blk.4.attn_k.weight"] == "Q5_K"            # ...and its widening

    def test_the_depth_follows_the_model_not_a_fixed_count(self, tmp_path):
        deep = _formats(_model(tmp_path / "deep.gguf", layers=16), "hnx_Q6_H_4")
        assert deep["blk.1.ffn_gate.weight"] == "Q5_K"     # 16 // 8 = 2 edge layers
        assert deep["blk.2.ffn_gate.weight"] == "Q4_K"
        assert deep["blk.8.ffn_gate.weight"] == "Q2_K"

    @pytest.mark.parametrize("name", HYBRIDS)
    def test_the_file_is_what_the_plan_said(self, model, tmp_path, name):
        out = tmp_path / f"{name}.gguf"
        report = quantize_gguf(model, out, name)
        written = GGUFFile.read(out)
        planned = {p.name: p.ggml_type for p in plan_tensors(GGUFFile.read(model), target_spec(name))}
        assert {t.name: int(t.ggml_type) for t in written.tensors} == {
            k: int(v) for k, v in planned.items()}
        assert written.metadata["hypernix.hybrid"] is True
        assert written.metadata["hypernix.needs"] == "ggml-hnx"
        assert written.metadata["general.file_type"] == file_type_for(target_spec(name))
        assert report.tensors_quantized == sum(1 for t in planned.values() if t != GGMLType.F32)

    @pytest.mark.parametrize("name", [*HYBRIDS, "Q8_K", "INT3", "FP8"])
    def test_it_decodes_back_close_to_the_source(self, model, tmp_path, name):
        """Through hyprslug's own reader: quantise, then widen to FP32."""
        quantised = tmp_path / "q.gguf"
        widened = tmp_path / "f32.gguf"
        quantize_gguf(model, quantised, name)
        quantize_gguf(quantised, widened, "FP32")
        source, back = GGUFFile.read(model), GGUFFile.read(widened)
        worst = 0.0
        for tensor in source.tensors:
            if len(tensor.shape) < 2:
                continue
            x = np.frombuffer(source.tensor_bytes(tensor), dtype=np.float32)
            other = next(t for t in back.tensors if t.name == tensor.name)
            y = np.frombuffer(back.tensor_bytes(other), dtype=np.float32)
            worst = max(worst, float(np.sqrt(np.mean((y - x) ** 2)) / x.std()))
        # IQ1_XS in hnx_Q6_H_2 is the loosest thing here, at 1.375 bits.
        assert worst < 0.75

    def test_the_file_types_do_not_claim_an_upstream_number(self):
        numbers = {file_type_for(target_spec(n)) for n in (*HYBRIDS, "Q8_K")}
        assert len(numbers) == 4
        assert all(n >= 1300 for n in numbers)
        assert not RECIPES["Q4_K_M"].needs_patched_llamacpp
        assert all(RECIPES[n].needs_patched_llamacpp for n in (*HYBRIDS, "Q8_K"))


class TestBrewerInput:
    """hyperNix0x-v2 models go in as they are: folder or .pt."""

    @pytest.fixture
    def folder(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("gguf")
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file

        from hypernix.models.brewer_gguf import write_char_vocab
        from hypernix.training.brewer import BrewerConfig, BrewerModel

        cfg = BrewerConfig(vocab_size=64, n_layers=2, n_heads=4, n_kv_heads=2, d_model=256,
                           d_ff=256, max_seq_len=64, rope_theta=10000.0, tie_embeddings=False,
                           name="slugged")
        folder = tmp_path / "slugged"
        folder.mkdir()
        torch.manual_seed(0)
        model = BrewerModel(cfg)
        cfg.save(folder / "config.json")
        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  str(folder / "model.safetensors"))
        rng = random.Random(0)
        chars = sorted({chr(32 + rng.randrange(90)) for _ in range(400)})[:60]
        write_char_vocab(folder / "char_vocab.json", chars)
        return folder

    def test_a_brewer_folder_is_recognised(self, folder, tmp_path):
        assert is_brewer_source(folder)
        assert is_brewer_source(tmp_path / "x.pt") is False     # does not exist
        (tmp_path / "x.pt").write_bytes(b"")
        assert is_brewer_source(tmp_path / "x.pt")

    def test_a_hugging_face_folder_is_not(self, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        (hf / "config.json").write_text(json.dumps({"hidden_size": 256, "num_hidden_layers": 2}))
        (hf / "model.safetensors").write_bytes(b"")
        assert not is_brewer_source(hf)

    def test_it_quantises_a_brewer_folder(self, folder, tmp_path):
        out = tmp_path / "out" / "slugged.q6h4.gguf"
        report = quantize_gguf(folder, out, "q6h4")
        assert report.converted_from == "hyperNix0x-v2"
        written = GGUFFile.read(out)
        assert written.metadata["general.architecture"] == "llama"
        assert written.metadata["hypernix.tier"] == "hnx_Q6_H_4"
        assert report.tensors_quantized > 0
        # The staged F16 copy is gone.
        assert [p.name for p in out.parent.iterdir()] == [out.name]
        assert "hyperNix0x-v2" in report.describe()
