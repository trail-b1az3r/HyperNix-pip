"""Why hyprslug's models were falling apart.

Not the attention weights, and not the codecs — those measure exactly
what their bitrates allow. It was ``ffn_gate_inp``.

A mixture-of-experts router is ``[n_embd, n_expert]``: a few hundred
kilobytes in a model of tens of gigabytes, and the one tensor in the
file whose output is an **argmax** rather than a sum. Every other weight
gets averaged over a reduction of thousands of terms, which is what
makes a 4-bit dot product survivable at all. The router's does not. A
2% error in its logits is nothing right up until two experts are within
2% of each other, and then it is a *different expert* — one that was
never trained for this token. The model does not degrade, it changes
subject.

The same argument covers Mamba's convolution, RWKV's time-mixing
constants and the positional and token-type tables: small, read
directly, catastrophic when moved. llama.cpp refuses to quantise exactly
this list, for exactly this reason. hyprslug's list was norms and
biases, so on every MoE model in circulation it crushed the router and
reported success.

The second half of the file is about the label. ``general.file_type``
was copied from the source and never rewritten, so a Q4_K_M made from an
F16 announced itself as F16 to llama.cpp's load banner, to a hub
listing, and to ``hypernix-t1 index`` — the tensor table was right and
the field everybody actually reads was wrong.
"""
from __future__ import annotations

import math
import random
import struct

import pytest

from hypernix.quant import hyprslug
from hypernix.quant.gguf import GGMLType, GGUFFile, GGUFWriter

GB = 1024 ** 3


def weights(count: int, seed: int) -> list[float]:
    """*count* weights shaped like a trained tensor's, reproducibly.

    Gaussian rather than a sine wave. A smooth function quantises far
    better than a real weight matrix does -- neighbouring values inside
    a 256-element block are nearly equal, so a block scale fits them
    almost exactly -- and a test built on one measures the quantiser at
    its best rather than at the job.
    """
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.02) for _ in range(count)]


def build_model(path, shapes, *, file_type: int = 1):
    """A GGUF with *shapes*, filled with reproducible weights."""
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.file_type", file_type)
    writer.set_metadata("llama.block_count", 1)
    payload = {}
    for index, (name, shape) in enumerate(shapes.items()):
        count = shape[0] * shape[1]
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = struct.pack(f"<{count}f", *weights(count, index + 1))
    writer.write(lambda tensor: payload[tensor.name])
    return payload


#: A block of a mixture-of-experts model, at the smallest shapes the
#: 256-element block size allows.
MOE_SHAPES = {
    "token_embd.weight": (512, 64),
    "blk.0.attn_norm.weight": (512, 1),
    "blk.0.attn_q.weight": (512, 512),
    "blk.0.ffn_gate_inp.weight": (512, 8),
    "blk.0.ffn_gate_exps.weight": (512, 256),
    "blk.0.ffn_down_exps.weight": (512, 256),
    "output.weight": (512, 64),
}


def types_in(path) -> dict[str, str]:
    model = GGUFFile.read(path)
    return {t.name: GGMLType(int(t.ggml_type)).name for t in model.tensors}


class TestTheRouterSurvives:
    """The bug, and the only test in this file that is about one tensor."""

    @pytest.mark.parametrize("tier", ["Q4_K_M", "Q2_K_S", "Q6_K", "IQ0.5_XXXL"])
    def test_it_is_never_quantised_at_any_tier(self, tmp_path, tier):
        source = tmp_path / "moe.gguf"
        build_model(source, MOE_SHAPES)
        out = tmp_path / f"out-{tier}.gguf"
        hyprslug.quantize_gguf(source, out, tier)
        assert types_in(out)["blk.0.ffn_gate_inp.weight"] == "F32"

    def test_the_experts_themselves_still_are(self, tmp_path):
        """Protecting the router must not turn into protecting the
        model: the expert tensors are the file, and a "Q4_K_M" that
        copied them would be an F32 model with a label."""
        source = tmp_path / "moe.gguf"
        build_model(source, MOE_SHAPES)
        out = tmp_path / "out.gguf"
        hyprslug.quantize_gguf(source, out, "Q4_K_M")
        found = types_in(out)
        assert found["blk.0.ffn_gate_exps.weight"].startswith("Q")
        assert found["blk.0.attn_q.weight"].startswith("Q")

    def test_the_run_says_it_skipped_it_and_why(self, tmp_path):
        """A protection nobody can see is one somebody removes."""
        source = tmp_path / "moe.gguf"
        build_model(source, MOE_SHAPES)
        report = hyprslug.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        reasons = dict(report.skipped)
        assert "argmax" in reasons["blk.0.ffn_gate_inp.weight"]

    def test_the_damage_it_was_taking_was_real(self, tmp_path):
        """Not a theoretical concern. Quantising the router directly and
        measuring is the difference between "we think this matters" and
        knowing it does: Q2_K moves a router's logits by tens of
        percent, and expert selection is a comparison between them."""
        from hypernix.quant.llamaquants import dequantize_array, quantize_array

        router = weights(4096, seed=11)
        packed = quantize_array(router, "Q2_K")
        recovered = [float(v) for v in dequantize_array(packed, int(GGMLType.Q2_K))]
        error = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(router, recovered, strict=True))
            / sum(a * a for a in router)
        )
        assert error > 0.1, "if this is small the protection is unnecessary"


class TestTheRestOfTheList:
    """Every architecture whose small 2-D weights were being crushed."""

    def test_mamba_keeps_its_convolution(self, tmp_path):
        shapes = dict(MOE_SHAPES, **{"blk.0.ssm_conv1d.weight": (256, 512)})
        source = tmp_path / "mamba.gguf"
        build_model(source, shapes)
        hyprslug.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        assert types_in(tmp_path / "out.gguf")["blk.0.ssm_conv1d.weight"] == "F32"

    def test_rwkv_keeps_its_time_mixing_constants(self, tmp_path):
        shapes = dict(MOE_SHAPES, **{
            "blk.0.time_mix_first.weight": (512, 8),
            "blk.0.time_mix_decay_w1.weight": (512, 8),
        })
        source = tmp_path / "rwkv.gguf"
        build_model(source, shapes)
        hyprslug.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        found = types_in(tmp_path / "out.gguf")
        assert found["blk.0.time_mix_first.weight"] == "F32"
        assert found["blk.0.time_mix_decay_w1.weight"] == "F32"

    def test_rwkv_still_quantises_its_projections(self, tmp_path):
        """``time_mix_key`` and ``time_mix_value`` are full-sized
        projections. A bare ``time_mix`` prefix would leave most of an
        RWKV model unquantised and still call it Q4_K_M."""
        shapes = dict(MOE_SHAPES, **{"blk.0.time_mix_key.weight": (512, 512)})
        source = tmp_path / "rwkv.gguf"
        build_model(source, shapes)
        hyprslug.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        assert types_in(tmp_path / "out.gguf")["blk.0.time_mix_key.weight"] == "Q4_K"

    def test_positional_and_token_type_tables_are_kept(self, tmp_path):
        shapes = dict(MOE_SHAPES, **{
            "position_embd.weight": (512, 512),
            "token_types.weight": (512, 2),
        })
        source = tmp_path / "bert.gguf"
        build_model(source, shapes)
        hyprslug.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        found = types_in(tmp_path / "out.gguf")
        assert found["position_embd.weight"] == "F32"
        assert found["token_types.weight"] == "F32"

    def test_norms_and_biases_are_still_kept(self, tmp_path):
        """The original list. Replacing it must not lose it."""
        assert hyprslug.never_quantize_reason("blk.0.attn_norm.weight")
        assert hyprslug.never_quantize_reason("output_norm.weight")
        assert hyprslug.never_quantize_reason("blk.0.attn_q.bias")

    def test_an_ordinary_weight_is_not_protected(self, tmp_path):
        for name in ("blk.0.attn_q.weight", "blk.0.ffn_down.weight",
                     "token_embd.weight", "output.weight"):
            assert hyprslug.never_quantize_reason(name) == "", name


class TestTheFileSaysWhatItIs:
    """``general.file_type`` was the source's, always."""

    def _file_type(self, path) -> int:
        return int(GGUFFile.read(path).metadata["general.file_type"])

    def test_a_q4_k_m_no_longer_claims_to_be_f16(self, tmp_path):
        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES, file_type=1)  # 1 == MOSTLY_F16
        out = tmp_path / "out.gguf"
        hyprslug.quantize_gguf(source, out, "Q4_K_M")
        assert self._file_type(out) == 15  # MOSTLY_Q4_K_M

    def test_each_recipe_gets_its_own_number(self, tmp_path):
        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES)
        for tier, expected in (("Q8_0", 7), ("Q6_K", 18), ("Q2_K_S", 10),
                               ("Q5_K_M", 17), ("Q4_K_S", 14)):
            out = tmp_path / f"{tier}.gguf"
            hyprslug.quantize_gguf(source, out, tier)
            assert self._file_type(out) == expected, tier

    def test_a_width_conversion_gets_the_width(self, tmp_path):
        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES, file_type=0)
        out = tmp_path / "bf16.gguf"
        hyprslug.quantize_gguf(source, out, "BF16")
        assert self._file_type(out) == 32  # MOSTLY_BF16

    def test_a_sub_bit_tier_does_not_borrow_an_upstream_number(self, tmp_path):
        """A half-bit file labelled Q2_K claims four times the precision
        it has. There is no upstream number for these, so it says so
        with one of its own rather than picking the nearest lie."""
        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES)
        out = tmp_path / "sub.gguf"
        hyprslug.quantize_gguf(source, out, "IQ0.5_XXXL")
        assert self._file_type(out) >= hyprslug.HNX_FILE_TYPE_BASE

    def test_sub_bit_numbers_are_distinct_per_tier(self, tmp_path):
        seen = {
            hyprslug.file_type_for(hyprslug.target_spec(tier))
            for tier in hyprslug.TIER_TYPES
        }
        assert len(seen) == len(hyprslug.TIER_TYPES)

    def test_the_quantisation_version_is_written(self, tmp_path):
        """llama.cpp writes this on every file it quantises, and readers
        use it to decide how to interpret the block layouts."""
        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES)
        out = tmp_path / "out.gguf"
        hyprslug.quantize_gguf(source, out, "Q4_K_M")
        assert GGUFFile.read(out).metadata["general.quantization_version"] == 2

    def test_it_is_a_u32_like_every_other_writer_makes_it(self, tmp_path):
        """A reader calling ``gguf_get_val_u32`` refuses an i32, so the
        key would be present, correct, and unreadable."""
        from hypernix.quant.gguf import GGUFValueType

        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES)
        out = tmp_path / "out.gguf"
        hyprslug.quantize_gguf(source, out, "Q4_K_M")
        types = GGUFFile.read(out).metadata_types
        assert types["general.file_type"][0] == int(GGUFValueType.UINT32)
        assert types["general.quantization_version"][0] == int(GGUFValueType.UINT32)

    def test_everything_else_still_comes_across(self, tmp_path):
        """Rewriting two keys must not become filtering the metadata."""
        source = tmp_path / "src.gguf"
        build_model(source, MOE_SHAPES)
        out = tmp_path / "out.gguf"
        hyprslug.quantize_gguf(source, out, "Q4_K_M")
        carried = GGUFFile.read(out).metadata
        assert carried["general.architecture"] == "llama"
        assert carried["llama.block_count"] == 1
