"""Several quantisations in one GGUF, and the two draft generations.

Three things are being tested and they fail in different ways.

The **new targets** (INT8, INT2, and the FP32/FP16/BF16 widths) fail
loudly: a wrong block size is a file that will not load. The tests here
are mostly that the encoder and the block-shape table agree, because
when they do not, every tensor after the first reads from the wrong
offset — and the model still loads, and still generates text, just wrong
text.

A **bundle** fails quietly. A stock llama.cpp opening one has to find
every tensor it expects under the name it expects and read past the rest;
if it does not, the failure is "missing tensor blk.0.attn_q.weight" from
a file that plainly contains it. So the compatibility assertions here are
about names, not about bytes.

A **draft** fails silently and is the worst of the three. A draft whose
tokenizer disagrees with its base has every proposal rejected: nothing
errors, nothing warns, generation just gets slower than not using a draft
at all. :func:`hypernix.quant.dflash1.derive` refuses to write one, and
that refusal is tested here because it is the only thing standing between
a user and a symptomless regression.
"""
from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from hypernix.quant import lowbit
from hypernix.quant.gguf import GGMLType, GGUFFile, GGUFWriter, type_block_size
from hypernix.quant.gguf import _BLOCK_SHAPE  # noqa: PLC2701 - the table under test
from hypernix.quant.hyprslug import (
    TIER_TYPES,
    WIDTHS,
    HyprslugError,
    quantize_gguf,
    resolve_target,
)
from hypernix.quant.multiquant import (
    PREFIX,
    MultiQuantError,
    bundle,
    extract,
    read_bundle_info,
    slug_for,
    strip,
    variants,
)


def _weights(count: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.05) for _ in range(count)]


def _model(path: Path, *, blocks: int = 4, tokenizer: bool = True) -> Path:
    """A GGUF shaped like a transformer, small enough to quantise in a test."""
    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (512, 8),
        "output.weight": (512, 8),
        "output_norm.weight": (512,),
    }
    for index in range(blocks):
        tensors[f"blk.{index}.attn_q.weight"] = (512, 8)
        tensors[f"blk.{index}.attn_v.weight"] = (512, 8)
        tensors[f"blk.{index}.ffn_down.weight"] = (512, 8)
        tensors[f"blk.{index}.attn_norm.weight"] = (512,)

    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.name", "Fixture")
    writer.set_metadata("llama.block_count", blocks)
    if tokenizer:
        writer.set_metadata("tokenizer.ggml.model", "gpt2")
        writer.set_metadata("tokenizer.ggml.tokens", ["a", "b", "c"])
    payload = {}
    for index, (name, shape) in enumerate(tensors.items()):
        count = 1
        for dim in shape:
            count *= dim
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = struct.pack(f"<{count}f", *_weights(count, seed=index))
    writer.write(lambda tensor: payload[tensor.name])
    return path


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return _model(tmp_path / "base.gguf")


class TestTheNewCodecs:
    """INT8 and INT2, against the block-shape table that reads them."""

    @pytest.mark.parametrize("name", ["INT8", "INT2"])
    def test_the_encoder_and_the_block_table_agree(self, name):
        """The one that matters. A table that drifts from the packer
        produces a file whose offsets are all wrong from the first
        tensor -- and it loads."""
        codec = lowbit.CODECS[name]
        ggml_type = TIER_TYPES[name][0]
        elements, block_bytes = _BLOCK_SHAPE[ggml_type]
        assert elements == lowbit.BLOCK_SIZE
        assert block_bytes == codec.block_bytes
        assert len(lowbit.quantize_array(_weights(elements * 3), name)) == block_bytes * 3

    @pytest.mark.parametrize("name", ["INT8", "INT2"])
    def test_a_round_trip_returns_the_right_count(self, name):
        values = _weights(lowbit.BLOCK_SIZE * 4, seed=7)
        back = lowbit.dequantize_array(lowbit.quantize_array(values, name), name)
        assert len(back) == len(values)

    def test_int8_is_close_and_int2_is_not(self):
        """Sanity on the two ends: 8 bits should reconstruct nearly
        exactly, 2 bits should not, and if they ever measure the same
        something is pointing at the wrong codec."""
        values = _weights(lowbit.BLOCK_SIZE * 8, seed=3)
        scale = (sum(v * v for v in values) / len(values)) ** 0.5

        def _error(name: str) -> float:
            back = lowbit.dequantize_array(lowbit.quantize_array(values, name), name)
            return (sum((a - b) ** 2 for a, b in zip(back, values)) / len(values)) ** 0.5

        assert _error("INT8") / scale < 0.02
        assert 0.2 < _error("INT2") / scale < 0.6

    def test_int2_has_a_zero_level_and_fp2_does_not(self):
        """The entire reason INT2 exists beside FP2."""
        assert 0.0 in lowbit.CODECS["INT2"].levels
        assert 0.0 not in lowbit.CODECS["FP2"].levels


class TestTheWidths:
    """FP32/FP16/BF16 are conversions, not quantisations."""

    @pytest.mark.parametrize("spelling,expected", [
        ("fp16", "FP16"), ("f16", "FP16"), ("half", "FP16"),
        ("bf16", "BF16"), ("bfloat16", "BF16"),
        ("fp32", "FP32"), ("f32", "FP32"), ("float32", "FP32"),
    ])
    def test_a_width_is_a_width_however_it_is_typed(self, spelling, expected):
        assert resolve_target(spelling) == ("width", expected)

    @pytest.mark.parametrize("width", sorted(WIDTHS))
    def test_every_tensor_converts_including_the_norms(self, base, tmp_path, width):
        """A quantiser leaves 1-D norms alone because they are all of the
        damage and none of the size. A width conversion has no such
        trade: leaving them would produce an "FP16" file with F32 norms
        in it."""
        out = tmp_path / f"{width}.gguf"
        quantize_gguf(base, out, width)
        types = {t.name: int(t.ggml_type) for t in GGUFFile.read(out).tensors}
        assert set(types.values()) == {WIDTHS[width][0]}

    def test_converting_to_the_width_it_already_is_copies(self, base, tmp_path):
        """The source here is F32 throughout, so FP32 has nothing to do.
        Copying is both faster and exactly lossless, where a decode and
        re-encode round trip is only nearly so."""
        report = quantize_gguf(base, tmp_path / "same.gguf", "FP32")
        assert report.tensors_quantized == 0
        assert report.tensors_copied == report.tensors_total
        assert all("already FP32" in reason for _, reason in report.skipped)

    def test_bf16_survives_what_f16_cannot(self, tmp_path):
        """F16 tops out at 65504. A weight past that becomes an inf that
        poisons every dot product it touches, so hyprslug saturates and
        -- this is the point -- counts it."""
        path = tmp_path / "big.gguf"
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", (256, 2), int(GGMLType.F32))
        values = [1e6] * 4 + [0.1] * 508
        writer.write(lambda _t: struct.pack(f"<{len(values)}f", *values))

        f16 = quantize_gguf(path, tmp_path / "f16.gguf", "FP16")
        assert f16.weights_saturated == 4
        assert "65504" in f16.describe()

        bf16 = quantize_gguf(path, tmp_path / "bf16.gguf", "BF16")
        assert bf16.weights_saturated == 0


class TestBundling:
    def test_the_default_keeps_the_ordinary_names(self, base, tmp_path):
        """The whole compatibility claim. A stock llama.cpp has to find
        every tensor it expects under the name it expects."""
        out = tmp_path / "bundle.gguf"
        bundle(base, out, ["Q8_0", "Q4_K_M", "INT2"])
        plain = {t.name for t in GGUFFile.read(base).tensors}
        bundled = {t.name for t in GGUFFile.read(out).tensors}
        assert plain <= bundled
        assert all(name.startswith(PREFIX) for name in bundled - plain)

    def test_the_default_variant_is_the_first_target(self, base, tmp_path):
        report = bundle(base, tmp_path / "b.gguf", ["Q4_K_M", "Q8_0"])
        assert report.default == "Q4_K_M"

    def test_a_named_default_wins(self, base, tmp_path):
        report = bundle(base, tmp_path / "b.gguf", ["Q4_K_M", "Q8_0"], default="q8")
        assert report.default == "Q8_0"

    def test_a_default_outside_the_targets_is_refused(self, base, tmp_path):
        """Otherwise the file would name a default it does not carry, and
        a loader would look for tensors that are not there."""
        with pytest.raises(MultiQuantError, match="not among the targets"):
            bundle(base, tmp_path / "b.gguf", ["Q4_K_M"], default="Q8_0")

    def test_the_same_tier_twice_is_refused(self, base, tmp_path):
        """`q4km` and `Q4_K_M` are one request, and a file carrying it
        twice under one slug has no way to say which is which."""
        with pytest.raises(MultiQuantError, match="same quantisation"):
            bundle(base, tmp_path / "b.gguf", ["Q4_K_M", "q4km"])

    def test_an_empty_target_list_is_refused(self, base, tmp_path):
        with pytest.raises(MultiQuantError, match="at least one target"):
            bundle(base, tmp_path / "b.gguf", [])

    def test_bundling_a_bundle_is_refused(self, base, tmp_path):
        """Nesting the prefixes would produce hnxq.a.hnxq.b.blk.0.… and
        no reader unwraps that."""
        first = tmp_path / "b.gguf"
        bundle(base, first, ["Q8_0", "Q4_K_M"])
        with pytest.raises(MultiQuantError, match="already a bundle"):
            bundle(first, tmp_path / "b2.gguf", ["Q8_0", "Q6_K"])

    def test_shared_tensors_are_stored_once(self, base, tmp_path):
        """Where the size saving comes from: a tensor every variant
        copies verbatim is the same bytes in each, so it is written once
        and pointed at."""
        out = tmp_path / "shared.gguf"
        report = bundle(base, out, ["Q8_0", "Q4_K_M"], share=True)
        assert report.shared_bytes > 0
        assert report.separate_bytes > report.output_bytes

        loose = tmp_path / "loose.gguf"
        unshared = bundle(base, loose, ["Q8_0", "Q4_K_M"], share=False)
        assert unshared.shared_bytes == 0
        assert loose.stat().st_size > out.stat().st_size

    def test_the_variants_are_readable_back(self, base, tmp_path):
        out = tmp_path / "b.gguf"
        bundle(base, out, ["Q8_0", "Q4_K_M", "IQ0.5_XXXL"])
        found = variants(out)
        assert [v.tier for v in found] == ["Q8_0", "Q4_K_M", "IQ0.5_XXXL"]
        assert found[0].default
        assert not any(v.default for v in found[1:])
        assert read_bundle_info(out)["default"] == "Q8_0"

    def test_an_ordinary_gguf_has_no_variants(self, base):
        """Not an error. Almost every GGUF in existence carries exactly
        one quantisation, and the answer to "how many" is "one"."""
        assert variants(base) == []
        assert read_bundle_info(base) == {}


class TestExtracting:
    @pytest.mark.parametrize("tier", ["Q8_0", "Q4_K_M", "INT2"])
    def test_what_comes_out_is_an_ordinary_gguf(self, base, tmp_path, tier):
        out = tmp_path / "b.gguf"
        bundle(base, out, ["Q8_0", "Q4_K_M", "INT2"])
        pulled = tmp_path / f"{slug_for(tier)}.gguf"
        extract(out, pulled, tier)

        model = GGUFFile.read(pulled)
        assert not any(t.name.startswith(PREFIX) for t in model.tensors)
        assert not any(k.startswith(PREFIX) for k in model.metadata)
        # Every tensor of the original, including the norms a variant
        # shared with the default rather than storing again.
        assert ({t.name for t in model.tensors}
                == {t.name for t in GGUFFile.read(base).tensors})

    def test_extracting_matches_quantising_directly(self, base, tmp_path):
        """A variant pulled out of a bundle has to be the same tensors,
        at the same types, as the same tier quantised on its own."""
        out = tmp_path / "b.gguf"
        bundle(base, out, ["Q8_0", "Q4_K_M"])
        extract(out, tmp_path / "pulled.gguf", "Q4_K_M")
        quantize_gguf(base, tmp_path / "direct.gguf", "Q4_K_M")

        def _types(path):
            return {t.name: int(t.ggml_type) for t in GGUFFile.read(path).tensors}

        assert _types(tmp_path / "pulled.gguf") == _types(tmp_path / "direct.gguf")

    def test_an_unknown_tier_says_what_is_there(self, base, tmp_path):
        out = tmp_path / "b.gguf"
        bundle(base, out, ["Q8_0", "Q4_K_M"])
        with pytest.raises(MultiQuantError, match="Q8_0, Q4_K_M"):
            extract(out, tmp_path / "x.gguf", "Q6_K")

    def test_extracting_from_a_plain_file_is_refused(self, base, tmp_path):
        with pytest.raises(MultiQuantError, match="Copy the file"):
            extract(base, tmp_path / "x.gguf", "Q8_0")

    def test_strip_keeps_the_default(self, base, tmp_path):
        out = tmp_path / "b.gguf"
        bundle(base, out, ["Q8_0", "Q4_K_M", "INT2"])
        assert strip(out, tmp_path / "plain.gguf") == 2
        model = GGUFFile.read(tmp_path / "plain.gguf")
        assert not any(t.name.startswith(PREFIX) for t in model.tensors)
        assert variants(tmp_path / "plain.gguf") == []


class TestTheDraftGenerations:
    def test_dflash1_writes_a_model_not_a_namespace(self, base, tmp_path):
        """The whole difference from dflash2: blocks numbered from zero
        under the ordinary names, because this is a model in its own
        right."""
        from hypernix.quant.dflash1 import derive

        out = tmp_path / "draft.gguf"
        report = derive(base, out, depth=0.5, quant="Q4_0")
        model = GGUFFile.read(out)
        names = {t.name for t in model.tensors}
        assert not any(name.startswith("dflash") for name in names)
        assert "blk.0.attn_q.weight" in names
        assert report.plan is not None
        highest = max(
            int(n.split(".")[1]) for n in names if n.startswith("blk.")
        )
        assert highest == len(report.plan.layers) - 1

    def test_the_block_count_is_rewritten(self, base, tmp_path):
        """llama.cpp allocates from this key before looking for a single
        tensor. Left at the base's value the draft does not load at
        all."""
        from hypernix.quant.dflash1 import derive

        out = tmp_path / "draft.gguf"
        report = derive(base, out, depth=0.5)
        model = GGUFFile.read(out)
        assert model.metadata["llama.block_count"] == len(report.plan.layers)

    def test_the_tokenizer_comes_across(self, base, tmp_path):
        """A draft proposes token ids. If the two files disagree about
        which id is which string, every proposal is rejected."""
        from hypernix.quant.dflash1 import derive

        out = tmp_path / "draft.gguf"
        derive(base, out)
        model = GGUFFile.read(out)
        assert model.metadata["tokenizer.ggml.model"] == "gpt2"
        assert model.metadata["tokenizer.ggml.tokens"] == ["a", "b", "c"]

    def test_a_base_with_no_tokenizer_is_refused(self, tmp_path):
        """The refusal that matters most, because the failure it prevents
        has no symptom: generation is just slower."""
        from hypernix.quant.dflash1 import Dflash1Error, derive

        naked = _model(tmp_path / "naked.gguf", tokenizer=False)
        with pytest.raises(Dflash1Error, match="no tokenizer metadata"):
            derive(naked, tmp_path / "draft.gguf")
        # And the escape hatch works, for fixtures like this one.
        derive(naked, tmp_path / "draft.gguf", require_tokenizer=False)

    def test_the_draft_is_much_smaller_than_the_base(self, base, tmp_path):
        from hypernix.quant.dflash1 import derive

        report = derive(base, tmp_path / "draft.gguf", depth=0.5, quant="Q4_0")
        assert 0 < report.ratio < 0.8

    def test_the_info_round_trips(self, base, tmp_path):
        from hypernix.quant.dflash1 import derive, read_draft_info

        out = tmp_path / "draft.gguf"
        report = derive(base, out, depth=0.5, quant="Q4_0", draft_tokens=6)
        info = read_draft_info(out)
        assert info["quant"] == "Q4_0"
        assert info["draft_tokens"] == 6
        assert info["source_block_count"] == 4
        assert list(info["layer_map"]) == list(report.plan.layers)

    def test_a_plain_model_reports_no_draft_info(self, base):
        from hypernix.quant.dflash1 import read_draft_info

        assert read_draft_info(base) == {}


class TestTheBlockTableCoversEverything:
    @pytest.mark.parametrize("tier", sorted(TIER_TYPES))
    def test_every_tier_has_a_block_shape(self, tier):
        """A tier whose type id is not in the table cannot have its
        tensor sizes checked, so `tensor_nbytes` refuses rather than
        guessing -- which means the tier cannot be written at all."""
        assert type_block_size(TIER_TYPES[tier][0]) == 256

    @pytest.mark.parametrize("width", sorted(WIDTHS))
    def test_every_width_has_a_block_shape(self, width):
        ggml_type, width_bytes = WIDTHS[width]
        assert _BLOCK_SHAPE[ggml_type] == (1, width_bytes)

    def test_an_unknown_target_lists_what_is_available(self):
        with pytest.raises(HyprslugError, match="hyprslug writes"):
            resolve_target("Q9_ULTRA")
