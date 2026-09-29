"""Running a sub-bit GGUF, which is the half that was missing.

The IQ0.x tiers have been real quantisations since 0.72.3 pt 2 — the
tensors genuinely carry 0.56 bits per weight, the container is a
well-formed GGUF. What they were not was *runnable*. Type ids at 200 and
above are unknown to every llama.cpp, and the reference ``gguf`` Python
reader rejects them outright, so the file was correct, small, and had
nowhere to go. "It is a real quantisation" is not much comfort when
nothing will load it.

So the tests here build an actual llama-architecture model, quantise it
through the whole ladder, and *run* each result. Two things are being
separated deliberately:

* **Does the quantiser work?** Measured at the weights, where the answer
  is arithmetic: IQ0.5 stores 2 signs of every 4, so 75% of signs should
  survive, and if it is 50% the packing is broken.
* **Is the model any good?** Measured at the logits, where the answer at
  half a bit is "no" and is *supposed* to be. A test that demanded good
  output from a 0.5-bit model would be demanding the impossible, and the
  only way to pass it would be to stop quantising.

Confusing the two is how a quantiser ends up secretly not quantising.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from hypernix.models import hnxrun
from hypernix.models.hnxtokenizer import tokenizer_from_metadata
from hypernix.quant.gguf import GGMLType, GGUFWriter
from hypernix.quant.hyprslug import quantize_gguf

# N_EMBD and N_FF are multiples of 256 because that is the sub-bit block
# size, and GGML quantises row by row: a tensor whose ne[0] is not a
# multiple of the block size cannot be that type at all. llama.cpp says
# so on load --
#
#     tensor '...' of type 202 (IQ0.5_XXXL) has 4 elements per row, not
#     a multiple of block size (256)
#
# -- and until 0.72.4.post16 this fixture used N_EMBD=64, so every
# tensor in it was 64 elements per row. hyprslug packed them anyway
# (it checked the element *total*, which 64x256 satisfies) and the
# round trip passed because hnxrun decoded them the same wrong way.
# Both halves agreed with each other and neither agreed with llama.cpp,
# which is the one reader that matters for a GGUF.
N_LAYER, N_EMBD, N_HEAD, N_KV, N_FF, VOCAB = 2, 256, 4, 2, 512, 256
HEAD_DIM = N_EMBD // N_HEAD

SUB_BIT_TIERS = ["IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL"]

#: Fraction of signs each tier should preserve, from its own design: it
#: stores ``kept`` of every ``group`` and the rest are reconstructed by
#: repeating the last stored one, which is right half the time.
EXPECTED_SIGN_ACCURACY = {
    "IQ0.9_L": 0.9375,      # 7 of 8 stored, + half of the remaining 1/8
    "IQ0.75_M": 0.875,      # 3 of 4
    "IQ0.5_XXXL": 0.75,     # 2 of 4
}


def _write_model(path: Path, *, seed: int = 0, tokenizer: bool = False) -> Path:
    """A real llama-architecture GGUF: every tensor the graph needs.

    Small enough to run in a test and shaped like the thing it stands
    for -- grouped-query attention included, since a KV head count that
    differs from the query head count is where a forward pass that
    "works" on the easy case falls over.
    """
    rng = np.random.default_rng(seed)
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.name", "tiny")
    writer.set_metadata("llama.block_count", N_LAYER)
    writer.set_metadata("llama.embedding_length", N_EMBD)
    writer.set_metadata("llama.attention.head_count", N_HEAD)
    writer.set_metadata("llama.attention.head_count_kv", N_KV)
    writer.set_metadata("llama.feed_forward_length", N_FF)
    writer.set_metadata("llama.context_length", 256)
    writer.set_metadata("llama.attention.layer_norm_rms_epsilon", 1e-5)
    writer.set_metadata("llama.rope.freq_base", 10000.0)
    if tokenizer:
        # A byte-level vocabulary: enough to encode and decode real text
        # without shipping a merge table.
        writer.set_metadata("tokenizer.ggml.model", "gpt2")
        tokens = [chr(256 + i) for i in range(VOCAB)]
        for byte in range(256):
            tokens[byte] = _byte_char(byte)
        writer.set_metadata("tokenizer.ggml.tokens", tokens)
        writer.set_metadata("tokenizer.ggml.merges", [])
        writer.set_metadata("tokenizer.ggml.bos_token_id", 1)
        writer.set_metadata("tokenizer.ggml.eos_token_id", 2)

    # GGUF shape is (n_input, n_output) -- fastest dimension first.
    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (N_EMBD, VOCAB),
        "output_norm.weight": (N_EMBD,),
        "output.weight": (N_EMBD, VOCAB),
    }
    for index in range(N_LAYER):
        shapes[f"blk.{index}.attn_norm.weight"] = (N_EMBD,)
        shapes[f"blk.{index}.attn_q.weight"] = (N_EMBD, N_HEAD * HEAD_DIM)
        shapes[f"blk.{index}.attn_k.weight"] = (N_EMBD, N_KV * HEAD_DIM)
        shapes[f"blk.{index}.attn_v.weight"] = (N_EMBD, N_KV * HEAD_DIM)
        shapes[f"blk.{index}.attn_output.weight"] = (N_HEAD * HEAD_DIM, N_EMBD)
        shapes[f"blk.{index}.ffn_norm.weight"] = (N_EMBD,)
        shapes[f"blk.{index}.ffn_gate.weight"] = (N_EMBD, N_FF)
        shapes[f"blk.{index}.ffn_up.weight"] = (N_EMBD, N_FF)
        shapes[f"blk.{index}.ffn_down.weight"] = (N_FF, N_EMBD)

    payload = {}
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        values = (
            np.ones(count, np.float32)
            if name.endswith("norm.weight")
            else rng.normal(0.0, 0.08, count).astype(np.float32)
        )
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = struct.pack(f"<{count}f", *values.tolist())
    writer.write(lambda tensor: payload[tensor.name])
    return path


def _byte_char(byte: int) -> str:
    from hypernix.models.hnxtokenizer import _BYTE_ENCODER

    return _BYTE_ENCODER[byte]


@pytest.fixture(scope="module")
def base_model(tmp_path_factory):
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("hnxrun")
    return _write_model(directory / "tiny.f32.gguf")


@pytest.fixture(scope="module")
def quantised(tmp_path_factory, base_model):
    """Every sub-bit tier, plus an upstream one for comparison."""
    directory = tmp_path_factory.mktemp("hnxrun-quants")
    built = {}
    for tier in [*SUB_BIT_TIERS, "Q4_K_M", "Q8_0"]:
        out = directory / f"tiny.{tier}.gguf"
        quantize_gguf(
            base_model, out, tier, quantize_embeddings=True, quantize_output=True
        )
        built[tier] = out
    return built


class TestItLoadsWhatNothingElseCan:
    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_a_sub_bit_model_loads(self, quantised, tier):
        model = hnxrun.load_model(quantised[tier])
        assert model.sub_bit is True
        assert model.config.block_count == N_LAYER
        assert model.config.vocab_size == VOCAB

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_every_tensor_came_back_finite(self, quantised, tier):
        """A dequantiser that divides by a zero scale produces NaN, and a
        model of NaN loads perfectly and generates nothing."""
        model = hnxrun.load_model(quantised[tier], materialize=True)
        for name, tensor in model.tensors.items():
            assert torch.isfinite(tensor).all(), name

    def test_the_reference_gguf_reader_cannot_open_these(self, quantised):
        """Not a defect -- the point. The type ids are deliberately
        outside upstream's range so a stock loader refuses the file by
        name instead of reading a 0.5-bit tensor as Q4_K. This pins that
        it is still true, because if upstream ever allocates 202 the
        collision would be silent and catastrophic.
        """
        gguf = pytest.importorskip("gguf")
        with pytest.raises(Exception):  # noqa: B017 - the library's own type varies
            gguf.GGUFReader(str(quantised["IQ0.5_XXXL"]))

    def test_the_reference_reader_opens_an_upstream_quant(self, quantised):
        """The control. If this also failed, the file would be malformed
        rather than merely carrying a type upstream does not know."""
        gguf = pytest.importorskip("gguf")
        reader = gguf.GGUFReader(str(quantised["Q8_0"]))
        assert len(reader.tensors) > 0

    def test_a_packed_weight_dequantises_to_the_same_thing(self, quantised):
        """to_dense() and materialize=True must not disagree, or the
        chunked path is quietly returning different weights."""
        packed = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        dense = hnxrun.load_model(quantised["IQ0.5_XXXL"], materialize=True)
        for name, weight in packed.tensors.items():
            got = weight.to_dense() if hasattr(weight, "to_dense") else weight
            assert torch.equal(got, dense.tensors[name]), name

    def test_the_shape_is_read_the_right_way_round(self, base_model):
        """GGUF stores the fastest dimension first, so a weight is
        (n_input, n_output) on disk and (out, in) in a linear layer.
        Backwards, the model loads cleanly and multiplies the wrong way."""
        model = hnxrun.load_model(base_model)
        assert tuple(model.tensors["token_embd.weight"].shape) == (VOCAB, N_EMBD)
        assert tuple(model.tensors["blk.0.ffn_up.weight"].shape) == (N_FF, N_EMBD)
        assert tuple(model.tensors["blk.0.ffn_down.weight"].shape) == (N_EMBD, N_FF)


class TestTheQuantiserItself:
    """Measured at the weights, where the answer is arithmetic."""

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_the_signs_survive_at_the_designed_rate(self, base_model, quantised, tier):
        """The number that says the packing is doing what it claims.

        IQ0.5 stores two signs of every four and reconstructs the other
        two by repeating the last stored one -- right half the time -- so
        75% of signs should come back. 50% would mean the stored signs
        are landing on the wrong weights, which is invisible in file size
        and fatal to the model.
        """
        original = hnxrun.load_model(base_model).tensors["blk.0.ffn_up.weight"]
        restored = hnxrun.load_model(
            quantised[tier], materialize=True
        ).tensors["blk.0.ffn_up.weight"]
        accuracy = float((torch.sign(original) == torch.sign(restored)).float().mean())
        expected = EXPECTED_SIGN_ACCURACY[tier]
        assert accuracy == pytest.approx(expected, abs=0.06), (
            f"{tier} kept {accuracy:.1%} of signs; its packing implies {expected:.1%}"
        )

    def test_a_wider_tier_keeps_more_signs(self, base_model, quantised):
        original = hnxrun.load_model(base_model).tensors["blk.0.ffn_up.weight"]

        def _accuracy(tier):
            restored = hnxrun.load_model(
                quantised[tier], materialize=True
            ).tensors["blk.0.ffn_up.weight"]
            return float((torch.sign(original) == torch.sign(restored)).float().mean())

        assert _accuracy("IQ0.9_L") > _accuracy("IQ0.75_M") > _accuracy("IQ0.5_XXXL")

    def test_the_file_really_is_smaller(self, base_model, quantised):
        sizes = {t: quantised[t].stat().st_size for t in SUB_BIT_TIERS}
        assert base_model.stat().st_size > sizes["IQ0.9_L"] * 10
        assert sizes["IQ0.9_L"] > sizes["IQ0.75_M"] > sizes["IQ0.5_XXXL"]


class TestItStaysSubBitInMemory:
    """The half that a runtime is most likely to quietly give away.

    Dequantising to float32 at load time turns 0.56 bits into 32 --
    *larger* than the F16 model the quantisation was made from. The file
    would still be small and the tier would be pointless, which is the
    failure that is easy to ship because everything still works.
    """

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_the_weights_stay_packed(self, quantised, tier):
        model = hnxrun.load_model(quantised[tier])
        assert model.packed_in_memory > 0
        packed = [
            name
            for name, weight in model.tensors.items()
            if isinstance(weight, hnxrun.PackedWeight)
        ]
        assert "token_embd.weight" in packed
        assert "blk.0.ffn_up.weight" in packed

    @pytest.mark.parametrize(
        ("tier", "ceiling"),
        [("IQ0.5_XXXL", 1.0), ("IQ0.75_M", 1.0), ("IQ0.9_L", 1.1)],
    )
    def test_resident_cost_is_about_a_bit_per_weight(self, quantised, tier, ceiling):
        """The claim, as a number.

        Slightly above the packing's own rate, because the norms are
        one-dimensional and stay in float32 -- they are a rounding error
        of a real model's size and most of the gap on a toy one.
        """
        model = hnxrun.load_model(quantised[tier])
        assert model.resident_bits_per_weight < ceiling, model.describe()

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_memory_tracks_the_file_rather_than_the_original(self, quantised, tier):
        model = hnxrun.load_model(quantised[tier])
        assert model.resident_bytes < model.file_bytes

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_materialising_costs_what_it_says(self, quantised, tier):
        """The opt-out is real, and expensive, and the report says so."""
        packed = hnxrun.load_model(quantised[tier])
        dense = hnxrun.load_model(quantised[tier], materialize=True)
        assert dense.resident_bits_per_weight == pytest.approx(32.0)
        assert dense.resident_bytes > packed.resident_bytes * 20
        assert dense.packed_in_memory == 0

    def test_a_narrower_tier_costs_less_memory(self, quantised):
        def _resident(tier):
            return hnxrun.load_model(quantised[tier]).resident_bytes

        assert _resident("Q8_0") > _resident("Q4_K_M") > _resident("IQ0.9_L")
        assert _resident("IQ0.9_L") > _resident("IQ0.75_M") > _resident("IQ0.5_XXXL")

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_the_weights_themselves_are_bit_identical(self, quantised, tier):
        """The two paths decode the same bytes with the same decoder, so
        at the weights the answer is exact. Any difference here is a bug
        in the slicing rather than a rounding question -- and it would be
        invisible, because a mis-sliced weight gives a plausible model."""
        packed = hnxrun.load_model(quantised[tier])
        dense = hnxrun.load_model(quantised[tier], materialize=True)
        for name, weight in packed.tensors.items():
            if isinstance(weight, hnxrun.PackedWeight):
                assert torch.equal(weight.to_dense(), dense.tensors[name]), name

    def test_the_logits_agree_to_float32_rounding_not_exactly(self, quantised):
        """And the distinction is the honest one.

        The packed path folds the input and sums the weight a chunk at a
        time, so the same terms are added in a different order. Float32
        addition is not associative, so the logits differ in the last
        bits. Claiming "bit-identical" here would be a claim that breaks
        the first time anyone runs a model wide enough for two chunks --
        which is every real model.
        """
        packed = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        dense = hnxrun.load_model(quantised["IQ0.5_XXXL"], materialize=True)
        prompt = [1, 5, 9, 13]
        first, _ = hnxrun.forward(packed, prompt)
        second, _ = hnxrun.forward(dense, prompt)
        assert torch.allclose(first, second, rtol=1e-4, atol=1e-4)

    def test_generation_is_the_same_either_way(self, quantised):
        packed = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        dense = hnxrun.load_model(quantised["IQ0.5_XXXL"], materialize=True)
        assert hnxrun.generate_tokens(
            packed, [1, 5, 9], max_new_tokens=6
        ) == hnxrun.generate_tokens(dense, [1, 5, 9], max_new_tokens=6)

    def test_row_groups_handle_a_row_narrower_than_a_block(self, quantised):
        """A 64-wide layer packed in 256-element blocks puts four rows in
        one block, so a row is not sliceable on its own. Getting this
        wrong reads the wrong bytes and produces a plausible, wrong
        model rather than an error."""
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        weight = model.tensors["blk.0.ffn_up.weight"]
        assert isinstance(weight, hnxrun.PackedWeight)
        assert weight.columns == N_EMBD  # 64, narrower than the 256 block
        dense = hnxrun.load_model(
            quantised["IQ0.5_XXXL"], materialize=True
        ).tensors["blk.0.ffn_up.weight"]
        for start in (0, 1, 3, 7):
            assert torch.equal(weight.rows_slice(start, start + 2), dense[start:start + 2])

    def test_an_embedding_lookup_unpacks_only_what_it_needs(self, quantised):
        """The table is usually the largest tensor in the file and a
        prompt touches a handful of its rows."""
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        table = model.tensors["token_embd.weight"]
        dense = hnxrun.load_model(
            quantised["IQ0.5_XXXL"], materialize=True
        ).tensors["token_embd.weight"]
        ids = [7, 3, 7, 100]
        assert torch.equal(table.select(ids), dense[ids])

    def test_describe_reports_the_resident_cost(self, quantised):
        text = hnxrun.load_model(quantised["IQ0.5_XXXL"]).describe()
        assert "bits/weight" in text
        assert "still packed" in text


class TestTheFoldedMatmul:
    """The packed path never widens a weight to one float per element.

    A dropped sign repeats its group's last stored one, so the group's
    contribution factors: fold the input to match and the dot product
    runs against the stored signs alone. It is the difference between
    holding the model at half a bit and *running* it at half a bit, and
    it is the kind of algebra that is easy to get subtly wrong -- an
    off-by-one in which position absorbs the tail produces a model that
    loads, runs, and is quietly not the model in the file.

    So every case here is checked against the dense product: the same
    signs expanded the obvious way and multiplied the obvious way, which
    is precisely what the fold claims to be equivalent to. The sign
    decode is shared -- what is under test is the algebra on top of it.
    """

    SHAPES = [(8, 256), (17, 512), (64, 64), (5, 1024), (256, 32)]
    PACKINGS = {200: "sign_scale_l", 201: "pair_code_m", 202: "quad_code_xxxl"}

    def _packed(self, kind, rows, columns, seed=0):
        from hypernix.quant import subbit

        rng = np.random.default_rng(seed)
        weights = rng.standard_normal(rows * columns).astype(np.float32)
        raw = subbit.quantize_tensor(weights.tolist(), self.PACKINGS[kind])
        return hnxrun.PackedWeight(raw, kind, (rows, columns), "cpu")

    @pytest.mark.parametrize("kind", sorted(PACKINGS))
    @pytest.mark.parametrize(("rows", "columns"), SHAPES)
    def test_it_agrees_with_the_dense_product(self, kind, rows, columns):
        weight = self._packed(kind, rows, columns)
        dense = weight.to_dense()
        for shape in [(columns,), (3, columns), (2, 4, columns)]:
            x = torch.randn(*shape, dtype=torch.float32)
            got = weight.matmul_t(x)
            want = x @ dense.T
            assert got.shape == want.shape
            assert torch.allclose(got, want, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("kind", sorted(PACKINGS))
    def test_it_agrees_when_the_rows_need_several_chunks(self, kind):
        """One chunk hides every slicing bug there is."""
        weight = self._packed(kind, 64, 256)
        dense = weight.to_dense()
        x = torch.randn(3, 256, dtype=torch.float32)
        want = x @ dense.T
        for chunk_rows in (1, 2, 7, 16, 1000):
            got = weight.matmul_t(x, chunk_rows=chunk_rows)
            assert torch.allclose(got, want, rtol=2e-5, atol=2e-5), chunk_rows

    @pytest.mark.parametrize("kind", sorted(PACKINGS))
    def test_the_fold_is_actually_taken(self, kind):
        """Otherwise every assertion above passes on the slow path and
        the optimisation is untested."""
        assert self._packed(kind, 8, 256)._packing == self.PACKINGS[kind]

    def test_an_upstream_quant_does_not_fold(self):
        """There are no dropped signs to reconstruct in a Q4_K block, so
        there is nothing to fold and the general path has to stay."""
        from hypernix.quant import llamaquants

        rng = np.random.default_rng(0)
        raw = llamaquants.quantize_array(
            rng.standard_normal(256 * 4).astype(np.float32), "Q4_K"
        )
        weight = hnxrun.PackedWeight(
            raw, llamaquants.FORMATS["Q4_K"].ggml_type, (4, 256), "cpu"
        )
        assert weight._packing is None

    def test_a_row_that_splits_a_sign_group_does_not_fold(self):
        """The fold rewrites ``x`` per group, so a group straddling two
        rows would mix two rows' inputs -- silently, and wrongly. No real
        model is shaped like this; the guard is for the one that is."""
        from hypernix.quant import subbit

        rng = np.random.default_rng(0)
        raw = subbit.quantize_tensor(
            rng.standard_normal(1024).astype(np.float32).tolist(), "sign_scale_l"
        )
        weight = hnxrun.PackedWeight(raw, 200, (256, 4), "cpu")
        assert weight._packing is None
        x = torch.randn(2, 4, dtype=torch.float32)
        assert torch.allclose(weight.matmul_t(x), x @ weight.to_dense().T)

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_it_never_allocates_a_float_per_weight(self, quantised, tier):
        """The claim the fold exists to make, measured where it counts.

        ``ffn_down`` is the widest weight in the model; decoding it the
        old way meant one float32 per element, which for a 0.5-bit tensor
        is 57x what the tensor costs. The folded decode is ``kept/group``
        of that and nothing else is materialised.
        """
        from hypernix.quant import subbit

        model = hnxrun.load_model(quantised[tier])
        weight = model.tensors["blk.0.ffn_down.weight"]
        spec = subbit.PACKINGS[weight._packing]
        signs = weight._folded_signs(0, weight.rows)
        assert signs.numel() == weight.rows * weight.columns * spec.kept // spec.group
        assert signs.numel() < weight.rows * weight.columns


class TestTheCacheBudget:
    """Between "costs what the file costs" and "runs at float32 speed".

    Neither end suits every machine, and the interesting question for one
    in between is not *whether* it pinned something but *what it now
    costs* -- so the report has to move with the budget rather than
    describing the packed case forever.
    """

    @pytest.fixture(scope="class")
    def partial_budget(self, quantised):
        """A budget that affords some packed tensors and not all of them.

        This was a hard-coded 100_000, chosen for a fixture whose largest
        packed tensor was 32 KiB. When the fixture grew, every packed
        tensor became larger than the budget, so nothing was pinned --
        and three tests that assert pinning costs memory started
        comparing a number with itself and passing on the tie until an
        unrelated change made them fail. Deriving it from the model keeps
        "partial" true whatever the fixture's dimensions are.
        """
        sizes = [
            weight.dense_bytes
            for weight in hnxrun.load_model(quantised["IQ0.5_XXXL"]).tensors.values()
            if isinstance(weight, hnxrun.PackedWeight)
        ]
        assert len(sizes) > 1, "a budget cannot be partial over one tensor"
        return sum(sizes) // 2

    def test_no_budget_pins_nothing(self, quantised):
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        assert model.pinned_in_memory == 0

    def test_a_budget_pins_the_largest_first(self, quantised):
        """Every forward pass touches every tensor exactly once, so there
        is no locality to exploit: the only question is how much decode
        work a byte of budget buys, and the biggest tensor buys most.

        Asserted by *size* rather than by name: several tensors tie for
        largest, and which of them the spender reaches first is a
        tie-break in a sort, not a promise. Naming one made this test
        agree with the implementation by luck.
        """
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        packed = {
            name: weight.dense_bytes
            for name, weight in model.tensors.items()
            if isinstance(weight, hnxrun.PackedWeight)
        }
        biggest = max(packed.values())
        budget = hnxrun.load_model(quantised["IQ0.5_XXXL"], cache_bytes=biggest)
        pinned = [
            name for name, weight in budget.tensors.items()
            if isinstance(weight, hnxrun.PackedWeight) and weight.pinned
        ]
        assert len(pinned) == 1, f"a budget for one tensor pinned {pinned}"
        assert packed[pinned[0]] == biggest

    def test_the_reported_cost_moves_with_the_budget(self, quantised, partial_budget):
        packed = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        cached = hnxrun.load_model(quantised["IQ0.5_XXXL"], cache_bytes=partial_budget)
        assert cached.resident_bytes > packed.resident_bytes
        assert cached.resident_bits_per_weight > packed.resident_bits_per_weight
        assert "pinned" in cached.describe()

    def test_a_generous_budget_still_beats_materialising(self, quantised, partial_budget):
        """Norms and any tensor the budget could not afford stay as they
        were, so this is a dial and not a second switch."""
        cached = hnxrun.load_model(quantised["IQ0.5_XXXL"], cache_bytes=partial_budget)
        dense = hnxrun.load_model(quantised["IQ0.5_XXXL"], materialize=True)
        assert cached.resident_bytes < dense.resident_bytes

    def test_pinning_does_not_change_the_answer(self, quantised, partial_budget):
        """The whole dial is worthless if the two ends disagree."""
        packed = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        cached = hnxrun.load_model(quantised["IQ0.5_XXXL"], cache_bytes=partial_budget)
        first, _ = hnxrun.forward(packed, [1, 5, 9, 13])
        second, _ = hnxrun.forward(cached, [1, 5, 9, 13])
        assert torch.allclose(first, second, rtol=1e-4, atol=1e-4)

    def test_unpinning_gives_the_memory_back(self, quantised, partial_budget):
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"], cache_bytes=partial_budget)
        before = model.resident_bytes
        for weight in model.tensors.values():
            if isinstance(weight, hnxrun.PackedWeight):
                weight.unpin()
        assert model.resident_bytes < before
        assert model.pinned_in_memory == 0


class TestItRuns:
    PROMPT = [1, 5, 9, 13, 21]

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_the_forward_pass_produces_usable_logits(self, quantised, tier):
        model = hnxrun.load_model(quantised[tier])
        logits, _cache = hnxrun.forward(model, self.PROMPT)
        assert tuple(logits.shape) == (len(self.PROMPT), VOCAB)
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("tier", SUB_BIT_TIERS)
    def test_it_generates_tokens(self, quantised, tier):
        """The whole claim, in one assertion: a 0.5-bit GGUF produces
        tokens."""
        model = hnxrun.load_model(quantised[tier])
        produced = hnxrun.generate_tokens(model, self.PROMPT, max_new_tokens=6)
        assert len(produced) == 6
        assert all(0 <= token < VOCAB for token in produced)

    def test_greedy_generation_is_deterministic(self, quantised):
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        first = hnxrun.generate_tokens(model, self.PROMPT, max_new_tokens=6)
        second = hnxrun.generate_tokens(model, self.PROMPT, max_new_tokens=6)
        assert first == second

    def test_sampling_with_a_seed_is_reproducible(self, quantised):
        model = hnxrun.load_model(quantised["IQ0.9_L"])
        kwargs = {"max_new_tokens": 6, "temperature": 0.8, "seed": 7}
        assert hnxrun.generate_tokens(
            model, self.PROMPT, **kwargs
        ) == hnxrun.generate_tokens(model, self.PROMPT, **kwargs)

    def test_the_kv_cache_agrees_with_a_full_recompute(self, quantised):
        """The classic way this goes quietly wrong: a cache that drifts
        from the uncached path gives output that is plausible and not
        what the model would have said."""
        model = hnxrun.load_model(quantised["IQ0.5_XXXL"])
        sequence = [1, 5, 9, 13, 21, 34]

        full, _ = hnxrun.forward(model, sequence)
        stepped, cache = hnxrun.forward(model, sequence[:3])
        for position, token in enumerate(sequence[3:], start=3):
            stepped, cache = hnxrun.forward(
                model, [token], cache=cache, start_position=position
            )
        assert torch.allclose(full[-1], stepped[-1], atol=1e-4)

    def test_grouped_query_attention_is_actually_grouped(self, base_model):
        model = hnxrun.load_model(base_model)
        assert model.config.head_count == N_HEAD
        assert model.config.head_count_kv == N_KV
        assert model.config.kv_groups == N_HEAD // N_KV

    def test_attention_is_causal(self, base_model):
        """Changing a later token must not move an earlier position's
        logits. A mask off by one is invisible in output that already
        looks like noise."""
        model = hnxrun.load_model(base_model)
        first, _ = hnxrun.forward(model, [1, 5, 9, 13])
        second, _ = hnxrun.forward(model, [1, 5, 9, 99])
        assert torch.allclose(first[:3], second[:3], atol=1e-5)
        assert not torch.allclose(first[3], second[3], atol=1e-3)


class TestFidelityDegradesAsItShould:
    """At the logits, where half a bit is supposed to be bad."""

    PROMPT = [1, 5, 9, 13, 21]

    def _agreement(self, reference, model) -> float:
        got, _ = hnxrun.forward(model, self.PROMPT)
        return float(
            np.corrcoef(reference[-1].numpy(), got[-1].detach().numpy())[0, 1]
        )

    def test_an_upstream_quant_tracks_the_original_closely(self, base_model, quantised):
        base = hnxrun.load_model(base_model)
        reference, _ = hnxrun.forward(base, self.PROMPT)
        agreement = self._agreement(reference, hnxrun.load_model(quantised["Q8_0"]))
        assert agreement > 0.95, (
            f"Q8_0 is meant to be near-lossless and agreed only {agreement:.2f}"
        )

    def test_the_sub_bit_tiers_are_ordered(self, base_model, quantised):
        """Not "good" -- ordered. More bits must not produce a worse
        model, and if they do the packing is wrong somewhere."""
        base = hnxrun.load_model(base_model)
        reference, _ = hnxrun.forward(base, self.PROMPT)
        wide = self._agreement(reference, hnxrun.load_model(quantised["Q4_K_M"]))
        narrow = self._agreement(reference, hnxrun.load_model(quantised["IQ0.5_XXXL"]))
        assert wide > narrow


class TestTheRefusals:
    def test_a_missing_model_says_so(self, tmp_path):
        with pytest.raises(hnxrun.HnxRunError, match="No such model"):
            hnxrun.load_model(tmp_path / "absent.gguf")

    def test_a_file_with_no_embedding_is_not_a_model(self, tmp_path):
        writer = GGUFWriter(tmp_path / "fragment.gguf")
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", (64, 4), int(GGMLType.F32))
        writer.write(lambda _t: struct.pack("<256f", *([0.0] * 256)))
        with pytest.raises(hnxrun.HnxRunError, match="token_embd"):
            hnxrun.load_model(tmp_path / "fragment.gguf")

    def test_an_architecture_it_does_not_implement_is_refused(self, tmp_path):
        """Running the llama graph over a model that is not one produces
        confident nonsense rather than an error, so the name is checked."""
        writer = GGUFWriter(tmp_path / "other.gguf")
        writer.set_metadata("general.architecture", "mamba")
        writer.add_tensor("token_embd.weight", (64, 8), int(GGMLType.F32))
        writer.write(lambda _t: struct.pack("<512f", *([0.01] * 512)))
        with pytest.raises(hnxrun.HnxRunError, match="llama-family"):
            hnxrun.load_model(tmp_path / "other.gguf")

    def test_an_out_of_range_token_is_refused(self, base_model):
        model = hnxrun.load_model(base_model)
        with pytest.raises(hnxrun.HnxRunError, match="out of range"):
            hnxrun.forward(model, [VOCAB + 5])

    def test_an_empty_prompt_is_refused(self, base_model):
        model = hnxrun.load_model(base_model)
        with pytest.raises(hnxrun.HnxRunError, match="no tokens"):
            hnxrun.generate_tokens(model, [])

    def test_text_generation_without_a_tokenizer_says_which_is_missing(self, base_model):
        """Guessing an encoding produces output that reads as a broken
        model rather than as a missing tokenizer."""
        with pytest.raises(hnxrun.HnxRunError, match="no tokenizer"):
            hnxrun.generate_text(base_model, "hello")


class TestTextInAndOut:
    def test_a_model_carrying_its_tokenizer_generates_text(self, tmp_path):
        """End to end from a string, which is what anyone actually
        wants: quantise to half a bit, then talk to it."""
        source = _write_model(tmp_path / "tok.f32.gguf", tokenizer=True)
        out = tmp_path / "tok.iq05.gguf"
        quantize_gguf(
            source, out, "IQ0.5_XXXL", quantize_embeddings=True, quantize_output=True
        )
        text = hnxrun.generate_text(out, "hello", max_new_tokens=4)
        assert isinstance(text, str)

    def test_the_tokenizer_round_trips_ascii(self, tmp_path):
        source = _write_model(tmp_path / "tok.gguf", tokenizer=True)
        model = hnxrun.load_model(source)
        assert model.tokenizer is not None
        ids = model.tokenizer.encode("hello world", add_bos=False)
        assert model.tokenizer.decode(ids) == "hello world"

    def test_a_file_without_tokenizer_metadata_reports_none(self, base_model):
        assert hnxrun.load_model(base_model).tokenizer is None

    def test_tokenizer_from_metadata_needs_tokens(self):
        assert tokenizer_from_metadata({}) is None
        assert tokenizer_from_metadata({"tokenizer.ggml.model": "gpt2"}) is None

    def test_a_sentencepiece_vocabulary_segments_by_score(self):
        """Viterbi over the scores, not longest-match-first: the greedy
        shortcut silently produces a different segmentation."""
        tokenizer = tokenizer_from_metadata({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["<unk>", "▁", "▁a", "b", "▁ab", "a"],
            "tokenizer.ggml.scores": [-100.0, -1.0, -3.0, -1.0, -1.5, -1.0],
        })
        assert tokenizer is not None and tokenizer.kind == "spm"
        # "▁ab" (-1.5) beats "▁a" + "b" (-3 + -1 = -4).
        assert tokenizer.encode("ab", add_bos=False) == [4]


class TestDescribe:
    def test_it_reports_the_architecture_and_the_packing(self, quantised):
        info = hnxrun.describe(quantised["IQ0.5_XXXL"])
        assert info["architecture"] == "llama"
        assert info["block_count"] == N_LAYER
        assert info["sub_bit"] is True
        assert "HNX_IQ0_5" in info["packed_as"]

    def test_an_upstream_quant_is_not_marked_sub_bit(self, quantised):
        info = hnxrun.describe(quantised["Q4_K_M"])
        assert info["sub_bit"] is False
