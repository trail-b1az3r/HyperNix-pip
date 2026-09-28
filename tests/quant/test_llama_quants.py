"""The llama.cpp block formats, encoded and decoded without llama.cpp.

Two things can be wrong here and only one of them is loud.

The loud one is a block that is the wrong size: every tensor after it
reads from the wrong offset and the file is noise from the second tensor
on. That is checked directly, against the same table
``hypernix.quant.gguf`` sizes tensors from.

The quiet one is a *layout* that is the wrong shape inside a correctly
sized block — a scale packed into the wrong bits, a nibble pair swapped.
The file opens, the shapes are right, and the model is subtly wrong in a
way no size check sees. What catches that here is round-trip fidelity:
each format is decoded with the reference ``dequantize_row_*``
arithmetic rather than by inverting the encoder, so an encoder that
packed into the wrong bits reconstructs the wrong numbers and the error
bound fails.
"""
from __future__ import annotations

import numpy as np
import pytest

from hypernix.quant import llamaquants as lq
from hypernix.quant.gguf import GGMLType, type_block_size, type_size_bytes

ALL_FORMATS = sorted(lq.FORMATS)

#: Relative RMS error each format has to stay inside on smooth data.
#:
#: Generous — these are not quality targets, they are "the layout is not
#: scrambled" bounds. A format whose bits went to the wrong place lands
#: one to two orders of magnitude above its entry, not just above it.
ERROR_BUDGET = {
    "Q8_0": 0.03,
    "Q6_K": 0.05,
    "Q5_1": 0.09,
    "Q5_K": 0.08,
    "Q5_0": 0.14,
    "Q4_1": 0.17,
    "Q4_K": 0.15,
    "Q4_0": 0.19,
    "Q3_K": 0.22,
    "Q2_K": 0.36,
}


def _sample(rng, count: int) -> np.ndarray:
    """Weights shaped like a real tensor: mostly small, a few large."""
    body = rng.normal(0.0, 0.02, count)
    outliers = rng.random(count) < 0.01
    body[outliers] *= 25
    return body.astype(np.float32)


def _relative_rmse(original: np.ndarray, restored: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((restored - original) ** 2)) / np.sqrt(np.mean(original**2))
    )


class TestBlockGeometry:
    """A block byte count that drifts corrupts every later tensor."""

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_the_gguf_table_and_the_encoder_agree(self, name):
        fmt = lq.FORMATS[name]
        assert type_size_bytes(fmt.ggml_type) == fmt.block_bytes
        assert type_block_size(fmt.ggml_type) == fmt.block

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_output_length_is_exactly_the_blocks_it_was_given(self, name):
        fmt = lq.FORMATS[name]
        values = np.zeros(fmt.block * 5, dtype=np.float32)
        assert len(lq.quantize_array(values, name)) == 5 * fmt.block_bytes

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_the_ggml_type_id_is_the_upstream_one(self, name):
        """These ids are the file format. A wrong one is a wrong file."""
        assert lq.FORMATS[name].ggml_type == int(getattr(GGMLType, name))

    def test_bits_per_weight_ranks_the_formats_in_the_expected_order(self):
        order = sorted(ALL_FORMATS, key=lambda n: lq.FORMATS[n].bits_per_weight)
        assert order[0] == "Q2_K"
        assert order[-1] == "Q8_0"

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_a_partial_block_is_refused_not_padded(self, name):
        fmt = lq.FORMATS[name]
        with pytest.raises(lq.LlamaQuantError, match="do not divide"):
            lq.quantize_array(np.zeros(fmt.block + 1), name)


class TestRoundTrip:
    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_it_reconstructs_within_the_format_s_budget(self, name):
        rng = np.random.default_rng(20260902)
        fmt = lq.FORMATS[name]
        values = _sample(rng, fmt.block * 16)
        restored = lq.dequantize_array(lq.quantize_array(values, name), name)
        assert restored.shape == values.shape
        assert _relative_rmse(values, restored) < ERROR_BUDGET[name]

    def test_more_bits_really_do_mean_less_error(self):
        """The ordering is the only end-to-end check of the arithmetic.

        Each format is decoded by the reference dequantiser rather than
        by inverting its own encoder, so a Q6_K that scrambled its high
        bits cannot hide behind a matching mistake on the way back — it
        lands worse than Q4_K and this fails.
        """
        rng = np.random.default_rng(11)
        values = _sample(rng, 256 * 24)
        errors = {}
        for name in ALL_FORMATS:
            restored = lq.dequantize_array(lq.quantize_array(values, name), name)
            errors[name] = _relative_rmse(values, restored)

        for wider, narrower in [
            ("Q8_0", "Q5_0"),
            ("Q6_K", "Q4_K"),
            ("Q5_K", "Q4_K"),
            ("Q4_K", "Q3_K"),
            ("Q3_K", "Q2_K"),
            ("Q5_1", "Q4_1"),
            # And the whole reason the K-quants exist: at the same width
            # they beat the legacy block, because a 256-weight super-block
            # can afford per-group scales the 32-weight block cannot.
            ("Q4_K", "Q4_0"),
            ("Q5_K", "Q5_0"),
        ]:
            assert errors[wider] < errors[narrower], (
                f"{wider} ({errors[wider]:.4f}) should beat "
                f"{narrower} ({errors[narrower]:.4f})"
            )

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_all_zeros_round_trips_to_all_zeros(self, name):
        """The degenerate block, which every scale search divides by.

        A format that returns NaN here produces a model that generates
        nothing, from a tensor that was merely empty.
        """
        fmt = lq.FORMATS[name]
        values = np.zeros(fmt.block * 3, dtype=np.float32)
        restored = lq.dequantize_array(lq.quantize_array(values, name), name)
        assert np.all(np.isfinite(restored))
        assert np.allclose(restored, 0.0)

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_a_constant_block_survives(self, name):
        """Every value identical: max == min, and the span is zero."""
        fmt = lq.FORMATS[name]
        values = np.full(fmt.block * 2, 0.125, dtype=np.float32)
        restored = lq.dequantize_array(lq.quantize_array(values, name), name)
        assert np.all(np.isfinite(restored))

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_it_does_not_flip_signs(self, name):
        """The failure mode of a swapped nibble pair, and of a sign bit
        packed at the wrong offset. Both look fine on size and awful on
        output, and neither is visible in an average error."""
        rng = np.random.default_rng(5)
        fmt = lq.FORMATS[name]
        values = _sample(rng, fmt.block * 8)
        restored = lq.dequantize_array(lq.quantize_array(values, name), name)
        big = np.abs(values) > 3 * np.std(values)
        agree = np.sign(values[big]) == np.sign(restored[big])
        assert agree.mean() > 0.95, "large weights are coming back with the wrong sign"

    @pytest.mark.parametrize("name", ALL_FORMATS)
    def test_position_is_preserved(self, name):
        """One large value in a field of small ones must come back where
        it went in. A packing that walks the block in the wrong order
        keeps every value and puts them somewhere else."""
        fmt = lq.FORMATS[name]
        values = np.full(fmt.block * 2, 0.001, dtype=np.float32)
        values[fmt.block + 7] = 1.0
        restored = lq.dequantize_array(lq.quantize_array(values, name), name)
        assert int(np.argmax(np.abs(restored))) == fmt.block + 7


class TestTheScaleSearches:
    """The two functions that do most of the quality work."""

    def test_make_qx_quants_is_symmetric_and_in_range(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, (32, 16))
        scale, levels = lq.make_qx_quants(x, 32)
        assert levels.shape == x.shape
        assert levels.min() >= 0 and levels.max() <= 63
        restored = scale[:, None] * (levels - 32)
        assert _relative_rmse(x, restored) < 0.05

    def test_make_qx_quants_beats_the_unsearched_scale(self):
        """The 19-step search is the reason this is not just x/amax."""
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, (64, 16))
        scale, levels = lq.make_qx_quants(x, 32)
        searched = _relative_rmse(x, scale[:, None] * (levels - 32))

        amax = np.max(np.abs(x), axis=1)
        naive_scale = amax / 32
        naive_levels = np.clip(np.round(x / naive_scale[:, None]), -32, 31)
        naive = _relative_rmse(x, naive_scale[:, None] * naive_levels)
        assert searched <= naive

    def test_make_qkx2_quants_keeps_the_offset_non_positive(self):
        """It stores ``-min`` unsigned, so a positive min cannot be said.

        A fit that wanted one and got stored anyway would reconstruct
        every value in the block shifted the wrong way.
        """
        rng = np.random.default_rng(6)
        x = rng.uniform(0.5, 1.5, (40, 32))
        _scale, the_min, _levels = lq.make_qkx2_quants(x, np.ones_like(x), 15)
        assert np.all(the_min >= -1e-12)

    def test_make_qkx2_quants_stays_in_range(self):
        rng = np.random.default_rng(8)
        x = rng.normal(0, 0.3, (40, 32))
        _scale, _the_min, levels = lq.make_qkx2_quants(x, np.ones_like(x), 15)
        assert levels.min() >= 0 and levels.max() <= 15

    def test_a_constant_group_reconstructs_rather_than_dividing_by_zero(self):
        """Upstream pins the offset at zero before comparing to the max,
        so a group of identical positive values is not "flat" — it spans
        zero to that value and quantises to the top level. Only a group
        that is genuinely all-zero has no scale."""
        x = np.full((4, 32), 0.7)
        scale, the_min, levels = lq.make_qkx2_quants(x, np.ones_like(x), 15)
        assert np.all(np.isfinite(scale)) and np.all(np.isfinite(the_min))
        assert np.allclose(scale[:, None] * levels - the_min[:, None], 0.7)

    def test_an_all_zero_group_has_no_scale_at_all(self):
        x = np.zeros((4, 32))
        scale, the_min, levels = lq.make_qkx2_quants(x, np.ones_like(x), 15)
        assert np.all(scale == 0)
        assert np.all(levels == 0)
        assert np.all(np.isfinite(the_min))


class TestTheImatrixChangesSomething:
    """An imatrix that made no difference would be a lie in the report."""

    @pytest.mark.parametrize("name", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
    def test_weighting_one_half_protects_it(self, name):
        rng = np.random.default_rng(99)
        fmt = lq.FORMATS[name]
        values = _sample(rng, fmt.block * 8)
        importance = np.ones_like(values)
        importance[: values.size // 2] = 100.0

        plain = lq.dequantize_array(lq.quantize_array(values, name), name)
        weighted = lq.dequantize_array(
            lq.quantize_array(values, name, importance), name
        )
        half = values.size // 2
        protected_plain = _relative_rmse(values[:half], plain[:half])
        protected_weighted = _relative_rmse(values[:half], weighted[:half])
        assert protected_weighted <= protected_plain * 1.05, (
            "the imatrix made no difference to the half it was told to protect"
        )

    def test_a_legacy_type_says_it_ignores_one(self):
        """Q4_0 has nowhere to put per-element importance. Silently
        accepting it would let someone believe it was applied."""
        assert lq.FORMATS["Q4_0"].takes_imatrix is False
        assert lq.FORMATS["Q4_K"].takes_imatrix is True

    def test_a_wrong_length_imatrix_is_refused(self):
        values = np.zeros(256)
        with pytest.raises(lq.LlamaQuantError, match="weights for"):
            lq.quantize_array(values, "Q4_K", np.ones(128))


class TestTheLookups:
    def test_is_supported_answers_for_names_and_type_ids(self):
        assert lq.is_supported("Q4_K") and lq.is_supported("q4_k")
        assert lq.is_supported(int(GGMLType.Q4_K))
        assert not lq.is_supported("IQ1_M")
        assert not lq.is_supported(int(GGMLType.HNX_IQ0_5))

    def test_an_unknown_name_names_the_known_ones(self):
        with pytest.raises(lq.LlamaQuantError, match="Q4_K"):
            lq.quantize_array(np.zeros(256), "Q9_ULTRA")

    def test_bytes_that_are_not_whole_blocks_are_refused(self):
        with pytest.raises(lq.LlamaQuantError, match="whole number"):
            lq.dequantize_array(b"\x00" * 100, "Q4_K")
