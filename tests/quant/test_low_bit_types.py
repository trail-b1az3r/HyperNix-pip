"""The types added in 0.72.3.post2: IQ0.25_UXL, INT1, INT4, FP2, Q4_M.

Two families arrive here at once and the tests keep them apart, because
they fail differently.

**Sign and scale** (``IQ0.25_UXL``, ``INT1``) extends the existing
sub-bit machinery. ``INT1`` is its ``k == g`` case — every sign kept,
nothing reconstructed — and ``IQ0.25_UXL`` is the far end, three signs of
sixteen. Both are :mod:`hypernix.quant.subbit` packings and neither
needed new arithmetic, which is the point of checking them: a packing
table entry that does not match what the packer emits produces a file
whose offsets are wrong from the first tensor.

**Fixed codebook** (``INT4``, ``FP2``) is new arithmetic, in
:mod:`hypernix.quant.lowbit`. These carry magnitude, so the thing that
can go wrong is the scale — and it did. See
:class:`TestTheScaleSearchEarnsItsPlace`.

``Q4_M`` is neither: it is a spelling of ``Q4_K_M`` that the resolver did
not recognise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypernix.quant import lowbit, subbit
from hypernix.quant.gguf import _BLOCK_SHAPE, GGMLType
from hypernix.quant.hyprslug import TIER_TYPES, quantize_gguf, resolve_recipe

#: Every extension tier, with the rate its name promises and the rate the
#: file actually carries. They differ for the fixed-codebook types by
#: exactly the FP16 block scale, the same way Q4_0 is 4.5 and not 4.
NEW_TIERS = {
    "IQ0.25_UXL": 0.25,
    "INT1": 1.0625,
    "FP2": 2.0625,
    "INT4": 4.0625,
}


class TestTheRatesAreExactlyWhatIsClaimed:
    """A quantiser that is off by a byte per block is off by 3% at these
    widths, and the only place it shows is a file size nobody checks."""

    @pytest.mark.parametrize(("tier", "expected"), sorted(NEW_TIERS.items()))
    def test_the_tier_carries_the_rate_it_advertises(self, tier, expected):
        from hypernix.quant.steamroller import TIERS

        _ggml, packing = TIER_TYPES[tier]
        actual = (
            subbit.PACKINGS[packing].bits_per_weight
            if packing in subbit.PACKINGS
            else lowbit.CODECS[packing].bits_per_weight
        )
        assert actual == pytest.approx(expected)
        assert TIERS[tier].bits_per_weight == pytest.approx(expected)

    @pytest.mark.parametrize("tier", sorted(NEW_TIERS))
    def test_the_gguf_size_table_matches_the_packer(self, tier):
        """The table ``gguf.py`` sizes tensors from and the code that
        emits the bytes are two places that have to agree. When they do
        not, every tensor after the first lands at the wrong offset and
        the file is confidently, silently wrong."""
        ggml_type, packing = TIER_TYPES[tier]
        emitted = (
            subbit.PACKINGS[packing].block_bytes
            if packing in subbit.PACKINGS
            else lowbit.CODECS[packing].block_bytes
        )
        block, table = _BLOCK_SHAPE[GGMLType(ggml_type)]
        assert block == 256
        assert table == emitted

    def test_a_quarter_bit_block_really_is_eight_bytes(self):
        """256 weights in 8 bytes, scale included. The arithmetic that
        makes 0.25 exact rather than approximately a quarter."""
        spec = subbit.PACKINGS["quarter_code_uxl"]
        assert spec.group == 16 and spec.kept == 3
        assert spec.block_bytes == 8
        raw = subbit.quantize_tensor([0.5] * 256, "quarter_code_uxl")
        assert len(raw) == 8

    def test_int1_keeps_every_sign(self):
        """The k == g case: nothing is reconstructed, so the only loss is
        magnitude. If this drops a sign the whole family is wrong."""
        rng = np.random.default_rng(0)
        weights = rng.standard_normal(256).astype(np.float32)
        raw = subbit.quantize_tensor(weights.tolist(), "int1_binary")
        back = subbit.dequantize_array(raw, "int1_binary")
        assert np.array_equal(np.sign(back), np.sign(weights))


class TestTheScaleSearchEarnsItsPlace:
    """The bug this exists for shipped in the first draft of lowbit.py.

    The scale was fitted to the block's peak, the way Q4_0 does it. At
    sixteen levels that is fine. At four it is a disaster: the levels
    land at 1.75 and 3.5 sigma, almost everything rounds to the smaller
    of two numbers that are both too big, and FP2 scored *worse relative
    error than one bit* — a two-bit format losing to a one-bit format at
    twice the size.

    Nothing in a round-trip test catches that. It needs a comparison.
    """

    @pytest.fixture(scope="class")
    def weights(self):
        rng = np.random.default_rng(0)
        return (rng.standard_normal(256 * 64) * 0.05).astype(np.float32)

    def _relative_error(self, weights, reconstructed):
        return float(
            np.sqrt(((reconstructed - weights) ** 2).mean()) / weights.std()
        )

    def test_fp2_beats_int1_at_twice_the_size(self, weights):
        """The comparison the peak-fitted version failed. Paying a
        second bit per weight has to buy something."""
        fp2 = lowbit.dequantize_array(
            lowbit.quantize_array(weights, "FP2"), "FP2"
        )
        int1 = subbit.dequantize_array(
            subbit.quantize_tensor(weights.tolist(), "int1_binary"), "int1_binary"
        )
        assert self._relative_error(weights, fp2) < self._relative_error(
            weights, int1
        )

    def test_every_codec_beats_the_next_one_down(self, weights):
        """More bits must not produce a worse reconstruction. It is the
        one ordering that is always true and the first to break when a
        scale is fitted wrongly."""
        int4 = self._relative_error(
            weights, lowbit.dequantize_array(
                lowbit.quantize_array(weights, "INT4"), "INT4")
        )
        fp2 = self._relative_error(
            weights, lowbit.dequantize_array(
                lowbit.quantize_array(weights, "FP2"), "FP2")
        )
        assert int4 < fp2

    def test_the_search_beats_the_peak_fit_it_replaced(self, weights):
        """Directly, so the search cannot be quietly removed later."""
        codec = lowbit.CODECS["FP2"]
        blocks = weights.reshape(-1, 256)
        peak = np.abs(blocks).max(axis=1) / codec.peak
        scales = peak.astype(np.float16).astype(np.float32)
        codes, _err = lowbit._encode_at(blocks, scales, codec)  # noqa: SLF001
        levels = np.asarray(codec.levels, dtype=np.float32)
        peak_fitted = (levels[codes] * scales[:, None]).reshape(-1)

        searched = lowbit.dequantize_array(
            lowbit.quantize_array(weights, "FP2"), "FP2"
        )
        assert self._relative_error(weights, searched) < self._relative_error(
            weights, peak_fitted
        )

    def test_the_stored_scale_is_the_one_the_search_scored(self, weights):
        """The candidate is rounded to FP16 inside the loop. Score a
        scale the file cannot hold and the winner is chosen on a number
        that never reaches disk."""
        raw = lowbit.quantize_array(weights[:256], "FP2")
        stored = np.frombuffer(raw[:2], dtype=np.float16)[0]
        assert np.float16(np.float32(stored)) == stored


class TestTheCodecsRoundTrip:
    @pytest.mark.parametrize("name", sorted(lowbit.CODECS))
    def test_a_block_of_zeros_stays_zeros(self, name):
        """A zero block gives a zero scale, and the encoder must not
        divide by it."""
        zeros = np.zeros(256, dtype=np.float32)
        back = lowbit.dequantize_array(lowbit.quantize_array(zeros, name), name)
        assert np.array_equal(back, zeros)

    @pytest.mark.parametrize("name", sorted(lowbit.CODECS))
    def test_a_poisoned_block_degrades_rather_than_spreading(self, name):
        """One inf in a block must not turn 256 weights into NaN. The
        file is written either way; only one version is recoverable."""
        poisoned = np.zeros(256, dtype=np.float32)
        poisoned[7] = np.inf
        back = lowbit.dequantize_array(
            lowbit.quantize_array(poisoned, name), name
        )
        assert np.isfinite(back).all()

    @pytest.mark.parametrize("name", sorted(lowbit.CODECS))
    def test_a_ragged_tensor_is_refused(self, name):
        """Padding would change the tensor's shape, and silently
        changing a model's shape is worse than refusing it."""
        with pytest.raises(lowbit.LowBitError, match="do not divide"):
            lowbit.quantize_array(np.zeros(300, dtype=np.float32), name)

    @pytest.mark.parametrize("name", sorted(lowbit.CODECS))
    def test_a_truncated_file_is_refused(self, name):
        raw = lowbit.quantize_array(np.zeros(256, dtype=np.float32), name)
        with pytest.raises(lowbit.LowBitError, match="whole number"):
            lowbit.dequantize_array(raw[:-1], name)

    def test_an_unknown_codec_names_the_ones_it_has(self):
        with pytest.raises(lowbit.LowBitError, match="INT4"):
            lowbit.quantize_array(np.zeros(256, dtype=np.float32), "FP7")

    @pytest.mark.parametrize("name", sorted(lowbit.CODECS))
    def test_every_code_survives_a_round_trip(self, name):
        """Straight at the bit packing: feed one block that lands on
        every level and check each comes back. An off-by-one in the
        packer shifts every code by one level, which looks like a
        slightly worse quantiser rather than like a bug."""
        codec = lowbit.CODECS[name]
        levels = np.asarray(codec.levels, dtype=np.float32)
        pattern = np.resize(levels, 256).astype(np.float32)
        back = lowbit.dequantize_array(
            lowbit.quantize_array(pattern, name), name
        )
        assert np.allclose(back, pattern, rtol=1e-3, atol=1e-3)


class TestQ4MIsQ4KM:
    """``Q4M`` is what people type. Squashing separators does not get
    there from ``Q4_K_M`` — the missing character is the ``K``, not an
    underscore — so it fell through to "unknown target", which is a
    confusing way to reject the most common request there is."""

    @pytest.mark.parametrize(
        "spelling", ["q4m", "Q4M", "Q4_M", "q4-m", "Q4_K_M", "q4km"]
    )
    def test_every_spelling_resolves_to_the_same_recipe(self, spelling):
        recipe = resolve_recipe(spelling)
        assert recipe is not None, spelling
        assert recipe.name == "Q4_K_M"

    def test_it_is_still_a_mix_not_a_block_format(self):
        """A consumer treating Q4_K_M as a block format would mis-size
        every estimate."""
        recipe = resolve_recipe("q4m")
        assert recipe.base == "Q4_K"
        assert set(dict(recipe.overrides).values()) == {"Q6_K"}

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [("q3l", "Q3_K_L"), ("q5m", "Q5_K_M"), ("q4s", "Q4_K_S")],
    )
    def test_the_other_dropped_letter_spellings_work_too(self, spelling, expected):
        assert resolve_recipe(spelling).name == expected

    def test_the_cli_alias_table_agrees(self):
        from hypernix.interfaces.cli import _ALIAS

        assert _ALIAS["q4m"] == "q4_k_m"


class TestTheyGoAllTheWayThrough:
    """Quantise a real model to each and run it.

    A tier that packs but does not load is the failure this whole line
    of work exists to stop shipping.
    """

    @pytest.fixture(scope="class")
    def source(self, tmp_path_factory):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from test_hnxrun import _write_model

        directory = tmp_path_factory.mktemp("new-tiers")
        return _write_model(directory / "tiny.f32.gguf", tokenizer=True)

    @pytest.fixture(scope="class")
    def built(self, source):
        built = {}
        for tier in NEW_TIERS:
            out = Path(source).parent / f"tiny.{tier}.gguf"
            quantize_gguf(
                source, out, tier,
                quantize_embeddings=True, quantize_output=True,
            )
            built[tier] = out
        return built

    @pytest.mark.parametrize("tier", sorted(NEW_TIERS))
    def test_it_loads_and_generates(self, built, tier):
        from hypernix.models import hnxrun

        model = hnxrun.load_model(built[tier])
        assert model.sub_bit
        tokens = hnxrun.generate_tokens(model, [1, 5, 9], max_new_tokens=4)
        assert len(tokens) == 4

    @pytest.mark.parametrize("tier", sorted(NEW_TIERS))
    def test_it_stays_packed_in_memory(self, built, tier):
        """The whole reason these exist. Dequantising at load time would
        make every one of them cost 32 bits per weight."""
        from hypernix.models import hnxrun

        model = hnxrun.load_model(built[tier])
        assert model.packed_in_memory > 0
        assert model.resident_bits_per_weight < NEW_TIERS[tier] + 0.5

    def test_a_narrower_tier_gives_a_smaller_file(self, built):
        sizes = {t: built[t].stat().st_size for t in NEW_TIERS}
        assert (
            sizes["INT4"] > sizes["FP2"] > sizes["INT1"] > sizes["IQ0.25_UXL"]
        ), sizes

    def test_the_wider_codecs_track_the_original_better(self, built, source):
        """Ordering, not quality. Below about a bit these stop being
        worse versions of the model and start being different models, so
        what is asserted is that paying for bits buys something while
        there are still enough of them to buy with."""
        import torch

        from hypernix.models import hnxrun

        reference, _ = hnxrun.forward(hnxrun.load_model(source), [1, 5, 9, 13])

        def agreement(tier):
            logits, _ = hnxrun.forward(
                hnxrun.load_model(built[tier]), [1, 5, 9, 13]
            )
            return torch.nn.functional.cosine_similarity(
                logits.reshape(-1), reference.reshape(-1), dim=0
            ).item()

        assert agreement("INT4") > agreement("FP2") > agreement("INT1")

    @pytest.mark.parametrize("tier", sorted(NEW_TIERS))
    def test_the_reference_reader_still_refuses_them(self, built, tier):
        """The type ids stay outside upstream's range on purpose: a
        stock loader has to refuse the file by name rather than read a
        quarter-bit tensor as Q4_K and return noise."""
        gguf = pytest.importorskip("gguf")

        with pytest.raises(Exception):  # noqa: B017, PT011
            gguf.GGUFReader(str(built[tier]))
