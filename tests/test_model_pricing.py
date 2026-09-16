"""Estimating what a model costs to serve.

`hypernix-t1 index` reads everything about a model except what to charge
for it, and wrote 0.0. A price of zero on a 70B is not a policy — it is
an unanswered question that bills the operator.

This is the check that the estimate is worth having. Most of the file is
*relationships* rather than absolute figures, because the absolute
figures depend on what an hour of GPU is worth to a given operator and
the relationships do not:

* a bigger model costs more than a smaller one
* the same model costs more on CPU than on a card that fits it
* a mixture-of-experts costs what its *active* parameters cost, not its
  total — the case that is wrong by an order of magnitude if you get it
  backwards
* output costs more than input, because decode is serial and prefill is
  not

The shapes below are real ones: Qwen3 8B at Q4_K_M really is about 4.9
GB, a 70B at Q4 really is about 40, and Qwen3-235B-A22B really does have
22B active. An estimator that only works on invented numbers is not one
anybody should price with.
"""
from __future__ import annotations

import logging

import pytest

from hypernix.t1api.pricing import FLOOR_PER_1K, PriceEstimate, estimate

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def qwen3_8b(**overrides) -> PriceEstimate:
    """Qwen3 8B at Q4_K_M — 4.9 GB, 8.2B parameters."""
    return estimate(**{
        "file_bytes": int(4.9 * GB), "total_parameters_b": 8.2,
        "bits_per_weight": 4.83, "quant": "Q4_K_M", "vram_bytes": 24 * GB,
        **overrides,
    })


def llama_70b(**overrides) -> PriceEstimate:
    """A 70B at Q4_K_M — about 40 GB."""
    return estimate(**{
        "file_bytes": int(40 * GB), "total_parameters_b": 70.0,
        "bits_per_weight": 4.5, "quant": "Q4_K_M", "vram_bytes": 80 * GB,
        **overrides,
    })


def qwen3_235b_a22b(**overrides) -> PriceEstimate:
    """235B total, 22B active. The case the whole feature turns on."""
    return estimate(**{
        "file_bytes": int(70 * GB), "total_parameters_b": 235.0,
        "active_parameters_b": 22.0, "bits_per_weight": 2.5, "quant": "IQ2_M",
        "vram_bytes": 80 * GB, **overrides,
    })


class TestTheRelationshipsHold:
    def test_a_bigger_model_costs_more(self):
        assert llama_70b().output_price_per_1k > qwen3_8b().output_price_per_1k

    def test_the_same_model_costs_more_without_a_gpu(self):
        """The biggest single factor, and the reason placement is decided
        from the file's real size rather than a parameter count."""
        assert (
            qwen3_8b(vram_bytes=0).output_price_per_1k
            > qwen3_8b(vram_bytes=24 * GB).output_price_per_1k
        )

    def test_output_costs_more_than_input(self):
        """Prefill is parallel across the sequence; decode is one token
        at a time."""
        priced = qwen3_8b()
        assert priced.output_price_per_1k > priced.input_price_per_1k

    def test_a_card_that_fits_it_beats_one_that_half_does(self):
        assert (
            llama_70b(vram_bytes=24 * GB).output_price_per_1k
            > llama_70b(vram_bytes=80 * GB).output_price_per_1k
        )

    def test_nothing_is_ever_free(self):
        """A registry full of 0.0 cannot express "cheap" — it is
        indistinguishable from "nobody set this"."""
        tiny = estimate(
            file_bytes=int(0.3 * GB), total_parameters_b=0.5,
            bits_per_weight=4.0, vram_bytes=80 * GB,
        )
        assert tiny.output_price_per_1k >= FLOOR_PER_1K
        assert tiny.input_price_per_1k >= FLOOR_PER_1K


class TestMixtureOfExperts:
    """The case that is wrong by an order of magnitude if you get it
    backwards, and the one the request specifically named."""

    def test_it_is_priced_by_active_parameters(self):
        by_active = qwen3_235b_a22b()
        by_total = qwen3_235b_a22b(active_parameters_b=0.0)
        assert by_active.output_price_per_1k < by_total.output_price_per_1k

    def test_the_difference_is_large_not_cosmetic(self):
        """235B/22B is more than a 5x difference in work per token. An
        estimator that got this within 20% would be getting it wrong."""
        by_active = qwen3_235b_a22b()
        by_total = qwen3_235b_a22b(active_parameters_b=0.0)
        assert by_total.output_price_per_1k / by_active.output_price_per_1k > 5

    def test_it_still_needs_the_memory_of_the_full_model(self):
        """22B of work per token, 70 GB of weights. Pricing it as a 22B
        would ignore the hardware it demands, which is why placement
        comes from the file and speed from the active count."""
        on_a_small_card = qwen3_235b_a22b(vram_bytes=24 * GB)
        # 2x80GB, because 70 GB of weights plus a KV cache does not fit
        # in 80 -- which the estimator is right about and this test
        # originally was not.
        on_enough_cards = qwen3_235b_a22b(vram_bytes=160 * GB)
        assert on_a_small_card.placement != "gpu"
        assert on_enough_cards.placement == "gpu"

    def test_a_dense_model_is_unaffected(self):
        """Passing the active count for a dense model is the same number
        twice, and must not change anything."""
        assert (
            llama_70b(active_parameters_b=70.0).output_price_per_1k
            == llama_70b().output_price_per_1k
        )


class TestItSaysWhatItAssumed:
    """A silently-wrong price is worse than an obviously uncertain one."""

    def test_a_dense_assumption_is_reported(self):
        priced = qwen3_235b_a22b(active_parameters_b=0.0)
        assert any("dense" in note for note in priced.assumptions)

    def test_a_missing_gpu_is_reported(self):
        assert any("no GPU" in note for note in qwen3_8b(vram_bytes=0).assumptions)

    def test_a_derived_parameter_count_is_reported(self):
        priced = estimate(
            file_bytes=int(4.9 * GB), total_parameters_b=0.0,
            bits_per_weight=4.83, vram_bytes=24 * GB,
        )
        assert any("derived" in note for note in priced.assumptions)

    def test_a_derived_count_is_close_to_the_real_one(self):
        """4.9 GB at 4.83 bpw really is about 8B. A derivation that was
        out by 3x would make the price it produces worthless."""
        priced = estimate(
            file_bytes=int(4.9 * GB), total_parameters_b=0.0,
            bits_per_weight=4.83, vram_bytes=24 * GB,
        )
        assert 6.0 < priced.total_parameters_b < 11.0

    def test_knowing_everything_assumes_nothing(self):
        assert qwen3_235b_a22b().assumptions == []

    def test_the_inputs_are_all_reported(self):
        """So somebody who disagrees can see *where* they disagree rather
        than being handed a figure."""
        data = qwen3_235b_a22b().to_dict()
        for field in ("total_parameters_b", "active_parameters_b", "file_bytes",
                      "bits_per_weight", "quant", "placement",
                      "estimated_tokens_per_second", "vram_bytes"):
            assert field in data


class TestTheNumbersAreNotAbsurd:
    """Not a market rate, and not allowed to be nonsense either. These
    bounds are wide on purpose — they catch an estimator that is out by
    orders of magnitude, which is the failure that matters."""

    def test_a_small_model_on_a_good_card_is_cheap(self):
        assert qwen3_8b().output_price_per_1k < 0.05

    def test_a_70b_is_not_cheaper_than_an_8b(self):
        assert llama_70b().output_price_per_1k > qwen3_8b().output_price_per_1k

    def test_nothing_costs_more_than_a_dollar_per_1k(self):
        """A price above this is not a model anybody serves; it is an
        arithmetic error."""
        for priced in (qwen3_8b(), llama_70b(), qwen3_235b_a22b(),
                       qwen3_8b(vram_bytes=0), llama_70b(vram_bytes=0)):
            assert priced.output_price_per_1k < 1.0, priced.describe()

    def test_the_speed_estimate_is_in_the_right_order(self):
        """An 8B on a card does tens of tokens a second, not thousands
        and not one."""
        speed = qwen3_8b().estimated_tokens_per_second
        assert 5 < speed < 500

    def test_a_70b_on_cpu_is_slow(self):
        assert llama_70b(vram_bytes=0).estimated_tokens_per_second < 5


class TestReadingAnActualFile:
    """The estimator takes what the indexer read. If those two disagree
    about field names it produces a confident price from nothing."""

    def test_it_reads_an_indexed_model(self, tmp_path):
        import struct

        from hypernix.quant.gguf import GGMLType, GGUFWriter
        from hypernix.t1api.modelindex import inspect
        from hypernix.t1api.pricing import estimate_for_indexed

        path = tmp_path / "model.gguf"
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        writer.set_metadata("llama.block_count", 1)
        writer.set_metadata("llama.context_length", 4096)
        shapes = {"token_embd.weight": (512, 8), "blk.0.attn_q.weight": (512, 8)}
        payload = {}
        for name, shape in shapes.items():
            count = shape[0] * shape[1]
            writer.add_tensor(name, shape, int(GGMLType.F32))
            payload[name] = struct.pack(f"<{count}f", *([0.01] * count))
        writer.write(lambda t: payload[t.name])

        found = inspect(path)
        priced = estimate_for_indexed(found, vram_bytes=24 * GB)
        assert priced.output_price_per_1k > 0
        assert priced.file_bytes == found.file_bytes

    def test_the_indexer_reports_active_parameters(self, tmp_path):
        """The field the estimator reads for the MoE case. If the indexer
        never sets it, every model is priced as dense and the feature is
        inert."""
        from hypernix.t1api.modelindex import IndexedModel

        assert "active_parameters_b" in IndexedModel.__dataclass_fields__

    def test_a_dense_file_reports_active_equal_to_total(self, tmp_path):
        import struct

        from hypernix.quant.gguf import GGMLType, GGUFWriter
        from hypernix.t1api.modelindex import inspect

        path = tmp_path / "dense.gguf"
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        writer.set_metadata("llama.block_count", 1)
        writer.add_tensor("token_embd.weight", (512, 8), int(GGMLType.F32))
        writer.write(lambda t: struct.pack("<4096f", *([0.01] * 4096)))

        found = inspect(path)
        assert found.active_parameters_b == found.parameters_b
        assert found.expert_count == 0
