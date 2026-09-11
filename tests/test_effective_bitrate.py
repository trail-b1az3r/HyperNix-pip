"""What the *file* costs per weight, not what the tier packs at.

From a report: two models quantised from the same BF16 Qwen3-class 2B,
one at IQ0.9_L and one at IQ0.5_XXXL, **both 1.4 GB**. A tier that
claims 0.56 bits per weight and one that claims 0.94 landing on the same
size is not a rounding coincidence — it means the tier barely touched
the file.

It did not, and by design. The default policy leaves ``token_embd`` and
``output`` at source precision, and Qwen3's vocabulary is 151,936
tokens: those two tensors are 622M of the model's 2.03B parameters, and
at BF16 they are 1.24 GB before a single packed tensor is written. The
sub-bit body adds 99 MB at IQ0.5 and 165 MB at IQ0.9 — which is the
whole of the difference between "1.34 GB" and "1.41 GB", both of which
read as "1.4 GB".

So the run reported its tier's rate, 0.562, for a file costing 5.3 bits
per weight. Nothing said so, and the flags that fix it were never
mentioned. These tests are about the report telling the truth.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant.gguf import GGMLType, GGUFWriter
from hypernix.quant.hyprslug import QuantizeReport, quantize_gguf

N_EMBD, N_FF = 256, 512


def _bf16(a: np.ndarray) -> bytes:
    """Round-to-nearest, the way a real BF16 export is written."""
    return ((a.astype(np.float32).view(np.uint32) + 0x8000) >> 16).astype("<u2").tobytes()


def _model(path: Path, vocab: int, dtype: int = int(GGMLType.BF16)) -> Path:
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "qwen3")
    writer.set_metadata("qwen3.block_count", 1)
    writer.set_metadata("qwen3.embedding_length", N_EMBD)
    writer.set_metadata("qwen3.attention.head_count", 8)
    writer.set_metadata("qwen3.attention.head_count_kv", 8)
    writer.set_metadata("qwen3.feed_forward_length", N_FF)
    shapes = {
        "token_embd.weight": (N_EMBD, vocab),
        "blk.0.attn_q.weight": (N_EMBD, N_EMBD),
        "blk.0.ffn_gate.weight": (N_EMBD, N_FF),
        "blk.0.ffn_down.weight": (N_FF, N_EMBD),
        "output.weight": (N_EMBD, vocab),
    }
    rng = np.random.default_rng(0)
    payload = {}
    for name, shape in shapes.items():
        values = rng.normal(0, 0.08, int(np.prod(shape))).astype(np.float32)
        payload[name] = (
            _bf16(values) if dtype == int(GGMLType.BF16)
            else struct.pack(f"<{values.size}f", *values.tolist())
        )
        writer.add_tensor(name, shape, dtype)
    writer.write(lambda tensor: payload[tensor.name])
    return path


@pytest.fixture(scope="module")
def big_vocab(tmp_path_factory):
    """A vocabulary large enough for the embedding table to dominate,
    which is the case the report came from and the case every modern
    model is in."""
    return _model(tmp_path_factory.mktemp("bitrate") / "m.bf16.gguf", vocab=16384)


class TestTheReportedRateIsTheFilesRate:
    def test_the_default_costs_far_more_than_the_tier_claims(self, big_vocab, tmp_path):
        report = quantize_gguf(big_vocab, tmp_path / "d.gguf", "IQ0.5_XXXL")
        assert report.tier_bits_per_weight == pytest.approx(0.5625)
        assert report.effective_bits_per_weight > 4.0, (
            "the untouched embedding table has stopped dominating; if the "
            "default policy changed, this test is the wrong shape"
        )

    def test_the_flags_deliver_the_advertised_rate(self, big_vocab, tmp_path):
        report = quantize_gguf(
            big_vocab, tmp_path / "f.gguf", "IQ0.5_XXXL",
            quantize_embeddings=True, quantize_output=True,
        )
        assert report.effective_bits_per_weight == pytest.approx(0.5625, abs=0.12)

    def test_two_tiers_land_on_the_same_size_by_default(self, big_vocab, tmp_path):
        """The report's actual symptom, reproduced.

        Both files are within a few percent, because what dominates them
        is identical in each: the tensors neither tier touched.
        """
        small = quantize_gguf(big_vocab, tmp_path / "s.gguf", "IQ0.5_XXXL").output_bytes
        large = quantize_gguf(big_vocab, tmp_path / "l.gguf", "IQ0.9_L").output_bytes
        assert abs(small - large) / large < 0.10, (small, large)

    def test_and_are_far_apart_once_the_flags_are_on(self, big_vocab, tmp_path):
        """Which is how you can tell the tiers do work."""
        kw = dict(quantize_embeddings=True, quantize_output=True)
        small = quantize_gguf(big_vocab, tmp_path / "s2.gguf", "IQ0.5_XXXL", **kw).output_bytes
        large = quantize_gguf(big_vocab, tmp_path / "l2.gguf", "IQ0.9_L", **kw).output_bytes
        assert large > small * 1.4, (small, large)

    def test_the_default_barely_compresses_at_all(self, big_vocab, tmp_path):
        report = quantize_gguf(big_vocab, tmp_path / "c.gguf", "IQ0.5_XXXL")
        assert report.compression < 1.5, report.describe()


class TestItSaysSo:
    def test_the_warning_fires_when_the_name_misleads(self, big_vocab, tmp_path):
        report = quantize_gguf(big_vocab, tmp_path / "w.gguf", "IQ0.5_XXXL")
        assert report.name_is_misleading
        text = report.describe()
        assert "what the tier name suggests" in text

    def test_the_warning_names_the_flags_that_fix_it(self, big_vocab, tmp_path):
        """A warning that does not say what to do instead is just a
        second way of being surprised."""
        text = quantize_gguf(big_vocab, tmp_path / "w2.gguf", "IQ0.5_XXXL").describe()
        assert "--quantize-embeddings" in text
        assert "--quantize-output" in text

    def test_it_stays_quiet_when_the_file_matches_its_tier(self, big_vocab, tmp_path):
        """A warning on every run is a warning nobody reads."""
        report = quantize_gguf(
            big_vocab, tmp_path / "q.gguf", "IQ0.5_XXXL",
            quantize_embeddings=True, quantize_output=True,
        )
        assert not report.name_is_misleading
        assert "what the tier name suggests" not in report.describe()

    def test_a_small_vocabulary_does_not_trip_it(self, tmp_path):
        """Norms and biases are always copied and always small; the
        threshold has to tolerate that or it fires on healthy runs."""
        source = _model(tmp_path / "tiny.gguf", vocab=256)
        report = quantize_gguf(
            source, tmp_path / "t.gguf", "IQ0.5_XXXL",
            quantize_embeddings=True, quantize_output=True,
        )
        assert not report.name_is_misleading, report.describe()

    def test_the_real_rate_is_in_the_description(self, big_vocab, tmp_path):
        text = quantize_gguf(big_vocab, tmp_path / "r.gguf", "IQ0.5_XXXL").describe()
        assert "bits/weight over the whole file" in text

    def test_the_json_carries_both_numbers(self, big_vocab, tmp_path):
        """Two different numbers that are easy to confuse, so a script
        gets both rather than guessing which one it has."""
        payload = quantize_gguf(big_vocab, tmp_path / "j.gguf", "IQ0.5_XXXL").to_dict()
        assert payload["tier_bits_per_weight"] == pytest.approx(0.562, abs=0.01)
        assert payload["effective_bits_per_weight"] > 4.0
        assert payload["name_is_misleading"] is True


class TestTheArithmeticIsWhatItSays:
    def test_effective_rate_is_output_bytes_over_weights(self):
        report = QuantizeReport(
            tier="IQ0.5_XXXL", packing="quad_code_xxxl",
            output_bytes=1000, elements_quantized=600, elements_copied=400,
        )
        assert report.elements_total == 1000
        assert report.effective_bits_per_weight == pytest.approx(8.0)

    def test_no_weights_is_not_a_division_by_zero(self):
        assert QuantizeReport().effective_bits_per_weight == 0.0

    def test_an_unknown_packing_does_not_claim_a_tier_rate(self):
        assert QuantizeReport(output_bytes=10, elements_copied=10).tier_bits_per_weight == 0.0
        assert not QuantizeReport(output_bytes=10, elements_copied=10).name_is_misleading
