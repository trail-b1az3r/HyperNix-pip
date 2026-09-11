"""The IQ0.x tiers, which used not to quantise anything.

``steamroller`` advertised ``IQ0.9_L``, ``IQ0.75_M`` and ``IQ0.5_XXXL``
for several releases. What ``pack_sub_bit`` actually did was copy the
Q3_K_L staging file and write a sidecar JSON naming a tier — so a
"0.5-bit model" was byte-identical to the 3-bit model it came from, the
same size on disk, and no more quantised than its input. The tier was a
label on an unchanged file, and nothing tested that it was not.

These tests are mostly about that: the output has to be smaller, the
tensors have to carry the sub-bit type, and the bytes have to differ from
the input.
"""
from __future__ import annotations

import math
import random
import struct
from pathlib import Path

import pytest

from hypernix.quant.gguf import (
    DEFAULT_ALIGNMENT,
    GGMLType,
    GGUFError,
    GGUFFile,
    GGUFWriter,
    tensor_nbytes,
    type_size_bytes,
)
from hypernix.quant.hyprslug import (
    ALIASES,
    TIER_TYPES,
    HyprslugError,
    quantize_gguf,
)
from hypernix.quant.subbit import (
    BLOCK_SIZE,
    PACKINGS,
    SubBitError,
    dequantize_block,
    dequantize_tensor,
    quantize_block,
    quantize_tensor,
)


def _weights(count: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.05) for _ in range(count)]


def _write_gguf(path: Path, tensors: dict[str, tuple[tuple[int, ...], list[float]]],
                metadata: dict | None = None) -> Path:
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    for key, value in (metadata or {}).items():
        writer.set_metadata(key, value)
    payload = {}
    for name, (shape, values) in tensors.items():
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = struct.pack(f"<{len(values)}f", *values)
    writer.write(lambda tensor: payload[tensor.name])
    return path


class TestTheBitRatesAreWhatTheTiersClaim:
    @pytest.mark.parametrize(
        ("packing", "expected_bpw"),
        [("sign_scale_l", 0.9375), ("pair_code_m", 0.8125), ("quad_code_xxxl", 0.5625)],
    )
    def test_each_packing_hits_its_advertised_rate(self, packing, expected_bpw):
        assert PACKINGS[packing].bits_per_weight == pytest.approx(expected_bpw)

    #: The packings that claim to be sub-bit. ``int1_binary`` is in this
    #: module's PACKINGS table but is not one of them: it is the k == g
    #: end of the same machinery, one whole bit per weight plus the
    #: scale, and it is named INT1 rather than IQ-something precisely so
    #: nobody expects otherwise.
    SUB_BIT_PACKINGS = [
        "sign_scale_l", "pair_code_m", "quad_code_xxxl", "quarter_code_uxl",
    ]

    def test_every_sub_bit_packing_is_below_one_bit_per_weight(self):
        """The entire premise. A 'sub-bit' tier at 1.06 bpw is not one."""
        for name in self.SUB_BIT_PACKINGS:
            assert PACKINGS[name].bits_per_weight < 1.0, name

    def test_the_one_packing_that_is_not_sub_bit_says_so_in_its_name(self):
        """It would be easy to let INT1 drift into the sub-bit list and
        quietly weaken what 'sub-bit' means."""
        not_sub_bit = [
            name for name, spec in PACKINGS.items()
            if spec.bits_per_weight >= 1.0
        ]
        # Both of these live in this module because they are the same
        # machinery -- signs and a scale -- not because they are sub-bit.
        # INT1 is the k == g end of it; hnx_1375bit is what comes after,
        # once every sign is stored and the only thing left to buy is
        # magnitude. Neither is named IQ0-anything, and that is the point.
        assert not_sub_bit == ["int1_binary", "hnx_1375bit"]
        assert TIER_TYPES["INT1"][1] == "int1_binary"
        assert TIER_TYPES["HNX_1375BIT"][1] == "hnx_1375bit"

    def test_more_bits_means_less_error(self):
        """Monotonic, or the tiers are not a ladder.

        This caught a real design fault: the encoder chose which signs to
        keep by magnitude while the decoder filled left to right, so the
        two disagreed about which weight each stored sign belonged to and
        the widest tier came out worst.
        """
        values = _weights(BLOCK_SIZE, seed=7)
        errors = {}
        for packing in ("quad_code_xxxl", "pair_code_m", "sign_scale_l"):
            back = dequantize_block(quantize_block(values, packing), packing)
            errors[packing] = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(values, back, strict=True)) / len(values)
            )
        assert errors["sign_scale_l"] < errors["pair_code_m"] < errors["quad_code_xxxl"]

    def test_the_packer_and_the_gguf_table_agree(self):
        """They are two statements of the same number in different files.

        If they drift, every tensor offset after the first is wrong and
        the file opens, reports sensible shapes and returns garbage.
        """
        from hypernix.quant.lowbit import CODECS

        for tier, (ggml_type, packing) in TIER_TYPES.items():
            emitted = (
                PACKINGS[packing].block_bytes
                if packing in PACKINGS
                else CODECS[packing].block_bytes
            )
            assert emitted == type_size_bytes(ggml_type), tier


class TestBlockRoundTrip:
    @pytest.mark.parametrize("packing", list(PACKINGS))
    def test_a_block_packs_to_exactly_its_declared_size(self, packing):
        packed = quantize_block(_weights(BLOCK_SIZE), packing)
        assert len(packed) == PACKINGS[packing].block_bytes

    @pytest.mark.parametrize("packing", list(PACKINGS))
    def test_kept_signs_come_back_exactly(self, packing):
        """The stored signs are the point; only the tail is approximated."""
        spec = PACKINGS[packing]
        values = _weights(BLOCK_SIZE, seed=3)
        back = dequantize_block(quantize_block(values, packing), packing)
        for start in range(0, BLOCK_SIZE, spec.group):
            for offset in range(spec.kept):
                index = start + offset
                assert (values[index] >= 0) == (back[index] >= 0), index

    @pytest.mark.parametrize("packing", list(PACKINGS))
    def test_a_block_of_zeros_round_trips_to_zeros(self, packing):
        back = dequantize_block(quantize_block([0.0] * BLOCK_SIZE, packing), packing)
        assert all(value == 0.0 for value in back)

    @pytest.mark.parametrize("packing", list(PACKINGS))
    def test_an_infinite_weight_does_not_poison_the_block(self, packing):
        """An inf scale serialises and dequantises every weight to NaN."""
        values = _weights(BLOCK_SIZE)
        values[0] = float("inf")
        back = dequantize_block(quantize_block(values, packing), packing)
        assert all(math.isfinite(v) for v in back)

    @pytest.mark.parametrize("packing", list(PACKINGS))
    def test_the_scale_is_the_mean_not_the_maximum(self, packing):
        """The maximum minimises the wrong thing and biases every
        reconstruction high."""
        values = [0.01] * (BLOCK_SIZE - 1) + [10.0]
        back = dequantize_block(quantize_block(values, packing), packing)
        assert abs(back[0]) < 1.0, "one outlier set the scale for the block"

    def test_a_wrong_sized_block_is_refused(self):
        with pytest.raises(SubBitError, match="block is"):
            quantize_block([0.0] * 10, "sign_scale_l")

    def test_an_unknown_packing_is_refused(self):
        with pytest.raises(SubBitError, match="Unknown packing"):
            quantize_block([0.0] * BLOCK_SIZE, "not-a-packing")

    def test_a_mismatched_importance_length_is_refused(self):
        with pytest.raises(SubBitError, match="same length"):
            quantize_block(_weights(BLOCK_SIZE), "sign_scale_l", [1.0] * 10)


class TestTheImportanceMatrixChangesTheScale:
    def test_weighting_moves_the_scale_toward_what_matters(self):
        """At these rates there is no magnitude left to allocate, so the
        scale is the only thing an imatrix can influence."""
        values = [0.5] * 128 + [0.01] * 128
        plain = dequantize_block(quantize_block(values, "sign_scale_l"), "sign_scale_l")
        weighted = dequantize_block(
            quantize_block(values, "sign_scale_l", [10.0] * 128 + [1.0] * 128),
            "sign_scale_l",
        )
        # The weighted scale should sit closer to the half that matters.
        assert abs(abs(weighted[0]) - 0.5) < abs(abs(plain[0]) - 0.5)

    def test_an_all_zero_importance_does_not_divide_by_zero(self):
        back = dequantize_block(
            quantize_block(_weights(BLOCK_SIZE), "sign_scale_l", [0.0] * BLOCK_SIZE),
            "sign_scale_l",
        )
        assert all(math.isfinite(v) for v in back)


class TestTensorRoundTrip:
    @pytest.mark.parametrize("packing", list(PACKINGS))
    def test_a_multi_block_tensor_round_trips(self, packing):
        values = _weights(BLOCK_SIZE * 4, seed=11)
        back = dequantize_tensor(quantize_tensor(values, packing), packing)
        assert len(back) == len(values)

    def test_a_ragged_tensor_is_refused_not_padded(self):
        """Padding would change the tensor's shape, which is worse than
        refusing it."""
        with pytest.raises(SubBitError, match="do not divide"):
            quantize_tensor(_weights(BLOCK_SIZE + 1), "sign_scale_l")


class TestTheGGUFItWrites:
    def test_it_is_smaller_than_the_source(self, tmp_path):
        """The failure this whole change is about: the output used to be
        byte-identical to its input."""
        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )
        out = tmp_path / "out.gguf"
        report = quantize_gguf(source, out, "IQ0.5_XXXL")

        assert out.stat().st_size < source.stat().st_size
        assert report.compression > 1.0
        assert out.read_bytes() != source.read_bytes()

    @pytest.mark.parametrize("tier", list(TIER_TYPES))
    def test_the_tensors_carry_the_sub_bit_type(self, tmp_path, tier):
        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )
        out = tmp_path / f"{tier}.gguf"
        quantize_gguf(source, out, tier)

        written = GGUFFile.read(out)
        tensor = written.get("blk.0.attn_q.weight")
        assert tensor.ggml_type == TIER_TYPES[tier][0]
        assert tensor.nbytes == tensor_nbytes(TIER_TYPES[tier][0], tensor.shape)

    def test_the_type_id_cannot_collide_with_llama_cpp(self, tmp_path):
        """Picked high on purpose: a stock loader must refuse the file by
        name, not read a 0.5-bit tensor as Q4_K and produce noise."""
        for ggml_type, _ in TIER_TYPES.values():
            assert ggml_type >= 200

    def test_norms_and_biases_are_left_alone(self, tmp_path):
        """All of the damage, none of the size."""
        source = _write_gguf(tmp_path / "src.gguf", {
            "blk.0.attn_q.weight": ((512, 8), _weights(512 * 8)),
            "blk.0.attn_norm.weight": ((512,), [1.0] * 512),
        })
        out = tmp_path / "out.gguf"
        report = quantize_gguf(source, out, "IQ0.5_XXXL")

        written = GGUFFile.read(out)
        assert written.get("blk.0.attn_norm.weight").ggml_type == int(GGMLType.F32)
        assert written.get("blk.0.attn_q.weight").ggml_type != int(GGMLType.F32)
        assert report.tensors_copied == 1
        assert any("1-D" in reason for _, reason in report.skipped)

    def test_embeddings_are_skipped_unless_asked_for(self, tmp_path):
        source = _write_gguf(tmp_path / "src.gguf", {
            "token_embd.weight": ((512, 8), _weights(512 * 8)),
        })
        default = quantize_gguf(source, tmp_path / "a.gguf", "IQ0.5_XXXL")
        opted_in = quantize_gguf(
            source, tmp_path / "b.gguf", "IQ0.5_XXXL", quantize_embeddings=True
        )
        assert default.tensors_quantized == 0
        assert opted_in.tensors_quantized == 1

    def test_metadata_survives_the_round_trip(self, tmp_path):
        """A reader that dropped keys it did not recognise would strip a
        model's chat template and tokenizer on every pass."""
        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
            metadata={
                "tokenizer.ggml.tokens": ["alpha", "beta"],
                "tokenizer.chat_template": "{{ bos }}",
                "llama.rope.freq_base": 10000.0,
                "llama.block_count": 32,
            },
        )
        out = tmp_path / "out.gguf"
        quantize_gguf(source, out, "IQ0.9_L")

        written = GGUFFile.read(out)
        assert written.metadata["tokenizer.ggml.tokens"] == ["alpha", "beta"]
        assert written.metadata["tokenizer.chat_template"] == "{{ bos }}"
        assert written.metadata["llama.rope.freq_base"] == pytest.approx(10000.0)
        assert written.metadata["llama.block_count"] == 32

    def test_the_tier_is_recorded_in_the_file_not_only_a_sidecar(self, tmp_path):
        """A sidecar can be lost in a copy, and then nothing about the
        model says what was done to it."""
        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )
        out = tmp_path / "out.gguf"
        quantize_gguf(source, out, "IQ0.75_M")

        written = GGUFFile.read(out)
        assert written.metadata["hypernix.tier"] == "IQ0.75_M"
        assert written.metadata["hypernix.quantiser"] == "hyprslug"
        assert written.metadata["hypernix.sub_bit"] is True

    def test_a_tensor_that_does_not_divide_is_copied_and_reported(self, tmp_path):
        """Never silently: a run that left half a model at F32 while
        reporting IQ0.5 is the failure this module exists to fix."""
        source = _write_gguf(tmp_path / "src.gguf", {
            "blk.0.odd.weight": ((100, 3), _weights(300)),
        })
        report = quantize_gguf(source, tmp_path / "out.gguf", "IQ0.5_XXXL")
        assert report.tensors_quantized == 0
        assert report.tensors_copied == 1
        assert any("divide into" in reason for _, reason in report.skipped)
        assert report.quantized_fraction == 0.0

    def test_an_already_quantised_source_is_requantised_and_said_so(self, tmp_path):
        """A Q8_0 GGUF is the only copy of the model most people have.

        "Quantise from the unquantised weights" is advice they cannot
        take, so this reads the source through the llama.cpp decoders
        instead -- and names the type it came from, because requantising
        compounds whatever the first pass lost and that is the
        operator's call to make.
        """
        writer = GGUFWriter(tmp_path / "q.gguf")
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", (512, 8), int(GGMLType.Q8_0))
        writer.write(lambda t: b"\x11" * t.nbytes)

        report = quantize_gguf(tmp_path / "q.gguf", tmp_path / "out.gguf", "IQ0.5_XXXL")
        assert report.tensors_quantized == 1
        assert report.requantized_from == {"Q8_0": 1}
        assert "requantised" in report.describe()

    def test_a_source_type_it_cannot_read_is_copied_and_reported(self, tmp_path):
        writer = GGUFWriter(tmp_path / "q.gguf")
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", (512, 8), int(GGMLType.IQ4_XS))
        writer.write(lambda t: b"\x00" * t.nbytes)

        report = quantize_gguf(tmp_path / "q.gguf", tmp_path / "out.gguf", "IQ0.5_XXXL")
        assert report.tensors_quantized == 0
        assert any("cannot read" in reason for _, reason in report.skipped)

    def test_an_unknown_target_is_refused(self, tmp_path):
        source = _write_gguf(
            tmp_path / "src.gguf", {"a.weight": ((512, 8), _weights(512 * 8))}
        )
        with pytest.raises(HyprslugError, match="Unknown target"):
            quantize_gguf(source, tmp_path / "out.gguf", "IQ9_NOPE")

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(HyprslugError, match="No such model"):
            quantize_gguf(tmp_path / "absent.gguf", tmp_path / "o.gguf", "IQ0.5_XXXL")

    def test_a_mismatched_imatrix_is_ignored_not_misapplied(self, tmp_path):
        """It belongs to a different model; weighting the wrong positions
        is worse than weighting none."""
        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )
        report = quantize_gguf(
            source, tmp_path / "out.gguf", "IQ0.5_XXXL",
            imatrix={"blk.0.attn_q.weight": [1.0, 2.0, 3.0]},
        )
        assert report.tensors_quantized == 1


class TestTheGGUFLayer:
    def test_a_non_gguf_file_is_refused_by_name(self, tmp_path):
        path = tmp_path / "not.gguf"
        path.write_bytes(b"this is not a model")
        with pytest.raises(GGUFError, match="GGUF magic"):
            GGUFFile.read(path)

    def test_tensor_offsets_are_aligned(self, tmp_path):
        """Misalignment produces a file that opens, reports sensible
        shapes, and returns garbage from the second tensor onward."""
        source = _write_gguf(tmp_path / "src.gguf", {
            "a.weight": ((512, 2), _weights(1024)),
            "b.weight": ((512, 3), _weights(1536)),
            "c.weight": ((512, 5), _weights(2560)),
        })
        model = GGUFFile.read(source)
        for tensor in model.tensors:
            assert tensor.offset % DEFAULT_ALIGNMENT == 0, tensor.name

    def test_tensor_bytes_round_trip(self, tmp_path):
        values = _weights(1024, seed=5)
        source = _write_gguf(tmp_path / "src.gguf", {"a.weight": ((512, 2), values)})
        model = GGUFFile.read(source)
        raw = model.tensor_bytes(model.tensors[0])
        assert list(struct.unpack(f"<{len(values)}f", raw)) == pytest.approx(values)

    def test_a_short_write_is_caught(self, tmp_path):
        writer = GGUFWriter(tmp_path / "bad.gguf")
        writer.add_tensor("a.weight", (512, 2), int(GGMLType.F32))
        with pytest.raises(GGUFError, match="the table says"):
            writer.write(lambda tensor: b"\x00" * 4)


class TestTheNamesItAnswersTo:
    def test_doomslug_is_among_them(self):
        assert "hyprslug" in ALIASES
        assert "doomslug" in ALIASES
        assert "doomslugthedestroyer" in ALIASES
        assert "dstd" in ALIASES


class TestSteamrollerActuallyQuantises:
    """`pack_sub_bit` used to copy the file and write a sidecar."""

    def _source(self, tmp_path: Path) -> Path:
        return _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )

    def test_hnx_mode_never_looks_for_llama_cpp(self, tmp_path, monkeypatch):
        """Not 'looks and does not use the result' — never looks.

        resolve_binary() downloads a llama.cpp build when it cannot find
        one, so a lookup that happens and goes unused still means the
        machine ends up with llama.cpp on it.
        """
        from hypernix.quant import steamroller as module

        def _explode(self):
            raise AssertionError("resolve_binary() was called in hnx mode")

        monkeypatch.setattr(module.Steamroller, "resolve_binary", _explode)

        source = self._source(tmp_path)
        out = tmp_path / "out.gguf"
        module.Steamroller(hnx_only=True).run(
            source, "IQ0.5_XXXL", out, source_format="FP32"
        )
        assert out.exists()

    def test_the_output_is_a_real_sub_bit_gguf(self, tmp_path):
        from hypernix.quant.steamroller import Steamroller

        source = self._source(tmp_path)
        out = tmp_path / "out.gguf"
        Steamroller(hnx_only=True).run(source, "IQ0.9_L", out, source_format="FP32")

        written = GGUFFile.read(out)
        assert written.get("blk.0.attn_q.weight").ggml_type == int(GGMLType.HNX_IQ0_9)
        assert out.stat().st_size < source.stat().st_size

    def test_the_output_is_not_a_copy_of_its_input(self, tmp_path):
        """Directly the old behaviour, stated as a test."""
        from hypernix.quant.steamroller import Steamroller

        source = self._source(tmp_path)
        out = tmp_path / "out.gguf"
        Steamroller(hnx_only=True).run(source, "IQ0.5_XXXL", out, source_format="FP32")
        assert out.read_bytes() != source.read_bytes()
        assert out.stat().st_size != source.stat().st_size

    def test_the_sidecar_still_describes_the_run(self, tmp_path):
        from hypernix.quant.steamroller import Steamroller

        source = self._source(tmp_path)
        out = tmp_path / "out.gguf"
        Steamroller(hnx_only=True).run(source, "IQ0.75_M", out, source_format="FP32")

        import json
        sidecar = out.with_suffix(out.suffix + ".hypernix.json")
        assert sidecar.exists()
        recorded = json.loads(sidecar.read_text())
        assert recorded["hypernix.tier"] == "IQ0.75_M"
        assert recorded["hypernix.report"]["tensors_quantized"] == 1

    def test_each_tier_produces_a_different_size(self, tmp_path):
        """Three tiers that emit identical files are three labels."""
        from hypernix.quant.steamroller import Steamroller

        source = self._source(tmp_path)
        sizes = {}
        for tier in TIER_TYPES:
            out = tmp_path / f"{tier}.gguf"
            Steamroller(hnx_only=True).run(source, tier, out, source_format="FP32")
            sizes[tier] = out.stat().st_size
        assert len(set(sizes.values())) == len(sizes), sizes
        assert sizes["IQ0.5_XXXL"] < sizes["IQ0.75_M"] < sizes["IQ0.9_L"]


class TestTheCLI:
    def test_hnx_is_accepted_and_wired(self):
        import contextlib
        import io

        from hypernix.quant.steamroller_cli import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--list-tiers"])
        assert code == 0
        assert "IQ0.5_XXXL" in buffer.getvalue()

    def test_the_module_runs_under_dash_m(self):
        """Without a __main__ guard it imports, runs nothing and exits 0
        — which looks exactly like a quantisation that wrote no file."""
        source = Path("src/hypernix/quant/steamroller_cli.py").read_text()
        assert '__name__ == "__main__"' in source


class TestTheEmbeddingPolicyIsReachable:
    """The tier's name has to describe the file it produces.

    A sub-bit tier leaves ``token_embd`` and the output head in float by
    default, and the reason is sound: at half a bit the embedding table
    is the model. But it has a size consequence nobody chose -- on a 7B
    an untouched F32 table and head are most of the resulting file, so a
    tier called ``IQ0.5_XXXL`` produced something closer to 1.7 bits per
    weight. The policy was reachable from ``hyprslug.quantize_gguf`` and
    from no command line at all, which meant the headline number in the
    docs could not actually be obtained with the tool.
    """

    @pytest.fixture(scope="class")
    def source(self, tmp_path_factory):
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from test_hnxrun import _write_model

        directory = tmp_path_factory.mktemp("embed-policy")
        return _write_model(directory / "tiny.f32.gguf", tokenizer=True)

    def _write(self, source, name, *flags):
        import contextlib
        import io

        from hypernix.interfaces import cli

        out = Path(source).parent / f"{name}.gguf"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = cli.main([
                "quantize", "--source", str(source), "--output", str(out),
                "--type", "IQ0.5_XXXL", "-hnx", *flags,
            ])
        assert code == 0, buffer.getvalue()
        return out

    def _quantize(self, source, name, *flags):
        from hypernix.models.ggufrun import load_gguf

        return load_gguf(
            self._write(source, name, *flags)
        ).model.resident_bits_per_weight

    def _type_of(self, path, tensor):
        """The GGML type ``llama.cpp`` would read for one tensor."""
        from hypernix.quant.gguf import GGUFFile

        for entry in GGUFFile.read(path).tensors:
            if entry.name == tensor:
                return entry.ggml_type
        raise AssertionError(f"{path} has no {tensor}")

    def test_the_default_leaves_the_table_in_float(self, source):
        """The policy this class is named for, read off the file.

        This used to assert a bits-per-weight floor, which measured the
        fixture rather than the policy: the average only stays high while
        the untouched table is a large share of the model, so growing the
        fixture broke a test about a decision that had not changed. The
        decision is per tensor, so the assertion is too.
        """
        from hypernix.quant.gguf import GGMLType

        out = self._write(source, "default")
        assert self._type_of(out, "token_embd.weight") == int(GGMLType.F32)
        assert self._type_of(out, "output.weight") == int(GGMLType.F32)
        # ...and the layers it *is* meant to touch really were touched,
        # so a quantiser that silently did nothing cannot pass this.
        assert self._type_of(out, "blk.0.attn_q.weight") == int(GGMLType.HNX_IQ0_5)

    def test_the_flags_reach_the_two_tensors(self, source):
        """The flags' whole purpose: the headline bit rate is only
        obtainable if these two stop being float."""
        from hypernix.quant.gguf import GGMLType

        out = self._write(
            source, "flagged", "--quantize-embeddings", "--quantize-output"
        )
        assert self._type_of(out, "token_embd.weight") == int(GGMLType.HNX_IQ0_5)
        assert self._type_of(out, "output.weight") == int(GGMLType.HNX_IQ0_5)

    def test_quantising_both_gets_under_a_bit(self, source):
        """The number the tier is named for, obtainable from the CLI."""
        both = self._quantize(
            source, "both", "--quantize-embeddings", "--quantize-output"
        )
        assert both < 1.0

    def test_each_flag_moves_it_on_its_own(self, source):
        default = self._quantize(source, "d2")
        embeddings = self._quantize(source, "e2", "--quantize-embeddings")
        both = self._quantize(
            source, "b2", "--quantize-embeddings", "--quantize-output"
        )
        assert default > embeddings > both

    def test_the_negative_form_is_the_default(self, source):
        """argparse's BooleanOptionalAction, so a script can be explicit
        about wanting the float table rather than relying on a default."""
        assert self._quantize(
            source, "explicit", "--no-quantize-embeddings", "--no-quantize-output"
        ) == self._quantize(source, "d3")

    def test_the_result_still_runs(self, source):
        """A smaller file that does not load is not an improvement."""
        import contextlib
        import io

        from hypernix.interfaces import cli

        out = Path(source).parent / "runs.gguf"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            assert cli.main([
                "quantize", "--source", str(source), "--output", str(out),
                "--type", "IQ0.5_XXXL", "-hnx",
                "--quantize-embeddings", "--quantize-output",
            ]) == 0
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = cli.main([
                "generate", "--model-dir", str(out), "--prompt", "hi",
                "--max-new-tokens", "3",
            ])
        assert code == 0
        assert printed.getvalue().strip()


class TestTheHyprslugCLI:
    def _run(self, *argv: str) -> tuple[int, str]:
        import contextlib
        import io

        from hypernix.quant.hyprslug_cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_list_tiers_names_every_tier(self):
        code, text = self._run("--list-tiers")
        assert code == 0
        for tier in TIER_TYPES:
            assert tier in text

    def test_it_reports_the_real_bit_rate(self):
        """The rate the packer emits, not the rate the name suggests --
        which for the fixed-codebook tiers are different numbers."""
        _, text = self._run("--list-tiers")
        for rate in ("0.938", "0.812", "0.562", "0.250", "1.062", "4.062", "2.062"):
            assert rate in text, rate

    def test_it_quantises_a_file(self, tmp_path):
        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )
        out = tmp_path / "out.gguf"
        code, text = self._run(str(source), "IQ0.5_XXXL", "-o", str(out), "--quiet")
        assert code == 0, text
        assert GGUFFile.read(out).get("blk.0.attn_q.weight").ggml_type == 202

    def test_a_bad_tier_fails_without_a_traceback(self, tmp_path):
        source = _write_gguf(
            tmp_path / "src.gguf", {"a.weight": ((512, 8), _weights(512 * 8))}
        )
        code, _ = self._run(str(source), "NOPE", "-o", str(tmp_path / "o.gguf"))
        assert code == 1

    def test_json_output_is_machine_readable(self, tmp_path):
        import json

        source = _write_gguf(
            tmp_path / "src.gguf",
            {"blk.0.attn_q.weight": ((512, 8), _weights(512 * 8))},
        )
        code, text = self._run(
            str(source), "IQ0.9_L", "-o", str(tmp_path / "o.gguf"), "--json"
        )
        assert code == 0
        report = json.loads(text)
        assert report["tier"] == "IQ0.9_L"
        assert report["tensors_quantized"] == 1

    def test_all_four_names_are_installed(self):
        """People type all four; one implementation."""
        pyproject = Path("pyproject.toml").read_text()
        for alias in ALIASES:
            assert f'{alias} = "hypernix.quant.hyprslug_cli:cli_main"' in pyproject, alias
