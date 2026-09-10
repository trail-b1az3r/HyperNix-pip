"""Row length, not element count, decides whether a tensor can be packed.

From a real failure. The desktop llama.cpp built fine with the HyperNix
patch, and then refused a model hyprslug had made:

    gguf_init_from_reader: tensor 'blk.0.ssm_conv1d.weight' of type 202
    (IQ0.5_XXXL) has 4 elements per row, not a multiple of block size
    (256)

GGML quantises **row by row**, so the constraint is on ``ne[0]`` — the
row length. `_should_quantize` checked the *total* element count
instead, and the two differ in exactly the case that bit: a state-space
convolution weight is ``[4, N]``, four elements per row, and a total of
``4N`` that divides into 256 whenever ``N`` does. The guard passed, the
tensor was packed, and the file would not load.

The row check subsumes the total: if ``ne[0]`` divides into the block
size then so does the product. It is also why packing the flattened
array is safe at all — with ``ne[0]`` a multiple of the block, no block
ever straddles two rows.
"""
from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from hypernix.quant.gguf import GGMLType, GGUFFile, GGUFTensor, GGUFWriter
from hypernix.quant.hyprslug import _should_quantize, quantize_gguf
from hypernix.quant.subbit import BLOCK_SIZE


def tensor(name: str, shape: tuple[int, ...]) -> GGUFTensor:
    return GGUFTensor(name=name, shape=shape, ggml_type=int(GGMLType.F32), offset=0)


def can_pack(shape: tuple[int, ...], *, block: int = BLOCK_SIZE, name: str = "blk.0.w.weight"):
    return _should_quantize(
        tensor(name, shape), block=block,
        quantize_embeddings=True, quantize_output=True,
    )


class TestTheReportedTensor:
    """`blk.0.ssm_conv1d.weight`, four elements per row."""

    @pytest.mark.parametrize("rows", [64, 1024, 5120])
    def test_it_is_not_packed(self, rows):
        ok, reason = can_pack((4, rows), name="blk.0.ssm_conv1d.weight")
        assert not ok
        assert "per row" in reason

    @pytest.mark.parametrize("rows", [64, 1024, 5120])
    def test_the_old_check_would_have_packed_it(self, rows):
        """Which is why this shipped: the totals all divide cleanly.

        If this ever fails, the shapes stopped being a counter-example
        and the test below is no longer testing anything.
        """
        assert (4 * rows) % BLOCK_SIZE == 0

    def test_the_reason_names_the_real_constraint(self):
        """"20480 elements do not divide into 256" was both wrong and
        confusing, because 20480 does divide into 256."""
        _, reason = can_pack((4, 5120), name="blk.0.ssm_conv1d.weight")
        assert "4 elements per row" in reason
        assert "20480" not in reason


class TestWhatIsAndIsNotPackable:
    @pytest.mark.parametrize(
        "shape",
        [(4096, 4096), (11008, 4096), (256, 3), (512, 4)],
    )
    def test_a_row_that_divides_is_packed(self, shape):
        ok, reason = can_pack(shape)
        assert ok, reason

    @pytest.mark.parametrize(
        "shape",
        [(4, 5120), (100, 256), (255, 256), (1, 4096), (128, 512)],
    )
    def test_a_row_that_does_not_is_copied(self, shape):
        ok, _ = can_pack(shape)
        assert not ok

    def test_a_one_dimensional_tensor_is_still_copied(self):
        """Norms and biases: all of the damage, none of the size."""
        ok, reason = can_pack((4096,))
        assert not ok
        assert "1-D" in reason

    def test_the_check_scales_with_the_block_size(self):
        """A 32-element block accepts rows a 256-element one refuses."""
        assert can_pack((128, 512), block=32)[0]
        assert not can_pack((128, 512), block=256)[0]


class TestTheFileActuallyLoads:
    """End to end, because the guard is only interesting if the output
    is readable — which is the property the real failure violated."""

    @staticmethod
    def _model(path: Path, extra: dict[str, tuple[int, ...]]) -> Path:
        shapes: dict[str, tuple[int, ...]] = {
            "token_embd.weight": (512, 4),
            "blk.0.attn_q.weight": (512, 4),
            "blk.0.ffn_down.weight": (512, 4),
            "blk.0.attn_norm.weight": (512,),
            "output.weight": (512, 4),
        }
        shapes.update(extra)
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        payload = {}
        rng = random.Random(0)
        for name, shape in shapes.items():
            count = 1
            for dim in shape:
                count *= dim
            writer.add_tensor(name, shape, int(GGMLType.F32))
            payload[name] = struct.pack(
                f"<{count}f", *[rng.gauss(0.0, 0.05) for _ in range(count)]
            )
        writer.write(lambda t: payload[t.name])
        return path

    def test_an_ssm_convolution_survives_a_quantise(self, tmp_path):
        source = self._model(
            tmp_path / "in.gguf", {"blk.0.ssm_conv1d.weight": (4, 5120)}
        )
        out = tmp_path / "out.gguf"
        report = quantize_gguf(source, out, tier="IQ0.5_XXXL")

        model = GGUFFile.read(out)
        by_name = {t.name: t for t in model.tensors}
        conv = by_name["blk.0.ssm_conv1d.weight"]
        assert int(conv.ggml_type) == int(GGMLType.F32), (
            "the SSM convolution was packed into a block type it does not fit"
        )
        assert conv.shape == (4, 5120)
        assert any(
            "ssm_conv1d" in name for name, _ in report.skipped
        ), "it was packed silently rather than reported as skipped"

    @pytest.mark.parametrize(
        "name", ["blk.0.attn_q.weight", "blk.0.ffn_down.weight"]
    )
    def test_the_rest_of_the_model_is_still_packed(self, tmp_path, name):
        """The guard must not turn into "copy everything".

        Named tensors rather than a count: embeddings and the output
        head are skipped by default, so a count is really a test of
        those defaults and moves whenever they do.
        """
        source = self._model(
            tmp_path / "in.gguf", {"blk.0.ssm_conv1d.weight": (4, 5120)}
        )
        out = tmp_path / "out.gguf"
        quantize_gguf(source, out, tier="IQ0.5_XXXL")
        packed = {
            t.name: int(t.ggml_type) for t in GGUFFile.read(out).tensors
        }
        assert packed[name] not in (int(GGMLType.F32), int(GGMLType.F16)), (
            f"{name} has a row of 512, which fits — it should have been packed"
        )

    def test_every_packed_tensor_has_a_row_that_fits(self, tmp_path):
        """The invariant the loader checks, asserted on the output."""
        source = self._model(
            tmp_path / "in.gguf",
            {
                "blk.0.ssm_conv1d.weight": (4, 5120),
                "blk.1.ssm_conv1d.weight": (4, 64),
                "blk.0.odd.weight": (100, 256),
            },
        )
        out = tmp_path / "out.gguf"
        quantize_gguf(source, out, tier="IQ0.5_XXXL")

        unquantised = {int(GGMLType.F32), int(GGMLType.F16)}
        for t in GGUFFile.read(out).tensors:
            if int(t.ggml_type) in unquantised:
                continue
            assert int(t.shape[0]) % BLOCK_SIZE == 0, (
                f"{t.name} is type {t.ggml_type} with {t.shape[0]} elements per "
                f"row — llama.cpp will refuse this file"
            )


class TestDflash2HasTheSameGuard:
    """It picked tensors the same wrong way, so a draft derived from an
    SSM architecture would have failed identically."""

    def test_it_checks_the_row_not_the_total(self):
        source = Path("src/hypernix/quant/dflash2.py").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        assert "tensor.elements % block_size" not in code, (
            "dflash2 is back to counting elements instead of the row length"
        )
        assert "int(tensor.shape[0]) % block_size" in code
