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


class TestTheWriterRefusesEvenIfTheCheckRegresses:
    """Defence in depth, and the reason this is worth having twice.

    ``_should_quantize`` deciding correctly is one line of code away from
    deciding incorrectly again -- it already did once, and the round trip
    passed because the reader shared the writer's misconception. So the
    *file format layer* refuses too: ``tensor_nbytes`` is on the path of
    every write, and it will not lay out a tensor whose ``ne[0]`` cannot
    divide into its type's block.

    That turns "we fixed this bug" into "this file cannot be produced".
    """

    def test_add_tensor_refuses_a_row_that_cannot_divide(self, tmp_path):
        from hypernix.quant.gguf import GGUFError

        writer = GGUFWriter(tmp_path / "refused.gguf")
        writer.set_metadata("general.architecture", "llama")
        with pytest.raises(GGUFError, match="elements per row"):
            writer.add_tensor(
                "blk.0.ssm_conv1d.weight", (4, 4096), int(GGMLType.HNX_IQ0_5)
            )

    def test_the_message_names_ne0_as_the_dimension(self, tmp_path):
        """A size error that says "16384 elements" sends somebody to
        look at a number that is fine."""
        from hypernix.quant.gguf import GGUFError

        writer = GGUFWriter(tmp_path / "refused.gguf")
        with pytest.raises(GGUFError, match="ne\\[0\\]"):
            writer.add_tensor("w", (4, 4096), int(GGMLType.HNX_IQ0_5))

    def test_a_legal_tensor_is_unaffected(self, tmp_path):
        writer = GGUFWriter(tmp_path / "fine.gguf")
        tensor = writer.add_tensor("w", (256, 512), int(GGMLType.HNX_IQ0_5))
        assert tensor.nbytes > 0

    def test_f32_of_any_shape_is_unaffected(self, tmp_path):
        """Block size 1 divides everything; norms and 1-D tensors must
        keep working whatever their shape."""
        writer = GGUFWriter(tmp_path / "fine.gguf")
        for shape in [(7,), (4, 4096), (3, 5, 7)]:
            assert writer.add_tensor(f"w{shape}", shape, int(GGMLType.F32)).nbytes > 0

    def test_the_reader_still_opens_a_file_that_has_the_bug(self, tmp_path):
        """The guard belongs on writes only.

        Files with this bug exist -- that is why ggufcheck exists -- and
        a reader that refused them would make them undiagnosable by the
        tool written to repair them.
        """
        from hypernix.quant import gguf

        path = tmp_path / "legacy.gguf"
        original = gguf.tensor_nbytes
        gguf.tensor_nbytes = gguf.tensor_nbytes_unchecked
        try:
            writer = GGUFWriter(path)
            writer.set_metadata("general.architecture", "llama")
            writer.add_tensor("blk.0.ssm_conv1d.weight", (4, 256), int(GGMLType.HNX_IQ0_5))
            writer.write(lambda t: bytes(t.nbytes))
        finally:
            gguf.tensor_nbytes = original

        model = GGUFFile.read(path)
        assert [t.name for t in model.tensors] == ["blk.0.ssm_conv1d.weight"]
        assert int(model.tensors[0].ggml_type) == int(GGMLType.HNX_IQ0_5)

    def test_quantize_gguf_cannot_write_one_even_with_the_check_regressed(
        self, tmp_path, monkeypatch
    ):
        """The whole point, stated as the scenario that produced the bug.

        With ``_should_quantize`` put back to the element-count test that
        shipped, quantising a model with an SSM convolution in it must
        fail loudly rather than produce a file that llama.cpp refuses
        after an hour of quantisation.
        """
        from hypernix.quant import hyprslug
        from hypernix.quant.gguf import GGUFError

        def regressed(tensor, *, block, **kwargs):
            if tensor.elements % block:
                return False, "the old, wrong check"
            return True, ""

        monkeypatch.setattr(hyprslug, "_should_quantize", regressed)
        source = self._model(tmp_path / "in.gguf", {"blk.0.ssm_conv1d.weight": (4, 4096)})
        with pytest.raises(GGUFError, match="elements per row"):
            quantize_gguf(source, tmp_path / "out.gguf", tier="IQ0.5_XXXL")

    #: The same builder the load tests use, rather than a second copy of
    #: a GGUF writer that could drift from it. Re-wrapped because
    #: reaching through the class hands back the plain function.
    _model = staticmethod(TestTheFileActuallyLoads._model)
