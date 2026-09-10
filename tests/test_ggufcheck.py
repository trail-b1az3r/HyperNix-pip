"""Checking and repairing a GGUF that llama.cpp refuses to load.

The bug these exist for shipped: files were written with a
block-quantised type on tensors whose ``ne[0]`` is not a multiple of the
block size, and llama.cpp rejects them at load. Fixing the quantiser
stops new ones being made and does nothing for the ones already on
disk, which is where every model somebody has already spent an hour
quantising lives.

So the fixture here *reintroduces the bug on purpose* -- it writes the
bad file directly rather than trusting a quantiser to misbehave -- and
the tests are about detecting and repairing it.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant.gguf import GGMLType, GGUFError, GGUFFile, GGUFWriter
from hypernix.quant.ggufcheck import (
    block_elements,
    check_gguf,
    repair_gguf,
    row_length,
)
from hypernix.quant.hyprslug import TIER_TYPES
from hypernix.quant.subbit import quantize_tensor

#: The tensor from the report: an SSM convolution weight is [4, N], so
#: four elements per row and a total of 4N that divides by 256 whenever
#: N does. Exactly the shape the element-count check let through.
SSM_SHAPE = (4, 4096)
CLEAN_SHAPE = (256, 256)


def _packed(shape, packing="quad_code_xxxl", seed=0):
    rng = np.random.default_rng(seed)
    count = int(np.prod(shape))
    return bytes(quantize_tensor(rng.normal(0, 0.08, count).astype(np.float32).tolist(), packing))


@pytest.fixture(scope="module")
def unloadable(tmp_path_factory):
    """A file with one good sub-bit tensor and one llama.cpp refuses.

    Built the way the pre-0.72.4.post16 writer built it: ``add_tensor``
    now refuses a row that cannot divide, which is the point of that
    guard, so producing the artefact under test means stepping around it
    deliberately rather than hoping the quantiser misbehaves.
    """
    from hypernix.quant import gguf

    path = tmp_path_factory.mktemp("ggufcheck") / "unloadable.gguf"
    original = gguf.tensor_nbytes
    gguf.tensor_nbytes = gguf.tensor_nbytes_unchecked
    try:
        return _write_unloadable(path)
    finally:
        gguf.tensor_nbytes = original


def _write_unloadable(path: Path) -> Path:
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("hypernix.tier", "IQ0.5_XXXL")
    writer.set_metadata("hypernix.sub_bit", True)
    type_id = int(GGMLType.HNX_IQ0_5)
    payload = {
        "blk.0.attn_q.weight": _packed(CLEAN_SHAPE),
        "blk.0.ssm_conv1d.weight": _packed(SSM_SHAPE, seed=1),
    }
    writer.add_tensor("blk.0.attn_q.weight", CLEAN_SHAPE, type_id)
    writer.add_tensor("blk.0.ssm_conv1d.weight", SSM_SHAPE, type_id)
    writer.write(lambda tensor: payload[tensor.name])
    return path


@pytest.fixture(scope="module")
def loadable(tmp_path_factory):
    path = tmp_path_factory.mktemp("ggufcheck-ok") / "loadable.gguf"
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("hypernix.tier", "IQ0.5_XXXL")
    payload = {"blk.0.attn_q.weight": _packed(CLEAN_SHAPE)}
    writer.add_tensor("blk.0.attn_q.weight", CLEAN_SHAPE, int(GGMLType.HNX_IQ0_5))
    writer.write(lambda tensor: payload[tensor.name])
    return path


class TestItFindsWhatLlamaCppWouldRefuse:
    def test_a_bad_file_is_not_loadable(self, unloadable):
        assert check_gguf(unloadable).loadable is False

    def test_a_good_file_is(self, loadable):
        assert check_gguf(loadable).loadable is True

    def test_it_names_the_offending_tensor_and_only_that_one(self, unloadable):
        """A checker that flags the whole file is no help in finding the
        tensor to look at."""
        bad = check_gguf(unloadable).bad
        assert [entry.name for entry in bad] == ["blk.0.ssm_conv1d.weight"]

    def test_the_message_is_the_one_llama_cpp_prints(self, unloadable):
        """Word for word, so pasting the error into a search finds this.

        The name in it is the *tier* -- IQ0.5_XXXL, which is what the
        patched llama.cpp prints -- and not HNX_IQ0_5, which is only how
        Python spells the enum member.
        """
        entry = check_gguf(unloadable).bad[0]
        assert entry.llama_cpp_message == (
            "tensor 'blk.0.ssm_conv1d.weight' of type 202 (IQ0.5_XXXL) has 4 "
            "elements per row, not a multiple of block size (256)"
        )

    def test_the_tier_name_is_not_the_enum_spelling(self):
        """Guarding the line above against a well-meaning simplification
        back to ``GGMLType(id).name``."""
        from hypernix.quant.ggufcheck import _type_name

        assert _type_name(202) == "IQ0.5_XXXL"
        assert _type_name(int(GGMLType.F32)) == "F32"
        assert "type 9999" in _type_name(9999)

    @pytest.mark.parametrize("tier", sorted(TIER_TYPES))
    def test_every_tier_id_resolves_to_its_tier_name(self, tier):
        from hypernix.quant.ggufcheck import _type_name

        assert _type_name(TIER_TYPES[tier][0]) == tier

    def test_it_reads_the_table_and_not_the_data(self, unloadable, monkeypatch):
        """Checking a 40 GB model has to cost what checking a small one
        costs, or nobody will run it on the file that needs it."""
        from hypernix.quant import gguf

        def refuse(self, tensor):
            raise AssertionError(f"check_gguf read {tensor.name}'s data")

        monkeypatch.setattr(gguf.GGUFFile, "tensor_bytes", refuse)
        assert check_gguf(unloadable).bad

    def test_f32_is_never_flagged(self, tmp_path):
        """Block size 1 divides everything; a 1-D F32 norm must not be
        reported as a problem."""
        path = tmp_path / "norms.gguf"
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_norm.weight", (7,), int(GGMLType.F32))
        writer.write(lambda t: struct.pack("<7f", *([1.0] * 7)))
        assert check_gguf(path).loadable

    def test_row_length_is_the_first_dimension(self):
        from hypernix.quant.gguf import GGUFTensor

        assert row_length(GGUFTensor("t", (4, 4096), 202, 0)) == 4

    def test_block_elements_is_elements_not_bytes(self):
        """The two live in the same table and are easy to swap; 256
        elements of IQ0.5 are 18 bytes, so a swap is loud here and
        silent in arithmetic."""
        from hypernix.quant.gguf import type_size_bytes

        assert block_elements(int(GGMLType.HNX_IQ0_5)) == 256
        assert type_size_bytes(int(GGMLType.HNX_IQ0_5)) != 256

    def test_an_unknown_type_is_not_our_call(self, tmp_path):
        assert block_elements(int(GGMLType.F32)) == 1
        with pytest.raises(GGUFError, match="Unknown GGML type"):
            block_elements(9999)

    def test_the_report_counts_types_and_carries_the_tier(self, unloadable):
        report = check_gguf(unloadable)
        assert report.types == {"IQ0.5_XXXL": 2}
        assert report.tier == "IQ0.5_XXXL"
        assert report.tensors == 2

    def test_the_json_form_carries_the_message(self, unloadable):
        payload = check_gguf(unloadable).as_dict()
        assert payload["loadable"] is False
        assert payload["bad"][0]["elements_per_row"] == 4
        assert payload["bad"][0]["block"] == 256
        assert "not a multiple of block size" in payload["bad"][0]["llama_cpp"]

    def test_the_description_tells_you_what_to_do(self, unloadable):
        text = check_gguf(unloadable).describe()
        assert "blk.0.ssm_conv1d.weight" in text
        assert "--repair-to" in text
        assert "Re-quantise" in text


class TestRepair:
    def test_the_repaired_file_loads(self, unloadable, tmp_path):
        out = tmp_path / "repaired.gguf"
        repair_gguf(unloadable, out)
        assert check_gguf(out).loadable

    def test_only_the_broken_tensor_changed_type(self, unloadable, tmp_path):
        """A repair that re-typed everything would be a silent
        dequantisation of the whole model."""
        out = tmp_path / "repaired.gguf"
        report = repair_gguf(unloadable, out)
        assert report.repaired == ["blk.0.ssm_conv1d.weight"]
        assert report.copied == 1
        types = {t.name: t.ggml_type for t in GGUFFile.read(out).tensors}
        assert types["blk.0.ssm_conv1d.weight"] == int(GGMLType.F32)
        assert types["blk.0.attn_q.weight"] == int(GGMLType.HNX_IQ0_5)

    def test_the_untouched_tensor_is_byte_identical(self, unloadable, tmp_path):
        """Copied through, not round-tripped: a repair that re-encoded
        the good tensors would lose a little more of the model every
        time somebody ran it."""
        out = tmp_path / "repaired.gguf"
        repair_gguf(unloadable, out)
        before = GGUFFile.read(unloadable)
        after = GGUFFile.read(out)
        name = "blk.0.attn_q.weight"
        original = before.tensor_bytes(next(t for t in before.tensors if t.name == name))
        copied = after.tensor_bytes(next(t for t in after.tensors if t.name == name))
        assert original == copied

    def test_the_values_are_what_the_packer_produced(self, unloadable, tmp_path):
        """Widened, not invented: the F32 it writes has to be the
        dequantisation of the bytes that were there."""
        from hypernix.models.hnxrun import _dequantize

        out = tmp_path / "repaired.gguf"
        repair_gguf(unloadable, out)
        before = GGUFFile.read(unloadable)
        source = next(t for t in before.tensors if t.name == "blk.0.ssm_conv1d.weight")
        expected = _dequantize(before.tensor_bytes(source), source.ggml_type, source.elements)
        after = GGUFFile.read(out)
        target = next(t for t in after.tensors if t.name == "blk.0.ssm_conv1d.weight")
        got = np.frombuffer(after.tensor_bytes(target), dtype="<f4", count=target.elements)
        assert np.allclose(got, np.asarray(expected)[: target.elements])

    def test_the_shape_survives(self, unloadable, tmp_path):
        out = tmp_path / "repaired.gguf"
        repair_gguf(unloadable, out)
        after = GGUFFile.read(out)
        target = next(t for t in after.tensors if t.name == "blk.0.ssm_conv1d.weight")
        assert target.shape == SSM_SHAPE

    def test_metadata_comes_across(self, unloadable, tmp_path):
        """Dropping keys is how a repair strips a chat template."""
        out = tmp_path / "repaired.gguf"
        repair_gguf(unloadable, out)
        after = GGUFFile.read(out)
        assert after.metadata["hypernix.tier"] == "IQ0.5_XXXL"
        assert after.metadata["general.architecture"] == "llama"

    def test_repairing_in_place_is_refused(self, unloadable):
        """Writing over the input mid-read would destroy the only copy
        of a model somebody may not be able to rebuild."""
        with pytest.raises(GGUFError, match="not the model being repaired"):
            repair_gguf(unloadable, unloadable)

    def test_it_is_idempotent(self, unloadable, tmp_path):
        once = tmp_path / "once.gguf"
        twice = tmp_path / "twice.gguf"
        repair_gguf(unloadable, once)
        report = repair_gguf(once, twice)
        assert report.repaired == []

    def test_the_report_admits_the_file_grew(self, unloadable, tmp_path):
        """F32 is 64x the size of half a bit. A report implying a repair
        is free would be lying about the trade."""
        out = tmp_path / "repaired.gguf"
        report = repair_gguf(unloadable, out)
        assert report.grew_by > 0
        assert "Re-quantising from the base model is better." in report.describe()


class TestTheCLI:
    def _run(self, *argv):
        import contextlib
        import io

        from hypernix.quant.hyprslug_cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_check_exits_non_zero_on_a_bad_file(self, unloadable):
        """So a build step can gate on it without parsing anything."""
        code, text = self._run(str(unloadable), "--check")
        assert code == 1
        assert "blk.0.ssm_conv1d.weight" in text

    def test_check_exits_zero_on_a_good_one(self, loadable):
        code, _ = self._run(str(loadable), "--check")
        assert code == 0

    def test_check_does_not_need_a_tier(self, unloadable):
        """The file already has one; asking again is how a repair gets
        run with the wrong one."""
        code, _ = self._run(str(unloadable), "--check")
        assert code == 1

    def test_a_missing_file_says_so(self, tmp_path):
        code, text = self._run(str(tmp_path / "nope.gguf"), "--check")
        assert code == 2
        assert "No such file" in text

    def test_check_with_no_source_is_an_error_not_a_traceback(self):
        code, text = self._run("--check")
        assert code == 2
        assert "need a GGUF" in text

    def test_json_is_parseable(self, unloadable):
        import json

        code, text = self._run(str(unloadable), "--check", "--json")
        assert code == 1
        assert json.loads(text)["bad"][0]["elements_per_row"] == 4

    def test_repair_writes_a_file_that_checks_clean(self, unloadable, tmp_path):
        out = tmp_path / "cli-repaired.gguf"
        code, _ = self._run(str(unloadable), "--repair-to", str(out))
        assert code == 0
        assert check_gguf(out).loadable

    def test_repairing_a_healthy_file_says_so_and_writes_nothing(
        self, loadable, tmp_path
    ):
        out = tmp_path / "unnecessary.gguf"
        code, text = self._run(str(loadable), "--repair-to", str(out))
        assert code == 0
        assert "nothing to repair" in text
        assert not out.exists()


class TestTheTiersLlamaCppCanActuallyLoad:
    """hyprslug offers seven tiers; the ggml patch registers five.

    INT4 (205) and FP2 (206) quantise, write a well-formed GGUF, and run
    under HyperNix's own runtime. No llama.cpp build can open them: the
    patch adds enum members 200-204 and pins ``GGML_TYPE_COUNT`` to 205,
    so those two ids are past the end of both trait tables and gguf.cpp
    rejects them on the type check.

    Nothing said so, which is how somebody spends an hour quantising to
    a tier ``llama-server`` will never load.
    """

    PATCH = Path(__file__).resolve().parents[1] / "native/ggml-hnx/tools/patch_llamacpp.py"

    def _registered_in_c(self) -> set[int]:
        """The ids the patch script really inserts, parsed from the C it
        emits -- not from a list restated in a comment.

        The difference matters: a check that read a docstring would go on
        passing after somebody added a type to the enum and not to this
        table, which is the exact failure this test exists to catch.
        """
        import re

        text = self.PATCH.read_text()
        return {
            int(value)
            for value in re.findall(
                r"GGML_TYPE_HNX_\w+\s*=\s*(\d+)\s*,", text
            )
        }

    def test_the_patch_script_exists_where_this_test_thinks(self):
        assert self.PATCH.is_file()

    def test_the_table_matches_the_c_enum(self):
        from hypernix.quant.ggufcheck import LLAMA_CPP_REGISTERED_TYPES

        assert self._registered_in_c() == set(LLAMA_CPP_REGISTERED_TYPES)

    def test_the_type_count_is_one_past_the_highest_registered_id(self):
        """``GGML_TYPE_COUNT`` sizes both trait tables, so it is what
        actually decides which ids gguf.cpp will accept."""
        import re

        from hypernix.quant.ggufcheck import LLAMA_CPP_REGISTERED_TYPES

        text = self.PATCH.read_text()
        count = int(re.search(r"^HNX_TYPE_COUNT\s*=\s*(\d+)", text, re.M).group(1))
        assert count == max(LLAMA_CPP_REGISTERED_TYPES) + 1

    @pytest.mark.parametrize("tier", ["INT4", "FP2"])
    def test_the_two_python_only_tiers_are_reported_as_such(self, tier):
        from hypernix.quant.ggufcheck import llama_cpp_can_load_type

        assert not llama_cpp_can_load_type(TIER_TYPES[tier][0])

    @pytest.mark.parametrize(
        "tier", ["IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL", "IQ0.25_UXL", "INT1"]
    )
    def test_the_five_registered_tiers_are_loadable(self, tier):
        from hypernix.quant.ggufcheck import llama_cpp_can_load_type

        assert llama_cpp_can_load_type(TIER_TYPES[tier][0])

    def test_every_tier_is_accounted_for_either_way(self):
        """So a new tier cannot be added without landing on one side."""
        from hypernix.quant.ggufcheck import llama_cpp_can_load_type

        verdicts = {
            tier: llama_cpp_can_load_type(type_id)
            for tier, (type_id, _packing) in TIER_TYPES.items()
        }
        assert len(verdicts) == len(TIER_TYPES)
        assert sum(verdicts.values()) == 5, verdicts

    def test_upstream_types_are_never_flagged(self):
        from hypernix.quant.ggufcheck import llama_cpp_can_load_type

        for kind in (GGMLType.F32, GGMLType.F16, GGMLType.Q4_K, GGMLType.Q6_K):
            assert llama_cpp_can_load_type(int(kind))

    def test_a_file_in_an_unregistered_type_is_reported(self, tmp_path):
        path = tmp_path / "int4.gguf"
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        writer.set_metadata("hypernix.tier", "INT4")
        writer.add_tensor("blk.0.attn_q.weight", CLEAN_SHAPE, int(GGMLType.HNX_INT4))
        writer.write(lambda t: bytes(t.nbytes))
        report = check_gguf(path)
        assert report.unregistered == ["INT4"]
        assert report.loadable is False
        assert report.runs_under_hnxrun is True

    def test_the_advice_names_the_tiers_that_would_work(self, tmp_path):
        """"cannot be loaded" without a way forward is not much help."""
        path = tmp_path / "fp2.gguf"
        writer = GGUFWriter(path)
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("blk.0.attn_q.weight", CLEAN_SHAPE, int(GGMLType.HNX_FP2))
        writer.write(lambda t: bytes(t.nbytes))
        text = check_gguf(path).describe()
        assert "IQ0.5_XXXL" in text
        assert "hnx generate" in text

    def test_a_registered_tier_is_still_reported_loadable(self, loadable):
        report = check_gguf(loadable)
        assert report.unregistered == []
        assert report.loadable is True
