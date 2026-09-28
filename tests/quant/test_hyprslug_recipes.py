"""hyprslug writing upstream quant types, with no llama.cpp anywhere.

The sub-bit tiers were always going to need their own quantiser — they
are HyperNix types ``llama-quantize`` has never heard of. The upstream
types did not: they had a perfectly good quantiser, in a binary the
machine might not be able to build. So "hyprslug quantises without
llama.cpp" was only true of the tiers nobody was asking for.

These tests are about the other half: that ``Q4_K_M`` produces a real
Q4_K_M, that the mix puts the wider format where it says it does, and
that an already-quantised GGUF can be requantised — because a Q8_0 file
is the only copy of the model most people have, and "quantise from the
unquantised weights" is advice they cannot take.
"""
from __future__ import annotations

import json
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant import llamaquants as lq
from hypernix.quant.gguf import GGMLType, GGUFFile, GGUFWriter
from hypernix.quant.hyprslug import (
    RECIPES,
    HyprslugError,
    all_targets,
    quantize_gguf,
    resolve_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _weights(count: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.05) for _ in range(count)]


def _model(path: Path, extra: dict | None = None) -> Path:
    """A GGUF shaped like a transformer block, so the mixes have targets."""
    tensors = {
        "token_embd.weight": (512, 4),
        "blk.0.attn_q.weight": (512, 4),
        "blk.0.attn_k.weight": (512, 4),
        "blk.0.attn_v.weight": (512, 4),
        "blk.0.ffn_down.weight": (512, 4),
        "blk.0.ffn_up.weight": (512, 4),
        "blk.0.attn_norm.weight": (512,),
        "output.weight": (512, 4),
    }
    tensors.update(extra or {})
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    payload = {}
    for index, (name, shape) in enumerate(tensors.items()):
        count = 1
        for dim in shape:
            count *= dim
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = struct.pack(f"<{count}f", *_weights(count, seed=index))
    writer.write(lambda tensor: payload[tensor.name])
    return path


def _types(path: Path) -> dict[str, int]:
    model = GGUFFile.read(path)
    return {tensor.name: int(tensor.ggml_type) for tensor in model.tensors}


class TestTheRecipeTable:
    def test_every_recipe_names_a_format_that_exists(self):
        for recipe in RECIPES.values():
            assert recipe.base in lq.FORMATS, recipe.name
            for _, fmt in recipe.overrides:
                assert fmt in lq.FORMATS, f"{recipe.name} -> {fmt}"
            if recipe.output:
                assert recipe.output in lq.FORMATS

    @pytest.mark.parametrize("spelling", ["Q4_K_M", "q4_k_m", "q4-k-m", "q4km", "Q4KM"])
    def test_a_name_is_a_name_however_it_is_typed(self, spelling):
        assert resolve_recipe(spelling) is RECIPES["Q4_K_M"]

    def test_a_sub_bit_tier_is_not_a_recipe(self):
        """The two dispatch to different encoders, so conflating them
        would send a 0.5-bit tier through the K-quant packer."""
        assert resolve_recipe("IQ0.5_XXXL") is None

    def test_the_m_mixes_widen_the_tensors_upstream_widens(self):
        recipe = RECIPES["Q4_K_M"]
        assert recipe.format_for("blk.0.attn_v.weight") == "Q6_K"
        assert recipe.format_for("blk.0.ffn_down.weight") == "Q6_K"
        assert recipe.format_for("blk.0.attn_q.weight") == "Q4_K"
        assert recipe.format_for("output.weight") == "Q6_K"

    def test_the_s_mixes_are_uniform_apart_from_the_head(self):
        recipe = RECIPES["Q4_K_S"]
        for name in ("blk.0.attn_v.weight", "blk.0.ffn_down.weight",
                     "blk.0.attn_q.weight"):
            assert recipe.format_for(name) == "Q4_K"
        assert recipe.format_for("output.weight") == "Q6_K"

    def test_all_targets_lists_both_families(self):
        targets = all_targets()
        assert "Q4_K_M" in targets
        assert "IQ0.5_XXXL" in targets


class TestWritingAnUpstreamQuant:
    @pytest.mark.parametrize("target", ["Q4_0", "Q5_1", "Q8_0", "Q2_K", "Q6_K"])
    def test_the_tensors_carry_the_upstream_type_id(self, tmp_path, target):
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "out.gguf", target)
        types = _types(tmp_path / "out.gguf")
        assert types["blk.0.attn_q.weight"] == int(getattr(GGMLType, target))

    def test_a_mix_really_is_mixed(self, tmp_path):
        """The point of the _M suffix. A "Q4_K_M" whose every tensor is
        Q4_K is a Q4_K_S wearing the wrong name."""
        source = _model(tmp_path / "src.gguf")
        report = quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        types = _types(tmp_path / "out.gguf")

        assert types["blk.0.attn_q.weight"] == int(GGMLType.Q4_K)
        assert types["blk.0.attn_v.weight"] == int(GGMLType.Q6_K)
        assert types["blk.0.ffn_down.weight"] == int(GGMLType.Q6_K)
        assert types["output.weight"] == int(GGMLType.Q6_K)
        assert report.formats == {"Q4_K": 4, "Q6_K": 3}

    def test_the_norm_is_left_alone(self, tmp_path):
        """1-D weights are a rounding error of the size and a large
        fraction of the damage. Every serious quantiser skips them."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        assert _types(tmp_path / "out.gguf")["blk.0.attn_norm.weight"] == int(GGMLType.F32)

    def test_the_file_actually_shrinks(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        assert (tmp_path / "out.gguf").stat().st_size < source.stat().st_size / 3

    def test_a_narrower_target_makes_a_smaller_file(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        sizes = {}
        for target in ("Q8_0", "Q5_K", "Q4_K", "Q2_K"):
            out = tmp_path / f"{target}.gguf"
            quantize_gguf(source, out, target)
            sizes[target] = out.stat().st_size
        assert sizes["Q8_0"] > sizes["Q5_K"] > sizes["Q4_K"] > sizes["Q2_K"]

    def test_the_weights_survive_the_round_trip(self, tmp_path):
        """The check that the file is a model and not a well-formed
        container of noise."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "out.gguf", "Q6_K")

        original = GGUFFile.read(source)
        quantised = GGUFFile.read(tmp_path / "out.gguf")
        name = "blk.0.attn_q.weight"
        before = np.frombuffer(
            original.tensor_bytes(next(t for t in original.tensors if t.name == name)),
            dtype=np.float32,
        )
        tensor = next(t for t in quantised.tensors if t.name == name)
        after = lq.dequantize_array(quantised.tensor_bytes(tensor), int(tensor.ggml_type))
        error = np.sqrt(np.mean((after - before) ** 2)) / np.sqrt(np.mean(before**2))
        assert error < 0.05

    def test_the_metadata_says_what_was_done(self, tmp_path):
        """A sidecar can be lost in a copy; the file itself cannot."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        meta = GGUFFile.read(tmp_path / "out.gguf").metadata
        assert meta["hypernix.quantiser"] == "hyprslug"
        assert meta["hypernix.tier"] == "Q4_K_M"
        assert meta["hypernix.base_format"] == "Q4_K"
        assert meta["hypernix.sub_bit"] is False

    def test_the_architecture_and_tokenizer_metadata_survive(self, tmp_path):
        """A reader that dropped what it did not recognise would strip a
        model's chat template on every quantisation."""
        source = _model(tmp_path / "src.gguf")
        model = GGUFFile.read(source)
        assert model.metadata["general.architecture"] == "llama"
        quantize_gguf(source, tmp_path / "out.gguf", "Q4_K")
        assert GGUFFile.read(tmp_path / "out.gguf").metadata["general.architecture"] == "llama"


class TestRequantising:
    def test_an_existing_quant_can_be_narrowed(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "q8.gguf", "Q8_0")
        report = quantize_gguf(tmp_path / "q8.gguf", tmp_path / "q4.gguf", "Q4_K")

        assert report.tensors_quantized > 0
        assert report.requantized_from == {"Q8_0": 7}
        assert (tmp_path / "q4.gguf").stat().st_size < (tmp_path / "q8.gguf").stat().st_size

    def test_it_says_the_source_was_already_quantised(self, tmp_path):
        """Requantising compounds whatever the first pass lost. Whether
        that matters is the operator's call; whether they get to make it
        is not."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "q8.gguf", "Q8_0")
        report = quantize_gguf(tmp_path / "q8.gguf", tmp_path / "q4.gguf", "Q4_K")
        assert "requantised" in report.describe()
        assert report.to_dict()["requantized_from"] == {"Q8_0": 7}

    def test_going_through_a_wide_quant_costs_little(self, tmp_path):
        """Q8_0 is close enough to lossless that the two-step result
        should be near the one-step one. If it is not, the decoder is
        wrong rather than the idea."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "direct.gguf", "Q4_K")
        quantize_gguf(source, tmp_path / "q8.gguf", "Q8_0")
        quantize_gguf(tmp_path / "q8.gguf", tmp_path / "twostep.gguf", "Q4_K")

        name = "blk.0.attn_q.weight"

        def _values(path):
            model = GGUFFile.read(path)
            tensor = next(t for t in model.tensors if t.name == name)
            raw = model.tensor_bytes(tensor)
            if int(tensor.ggml_type) == int(GGMLType.F32):
                return np.frombuffer(raw, dtype=np.float32)
            return lq.dequantize_array(raw, int(tensor.ggml_type))

        original = _values(source)
        direct = _values(tmp_path / "direct.gguf")
        twostep = _values(tmp_path / "twostep.gguf")

        def _err(candidate):
            return float(
                np.sqrt(np.mean((candidate - original) ** 2))
                / np.sqrt(np.mean(original**2))
            )

        assert _err(twostep) < _err(direct) * 1.5


class TestTheDefaultsMatchTheTarget:
    def test_a_llama_mix_quantises_the_embeddings_and_the_head(self, tmp_path):
        """Upstream does, and a "Q4_K_M" that left the embedding table at
        F32 would be several times the size anyone expects."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M")
        types = _types(tmp_path / "out.gguf")
        assert types["token_embd.weight"] != int(GGMLType.F32)
        assert types["output.weight"] == int(GGMLType.Q6_K)

    def test_a_sub_bit_tier_leaves_them_alone(self, tmp_path):
        """At half a bit the embedding table is the model."""
        source = _model(tmp_path / "src.gguf")
        report = quantize_gguf(source, tmp_path / "out.gguf", "IQ0.5_XXXL")
        reasons = dict(report.skipped)
        assert "token embeddings" in reasons["token_embd.weight"]
        assert "output head" in reasons["output.weight"]

    def test_an_explicit_choice_still_wins(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(
            source, tmp_path / "out.gguf", "Q4_K_M", quantize_embeddings=False
        )
        assert _types(tmp_path / "out.gguf")["token_embd.weight"] == int(GGMLType.F32)


class TestTheImatrixReachesTheEncoder:
    def test_it_is_recorded_in_the_file(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        imatrix = {"blk.0.attn_q.weight": [1.0] * (512 * 4)}
        quantize_gguf(source, tmp_path / "out.gguf", "Q4_K", imatrix=imatrix)
        assert GGUFFile.read(tmp_path / "out.gguf").metadata["hypernix.imatrix"] is True

    def test_it_changes_the_bytes(self, tmp_path):
        """An imatrix that produced an identical file would be a lie in
        the metadata, and there would be nothing to notice it."""
        source = _model(tmp_path / "src.gguf")
        quantize_gguf(source, tmp_path / "plain.gguf", "Q4_K")
        weights = [1.0] * (512 * 4)
        for index in range(0, len(weights), 2):
            weights[index] = 200.0
        quantize_gguf(
            source, tmp_path / "weighted.gguf", "Q4_K",
            imatrix={"blk.0.attn_q.weight": weights},
        )
        assert (tmp_path / "plain.gguf").read_bytes() != (
            tmp_path / "weighted.gguf"
        ).read_bytes()


class TestTheCommandLine:
    def _run(self, *argv):
        return subprocess.run(
            [sys.executable, "-m", "hypernix.quant.hyprslug_cli", *argv],
            capture_output=True, text=True, timeout=300,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        )

    def test_list_tiers_names_both_families(self):
        result = self._run("--list-tiers")
        assert result.returncode == 0, result.stderr
        assert "Q4_K_M" in result.stdout
        assert "IQ0.5_XXXL" in result.stdout

    def test_it_writes_an_upstream_quant(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        out = tmp_path / "out.gguf"
        result = self._run(str(source), "Q4_K_M", "-o", str(out), "--json")
        assert result.returncode == 0, result.stderr
        assert out.exists()
        report = json.loads(result.stdout)
        assert report["tier"] == "Q4_K_M"
        assert report["formats"] == {"Q4_K": 4, "Q6_K": 3}

    def test_an_unknown_target_names_the_known_ones(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        result = self._run(str(source), "Q9_ULTRA", "-o", str(tmp_path / "o.gguf"))
        assert result.returncode == 1
        assert "Q4_K_M" in result.stderr


class TestSteamrollerInHnxMode:
    """``-hnx`` promises no llama.cpp is looked for, downloaded or built.

    Before this it also meant "only the IQ0.x tiers work": a
    ``quantize`` step in hnx mode was skipped outright, so asking for an
    upstream tier produced no output at all.
    """

    def test_an_upstream_tier_is_written_by_hyprslug(self, tmp_path, monkeypatch):
        from hypernix.quant import steamroller

        def _refuse(*_args, **_kwargs):
            raise AssertionError("-hnx looked for llama-quantize")

        monkeypatch.setattr(steamroller.Steamroller, "resolve_binary", _refuse)
        source = _model(tmp_path / "src.gguf")
        out = tmp_path / "out.gguf"
        result = steamroller.Steamroller(hnx_only=True).run(
            source, "Q3_K_L", out, source_format="FP16"
        )
        assert out.exists()
        assert result["bytes"] > 0
        types = _types(out)
        assert types["blk.0.attn_q.weight"] == int(GGMLType.Q3_K)

    def test_a_tier_hyprslug_cannot_write_says_so(self, tmp_path, monkeypatch):
        from hypernix.quant import steamroller

        monkeypatch.setattr(
            steamroller.Steamroller, "resolve_binary",
            lambda self: (_ for _ in ()).throw(AssertionError("looked for the binary")),
        )
        source = _model(tmp_path / "src.gguf")
        with pytest.raises(steamroller.SteamrollerError, match="no encoder"):
            steamroller.Steamroller(hnx_only=True).run(
                source, "IQ1_M", tmp_path / "out.gguf", source_format="FP16"
            )


class TestTheRefusals:
    def test_an_unknown_target_lists_what_it_does_write(self, tmp_path):
        source = _model(tmp_path / "src.gguf")
        with pytest.raises(HyprslugError, match="Q4_K_M"):
            quantize_gguf(source, tmp_path / "o.gguf", "Q9_ULTRA")

    def test_a_tensor_that_does_not_divide_is_copied_not_mangled(self, tmp_path):
        source = _model(tmp_path / "src.gguf", extra={"blk.0.odd.weight": (100, 3)})
        report = quantize_gguf(source, tmp_path / "out.gguf", "Q4_K")
        reasons = dict(report.skipped)
        assert "divide into 256" in reasons["blk.0.odd.weight"]
        assert _types(tmp_path / "out.gguf")["blk.0.odd.weight"] == int(GGMLType.F32)

    def test_a_legacy_target_takes_a_tensor_the_k_quants_cannot(self, tmp_path):
        """A 32-element block divides where a 256-element one does not,
        so the same tensor is packable at Q4_0 and not at Q4_K. Reporting
        it as "cannot be packed" for both would be false for one."""
        source = _model(tmp_path / "src.gguf", extra={"blk.0.small.weight": (32, 3)})
        report = quantize_gguf(source, tmp_path / "out.gguf", "Q4_0")
        assert "blk.0.small.weight" not in dict(report.skipped)
        assert _types(tmp_path / "out.gguf")["blk.0.small.weight"] == int(GGMLType.Q4_0)
