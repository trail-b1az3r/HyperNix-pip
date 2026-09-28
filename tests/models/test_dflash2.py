"""Dflash2: a draft model carried inside the model it drafts for.

Two properties matter more than everything else here, and both are the
kind that fail silently.

The first is that attaching a draft must not change the model. A file
that gained a draft and lost a chat template, or whose base tensors moved
by a byte, is a regression nobody notices until generation is subtly
wrong — so `strip` is checked to reproduce the original exactly, and the
base tensors are compared byte for byte through the attached file.

The second is that speculative decoding produces the *same tokens* the
base model would have produced alone. That is the whole reason it is
safe: a bad draft costs time, never correctness. So the acceptance loop
is tested against a target whose answers are known, including the cases
where the draft is perfect, useless, and wrong halfway through.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from hypernix.quant import llamaquants as lq
from hypernix.quant.dflash2 import (
    PREFIX,
    VERSION,
    Dflash2Error,
    attach,
    extract,
    has_draft,
    plan_draft,
    read_draft_info,
    speculate,
    strip,
)
from hypernix.quant.gguf import GGMLType, GGUFFile, GGUFWriter

REPO_ROOT = Path(__file__).resolve().parents[2]

PARTS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def _base_model(path: Path, *, blocks: int = 8, extra_metadata: dict | None = None) -> Path:
    """A GGUF shaped like a transformer, small enough to be quick."""
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.name", "tiny")
    writer.set_metadata("llama.block_count", blocks)
    writer.set_metadata("tokenizer.chat_template", "{{ messages }}")
    for key, value in (extra_metadata or {}).items():
        writer.set_metadata(key, value)

    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (256, 4),
        "output.weight": (256, 4),
        "output_norm.weight": (256,),
    }
    for index in range(blocks):
        for part in PARTS:
            shapes[f"blk.{index}.{part}.weight"] = (256, 4)
        shapes[f"blk.{index}.attn_norm.weight"] = (256,)

    payload = {}
    for seed, (name, shape) in enumerate(shapes.items()):
        count = 1
        for dim in shape:
            count *= dim
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = struct.pack(
            f"<{count}f", *[0.01 * (((i + seed) % 13) - 6) for i in range(count)]
        )
    writer.write(lambda tensor: payload[tensor.name])
    return path


@pytest.fixture
def base(tmp_path):
    return _base_model(tmp_path / "base.gguf")


@pytest.fixture
def attached(tmp_path, base):
    attach(base, tmp_path / "with-draft.gguf", depth=0.4)
    return tmp_path / "with-draft.gguf"


class TestChoosingLayers:
    def test_the_first_and_last_are_always_kept(self, base):
        """A pruned model missing its first block does not produce
        slightly worse tokens; it produces tokens from a different
        distribution, and every proposal is rejected."""
        model = GGUFFile.read(base)
        for depth in (0.25, 0.4, 0.6, 0.9):
            plan = plan_draft(model, depth=depth)
            assert plan.layers[0] == 0
            assert plan.layers[-1] == 7

    def test_a_deeper_draft_keeps_more_layers(self, base):
        model = GGUFFile.read(base)
        assert len(plan_draft(model, depth=0.25).layers) <= len(
            plan_draft(model, depth=0.75).layers
        )

    def test_a_depth_of_one_keeps_every_layer(self, base):
        model = GGUFFile.read(base)
        assert plan_draft(model, depth=1.0).layers == tuple(range(8))

    def test_the_layers_are_distinct_and_in_order(self, base):
        model = GGUFFile.read(base)
        layers = plan_draft(model, depth=0.6).layers
        assert list(layers) == sorted(set(layers))

    def test_an_explicit_list_is_honoured(self, base):
        model = GGUFFile.read(base)
        assert plan_draft(model, layers=[5, 1, 1, 0]).layers == (0, 1, 5)

    def test_a_layer_the_model_does_not_have_is_refused(self, base):
        model = GGUFFile.read(base)
        with pytest.raises(Dflash2Error, match="outside this model"):
            plan_draft(model, layers=[0, 99])

    def test_an_unknown_quant_is_refused(self, base):
        model = GGUFFile.read(base)
        with pytest.raises(Dflash2Error, match="Unknown draft quantisation"):
            plan_draft(model, quant="Q9_ULTRA")

    def test_a_model_with_no_blocks_says_what_is_wrong(self, tmp_path):
        writer = GGUFWriter(tmp_path / "flat.gguf")
        writer.set_metadata("general.architecture", "llama")
        writer.add_tensor("token_embd.weight", (256, 4), int(GGMLType.F32))
        writer.write(lambda _t: struct.pack("<1024f", *([0.0] * 1024)))
        model = GGUFFile.read(tmp_path / "flat.gguf")
        with pytest.raises(Dflash2Error, match="nothing to draft from"):
            plan_draft(model)


class TestAttaching:
    def test_the_draft_tensors_are_namespaced(self, attached):
        names = [t.name for t in GGUFFile.read(attached).tensors]
        drafted = [n for n in names if n.startswith(PREFIX)]
        assert drafted
        assert f"{PREFIX}blk.0.attn_q.weight" in drafted

    def test_the_base_tensors_come_through_byte_for_byte(self, base, attached):
        """Attaching a draft must not change the model anyone was
        already running."""
        original = GGUFFile.read(base)
        combined = GGUFFile.read(attached)
        after = {t.name: t for t in combined.tensors}
        for tensor in original.tensors:
            assert tensor.name in after
            assert original.tensor_bytes(tensor) == combined.tensor_bytes(
                after[tensor.name]
            )

    def test_the_metadata_survives(self, attached):
        meta = GGUFFile.read(attached).metadata
        assert meta["general.architecture"] == "llama"
        assert meta["tokenizer.chat_template"] == "{{ messages }}"
        assert meta["llama.block_count"] == 8

    def test_the_draft_metadata_says_what_it_is(self, attached):
        info = read_draft_info(attached)
        assert info["present"] is True
        assert info["version"] == VERSION
        assert info["source_block_count"] == 8
        assert info["block_count"] == len(info["layer_map"])
        assert info["quant"] == "Q4_0"
        assert info["draft_tokens"] == 4

    def test_the_draft_is_actually_quantised(self, attached):
        """A draft the same size as the layers it came from is not a
        draft, it is a copy with a prefix."""
        model = GGUFFile.read(attached)
        drafted = next(
            t for t in model.tensors if t.name == f"{PREFIX}blk.0.attn_q.weight"
        )
        assert int(drafted.ggml_type) == int(GGMLType.Q4_0)
        original = next(t for t in model.tensors if t.name == "blk.0.attn_q.weight")
        assert drafted.nbytes < original.nbytes / 4

    def test_the_norms_are_copied_not_crushed(self, attached):
        model = GGUFFile.read(attached)
        norm = next(
            t for t in model.tensors if t.name == f"{PREFIX}blk.0.attn_norm.weight"
        )
        assert int(norm.ggml_type) == int(GGMLType.F32)

    def test_the_overhead_is_a_fraction_of_the_base(self, tmp_path, base):
        report = attach(base, tmp_path / "out.gguf", depth=0.25)
        assert 0 < report.overhead < 0.25, report.describe()

    def test_a_deeper_draft_costs_more(self, tmp_path, base):
        shallow = attach(base, tmp_path / "shallow.gguf", depth=0.25)
        deep = attach(base, tmp_path / "deep.gguf", depth=0.75)
        assert deep.draft_bytes > shallow.draft_bytes

    def test_a_narrower_quant_costs_less(self, tmp_path, base):
        wide = attach(base, tmp_path / "wide.gguf", depth=0.5, quant="Q8_0")
        narrow = attach(base, tmp_path / "narrow.gguf", depth=0.5, quant="Q2_K")
        assert narrow.draft_bytes < wide.draft_bytes

    def test_the_embeddings_are_shared_by_default(self, attached):
        """The draft has to speak the same vocabulary as the model it
        drafts for; a second copy of the embedding table would be the
        largest thing in the draft and identical to one already present."""
        info = read_draft_info(attached)
        assert "token_embd.weight" in info["shared"]
        names = [t.name for t in GGUFFile.read(attached).tensors]
        assert f"{PREFIX}token_embd.weight" not in names

    def test_attaching_twice_is_refused(self, tmp_path, attached):
        with pytest.raises(Dflash2Error, match="already carries"):
            attach(attached, tmp_path / "again.gguf")

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(Dflash2Error, match="No such model"):
            attach(tmp_path / "absent.gguf", tmp_path / "o.gguf")

    def test_progress_is_reported_per_tensor(self, tmp_path, base):
        events = []
        attach(base, tmp_path / "out.gguf", depth=0.5, progress=events.append)
        tensors = [e for e in events if e["event"] == "tensor"]
        assert tensors
        assert any(e["draft"] for e in tensors)
        assert any(not e["draft"] for e in tensors)
        assert events[-1]["event"] == "done"


class TestReadingItBack:
    def test_a_model_without_a_draft_says_so_rather_than_failing(self, base):
        """Most GGUFs have no draft. Asking is the normal way to find out."""
        assert read_draft_info(base) == {"present": False}
        assert has_draft(base) is False

    def test_a_model_with_one_says_so(self, attached):
        assert has_draft(attached) is True

    def test_metadata_claiming_a_draft_that_is_not_there_is_an_error(self, tmp_path, base):
        """Something rewrote the file and dropped the tensors. Better to
        say the file is inconsistent than to report a draft nothing can
        run."""
        model = GGUFFile.read(base)
        writer = GGUFWriter(tmp_path / "lying.gguf")
        writer.copy_metadata_from(model)
        writer.set_metadata("dflash2.present", True)
        for tensor in model.tensors:
            writer.add_tensor(tensor.name, tensor.shape, tensor.ggml_type)
        by_name = {t.name: t for t in model.tensors}
        writer.write(lambda d: model.tensor_bytes(by_name[d.name]))

        with pytest.raises(Dflash2Error, match="metadata is describing a draft"):
            read_draft_info(tmp_path / "lying.gguf")
        assert has_draft(tmp_path / "lying.gguf") is False


class TestExtracting:
    def test_the_draft_becomes_a_gguf_of_its_own(self, tmp_path, attached):
        result = extract(attached, tmp_path / "draft.gguf")
        assert Path(result["path"]).exists()
        draft = GGUFFile.read(tmp_path / "draft.gguf")
        names = [t.name for t in draft.tensors]
        assert "blk.0.attn_q.weight" in names
        assert not any(n.startswith(PREFIX) for n in names)

    def test_the_block_count_is_the_draft_s_own(self, tmp_path, attached):
        """A draft that claims the base's layer count describes tensors it
        does not have, and a loader fails on the first missing block
        rather than on the metadata."""
        info = read_draft_info(attached)
        extract(attached, tmp_path / "draft.gguf")
        meta = GGUFFile.read(tmp_path / "draft.gguf").metadata
        assert meta["llama.block_count"] == info["block_count"]

    def test_the_draft_carries_the_vocabulary_it_shares(self, tmp_path, attached):
        extract(attached, tmp_path / "draft.gguf")
        names = [t.name for t in GGUFFile.read(tmp_path / "draft.gguf").tensors]
        assert "token_embd.weight" in names
        assert "output.weight" in names

    def test_the_extracted_draft_carries_no_dflash2_metadata(self, tmp_path, attached):
        extract(attached, tmp_path / "draft.gguf")
        meta = GGUFFile.read(tmp_path / "draft.gguf").metadata
        assert not any(key.startswith("dflash2.") for key in meta)
        assert meta["hypernix.dflash2_draft"] is True

    def test_the_draft_weights_are_readable(self, tmp_path, attached):
        extract(attached, tmp_path / "draft.gguf")
        draft = GGUFFile.read(tmp_path / "draft.gguf")
        tensor = next(t for t in draft.tensors if t.name == "blk.0.attn_q.weight")
        values = lq.dequantize_array(draft.tensor_bytes(tensor), int(tensor.ggml_type))
        assert len(values) == 1024
        assert any(v != 0 for v in values)

    def test_a_model_without_a_draft_is_refused(self, tmp_path, base):
        with pytest.raises(Dflash2Error, match="no Dflash2 draft"):
            extract(base, tmp_path / "draft.gguf")


class TestStripping:
    def test_it_reproduces_the_original_exactly(self, tmp_path, base, attached):
        """The strongest statement available that attaching is
        non-destructive."""
        strip(attached, tmp_path / "stripped.gguf")
        assert (tmp_path / "stripped.gguf").read_bytes() == base.read_bytes()

    def test_stripping_a_model_without_a_draft_is_a_copy(self, tmp_path, base):
        strip(base, tmp_path / "same.gguf")
        assert (tmp_path / "same.gguf").read_bytes() == base.read_bytes()


class TestSpeculating:
    """The acceptance loop, against a target whose answers are known.

    The guarantee under test: the tokens produced are exactly those the
    target alone would have produced. A bad draft costs time, never
    correctness.
    """

    TRUTH = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    def _verify(self, context, proposal):
        """The target: token N of the continuation is always TRUTH[N]."""
        start = len(context) - 1
        needed = len(proposal) + 1
        return self.TRUTH[start:start + needed]

    def _greedy(self, prefix, count):
        start = len(prefix) - 1
        return self.TRUTH[start:start + count]

    def test_a_perfect_draft_gives_the_target_s_own_tokens(self):
        result = speculate(
            [0],
            propose=lambda context, k: self._greedy(context, k),
            verify=self._verify,
            draft_tokens=4,
            max_new_tokens=8,
        )
        assert result.tokens == self.TRUTH[:8]
        assert result.acceptance_rate == 1.0

    def test_a_useless_draft_gives_the_same_tokens(self):
        """The property that makes this safe. A draft proposing garbage
        must not change one token of the output."""
        result = speculate(
            [0],
            propose=lambda _context, k: [999] * k,
            verify=self._verify,
            draft_tokens=4,
            max_new_tokens=8,
        )
        assert result.tokens == self.TRUTH[:8]
        assert result.accepted == 0

    def test_a_draft_that_is_right_then_wrong_gives_the_same_tokens(self):
        def propose(context, k):
            good = self._greedy(context, k)
            return good[:2] + [777] * (k - 2)

        result = speculate(
            [0], propose=propose, verify=self._verify,
            draft_tokens=4, max_new_tokens=8,
        )
        assert result.tokens == self.TRUTH[:8]
        assert 0 < result.acceptance_rate < 1

    def test_a_useless_draft_costs_one_target_call_per_token(self):
        useless = speculate(
            [0], propose=lambda _c, k: [999] * k, verify=self._verify,
            draft_tokens=4, max_new_tokens=8,
        )
        assert useless.tokens_per_target_call == pytest.approx(1.0)

    def test_a_perfect_draft_gets_more_tokens_per_target_call(self):
        """The entire point. If this ratio is not above one, the draft
        bought nothing and cost its own runtime."""
        perfect = speculate(
            [0], propose=lambda c, k: self._greedy(c, k), verify=self._verify,
            draft_tokens=4, max_new_tokens=8,
        )
        assert perfect.tokens_per_target_call > 1.0
        assert perfect.target_calls < 8

    def test_it_stops_at_a_stop_token(self):
        result = speculate(
            [0], propose=lambda c, k: self._greedy(c, k), verify=self._verify,
            draft_tokens=4, max_new_tokens=20, stop=[13],
        )
        assert result.tokens == [10, 11, 12, 13]

    def test_it_never_exceeds_max_new_tokens(self):
        for limit in (1, 2, 3, 5, 7):
            result = speculate(
                [0], propose=lambda c, k: self._greedy(c, k), verify=self._verify,
                draft_tokens=4, max_new_tokens=limit,
            )
            assert len(result.tokens) == limit, limit

    def test_a_draft_that_returns_fewer_tokens_than_asked_still_works(self):
        result = speculate(
            [0], propose=lambda c, _k: self._greedy(c, 1), verify=self._verify,
            draft_tokens=4, max_new_tokens=6,
        )
        assert result.tokens == self.TRUTH[:6]

    def test_a_draft_that_returns_nothing_still_advances(self):
        """Otherwise an empty proposal is an infinite loop rather than a
        slow generation."""
        result = speculate(
            [0], propose=lambda _c, _k: [], verify=self._verify,
            draft_tokens=4, max_new_tokens=5,
        )
        assert result.tokens == self.TRUTH[:5]
        assert result.proposed == 0

    def test_a_verifier_that_returns_too_few_tokens_is_refused(self):
        with pytest.raises(Dflash2Error, match="one per position plus one more"):
            speculate(
                [0], propose=lambda _c, k: [1] * k, verify=lambda _c, _p: [1],
                draft_tokens=4, max_new_tokens=4,
            )

    def test_zero_draft_tokens_is_refused(self):
        with pytest.raises(Dflash2Error, match="at least 1"):
            speculate([0], propose=lambda _c, k: [], verify=self._verify, draft_tokens=0)

    def test_the_report_reads_as_a_decision(self):
        result = speculate(
            [0], propose=lambda c, k: self._greedy(c, k), verify=self._verify,
            draft_tokens=4, max_new_tokens=8,
        )
        text = result.describe()
        assert "per call" in text
        assert "accepted" in text
        assert result.to_dict()["acceptance_rate"] == 1.0


class TestTheRuntimeSeesIt:
    def test_describe_gguf_reports_the_draft(self, attached):
        from hypernix.models.ggufrun import describe_gguf

        info = describe_gguf(attached)
        assert info["dflash2"]["present"] is True
        assert info["dflash2"]["quant"] == "Q4_0"

    def test_describe_gguf_reports_its_absence(self, base):
        from hypernix.models.ggufrun import describe_gguf

        assert describe_gguf(base)["dflash2"] == {"present": False}

    def test_the_draft_can_be_handed_to_a_runtime_as_a_path(self, tmp_path, attached):
        """Every llama.cpp that speculates wants the draft as a second
        path. Carrying it inside the model means the person still only
        downloads one file; this is where it becomes two."""
        from hypernix.models.ggufrun import materialize_draft

        path = materialize_draft(attached, cache_dir=tmp_path / "cache")
        assert path is not None and path.exists()
        assert GGUFFile.read(path).metadata["hypernix.dflash2_draft"] is True

    def test_it_is_reused_rather_than_rewritten(self, tmp_path, attached):
        from hypernix.models.ggufrun import materialize_draft

        first = materialize_draft(attached, cache_dir=tmp_path / "cache")
        stamp = first.stat().st_mtime_ns
        second = materialize_draft(attached, cache_dir=tmp_path / "cache")
        assert second == first
        assert second.stat().st_mtime_ns == stamp

    def test_a_model_without_a_draft_materialises_nothing(self, tmp_path, base):
        from hypernix.models.ggufrun import materialize_draft

        assert materialize_draft(base, cache_dir=tmp_path / "cache") is None


class TestTheCommandLine:
    def _run(self, *argv):
        return subprocess.run(
            [sys.executable, "-m", "hypernix.quant.dflash2_cli", *argv],
            capture_output=True, text=True, timeout=300,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        )

    def test_attach_writes_a_file_and_reports_it(self, tmp_path, base):
        out = tmp_path / "out.gguf"
        result = self._run("attach", str(base), "-o", str(out), "--json")
        assert result.returncode == 0, result.stderr
        assert out.exists()
        report = json.loads(result.stdout)
        assert report["layers"][0] == 0
        assert report["quant"] == "Q4_0"

    def test_info_on_a_plain_model_says_there_is_none(self, base):
        result = self._run("info", str(base))
        assert result.returncode == 0
        assert "no Dflash2 draft" in result.stdout

    def test_the_round_trip_through_the_cli(self, tmp_path, base):
        attached = tmp_path / "attached.gguf"
        assert self._run("attach", str(base), "-o", str(attached), "-q").returncode == 0
        assert self._run("info", str(attached)).returncode == 0
        assert self._run(
            "extract", str(attached), "-o", str(tmp_path / "draft.gguf")
        ).returncode == 0
        assert self._run(
            "strip", str(attached), "-o", str(tmp_path / "stripped.gguf")
        ).returncode == 0
        assert (tmp_path / "stripped.gguf").read_bytes() == base.read_bytes()

    def test_explicit_layers_are_passed_through(self, tmp_path, base):
        out = tmp_path / "out.gguf"
        result = self._run(
            "attach", str(base), "-o", str(out), "--layers", "0,4,7", "--json"
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["layers"] == [0, 4, 7]

    def test_a_bad_layer_list_is_refused_before_anything_is_written(self, tmp_path, base):
        out = tmp_path / "out.gguf"
        result = self._run("attach", str(base), "-o", str(out), "--layers", "a,b")
        assert result.returncode == 2
        assert not out.exists()

    def test_no_command_prints_help(self):
        result = self._run()
        assert result.returncode == 2
        assert "attach" in result.stdout
