"""`generate` and `chat` reading a GGUF.

Both commands took a snapshot directory — config.json plus safetensors,
loaded through torch — so the one format this package spends most of its
time *producing* was the one format its own inference commands could not
read.

The interesting case is the routing. The sub-bit tiers hyprslug writes
use GGML type ids at 200 and above, which no llama.cpp knows; handing one
to a runtime that does not know them gets either a refusal in someone
else's words or, worse, tensors read as the wrong type. So they go to
HyperNix's own runtime instead, and the upstream types still go to
llama.cpp, which is better at them than anything here will be.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from hypernix.models.ggufrun import (
    GGUFRunError,
    HnxSession,
    describe_gguf,
    is_gguf,
    load_gguf,
)
from hypernix.quant.gguf import GGMLType, GGUFWriter
from hypernix.quant.hyprslug import quantize_gguf


def _plain_gguf(path: Path, *, name: str = "tiny") -> Path:
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.name", name)
    writer.add_tensor("blk.0.attn_q.weight", (512, 4), int(GGMLType.F32))
    values = [0.01] * (512 * 4)
    writer.write(lambda t: struct.pack(f"<{len(values)}f", *values))
    return path


class TestDetection:
    def test_a_gguf_is_recognised(self, tmp_path):
        assert is_gguf(_plain_gguf(tmp_path / "m.gguf"))

    def test_it_reads_the_magic_not_the_extension(self, tmp_path):
        """A file named .gguf that is not one should fail as 'not a GGUF',
        not somewhere deep inside a loader."""
        impostor = tmp_path / "lies.gguf"
        impostor.write_bytes(b"I am a text file")
        assert not is_gguf(impostor)

        real = _plain_gguf(tmp_path / "no-extension")
        assert is_gguf(real)

    def test_a_directory_is_not_a_gguf(self, tmp_path):
        assert not is_gguf(tmp_path)

    def test_a_missing_path_is_not_a_gguf(self, tmp_path):
        assert not is_gguf(tmp_path / "absent.gguf")


class TestDescribing:
    def test_it_reports_the_architecture_and_name(self, tmp_path):
        info = describe_gguf(_plain_gguf(tmp_path / "m.gguf", name="tiny-test"))
        assert info["architecture"] == "llama"
        assert info["name"] == "tiny-test"
        assert info["sub_bit"] is False

    def test_it_reports_a_sub_bit_tier(self, tmp_path):
        source = _plain_gguf(tmp_path / "src.gguf")
        out = tmp_path / "sub.gguf"
        quantize_gguf(source, out, "IQ0.5_XXXL")

        info = describe_gguf(out)
        assert info["sub_bit"] is True
        assert info["tier"] == "IQ0.5_XXXL"
        assert info["quantiser"] == "hyprslug"

    def test_a_non_gguf_is_refused_by_name(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"nope")
        with pytest.raises(GGUFRunError, match="GGUF magic"):
            describe_gguf(bad)


class TestTheSubBitRouting:
    """Type 200+ has nowhere to go in llama.cpp, and somewhere here."""

    @pytest.mark.parametrize("tier", ["IQ0.9_L", "IQ0.75_M", "IQ0.5_XXXL"])
    def test_a_sub_bit_model_is_not_sent_to_llama_cpp(self, tmp_path, tier, monkeypatch):
        """The routing decision, checked without needing a whole model.

        multilama would either refuse in someone else's words or read a
        0.5-bit tensor as Q4_K and return noise, so it must not be
        reached at all.
        """
        from hypernix.models import multilama

        def _never(*_args, **_kwargs):
            raise AssertionError("a sub-bit GGUF was handed to llama.cpp")

        monkeypatch.setattr(multilama, "load", _never)

        source = _plain_gguf(tmp_path / "src.gguf")
        out = tmp_path / f"{tier}.gguf"
        quantize_gguf(source, out, tier)

        # This fragment is not a runnable model -- it has one tensor --
        # so loading fails, but on the *model's* terms rather than on
        # "no llama.cpp can read this".
        with pytest.raises(GGUFRunError) as caught:
            load_gguf(out)
        assert "token_embd" in str(caught.value)

    def test_an_upstream_quant_still_goes_to_llama_cpp(self, tmp_path, monkeypatch):
        """The other half of the routing. llama.cpp is better at Q4_K_M
        than a reference implementation in torch will ever be."""
        from hypernix.models import multilama

        seen = {}

        def _record(*args, **kwargs):
            seen["called"] = True
            raise multilama.MultiLlamaError("ML-TEST", "stopped here on purpose")

        monkeypatch.setattr(multilama, "load", _record)

        source = _plain_gguf(tmp_path / "src.gguf")
        out = tmp_path / "q4.gguf"
        quantize_gguf(source, out, "Q4_0")

        with pytest.raises(GGUFRunError):
            load_gguf(out)
        assert seen.get("called"), "an upstream quant did not reach llama.cpp"

    def test_describe_still_names_the_tier(self, tmp_path):
        source = _plain_gguf(tmp_path / "src.gguf")
        out = tmp_path / "sub.gguf"
        quantize_gguf(source, out, "IQ0.5_XXXL")
        info = describe_gguf(out)
        assert info["sub_bit"] is True
        assert info["tier"] == "IQ0.5_XXXL"


class TestLoadingRefusals:
    def test_a_missing_model_says_so(self, tmp_path):
        with pytest.raises(GGUFRunError, match="No such model"):
            load_gguf(tmp_path / "absent.gguf")

    def test_a_non_gguf_says_so(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"not a model at all")
        with pytest.raises(GGUFRunError, match="not a GGUF"):
            load_gguf(bad)


@pytest.fixture(scope="module")
def runnable_sub_bit(tmp_path_factory):
    """A whole sub-bit model, with a tokenizer, that really runs.

    The fragments elsewhere in this file are enough to test *routing*.
    They are not enough to test that the route arrives anywhere, which is
    a different question and the one that was being missed.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_hnxrun import _write_model

    directory = tmp_path_factory.mktemp("ggufrun-runnable")
    source = _write_model(directory / "tiny.f32.gguf", tokenizer=True)
    out = directory / "tiny.IQ0.5_XXXL.gguf"
    quantize_gguf(
        source, out, "IQ0.5_XXXL", quantize_embeddings=True, quantize_output=True
    )
    return out


class TestASubBitModelIsActuallyUsable:
    """Routing to a runtime is not the same as being able to run.

    ``load_gguf`` handed back a bare ``LoadedModel`` for these files, and
    a ``LoadedModel`` has no ``.chat()``. So ``hypernix chat`` on a
    0.5-bit model loaded successfully, printed nothing, and died with
    ``AttributeError: 'LoadedModel' object has no attribute 'chat'`` on
    the first message. Every test that existed passed: they checked that
    the CLI *mentions* ``load_gguf`` and that the routing does not reach
    llama.cpp, and both were true of the broken version.

    So these run the thing.
    """

    def test_load_gguf_returns_something_that_can_chat(self, runnable_sub_bit):
        session = load_gguf(runnable_sub_bit)
        assert isinstance(session, HnxSession)
        assert callable(session.chat)
        reply = session.chat([{"role": "user", "content": "hello"}], max_tokens=4)
        assert isinstance(reply, str)

    def test_the_loaded_model_is_still_reachable(self, runnable_sub_bit):
        """The session is a wrapper, not a wall: describe(), the resident
        cost, and the tensors are what someone loads one of these to look
        at."""
        session = load_gguf(runnable_sub_bit)
        assert session.model.resident_bits_per_weight < 1.0
        assert "bits/weight" in session.describe()

    def test_the_budget_reaches_the_runtime(self, runnable_sub_bit):
        """``--cache-bytes`` is the memory-for-speed dial, and a flag that
        is accepted and dropped on the floor is worse than no flag."""
        packed = load_gguf(runnable_sub_bit)
        budgeted = load_gguf(runnable_sub_bit, cache_bytes=1 << 20)
        assert packed.model.pinned_in_memory == 0
        assert budgeted.model.pinned_in_memory > 0
        assert (
            budgeted.model.resident_bits_per_weight
            > packed.model.resident_bits_per_weight
        )

    def test_the_budget_does_not_change_the_answer(self, runnable_sub_bit):
        """Greedy, because the dial is a memory-for-time trade and would
        be worthless if its two ends disagreed about the model."""
        messages = [{"role": "user", "content": "hello"}]
        packed = load_gguf(runnable_sub_bit).chat(
            messages, max_tokens=6, temperature=0.0
        )
        budgeted = load_gguf(runnable_sub_bit, cache_bytes=1 << 20).chat(
            messages, max_tokens=6, temperature=0.0
        )
        assert packed == budgeted

    def test_an_upstream_quant_does_not_get_a_budget_it_cannot_use(self, tmp_path):
        """llama.cpp has its own answer to how much to keep resident, and
        passing this one through would be guessing on its behalf."""
        import inspect

        from hypernix.models import ggufrun

        source = inspect.getsource(ggufrun.load_gguf)
        after = source.split("multilama.load(")[1]
        assert "cache_bytes" not in after


class TestTheCLIRunsItEndToEnd:
    """Source-text assertions caught the routing and missed the crash, so
    these call ``main`` and read what comes out."""

    def _run(self, capsys, argv):
        from hypernix.interfaces import cli

        code = cli.main(list(argv))
        return code, capsys.readouterr().out

    def test_generate_produces_text(self, capsys, runnable_sub_bit):
        code, out = self._run(
            capsys,
            ["generate", "--model-dir", str(runnable_sub_bit),
             "--prompt", "hi", "--max-new-tokens", "3"],
        )
        assert code == 0
        assert out.strip()

    def test_chat_produces_a_turn(self, capsys, runnable_sub_bit):
        """The one that was broken."""
        code, out = self._run(
            capsys,
            ["chat", "--model-dir", str(runnable_sub_bit),
             "--message", "hi", "--max-new-tokens", "3"],
        )
        assert code == 0
        assert out.strip()

    @pytest.mark.parametrize("command", ["generate", "chat"])
    def test_both_take_a_cache_budget(self, capsys, runnable_sub_bit, command):
        argv = [command, "--model-dir", str(runnable_sub_bit),
                "--max-new-tokens", "3", "--cache-bytes", "1M"]
        argv += ["--prompt", "hi"] if command == "generate" else ["--message", "hi"]
        code, out = self._run(capsys, argv)
        assert code == 0
        assert out.strip()

    def test_chat_loads_once_for_two_turns(self, monkeypatch, runnable_sub_bit):
        """The REPL's whole reason for holding the model open. A sub-bit
        load is the expensive one, so reloading per turn would be worst
        exactly where it matters."""
        from hypernix.models import hnxrun

        loads = []
        real = hnxrun.load_model

        def _counting(*args, **kwargs):
            loads.append(args[0] if args else kwargs.get("path"))
            return real(*args, **kwargs)

        monkeypatch.setattr(hnxrun, "load_model", _counting)

        session = load_gguf(runnable_sub_bit)
        session.chat([{"role": "user", "content": "one"}], max_tokens=2)
        session.chat([{"role": "user", "content": "two"}], max_tokens=2)
        assert len(loads) == 1, f"loaded {len(loads)} times for two turns"


class TestTheSizeFlagParses:
    """``--cache-bytes 2000000000`` is a number nobody types correctly."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0", 0),
            ("1024", 1024),
            ("1K", 1 << 10),
            ("512M", 512 << 20),
            ("2G", 2 << 30),
            ("2GB", 2 << 30),
            ("2GiB", 2 << 30),
            ("1.5G", int(1.5 * (1 << 30))),
            ("  4g  ", 4 << 30),
        ],
    )
    def test_it_reads_human_sizes(self, text, expected):
        from hypernix.interfaces.cli import _parse_size

        assert _parse_size(text) == expected

    @pytest.mark.parametrize("text", ["", "lots", "-1G", "G", "1Q"])
    def test_a_size_it_cannot_read_is_refused(self, text):
        """Rather than silently becoming zero, which is a memory limit
        that does not hold and looks like the tool ignoring the flag."""
        import argparse

        from hypernix.interfaces.cli import _parse_size

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_size(text)


class TestTheCLIRoutesToIt:
    def test_generate_checks_for_a_gguf(self):
        source = Path("src/hypernix/interfaces/cli.py").read_text(encoding="utf-8")
        generate = source.split("def _run_generate(")[1].split("\ndef ")[0]
        assert "is_gguf" in generate
        assert "generate_with_gguf" in generate

    def test_chat_checks_for_a_gguf(self):
        source = Path("src/hypernix/interfaces/cli.py").read_text(encoding="utf-8")
        chat = source.split("def _run_chat(")[1].split("\ndef ")[0]
        assert "is_gguf" in chat
        assert "load_gguf" in chat

    def test_chat_loads_the_model_once_not_per_turn(self):
        """A GGUF load is seconds to minutes; paying it per message would
        make the REPL unusable."""
        source = Path("src/hypernix/interfaces/cli.py").read_text(encoding="utf-8")
        chat = source.split("def _run_chat(")[1].split("\ndef ")[0]
        turn = chat.split("def turn(")[1]
        assert "load_gguf" not in turn, "the model is reloaded on every turn"

    def test_both_help_texts_mention_gguf(self):
        source = Path("src/hypernix/interfaces/cli.py").read_text(encoding="utf-8")
        for name in ("_run_generate", "_run_chat"):
            body = source.split(f"def {name}(")[1].split("\ndef ")[0]
            assert ".gguf" in body, name
