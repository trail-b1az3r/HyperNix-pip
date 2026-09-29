"""``hypernix hyprslug-headers`` — and the two bugs it found on the way.

The request behind this module was "headers that let any GGUF load".
That is not a thing a header can do, and the first job of the tests is to
pin down what it *does* do, so the promise stays the one that is true:

* **stamp** makes the file describe itself. It does not make it loadable.
* **wrap** re-encodes to a type stock llama.cpp has. The result loads
  everywhere and is not sub-bit any more.
* **serve** keeps the tier and moves the boundary to HTTP.

Two real bugs turned up while wiring it, and both have tests here because
both were invisible from the outside:

``wrap`` reported success on a file it had not converted.
    ``_readable()`` did not list the extension types, so
    ``_should_quantize`` declined every tensor with "source type 200 is
    one hyprslug cannot read" and copied it verbatim. The output was a
    Q2_K-labelled file still full of type-200 tensors — refused by
    exactly the loader the command exists to satisfy.

``stamp`` corrupted ``general.alignment``.
    Copying metadata with ``set_metadata`` re-infers a GGUF type per
    value, and nothing about the number ``32`` says UINT32 rather than
    INT32. The reference reader rejected the result with "Bad type for
    general.alignment field": a file this package could still read and
    nothing else could, which is the worst of both.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant import hyprslug_headers as headers
from hypernix.quant.gguf import GGMLType, GGUFFile, GGUFWriter
from hypernix.quant.hyprslug import TIER_TYPES, quantize_gguf


def _wide_model(path: Path, *, vocab: int = 512) -> Path:
    """A llama model whose rows divide a 256-element block.

    The toy model the rest of the suite uses is 64 wide, which no K-quant
    can encode — so a wrap of it is correct and still unreadable by
    anything, for a reason that has nothing to do with wrapping. This one
    is shaped like a real model so "does the output open" is a question
    about the output.
    """
    from hypernix.models.hnxtokenizer import _BYTE_ENCODER

    n_layer, n_embd, n_head, n_kv, n_ff = 2, 256, 4, 2, 512
    head_dim = n_embd // n_head
    rng = np.random.default_rng(0)
    writer = GGUFWriter(path)
    for key, value in [
        ("general.architecture", "llama"),
        ("llama.block_count", n_layer),
        ("llama.embedding_length", n_embd),
        ("llama.attention.head_count", n_head),
        ("llama.attention.head_count_kv", n_kv),
        ("llama.feed_forward_length", n_ff),
        ("llama.attention.layer_norm_rms_epsilon", 1e-5),
        ("llama.rope.freq_base", 10000.0),
    ]:
        writer.set_metadata(key, value)
    tokens = [chr(256 + i) for i in range(vocab)]
    for byte in range(256):
        tokens[byte] = _BYTE_ENCODER[byte]
    writer.set_metadata("tokenizer.ggml.model", "gpt2")
    writer.set_metadata("tokenizer.ggml.tokens", tokens)
    writer.set_metadata("tokenizer.ggml.merges", [])

    shapes = {
        "token_embd.weight": (n_embd, vocab),
        "output_norm.weight": (n_embd,),
        "output.weight": (n_embd, vocab),
    }
    for index in range(n_layer):
        shapes[f"blk.{index}.attn_norm.weight"] = (n_embd,)
        shapes[f"blk.{index}.ffn_norm.weight"] = (n_embd,)
        shapes[f"blk.{index}.attn_q.weight"] = (n_embd, n_head * head_dim)
        shapes[f"blk.{index}.attn_k.weight"] = (n_embd, n_kv * head_dim)
        shapes[f"blk.{index}.attn_v.weight"] = (n_embd, n_kv * head_dim)
        shapes[f"blk.{index}.attn_output.weight"] = (n_head * head_dim, n_embd)
        shapes[f"blk.{index}.ffn_gate.weight"] = (n_embd, n_ff)
        shapes[f"blk.{index}.ffn_up.weight"] = (n_embd, n_ff)
        shapes[f"blk.{index}.ffn_down.weight"] = (n_ff, n_embd)

    payload = {}
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        values = (
            np.ones(count, np.float32)
            if name.endswith("norm.weight")
            else rng.normal(0.0, 0.05, count).astype(np.float32)
        )
        writer.add_tensor(name, shape, int(GGMLType.F32))
        payload[name] = values.tobytes()
    writer.write(lambda tensor: payload[tensor.name])
    return path


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _wide_model(tmp_path_factory.mktemp("headers") / "wide.f32.gguf")


@pytest.fixture(scope="module")
def sub_bit(source):
    out = Path(source).parent / "wide.IQ0.9_L.gguf"
    quantize_gguf(
        source, out, "IQ0.9_L", quantize_embeddings=True, quantize_output=True
    )
    return out


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A runtime directory and an LM Studio tree of our own."""
    monkeypatch.setenv("HYPRSLUG_HEADERS_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "lms"))
    (tmp_path / "lms" / "models").mkdir(parents=True)
    return tmp_path


class TestTheHeaderDescribesTheCodec:
    """A header that disagreed with the packer would be worse than no
    header, because it would be believed."""

    @pytest.mark.parametrize("tier", sorted(TIER_TYPES))
    def test_it_is_read_out_of_the_packing_tables(self, tier):
        from hypernix.quant.lowbit import CODECS
        from hypernix.quant.subbit import PACKINGS

        ggml_type, packing = TIER_TYPES[tier]
        header = headers.header_for_type(ggml_type)
        assert header.tier == tier
        assert header.packing == packing
        assert header.block_elements == 256
        if packing in PACKINGS:
            spec = PACKINGS[packing]
            assert header.family == "sign-and-scale"
            assert (header.kept, header.group) == (spec.kept, spec.group)
            assert header.block_bytes == spec.block_bytes
        else:
            codec = CODECS[packing]
            assert header.family == "fixed-codebook"
            assert header.levels == list(codec.levels)
            assert header.block_bytes == codec.block_bytes

    @pytest.mark.parametrize("tier", sorted(TIER_TYPES))
    def test_every_tier_names_a_fallback_that_exists(self, tier):
        """``wrap`` uses it as a default, so an unwritable name here is a
        command that fails on the tier nobody tested."""
        from hypernix.quant.hyprslug import resolve_recipe

        header = headers.header_for_type(TIER_TYPES[tier][0])
        assert resolve_recipe(header.fallback) is not None, header.fallback

    def test_an_ordinary_type_is_not_an_extension(self):
        header = headers.header_for_type(int(GGMLType.Q4_K))
        assert not header.is_extension
        assert "no HyperNix extension types" in header.describe()

    def test_the_geometry_is_enough_to_write_a_decoder(self, sub_bit):
        """The point of stamping arithmetic rather than a codec name: a
        reader without this package installed still knows what it holds."""
        header = headers.read_header(sub_bit)
        assert header.block_elements * header.bits_per_weight / 8 == (
            header.block_bytes
        )
        assert header.scale_dtype == "f16"
        assert header.kept and header.group


class TestStampingDoesNotBreakTheFile:
    def test_the_header_survives_a_round_trip(self, sub_bit, tmp_path):
        stamped = tmp_path / "stamped.gguf"
        written = headers.stamp(sub_bit, stamped)
        read_back = headers.read_header(stamped)
        assert read_back.tier == written.tier == "IQ0.9_L"
        assert (read_back.kept, read_back.group) == (7, 8)
        assert read_back.version == headers.HEADER_VERSION

    def test_the_model_still_loads_and_runs(self, sub_bit, tmp_path):
        """Stamping rewrites the file — every tensor offset moves. A
        stamp that shifted one would give a plausible, wrong model."""
        from hypernix.models import hnxrun

        stamped = tmp_path / "stamped.gguf"
        headers.stamp(sub_bit, stamped)
        before, _ = hnxrun.forward(hnxrun.load_model(sub_bit), [1, 5, 9])
        after, _ = hnxrun.forward(hnxrun.load_model(stamped), [1, 5, 9])
        import torch

        assert torch.equal(before, after)

    def test_it_preserves_the_metadata_types(self, sub_bit, tmp_path):
        """The bug: copying with set_metadata re-infers a GGUF type per
        value, and ``32`` does not say whether it is UINT32 or INT32.
        general.alignment came back INT32 and the reference reader
        rejected the file outright."""
        stamped = tmp_path / "stamped.gguf"
        headers.stamp(sub_bit, stamped)
        original = GGUFFile.read(sub_bit)
        after = GGUFFile.read(stamped)
        for key, kind in original.metadata_types.items():
            assert after.metadata_types.get(key) == kind, key

    def test_it_keeps_every_key_it_did_not_write(self, sub_bit, tmp_path):
        """Dropping keys a reader does not recognise is how a round trip
        silently strips a chat template or a rope scaling factor."""
        stamped = tmp_path / "stamped.gguf"
        headers.stamp(sub_bit, stamped)
        original = GGUFFile.read(sub_bit).metadata
        after = GGUFFile.read(stamped).metadata
        assert set(original) <= set(after)
        for key, value in original.items():
            assert after[key] == value, key

    def test_stamping_twice_does_not_accumulate(self, sub_bit, tmp_path):
        once = tmp_path / "once.gguf"
        headers.stamp(sub_bit, once)
        first = len(GGUFFile.read(once).metadata)
        headers.stamp(once)
        assert len(GGUFFile.read(once).metadata) == first

    def test_an_ordinary_gguf_is_refused(self, source, tmp_path):
        """Stamping it would add keys saying 'this is an ordinary GGUF',
        which it already says by being one."""
        with pytest.raises(headers.HeaderError, match="nothing to describe"):
            headers.stamp(source, tmp_path / "no.gguf")


class TestWrappingActuallyConverts:
    """The bug this class exists for: ``wrap`` returned a success report
    for a file it had copied rather than converted."""

    @pytest.mark.parametrize("target", ["Q2_K", "Q4_K_M", "Q8_0"])
    def test_no_extension_type_survives(self, sub_bit, tmp_path, target):
        out = tmp_path / f"wrap.{target}.gguf"
        headers.wrap(sub_bit, out, to=target)
        model = GGUFFile.read(out)
        assert model.tensors
        assert all(int(t.ggml_type) < 200 for t in model.tensors), (
            f"{target} wrap left an extension type behind"
        )

    @pytest.mark.parametrize("target", ["Q2_K", "Q4_K_M", "Q8_0"])
    def test_the_reference_reader_opens_the_result(self, sub_bit, tmp_path, target):
        """The whole promise, checked against a reader that is not ours.
        ``gguf.GGUFReader`` is what LM Studio's loader agrees with about
        types, so it opening the file is the closest thing to the real
        test that runs without LM Studio."""
        gguf = pytest.importorskip("gguf")

        out = tmp_path / f"wrap.{target}.gguf"
        headers.wrap(sub_bit, out, to=target)
        reader = gguf.GGUFReader(str(out))
        assert len(reader.tensors) > 0

    def test_the_original_is_still_refused(self, sub_bit):
        """Wrapping must not be reachable by making the source loadable —
        the type ids stay outside upstream's range on purpose."""
        gguf = pytest.importorskip("gguf")

        with pytest.raises(Exception):  # noqa: B017, PT011
            gguf.GGUFReader(str(sub_bit))

    def test_the_report_says_it_got_bigger(self, sub_bit, tmp_path):
        """A compatibility export presented as the original is the lie
        this package exists to stop telling."""
        result = headers.wrap(sub_bit, tmp_path / "w.gguf", to="Q4_K_M")
        assert result["growth"] > 1.0
        assert result["from_tier"] == "IQ0.9_L"
        assert result["to_type"] == "Q4_K_M"
        assert "not a IQ0.9_L model" in result["honest_warning"]

    def test_it_uses_the_tier_default_when_none_is_given(self, sub_bit, tmp_path):
        result = headers.wrap(sub_bit, tmp_path / "d.gguf")
        assert result["to_type"] == headers.FALLBACKS["IQ0.9_L"]

    def test_wrapping_an_ordinary_gguf_is_refused(self, source, tmp_path):
        with pytest.raises(headers.HeaderError, match="already a type"):
            headers.wrap(source, tmp_path / "no.gguf")

    def test_a_wrap_that_left_a_type_behind_removes_its_output(
        self, sub_bit, tmp_path, monkeypatch
    ):
        """The guard, exercised. Simulating the old bug has to leave no
        file: a half-converted GGUF on disk is one somebody loads."""
        out = tmp_path / "bad.gguf"
        monkeypatch.setattr(headers, "_extension_type", lambda _p: 200)
        with pytest.raises(headers.HeaderError, match="still be refused"):
            headers.wrap(sub_bit, out, to="Q4_K_M")
        assert not out.exists()


class TestScanningAndInstalling:
    def test_it_tells_the_two_kinds_apart(self, isolated, sub_bit, source):
        import shutil

        models = isolated / "lms" / "models"
        shutil.copy(sub_bit, models / "sub.gguf")
        upstream = models / "up.gguf"
        quantize_gguf(source, upstream, "Q4_K_M")

        rows = {Path(r["path"]).name: r for r in headers.scan(models)}
        assert rows["sub.gguf"]["extension"] is True
        assert rows["sub.gguf"]["stock_llama_cpp"] is False
        assert rows["sub.gguf"]["tier"] == "IQ0.9_L"
        assert rows["up.gguf"]["extension"] is False
        assert rows["up.gguf"]["stock_llama_cpp"] is True

    def test_an_unreadable_file_is_reported_not_raised(self, isolated):
        """One corrupt file in a model directory must not end the scan;
        the whole point is to find out which of many is the problem."""
        bad = isolated / "lms" / "models" / "broken.gguf"
        bad.write_bytes(b"not a gguf at all")
        rows = headers.scan(isolated / "lms" / "models")
        assert len(rows) == 1
        assert rows[0]["readable"] is False
        assert rows[0]["error"]

    def test_install_writes_a_config_and_finds_the_models(self, isolated, sub_bit):
        import shutil

        shutil.copy(sub_bit, isolated / "lms" / "models" / "sub.gguf")
        result = headers.install()
        config = Path(result["config"])
        assert config.is_file()
        payload = json.loads(config.read_text())
        assert payload["header_version"] == headers.HEADER_VERSION
        assert set(payload["types"]) == set(TIER_TYPES)
        assert len(result["models_needing_the_runtime"]) == 1

    def test_status_before_and_after(self, isolated):
        assert headers.status()["installed"] is False
        headers.install(scan_lmstudio=False)
        after = headers.status()
        assert after["installed"] is True
        assert set(after["types"]) == set(TIER_TYPES)

    def test_uninstall_removes_the_config_and_nothing_else(self, isolated, sub_bit):
        import shutil

        model = isolated / "lms" / "models" / "sub.gguf"
        shutil.copy(sub_bit, model)
        headers.install()
        headers.uninstall()
        assert headers.status()["installed"] is False
        assert model.is_file(), "uninstall touched a model"


class TestInstallingIntoLMStudio:
    """LM Studio and Bionic read the same tree and list whatever their
    llama.cpp can open. Which is the constraint: a sub-bit model cannot
    go there as it stands, so what gets installed is a wrap of it."""

    def test_it_lands_where_lm_studio_looks(self, isolated, sub_bit):
        result = headers.install_model(sub_bit, name="Tiny", publisher="Acme")
        installed = Path(result["installed_to"])
        assert installed.is_file()
        assert installed.parent.name == "Tiny"
        assert installed.parent.parent.name == "Acme"
        assert installed.name == "Tiny.gguf"

    def test_what_it_installs_actually_opens(self, isolated, sub_bit):
        """The point of the exercise. If the installed file is refused
        too, nothing has been achieved but a copy."""
        gguf = pytest.importorskip("gguf")

        result = headers.install_model(sub_bit, to="Q4_K_M", name="Tiny")
        reader = gguf.GGUFReader(result["installed_to"])
        assert len(reader.tensors) > 0

    def test_it_says_the_installed_copy_is_not_the_model(self, isolated, sub_bit):
        """"Installed into LM Studio" is exactly the phrase under which
        someone assumes the 0.9-bit file now works there."""
        result = headers.install_model(sub_bit, to="Q4_K_M", name="Tiny")
        assert result["converted"] is True
        assert result["growth"] > 1.0
        assert "not a IQ0.9_L model" in result["honest_warning"]

    def test_the_original_is_left_alone(self, isolated, sub_bit):
        before = Path(sub_bit).read_bytes()
        headers.install_model(sub_bit, name="Tiny")
        assert Path(sub_bit).read_bytes() == before

    def test_an_upstream_model_is_copied_not_requantised(self, isolated, source):
        """It already opens. Re-encoding would lose a generation of
        quality for nothing."""
        upstream = Path(source).parent / "already-fine.gguf"
        quantize_gguf(source, upstream, "Q4_K_M")
        result = headers.install_model(upstream, name="Fine")
        assert result["converted"] is False
        assert result["growth"] == 1.0
        assert Path(result["installed_to"]).stat().st_size == upstream.stat().st_size

    def test_a_missing_model_says_so(self, isolated, tmp_path):
        with pytest.raises(headers.HeaderError, match="No such model"):
            headers.install_model(tmp_path / "absent.gguf")

    def test_no_lmstudio_directory_names_the_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_HOME", str(tmp_path / "nowhere"))
        with pytest.raises(headers.HeaderError, match="lmstudio"):
            headers.install_model(tmp_path / "x.gguf", root=None)

    def test_from_the_command_line(self, isolated, sub_bit):
        import contextlib
        import io

        from hypernix.quant.hyprslug_headers_cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["install-model", str(sub_bit), "--name", "Tiny"])
        text = out.getvalue()
        assert code == 0
        assert "installed" in text
        # It must also point at the way to keep the tier.
        assert "serve" in text


class TestTheServer:
    """The mechanism that keeps the tier: LM Studio talks HTTP, the
    model stays sub-bit."""

    @pytest.fixture(scope="class")
    def running(self, sub_bit):
        from hypernix.quant.hyprslug_server import HyprslugModel, build_server

        model = HyprslugModel(sub_bit, name="under-test")
        server = build_server(model, host="127.0.0.1", port=0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{port}"
        server.shutdown()
        server.server_close()

    def _get(self, base, path):
        with urllib.request.urlopen(f"{base}{path}", timeout=60) as response:
            return json.loads(response.read())

    def _post(self, base, path, payload):
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def test_health(self, running):
        assert self._get(running, "/health")["status"] == "ok"

    def test_models_reports_the_tier_and_the_resident_cost(self, running):
        data = self._get(running, "/v1/models")["data"][0]
        assert data["id"] == "under-test"
        assert data["hypernix"]["tier"] == "IQ0.9_L"
        # The number that says the tier survived being served.
        assert data["hypernix"]["resident_bits_per_weight"] < 1.5

    def test_a_chat_turn(self, running):
        reply = self._post(running, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        })
        assert reply["object"] == "chat.completion"
        assert reply["choices"][0]["message"]["role"] == "assistant"
        assert reply["choices"][0]["finish_reason"] == "stop"

    def test_a_completion(self, running):
        reply = self._post(
            running, "/v1/completions", {"prompt": "hello", "max_tokens": 4}
        )
        assert reply["object"] == "text_completion"
        assert isinstance(reply["choices"][0]["text"], str)

    def test_the_openai_content_parts_shape(self, running):
        """What a newer client sends. Refusing it would look like the
        server being broken rather than being old."""
        reply = self._post(running, "/v1/chat/completions", {
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }],
            "max_tokens": 4,
        })
        assert reply["choices"][0]["message"]["content"] is not None

    @pytest.mark.parametrize(
        ("path", "payload", "fragment"),
        [
            ("/v1/chat/completions", {"messages": []}, "non-empty"),
            ("/v1/chat/completions", {"messages": [{"content": "  "}]}, "empty"),
            ("/v1/completions", {"prompt": ""}, "empty"),
            (
                "/v1/chat/completions",
                {"messages": [{"content": "hi"}], "stream": True},
                "does not stream",
            ),
        ],
    )
    def test_a_bad_request_is_a_400_with_a_reason(
        self, running, path, payload, fragment
    ):
        with pytest.raises(urllib.error.HTTPError) as caught:
            self._post(running, path, payload)
        assert caught.value.code == 400
        assert fragment in json.loads(caught.value.read())["error"]["message"]

    def test_an_unknown_route_is_a_404(self, running):
        with pytest.raises(urllib.error.HTTPError) as caught:
            self._get(running, "/v1/embeddings")
        assert caught.value.code == 404

    def test_usage_is_omitted_rather_than_invented(self, running):
        """A wrong token count silently corrupts whatever is budgeting
        on it, and this runtime cannot produce one that matches anyone
        else's tokenizer."""
        reply = self._post(running, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2,
        })
        assert reply["usage"] == {}

    def test_a_missing_model_is_refused_at_load(self, tmp_path):
        from hypernix.quant.hyprslug_server import HyprslugModel, ServerError

        with pytest.raises(ServerError, match="No such model"):
            HyprslugModel(tmp_path / "absent.gguf")


class TestTheCommandLine:
    def _run(self, *argv):
        import contextlib
        import io

        from hypernix.quant.hyprslug_headers_cli import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue()

    def test_bare_invocation_prints_help_rather_than_failing(self):
        code, text = self._run()
        assert code == 0
        assert "serve" in text and "wrap" in text and "stamp" in text

    def test_the_help_says_what_a_header_cannot_do(self):
        """Someone arrives here because a model would not open. Letting
        them believe a header will fix that wastes an afternoon."""
        import contextlib
        import io

        from hypernix.quant.hyprslug_headers_cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exit_:
            main(["--help"])
        assert exit_.value.code == 0
        text = out.getvalue()
        assert "dequantisation kernel" in text
        assert "keep the tier" in text and "open it anywhere" in text

    @pytest.mark.parametrize(
        "command", [["status"], ["scan"], ["show", "MODEL"]]
    )
    def test_json_output_parses(self, isolated, sub_bit, command, monkeypatch):
        import shutil

        shutil.copy(sub_bit, isolated / "lms" / "models" / "sub.gguf")
        argv = ["--json", *[
            str(sub_bit) if part == "MODEL" else part for part in command
        ]]
        _code, text = self._run(*argv)
        assert json.loads(text) is not None

    def test_status_exits_nonzero_when_not_installed(self, isolated):
        """A script asking "is this set up" needs an answer in the exit
        code, not only in the text."""
        code, _text = self._run("status")
        assert code == 1
        headers.install(scan_lmstudio=False)
        code, _text = self._run("status")
        assert code == 0

    def test_install_then_scan(self, isolated, sub_bit):
        import shutil

        shutil.copy(sub_bit, isolated / "lms" / "models" / "sub.gguf")
        code, text = self._run("install")
        assert code == 0
        assert "need this runtime" in text
        code, text = self._run("scan")
        assert code == 0
        assert "needs hnxrun" in text

    def test_wrap_from_the_command_line(self, isolated, sub_bit, tmp_path):
        out = tmp_path / "wrapped.gguf"
        code, text = self._run("wrap", str(sub_bit), "-o", str(out), "--to", "Q4_K_M")
        assert code == 0
        assert out.is_file()
        assert "not a IQ0.9_L model" in text

    def test_an_error_is_a_message_not_a_traceback(self, isolated, source):
        code, _text = self._run("wrap", str(source), "-o", "/dev/null")
        assert code == 1
