"""Strife (0.73.0 alpha 1): HyperNix's own GGUF runtime.

Tiny models written by ``hypernix.strife.fixtures`` -- real GGUF files
with random weights -- so nothing is downloaded. Correctness is checked
against ``transformers`` running the same weights (the trusted baseline),
with explicit tolerances.
"""
from __future__ import annotations

import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from hypernix.quant.gguf import GGUFFile
from hypernix.strife import Engine, SamplerConfig, StrifeError, inspect_model
from hypernix.strife.fixtures import QUANT_TYPES, make_model
from hypernix.strife.inspect import read_gguf
from hypernix.strife.model import Transformer
from hypernix.strife.quants import decode, supported_type, supported_types
from hypernix.strife.weights import WeightStore

GREEDY = SamplerConfig(temperature=0, repeat_penalty=1.0)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    base = tmp_path_factory.mktemp("strife")
    out = {}
    for arch in ("llama", "qwen2", "qwen3"):
        out[arch] = make_model(base / f"{arch}.gguf", arch=arch, seed=1)
    out["spm"] = make_model(base / "spm.gguf", arch="llama", tokenizer="llama", seed=2)
    out["tied"] = make_model(base / "tied.gguf", arch="qwen2", tied=True, seed=3)
    return out


def _transformer(path: str, cache_bytes: int | None = None) -> Transformer:
    gguf, report = read_gguf(path)
    assert report.runnable, report.problems
    return Transformer(WeightStore(gguf, cache_bytes=cache_bytes), report)


def _hf_logits(model: dict, ids: list[int], hf_weights: dict | None = None) -> np.ndarray:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    c = model["config"]
    classes = {"llama": ("LlamaConfig", "LlamaForCausalLM"), "qwen2": ("Qwen2Config", "Qwen2ForCausalLM"),
               "qwen3": ("Qwen3Config", "Qwen3ForCausalLM")}[c["arch"]]
    cfg = getattr(transformers, classes[0])(
        vocab_size=c["n_vocab"], hidden_size=c["n_embd"], intermediate_size=c["n_ff"],
        num_hidden_layers=c["n_layers"], num_attention_heads=c["n_heads"],
        num_key_value_heads=c["n_kv_heads"], rms_norm_eps=c["eps"], rope_theta=c["rope_base"],
        max_position_embeddings=c["context"], tie_word_embeddings=c["tied"],
        head_dim=c["head_dim"])
    hf = getattr(transformers, classes[1])(cfg).eval()
    weights = hf_weights or model["hf"]
    hf.load_state_dict({k: torch.tensor(v) for k, v in weights.items()}, strict=False)
    with torch.no_grad():
        return hf(torch.tensor([ids])).logits[0].numpy()


IDS = [5, 77, 300, 12, 9, 200, 41, 3]


class TestAgainstTransformers:
    @pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3"])
    def test_every_position_matches(self, models, arch):
        tr = _transformer(models[arch]["path"])
        ours = tr.forward(IDS, tr.new_state(64), all_logits=True)
        np.testing.assert_allclose(ours, _hf_logits(models[arch], IDS), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3"])
    def test_decoding_one_token_at_a_time_through_the_cache_agrees(self, models, arch):
        tr = _transformer(models[arch]["path"])
        cache = tr.new_state(64)
        tr.forward(IDS[:3], cache)
        for i, token in enumerate(IDS[3:], start=3):
            last = tr.forward([token], cache)
            whole = _transformer(models[arch]["path"]).forward(
                IDS[: i + 1], tr.new_state(64))
            np.testing.assert_allclose(last, whole, atol=1e-4)

    def test_a_tied_output_uses_the_embedding(self, models):
        tr = _transformer(models["tied"]["path"])
        assert tr.plan.tied
        np.testing.assert_allclose(tr.forward(IDS, tr.new_state(64), all_logits=True),
                                   _hf_logits(models["tied"], IDS), atol=1e-4)

    def test_a_tiny_decode_cache_changes_nothing(self, models):
        """Over budget, weights are multiplied a slab at a time instead."""
        roomy = _transformer(models["qwen2"]["path"])
        tight = _transformer(models["qwen2"]["path"], cache_bytes=0)
        a = roomy.forward(IDS, roomy.new_state(64), all_logits=True)
        b = tight.forward(IDS, tight.new_state(64), all_logits=True)
        np.testing.assert_allclose(a, b, atol=1e-5)
        assert tight.w.stats()["cached_bytes"] == 0


def _dequantized_hf(path: str, model: dict) -> dict[str, np.ndarray]:
    """The HF weights as the quantised file actually stores them (qwen2 layout)."""
    gguf = GGUFFile.read(path)
    store = WeightStore(gguf)
    names = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
             "output.weight": "lm_head.weight"}
    for i in range(model["config"]["n_layers"]):
        for g, h in (("attn_norm.weight", "input_layernorm.weight"),
                     ("ffn_norm.weight", "post_attention_layernorm.weight"),
                     ("attn_q.weight", "self_attn.q_proj.weight"), ("attn_k.weight", "self_attn.k_proj.weight"),
                     ("attn_v.weight", "self_attn.v_proj.weight"), ("attn_output.weight", "self_attn.o_proj.weight"),
                     ("ffn_gate.weight", "mlp.gate_proj.weight"), ("ffn_up.weight", "mlp.up_proj.weight"),
                     ("ffn_down.weight", "mlp.down_proj.weight"), ("attn_q.bias", "self_attn.q_proj.bias"),
                     ("attn_k.bias", "self_attn.k_proj.bias"), ("attn_v.bias", "self_attn.v_proj.bias")):
            names[f"blk.{i}.{g}"] = f"model.layers.{i}.{h}"
    return {hf: store.get(g).copy() for g, hf in names.items() if store.has(g)}


K_QUANTS = ["Q8_0", "Q4_0", "Q4_K", "Q3_K", "Q6_K", "Q5_K", "Q2_K"]


class TestQuantisedModels:
    @pytest.mark.parametrize("quant", K_QUANTS)
    def test_runs_exactly_what_the_file_stores(self, tmp_path, quant):
        """The forward pass over a quantised file equals transformers over the
        same weights decoded -- so the quantised kernels add no error of their own."""
        model = make_model(tmp_path / "m.gguf", arch="qwen2", quant=quant, n_embd=256, n_ff=512,
                           n_heads=4, n_kv_heads=2, n_layers=1, seed=7)
        tr = _transformer(model["path"])
        ours = tr.forward(IDS, tr.new_state(32), all_logits=True)
        ref = _hf_logits(model, IDS, _dequantized_hf(model["path"], model))
        np.testing.assert_allclose(ours, ref, atol=2e-4, rtol=1e-3)

    @pytest.mark.parametrize("quant", ["Q8_0", "Q4_K", "Q3_K"])
    def test_and_stays_close_to_the_unquantised_model(self, tmp_path, quant):
        f32 = make_model(tmp_path / "f.gguf", arch="qwen2", n_embd=256, n_ff=512, n_layers=1, seed=7)
        q = make_model(tmp_path / "q.gguf", arch="qwen2", quant=quant, n_embd=256, n_ff=512,
                       n_layers=1, seed=7)
        a, b = _transformer(f32["path"]), _transformer(q["path"])
        la = a.forward(IDS, a.new_state(32), all_logits=True)
        lb = b.forward(IDS, b.new_state(32), all_logits=True)
        corr = np.corrcoef(la.reshape(-1), lb.reshape(-1))[0, 1]
        # Measured on these random weights: Q8_0 0.9999, Q4_K 0.98, Q3_K 0.92-0.93.
        assert corr > {"Q8_0": 0.999, "Q4_K": 0.97, "Q3_K": 0.90}[quant], corr


HNX = [name for name in QUANT_TYPES if not name.startswith(("F", "Q"))]


class TestHyperNixTypes:
    def test_every_hnx_type_is_advertised(self):
        names = set(supported_types().values())
        assert {"HNX_IQ0.5_XXXL", "HNX_IQ0.25_UXL", "HNX_IQ0.75_M", "HNX_IQ0.9_L"} <= names

    @pytest.mark.parametrize("tier", HNX)
    def test_every_advertised_hnx_type_loads_and_runs(self, tmp_path, tier):
        model = make_model(tmp_path / "h.gguf", arch="qwen2", quant=tier, n_embd=256, n_ff=512,
                           n_layers=1, seed=5)
        report = inspect_model(model["path"])
        assert report.runnable, report.problems
        tr = _transformer(model["path"])
        logits = tr.forward(IDS, tr.new_state(32))
        assert np.isfinite(logits).all()
        ref = _hf_logits(model, IDS, _dequantized_hf(model["path"], model))[-1]
        np.testing.assert_allclose(logits, ref, atol=2e-4, rtol=1e-3)

    def test_an_unknown_type_is_never_read_as_another(self):
        assert not supported_type(16)          # IQ2_XXS: no verified decoder
        with pytest.raises(StrifeError, match="no verified decoder"):
            decode(b"\0" * 66, 16)


def _set_type(path: Path, tensor: str, new_type: int) -> None:
    """Rewrite one tensor's type id in the header, in place."""
    data = bytearray(path.read_bytes())
    key = tensor.encode()
    at = data.find(struct.pack("<Q", len(key)) + key)
    assert at > 0
    dims = struct.unpack_from("<I", data, at + 8 + len(key))[0]
    type_at = at + 8 + len(key) + 4 + 8 * dims
    struct.pack_into("<I", data, type_at, new_type)
    path.write_bytes(bytes(data))


class TestInspection:
    def test_a_good_model(self, models):
        r = inspect_model(models["qwen2"]["path"])
        assert r.runnable and r.architecture == "qwen2" and r.n_layers == 2
        assert r.tokenizer == "gpt2" and r.n_vocab == models["qwen2"]["config"]["n_vocab"]

    def test_truncated(self, models, tmp_path):
        src = Path(models["llama"]["path"]).read_bytes()
        bad = tmp_path / "cut.gguf"
        bad.write_bytes(src[: len(src) - 1000])
        r = inspect_model(bad)
        assert not r.runnable and r.problems[0]["code"] == "strife.model.truncated"

    def test_not_gguf(self, tmp_path):
        bad = tmp_path / "x.gguf"
        bad.write_bytes(b"NOPE" + b"\0" * 100)
        assert inspect_model(bad).problems[0]["code"] == "strife.model.not_gguf"

    def test_a_hostile_length_does_not_allocate_it(self, tmp_path):
        """A 40-byte file claiming a 2^33-element array is refused, not allocated."""
        bad = tmp_path / "huge.gguf"
        header = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
        kv = struct.pack("<Q", 1) + b"k" + struct.pack("<IIQ", 9, 4, 1 << 33)
        bad.write_bytes(header + kv)
        r = inspect_model(bad)
        assert r.problems[0]["code"] == "strife.model.not_gguf"

    def test_an_unsupported_tensor_type(self, models, tmp_path):
        bad = tmp_path / "t.gguf"
        bad.write_bytes(Path(models["llama"]["path"]).read_bytes())
        _set_type(bad, "blk.0.attn_q.weight", 16)
        r = inspect_model(bad)
        codes = {p["code"] for p in r.problems}
        assert "strife.quant.unsupported" in codes
        with pytest.raises(StrifeError) as info:
            Engine(bad)
        assert info.value.code == "strife.quant.unsupported"

    def test_an_unsupported_architecture(self, models, tmp_path):
        bad = tmp_path / "a.gguf"
        data = Path(models["llama"]["path"]).read_bytes()
        bad.write_bytes(data.replace(b"\x05\x00\x00\x00\x00\x00\x00\x00llama", b"\x05\x00\x00\x00\x00\x00\x00\x00gemma", 1))
        r = inspect_model(bad)
        assert r.problems[0]["code"] == "strife.arch.unsupported"
        assert "llama" in r.problems[0]["remedy"]

    def test_missing_file(self, tmp_path):
        assert inspect_model(tmp_path / "nope.gguf").problems[0]["code"] == "strife.model.unreadable"


class TestTokenizer:
    def test_gpt2_round_trips_any_text(self, models):
        engine = Engine(models["qwen2"]["path"], n_ctx=64)
        tok = engine.tokenizer
        for text in ["hello world", "Héllo wörld 💥 — 123", "the model\n\n  and   strife", ""]:
            assert tok.decode(tok.encode(text)) == text

    def test_gpt2_matches_the_tokenizers_library(self, models):
        tokenizers = pytest.importorskip("tokenizers")
        from tokenizers import Regex, decoders, pre_tokenizers
        from tokenizers import models as tmodels

        from hypernix.strife.tokenizer import _QWEN2

        engine = Engine(models["qwen2"]["path"], n_ctx=64)
        tok = engine.tokenizer
        c = models["qwen2"]["config"]
        vocab = {t: i for i, t in enumerate(c["tokens"])}
        merges = [tuple(m.split(" ")) for m in c["merges"]]
        ref = tokenizers.Tokenizer(tmodels.BPE(vocab=vocab, merges=merges))
        ref.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(_QWEN2), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        ref.decoder = decoders.ByteLevel()
        for text in ["hello world and the model", "strife's 42 models!!\n\nok", "über naïve"]:
            assert tok.encode(text, special=False) == ref.encode(text).ids, text

    def test_specials_are_one_token(self, models):
        tok = Engine(models["qwen2"]["path"], n_ctx=64).tokenizer
        ids = tok.encode("<|im_start|>user\nhi<|im_end|>")
        assert ids[0] == tok.vocab["<|im_start|>"] and ids[-1] == tok.vocab["<|im_end|>"]
        assert tok.vocab["<|im_end|>"] in tok.stop_ids

    def test_sentencepiece_round_trips_with_byte_fallback(self, models):
        tok = Engine(models["spm"]["path"], n_ctx=64).tokenizer
        ids = tok.encode("hello world, strife!")
        assert ids[0] == tok.bos_id
        assert tok.decode(ids[1:]) == "hello world, strife!"
        assert tok.decode(tok.encode("ü ☃", add_bos=False)) == "ü ☃"  # through <0xNN>
        assert tok.vocab["▁hello"] in ids  # merged, not spelled

    def test_chat_template_renders_sandboxed(self, tmp_path):
        tpl = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
               "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        model = make_model(tmp_path / "c.gguf", arch="qwen2", chat_template=tpl)
        tok = Engine(model["path"], n_ctx=64).tokenizer
        text = tok.apply_chat_template([{"role": "user", "content": "hi"}])
        assert text == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        evil = make_model(tmp_path / "e.gguf", arch="qwen2",
                          chat_template="{{ ''.__class__.__mro__[1].__subclasses__() }}")
        with pytest.raises(StrifeError, match="chat template"):
            Engine(evil["path"], n_ctx=64).tokenizer.apply_chat_template([])


class TestEngine:
    @pytest.fixture(scope="class")
    def engine(self, models):
        with Engine(models["qwen2"]["path"], n_ctx=128, max_concurrent=2, max_sessions=2) as e:
            yield e

    def test_greedy_is_deterministic(self, engine):
        a = engine.generate("hello world", max_tokens=12, sampling=GREEDY)
        b = engine.generate("hello world", max_tokens=12, sampling=GREEDY)
        assert a.tokens == b.tokens and len(a.tokens) == 12 and a.stop_reason == "length"

    def test_a_seed_is_reproducible_and_seeds_differ(self, engine):
        cfg = SamplerConfig(temperature=1.0, seed=11)
        a = engine.generate("the model", max_tokens=16, sampling=cfg)
        b = engine.generate("the model", max_tokens=16, sampling=cfg)
        c = engine.generate("the model", max_tokens=16, sampling=SamplerConfig(temperature=1.0, seed=12))
        assert a.tokens == b.tokens and a.tokens != c.tokens

    def test_streaming_adds_up_to_the_result(self, engine):
        events = list(engine.stream("hello", max_tokens=10, sampling=GREEDY))
        assert events[-1]["type"] == "done"
        assert "".join(e["text"] for e in events[:-1]) == events[-1]["text"]
        stats = events[-1]["stats"]
        assert stats["generated_tokens"] == 10 and stats["prompt_tokens"] > 0
        assert stats["time_to_first_token_seconds"] >= 0

    def test_stop_strings_cut_the_text(self, engine):
        full = engine.generate("hello world", max_tokens=20, sampling=GREEDY).text
        stop = full[5:8]
        assert stop and full.find(stop) >= 0
        cut = engine.generate("hello world", max_tokens=20, sampling=GREEDY, stop=[stop])
        assert cut.stop_reason == "stop_string" and cut.text == full[: full.find(stop)]
        streamed = "".join(e["text"] for e in engine.stream(
            "hello world", max_tokens=20, sampling=GREEDY, stop=[stop]) if e["type"] == "token")
        assert streamed == cut.text

    def test_cancellation(self, engine):
        cancel = threading.Event()
        cancel.set()
        result = engine.generate("hello", max_tokens=50, sampling=GREEDY, cancel=cancel)
        assert result.stop_reason == "cancelled" and len(result.tokens) < 50

    def test_cancel_mid_stream(self, engine):
        cancel = threading.Event()
        seen = 0
        for event in engine.stream("hello", max_tokens=50, sampling=GREEDY, cancel=cancel):
            if event["type"] == "token":
                seen += 1
                if seen == 3:
                    cancel.set()
            else:
                assert event["stop_reason"] == "cancelled"
                assert len(event["tokens"]) <= 4

    def test_timeout(self, engine):
        result = engine.generate("hello", max_tokens=500, sampling=GREEDY, timeout=1e-9)
        assert result.stop_reason == "timeout"

    def test_a_full_context_stops_cleanly(self, models):
        with Engine(models["qwen2"]["path"], n_ctx=16) as small:
            result = small.generate("hello world", max_tokens=100, sampling=GREEDY)
            assert result.stop_reason == "context_full"
            with pytest.raises(StrifeError) as info:
                small.generate("hello world " * 20, max_tokens=1)
            assert info.value.code == "strife.runtime.context_full"

    def test_a_session_reuses_its_prefix_and_gives_the_same_answer(self, engine):
        sid = engine.new_session()
        first = engine.generate("hello world and the model", session=sid, max_tokens=4, sampling=GREEDY)
        again = engine.generate("hello world and the model then", session=sid, max_tokens=6,
                                sampling=GREEDY)
        fresh = engine.generate("hello world and the model then", max_tokens=6, sampling=GREEDY)
        assert again.stats["prompt_tokens_reused"] > 0
        assert again.tokens == fresh.tokens
        assert first.stats["prompt_tokens_reused"] == 0

    def test_sessions_are_bounded(self, engine):
        ids = [engine.new_session() for _ in range(3)]
        with pytest.raises(StrifeError) as info:
            engine.generate("hi", session=ids[0], max_tokens=1)
        assert info.value.code == "strife.engine.no_session"
        assert engine.close_session(ids[2])

    def test_concurrent_requests_do_not_disturb_each_other(self, engine):
        prompts = ["hello world", "the model and", "strife on", "and the world"]
        alone = {p: engine.generate(p, max_tokens=10, sampling=GREEDY).tokens for p in prompts}
        together: dict[str, list[int]] = {}

        def run(p: str) -> None:
            together[p] = engine.generate(p, max_tokens=10, sampling=GREEDY).tokens

        threads = [threading.Thread(target=run, args=(p,)) for p in prompts * 2]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert together == alone

    def test_a_full_queue_is_refused_not_grown(self, models):
        with Engine(models["qwen2"]["path"], n_ctx=64, max_concurrent=1, max_queue=0) as e:
            first = e.stream("hello", max_tokens=5, sampling=GREEDY)
            next(first)  # holds the only slot
            with pytest.raises(StrifeError) as info:
                next(e.stream("hello", max_tokens=5, sampling=GREEDY))
            assert info.value.code == "strife.engine.busy"
            list(first)

    def test_chat_messages(self, models, tmp_path):
        engine = Engine(models["qwen2"]["path"], n_ctx=128)
        result = engine.generate(messages=[{"role": "user", "content": "hello"}], max_tokens=4,
                                 sampling=GREEDY)
        assert result.stats["prompt_tokens"] > 3

    def test_the_status_reports_memory_separately(self, engine):
        status = engine.status()
        assert status["weights"]["mapped_bytes"] > 0
        assert status["placement"]["kv_cache_bytes_per_session"] > 0
        assert status["placement"]["gpu_layers_used"] == 0


class TestPlacement:
    def test_a_gpu_request_is_answered_honestly(self, models):
        from hypernix.strife.placement import plan

        report = inspect_model(models["qwen2"]["path"])
        hw = {"gpus": [], "ram": {"available_bytes": 1 << 34, "total_bytes": 1 << 35}}
        p = plan(report, n_ctx=64, gpu_layers=10, hardware=hw)
        assert p["gpu_layers_used"] == 0 and p["cpu_layers"] == report.n_layers
        assert any("no GPU" in r for r in p["fallback_reasons"])
        hw["gpus"] = [{"index": 0, "name": "GTX 1080", "vram_bytes": 8 << 30}]
        p = plan(report, n_ctx=64, gpu_layers=10, hardware=hw)
        assert any("not in Strife alpha 1" in r for r in p["fallback_reasons"])

    @pytest.mark.parametrize("chip,tier", [("Apple M5 Pro", "experimental"), ("Apple M2", "unsupported"),
                                           ("Intel(R) Core(TM) i9", "unsupported")])
    def test_macs_are_scoped_to_m5_and_newer(self, chip, tier):
        from hypernix.strife.placement import platform_support

        assert platform_support("Darwin", "arm64", chip)["tier"] == tier


class TestCLI:
    def test_selftest(self, capsys):
        from hypernix.strife.cli import cli_main

        assert cli_main(["selftest", "--json"]) == 0
        assert '"ok": true' in capsys.readouterr().out

    def test_inspect_and_run(self, models, capsys):
        import json

        from hypernix.strife.cli import cli_main

        assert cli_main(["inspect", models["qwen2"]["path"], "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["runnable"] is True
        assert cli_main(["run", models["qwen2"]["path"], "-p", "hello", "-n", "4", "--temp", "0",
                         "--json"]) == 0
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert lines[-1]["type"] == "done" and len(lines[-1]["tokens"]) == 4

    def test_errors_carry_a_code(self, tmp_path, capsys):
        from hypernix.strife.cli import cli_main

        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"nope")
        assert cli_main(["run", str(bad), "-p", "x"]) == 2
        assert "strife.model.not_gguf" in capsys.readouterr().err

    def test_hnx_strife(self, capsys):
        from hypernix.interfaces.cli import main

        assert main(["strife", "types"]) == 0
        assert "Q4_K" in capsys.readouterr().out


@pytest.mark.skipif(not __import__("os").environ.get("STRIFE_TEST_MODEL"),
                    reason="set STRIFE_TEST_MODEL to a local GGUF (opt-in; nothing is downloaded)")
def test_a_real_model_against_transformers():
    """Opt-in: a real GGUF, compared with transformers loading the same file.

    Run on Qwen2.5-0.5B-Instruct Q4_K_M (0.73.0 alpha 1): identical token
    ids, logits within 1.5e-4 of a ±25 range, every argmax the same.
    """
    import os

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("gguf")
    path = Path(os.environ["STRIFE_TEST_MODEL"])
    engine = Engine(path, n_ctx=256)
    text = "The quick brown fox jumps over the lazy dog. Grüße, 東京! 12345"
    ids = engine.tokenizer.encode(text)
    tok = transformers.AutoTokenizer.from_pretrained(path.parent, gguf_file=path.name)
    assert ids == tok(text)["input_ids"]
    hf = transformers.AutoModelForCausalLM.from_pretrained(path.parent, gguf_file=path.name,
                                                           dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()
    ours = engine.model.forward(ids, engine.model.new_state(256), all_logits=True)
    assert np.abs(ours - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())
    assert (ours.argmax(-1) == ref.argmax(-1)).mean() == 1.0
