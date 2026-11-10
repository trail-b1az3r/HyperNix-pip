"""Strife: LFM3, Qwen3.5 (dense and MoE) and GLM-4.5 (0.73.0 alpha 1).

Each architecture is written as a tiny GGUF the way llama.cpp's converter
writes it, then compared with a trusted implementation of the original:

* Qwen3.5, Qwen3.5-MoE and GLM-4.5 with ``transformers``' own models;
* LFM3, which ``transformers`` does not have yet, with an independent
  PyTorch implementation in this file, written from Liquid's published
  modeling code (``LiquidAI/LFM3-1.7B-Base``) rather than from Strife.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from hypernix.strife import Engine, SamplerConfig, StrifeError, inspect_model
from hypernix.strife.archs import PlanError, build_plan
from hypernix.strife.fixtures import make_model
from hypernix.strife.inspect import read_gguf, shapes_of
from hypernix.strife.model import Transformer
from hypernix.strife.weights import WeightStore

torch = pytest.importorskip("torch")

IDS = [5, 77, 300, 12, 9, 200, 41, 3, 17, 250, 64, 8]
GREEDY = SamplerConfig(temperature=0, repeat_penalty=1.0)


def _strife(path: str):
    gguf, report = read_gguf(path)
    assert report.runnable, report.problems
    return Transformer(WeightStore(gguf), report)


def _hf(model: dict, config_cls: str, model_cls: str) -> np.ndarray:
    transformers = pytest.importorskip("transformers")
    cfg = {k: v for k, v in model["config"].items() if k != "arch"}
    hf = getattr(transformers, model_cls)(getattr(transformers, config_cls)(**cfg)).eval()
    missing, unexpected = hf.load_state_dict({k: torch.tensor(v) for k, v in model["hf"].items()},
                                             strict=False)
    assert not [m for m in missing if "rotary" not in m] and not unexpected
    with torch.no_grad():
        return hf(torch.tensor([IDS])).logits[0].numpy()


REFERENCES = {
    "qwen35": ("Qwen3_5TextConfig", "Qwen3_5ForCausalLM"),
    "qwen35moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM"),
    "glm4moe": ("Glm4MoeConfig", "Glm4MoeForCausalLM"),
}


# ---------------------------------------------------------------------------
# LFM3, independently
# ---------------------------------------------------------------------------

def _lfm3_reference(model: dict, ids: list[int]) -> np.ndarray:
    """LFM3's forward pass in PyTorch, from Liquid's ``modeling_lfm3.py``."""
    F = torch.nn.functional
    w = {k: torch.tensor(v) for k, v in model["hf"].items()}
    c = model["config"]
    E, H, KV, D, eps = c["hidden_size"], c["heads"], c["kv_heads"], c["head_dim"], c["eps"]

    def rms(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def rotary(kind: str):
        base = c["rope_base"] if kind == "full_attention" else c["rope_base_swa"]
        factor = c["partial"] if kind == "full_attention" else 1.0
        angles = int(factor * D) // 2
        inv = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / D))
        inv = torch.cat((inv, torch.zeros(D // 2 - angles)))
        freqs = torch.arange(len(ids), dtype=torch.float32)[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def rotate_half(x):
        return torch.cat((-x[..., D // 2:], x[..., : D // 2]), dim=-1)

    T = len(ids)
    x = w["model.embed_tokens.weight"][torch.tensor(ids)]
    for i, kind in enumerate(c["layer_types"]):
        p = f"model.layers.{i}."
        h = rms(x, w[p + "operator_norm.weight"])
        if kind == "conv":
            bcx = h @ w[p + "conv.in_proj.weight"].T
            b, cc, xx = bcx.chunk(3, dim=-1)
            bx = (b * xx).T[None]                                   # (1, E, T)
            conv = F.conv1d(bx, w[p + "conv.conv.weight"], padding=c["kernel"] - 1, groups=E)[..., :T]
            y = (cc * conv[0].T) @ w[p + "conv.out_proj.weight"].T
        else:
            a = p + "self_attn."
            q = rms((h @ w[a + "q_proj.weight"].T).view(T, H, D), w[a + "q_norm.weight"])
            k = rms((h @ w[a + "k_proj.weight"].T).view(T, KV, D), w[a + "k_norm.weight"])
            v = (h @ w[a + "v_proj.weight"].T).view(T, KV, D)
            cos, sin = rotary(kind)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            k = k.repeat_interleave(H // KV, dim=1)
            v = v.repeat_interleave(H // KV, dim=1)
            scores = torch.einsum("thd,shd->hts", q, k) * D ** -0.5
            pos = torch.arange(T)
            hidden = pos[None, :] > pos[:, None]
            if kind == "sliding_attention":
                hidden |= (pos[:, None] - pos[None, :]) >= c["window"]
            scores = scores.masked_fill(hidden[None], float("-inf")).softmax(-1)
            out = torch.einsum("hts,shd->thd", scores, v)
            gate = torch.sigmoid(h @ w[a + "gate_proj.weight"].T)[..., None]
            y = (out * gate).reshape(T, H * D) @ w[a + "o_proj.weight"].T
        y = rms(y, w[p + "post_operator_norm.weight"])
        x = x + y
        n = rms(x, w[p + "ffn_norm.weight"])
        f = (F.silu(n @ w[p + "feed_forward.w1.weight"].T) * (n @ w[p + "feed_forward.w3.weight"].T)) \
            @ w[p + "feed_forward.w2.weight"].T
        x = x + rms(f, w[p + "post_ffn_norm.weight"])
    x = rms(x, w["model.embedding_norm.weight"])
    return (x @ w["model.embed_tokens.weight"].T).numpy()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("archs")
    return {arch: make_model(base / f"{arch}.gguf", arch=arch, seed=3)
            for arch in ("lfm3", "qwen35", "qwen35moe", "glm4moe")}


class TestAgainstReferences:
    @pytest.mark.parametrize("arch", ["qwen35", "qwen35moe", "glm4moe"])
    def test_every_position_matches_transformers(self, built, arch):
        tr = _strife(built[arch]["path"])
        ours = tr.forward(IDS, tr.new_state(64), all_logits=True)
        np.testing.assert_allclose(ours, _hf(built[arch], *REFERENCES[arch]), atol=1e-4, rtol=1e-4)

    def test_lfm3_matches_the_independent_reference(self, built):
        tr = _strife(built["lfm3"]["path"])
        ours = tr.forward(IDS, tr.new_state(64), all_logits=True)
        np.testing.assert_allclose(ours, _lfm3_reference(built["lfm3"], IDS), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("arch", ["lfm3", "qwen35", "qwen35moe", "glm4moe"])
    def test_one_token_at_a_time_through_the_state_agrees(self, built, arch):
        tr = _strife(built[arch]["path"])
        whole = tr.forward(IDS, tr.new_state(64), all_logits=True)
        state = tr.new_state(64)
        tr.forward(IDS[:5], state)
        for i, token in enumerate(IDS[5:], start=5):
            np.testing.assert_allclose(tr.forward([token], state), whole[i], atol=1e-4)

    def test_the_sliding_window_matters(self, built):
        """Past the window (4 here), LFM3's sliding layer must forget."""
        tr = _strife(built["lfm3"]["path"])
        plan = tr.plan
        assert [lay.attn.window for lay in plan.layers if lay.attn] == [0, 4, 0]
        np.testing.assert_allclose(tr.forward(IDS, tr.new_state(64), all_logits=True),
                                   _lfm3_reference(built["lfm3"], IDS), atol=1e-4)

    def test_fused_expert_gate_up_is_the_same_model(self, tmp_path):
        a = make_model(tmp_path / "a.gguf", arch="qwen35moe", seed=5)
        from hypernix.strife.fixtures import _make_qwen35

        b = _make_qwen35(tmp_path / "b.gguf", moe=True, fused=True, seed=5)
        assert "blk.0.ffn_gate_up_exps.weight" in shapes_of(read_gguf(b["path"])[0])
        ta, tb = _strife(a["path"]), _strife(b["path"])
        np.testing.assert_allclose(ta.forward(IDS, ta.new_state(32), all_logits=True),
                                   tb.forward(IDS, tb.new_state(32), all_logits=True), atol=1e-5)

    def test_glm_without_qk_norm(self, tmp_path):
        from hypernix.strife.fixtures import _make_glm4moe

        model = _make_glm4moe(tmp_path / "g.gguf", qk_norm=False, seed=6)
        tr = _strife(model["path"])
        assert not tr.plan.layers[0].attn.qk_norm
        np.testing.assert_allclose(tr.forward(IDS, tr.new_state(32), all_logits=True),
                                   _hf(model, *REFERENCES["glm4moe"]), atol=1e-4)

    @pytest.mark.parametrize("arch", ["lfm3", "qwen35", "glm4moe"])
    def test_quantised_still_close(self, tmp_path, arch):
        f32 = make_model(tmp_path / "f.gguf", arch=arch, seed=4)
        q8 = make_model(tmp_path / "q.gguf", arch=arch, quant="Q8_0", seed=4)
        types = inspect_model(q8["path"]).types
        assert types.get("Q8_0"), types
        a, b = _strife(f32["path"]), _strife(q8["path"])
        la = a.forward(IDS, a.new_state(32), all_logits=True)
        lb = b.forward(IDS, b.new_state(32), all_logits=True)
        assert np.corrcoef(la.ravel(), lb.ravel())[0, 1] > 0.999


class TestInspection:
    def test_reports_say_what_the_model_is(self, built):
        r = inspect_model(built["lfm3"]["path"])
        assert r.runnable and r.details["layers"] == {"conv": 3, "attn": 2, "swa": 1}
        r = inspect_model(built["qwen35"]["path"])
        assert r.details["linear_layers"] == 3 and r.details["attention_layers"] == 1
        r = inspect_model(built["glm4moe"]["path"])
        assert r.details["experts"] == "2 of 8" and r.details["dense_layers"] == 1

    def test_a_non_causal_lfm3_is_refused(self, built):
        gguf, _ = read_gguf(built["lfm3"]["path"])
        md = dict(gguf.metadata, **{"lfm3.attention.causal": False})
        with pytest.raises(PlanError) as info:
            build_plan(md, shapes_of(gguf))
        assert info.value.code == "strife.arch.non_causal"

    def test_group_limited_routing_is_refused_not_ignored(self, built):
        gguf, _ = read_gguf(built["glm4moe"]["path"])
        md = dict(gguf.metadata, **{"glm4moe.expert_group_count": 4,
                                    "glm4moe.expert_group_used_count": 2})
        with pytest.raises(PlanError) as info:
            build_plan(md, shapes_of(gguf))
        assert info.value.code == "strife.arch.moe_groups"

    def test_a_missing_tensor_is_named(self, built):
        gguf, _ = read_gguf(built["qwen35"]["path"])
        shapes = shapes_of(gguf)
        del shapes["blk.1.ssm_a"]
        with pytest.raises(PlanError, match="blk.1.ssm_a"):
            build_plan(gguf.metadata, shapes)

    def test_mtp_layers_are_not_run(self, built):
        gguf, _ = read_gguf(built["glm4moe"]["path"])
        md = dict(gguf.metadata, **{"glm4moe.block_count": 4, "glm4moe.nextn_predict_layers": 1})
        plan = build_plan(md, shapes_of(gguf))
        assert plan.n_layers == 3 and plan.summary["mtp_layers_skipped"] == 1


class TestRecurrentSessions:
    @pytest.fixture(scope="class")
    def engine(self, tmp_path_factory):
        model = make_model(tmp_path_factory.mktemp("rs") / "q.gguf", arch="qwen35", seed=8)
        with Engine(model["path"], n_ctx=128, max_sessions=4) as e:
            yield e

    def test_a_conversation_continues_from_its_checkpoint(self, engine):
        sid = engine.new_session()
        first = "hello world and the model"
        engine.generate(first, session=sid, max_tokens=3, sampling=GREEDY)
        follow = first + " then the strife"
        again = engine.generate(follow, session=sid, max_tokens=6, sampling=GREEDY)
        fresh = engine.generate(follow, max_tokens=6, sampling=GREEDY)
        assert again.stats["prompt_tokens_reused"] > 0
        assert again.tokens == fresh.tokens

    def test_an_unrelated_prompt_starts_over(self, engine):
        sid = engine.new_session()
        engine.generate("hello world", session=sid, max_tokens=3, sampling=GREEDY)
        other = engine.generate("the model on", session=sid, max_tokens=5, sampling=GREEDY)
        assert other.stats["prompt_tokens_reused"] == 0
        assert other.tokens == engine.generate("the model on", max_tokens=5, sampling=GREEDY).tokens

    def test_cancelling_mid_reply_leaves_the_session_usable(self, engine):
        sid = engine.new_session()
        cancel = threading.Event()
        for event in engine.stream("hello world", session=sid, max_tokens=40, sampling=GREEDY,
                                   cancel=cancel):
            if event["type"] == "token":
                cancel.set()
        follow = "hello world and"
        again = engine.generate(follow, session=sid, max_tokens=6, sampling=GREEDY)
        assert again.tokens == engine.generate(follow, max_tokens=6, sampling=GREEDY).tokens

    def test_state_size_is_reported(self, engine):
        p = engine.status()["placement"]
        assert p["recurrent_state_bytes_per_session"] > 0 and p["kv_cache_bytes_per_session"] > 0

    def test_truncating_to_a_place_it_cannot_reach_says_so(self, engine):
        state = engine.model.new_state(64)
        engine.model.forward([5, 6, 7, 8], state)
        with pytest.raises(StrifeError) as info:
            state.truncate(2)
        assert info.value.code == "strife.runtime.no_rewind"


class TestTokenizers:
    def _tokenizer(self, tmp_path, pre: str):
        from hypernix.strife.tokenizer import Tokenizer

        model = make_model(tmp_path / "t.gguf", arch="qwen2")
        gguf, _ = read_gguf(model["path"])
        return Tokenizer(dict(gguf.metadata, **{"tokenizer.ggml.pre": pre})), model

    def test_ignore_merges_takes_a_whole_word_when_it_exists(self, tmp_path):
        tok, _ = self._tokenizer(tmp_path, "lfm2")
        for text in ("hello", " the", "strife", " world"):
            assert len(tok.encode(text)) == 1, text
            assert tok.decode(tok.encode(text)) == text

    @pytest.mark.parametrize("pre", ["lfm2", "glm4", "qwen35"])
    def test_the_new_pre_tokenizers_match_the_tokenizers_library(self, tmp_path, pre):
        tokenizers = pytest.importorskip("tokenizers")
        from tokenizers import Regex, pre_tokenizers
        from tokenizers import models as tmodels

        from hypernix.strife.tokenizer import _IGNORE_MERGES, _PRE

        tok, model = self._tokenizer(tmp_path, pre)
        vocab = {t: i for i, t in enumerate(model["config"]["tokens"])}
        merges = [tuple(m.split(" ")) for m in model["config"]["merges"]]
        ref = tokenizers.Tokenizer(tmodels.BPE(vocab=vocab, merges=merges,
                                               ignore_merges=pre in _IGNORE_MERGES))
        ref.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(_PRE[pre]), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        for text in ["hello world and the model", "strife's 1234 models!!\n\nok", "naïve café",
                     "ét́é ok"]:
            assert tok.encode(text, special=False) == ref.encode(text).ids, (pre, text)

    def test_an_unknown_pre_tokenizer_is_refused(self, tmp_path):
        model = make_model(tmp_path / "u.gguf", arch="qwen2")
        from hypernix.strife.inspect import ModelReport, _check_tokenizer

        gguf, _ = read_gguf(model["path"])
        gguf.metadata["tokenizer.ggml.pre"] = "made-up-pre"
        report = ModelReport(path=model["path"])
        _check_tokenizer(gguf, report)
        assert report.problems[0]["code"] == "strife.tokenizer.pre_unsupported"


class TestEngineOnEveryArchitecture:
    @pytest.mark.parametrize("arch", ["lfm3", "qwen35", "qwen35moe", "glm4moe"])
    def test_generates_deterministically(self, built, arch):
        with Engine(built[arch]["path"], n_ctx=64) as engine:
            a = engine.generate("hello world", max_tokens=8, sampling=GREEDY)
            b = engine.generate("hello world", max_tokens=8, sampling=GREEDY)
        assert a.tokens == b.tokens and len(a.tokens) == 8
