"""strife.fixtures — tiny, real GGUF models for tests and ``hnx strife selftest``.

Random weights, but a real file: the same metadata keys, tensor names,
layouts and tokenizers a converted model has, written by
:class:`hypernix.quant.gguf.GGUFWriter`. Small enough to build in a
second, so tests never need a download.

:func:`make_model` also returns the weights in Hugging Face layout, which
is what lets a test build the same model in ``transformers`` and compare
logits: the trusted baseline. For llama the q and k projections are
permuted exactly as llama.cpp's ``convert_hf_to_gguf.py`` permutes them,
because GGUF llama files use adjacent-pair rope.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..quant.gguf import GGMLType, GGUFValueType, GGUFWriter

__all__ = ["make_model", "byte_bpe_vocab", "spm_vocab", "QUANT_TYPES"]

#: Quant names :func:`make_model` can write, to their GGML type.
QUANT_TYPES: dict[str, int] = {
    "F32": int(GGMLType.F32), "F16": int(GGMLType.F16), "Q8_0": int(GGMLType.Q8_0),
    "Q4_0": int(GGMLType.Q4_0), "Q4_K": int(GGMLType.Q4_K), "Q6_K": int(GGMLType.Q6_K),
    "Q3_K": int(GGMLType.Q3_K), "Q5_K": int(GGMLType.Q5_K), "Q2_K": int(GGMLType.Q2_K),
}


def _hnx_tiers() -> dict[str, tuple[int, str]]:
    from ..quant.hyprslug import TIER_TYPES

    return dict(TIER_TYPES)


#: The HyperNix tiers (hyprslug names: "IQ0.5_XXXL", "INT4", ...) are
#: writable too, through the package's own encoders.
QUANT_TYPES.update({tier: type_id for tier, (type_id, _p) in _hnx_tiers().items()})

_STR, _U32, _I32, _F32 = (int(GGUFValueType.STRING), int(GGUFValueType.UINT32),
                          int(GGUFValueType.INT32), int(GGUFValueType.FLOAT32))
_ARR = int(GGUFValueType.ARRAY)
_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
_MERGE_WORDS = ["the", "and", "hello", "world", "strife", "model", "ing", "er", "on", "in"]


def byte_bpe_vocab() -> tuple[list[str], list[int], list[str]]:
    """A small gpt2-style vocabulary: 256 byte symbols, merges, specials."""
    from .tokenizer import _byte_to_unicode

    b2u = _byte_to_unicode()
    tokens = [b2u[b] for b in range(256)]
    merges: list[str] = []
    seen = set(tokens)
    for word in _MERGE_WORDS + [" " + w for w in _MERGE_WORDS]:
        chars = [b2u[b] for b in word.encode()]
        while len(chars) > 1:
            merged = chars[0] + chars[1]
            if merged not in seen:
                merges.append(f"{chars[0]} {chars[1]}")
                tokens.append(merged)
                seen.add(merged)
            chars = [merged] + chars[2:]
    types = [1] * len(tokens)
    for special in _SPECIALS:
        tokens.append(special)
        types.append(3)
    return tokens, types, merges


def spm_vocab() -> tuple[list[str], list[int], list[float]]:
    """A small SentencePiece vocabulary with byte fallback."""
    tokens = ["<unk>", "<s>", "</s>"]
    types = [2, 3, 3]
    scores = [0.0, 0.0, 0.0]
    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        types.append(6)
        scores.append(0.0)
    pieces = ["▁"] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + list(".,!?'")
    for word in _MERGE_WORDS:
        for i in range(2, len(word) + 1):
            pieces.append(word[:i])
            pieces.append("▁" + word[:i])
    for piece in dict.fromkeys(pieces):
        if piece not in tokens:
            tokens.append(piece)
            types.append(1)
            scores.append(-float(len(tokens)) / 1000.0 + len(piece))
    return tokens, types, scores


def _permute_for_norm_rope(w: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp convert_hf_to_gguf.py's ``permute`` for q and k."""
    out, dim = w.shape
    return (w.reshape(n_head, 2, out // n_head // 2, dim).swapaxes(1, 2).reshape(out, dim))


def _encode(values: np.ndarray, qtype: str) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if qtype == "F32":
        return flat.astype("<f4").tobytes()
    if qtype == "F16":
        return flat.astype("<f2").tobytes()
    tiers = _hnx_tiers()
    if qtype in tiers:
        from ..quant import lowbit, subbit

        packing = tiers[qtype][1]
        if packing in subbit.PACKINGS:
            return subbit.quantize_tensor(flat.tolist(), packing)
        return lowbit.quantize_array(flat, packing)
    from ..quant import llamaquants

    return llamaquants.quantize_array(flat, qtype)


def make_model(path: str | Path, *, arch: str = "llama", tokenizer: str = "gpt2",
               quant: str = "F32", n_layers: int = 2, n_embd: int = 64, n_heads: int = 4,
               n_kv_heads: int = 2, n_ff: int = 128, context: int = 256, seed: int = 0,
               tied: bool = False, chat_template: str | None = None,
               rope_base: float = 10000.0) -> dict[str, Any]:
    """Write a tiny model to *path*; return its HF-layout weights and config.

    Block-quantised types need rows that divide into blocks, so with a
    K-quant ``n_embd`` and ``n_ff`` must be multiples of 256 (norms,
    biases and the embedding stay F32 unless *quant* is a plain type).
    """
    if quant not in QUANT_TYPES:
        raise ValueError(f"quant must be one of {', '.join(QUANT_TYPES)}")
    if arch in _HYBRID:
        return _HYBRID[arch](path, quant=quant, seed=seed, chat_template=chat_template)
    rng = np.random.default_rng(seed)
    head_dim = n_embd // n_heads
    if tokenizer == "gpt2":
        tokens, types, merges = byte_bpe_vocab()
        scores: list[float] = []
    else:
        tokens, types, scores = spm_vocab()
        merges = []
    n_vocab = len(tokens)

    def mat(rows: int, cols: int, scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    hf: dict[str, np.ndarray] = {"model.embed_tokens.weight": mat(n_vocab, n_embd, 0.5)}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        hf[p + "input_layernorm.weight"] = (1.0 + rng.standard_normal(n_embd) * 0.1).astype(np.float32)
        hf[p + "post_attention_layernorm.weight"] = (1.0 + rng.standard_normal(n_embd) * 0.1).astype(np.float32)
        hf[p + "self_attn.q_proj.weight"] = mat(n_heads * head_dim, n_embd)
        hf[p + "self_attn.k_proj.weight"] = mat(n_kv_heads * head_dim, n_embd)
        hf[p + "self_attn.v_proj.weight"] = mat(n_kv_heads * head_dim, n_embd)
        hf[p + "self_attn.o_proj.weight"] = mat(n_embd, n_heads * head_dim)
        hf[p + "mlp.gate_proj.weight"] = mat(n_ff, n_embd)
        hf[p + "mlp.up_proj.weight"] = mat(n_ff, n_embd)
        hf[p + "mlp.down_proj.weight"] = mat(n_embd, n_ff)
        if arch == "qwen2":
            for name, size in (("q", n_heads), ("k", n_kv_heads), ("v", n_kv_heads)):
                hf[p + f"self_attn.{name}_proj.bias"] = (rng.standard_normal(size * head_dim) * 0.05).astype(np.float32)
        if arch == "qwen3":
            hf[p + "self_attn.q_norm.weight"] = (1.0 + rng.standard_normal(head_dim) * 0.1).astype(np.float32)
            hf[p + "self_attn.k_norm.weight"] = (1.0 + rng.standard_normal(head_dim) * 0.1).astype(np.float32)
    hf["model.norm.weight"] = (1.0 + rng.standard_normal(n_embd) * 0.1).astype(np.float32)
    if not tied:
        hf["lm_head.weight"] = mat(n_vocab, n_embd, 0.3)

    # GGUF names, values (out, in), and the type each is stored in.
    plain = quant in ("F32", "F16")
    big = quant
    gg: list[tuple[str, np.ndarray, str]] = [
        ("token_embd.weight", hf["model.embed_tokens.weight"], quant if plain else "F32")]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        q = hf[p + "self_attn.q_proj.weight"]
        k = hf[p + "self_attn.k_proj.weight"]
        if arch in ("llama", "mistral"):
            q = _permute_for_norm_rope(q, n_heads)
            k = _permute_for_norm_rope(k, n_kv_heads)
        b = f"blk.{i}."
        gg += [
            (b + "attn_norm.weight", hf[p + "input_layernorm.weight"], "F32"),
            (b + "attn_q.weight", q, big), (b + "attn_k.weight", k, big),
            (b + "attn_v.weight", hf[p + "self_attn.v_proj.weight"], big),
            (b + "attn_output.weight", hf[p + "self_attn.o_proj.weight"], big),
            (b + "ffn_norm.weight", hf[p + "post_attention_layernorm.weight"], "F32"),
            (b + "ffn_gate.weight", hf[p + "mlp.gate_proj.weight"], big),
            (b + "ffn_up.weight", hf[p + "mlp.up_proj.weight"], big),
            (b + "ffn_down.weight", hf[p + "mlp.down_proj.weight"], big),
        ]
        if arch == "qwen2":
            for name in ("q", "k", "v"):
                gg.append((b + f"attn_{name}.bias", hf[p + f"self_attn.{name}_proj.bias"], "F32"))
        if arch == "qwen3":
            gg.append((b + "attn_q_norm.weight", hf[p + "self_attn.q_norm.weight"], "F32"))
            gg.append((b + "attn_k_norm.weight", hf[p + "self_attn.k_norm.weight"], "F32"))
    gg.append(("output_norm.weight", hf["model.norm.weight"], "F32"))
    if not tied:
        gg.append(("output.weight", hf["lm_head.weight"], quant if plain else "Q8_0"
                   if n_embd % 32 == 0 else "F32"))

    writer = GGUFWriter(path)
    a = arch
    for key, value, kind in (
        ("general.architecture", arch, _STR), ("general.name", f"strife-fixture-{arch}", _STR),
        (f"{a}.block_count", n_layers, _U32), (f"{a}.embedding_length", n_embd, _U32),
        (f"{a}.attention.head_count", n_heads, _U32),
        (f"{a}.attention.head_count_kv", n_kv_heads, _U32),
        (f"{a}.feed_forward_length", n_ff, _U32), (f"{a}.context_length", context, _U32),
        (f"{a}.rope.freq_base", float(rope_base), _F32),
        (f"{a}.attention.layer_norm_rms_epsilon", 1e-6, _F32),
        ("tokenizer.ggml.model", tokenizer, _STR),
    ):
        writer.set_metadata(key, value, type_hint=(kind, None))
    writer.set_metadata("tokenizer.ggml.tokens", tokens, type_hint=(_ARR, _STR))
    writer.set_metadata("tokenizer.ggml.token_type", types, type_hint=(_ARR, _I32))
    if tokenizer == "gpt2":
        writer.set_metadata("tokenizer.ggml.pre", "qwen2", type_hint=(_STR, None))
        writer.set_metadata("tokenizer.ggml.merges", merges, type_hint=(_ARR, _STR))
        eos = tokens.index("<|im_end|>")
        writer.set_metadata("tokenizer.ggml.eos_token_id", eos, type_hint=(_U32, None))
        writer.set_metadata("tokenizer.ggml.bos_token_id", tokens.index("<|endoftext|>"),
                            type_hint=(_U32, None))
        writer.set_metadata("tokenizer.ggml.add_bos_token", False,
                            type_hint=(int(GGUFValueType.BOOL), None))
    else:
        writer.set_metadata("tokenizer.ggml.scores", scores, type_hint=(_ARR, _F32))
        writer.set_metadata("tokenizer.ggml.bos_token_id", 1, type_hint=(_U32, None))
        writer.set_metadata("tokenizer.ggml.eos_token_id", 2, type_hint=(_U32, None))
        writer.set_metadata("tokenizer.ggml.unknown_token_id", 0, type_hint=(_U32, None))
    if chat_template:
        writer.set_metadata("tokenizer.chat_template", chat_template, type_hint=(_STR, None))

    payload: dict[str, bytes] = {}
    for name, values, qtype in gg:
        shape = tuple(reversed(values.shape))  # GGUF ne: row length first
        writer.add_tensor(name, shape, QUANT_TYPES[qtype])
        payload[name] = _encode(values, qtype)
    writer.write(lambda t: payload[t.name])
    config = {"arch": arch, "n_layers": n_layers, "n_embd": n_embd, "n_heads": n_heads,
              "n_kv_heads": n_kv_heads, "n_ff": n_ff, "n_vocab": n_vocab, "head_dim": head_dim,
              "context": context, "rope_base": rope_base, "eps": 1e-6, "tied": tied,
              "tokens": tokens, "merges": merges}
    return {"path": str(path), "hf": hf, "config": config}


# ---------------------------------------------------------------------------
# 0.73.0: LFM3, Qwen3.5 (dense and MoE) and GLM-4.5, written the way
# llama.cpp's converter writes them (conversion/lfm3.py, qwen.py, glm.py).
# ---------------------------------------------------------------------------

_BOOL, _ARR_U32 = int(GGUFValueType.BOOL), (_ARR, _U32)


class _Writer:
    """Collect GGUF tensors and metadata, then write them in one go."""

    def __init__(self, path: str | Path, arch: str, quant: str) -> None:
        self.path, self.arch, self.quant = Path(path), arch, quant
        self.meta: list[tuple[str, Any, tuple[int, int | None]]] = [
            ("general.architecture", arch, (_STR, None)),
            ("general.name", f"strife-fixture-{arch}", (_STR, None))]
        self.tensors: list[tuple[str, np.ndarray, str]] = []

    def m(self, key: str, value: Any, kind: int = _U32, item: int | None = None) -> None:
        name = key if key.startswith(("general.", "tokenizer.")) else f"{self.arch}.{key}"
        self.meta.append((name, value, (kind, item)))

    def t(self, name: str, values: np.ndarray, *, big: bool = False) -> None:
        """*values* in numpy layout (rows = output); *big* matrices take the quant."""
        kind = self.quant if big and values.ndim >= 2 else "F32"
        if kind not in ("F32", "F16") and values.shape[-1] % 256 and kind not in ("Q8_0", "Q4_0"):
            kind = "F32"
        if kind in ("Q8_0", "Q4_0") and values.shape[-1] % 32:
            kind = "F32"
        self.tensors.append((name, np.ascontiguousarray(values, dtype=np.float32), kind))

    def write(self, tokenizer: str, chat_template: str | None, pre: str) -> None:
        tokens, types, merges = byte_bpe_vocab()
        w = GGUFWriter(self.path)
        for key, value, hint in self.meta:
            w.set_metadata(key, value, type_hint=hint)
        w.set_metadata("tokenizer.ggml.model", "gpt2", type_hint=(_STR, None))
        w.set_metadata("tokenizer.ggml.pre", pre, type_hint=(_STR, None))
        w.set_metadata("tokenizer.ggml.tokens", tokens, type_hint=(_ARR, _STR))
        w.set_metadata("tokenizer.ggml.token_type", types, type_hint=(_ARR, _I32))
        w.set_metadata("tokenizer.ggml.merges", merges, type_hint=(_ARR, _STR))
        w.set_metadata("tokenizer.ggml.eos_token_id", tokens.index("<|im_end|>"), type_hint=(_U32, None))
        w.set_metadata("tokenizer.ggml.bos_token_id", tokens.index("<|endoftext|>"), type_hint=(_U32, None))
        w.set_metadata("tokenizer.ggml.add_bos_token", False, type_hint=(_BOOL, None))
        if chat_template:
            w.set_metadata("tokenizer.chat_template", chat_template, type_hint=(_STR, None))
        payload = {}
        for name, values, kind in self.tensors:
            w.add_tensor(name, tuple(reversed(values.shape)), QUANT_TYPES[kind])
            payload[name] = _encode(values, kind)
        w.write(lambda t: payload[t.name])

    @property
    def n_vocab(self) -> int:
        return len(byte_bpe_vocab()[0])


def _rand(rng: np.random.Generator, *shape: int, scale: float = 0.08) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _norm_w(rng: np.random.Generator, n: int) -> np.ndarray:
    return (1.0 + rng.standard_normal(n) * 0.1).astype(np.float32)


def _reorder_v_heads(x: np.ndarray, axis: int, n_k: int, per_k: int, head: int) -> np.ndarray:
    """llama.cpp's ``_reorder_v_heads``: grouped value heads to tiled order."""
    x = np.moveaxis(x, axis, 0)
    rest = x.shape[1:]
    x = x.reshape(n_k, per_k, head, *rest).swapaxes(0, 1).reshape(n_k * per_k * head, *rest)
    return np.moveaxis(x, 0, axis)


def _make_lfm3(path: str | Path, *, quant: str = "F32", seed: int = 0,
               chat_template: str | None = None) -> dict[str, Any]:
    """LFM3 in Liquid's own layout (``model.layers.N.operator_norm`` ...)."""
    rng = np.random.default_rng(seed)
    w = _Writer(path, "lfm3", quant)
    E, H, KV, D, FF, K, window = 64, 4, 2, 16, 96, 7, 4
    kinds = ["conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention"]
    V = w.n_vocab
    hf: dict[str, np.ndarray] = {"model.embed_tokens.weight": _rand(rng, V, E, scale=0.5),
                                 "model.embedding_norm.weight": _norm_w(rng, E)}
    for i, kind in enumerate(kinds):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        for hn, gn in (("operator_norm", "attn_norm"), ("ffn_norm", "ffn_norm"),
                       ("post_operator_norm", "post_attention_norm"), ("post_ffn_norm", "post_ffw_norm")):
            hf[p + hn + ".weight"] = _norm_w(rng, E)
            w.t(b + gn + ".weight", hf[p + hn + ".weight"])
        for hn, gn, shape in (("w1", "ffn_gate", (FF, E)), ("w3", "ffn_up", (FF, E)),
                              ("w2", "ffn_down", (E, FF))):
            hf[p + f"feed_forward.{hn}.weight"] = _rand(rng, *shape)
            w.t(b + gn + ".weight", hf[p + f"feed_forward.{hn}.weight"], big=True)
        if kind == "conv":
            hf[p + "conv.in_proj.weight"] = _rand(rng, 3 * E, E)
            hf[p + "conv.conv.weight"] = _rand(rng, E, 1, K, scale=0.3)
            hf[p + "conv.out_proj.weight"] = _rand(rng, E, E)
            w.t(b + "shortconv.in_proj.weight", hf[p + "conv.in_proj.weight"], big=True)
            w.t(b + "shortconv.conv.weight", hf[p + "conv.conv.weight"][:, 0, :])
            w.t(b + "shortconv.out_proj.weight", hf[p + "conv.out_proj.weight"], big=True)
            continue
        a = p + "self_attn."
        hf[a + "q_proj.weight"] = _rand(rng, H * D, E)
        hf[a + "k_proj.weight"] = _rand(rng, KV * D, E)
        hf[a + "v_proj.weight"] = _rand(rng, KV * D, E)
        hf[a + "o_proj.weight"] = _rand(rng, E, H * D)
        hf[a + "q_norm.weight"] = _norm_w(rng, D)
        hf[a + "k_norm.weight"] = _norm_w(rng, D)
        hf[a + "gate_proj.weight"] = _rand(rng, H, E, scale=0.3)
        for hn, gn in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                       ("o_proj", "attn_output"), ("gate_proj", "attn_gate")):
            w.t(b + gn + ".weight", hf[a + hn + ".weight"], big=hn != "gate_proj")
        w.t(b + "attn_q_norm.weight", hf[a + "q_norm.weight"])
        w.t(b + "attn_k_norm.weight", hf[a + "k_norm.weight"])
    w.t("output_norm.weight", hf["model.embedding_norm.weight"])
    w.t("token_embd.weight", hf["model.embed_tokens.weight"])
    rope = np.ones(D // 2, dtype=np.float32)
    rope[int(0.5 * D) // 2:] = 1e30  # zero tail: half the head rotates
    w.t("rope_freqs.weight", rope)
    n = len(kinds)
    w.m("block_count", n)
    w.m("context_length", 256)
    w.m("embedding_length", E)
    w.m("feed_forward_length", FF)
    w.m("attention.head_count", H)
    w.m("attention.head_count_kv", [0 if k == "conv" else KV for k in kinds], _ARR, _U32)
    w.m("attention.key_length", D)
    w.m("attention.value_length", D)
    w.m("rope.dimension_count", D)
    w.m("rope.freq_base", 1e6, _F32)
    w.m("rope.freq_base_swa", 1e4, _F32)
    w.m("attention.layer_norm_rms_epsilon", 1e-5, _F32)
    w.m("attention.sliding_window", window)
    w.m("attention.sliding_window_pattern", [k == "sliding_attention" for k in kinds], _ARR, _BOOL)
    w.m("shortconv.l_cache", K)
    w.write("gpt2", chat_template, "lfm2")
    return {"path": str(path), "hf": hf, "config": {
        "arch": "lfm3", "layer_types": kinds, "hidden_size": E, "heads": H, "kv_heads": KV,
        "head_dim": D, "ff": FF, "kernel": K, "window": window, "eps": 1e-5, "vocab": V,
        "rope_base": 1e6, "rope_base_swa": 1e4, "partial": 0.5}}


def _make_qwen35(path: str | Path, *, quant: str = "F32", seed: int = 0, moe: bool = False,
                 fused: bool = False, chat_template: str | None = None) -> dict[str, Any]:
    """Qwen3.5 (or Qwen3.5-MoE) from transformers' layout, converted as llama.cpp does."""
    rng = np.random.default_rng(seed)
    arch = "qwen35moe" if moe else "qwen35"
    w = _Writer(path, arch, quant)
    E, H, KV, D, FF, L = 64, 4, 2, 32, 96, 4
    NK, NV, DK, DV, K = 2, 4, 16, 16, 4
    X, XFF, SFF, USED = 8, 32, 48, 2
    V = w.n_vocab
    key_dim, value_dim = NK * DK, NV * DV
    conv_dim = 2 * key_dim + value_dim
    per_k = NV // NK
    hf: dict[str, np.ndarray] = {"model.embed_tokens.weight": _rand(rng, V, E, scale=0.5),
                                 "model.norm.weight": _rand(rng, E, scale=0.1),
                                 "lm_head.weight": _rand(rng, V, E, scale=0.3)}
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    for i, kind in enumerate(layer_types):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        hf[p + "input_layernorm.weight"] = _rand(rng, E, scale=0.1)
        hf[p + "post_attention_layernorm.weight"] = _rand(rng, E, scale=0.1)
        w.t(b + "attn_norm.weight", hf[p + "input_layernorm.weight"] + 1)
        w.t(b + "post_attention_norm.weight", hf[p + "post_attention_layernorm.weight"] + 1)
        if kind == "linear_attention":
            la = p + "linear_attn."
            hf[la + "in_proj_qkv.weight"] = _rand(rng, conv_dim, E)
            hf[la + "in_proj_z.weight"] = _rand(rng, value_dim, E)
            hf[la + "in_proj_b.weight"] = _rand(rng, NV, E, scale=0.3)
            hf[la + "in_proj_a.weight"] = _rand(rng, NV, E, scale=0.3)
            hf[la + "A_log"] = (rng.standard_normal(NV) * 0.5).astype(np.float32)
            hf[la + "dt_bias"] = (rng.standard_normal(NV) * 0.5).astype(np.float32)
            hf[la + "conv1d.weight"] = _rand(rng, conv_dim, 1, K, scale=0.3)
            hf[la + "norm.weight"] = _norm_w(rng, DV)
            hf[la + "out_proj.weight"] = _rand(rng, E, value_dim)
            qkv = hf[la + "in_proj_qkv.weight"]
            qkv = np.concatenate([qkv[:2 * key_dim],
                                  _reorder_v_heads(qkv[2 * key_dim:], 0, NK, per_k, DV)])
            conv = hf[la + "conv1d.weight"][:, 0, :]
            conv = np.concatenate([conv[:2 * key_dim],
                                   _reorder_v_heads(conv[2 * key_dim:], 0, NK, per_k, DV)])
            w.t(b + "attn_qkv.weight", qkv, big=True)
            w.t(b + "attn_gate.weight", _reorder_v_heads(hf[la + "in_proj_z.weight"], 0, NK, per_k, DV), big=True)
            w.t(b + "ssm_beta.weight", _reorder_v_heads(hf[la + "in_proj_b.weight"], 0, NK, per_k, 1))
            w.t(b + "ssm_alpha.weight", _reorder_v_heads(hf[la + "in_proj_a.weight"], 0, NK, per_k, 1))
            w.t(b + "ssm_a", -np.exp(_reorder_v_heads(hf[la + "A_log"], 0, NK, per_k, 1)))
            w.t(b + "ssm_dt.bias", _reorder_v_heads(hf[la + "dt_bias"], 0, NK, per_k, 1))
            w.t(b + "ssm_conv1d.weight", conv)
            w.t(b + "ssm_norm.weight", hf[la + "norm.weight"])
            w.t(b + "ssm_out.weight", _reorder_v_heads(hf[la + "out_proj.weight"], 1, NK, per_k, DV), big=True)
        else:
            a = p + "self_attn."
            hf[a + "q_proj.weight"] = _rand(rng, 2 * H * D, E)
            hf[a + "k_proj.weight"] = _rand(rng, KV * D, E)
            hf[a + "v_proj.weight"] = _rand(rng, KV * D, E)
            hf[a + "o_proj.weight"] = _rand(rng, E, H * D)
            hf[a + "q_norm.weight"] = _rand(rng, D, scale=0.1)
            hf[a + "k_norm.weight"] = _rand(rng, D, scale=0.1)
            for hn, gn in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                           ("o_proj", "attn_output")):
                w.t(b + gn + ".weight", hf[a + hn + ".weight"], big=True)
            w.t(b + "attn_q_norm.weight", hf[a + "q_norm.weight"] + 1)
            w.t(b + "attn_k_norm.weight", hf[a + "k_norm.weight"] + 1)
        m = p + "mlp."
        if not moe:
            for hn, gn, shape in (("gate_proj", "ffn_gate", (FF, E)), ("up_proj", "ffn_up", (FF, E)),
                                  ("down_proj", "ffn_down", (E, FF))):
                hf[m + hn + ".weight"] = _rand(rng, *shape)
                w.t(b + gn + ".weight", hf[m + hn + ".weight"], big=True)
            continue
        hf[m + "gate.weight"] = _rand(rng, X, E, scale=0.5)
        hf[m + "experts.gate_up_proj"] = _rand(rng, X, 2 * XFF, E)
        hf[m + "experts.down_proj"] = _rand(rng, X, E, XFF)
        for hn, gn, shape in (("gate_proj", "ffn_gate_shexp", (SFF, E)), ("up_proj", "ffn_up_shexp", (SFF, E)),
                              ("down_proj", "ffn_down_shexp", (E, SFF))):
            hf[m + f"shared_expert.{hn}.weight"] = _rand(rng, *shape)
            w.t(b + gn + ".weight", hf[m + f"shared_expert.{hn}.weight"], big=True)
        hf[m + "shared_expert_gate.weight"] = _rand(rng, 1, E, scale=0.3)
        w.t(b + "ffn_gate_inp.weight", hf[m + "gate.weight"])
        gu = hf[m + "experts.gate_up_proj"]
        if fused:
            w.t(b + "ffn_gate_up_exps.weight", gu, big=True)
        else:
            w.t(b + "ffn_gate_exps.weight", gu[:, :XFF], big=True)
            w.t(b + "ffn_up_exps.weight", gu[:, XFF:], big=True)
        w.t(b + "ffn_down_exps.weight", hf[m + "experts.down_proj"], big=True)
        w.t(b + "ffn_gate_inp_shexp.weight", hf[m + "shared_expert_gate.weight"][0])
    w.t("token_embd.weight", hf["model.embed_tokens.weight"])
    w.t("output_norm.weight", hf["model.norm.weight"] + 1)
    w.t("output.weight", hf["lm_head.weight"], big=True)
    w.m("block_count", L)
    w.m("context_length", 256)
    w.m("embedding_length", E)
    if not moe:
        w.m("feed_forward_length", FF)
    w.m("attention.head_count", H)
    w.m("attention.head_count_kv", KV)
    w.m("attention.key_length", D)
    w.m("attention.value_length", D)
    w.m("rope.freq_base", 10000.0, _F32)
    w.m("rope.dimension_count", int(D * 0.25))
    w.m("rope.dimension_sections", [11, 11, 10, 0], _ARR, _I32)
    w.m("attention.layer_norm_rms_epsilon", 1e-6, _F32)
    w.m("ssm.conv_kernel", K)
    w.m("ssm.state_size", DK)
    w.m("ssm.group_count", NK)
    w.m("ssm.time_step_rank", NV)
    w.m("ssm.inner_size", value_dim)
    w.m("full_attention_interval", 4)
    if moe:
        w.m("expert_count", X)
        w.m("expert_used_count", USED)
        w.m("expert_feed_forward_length", XFF)
        w.m("expert_shared_feed_forward_length", SFF)
    w.write("gpt2", chat_template, "qwen35")
    return {"path": str(path), "hf": hf, "config": {
        "arch": arch, "vocab_size": V, "hidden_size": E, "intermediate_size": FF, "num_hidden_layers": L,
        "num_attention_heads": H, "num_key_value_heads": KV, "head_dim": D,
        "linear_num_key_heads": NK, "linear_num_value_heads": NV, "linear_key_head_dim": DK,
        "linear_value_head_dim": DV, "linear_conv_kernel_dim": K, "full_attention_interval": 4,
        "rms_norm_eps": 1e-6, "layer_types": layer_types,
        **({"moe_intermediate_size": XFF, "shared_expert_intermediate_size": SFF,
            "num_experts": X, "num_experts_per_tok": USED} if moe else {})}}


def _make_glm4moe(path: str | Path, *, quant: str = "F32", seed: int = 0, qk_norm: bool = True,
                  chat_template: str | None = None) -> dict[str, Any]:
    """GLM-4.5 from transformers' layout: one leading dense layer, then experts."""
    rng = np.random.default_rng(seed)
    w = _Writer(path, "glm4moe", quant)
    E, H, KV, D, FF, L = 64, 4, 2, 16, 96, 3
    X, XFF, USED, SCALE = 8, 32, 2, 2.5
    V = w.n_vocab
    hf: dict[str, np.ndarray] = {"model.embed_tokens.weight": _rand(rng, V, E, scale=0.5),
                                 "model.norm.weight": _norm_w(rng, E),
                                 "lm_head.weight": _rand(rng, V, E, scale=0.3)}
    for i in range(L):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        hf[p + "input_layernorm.weight"] = _norm_w(rng, E)
        hf[p + "post_attention_layernorm.weight"] = _norm_w(rng, E)
        w.t(b + "attn_norm.weight", hf[p + "input_layernorm.weight"])
        w.t(b + "post_attention_norm.weight", hf[p + "post_attention_layernorm.weight"])
        a = p + "self_attn."
        for hn, gn, rows in (("q_proj", "attn_q", H * D), ("k_proj", "attn_k", KV * D),
                             ("v_proj", "attn_v", KV * D)):
            hf[a + hn + ".weight"] = _rand(rng, rows, E)
            hf[a + hn + ".bias"] = _rand(rng, rows, scale=0.05)
            w.t(b + gn + ".weight", hf[a + hn + ".weight"], big=True)
            w.t(b + gn + ".bias", hf[a + hn + ".bias"])
        hf[a + "o_proj.weight"] = _rand(rng, E, H * D)
        w.t(b + "attn_output.weight", hf[a + "o_proj.weight"], big=True)
        if qk_norm:
            hf[a + "q_norm.weight"] = _norm_w(rng, D)
            hf[a + "k_norm.weight"] = _norm_w(rng, D)
            w.t(b + "attn_q_norm.weight", hf[a + "q_norm.weight"])
            w.t(b + "attn_k_norm.weight", hf[a + "k_norm.weight"])
        m = p + "mlp."
        if i == 0:
            for hn, gn, shape in (("gate_proj", "ffn_gate", (FF, E)), ("up_proj", "ffn_up", (FF, E)),
                                  ("down_proj", "ffn_down", (E, FF))):
                hf[m + hn + ".weight"] = _rand(rng, *shape)
                w.t(b + gn + ".weight", hf[m + hn + ".weight"], big=True)
            continue
        hf[m + "gate.weight"] = _rand(rng, X, E, scale=0.5)
        hf[m + "gate.e_score_correction_bias"] = _rand(rng, X, scale=0.2)
        hf[m + "experts.gate_up_proj"] = _rand(rng, X, 2 * XFF, E)
        hf[m + "experts.down_proj"] = _rand(rng, X, E, XFF)
        for hn, gn, shape in (("gate_proj", "ffn_gate_shexp", (XFF, E)), ("up_proj", "ffn_up_shexp", (XFF, E)),
                              ("down_proj", "ffn_down_shexp", (E, XFF))):
            hf[m + f"shared_experts.{hn}.weight"] = _rand(rng, *shape)
            w.t(b + gn + ".weight", hf[m + f"shared_experts.{hn}.weight"], big=True)
        w.t(b + "ffn_gate_inp.weight", hf[m + "gate.weight"])
        w.t(b + "exp_probs_b.bias", hf[m + "gate.e_score_correction_bias"])
        gu = hf[m + "experts.gate_up_proj"]
        w.t(b + "ffn_gate_exps.weight", gu[:, :XFF], big=True)
        w.t(b + "ffn_up_exps.weight", gu[:, XFF:], big=True)
        w.t(b + "ffn_down_exps.weight", hf[m + "experts.down_proj"], big=True)
    w.t("token_embd.weight", hf["model.embed_tokens.weight"])
    w.t("output_norm.weight", hf["model.norm.weight"])
    w.t("output.weight", hf["lm_head.weight"], big=True)
    w.m("block_count", L)
    w.m("context_length", 256)
    w.m("embedding_length", E)
    w.m("feed_forward_length", FF)
    w.m("attention.head_count", H)
    w.m("attention.head_count_kv", KV)
    w.m("attention.key_length", D)
    w.m("attention.value_length", D)
    w.m("rope.freq_base", 10000.0, _F32)
    w.m("rope.dimension_count", int(D * 0.5))
    w.m("attention.layer_norm_rms_epsilon", 1e-5, _F32)
    w.m("expert_count", X)
    w.m("expert_used_count", USED)
    w.m("expert_feed_forward_length", XFF)
    w.m("expert_shared_count", 1)
    w.m("leading_dense_block_count", 1)
    w.m("expert_weights_scale", SCALE, _F32)
    w.m("expert_weights_norm", True, _BOOL)
    w.m("expert_gating_func", 2)
    w.write("gpt2", chat_template, "glm4")
    return {"path": str(path), "hf": hf, "config": {
        "arch": "glm4moe", "vocab_size": V, "hidden_size": E, "intermediate_size": FF,
        "moe_intermediate_size": XFF, "n_routed_experts": X, "num_experts_per_tok": USED,
        "n_shared_experts": 1, "first_k_dense_replace": 1, "num_hidden_layers": L,
        "num_attention_heads": H, "num_key_value_heads": KV, "head_dim": D, "attention_bias": True,
        "use_qk_norm": qk_norm, "routed_scaling_factor": SCALE, "norm_topk_prob": True,
        "rms_norm_eps": 1e-5}}


_HYBRID = {"lfm3": _make_lfm3, "qwen35": _make_qwen35,
           "qwen35moe": lambda path, **kw: _make_qwen35(path, moe=True, **kw),
           "glm4moe": _make_glm4moe}
