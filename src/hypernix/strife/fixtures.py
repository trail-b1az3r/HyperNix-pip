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
