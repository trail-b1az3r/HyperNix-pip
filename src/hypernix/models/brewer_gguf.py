"""hyperNix0x-v2 (Brewer) models as real GGUF files llama.cpp runs.

Brewer's ``export(fmt="gguf")`` used to write a file with the magic
``HNXG``, a JSON header and raw F32 tensors: GGUF in name only. No
llama.cpp could open it, the HyperNix-patched one included, so a brewed
model could run in Python and nowhere else. This writes an actual GGUF.

Why it is exported as ``llama``
-------------------------------
A Brewer block is a Llama block: RMSNorm before attention and before the
FFN, grouped-query attention with RoPE, a SwiGLU FFN, no biases, a final
RMSNorm and a (usually tied) LM head. llama.cpp's ``llama`` architecture
computes exactly that, and every build has it -- stock, LM Studio's, and
the one ``native/ggml-hnx`` patches. A new architecture name would need a
new llama.cpp build to open the file at all, which is the problem this
exists to remove.

Three things differ, and each is handled here rather than approximated:

* **RoPE layout.** Brewer rotates the two halves of each head
  (``rotate_half``, the NeoX / Hugging Face layout); llama.cpp's llama
  rotates adjacent pairs. The Q and K projections are permuted so the
  same rotation lands on the same dimensions -- the transform llama.cpp's
  own ``convert_hf_to_gguf.py`` applies to every Llama checkpoint.
* **Sliding-window attention** on odd layers. The ``llama`` architecture
  has none, so beyond the window its odd layers would see further back
  than Brewer's. Up to the window the two are identical, so by default
  the exported context length is capped at the window: exact, never
  "close". ``context="full"`` keeps the trained length and says what that
  costs.
* **The tokenizer** is not in the weights. It comes from the model's
  ``tokenizer.json`` (a byte-level BPE, as HyperNix.3-mini uses) or, for a
  model ``brew train`` trained on characters, from the ``char_vocab.json``
  it now saves, rebuilt as a byte-level vocabulary with the merges that put
  multi-byte characters back together.

Improvements over a plain conversion
------------------------------------
* The FFN is zero-padded to a multiple of 256 (``pad_ffn_to``).
  HyperNix.3-mini's ``d_ff`` is 2203, and a tensor row that is not a
  multiple of the block size cannot be k-quantised, so every
  ``ffn_down`` fell back to F16 in ``llama-quantize``. Zero rows in gate
  and up and zero columns in down contribute exactly nothing
  (``silu(0) * 0 = 0``), so the outputs do not change.
* ``outtype`` of f32, f16, bf16 or q8_0, norms always F32.
* The Brewer config is kept under ``hypernix.brewer.*``, so the file
  says where it came from.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CHAR_VOCAB_FILE",
    "ExportReport",
    "GGUFExportError",
    "OUTTYPES",
    "brewer_to_llama_name",
    "export_gguf",
    "load_source",
    "read_char_vocab",
    "write_char_vocab",
]

#: Where ``brew train`` keeps a character-level model's vocabulary. Not
#: ``vocab.json``: with a ``merges.txt`` beside it that name means a
#: Hugging Face BPE, and loaders look for it.
CHAR_VOCAB_FILE = "char_vocab.json"


def write_char_vocab(path: str | Path, chars: list[str]) -> Path:
    out = Path(path)
    out.write_text(json.dumps({"type": "char", "chars": list(chars)}, ensure_ascii=False),
                   encoding="utf-8")
    return out


def read_char_vocab(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "char" or not isinstance(data.get("chars"), list):
        raise GGUFExportError(f"{path} is not a character vocabulary.")
    return [str(c) for c in data["chars"]]

#: Output tensor types. Norms and other 1-D tensors are always F32.
OUTTYPES = ("f32", "f16", "bf16", "q8_0")

#: Tensor rows a padded FFN is rounded up to: the k-quant super-block.
DEFAULT_FFN_MULTIPLE = 256


class GGUFExportError(ValueError):
    """The model or its tokenizer cannot be written as a runnable GGUF."""


@dataclass
class ExportReport:
    path: str
    outtype: str
    context_length: int
    tokenizer: str
    n_tensors: int = 0
    n_vocab: int = 0
    n_ff: int = 0
    n_ff_trained: int = 0
    bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        size = f"{self.bytes / 1e6:.1f} MB" if self.bytes else "?"
        ff = (f"n_ff {self.n_ff} (padded from {self.n_ff_trained})"
              if self.n_ff != self.n_ff_trained else f"n_ff {self.n_ff}")
        return (f"{self.path}: llama, {self.outtype}, {self.n_tensors} tensors, "
                f"{size}, ctx {self.context_length}, vocab {self.n_vocab} "
                f"({self.tokenizer}), {ff}")


# ---------------------------------------------------------------------------
# Reading a Brewer model
# ---------------------------------------------------------------------------

_WEIGHT_FILES = ("model.safetensors", "model.pt", "pytorch_model.bin", "weights.pt")


def _strip_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Drop a wrapper prefix (``_orig_mod.`` from torch.compile, ``model.``)."""
    anchor = next((k for k in state if k.endswith("embed.embed.weight")), None)
    if anchor is None:
        raise GGUFExportError("Not a hyperNix0x-v2 state dict: no embed.embed.weight.")
    prefix = anchor[: -len("embed.embed.weight")]
    if not prefix:
        return state
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_source(source: str | Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """``(state_dict, BrewerConfig, extras)`` from a model folder or a ``.pt``.

    *extras* carries what the file had beyond weights: ``char_vocab`` from
    ``brew train``, and ``folder`` for finding a tokenizer beside it.
    """
    from ..training.brewer import BrewerConfig

    path = Path(source).expanduser()
    extras: dict[str, Any] = {}
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise GGUFExportError(f"No config.json in {path}; is it a Brewer model folder?")
        config = BrewerConfig.load(config_path)
        weights = next((path / n for n in _WEIGHT_FILES if (path / n).is_file()), None)
        if weights is None:
            raise GGUFExportError(
                f"{path} has a config.json but no weights ({', '.join(_WEIGHT_FILES)}): "
                f"there is nothing to run yet. Train it first."
            )
        if weights.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(weights), device="cpu")
        else:
            from ..security.safeload import load_checkpoint

            loaded = load_checkpoint(weights)
            state = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
            if isinstance(loaded, dict) and loaded.get("char_vocab"):
                extras["char_vocab"] = list(loaded["char_vocab"])
        vocab_file = path / CHAR_VOCAB_FILE
        if "char_vocab" not in extras and vocab_file.is_file():
            extras["char_vocab"] = read_char_vocab(vocab_file)
        extras["folder"] = path
    elif path.is_file():
        from ..security.safeload import load_checkpoint

        payload = load_checkpoint(path)
        if not isinstance(payload, dict) or "model_state_dict" not in payload or "config" not in payload:
            raise GGUFExportError(
                f"{path} is not a hyperNix0x-v2 checkpoint (needs 'config' and "
                f"'model_state_dict')."
            )
        config = BrewerConfig.from_dict(payload["config"])
        state = payload["model_state_dict"]
        if payload.get("char_vocab"):
            extras["char_vocab"] = list(payload["char_vocab"])
        extras["folder"] = path.parent
    else:
        raise GGUFExportError(f"{path} does not exist.")
    return _strip_prefix(dict(state)), config, extras


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------

_BLOCK_MAP = {
    "norm1.weight": "attn_norm.weight",
    "attn.q_proj.weight": "attn_q.weight",
    "attn.k_proj.weight": "attn_k.weight",
    "attn.v_proj.weight": "attn_v.weight",
    "attn.o_proj.weight": "attn_output.weight",
    "norm2.weight": "ffn_norm.weight",
    "ffn.gate_proj.weight": "ffn_gate.weight",
    "ffn.up_proj.weight": "ffn_up.weight",
    "ffn.down_proj.weight": "ffn_down.weight",
}
_TOP_MAP = {
    "embed.embed.weight": "token_embd.weight",
    "norm_out.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def brewer_to_llama_name(name: str) -> str | None:
    """The llama.cpp tensor name for a Brewer one, or ``None`` if it has none."""
    if name in _TOP_MAP:
        return _TOP_MAP[name]
    if name.startswith("blocks."):
        _, index, rest = name.split(".", 2)
        mapped = _BLOCK_MAP.get(rest)
        return f"blk.{int(index)}.{mapped}" if mapped else None
    return None


def _permute(weight, n_head: int):
    """Hugging Face / NeoX half-rotation layout -> llama.cpp pair layout."""
    return (weight.reshape(n_head, 2, weight.shape[0] // n_head // 2, *weight.shape[1:])
            .swapaxes(1, 2)
            .reshape(weight.shape))


def _padded(n: int, multiple: int) -> int:
    return n if multiple <= 1 or n % multiple == 0 else n + multiple - n % multiple


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable character table, as tokenizers uses."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


@dataclass
class _Vocab:
    kind: str
    tokens: list[str]
    types: list[int]
    merges: list[str]
    pre: str
    bos: int | None = None
    eos: int | None = None
    pad: int | None = None
    add_bos: bool = False
    notes: list[str] = field(default_factory=list)


def _char_vocab(chars: list[str]) -> _Vocab:
    """A character vocabulary as a byte-level BPE llama.cpp's gpt2 tokenizer runs.

    Each character becomes its UTF-8 bytes in GPT-2's byte alphabet, and a
    multi-byte character gets the merges that rebuild it from its bytes.
    """
    from gguf import TokenType

    table = _bytes_to_unicode()
    tokens: list[str] = []
    merges: list[str] = []
    seen: set[str] = set()
    for ch in chars:
        symbols = [table[b] for b in ch.encode("utf-8")]
        tokens.append("".join(symbols))
        left = symbols[0]
        for right in symbols[1:]:
            merge = f"{left} {right}"
            if merge not in seen:
                seen.add(merge)
                merges.append(merge)
            left += right
    if len(set(tokens)) != len(tokens):
        raise GGUFExportError("The character vocabulary has duplicate entries.")
    return _Vocab("char", tokens, [int(TokenType.NORMAL)] * len(tokens), merges, "gpt-2")


def _special_id(value: Any, by_content: dict[str, int]) -> int | None:
    if isinstance(value, dict):
        value = value.get("content")
    if isinstance(value, str):
        return by_content.get(value)
    return None


def _pre_for(tok: dict[str, Any]) -> tuple[str, list[str]]:
    """The llama.cpp pre-tokenizer name for a tokenizers pre_tokenizer."""
    pre = tok.get("pre_tokenizer") or {}
    kind = pre.get("type")
    if kind == "ByteLevel" and pre.get("use_regex", True):
        return "gpt-2", []
    if kind == "Sequence":
        parts = pre.get("pretokenizers") or []
        regexes = [p.get("pattern", {}).get("Regex", "") for p in parts if p.get("type") == "Split"]
        joined = " ".join(regexes)
        if "(?i:'s|'t|'re|'ve|'m|'ll|'d)" in joined and r"\p{N}{1,3}" in joined:
            return "llama-bpe", []
        if r"\p{N}" in joined and "(?i:'s|'t|'re|'ve|'m|'ll|'d)" in joined:
            return "qwen2", []
    return "default", [
        f"pre-tokenizer {kind or 'none'} has no exact llama.cpp equivalent; using "
        f"'default', so a few strings may split differently than in Python"
    ]


def _json_vocab(folder: Path) -> _Vocab:
    """A byte-level BPE ``tokenizer.json`` as llama.cpp's gpt2 vocabulary."""
    from gguf import TokenType

    tok = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    model = tok.get("model") or {}
    if model.get("type") != "BPE":
        raise GGUFExportError(
            f"{folder / 'tokenizer.json'} is a {model.get('type')} tokenizer; only byte-level "
            f"BPE is supported (what `brew` and HyperNix.3-mini use)."
        )
    if model.get("byte_fallback"):
        raise GGUFExportError(
            "This tokenizer is a SentencePiece-style BPE with byte fallback, which needs "
            "the 'llama' tokenizer and scores that tokenizer.json does not carry."
        )
    vocab: dict[str, int] = dict(model.get("vocab") or {})
    added = {a["content"]: a for a in tok.get("added_tokens") or []}
    for content, entry in added.items():
        vocab.setdefault(content, int(entry["id"]))
    size = max(vocab.values()) + 1 if vocab else 0
    tokens = [f"[PAD{i}]" for i in range(size)]
    types = [int(TokenType.UNUSED)] * size
    for content, idx in vocab.items():
        tokens[idx] = content
        entry = added.get(content)
        if entry is None:
            types[idx] = int(TokenType.NORMAL)
        else:
            types[idx] = int(TokenType.CONTROL if entry.get("special") else TokenType.USER_DEFINED)
    raw_merges = model.get("merges") or []
    merges = [m if isinstance(m, str) else " ".join(m) for m in raw_merges]
    pre, notes = _pre_for(tok)

    config: dict[str, Any] = {}
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        path = folder / name
        if path.is_file():
            config.update(json.loads(path.read_text(encoding="utf-8")))
    by_content = {t: i for i, t in enumerate(tokens)}
    add_bos = bool(config.get("add_bos_token", False))
    post = tok.get("post_processor") or {}
    if post.get("type") == "TemplateProcessing":
        first = (post.get("single") or [{}])[0]
        special = first.get("SpecialToken", {}).get("id")
        if special and special == (config.get("bos_token") if isinstance(config.get("bos_token"), str)
                                   else (config.get("bos_token") or {}).get("content")):
            add_bos = True
    return _Vocab(
        "tokenizer.json", tokens, types, merges, pre,
        bos=_special_id(config.get("bos_token"), by_content),
        eos=_special_id(config.get("eos_token"), by_content),
        pad=_special_id(config.get("pad_token"), by_content),
        add_bos=add_bos, notes=notes,
    )


def _find_vocab(extras: dict[str, Any], tokenizer: str | Path | None) -> _Vocab:
    if tokenizer is not None:
        path = Path(tokenizer).expanduser()
        folder = path if path.is_dir() else path.parent
        if path.is_file() and path.name == CHAR_VOCAB_FILE:
            return _char_vocab(read_char_vocab(path))
        if not (folder / "tokenizer.json").is_file():
            raise GGUFExportError(f"No tokenizer.json at {path}.")
        return _json_vocab(folder)
    folder = extras.get("folder")
    if folder is not None and (Path(folder) / "tokenizer.json").is_file():
        return _json_vocab(Path(folder))
    if extras.get("char_vocab"):
        return _char_vocab(extras["char_vocab"])
    raise GGUFExportError(
        "No tokenizer: llama.cpp needs one inside the GGUF. Put the model's "
        "tokenizer.json beside it, pass tokenizer=, or retrain with `brew train`, "
        "which now saves its character vocabulary as char_vocab.json."
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def export_gguf(
    source: str | Path | None = None,
    out_path: str | Path | None = None,
    *,
    state_dict: dict[str, Any] | None = None,
    config: Any = None,
    char_vocab: list[str] | None = None,
    tokenizer: str | Path | None = None,
    outtype: str = "f16",
    context: str | int = "auto",
    pad_ffn_to: int = DEFAULT_FFN_MULTIPLE,
    name: str | None = None,
) -> ExportReport:
    """Write a Brewer model as a GGUF llama.cpp loads as ``llama``.

    Give either *source* (a model folder or a ``.pt`` checkpoint) or
    *state_dict* with *config*. *out_path* defaults to
    ``<folder>/<name>.<outtype>.gguf``.
    """
    import gguf
    import numpy as np
    import torch

    outtype = outtype.lower()
    if outtype not in OUTTYPES:
        raise GGUFExportError(f"outtype must be one of {', '.join(OUTTYPES)}, not {outtype!r}.")

    extras: dict[str, Any] = {}
    if state_dict is None:
        if source is None:
            raise GGUFExportError("Give a source folder or checkpoint, or state_dict and config.")
        state_dict, config, extras = load_source(source)
    else:
        if config is None:
            raise GGUFExportError("state_dict needs its config.")
        state_dict = _strip_prefix(dict(state_dict))
        if source is not None:
            extras["folder"] = Path(source)
    if char_vocab:
        extras["char_vocab"] = list(char_vocab)
    vocab = _find_vocab(extras, tokenizer)

    cfg = config
    n_head, n_kv = int(cfg.n_heads), int(cfg.n_kv_heads)
    head_dim = int(cfg.d_model) // n_head
    notes = list(vocab.notes)

    # The vocabulary must be exactly the embedding's rows.
    n_vocab = int(state_dict["embed.embed.weight"].shape[0])
    if len(vocab.tokens) > n_vocab:
        raise GGUFExportError(
            f"The tokenizer has {len(vocab.tokens)} tokens but the model only {n_vocab} "
            f"embeddings: they are not a pair."
        )
    if len(vocab.tokens) < n_vocab:
        extra = n_vocab - len(vocab.tokens)
        vocab.tokens += [f"[PAD{i}]" for i in range(len(vocab.tokens), n_vocab)]
        vocab.types += [int(gguf.TokenType.UNUSED)] * extra
        if vocab.kind == "char":
            notes.append(f"{extra} embedding rows beyond the {n_vocab - extra} characters "
                         f"are unused tokens")

    # Context.
    trained = int(cfg.max_seq_len)
    windowed = bool(cfg.use_sliding_window) and int(cfg.n_layers) > 1
    window = int(cfg.sliding_window_size)
    if isinstance(context, int) or (isinstance(context, str) and context.isdigit()):
        ctx = int(context)
        if windowed and ctx > window:
            notes.append(f"context {ctx} is past the {window}-token window: odd layers attend "
                         f"further back than in training")
    elif context == "full":
        ctx = trained
        if windowed and trained > window:
            notes.append(f"context {trained} is past the {window}-token window: odd layers attend "
                         f"further back than in training")
    elif context == "auto":
        ctx = min(trained, window) if windowed else trained
        if windowed and trained > window:
            notes.append(f"context capped at the {window}-token sliding window, where llama "
                         f"attention is identical to Brewer's; context='full' keeps {trained}")
    else:
        raise GGUFExportError("context must be 'auto', 'full' or a number of tokens.")

    # FFN padding.
    n_ff_trained = int(state_dict["blocks.0.ffn.gate_proj.weight"].shape[0])
    n_ff = _padded(n_ff_trained, pad_ffn_to)

    model_name = name or getattr(cfg, "name", None) or "hypernix0x-v2"
    if out_path is None:
        base = Path(extras.get("folder") or Path.cwd())
        out_path = base / f"{model_name}.{outtype}.gguf"
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(out), "llama")
    writer.add_name(model_name)
    writer.add_type("model")
    writer.add_context_length(ctx)
    writer.add_embedding_length(int(cfg.d_model))
    writer.add_block_count(int(cfg.n_layers))
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv)
    writer.add_rope_dimension_count(head_dim)
    if head_dim * n_head != int(cfg.d_model):
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
    writer.add_rope_freq_base(float(cfg.rope_theta))
    writer.add_layer_norm_rms_eps(float(cfg.norm_eps))
    writer.add_vocab_size(n_vocab)
    file_type = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                 "bf16": gguf.LlamaFileType.MOSTLY_BF16, "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}
    writer.add_file_type(int(file_type[outtype]))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(vocab.pre)
    writer.add_token_list(vocab.tokens)
    writer.add_token_types(vocab.types)
    if vocab.merges:
        writer.add_token_merges(vocab.merges)
    for setter, value in ((writer.add_bos_token_id, vocab.bos), (writer.add_eos_token_id, vocab.eos),
                          (writer.add_pad_token_id, vocab.pad)):
        if value is not None:
            setter(int(value))
    writer.add_add_bos_token(vocab.add_bos)
    writer.add_add_eos_token(False)

    writer.add_string("hypernix.brewer.arch", "hypernix0x-v2")
    writer.add_string("hypernix.brewer.config", json.dumps(cfg.to_dict(), sort_keys=True))
    writer.add_uint32("hypernix.brewer.ffn_trained", n_ff_trained)
    writer.add_string("hypernix.brewer.tokenizer", vocab.kind)

    # Tensors.
    tied = bool(cfg.tie_embeddings) or "lm_head.weight" not in state_dict or (
        state_dict["lm_head.weight"].data_ptr() == state_dict["embed.embed.weight"].data_ptr()
        if hasattr(state_dict["lm_head.weight"], "data_ptr") else False
    )
    written = 0
    skipped: list[str] = []
    for key in sorted(state_dict, key=_tensor_order):
        target = brewer_to_llama_name(key)
        if target is None:
            skipped.append(key)
            continue
        if target == "output.weight" and tied:
            continue
        t = state_dict[key]
        t = t.detach().to("cpu", torch.float32) if hasattr(t, "detach") else torch.as_tensor(t, dtype=torch.float32)
        if target.endswith("attn_q.weight"):
            t = _permute(t, n_head)
        elif target.endswith("attn_k.weight"):
            t = _permute(t, n_kv)
        if n_ff != n_ff_trained:
            if target.endswith(("ffn_gate.weight", "ffn_up.weight")):
                t = torch.cat([t, t.new_zeros(n_ff - n_ff_trained, t.shape[1])], dim=0)
            elif target.endswith("ffn_down.weight"):
                t = torch.cat([t, t.new_zeros(t.shape[0], n_ff - n_ff_trained)], dim=1)
        data = t.contiguous().numpy()
        _add_tensor(writer, target, data, outtype, gguf, np)
        written += 1
    if skipped:
        notes.append(f"not exported (no llama.cpp counterpart): {', '.join(skipped[:5])}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = ExportReport(
        path=str(out), outtype=outtype, context_length=ctx, tokenizer=vocab.kind,
        n_tensors=written, n_vocab=n_vocab, n_ff=n_ff, n_ff_trained=n_ff_trained,
        bytes=out.stat().st_size, notes=notes,
    )
    logger.info("brewer_gguf: %s", report.summary())
    return report


def _tensor_order(key: str) -> tuple[int, int, str]:
    """Embedding, then blocks in order, then the output norm and head."""
    if key.startswith("blocks."):
        return (1, int(key.split(".")[1]), key)
    return (0 if key.startswith("embed.") else 2, 0, key)


def _add_tensor(writer, name: str, data, outtype: str, gguf, np) -> None:
    if data.ndim == 1 or outtype == "f32":
        writer.add_tensor(name, data.astype(np.float32))
        return
    if outtype == "f16":
        writer.add_tensor(name, data.astype(np.float16))
        return
    if outtype == "bf16":
        packed = gguf.quants.quantize(data.astype(np.float32), gguf.GGMLQuantizationType.BF16)
        writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.BF16)
        return
    # q8_0: rows must be whole blocks of 32; anything else stays F16.
    if data.shape[-1] % 32 == 0:
        packed = gguf.quants.quantize(data.astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
        writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    else:
        writer.add_tensor(name, data.astype(np.float16))
