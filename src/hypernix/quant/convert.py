"""Convert a HyperNix PyTorch checkpoint to GGUF (fp32 or fp16)."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType
from tqdm import tqdm

from hypernix.models.arch import ArchInfo, infer_arch, iter_state_dict_names, map_tensor_name

_SUPPORTED_BASE = {"fp32", "f32", "fp16", "f16"}


def _collect_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load tensors from a HuggingFace-style snapshot directory.

    Supports safetensors (sharded or single), pytorch_model.bin (sharded or
    single), and a plain ``*.pt`` / ``*.pth`` checkpoint at the repo root.
    """
    state: dict[str, torch.Tensor] = {}

    st_index = model_dir / "model.safetensors.index.json"
    if st_index.exists():
        from safetensors.torch import load_file

        shard_map = json.loads(st_index.read_text())["weight_map"]
        for shard in sorted(set(shard_map.values())):
            state.update(load_file(str(model_dir / shard)))
        return state

    single_st = model_dir / "model.safetensors"
    if single_st.exists():
        from safetensors.torch import load_file

        state.update(load_file(str(single_st)))
        return state

    other_st = sorted(model_dir.glob("*.safetensors"))
    if other_st:
        from safetensors.torch import load_file

        for shard in other_st:
            state.update(load_file(str(shard)))
        return state

    bin_index = model_dir / "pytorch_model.bin.index.json"
    if bin_index.exists():
        shard_map = json.loads(bin_index.read_text())["weight_map"]
        for shard in sorted(set(shard_map.values())):
            state.update(torch.load(model_dir / shard, map_location="cpu", weights_only=True))
        return state

    single_bin = model_dir / "pytorch_model.bin"
    if single_bin.exists():
        state.update(torch.load(single_bin, map_location="cpu", weights_only=True))
        return state

    loose = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth")) + list(model_dir.glob("*.bin"))
    if loose:
        from ..security.safeload import load_checkpoint

        for path in sorted(loose):
            blob: Any = load_checkpoint(path)
            if isinstance(blob, dict):
                # Could be a bare state_dict or a checkpoint wrapper.
                inner = blob.get("state_dict") or blob.get("model") or blob
                if isinstance(inner, dict):
                    state.update(inner)
        if state:
            return state

    raise FileNotFoundError(
        f"No model weights found under {model_dir!s}. "
        "Expected *.safetensors, pytorch_model.bin*, or a loose *.pt/*.pth."
    )


def _load_config(model_dir: Path) -> dict[str, Any]:
    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _load_tokenizer_tokens(model_dir: Path) -> dict[str, Any] | None:
    """Best-effort extraction of vocab / merges so the GGUF is self-contained."""
    tok_json = model_dir / "tokenizer.json"
    if tok_json.exists():
        try:
            data = json.loads(tok_json.read_text())
        except json.JSONDecodeError:
            return None
        model = data.get("model") or {}
        vocab = model.get("vocab") or {}
        merges = model.get("merges") or []
        if vocab:
            tokens = [""] * len(vocab)
            for tok, idx in vocab.items():
                if 0 <= idx < len(tokens):
                    tokens[idx] = tok
            return {"tokens": tokens, "merges": merges, "kind": "bpe"}

    sp = model_dir / "tokenizer.model"
    if sp.exists():
        try:
            import sentencepiece as spm

            p = spm.SentencePieceProcessor()
            p.Load(str(sp))
            tokens = [p.IdToPiece(i) for i in range(p.GetPieceSize())]
            scores = [p.GetScore(i) for i in range(p.GetPieceSize())]
            token_types = [p.PieceType(i) for i in range(p.GetPieceSize())]
            return {
                "tokens": tokens,
                "scores": scores,
                "types": token_types,
                "kind": "spm",
            }
        except Exception:
            return None

    vocab_txt = model_dir / "vocab.txt"
    if vocab_txt.exists():
        tokens = [line.rstrip("\n") for line in vocab_txt.read_text().splitlines()]
        return {"tokens": tokens, "kind": "wordpiece"}

    return None


def _tensor_to_numpy(tensor: torch.Tensor, dtype: str) -> tuple[np.ndarray, GGMLQuantizationType]:
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous()
    if dtype in {"fp16", "f16"}:
        # Keep embeddings + output in F32 when F16 is requested (llama.cpp convention).
        return t.to(torch.float16).numpy(), GGMLQuantizationType.F16
    return t.to(torch.float32).numpy(), GGMLQuantizationType.F32


def _iter_named_tensors(state_dict: dict[str, torch.Tensor]) -> Iterable[tuple[str, str, torch.Tensor]]:
    for key in iter_state_dict_names(state_dict.keys()):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        mapped = map_tensor_name(key) or key
        yield key, mapped, tensor


def convert_to_gguf(
    model_dir: Path | str,
    output: Path | str,
    dtype: str = "fp16",
    arch_name: str = "hypernix",
    name: str = "HyperNix",
    n_head_hint: int | None = None,
    context_length: int | None = None,
) -> Path:
    """Write an uncompressed GGUF file at fp32 or fp16 precision.

    The writer is architecture-agnostic: it discovers the number of transformer
    blocks, the hidden size, and the FFN width from tensor shapes, so it works
    for any HyperNix checkpoint regardless of depth/width.

    Args:
        model_dir: Local HuggingFace snapshot directory.
        output: Destination GGUF path.
        dtype: ``"fp32"`` or ``"fp16"`` (use :func:`quantize_gguf` afterwards
            for k-quants).
        arch_name: GGUF architecture id written into the metadata.
        name: Model display name.
        n_head_hint: Explicit attention head count (overrides the heuristic).
        context_length: Override for the sequence-length metadata.
    """
    dtype = dtype.lower()
    if dtype not in _SUPPORTED_BASE:
        raise ValueError(
            f"convert_to_gguf only emits fp32/fp16; got {dtype!r}. Use quantize_gguf for k-quants."
        )

    model_dir = Path(model_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(model_dir)
    state = _collect_state_dict(model_dir)
    arch: ArchInfo = infer_arch(state, hint_n_head=n_head_hint or cfg.get("num_attention_heads"))

    n_layers = int(cfg.get("num_hidden_layers") or arch.n_layers)
    n_embd = int(cfg.get("hidden_size") or arch.n_embd)
    n_head = int(cfg.get("num_attention_heads") or arch.n_head or 1)
    n_head_kv = int(cfg.get("num_key_value_heads") or n_head)
    n_ff = int(cfg.get("intermediate_size") or arch.n_ff or (4 * n_embd))
    vocab_size = int(cfg.get("vocab_size") or arch.vocab_size or 0)
    ctx_len = int(context_length or cfg.get("max_position_embeddings") or cfg.get("n_positions") or 2048)
    rms_eps = float(cfg.get("rms_norm_eps") or cfg.get("layer_norm_epsilon") or 1e-5)
    rope_theta = float(cfg.get("rope_theta") or 10000.0)

    writer = GGUFWriter(str(output), arch_name)
    writer.add_name(name)
    writer.add_description(
        f"HyperNix custom architecture checkpoint converted to GGUF ({dtype})."
    )
    writer.add_context_length(ctx_len)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(rms_eps)
    writer.add_rope_freq_base(rope_theta)
    if vocab_size:
        writer.add_uint32("hypernix.vocab_size", vocab_size)
    writer.add_file_type(1 if dtype in {"fp16", "f16"} else 0)

    tok = _load_tokenizer_tokens(model_dir)
    if tok:
        kind = tok["kind"]
        writer.add_tokenizer_model(kind)
        # llama.cpp 2024+ requires `tokenizer.ggml.pre` on BPE GGUFs.
        # Without it, loaders (llama.cpp / LM Studio) fail with
        # "invalid GGUF type 9" on the merges field. "default" is the
        # safe catch-all pre-tokenizer identifier.
        if kind == "bpe":
            try:
                writer.add_tokenizer_pre("default")
            except AttributeError:
                writer.add_string("tokenizer.ggml.pre", "default")
        writer.add_token_list(tok["tokens"])
        if "scores" in tok:
            writer.add_token_scores(tok["scores"])
        if "types" in tok:
            writer.add_token_types(tok["types"])
        merges = tok.get("merges")
        if merges:
            # HF tokenizer.json v2+ stores merges as [["a","b"], ...];
            # llama.cpp wants a flat list of "a b" strings. Normalize so the
            # GGUF passes `gguf_init_from_file_impl` validation.
            if isinstance(merges[0], list | tuple):
                merges = [" ".join(m) for m in merges]
            writer.add_token_merges(merges)
        bos = cfg.get("bos_token_id")
        eos = cfg.get("eos_token_id")
        pad = cfg.get("pad_token_id")
        unk = cfg.get("unk_token_id")
        if isinstance(bos, int):
            writer.add_bos_token_id(bos)
        if isinstance(eos, int):
            writer.add_eos_token_id(eos)
        if isinstance(pad, int):
            writer.add_pad_token_id(pad)
        if isinstance(unk, int):
            writer.add_unk_token_id(unk)

    for _original, mapped, tensor in tqdm(list(_iter_named_tensors(state)), desc=f"writing {dtype} gguf"):
        # Token/output embeddings and norms stay in F32 even in F16 mode
        # so we preserve accuracy on the rarely-referenced tables.
        force_f32 = (
            mapped.endswith("token_embd.weight")
            or mapped.endswith("output.weight")
            or "_norm" in mapped
        )
        arr, gtype = _tensor_to_numpy(tensor, "fp32" if force_f32 and dtype in {"fp16", "f16"} else dtype)
        writer.add_tensor(mapped, arr, raw_dtype=gtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output
