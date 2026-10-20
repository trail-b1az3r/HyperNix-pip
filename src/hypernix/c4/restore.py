"""hypernix.c4.restore — putting pieces back, and writing a model out.

Testing a piece means changing it: zeroing a stem, replacing a Pdial with
its mean, adding noise to an mdial. Everything here exists so that is
never a one-way door:

* :class:`Snapshot` copies the pieces about to change (to the CPU, so a
  snapshot of a large stem does not double its GPU memory) and puts them
  back exactly;
* :func:`ablated` is a context manager that changes a piece for the
  length of a ``with`` block and restores it however the block ends;
* :func:`write` writes the model as safetensors (with its config.json
  and tokenizer files), a ``.pt`` state dict, or a GGUF.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .explode import Dial, Exploded, MDial, MStem, PDial, Stem, address
from .model import C4Error, C4Model

__all__ = ["ABLATIONS", "Snapshot", "ablated", "apply_ablation", "write"]

#: Ways to take a piece out: ``zero`` it, set it to its ``mean``,
#: replace it with ``noise`` of its own spread, or ``shuffle`` it.
ABLATIONS = ("zero", "mean", "noise", "shuffle")

Piece = Dial | MDial | PDial | Stem | MStem | str


def _pieces(exploded: Exploded, piece: Piece) -> list[Dial | MDial | PDial | Stem]:
    if isinstance(piece, str):
        piece = exploded.mstems[int(piece.split(".")[1])] if piece.startswith("layer.") else address(piece, exploded)
    if isinstance(piece, MStem):
        return list(piece.stems)
    return [piece]


def _stem_name(piece: Dial | MDial | PDial | Stem) -> str:
    return piece.name if isinstance(piece, Stem) else piece.stem


@dataclass
class Snapshot:
    """Copies of the stems a change will touch, to put back exactly."""

    exploded: Exploded
    saved: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def take(cls, exploded: Exploded, pieces: Iterable[Piece]) -> Snapshot:
        snap = cls(exploded)
        for piece in pieces:
            for part in _pieces(exploded, piece):
                name = _stem_name(part)
                if name not in snap.saved:
                    snap.saved[name] = exploded.tensor(name).detach().to("cpu", copy=True)
        return snap

    def restore(self, names: Iterable[str] | None = None) -> list[str]:
        """Put the saved stems back. Returns which."""
        import torch

        done = []
        with torch.no_grad():
            for name in (list(names) if names is not None else list(self.saved)):
                tensor = self.exploded.tensor(name)
                tensor.copy_(self.saved[name].to(tensor.device, tensor.dtype))
                done.append(name)
        return done

    @property
    def bytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.saved.values()))


def apply_ablation(exploded: Exploded, piece: Dial | MDial | PDial | Stem, how: str = "zero",
                   *, seed: int = 0) -> None:
    """Take *piece* out of the model in place (see :data:`ABLATIONS`)."""
    import torch

    if how not in ABLATIONS:
        raise C4Error(f"ablate by one of {', '.join(ABLATIONS)}, not {how!r}")
    tensor = exploded.tensor(_stem_name(piece))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def replace(view: Any) -> Any:
        if how == "zero":
            return torch.zeros_like(view)
        if how == "mean":
            return torch.full_like(view, float(view.float().mean()))
        if how == "noise":
            std = float(view.float().std()) if view.numel() > 1 else float(view.abs().mean())
            return (torch.randn(view.shape, generator=generator) * std).to(view.device, view.dtype)
        order = torch.randperm(view.numel(), generator=generator).to(view.device)
        return view.reshape(-1)[order].reshape(view.shape)

    with torch.no_grad():
        if isinstance(piece, Stem):
            tensor.copy_(replace(tensor))
        elif isinstance(piece, Dial):
            tensor[piece.index] = 0.0 if how != "mean" else float(tensor.float().mean())
        elif isinstance(piece, MDial):
            for index in (piece.first, piece.second):
                tensor[index] = 0.0 if how != "mean" else float(tensor.float().mean())
        else:
            selector = piece.selector(tuple(tensor.shape))
            if piece.axis in ("block", "all"):
                flat = tensor.view(-1)
                flat[selector] = replace(flat[selector])
            else:
                tensor[selector] = replace(tensor[selector])


@contextmanager
def ablated(exploded: Exploded, pieces: Piece | Iterable[Piece], how: str = "zero",
            *, seed: int = 0) -> Iterator[Snapshot]:
    """Take *pieces* out for the length of the block, then put them back."""
    group = [pieces] if isinstance(pieces, (Dial, MDial, PDial, Stem, MStem, str)) else list(pieces)
    snap = Snapshot.take(exploded, group)
    try:
        for piece in group:
            for part in _pieces(exploded, piece):
                apply_ablation(exploded, part, how, seed=seed)
        yield snap
    finally:
        snap.restore()


_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                    "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt",
                    "added_tokens.json", "chat_template.jinja")


def write(model: C4Model, out: str | Path, *, format: str = "safetensors",
          tensors: dict[str, Any] | None = None, config: dict[str, Any] | None = None,
          source_folder: str | Path | None = None, dtype: Any = None) -> Path:
    """Write *model* (or *tensors* with *config*) to *out*.

    ``safetensors`` and ``pt`` write a folder with ``config.json`` and the
    tokenizer files copied from the source; ``gguf`` writes one file,
    float16, carrying the source GGUF's metadata when there is one.
    """
    import torch

    state = tensors if tensors is not None else model.tensors
    cfg = dict(config if config is not None else model.config)
    target = Path(out).expanduser()
    fmt = format.lower()
    if fmt in ("safetensors", "pt"):
        target.mkdir(parents=True, exist_ok=True)
        clean = {}
        for name, value in state.items():
            value = value.detach()
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
            clean[name] = value.contiguous().cpu()
        _untie(clean, cfg)
        if fmt == "safetensors":
            from safetensors.torch import save_file

            save_file(clean, str(target / "model.safetensors"), metadata={"format": "pt", "written_by": "hypernix.c4"})
        else:
            torch.save(clean, target / "pytorch_model.bin")
        (target / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        source = Path(source_folder) if source_folder else (model.path if model.path.is_dir() else model.path.parent)
        for name in _TOKENIZER_FILES:
            if (source / name).is_file() and not (target / name).exists():
                shutil.copy2(source / name, target / name)
        return target
    if fmt == "gguf":
        return _write_gguf(model, target, state, cfg)
    raise C4Error(f"write safetensors, pt or gguf, not {format!r}")


def _untie(state: dict[str, Any], config: dict[str, Any]) -> None:
    """safetensors refuses two names for one storage; a tied head is written once."""
    head, embed = state.get("lm_head.weight"), state.get("model.embed_tokens.weight")
    if head is not None and embed is not None and config.get("tie_word_embeddings") \
            and head.data_ptr() == embed.data_ptr():
        del state["lm_head.weight"]
    seen: dict[int, str] = {}
    for name in list(state):
        ptr = state[name].data_ptr()
        if ptr in seen:
            state[name] = state[name].clone()
        seen.setdefault(ptr, name)


def _write_gguf(model: C4Model, target: Path, state: dict[str, Any], config: dict[str, Any]) -> Path:
    from ..quant.gguf import GGMLType, GGUFWriter

    if any(name.startswith(("model.layers.", "model.embed_tokens")) for name in state):
        raise C4Error(
            "these tensors carry Hugging Face names, which no llama.cpp reads from a GGUF. "
            "Write safetensors and convert it: `hnx convert OUT -P -Q F16`."
        )
    if target.suffix.lower() != ".gguf":
        target = target.with_suffix(".gguf")
    writer = GGUFWriter(target)
    metadata = dict(model.metadata) if model.metadata else {
        "general.architecture": str(config.get("model_type") or "llama")}
    arch = str(metadata.get("general.architecture") or "llama")
    if config.get("num_hidden_layers") is not None:
        metadata[f"{arch}.block_count"] = int(config["num_hidden_layers"])
    if config.get("intermediate_size") is not None:
        metadata[f"{arch}.feed_forward_length"] = int(config["intermediate_size"])
    metadata["general.file_type"] = 1
    metadata.pop("general.alignment", None)
    for key, value in metadata.items():
        writer.set_metadata(key, value)
    payload = {}
    for name, value in state.items():
        array = value.detach().float().cpu().numpy()
        kind = GGMLType.F32 if array.ndim < 2 else GGMLType.F16
        writer.add_tensor(name, tuple(reversed(array.shape)), int(kind))
        payload[name] = array.astype("<f4" if kind == GGMLType.F32 else "<f2").tobytes()
    writer.write(lambda tensor: payload[tensor.name])
    return target
