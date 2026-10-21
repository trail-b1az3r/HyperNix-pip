"""Loading a model for c4: weights on the GPU, and what each tensor is.

c4 reads three kinds of checkpoint:

* **safetensors** — a folder (or one ``.safetensors`` file in it, or a
  sharded set with its ``model.safetensors.index.json``);
* **PyTorch** — ``pytorch_model.bin`` or a ``.pt`` state dict, loaded
  with :func:`hypernix.security.safeload.load_checkpoint`, which never
  runs code from the file unless the person has said to trust it;
* **GGUF** — any llama.cpp or HyperNix tensor type, dequantised to floats.

The first two need a ``config.json`` beside the weights: it is what says
how many layers, heads and channels there are, and without it a stem's
shape cannot be read as a connection. A GGUF carries the same facts in
its metadata.

Weights go to the person's GPU when there is one (CUDA, then Apple's
MPS), in float32 unless a ``dtype`` is asked for; ``device="cpu"`` keeps
them in RAM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "C4Error",
    "C4Model",
    "layer_of",
    "load",
    "pick_device",
    "role_of",
]


class C4Error(RuntimeError):
    """c4 could not do what was asked; the message says why."""


#: HF module names -> the role names GGUF uses, so one vocabulary covers both.
_HF_ROLES = (
    ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"), ("self_attn.qkv_proj", "attn_qkv"),
    ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down"),
    ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
    ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
    ("embed_tokens", "embed"), ("lm_head", "output"),
)
_GGUF_ROLES = ("attn_qkv", "attn_q_norm", "attn_k_norm", "attn_q", "attn_k", "attn_v",
               "attn_output", "attn_norm", "ffn_gate", "ffn_up", "ffn_down", "ffn_norm")
_LAYER = re.compile(r"(?:^|\.)(?:layers|blk|h|block|blocks)\.(\d+)\.")


def layer_of(name: str) -> int:
    """The layer index a tensor belongs to, or -1 for one outside every layer."""
    match = _LAYER.search(name)
    return int(match.group(1)) if match else -1


def role_of(name: str) -> str:
    """What a tensor does, in one vocabulary for HF and GGUF names."""
    lowered = name.lower()
    for fragment, role in _HF_ROLES:
        if fragment in lowered:
            return role
    if lowered.startswith("token_embd") or "tok_embeddings" in lowered or "wte" in lowered:
        return "embed"
    if lowered.startswith("output_norm") or lowered in ("model.norm.weight", "norm.weight") \
            or lowered.endswith(".ln_f.weight"):
        return "output_norm"
    if lowered.startswith("output.") or lowered == "output.weight":
        return "output"
    for role in _GGUF_ROLES:
        if f".{role}." in lowered or lowered.endswith(f".{role}"):
            return role
    return "other"


def pick_device(device: str | None = "auto") -> str:
    """``cuda`` when there is an NVIDIA GPU, ``mps`` on Apple silicon, else ``cpu``."""
    import torch

    wanted = (device or "auto").strip().lower()
    if wanted != "auto":
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            raise C4Error(f"device {device!r} was asked for, and PyTorch sees no CUDA GPU here")
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class C4Model:
    """A loaded model: its tensors by name, and where they came from."""

    path: Path
    format: str                      # safetensors | pt | gguf
    config: dict[str, Any]
    tensors: dict[str, Any]          # name -> torch.Tensor on `device`
    device: str = "cpu"
    #: GGUF only: the file's metadata, kept for writing one back.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The runnable module, when one has been built (see c4.probe). Its
    #: parameters *are* `tensors`: one copy of the weights, not two.
    module: Any = None

    @property
    def n_layers(self) -> int:
        layers = [layer_of(n) for n in self.tensors]
        found = max(layers, default=-1) + 1
        return int(self.config.get("num_hidden_layers") or found)

    @property
    def parameters(self) -> int:
        return int(sum(t.numel() for t in self.tensors.values()))

    def names(self, *, layer: int | None = None, role: str | None = None) -> list[str]:
        return [n for n in self.tensors
                if (layer is None or layer_of(n) == layer) and (role is None or role_of(n) == role)]

    def bind(self, module: Any) -> None:
        """Make *module*'s parameters the model's tensors (no copy)."""
        self.module = module
        named = dict(module.named_parameters())
        for name in list(self.tensors):
            if name in named:
                self.tensors[name] = named[name].data
        for name, param in named.items():
            self.tensors.setdefault(name, param.data)


# -- loading ------------------------------------------------------------------


def _config_for(path: Path) -> dict[str, Any]:
    folder = path if path.is_dir() else path.parent
    config = folder / "config.json"
    if not config.is_file():
        raise C4Error(
            f"{path} has no config.json beside it. c4 needs one for safetensors and .pt "
            f"weights: it is what says how many layers, heads and channels the tensors "
            f"form. (A GGUF carries the same facts itself.)"
        )
    try:
        return json.loads(config.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise C4Error(f"{config} is not valid JSON: {exc}") from exc


def _safetensor_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        return [path / s for s in shards]
    return sorted(path.glob("*.safetensors"))


def _torch_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    for name in ("pytorch_model.bin", "model.pt", "weights.pt", "model.bin"):
        if (path / name).is_file():
            return [path / name]
    return sorted(path.glob("*.pt")) + sorted(path.glob("pytorch_model-*.bin"))


def _format_of(path: Path) -> str:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            return "gguf"
        if suffix == ".safetensors":
            return "safetensors"
        if suffix in (".pt", ".bin", ".pth"):
            return "pt"
        raise C4Error(f"{path} is not a .safetensors, .pt/.bin or .gguf file")
    if path.is_dir():
        if _safetensor_files(path):
            return "safetensors"
        if _torch_files(path):
            return "pt"
        ggufs = sorted(path.glob("*.gguf"))
        if len(ggufs) == 1:
            return "gguf"
        raise C4Error(f"{path} holds no safetensors, .pt/.bin weights or a single .gguf")
    raise C4Error(f"No such model: {path}")


def _gguf_floats(doc: Any, tensor: Any) -> Any:
    """One GGUF tensor as a float32 numpy array in GGUF's dimension order reversed
    (row-major, as PyTorch holds it)."""
    import numpy as np

    from ..quant import llamaquants
    from ..quant.gguf import GGMLType

    data = doc.tensor_bytes(tensor)
    kind = int(tensor.ggml_type)
    if kind == int(GGMLType.F32):
        values = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif kind == int(GGMLType.F16):
        values = np.frombuffer(data, dtype="<f2").astype(np.float32)
    elif kind == int(GGMLType.BF16):
        raw = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        values = raw.view(np.float32)
    elif llamaquants.is_supported(kind):
        values = np.asarray(llamaquants.dequantize_array(data, kind), dtype=np.float32)
    else:
        from ..quant.hyprslug import _decode_floats

        values = np.asarray(_decode_floats(data, kind), dtype=np.float32)
    shape = tuple(int(d) for d in reversed(tensor.shape))
    return values[: int(np.prod(shape))].reshape(shape)


def _gguf_config(metadata: dict[str, Any]) -> dict[str, Any]:
    arch = str(metadata.get("general.architecture") or "")

    def key(name: str, default: Any = None) -> Any:
        return metadata.get(f"{arch}.{name}", default)

    return {
        "model_type": arch, "architecture": arch,
        "num_hidden_layers": key("block_count"),
        "hidden_size": key("embedding_length"),
        "intermediate_size": key("feed_forward_length"),
        "num_attention_heads": key("attention.head_count"),
        "num_key_value_heads": key("attention.head_count_kv"),
        "max_position_embeddings": key("context_length"),
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
        "rope_theta": key("rope.freq_base"),
    }


def load(path: str | Path, *, device: str | None = "auto", dtype: Any = None) -> C4Model:
    """Load *path* for c4. See the module docstring for what it accepts."""
    import torch

    source = Path(path).expanduser()
    fmt = _format_of(source)
    target = pick_device(device)
    dtype = dtype or torch.float32
    tensors: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    if fmt == "gguf":
        from ..quant.gguf import GGUFFile

        file = source if source.is_file() else next(iter(sorted(source.glob("*.gguf"))))
        doc = GGUFFile.read(file)
        metadata = dict(doc.metadata)
        config = _gguf_config(metadata)
        for tensor in doc.tensors:
            tensors[tensor.name] = torch.from_numpy(_gguf_floats(doc, tensor).copy()).to(target, dtype)
        source = file
    elif fmt == "safetensors":
        from safetensors.torch import load_file

        config = _config_for(source)
        for shard in _safetensor_files(source):
            for name, value in load_file(str(shard), device="cpu").items():
                tensors[name] = value.to(target, dtype if value.is_floating_point() else value.dtype)
    else:
        from ..security.safeload import load_checkpoint

        config = _config_for(source)
        for file in _torch_files(source):
            state = load_checkpoint(file, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            if not isinstance(state, dict):
                raise C4Error(f"{file} is not a state dict (a mapping of names to tensors)")
            for name, value in state.items():
                if hasattr(value, "is_floating_point"):
                    tensors[name] = value.to(target, dtype if value.is_floating_point() else value.dtype)
    if not tensors:
        raise C4Error(f"{source} has no tensors c4 could read")
    return C4Model(path=source, format=fmt, config=config, tensors=tensors, device=target,
                   metadata=metadata)
