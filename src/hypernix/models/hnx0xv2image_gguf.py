"""hnx0xv2image in a GGUF, so hyprslug can quantise it.

hyprslug quantises GGUFs, and a fused image model was a folder of two
safetensors files: ``hyprslug ./lfm-eye Q4_K_M`` died on
``IsADirectoryError``. Here the whole fused model -- the image tower, its
projector and the text model, LFM2.5, Qwen3.5 or Brewer alike -- goes
into one GGUF, ``general.architecture = hnx0xv2image-fused``, with both
configs in its metadata. hyprslug packs it like any other (2-D weights
to the tier, norms and biases kept at source precision), and
:func:`load_gguf` reads it back, dequantising each tensor, so ``hnx image
caption`` answers from the quantised file.

What this is not: something llama.cpp runs. Its multimodal path wants an
``mmproj`` of a vision tower it already knows. This is the file at the
size the tier promises, for storing and moving; the model is float again
once loaded.

A quantised model is a folder::

    lfm-eye.q4km/
      config.json      model_type hnx0xv2image-fused, "gguf": "model.gguf", the tier
      model.gguf       every weight, at the tier
      tokenizer/       the text model's tokenizer (none for a character vocabulary)
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import hnx0xv2image as hi

__all__ = ["GGUF_ARCH", "is_fused_folder", "load_gguf", "quantize", "to_gguf"]

GGUF_ARCH = hi.FUSED_ARCH
_KEY = "hnx0xv2image"
GGUF_FILE = "model.gguf"


def is_fused_folder(path: str | Path) -> bool:
    """A fused model folder, as :meth:`FusedModel.save` or :func:`quantize` writes it."""
    config = Path(path) / "config.json"
    if not config.is_file():
        return False
    try:
        return json.loads(config.read_text(encoding="utf-8")).get("model_type") == hi.FUSED_ARCH
    except ValueError:
        return False


def _text_config(text: Any) -> tuple[str, dict[str, Any]]:
    if text.is_brewer:
        return "brewer", text.model.cfg.to_dict()
    return "transformers", text.model.config.to_dict()


def _tensors(fused: hi.FusedModel) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, str]]:
    """``(tensors, aliases, dtypes)``: every tensor under ``image.`` and
    ``text.``, one copy of a tied one (the other named in *aliases*), and
    the dtype of any that is not floating point."""
    out: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    seen: dict[tuple[int, int, tuple[int, ...]], str] = {}
    for prefix, module in (("image.", fused.image), ("text.", fused.text.model)):
        for name, tensor in module.state_dict().items():
            key = prefix + name
            ident = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
            if ident in seen:
                aliases[key] = seen[ident]
                continue
            seen[ident] = key
            if not tensor.is_floating_point():
                dtypes[key] = str(tensor.dtype).removeprefix("torch.")
            out[key] = tensor.detach().cpu()
    return out, aliases, dtypes


def to_gguf(fused: hi.FusedModel | str | Path, destination: str | Path) -> Path:
    """Write *fused* (a model or a saved folder) as one F32 GGUF."""
    from ..quant.gguf import GGMLType, GGUFValueType, GGUFWriter

    model = hi.load(fused) if not isinstance(fused, hi.FusedModel) else fused
    tensors, aliases, dtypes = _tensors(model)
    kind, text_config = _text_config(model.text)
    destination = Path(destination)
    writer = GGUFWriter(destination)
    string = (int(GGUFValueType.STRING), None)
    writer.set_metadata("general.architecture", GGUF_ARCH, type_hint=string)
    writer.set_metadata("general.name", str(model.image.config.name), type_hint=string)
    for key, value in {
        "family": model.family,
        "image_config": json.dumps(model.image.config.to_dict()),
        "text_kind": kind,
        "text_config": json.dumps(text_config, default=str),
        "aliases": json.dumps(aliases),
        "dtypes": json.dumps(dtypes),
        "shapes": json.dumps({k: list(t.shape) for k, t in tensors.items()}),
        "image_token": hi.IMAGE_TOKEN,
    }.items():
        writer.set_metadata(f"{_KEY}.{key}", value, type_hint=string)
    if isinstance(model.text.tokenizer, hi._CharTokenizer):
        writer.set_metadata(f"{_KEY}.char_vocab", json.dumps(model.text.tokenizer.vocab), type_hint=string)

    arrays: dict[str, bytes] = {}
    for name, tensor in tensors.items():
        values = tensor.to(torch.float32).contiguous().numpy()
        arrays[name] = values.astype("<f4").tobytes()
        # GGUF lists dimensions fastest first: a torch [out, in] is ne [in, out].
        shape = tuple(reversed(values.shape)) or (1,)
        writer.add_tensor(name, shape, int(GGMLType.F32))
    writer.write(lambda t: arrays[t.name])
    return destination


def _values(data: bytes, ggml_type: int) -> np.ndarray:
    """A tensor's bytes as float32, whatever it was packed as."""
    from ..quant import hyprslug, llamaquants
    from ..quant.gguf import GGMLType

    kind = int(ggml_type)
    if kind == int(GGMLType.F32):
        return np.frombuffer(data, dtype="<f4").astype(np.float32)
    if kind == int(GGMLType.F16):
        return np.frombuffer(data, dtype="<f2").astype(np.float32)
    if kind == int(GGMLType.BF16):
        return (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if llamaquants.is_supported(kind):
        return np.asarray(llamaquants.dequantize_array(data, kind), dtype=np.float32)
    return np.asarray(hyprslug._decode_floats(data, kind), dtype=np.float32)


def load_gguf(path: str | Path, *, device: str = "cpu", dtype: torch.dtype | None = None) -> hi.FusedModel:
    """A fused model from :func:`to_gguf`'s file (quantised or not), or a
    folder :func:`quantize` wrote."""
    from ..quant.gguf import GGUFError, GGUFFile

    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if path.is_dir():
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        path = path / config.get("gguf", GGUF_FILE)
    try:
        model = GGUFFile.read(path)
    except (GGUFError, OSError) as exc:
        raise hi.ImageError(f"{path}: {exc}") from exc
    meta = model.metadata
    if meta.get("general.architecture") != GGUF_ARCH:
        raise hi.ImageError(f"{path} is not an {GGUF_ARCH} GGUF "
                            f"(general.architecture is {meta.get('general.architecture')!r})")

    def field(name: str, default: str = "") -> Any:
        return json.loads(meta.get(f"{_KEY}.{name}", default) or "null")

    shapes = field("shapes", "{}")
    dtypes = field("dtypes", "{}")
    state: dict[str, torch.Tensor] = {}
    with path.open("rb") as stream:
        for tensor in model.tensors:
            stream.seek(model.data_start + tensor.offset)
            raw = stream.read(tensor.nbytes)
            values = _values(raw, tensor.ggml_type)
            shape = shapes.get(tensor.name) or list(reversed(tensor.shape))
            out = torch.from_numpy(values[: int(np.prod(shape)) if shape else 1].copy()).reshape(shape)
            if tensor.name in dtypes:
                out = out.round().to(getattr(torch, dtypes[tensor.name]))
            state[tensor.name] = out
    for alias, original in (field("aliases", "{}") or {}).items():
        state[alias] = state[original]

    image = hi.ImageModel(hi.ImageConfig.from_dict(field("image_config")))
    image.load_state_dict({k[len("image."):]: v for k, v in state.items() if k.startswith("image.")})

    # Plain strings, not JSON.
    kind = str(meta.get(f"{_KEY}.text_kind", ""))
    family = str(meta.get(f"{_KEY}.family", ""))
    text_config = field("text_config")
    text_state = {k[len("text."):]: v for k, v in state.items() if k.startswith("text.")}
    if kind == "brewer":
        from ..training.brewer import Brewer, BrewerConfig

        cfg = BrewerConfig.from_dict(text_config)
        brewer = Brewer(cfg, name=cfg.name, save_dir=folder)
        brewer.build()
        text_model: torch.nn.Module = brewer.model
    else:
        from transformers import AutoConfig, AutoModelForCausalLM

        settings = dict(text_config)
        config = AutoConfig.for_model(settings.pop("model_type"), **settings)
        text_model = AutoModelForCausalLM.from_config(config)
    missing, unexpected = text_model.load_state_dict(text_state, strict=False)
    if unexpected or [k for k in missing if not k.endswith("lm_head.weight")]:
        raise hi.ImageError(f"{path}: the text weights do not match its config "
                            f"(missing {missing[:5]}, unexpected {unexpected[:5]})")
    if missing and hasattr(text_model, "tie_weights"):
        text_model.tie_weights()
    chars = field("char_vocab")
    if chars:
        tokenizer: Any = hi._CharTokenizer(list(chars))
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")
    if dtype is not None:
        text_model = text_model.to(dtype)
    text = hi._TextModel(family, text_model.to(device), tokenizer, str(path))
    return hi.FusedModel(image.to(device), text)


def quantize(source: str | Path | hi.FusedModel, destination: str | Path, tier: str,
             **options: Any) -> Any:
    """Quantise a fused model to *tier* with hyprslug, into a folder.

    *options* go to :func:`hypernix.quant.hyprslug.quantize_gguf`
    (``imatrix``, ``quantize_embeddings``, ``quantize_output``,
    ``progress``). Returns its report.
    """
    from ..quant.hyprslug import quantize_gguf

    model = hi.load(source) if not isinstance(source, hi.FusedModel) else source
    destination = Path(destination)
    if destination.suffix.lower() == ".gguf":
        destination = destination.with_suffix("")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hnx0xv2image-") as scratch:
        staged = to_gguf(model, Path(scratch) / "f32.gguf")
        report = quantize_gguf(staged, destination / GGUF_FILE, tier, **options)
    tokenizer = model.text.tokenizer
    target = destination / "tokenizer"
    if target.exists():
        shutil.rmtree(target)
    if not isinstance(tokenizer, hi._CharTokenizer) and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(target)
    config = {"model_type": hi.FUSED_ARCH, "format": "gguf", "gguf": GGUF_FILE, "tier": tier,
              "family": model.family, "family_name": hi.FUSABLE.get(model.family, model.family),
              "image_token": hi.IMAGE_TOKEN, "tokens_per_image": model.image.config.tokens_per_image}
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return report
