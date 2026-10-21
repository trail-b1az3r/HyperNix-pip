"""hnx0xv2image — a HyperNix image-to-text model you can fuse with a text model.

Two parts, kept apart on purpose:

* **The image model** (``model_type: "hnx0xv2image"``): a vision
  transformer in the hyperNix0x-v2 style (RMSNorm, SwiGLU, no biases)
  that reads an image as patches, and a projector that pools the patch
  grid 2x2 and maps it into a text model's embedding space. It is saved
  and trained on its own.
* **A text model** it is fused with: any LFM2.5 model that is not
  already a vision model (``lfm2``, ``lfm2_moe``), any Qwen3.5 model
  (``qwen3_5``, ``qwen3_5_text`` and the MoE ones, text part), or any
  HyperNix model — a hyperNix0x-v2 Brewer folder or a HyperNix snapshot
  transformers can load.

Fusing does not touch the text model's vocabulary or weights. The prompt
is split at each ``<image>`` and the image's tokens are spliced in at
the embedding level, so the text model is exactly the one you had, and
either half can be retrained alone (``stage="projector" | "image" |
"text" | "all"``) and the halves re-fused.

::

    from hypernix.models import hnx0xv2image as hi

    image = hi.ImageModel(hi.preset("small"))
    fused = hi.fuse(image, "LiquidAI/LFM2.5-1.2B-Instruct")   # or a path
    hi.train(fused, "captions.jsonl", stage="projector", steps=2000)
    fused.save("./lfm2.5-hnx0xv2image")
    print(fused.generate("photo.jpg", "Describe the image."))

A JSONL row is ``{"image": PATH, "text": CAPTION}``, or ``{"image",
"prompt", "response"}``; a relative path is read from beside the file.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ARCH",
    "FUSED_ARCH",
    "FUSABLE",
    "FusedModel",
    "ImageConfig",
    "ImageError",
    "ImageModel",
    "PRESETS",
    "family_of",
    "fuse",
    "load",
    "load_image_model",
    "preprocess",
    "preset",
    "read_jsonl",
    "train",
]

ARCH = "hnx0xv2image"
FUSED_ARCH = "hnx0xv2image-fused"
IMAGE_TOKEN = "<image>"

#: Text model families an image model fuses with, by ``model_type``.
FUSABLE: dict[str, str] = {
    "lfm2": "LFM2.5",
    "lfm2_moe": "LFM2.5",
    "qwen3_5": "Qwen3.5",
    "qwen3_5_text": "Qwen3.5",
    "qwen3_5_moe": "Qwen3.5",
    "qwen3_5_moe_text": "Qwen3.5",
    "hypernix": "HyperNix",
    "hypernix0x-v2": "HyperNix",
}
#: Already image models: fusing a second vision tower onto one is a mistake.
_ALREADY_VISION = {"lfm2_vl", "qwen3_vl", "qwen3_vl_moe", "qwen2_5_vl", "qwen2_vl", "llava"}


class ImageError(ValueError):
    """A model, a file or a request hnx0xv2image cannot use, and why."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ImageConfig:
    image_size: int = 384
    patch_size: int = 16
    channels: int = 3
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_size: int = 0            # 0: 8/3 of hidden, rounded up to 64
    norm_eps: float = 1e-6
    pool: int = 2                # patch grid pooled pool x pool before projecting
    text_hidden_size: int = 0    # the projector's output; set by fuse()
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    name: str = "hnx0xv2image"
    model_type: str = ARCH

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ImageError(f"image_size {self.image_size} is not a multiple of patch_size {self.patch_size}")
        if self.grid % self.pool:
            raise ImageError(f"the {self.grid}x{self.grid} patch grid does not pool by {self.pool}")
        if self.hidden_size % self.num_heads:
            raise ImageError(f"hidden_size {self.hidden_size} does not divide into {self.num_heads} heads")
        if not self.ffn_size:
            self.ffn_size = (int(self.hidden_size * 8 / 3) + 63) // 64 * 64
        self.image_mean = tuple(self.image_mean)
        self.image_std = tuple(self.image_std)

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def tokens_per_image(self) -> int:
        return (self.grid // self.pool) ** 2

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_mean"] = list(self.image_mean)
        data["image_std"] = list(self.image_std)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageConfig:
        if data.get("model_type", ARCH) != ARCH:
            raise ImageError(f"not an {ARCH} config: model_type is {data.get('model_type')!r}")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


PRESETS: dict[str, dict[str, Any]] = {
    # For tests and a CPU: 64 px, 16 tokens an image.
    "tiny": {"image_size": 64, "patch_size": 8, "hidden_size": 64, "num_layers": 2, "num_heads": 4},
    # ~27 M parameters, 144 tokens an image.
    "small": {"image_size": 384, "patch_size": 16, "hidden_size": 512, "num_layers": 8, "num_heads": 8},
    # ~100 M parameters, 256 tokens an image.
    "base": {"image_size": 512, "patch_size": 16, "hidden_size": 768, "num_layers": 12, "num_heads": 12},
}


def preset(name: str, **overrides: Any) -> ImageConfig:
    if name not in PRESETS:
        raise ImageError(f"no preset {name!r}; choose from {', '.join(PRESETS)}")
    return ImageConfig(**{**PRESETS[name], "name": f"{ARCH}-{name}", **overrides})


# ---------------------------------------------------------------------------
# The image model
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class _Block(nn.Module):
    """Pre-norm block: bidirectional attention, then SwiGLU."""

    def __init__(self, cfg: ImageConfig) -> None:
        super().__init__()
        self.heads = cfg.num_heads
        self.norm1 = _RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.norm2 = _RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.ffn_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.ffn_size, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, n, d))
        h = self.norm2(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class ImageModel(nn.Module):
    """The vision encoder and the projector into a text model's embeddings."""

    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.patch_embed = nn.Conv2d(c.channels, c.hidden_size, c.patch_size, c.patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, c.grid * c.grid, c.hidden_size))
        self.blocks = nn.ModuleList(_Block(c) for _ in range(c.num_layers))
        self.norm = _RMSNorm(c.hidden_size, c.norm_eps)
        pooled = c.hidden_size * c.pool * c.pool
        self.proj_norm = _RMSNorm(pooled, c.norm_eps)
        out = c.text_hidden_size or c.hidden_size
        self.proj_in = nn.Linear(pooled, out, bias=False)
        self.proj_out = nn.Linear(out, out, bias=False)
        self._init()

    def _init(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Conv2d):
                nn.init.normal_(module.weight, std=0.02)
        scale = (2 * max(1, self.config.num_layers)) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.o_proj.weight, std=0.02 * scale)
            nn.init.normal_(block.down_proj.weight, std=0.02 * scale)

    def reset_projector(self, text_hidden_size: int) -> None:
        """A new projector for a text model of another width. Needs training."""
        c = self.config
        c.text_hidden_size = text_hidden_size
        pooled = c.hidden_size * c.pool * c.pool
        device = self.proj_in.weight.device
        self.proj_in = nn.Linear(pooled, text_hidden_size, bias=False, device=device)
        self.proj_out = nn.Linear(text_hidden_size, text_hidden_size, bias=False, device=device)
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

    @property
    def output_size(self) -> int:
        return self.proj_out.weight.shape[0]

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        """The patch grid after the encoder: (B, grid*grid, hidden)."""
        x = self.patch_embed(pixels).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Image tokens in the text model's space: (B, tokens_per_image, text hidden)."""
        c = self.config
        x = self.features(pixels)
        b, _, d = x.shape
        g, p = c.grid, c.pool
        # Pixel-unshuffle the grid: each p x p neighbourhood becomes one token.
        x = x.view(b, g // p, p, g // p, p, d).permute(0, 1, 3, 2, 4, 5).reshape(b, (g // p) ** 2, p * p * d)
        return self.proj_out(F.gelu(self.proj_in(self.proj_norm(x))))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, folder: str | Path) -> Path:
        from safetensors.torch import save_file

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(folder / "model.safetensors"))
        return folder


def load_image_model(folder: str | Path, *, device: str = "cpu") -> ImageModel:
    from safetensors.torch import load_file

    folder = Path(folder)
    config_file = folder / "config.json"
    if not config_file.is_file():
        raise ImageError(f"{folder} has no config.json")
    model = ImageModel(ImageConfig.from_dict(json.loads(config_file.read_text(encoding="utf-8"))))
    weights = folder / "model.safetensors"
    if weights.is_file():
        model.load_state_dict(load_file(str(weights), device="cpu"))
    return model.to(device)


# ---------------------------------------------------------------------------
# Images in
# ---------------------------------------------------------------------------

def _open(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, bytes | bytearray):
        import io

        return Image.open(io.BytesIO(bytes(image)))
    text = str(image)
    if text.startswith("data:") and ";base64," in text:
        import base64
        import io

        return Image.open(io.BytesIO(base64.b64decode(text.split(";base64,", 1)[1])))
    path = Path(text).expanduser()
    if not path.is_file():
        raise ImageError(f"no image at {path}")
    return Image.open(path)


def preprocess(images: Any | Sequence[Any], config: ImageConfig) -> torch.Tensor:
    """Pixels for the model: (B, 3, size, size), letterboxed, normalised.

    Letterboxed rather than stretched or cropped: a caption should be of
    the whole picture, in its own proportions.
    """
    from PIL import Image

    if not isinstance(images, list | tuple):
        images = [images]
    size = config.image_size
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    out = []
    for item in images:
        img = _open(item).convert("RGB")
        scale = size / max(img.size)
        resized = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                             Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), tuple(int(m * 255) for m in config.image_mean))
        canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
        data = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        pixels = data.view(size, size, 3).permute(2, 0, 1).float() / 255.0
        out.append((pixels - mean) / std)
    return torch.stack(out)


# ---------------------------------------------------------------------------
# Text models
# ---------------------------------------------------------------------------

def family_of(config: dict[str, Any]) -> str:
    """The fusable family of a text model's config.json, or an ImageError."""
    model_type = str(config.get("model_type", "")).lower()
    if not model_type and {"d_model", "n_layers", "n_heads"} <= set(config):
        return "hypernix0x-v2"
    if model_type in _ALREADY_VISION:
        raise ImageError(f"{model_type} is already an image model; fuse with its text-only "
                         "sibling (an LFM2.5 that is not -VL, or a Qwen3.5)")
    if model_type in FUSABLE:
        return model_type
    if model_type.startswith("hypernix") or str(config.get("name", "")).lower().startswith("hypernix"):
        return "hypernix"
    raise ImageError(f"model_type {model_type or '(none)'} is not fusable: hnx0xv2image fuses with "
                     "LFM2.5 (non-VL), Qwen3.5 and HyperNix models")


class _CharTokenizer:
    """A Brewer model's character vocabulary, as a tokenizer."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.index = {ch: i for i, ch in enumerate(vocab)}
        self.eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [self.index[ch] for ch in text if ch in self.index]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.vocab[i] for i in ids if 0 <= i < len(self.vocab))


class _TextModel:
    """One interface over a transformers model and a Brewer model."""

    def __init__(self, family: str, model: nn.Module, tokenizer: Any, source: str) -> None:
        self.family = family
        self.model = model
        self.tokenizer = tokenizer
        self.source = source

    @property
    def is_brewer(self) -> bool:
        return self.family == "hypernix0x-v2"

    @property
    def hidden_size(self) -> int:
        if self.is_brewer:
            return int(self.model.cfg.d_model)
        return int(self.embedding.weight.shape[1])

    @property
    def embedding(self) -> nn.Embedding:
        return self.model.embed.embed if self.is_brewer else self.model.get_input_embeddings()

    @property
    def eos(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    def encode(self, text: str) -> list[int]:
        if isinstance(self.tokenizer, _CharTokenizer):
            return self.tokenizer.encode(text)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        if isinstance(self.tokenizer, _CharTokenizer):
            return self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def logits(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_brewer:
            pad = (1.0 - attention_mask[:, None, None, :].to(embeds.dtype)) * torch.finfo(embeds.dtype).min
            x = embeds
            for block in self.model.blocks:
                x = block(x, pad)
            return self.model.lm_head(self.model.norm_out(x))
        return self.model(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False).logits

    def save(self, folder: Path) -> None:
        if self.is_brewer:
            from ..training.brewer import Brewer

            brewer = Brewer(self.model.cfg, name=self.model.cfg.name, save_dir=folder)
            brewer._model = self.model
            if isinstance(self.tokenizer, _CharTokenizer):
                brewer.char_vocab = self.tokenizer.vocab
            brewer.save(folder)
            return
        self.model.save_pretrained(folder)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(folder)


def _load_text(source: Any, *, device: str, dtype: torch.dtype | None, allow_other: bool,
               token: str | None = None) -> _TextModel:
    if isinstance(source, _TextModel):
        return source
    if isinstance(source, tuple):          # (model, tokenizer), already loaded
        model, tokenizer = source
        config = model.config.to_dict() if hasattr(model, "config") and hasattr(model.config, "to_dict") \
            else {"d_model": 0, "n_layers": 0, "n_heads": 0}
        family = _family_or_other(config, allow_other)
        return _TextModel(family, model.to(device), tokenizer, getattr(model, "name_or_path", "memory"))
    text = str(source)
    folder = Path(text).expanduser()
    if not folder.exists():
        from huggingface_hub import snapshot_download

        from ..security.hfpin import hub_revision

        folder = Path(snapshot_download(text, revision=hub_revision(text), token=token,
                                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                        "*.jinja", "*.tiktoken"]))
    config_file = folder / "config.json"
    if not config_file.is_file():
        raise ImageError(f"{folder} has no config.json")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    family = _family_or_other(config, allow_other)
    if family == "hypernix0x-v2":
        from ..training.brewer import Brewer

        brewer = Brewer.from_dir(folder)
        if brewer.char_vocab:
            tokenizer: Any = _CharTokenizer(list(brewer.char_vocab))
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(folder)
        return _TextModel(family, brewer.model.to(device), tokenizer, str(folder))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(folder, dtype=dtype or torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    return _TextModel(family, model, tokenizer, str(folder))


def _family_or_other(config: dict[str, Any], allow_other: bool) -> str:
    try:
        return family_of(config)
    except ImageError:
        if allow_other and str(config.get("model_type", "")) not in _ALREADY_VISION:
            return str(config.get("model_type") or "other")
        raise


# ---------------------------------------------------------------------------
# The fused model
# ---------------------------------------------------------------------------

@dataclass
class _Batch:
    embeds: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor


class FusedModel(nn.Module):
    """An image model and a text model, answering about images."""

    def __init__(self, image: ImageModel, text: _TextModel) -> None:
        super().__init__()
        if image.output_size != text.hidden_size:
            raise ImageError(f"the image model projects to {image.output_size}, the text model is "
                             f"{text.hidden_size} wide: use fuse(), which re-heads the projector")
        self.image = image
        self.text = text
        self.text_model = text.model      # registered, so .to() and .parameters() see it

    @property
    def device(self) -> torch.device:
        return self.image.proj_out.weight.device

    @property
    def family(self) -> str:
        return self.text.family

    def _segments(self, prompt: str, images: int) -> list[str]:
        parts = prompt.split(IMAGE_TOKEN)
        if len(parts) == 1:
            # No placeholder: the images go first, which is where every
            # captioning prompt would have put them.
            parts = [""] * images + [prompt]
        if len(parts) - 1 != images:
            raise ImageError(f"the prompt has {len(parts) - 1} {IMAGE_TOKEN} but {images} image(s) came with it")
        return parts

    def _sequence(self, prompt: str, image_tokens: torch.Tensor, answer: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.text.embedding
        dtype = embed.weight.dtype
        pieces, labels = [], []
        parts = self._segments(prompt, image_tokens.shape[0])
        for i, part in enumerate(parts):
            if part:
                ids = torch.tensor(self.text.encode(part), dtype=torch.long, device=self.device)
                if ids.numel():
                    pieces.append(embed(ids))
                    labels.append(torch.full_like(ids, -100))
            if i < image_tokens.shape[0]:
                pieces.append(image_tokens[i].to(dtype))
                labels.append(torch.full((image_tokens.shape[1],), -100, dtype=torch.long, device=self.device))
        if answer:
            ids = self.text.encode(answer)
            if self.text.eos is not None:
                ids = [*ids, int(self.text.eos)]
            ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
            pieces.append(embed(ids_t))
            labels.append(ids_t)
        return torch.cat(pieces), torch.cat(labels)

    def batch(self, rows: Sequence[dict[str, Any]], *, base: Path | None = None) -> _Batch:
        images = [self._image_list(row, base) for row in rows]
        flat = [img for imgs in images for img in imgs]
        tokens = self.image(preprocess(flat, self.image.config).to(self.device, self.image.proj_out.weight.dtype)) \
            if flat else None
        seqs, labs, at = [], [], 0
        for row, imgs in zip(rows, images, strict=True):
            prompt, answer = _prompt_answer(row)
            mine = tokens[at:at + len(imgs)] if tokens is not None else torch.zeros(0, 0, self.text.hidden_size)
            at += len(imgs)
            seq, lab = self._sequence(prompt, mine, answer)
            seqs.append(seq)
            labs.append(lab)
        width = max(s.shape[0] for s in seqs)
        hidden = seqs[0].shape[1]
        embeds = seqs[0].new_zeros(len(seqs), width, hidden)
        mask = torch.zeros(len(seqs), width, dtype=torch.long, device=self.device)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long, device=self.device)
        for i, (seq, lab) in enumerate(zip(seqs, labs, strict=True)):
            embeds[i, :seq.shape[0]] = seq
            mask[i, :seq.shape[0]] = 1
            labels[i, :lab.shape[0]] = lab
        return _Batch(embeds, mask, labels)

    @staticmethod
    def _image_list(row: dict[str, Any], base: Path | None) -> list[Any]:
        value = row.get("images", row.get("image"))
        values = value if isinstance(value, list) else ([] if value is None else [value])
        out = []
        for item in values:
            if isinstance(item, str) and base is not None and not item.startswith("data:"):
                path = Path(item).expanduser()
                out.append(path if path.is_absolute() else base / path)
            else:
                out.append(item)
        return out

    def loss(self, rows: Sequence[dict[str, Any]], *, base: Path | None = None) -> torch.Tensor:
        b = self.batch(rows, base=base)
        logits = self.text.logits(b.embeds, b.mask)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                               b.labels[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, image: Any | Sequence[Any], prompt: str = "Describe the image.", *,
                 max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0) -> str:
        """Answer *prompt* about one image (or several, one per ``<image>``)."""
        was = self.training
        self.eval()
        try:
            images = list(image) if isinstance(image, list | tuple) else [image]
            tokens = self.image(preprocess(images, self.image.config)
                                .to(self.device, self.image.proj_out.weight.dtype))
            embeds, _ = self._sequence(prompt, tokens)
            out: list[int] = []
            seq = embeds.unsqueeze(0)
            for _ in range(max_new_tokens):
                mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=self.device)
                logits = self.text.logits(seq, mask)[0, -1].float()
                nxt = _sample(logits, temperature, top_p)
                if self.text.eos is not None and nxt == int(self.text.eos):
                    break
                out.append(nxt)
                step = self.text.embedding(torch.tensor([[nxt]], device=self.device))
                seq = torch.cat([seq, step.to(seq.dtype)], dim=1)
            return self.text.decode(out)
        finally:
            self.train(was)

    def save(self, folder: str | Path, *, text: bool = True) -> Path:
        """``image/`` (the image model), ``text/`` (the text model) and a config.

        *text* False writes only a reference to where the text model came
        from — for a large text model you already have on disk.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.image.save(folder / "image")
        if text:
            self.text.save(folder / "text")
        config = {"model_type": FUSED_ARCH, "family": self.family,
                  "family_name": FUSABLE.get(self.family, self.family),
                  "image": "image", "text": "text" if text else self.text.source,
                  "image_token": IMAGE_TOKEN, "tokens_per_image": self.image.config.tokens_per_image}
        (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        return folder


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        ordered, index = probs.sort(descending=True)
        keep = ordered.cumsum(-1) - ordered < top_p
        probs = torch.zeros_like(probs).scatter(0, index[keep], ordered[keep])
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


def _prompt_answer(row: dict[str, Any]) -> tuple[str, str]:
    if "messages" in row:
        user = next((m.get("content", "") for m in row["messages"] if m.get("role") == "user"), "")
        reply = next((m.get("content", "") for m in reversed(row["messages"]) if m.get("role") == "assistant"), "")
        return str(user), str(reply)
    if "prompt" in row or "response" in row:
        return str(row.get("prompt", "")), str(row.get("response", ""))
    return "", str(row.get("text", row.get("caption", "")))


def fuse(image: ImageModel | str | Path, text: Any, *, device: str = "cpu",
         dtype: torch.dtype | None = None, allow_other: bool = False, token: str | None = None) -> FusedModel:
    """An image model on a text model. A projector of the wrong width is replaced.

    *text* is a folder, a Hugging Face repo id, a ``(model, tokenizer)``
    pair, or a Brewer folder. A replaced projector knows nothing yet:
    train ``stage="projector"`` before trusting a caption.
    """
    image_model = image if isinstance(image, ImageModel) else load_image_model(image, device=device)
    text_model = _load_text(text, device=device, dtype=dtype, allow_other=allow_other, token=token)
    if image_model.output_size != text_model.hidden_size or not image_model.config.text_hidden_size:
        image_model.reset_projector(text_model.hidden_size)
    return FusedModel(image_model.to(device), text_model)


def load(folder: str | Path, *, device: str = "cpu", dtype: torch.dtype | None = None,
         allow_other: bool = True) -> FusedModel:
    """A fused model saved by :meth:`FusedModel.save`."""
    folder = Path(folder)
    config_file = folder / "config.json"
    if not config_file.is_file():
        raise ImageError(f"{folder} has no config.json")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") == ARCH:
        raise ImageError(f"{folder} is an image model on its own; fuse it with a text model first")
    if config.get("model_type") != FUSED_ARCH:
        raise ImageError(f"{folder} is not a fused {ARCH} model")
    image = load_image_model(folder / config.get("image", "image"), device=device)
    text_ref = config.get("text", "text")
    text_path = folder / text_ref if (folder / text_ref).exists() else text_ref
    text = _load_text(text_path, device=device, dtype=dtype, allow_other=allow_other)
    return FusedModel(image, text)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

STAGES = ("projector", "image", "text", "all")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImageError(f"{path}:{number}: not JSON ({exc.msg})") from exc
            if not isinstance(row, dict) or ("image" not in row and "images" not in row):
                raise ImageError(f"{path}:{number}: a row needs an \"image\"")
            rows.append(row)
    if not rows:
        raise ImageError(f"{path} has no rows")
    return rows


@dataclass
class TrainReport:
    stage: str
    steps: int
    trainable: int
    losses: list[float] = field(default_factory=list)

    @property
    def first(self) -> float:
        return self.losses[0] if self.losses else math.nan

    @property
    def last(self) -> float:
        return sum(self.losses[-5:]) / len(self.losses[-5:]) if self.losses else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "steps": self.steps, "trainable": self.trainable,
                "loss_first": self.first, "loss_last": self.last}


def _stage_params(model: FusedModel, stage: str) -> list[nn.Parameter]:
    if stage not in STAGES:
        raise ImageError(f"stage is one of {', '.join(STAGES)}, not {stage!r}")
    projector = [*model.image.proj_norm.parameters(), *model.image.proj_in.parameters(),
                 *model.image.proj_out.parameters()]
    groups = {"projector": projector, "image": list(model.image.parameters()),
              "text": list(model.text_model.parameters()),
              "all": [*model.image.parameters(), *model.text_model.parameters()]}
    chosen = {id(p) for p in groups[stage]}
    for p in model.parameters():
        p.requires_grad_(id(p) in chosen)
    return [p for p in model.parameters() if p.requires_grad]


def train(model: FusedModel, data: str | Path | Sequence[dict[str, Any]], *, stage: str = "projector",
          steps: int = 100, lr: float = 1e-3, batch_size: int = 4, optimizer: str = "adamw",
          seed: int = 0, log_every: int = 0) -> TrainReport:
    """Train one part, the others frozen. The text model is untouched unless *stage* says so.

    ``optimizer`` is ``adamw`` or ``c4`` (the c4 hybrid AdamW / Pressure
    Cooker V5S, which checks each part's gradient before it updates).
    """
    base = Path(data).parent if isinstance(data, str | Path) else None
    rows = read_jsonl(data) if isinstance(data, str | Path) else list(data)
    params = _stage_params(model, stage)
    if optimizer == "c4":
        from ..c4.optim import C4HybridAdamW5S

        opt: torch.optim.Optimizer = C4HybridAdamW5S(params, lr=lr)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if stage == "projector" else 0.01)
    rng = random.Random(seed)
    report = TrainReport(stage, steps, sum(p.numel() for p in params))
    model.train()
    for step in range(steps):
        chosen = [rows[rng.randrange(len(rows))] for _ in range(min(batch_size, len(rows)))]
        loss = model.loss(chosen, base=base)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        report.losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(f"[hnx0xv2image] {stage} step {step + 1}/{steps}  loss {report.last:.4f}")
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return report
