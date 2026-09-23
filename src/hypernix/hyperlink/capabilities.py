"""hyperlink.capabilities — what a model can take, before it is sent it.

The app offered a photo button whatever model was picked, so a picture
went to a text-only model, which either failed with an error about
message formats or — worse — answered as if it had seen something. The
question the app needs answered is small: *can this model see images?*

Three kinds of evidence, strongest first:

1. The runtime says so. LM Studio lists a vision model as ``vlm``.
2. The files say so. A GGUF vision model needs its projector, an
   ``mmproj*.gguf`` beside it; without one llama.cpp cannot see even
   with a vision architecture.
3. The name or architecture says so. Families are recognised by the
   patterns their authors use (``-VL``, ``vision``, ``llava`` …).

The answer is ``True``, ``False``, or ``None`` for "nothing says either
way". The app shows the button for True, hides it for False, and for
None shows it with a warning rather than guessing: refusing a model that
can see is as wrong as sending a picture to one that cannot.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["VISION_PATTERNS", "TEXT_ONLY_PATTERNS", "supports_images", "has_projector"]

#: Names and architectures of families that take images. Matched against
#: a lower-cased model id, display name and architecture.
VISION_PATTERNS: tuple[str, ...] = (
    r"llava", r"bakllava", r"\bvl\b", r"[-_.]vl[-_.\d]", r"[-_.]vl$", r"vision",
    r"pixtral", r"minicpm-?v", r"moondream", r"internvl", r"idefics",
    r"smolvlm", r"paligemma", r"molmo", r"cogvlm", r"glm-?4v", r"kimi-?vl",
    r"deepseek-?vl", r"janus", r"florence", r"llama-?4", r"mllama",
    r"gemma-?3(?!.*\b1b\b)(?!n)", r"mistral-small-3\.[12]", r"phi-?3\.5-vision",
    r"phi-?4-multimodal", r"qwen2(\.5)?-?vl", r"qwen3-?vl", r"qwen2(\.5)?-?omni",
    r"qwen3[._]5(?![._]?text)", r"\bvlm\b", r"multimodal",
)

#: Families that share a vision family's name but have no eyes: the text
#: tower of a composite, a 1B that shipped without the encoder.
TEXT_ONLY_PATTERNS: tuple[str, ...] = (
    r"qwen3[._]5[._]?text", r"gemma-?3[-_]?1b", r"gemma-?3n", r"text-only",
)

_VISION = [re.compile(p) for p in VISION_PATTERNS]
_TEXT_ONLY = [re.compile(p) for p in TEXT_ONLY_PATTERNS]


def has_projector(path: str | Path | None) -> bool:
    """Whether a GGUF has an ``mmproj`` projector file beside it."""
    if not path:
        return False
    model = Path(path)
    try:
        siblings = list(model.parent.iterdir())
    except OSError:
        return False
    return any(
        s.name.lower().startswith("mmproj") or "mmproj" in s.name.lower()
        for s in siblings
        if s.suffix.lower() == ".gguf" and s != model
    )


def supports_images(
    model_id: str = "",
    *,
    name: str = "",
    architecture: str = "",
    path: str | Path | None = None,
    runtime_says: bool | None = None,
) -> bool | None:
    """True, False, or None when nothing says either way."""
    if runtime_says is not None:
        return bool(runtime_says)
    text = " ".join(part for part in (model_id, name, architecture) if part).lower()
    if not text:
        return None
    if any(p.search(text) for p in _TEXT_ONLY):
        return False
    looks_visual = any(p.search(text) for p in _VISION)
    if path is not None and str(path).lower().endswith(".gguf"):
        # A GGUF sees only with its projector, whatever it is called.
        if has_projector(path):
            return True
        return False if looks_visual else None
    return True if looks_visual else None
