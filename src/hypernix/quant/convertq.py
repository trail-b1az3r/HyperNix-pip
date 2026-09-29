"""``hypernix convert -P``: safetensors to a quantised GGUF in one step.

Converting and quantising were two commands with an F16 file in between
that nobody wanted to keep. This runs both and removes the F16 unless
asked to keep it.

What it accepts, and what happens to each:

* a Hugging Face folder, or one ``.safetensors`` file in one, is
  converted with :func:`hypernix.quant.convert.convert_to_gguf`;
* a hyperNix0x-v2 (Brewer) folder or ``.pt`` goes to hyprslug as is,
  which exports it itself;
* a ``.gguf`` needs no conversion and goes straight to hyprslug.

The quantisation is hyprslug's, so every target it knows works here:
llama.cpp's types and mixes, the HyperNix tiers, and the hybrids.
"""
from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ConvertQuantizeError", "ConvertQuantizeResult", "convert_and_quantize",
           "source_kind", "default_output"]


class ConvertQuantizeError(ValueError):
    """The source is not something -P can convert."""


@dataclass
class ConvertQuantizeResult:
    output: Path
    kind: str
    target: str
    report: object
    intermediate: Path | None = None


def source_kind(path: str | Path) -> str:
    """``"gguf"``, ``"brewer"``, ``"safetensors"``, or ``""`` for none of them."""
    from .hyprslug import is_brewer_source

    candidate = Path(path).expanduser()
    if candidate.is_file():
        suffix = candidate.suffix.lower()
        if suffix == ".gguf":
            return "gguf"
        if suffix == ".pt":
            return "brewer"
        if suffix == ".safetensors":
            return "brewer" if is_brewer_source(candidate.parent) else "safetensors"
        return ""
    if candidate.is_dir():
        if is_brewer_source(candidate):
            return "brewer"
        if any(candidate.glob("*.safetensors")):
            return "safetensors"
    return ""


def _model_dir(path: Path) -> Path:
    return path.parent if path.is_file() else path


def default_output(source: str | Path, target: str) -> Path:
    """``<model>.<target>.gguf`` beside the source."""
    path = Path(source).expanduser()
    folder = _model_dir(path) if path.suffix.lower() == ".safetensors" else path
    stem = folder.stem if folder.is_file() else folder.name
    return folder.parent / f"{stem}.{target}.gguf"


def convert_and_quantize(
    source: str | Path,
    target: str,
    output: str | Path | None = None,
    *,
    dtype: str = "fp16",
    arch_name: str = "hypernix",
    name: str | None = None,
    keep_intermediate: bool = False,
    imatrix: str | Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> ConvertQuantizeResult:
    """Convert *source* if it needs it, then quantise it to *target*."""
    from .hyprslug import quantize_gguf, resolve_target

    path = Path(source).expanduser()
    if not path.exists():
        raise ConvertQuantizeError(f"No such model: {path}")
    kind = source_kind(path)
    if not kind:
        raise ConvertQuantizeError(
            f"{path} is not safetensors, a hyperNix0x-v2 model or a GGUF. "
            f"-P converts a Hugging Face folder (or a .safetensors in one), "
            f"a Brewer folder or .pt, and quantises a .gguf as it is."
        )
    _kind, canonical = resolve_target(target)          # refuse a bad -Q first
    out = Path(output).expanduser() if output else default_output(path, canonical)
    out.parent.mkdir(parents=True, exist_ok=True)

    if kind != "safetensors":
        report = quantize_gguf(path, out, canonical, imatrix=imatrix, progress=progress)
        return ConvertQuantizeResult(out, kind, canonical, report)

    from .convert import convert_to_gguf

    model_dir = _model_dir(path)
    label = name or model_dir.name
    if keep_intermediate:
        staged = out.with_name(f"{model_dir.name}.{dtype}.gguf")
        convert_to_gguf(model_dir, staged, dtype=dtype, arch_name=arch_name, name=label)
        report = quantize_gguf(staged, out, canonical, imatrix=imatrix, progress=progress)
        return ConvertQuantizeResult(out, kind, canonical, report, staged)
    # Beside the output, not in /tmp: the F16 copy is the biggest file in
    # the pipeline and /tmp is often a small tmpfs.
    with tempfile.TemporaryDirectory(prefix=".hnx-convert-", dir=out.parent) as scratch:
        staged = Path(scratch) / f"{model_dir.name}.{dtype}.gguf"
        convert_to_gguf(model_dir, staged, dtype=dtype, arch_name=arch_name, name=label)
        report = quantize_gguf(staged, out, canonical, imatrix=imatrix, progress=progress)
    return ConvertQuantizeResult(out, kind, canonical, report)
