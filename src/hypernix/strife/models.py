"""strife.models — turn what somebody typed into a GGUF on disk.

``hnx strife chat qwen3.5-0.8b-q4_k_m`` should find the same model the
HyperLink catalogue lists under that id, so this resolves exactly as the
catalogue does:

1. an existing file path, or a folder holding exactly one ``.gguf``;
2. a catalogue id (:func:`hypernix.t1api.modelindex.model_id_for` of the
   file name) or a file name, in the models folder (``T1_HF_DOWNLOAD_DIR``
   when set, and ``~/.hypernix/models``);
3. a part of one, when it matches exactly one model.

Anything else is a coded error that lists what is there, and, for
something shaped like a Hugging Face id, how to download it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import StrifeError

__all__ = ["LocalModel", "local_models", "resolve_model"]


@dataclass(frozen=True)
class LocalModel:
    model_id: str
    path: Path


def _folders() -> list[Path]:
    from ..hyperlink.catalogue import DEFAULT_LOCAL_DIR

    out = []
    configured = os.environ.get("T1_HF_DOWNLOAD_DIR")
    if configured:
        out.append(Path(configured).expanduser())
    out.append(Path(os.path.expanduser("~")) / ".hypernix" / "models")
    out.append(DEFAULT_LOCAL_DIR)
    seen, unique = set(), []
    for folder in out:
        key = str(folder)
        if key not in seen:
            seen.add(key)
            unique.append(folder)
    return unique


def local_models() -> list[LocalModel]:
    """Every GGUF in the models folders, by catalogue id."""
    from ..t1api.modelindex import model_id_for

    found: dict[str, LocalModel] = {}
    for folder in _folders():
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.gguf")):
            name = path.name.lower()
            if name.startswith("mmproj") or not path.is_file():
                continue
            found.setdefault(model_id_for(path), LocalModel(model_id_for(path), path))
    return list(found.values())


def resolve_model(spec: str) -> Path:
    path = Path(spec).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        ggufs = [p for p in path.glob("*.gguf") if not p.name.lower().startswith("mmproj")]
        if len(ggufs) == 1:
            return ggufs[0]
        raise StrifeError("strife.model.ambiguous", f"{path} holds {len(ggufs)} .gguf files",
                          remedy="Name the file itself.")
    wanted = spec.strip().lower()
    models = local_models()
    exact = [m for m in models if wanted in (m.model_id, m.path.name.lower(), m.path.stem.lower())]
    if len(exact) == 1:
        return exact[0].path
    partial = [m for m in models if wanted and wanted in m.model_id]
    if len(partial) == 1:
        return partial[0].path
    if len(partial) > 1:
        raise StrifeError("strife.model.ambiguous",
                          f"{spec!r} matches {len(partial)} models: "
                          + ", ".join(m.model_id for m in partial[:8]),
                          remedy="Use more of the id.")
    folders = ", ".join(str(f) for f in _folders())
    remedy = (f"Download it first: hnx download {spec}" if "/" in spec
              else "Give a path to a .gguf, or one of: "
              + (", ".join(m.model_id for m in models[:10]) or f"(no models in {folders})"))
    raise StrifeError("strife.model.not_found", f"no model {spec!r} here", remedy=remedy,
                      details={"searched": folders})
