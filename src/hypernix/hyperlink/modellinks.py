"""Link a model that is already on the server into HyperLink's models folder.

A model on another disk used to need a copy, or an ``ln -s`` in a shell
on the server -- and until 0.72.6.post3 the link did not even work, since
every scanner skipped symlinked folders. This makes the link from the
app: pick a path on the server, and it appears in the model picker like
anything downloaded.

Only a symlink is ever created or removed here. :func:`unlink_model`
refuses anything that is not one, so "remove from the list" can never
delete the model it points at.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

__all__ = ["ModelLinkError", "link_model", "unlink_model", "NAME_PATTERN"]

#: What a link may be called: one path segment, no leading dot (hidden
#: files are skipped by the scanners), nothing a shell or a URL mangles.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ModelLinkError(ValueError):
    """The link could not be made, with the reason."""

    def __init__(self, message: str, *, conflict: bool = False, missing: bool = False):
        super().__init__(message)
        self.conflict = conflict
        self.missing = missing


def _kind(source: Path) -> str:
    from ..system.linkwalk import walk_files
    from .brewed import is_brewed_dir

    if source.is_file():
        if source.suffix.lower() == ".gguf":
            return "gguf"
        raise ModelLinkError(
            f"{source} is not a .gguf. Link a GGUF file, or a folder with one "
            f"in it, or a hyperNix0x-v2 model folder."
        )
    if source.is_dir():
        if is_brewed_dir(source):
            return "hypernix"
        if next(walk_files(source, ".gguf"), None) is not None:
            return "folder"
        raise ModelLinkError(
            f"{source} has no .gguf in it and is not a hyperNix0x-v2 model. "
            f"A Hugging Face folder of safetensors needs converting first: "
            f"hnx convert {source} -P -Q Q4_K_M"
        )
    raise ModelLinkError(f"{source} is neither a file nor a folder.")


def link_model(path: str | Path, models_dir: str | Path, name: str = "") -> dict[str, Any]:
    """Symlink *path* into *models_dir* as *name* (default: its own name)."""
    raw = str(path or "").strip()
    if not raw:
        raise ModelLinkError("Give the path of a model on this server.")
    source = Path(raw).expanduser()
    if not source.is_absolute():
        raise ModelLinkError(
            f"{raw} is relative. Give the full path on the server, e.g. "
            f"/data/models/qwen3-8b-q4_k_m.gguf"
        )
    if not source.exists():
        raise ModelLinkError(f"Nothing at {source} on this server.", missing=True)
    real = source.resolve()
    kind = _kind(real)

    folder = Path(models_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    if real == folder.resolve() or real.is_relative_to(folder.resolve()):
        raise ModelLinkError(f"{source} is already in {folder}; it is listed as it is.")

    wanted = (name or source.name).strip()
    if not NAME_PATTERN.fullmatch(wanted):
        raise ModelLinkError(
            f"{wanted!r} cannot be a model name here: use letters, digits, '.', "
            f"'_' and '-', starting with a letter or digit."
        )
    target = folder / wanted
    if target.is_symlink() or target.exists():
        raise ModelLinkError(
            f"{target} already exists. Pick another name, or remove that one first.",
            conflict=True,
        )
    os.symlink(real, target, target_is_directory=real.is_dir())
    return {"name": wanted, "path": str(target), "linked_to": str(real), "kind": kind}


def unlink_model(name: str, models_dir: str | Path) -> dict[str, Any]:
    """Remove the link *name* from *models_dir*. Never the model itself."""
    if not NAME_PATTERN.fullmatch(name or ""):
        raise ModelLinkError(f"{name!r} is not a model link name.")
    target = Path(models_dir).expanduser() / name
    if not target.is_symlink():
        if target.exists():
            raise ModelLinkError(
                f"{target} is a real file or folder, not a link. Only links are "
                f"removed from here; delete a model on the server itself.",
                conflict=True,
            )
        raise ModelLinkError(f"No link called {name} in {models_dir}.", missing=True)
    pointed = os.readlink(target)
    target.unlink()
    return {"name": name, "path": str(target), "linked_to": pointed, "kind": "removed"}
