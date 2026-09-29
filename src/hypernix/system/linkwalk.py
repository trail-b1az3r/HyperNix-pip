"""Walk a model folder the way a person thinks of it: through symlinks.

``Path.rglob`` does not descend into a symlinked directory, on any
Python this package supports (3.13 added ``recurse_symlinks`` and left
it off). So ``ln -s /data/qwen ~/.hypernix/models/qwen`` -- the obvious
way to keep a big model on another disk -- made the model invisible:
the link was there, ``ls`` showed the files, and every scanner that
listed models walked straight past it.

:func:`walk_files` follows directory links, and remembers every real
directory it has entered, so a link back up the tree ends the walk
rather than looping. Paths come back as they appear under the root --
``models/qwen/model.gguf``, not ``/data/qwen/model.gguf`` -- because that
is the name the rest of the server knows the model by.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

__all__ = ["walk_files", "broken_links", "link_target"]


def walk_files(root: str | Path, suffix: str | None = None) -> Iterator[Path]:
    """Every file under *root*, through symlinked directories and files.

    *suffix* filters by extension, case-insensitively (``".gguf"``). A
    dangling link is not a file and is skipped; :func:`broken_links`
    finds those.
    """
    base = Path(root)
    if not base.is_dir():
        return
    wanted = suffix.lower() if suffix else None
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            # A link to a directory already walked: its files are
            # already listed once, under the first name reached.
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames.sort()
        for name in sorted(filenames):
            if wanted and not name.lower().endswith(wanted):
                continue
            path = Path(dirpath) / name
            if path.is_file():
                yield path


def broken_links(root: str | Path) -> list[tuple[Path, str]]:
    """``(link, what it points at)`` for every dangling symlink under *root*.

    The model a person linked in and then moved, or whose disk is not
    mounted. Worth showing rather than skipping: "why is my model gone"
    is answered by the link's target.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        for name in sorted(dirnames + filenames):
            path = Path(dirpath) / name
            if path.is_symlink() and not path.exists():
                found.append((path, link_target(path)))
    return found


def link_target(path: str | Path) -> str:
    """What a symlink points at, as written; ``""`` for anything else."""
    try:
        return os.readlink(path)
    except OSError:
        return ""
