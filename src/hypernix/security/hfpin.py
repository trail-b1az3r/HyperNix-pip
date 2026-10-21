"""Pinning what is downloaded from the Hugging Face Hub (0.72.8).

A repo id names whatever its owner pushed last. Fetching ``main`` means a
model, tokenizer or dataset can change under a working setup -- by
accident, or because the repo was taken over. Every Hub download in
HyperNix asks :func:`hub_revision` which revision to fetch:

1. the revision the caller was given (``--revision``, a ``revision=``
   argument), when there is one;
2. otherwise a pin for that repo: ``HYPERNIX_HF_REVISION_<REPO>`` in the
   environment (the repo id upper-cased, every non-alphanumeric character
   an underscore), then the JSON file named by ``HYPERNIX_HF_PINS``
   (default ``~/.hypernix/hf-pins.json``), mapping ``"owner/repo"`` to a
   commit hash;
3. otherwise ``None``, which is the Hub's default branch -- what every
   download did before.

A local path is never pinned: it is not fetched.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

__all__ = ["hub_revision", "pins_path"]

_NOT_ALNUM = re.compile(r"[^A-Za-z0-9]")


def pins_path() -> Path:
    override = os.environ.get("HYPERNIX_HF_PINS", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".hypernix" / "hf-pins.json"


def _is_local(name: str) -> bool:
    if name.startswith(("./", "../", "/", "~")) or "\\" in name:
        return True
    try:
        return Path(name).expanduser().exists()
    except OSError:
        return False


def hub_revision(repo_or_path: object, revision: str | None = None) -> str | None:
    """The revision to fetch *repo_or_path* at, or ``None`` for the default."""
    if revision:
        return revision
    name = str(repo_or_path or "").strip()
    if not name or _is_local(name):
        return None
    env = os.environ.get("HYPERNIX_HF_REVISION_" + _NOT_ALNUM.sub("_", name).upper(), "").strip()
    if env:
        return env
    try:
        pins = json.loads(pins_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pinned = pins.get(name) if isinstance(pins, dict) else None
    return str(pinned).strip() or None if pinned else None
