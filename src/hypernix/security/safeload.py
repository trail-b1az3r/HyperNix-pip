"""Loading PyTorch checkpoints without running what is inside them.

``torch.load(path, weights_only=False)`` unpickles the file, and a pickle
can run any code it likes as it is read: a ``.pt`` downloaded from a model
hub, or handed over by someone else, is a program. Every checkpoint
HyperNix writes is tensors, dicts, lists, strings and numbers, which
``weights_only=True`` loads without executing anything, so that is how
HyperNix reads them.

A file that needs more than that (one written by another tool with its
own classes in it) is refused with the reason, unless the person says
they trust it: ``HYPERNIX_TRUST_PICKLE=1`` in the environment, or
``trust_pickle=True`` from code that knows where the file came from.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["UntrustedCheckpoint", "load_checkpoint", "pickle_trusted"]

#: Set to 1 to let a checkpoint that is not plain data be unpickled.
TRUST_ENV = "HYPERNIX_TRUST_PICKLE"


class UntrustedCheckpoint(RuntimeError):
    """The file holds objects that only full unpickling can rebuild."""


def pickle_trusted() -> bool:
    return os.environ.get(TRUST_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu",
                    trust_pickle: bool | None = None) -> Any:
    """``torch.load`` that does not execute the file.

    Loads with ``weights_only=True``. If the file needs full unpickling it
    is loaded that way only when *trust_pickle* is true (or, when it is
    None, when ``HYPERNIX_TRUST_PICKLE`` is set); otherwise
    :class:`UntrustedCheckpoint` says why and how to allow it.
    """
    import pickle

    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        trusted = pickle_trusted() if trust_pickle is None else trust_pickle
        if not trusted:
            raise UntrustedCheckpoint(
                f"{path} holds Python objects, not just tensors and plain data, "
                f"and loading it would run code from the file. If you trust "
                f"where it came from, set {TRUST_ENV}=1 and load it again. "
                f"({str(exc).splitlines()[0]})"
            ) from exc
        # The person has said this file is theirs to trust.
        return torch.load(path, map_location=map_location, weights_only=False)  # nosec B614  # nosemgrep
