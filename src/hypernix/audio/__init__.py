"""hypernix.audio — audio subsystems.

:mod:`hypernix.audio.audiofile` reads audio in whatever format it
arrives in. :mod:`hypernix.audio.processor` is the signal processing
between reading and using it — resampling, filtering, levelling,
silence — which everything else used to do by hand, slightly
differently each time. :mod:`hypernix.audio.features` turns audio into
log-mel frames, and :mod:`hypernix.audio.wakeup` is the wake-word
trainer and streaming detector built on all three.

Submodules are imported lazily so importing this package costs nothing
until you touch one.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = ["audiofile", "features", "processor", "wakeup"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
