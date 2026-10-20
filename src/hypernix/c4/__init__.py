"""hypernix.c4 — explode a model into its pieces, measure and test them,
reap the best and sow a new model from them.

::

    from hypernix import c4

    ex = c4.explode.explode("./my-model", device="auto")   # GPU when there is one
    prober = c4.probe.Prober.for_model(ex.model, texts=open("calib.txt").read().split("\\n\\n"))

    layers = c4.stem.layer_report(ex, prober)               # which layers are needed
    scores = c4.dial.score(ex, prober, kind="taylor")       # which weights matter
    cut = c4.dial.unneeded(scores, fraction=0.2)
    todo = c4.dial.train_priority(ex, scores, granularity="pdial")

    harvest = c4.reap.reap(ex, report=layers, drop_layers=2, unneeded=cut, ffn_keep=0.75)
    c4.sow.sow(harvest, "./my-model-c4", explodeds=[ex], prober=prober)

The sections:

* :mod:`hypernix.c4.explode` — mstems, stems, Pdials, mdials and dials;
* :mod:`hypernix.c4.stem` — measuring stems and testing layers;
* :mod:`hypernix.c4.dial` — per-parameter scores, what is unneeded,
  what to train next;
* :mod:`hypernix.c4.restore` — snapshots, temporary ablations, writing;
* :mod:`hypernix.c4.reap` — choosing the layers, dials and channels;
* :mod:`hypernix.c4.sow` — building the new model, and healing it;
* :mod:`hypernix.c4.optim` — the separate-and-check optimizer, with the
  c4 Pressure Cooker V5S and the V5S/AdamW hybrid;
* :mod:`hypernix.c4.probe` — running the model on calibration data.

Command line: ``hnx c4 explode|stem|dial|reap|sow`` (see ``hnx c4 -h``).
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = ["dial", "explode", "load", "optim", "probe", "reap", "restore", "sow", "stem"]

_SECTIONS = {"dial", "explode", "optim", "probe", "reap", "restore", "sow", "stem", "model"}


def __getattr__(name: str) -> Any:
    if name in _SECTIONS:
        return importlib.import_module(f"{__name__}.{name}")
    if name == "load":
        return importlib.import_module(f"{__name__}.model").load
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
