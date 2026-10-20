"""hypernix.c4.reap — gathering the best layers and weights for :mod:`~hypernix.c4.sow`.

A reap reads what :mod:`~hypernix.c4.stem` and :mod:`~hypernix.c4.dial`
found and decides what the new model is made of:

* **layers** — which mstems to keep, in order, from one model or from
  several models of the same shape (:func:`reap_best` takes, at every
  depth, the donor whose layer scored best there: cherry-picked layers);
* **dials** — which weights are unneeded and will be zeroed;
* **channels** — which feed-forward channels each kept layer keeps, when
  a structured prune is asked for (rows of ``ffn_gate``/``ffn_up`` and
  the matching columns of ``ffn_down``, scored together so a channel
  goes or stays as a unit).

The result is a :class:`Harvest`: a small JSON manifest plus a
safetensors file of masks, written by :meth:`Harvest.save`, so a reap
can be reviewed, edited and sown later or elsewhere.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .explode import Exploded
from .model import C4Error
from .stem import LayerVerdict, layer_report

__all__ = ["Harvest", "channel_scores", "reap", "reap_best"]


@dataclass
class Harvest:
    """What a sown model is made of."""

    donors: list[str]
    #: The new model's layers, in order: (donor index, that donor's layer).
    layers: list[tuple[int, int]]
    #: Which donor the embeddings, head and final norm come from.
    loose_from: int = 0
    #: (donor index, stem name) -> bool mask of dials to zero (True = unneeded).
    masks: dict[tuple[int, str], Any] = field(default_factory=dict)
    #: New layer position -> the feed-forward channels it keeps, sorted.
    channels: dict[int, list[int]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def zeroed(self) -> int:
        return int(sum(int(m.sum()) for m in self.masks.values()))

    def summary(self) -> dict[str, Any]:
        return {
            "donors": self.donors, "layers": [list(x) for x in self.layers],
            "loose_from": self.loose_from, "zeroed_dials": self.zeroed,
            "ffn_channels": (len(next(iter(self.channels.values()))) if self.channels else None),
            "reasons": self.reasons,
        }

    def save(self, folder: str | Path) -> Path:
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {**self.summary(), "channels": {str(k): v for k, v in self.channels.items()},
                    "scores": self.scores,
                    "masks": [{"donor": d, "stem": s} for d, s in self.masks]}
        (out / "harvest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if self.masks:
            import torch
            from safetensors.torch import save_file

            save_file({f"{d}::{s}": m.to("cpu").to(torch.uint8).contiguous()
                       for (d, s), m in self.masks.items()}, str(out / "harvest-masks.safetensors"))
        return out

    @classmethod
    def load(cls, folder: str | Path) -> Harvest:
        out = Path(folder)
        data = json.loads((out / "harvest.json").read_text(encoding="utf-8"))
        masks: dict[tuple[int, str], Any] = {}
        if (out / "harvest-masks.safetensors").is_file():
            from safetensors.torch import load_file

            for key, value in load_file(str(out / "harvest-masks.safetensors")).items():
                donor, stem = key.split("::", 1)
                masks[(int(donor), stem)] = value.bool()
        return cls(donors=data["donors"], layers=[tuple(x) for x in data["layers"]],
                   loose_from=int(data.get("loose_from", 0)), masks=masks,
                   channels={int(k): v for k, v in data.get("channels", {}).items()},
                   reasons=list(data.get("reasons", [])), scores=dict(data.get("scores", {})))


def channel_scores(exploded: Exploded, layer: int, dial_scores: Any = None) -> Any:
    """One score per feed-forward channel of *layer*: the summed dial scores
    of its ``ffn_gate`` and ``ffn_up`` rows and its ``ffn_down`` column
    (weight magnitudes when no scores are given)."""
    import torch

    mstem = exploded.mstems[layer]
    up, gate, down = mstem.stem("ffn_up"), mstem.stem("ffn_gate"), mstem.stem("ffn_down")
    if up is None or down is None:
        raise C4Error(f"layer {layer} has no ffn_up/ffn_down to prune channels of")

    def values(stem: Any) -> Any:
        if dial_scores is not None and stem.name in dial_scores.scores:
            return dial_scores.scores[stem.name].float()
        return exploded.tensor(stem).detach().float().abs()

    total = values(up).sum(dim=1) + values(down).sum(dim=0)
    if gate is not None:
        total = total + values(gate).sum(dim=1)
    return total.to(torch.float32)


def _pick_layers(report: list[LayerVerdict], keep: int | None, drop: int | None,
                 drop_unneeded: bool) -> list[int]:
    count = len(report)
    if keep is None and drop is None:
        keep = count - sum(1 for v in report if v.verdict == "unneeded") if drop_unneeded else count
    if drop is not None:
        keep = count - drop
    keep = max(1, min(count, int(keep)))
    ranked = sorted(report, key=lambda v: v.score, reverse=True)[:keep]
    return sorted(v.index for v in ranked)


def reap(exploded: Exploded, *, report: list[LayerVerdict] | None = None, prober: Any = None,
         keep_layers: int | None = None, drop_layers: int | None = None, drop_unneeded: bool = True,
         unneeded: Any = None, dial_scores: Any = None, ffn_keep: float = 1.0,
         mask_loose: bool = False) -> Harvest:
    """Reap one model.

    Layers: the *keep_layers* best (or all but *drop_layers*; by default
    all but those the tests found the model better without). Dials: the
    masks of an :func:`~hypernix.c4.dial.unneeded` result. Channels: with
    *ffn_keep* below 1, that share of each kept layer's feed-forward
    channels, the best by :func:`channel_scores`. Unneeded dials of the
    embeddings and head are left alone unless *mask_loose*: a zeroed
    embedding row is a token the model can no longer read.
    """
    import torch

    report = report or layer_report(exploded, prober)
    kept = _pick_layers(report, keep_layers, drop_layers, drop_unneeded)
    harvest = Harvest(donors=[str(exploded.model.path)], layers=[(0, i) for i in kept])
    harvest.scores = {f"layer.{v.index}": v.score for v in report}
    dropped = sorted(set(range(len(report))) - set(kept))
    if dropped:
        harvest.reasons.append(f"dropped layers {dropped}: the lowest scores "
                               f"({', '.join(f'{report[i].score:+.4f}' for i in dropped)})")
    if unneeded is not None:
        for name, mask in unneeded.masks.items():
            layer = exploded.stems[name].layer
            if (layer < 0 and mask_loose) or layer in kept:
                harvest.masks[(0, name)] = mask.to("cpu")
        harvest.reasons.append(f"{harvest.zeroed} unneeded dials ({unneeded.kind}, "
                               f"lowest {unneeded.fraction:.0%} per stem) will be zeroed")
    if ffn_keep < 1.0:
        widths = [exploded.mstems[i].stem("ffn_up").fan_out for i in kept]
        keep_n = max(1, int(round(min(widths) * ffn_keep)))
        for position, layer in enumerate(kept):
            ranks = channel_scores(exploded, layer, dial_scores)
            harvest.channels[position] = sorted(torch.topk(ranks, keep_n).indices.tolist())
        harvest.reasons.append(f"kept the best {keep_n} of {min(widths)} feed-forward channels per layer")
    return harvest


def reap_best(explodeds: Sequence[Exploded], reports: Sequence[list[LayerVerdict]] | None = None,
              *, probers: Sequence[Any] | None = None, layers: int | None = None) -> Harvest:
    """Cherry-pick layers from several same-shaped models.

    At every depth, the donor whose layer scored best there. The models
    must share their layer count and every stem's shape; their scores
    must be comparable, so test them on the same calibration set.
    """
    if len(explodeds) < 2:
        raise C4Error("reap_best compares two or more models; use reap for one")
    first = explodeds[0]
    for other in explodeds[1:]:
        if len(other.mstems) != len(first.mstems):
            raise C4Error("the models have different numbers of layers")
        for a, b in zip(first.mstems, other.mstems, strict=True):
            if sorted((s.role, s.shape) for s in a.stems) != sorted((s.role, s.shape) for s in b.stems):
                raise C4Error(f"layer {a.index} differs in shape between the models")
    if reports is None:
        probers = probers or [None] * len(explodeds)
        reports = [layer_report(e, p) for e, p in zip(explodeds, probers, strict=True)]
    depth = len(first.mstems)
    picks: list[tuple[int, int]] = []
    reasons = []
    for i in range(depth):
        donor = max(range(len(explodeds)), key=lambda d: reports[d][i].score)
        picks.append((donor, i))
        reasons.append(f"layer {i} from donor {donor} ({reports[donor][i].score:+.4f})")
    if layers is not None and layers < depth:
        best = sorted(range(depth), key=lambda i: reports[picks[i][0]][i].score, reverse=True)[:layers]
        picks = [picks[i] for i in sorted(best)]
    harvest = Harvest(donors=[str(e.model.path) for e in explodeds], layers=picks, reasons=reasons)
    return harvest
