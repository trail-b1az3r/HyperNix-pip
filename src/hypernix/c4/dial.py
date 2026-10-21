"""hypernix.c4.dial — which parameters matter, which do not, and which to train.

Every score here is per parameter (per dial), computed once over the
calibration set with one backward pass per batch, and kept on the GPU
beside the weight it scores:

``magnitude``
    ``|w|``. Needs no model run; a weak signal on its own.
``taylor``
    ``|w · g|`` — the first-order estimate of how much the loss changes
    if the weight is removed (LeCun et al. 1989; Molchanov et al. 2019).
``fisher``
    ``g²`` accumulated over batches — how sensitive the loss is to the
    weight moving at all (an empirical Fisher diagonal).
``wanda``
    ``|w| · ||x||`` — the weight times the norm of the input it multiplies
    (Sun et al. 2023), measured with forward hooks; no backward pass.

From those, :func:`unneeded` picks the dials (or rows: Pdials) that can
go, :func:`train_priority` ranks stems, Pdials or mstems by how much one
more step of training on them would be expected to lower the loss, and
:func:`interaction` tests whether an mdial's two weights matter more
together than apart.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .explode import Exploded, MDial, PDial
from .model import C4Error

__all__ = [
    "SCORES",
    "DialScores",
    "Priority",
    "interaction",
    "score",
    "train_priority",
    "unneeded",
]

SCORES = ("magnitude", "taylor", "fisher", "wanda")


@dataclass
class DialScores:
    """A score per parameter, by stem name, and the gradient statistics behind it."""

    kind: str
    scores: dict[str, Any]
    #: Per stem: mean gradient (accumulated over batches), for priorities.
    grads: dict[str, Any] = field(default_factory=dict)
    batches: int = 0
    synthetic: bool = False

    def stem_total(self, name: str) -> float:
        return float(self.scores[name].float().sum())

    def top(self, k: int = 10, *, largest: bool = True) -> list[tuple[str, tuple[int, ...], float]]:
        """The *k* highest (or lowest) scoring dials across the model."""
        import torch

        found: list[tuple[str, tuple[int, ...], float]] = []
        for name, values in self.scores.items():
            flat = values.float().reshape(-1)
            count = min(k, flat.numel())
            vals, idx = torch.topk(flat, count, largest=largest)
            for v, i in zip(vals.tolist(), idx.tolist(), strict=True):
                found.append((name, tuple(int(x) for x in _unravel(i, tuple(values.shape))), v))
        found.sort(key=lambda item: item[2], reverse=largest)
        return found[:k]


def _unravel(index: int, shape: tuple[int, ...]) -> list[int]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index //= dim
    return list(reversed(out)) or [index]


def _selected(exploded: Exploded, stems: Iterable[str] | None, matrices_only: bool) -> list[str]:
    names = list(stems) if stems is not None else list(exploded.stems)
    if matrices_only:
        names = [n for n in names if exploded.stems[n].kind == "matrix"]
    return names


def score(exploded: Exploded, prober: Any = None, *, kind: str = "taylor",
          stems: Iterable[str] | None = None, matrices_only: bool = True) -> DialScores:
    """Score every dial of *stems* (default: every matrix stem)."""
    import torch

    if kind not in SCORES:
        raise C4Error(f"score dials by one of {', '.join(SCORES)}, not {kind!r}")
    names = _selected(exploded, stems, matrices_only)
    if kind == "magnitude":
        return DialScores(kind, {n: exploded.tensor(n).detach().abs().to(torch.float16) for n in names})
    if prober is None:
        raise C4Error(f"{kind} scores need the model to run: give a Prober")
    synthetic = bool(prober.calibration.synthetic)
    if kind == "wanda":
        return _wanda(exploded, prober, names, synthetic)

    params = dict(prober.module.named_parameters())
    missing = [n for n in names if n not in params]
    if missing:
        raise C4Error(f"these stems are not parameters of the running model: {missing[:3]}")
    accum = {n: torch.zeros_like(params[n], dtype=torch.float32) for n in names}
    grads = {n: torch.zeros_like(params[n], dtype=torch.float32) for n in names}
    batches = 0
    for batch in prober.calibration.batches:
        prober.module.zero_grad(set_to_none=True)
        out = prober.module(input_ids=batch.to(exploded.model.device), labels=batch.to(exploded.model.device))
        out.loss.backward()
        batches += 1
        with torch.no_grad():
            for n in names:
                g = params[n].grad
                if g is None:
                    continue
                g = g.float()
                grads[n] += g
                accum[n] += (params[n].detach().float() * g).abs() if kind == "taylor" else g.pow(2)
    prober.module.zero_grad(set_to_none=True)
    scale = 1.0 / max(batches, 1)
    return DialScores(kind, {n: (accum[n] * scale).to(torch.float16) for n in names},
                      grads={n: grads[n] * scale for n in names}, batches=batches, synthetic=synthetic)


def _wanda(exploded: Exploded, prober: Any, names: list[str], synthetic: bool) -> DialScores:
    import torch

    modules = {f"{n}.weight": m for n, m in prober.module.named_modules() if isinstance(m, torch.nn.Linear)}
    norms: dict[str, Any] = {}
    hooks = []

    def hook(name: str):
        def record(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            norms[name] = norms.get(name, 0) + x.pow(2).sum(dim=0)
        return record

    for name in names:
        if name in modules:
            hooks.append(modules[name].register_forward_hook(hook(name)))
    try:
        prober.loss()
    finally:
        for h in hooks:
            h.remove()
    scores = {}
    for name in names:
        w = exploded.tensor(name).detach().float()
        if name in norms:
            scores[name] = (w.abs() * norms[name].sqrt().unsqueeze(0)).to(torch.float16)
        else:
            scores[name] = w.abs().to(torch.float16)
    return DialScores("wanda", scores, synthetic=synthetic)


@dataclass
class Unneeded:
    """Dials (as masks: True = unneeded) and the rows among them that can go whole."""

    masks: dict[str, Any]
    rows: dict[str, list[int]]
    fraction: float
    kind: str

    @property
    def dials(self) -> int:
        return int(sum(int(m.sum()) for m in self.masks.values()))

    def pdials(self) -> list[PDial]:
        return [PDial(name, "row", r, r + 1) for name, rows in self.rows.items() for r in rows]


def unneeded(scores: DialScores, *, fraction: float = 0.1, per_stem: bool = True,
             row_fraction: float = 0.9) -> Unneeded:
    """The lowest-scoring *fraction* of dials, per stem (or across the model).

    A row (Pdial) whose dials are almost all unneeded (*row_fraction*) is
    listed whole, since a whole row can be removed rather than zeroed.
    """
    import torch

    if not 0.0 < fraction < 1.0:
        raise C4Error("fraction is between 0 and 1")
    masks: dict[str, Any] = {}
    if per_stem:
        for name, values in scores.scores.items():
            flat = values.float().reshape(-1)
            k = int(flat.numel() * fraction)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            if k:
                mask[torch.topk(flat, k, largest=False).indices] = True
            masks[name] = mask.reshape(values.shape)
    else:
        everything = torch.cat([v.float().reshape(-1) for v in scores.scores.values()])
        threshold = float(torch.quantile(everything[torch.randperm(everything.numel())[:1_000_000]]
                                         if everything.numel() > 1_000_000 else everything, fraction))
        masks = {n: v.float() <= threshold for n, v in scores.scores.items()}
    rows = {}
    for name, mask in masks.items():
        if mask.dim() >= 2:
            share = mask.float().reshape(mask.shape[0], -1).mean(dim=1)
            picked = torch.nonzero(share >= row_fraction).reshape(-1).tolist()
            if picked:
                rows[name] = picked
    return Unneeded(masks, rows, fraction, scores.kind)


@dataclass
class Priority:
    piece: str
    granularity: str
    gain: float          # expected loss decrease per unit step: ||g||^2 / numel
    grad_norm: float
    weight_norm: float
    relative: float      # ||g|| / ||w||: high = far from settled
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("piece", "granularity", "gain", "grad_norm",
                                              "weight_norm", "relative", "reason")}


def train_priority(exploded: Exploded, scores: DialScores, *, granularity: str = "stem",
                   top: int = 20) -> list[Priority]:
    """What to train first: the pieces where a step would lower the loss most.

    One gradient step of size η on a piece lowers the loss by about
    η·||g||² (first order), so the gain per parameter is ||g||²/n. A large
    ||g||/||w|| says the piece is far from where the loss wants it — the
    ones that "specifically need retraining". Needs Taylor or Fisher
    scores (which carry the gradients).
    """
    import torch

    if not scores.grads:
        raise C4Error("train priorities need gradients: score with kind='taylor' or 'fisher'")
    found: list[Priority] = []

    def add(label: str, g: Any, w: Any, gran: str) -> None:
        gn, wn = float(g.norm()), float(w.norm())
        found.append(Priority(label, gran, float(g.pow(2).sum()) / max(g.numel(), 1), gn, wn,
                              gn / max(wn, 1e-12)))

    if granularity == "stem":
        for name, g in scores.grads.items():
            add(name, g, exploded.tensor(name).float(), "stem")
    elif granularity == "pdial":
        for name, g in scores.grads.items():
            if g.dim() < 2:
                continue
            w = exploded.tensor(name).float()
            gain = g.pow(2).reshape(g.shape[0], -1).mean(dim=1)
            best = torch.topk(gain, min(top, gain.numel())).indices.tolist()
            for r in best:
                add(f"{name}[{r}, :]", g[r], w[r], "pdial")
    elif granularity == "mstem":
        for mstem in exploded.mstems:
            parts = [n.name for n in mstem.stems if n.name in scores.grads]
            if not parts:
                continue
            g = torch.cat([scores.grads[n].reshape(-1) for n in parts])
            w = torch.cat([exploded.tensor(n).float().reshape(-1) for n in parts])
            add(mstem.name, g, w, "mstem")
    else:
        raise C4Error("granularity is stem, pdial or mstem")
    found.sort(key=lambda p: p.gain, reverse=True)
    if found:
        typical = sorted(p.relative for p in found)[len(found) // 2]
        for p in found:
            if p.relative > 3 * typical:
                p.reason = "needs retraining: its gradient is large for its weights"
            elif p.gain >= found[0].gain * 0.5:
                p.reason = "most to gain from more training"
            else:
                p.reason = "settled"
    return found[:top]


def interaction(exploded: Exploded, prober: Any, mdial: MDial, *, epsilon: float = 1e-2) -> dict[str, float]:
    """Does an mdial's pair matter more together than apart?

    Nudges each weight by *epsilon* (relative to the stem's spread),
    then both, and reports ``synergy = ΔL(both) - ΔL(first) - ΔL(second)``:
    near zero means they act independently; large means the pair works
    together (a second-order effect a per-dial score cannot see).
    """
    import torch

    tensor = exploded.tensor(mdial.stem)
    step = float(tensor.float().std()) * epsilon or epsilon
    base = prober.loss()

    def nudged(indices: list[tuple[int, ...]]) -> float:
        with torch.no_grad():
            saved = [float(tensor[i]) for i in indices]
            for i in indices:
                tensor[i] += step
        try:
            return prober.loss() - base
        finally:
            with torch.no_grad():
                for i, v in zip(indices, saved, strict=True):
                    tensor[i] = v

    first = nudged([mdial.first])
    second = nudged([mdial.second])
    both = nudged([mdial.first, mdial.second])
    return {"first": first, "second": second, "both": both, "synergy": both - first - second,
            "base_loss": base}
