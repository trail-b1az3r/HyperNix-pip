"""hypernix.c4.stem — measuring and testing stems and mstems.

Two kinds of evidence, kept apart because they answer different
questions:

**Measured** (weights only, no model run): a stem's spread, outliers,
sparsity, effective and stable rank, how much a 4-bit round trip would
cost it, and how alike two layers' stems are. Cheap, and available for
any model c4 can load.

**Tested** (needs a :class:`~hypernix.c4.probe.Prober`): the change in
calibration loss when a stem or a whole layer is taken out
(:func:`ablation_test`), and each layer's *block influence* — one minus
the cosine similarity between the hidden state entering a layer and the
one leaving it. A layer whose output is nearly its input is doing
little, and is the first candidate to drop (Men et al., 2024, ShortGPT).

:func:`layer_report` puts both together per mstem and ranks the layers
from most to least needed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from .explode import Exploded, MStem, Stem
from .model import C4Error
from .restore import ablated

__all__ = [
    "LayerVerdict",
    "StemStats",
    "ablation_test",
    "block_influence",
    "layer_report",
    "measure",
    "measure_all",
    "redundancy",
]

#: Above this many elements, SVD runs on a random sketch of the rows.
_SVD_LIMIT = 4096


@dataclass
class StemStats:
    name: str
    role: str
    layer: int
    numel: int
    mean: float
    std: float
    abs_max: float
    l2: float
    sparsity: float              # share of |w| below 1e-3 x the stem's std
    outliers: float              # share of |w| beyond 6 std
    kurtosis: float
    effective_rank: float = 0.0  # exp(entropy of the normalised singular values)
    stable_rank: float = 0.0     # ||W||_F^2 / ||W||_2^2
    rank_ratio: float = 0.0      # effective rank / min(shape)
    q4_error: float = 0.0        # relative RMS error of a symmetric 4-bit round trip

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _svdvals(matrix: Any) -> Any:
    import torch

    m = matrix.float()
    rows, cols = m.shape
    if min(rows, cols) > _SVD_LIMIT:
        generator = torch.Generator(device="cpu").manual_seed(0)
        sketch = torch.randn(_SVD_LIMIT, rows, generator=generator).to(m.device) / math.sqrt(_SVD_LIMIT)
        m = sketch @ m
    return torch.linalg.svdvals(m)


def _q4_error(values: Any) -> float:
    import torch

    flat = values.float().reshape(-1)
    pad = (-flat.numel()) % 32
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, 32)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7.0
    restored = torch.clamp(torch.round(blocks / scale), -8, 7) * scale
    return float((restored - blocks).pow(2).mean().sqrt() / blocks.pow(2).mean().sqrt().clamp(min=1e-12))


def measure(exploded: Exploded, stem: Stem | str, *, spectral: bool = True) -> StemStats:
    """A stem's weight statistics. Runs where the stem lives (the GPU)."""
    import torch

    info = exploded.stems[stem.name if isinstance(stem, Stem) else stem]
    w = exploded.tensor(info).detach().float()
    flat = w.reshape(-1)
    std = float(flat.std()) if flat.numel() > 1 else 0.0
    mean = float(flat.mean())
    centred = flat - mean
    kurt = float(centred.pow(4).mean() / centred.pow(2).mean().pow(2).clamp(min=1e-24)) if std else 0.0
    stats = StemStats(
        name=info.name, role=info.role, layer=info.layer, numel=info.numel,
        mean=mean, std=std, abs_max=float(flat.abs().max()), l2=float(flat.norm()),
        sparsity=float((flat.abs() < 1e-3 * (std or 1.0)).float().mean()),
        outliers=float((centred.abs() > 6 * std).float().mean()) if std else 0.0,
        kurtosis=kurt, q4_error=_q4_error(flat),
    )
    if spectral and info.kind == "matrix":
        with torch.no_grad():
            s = _svdvals(w.reshape(info.shape[0], -1))
        if s.numel() and float(s[0]) > 0:
            p = s / s.sum()
            entropy = float(-(p * torch.log(p.clamp(min=1e-30))).sum())
            stats.effective_rank = math.exp(entropy)
            stats.stable_rank = float((s.pow(2).sum()) / s[0].pow(2))
            stats.rank_ratio = stats.effective_rank / min(info.shape[0], info.shape[1])
    return stats


def measure_all(exploded: Exploded, *, spectral: bool = True) -> dict[str, StemStats]:
    return {name: measure(exploded, name, spectral=spectral) for name in exploded.stems}


def redundancy(exploded: Exploded, a: MStem, b: MStem) -> float:
    """Mean cosine similarity of two layers' same-role stems (1 = identical)."""
    import torch

    sims = []
    for stem in a.stems:
        twin = b.stem(stem.role)
        if twin is None or twin.shape != stem.shape or stem.kind != "matrix":
            continue
        x = exploded.tensor(stem).float().reshape(-1)
        y = exploded.tensor(twin).float().reshape(-1)
        sims.append(float(torch.nn.functional.cosine_similarity(x, y, dim=0)))
    return sum(sims) / len(sims) if sims else 0.0


def block_influence(prober: Any) -> list[float]:
    """Per layer, 1 - cos(input hidden state, output hidden state), averaged over tokens."""
    import torch

    states = prober.hidden_states()
    scores = []
    for before, after in zip(states[:-1], states[1:], strict=True):
        cos = torch.nn.functional.cosine_similarity(before.float(), after.float(), dim=-1)
        scores.append(float(1.0 - cos.mean()))
    return scores


@dataclass
class TestResult:
    piece: str
    how: str
    base_loss: float
    loss: float

    @property
    def delta(self) -> float:
        """How much worse the model got without the piece (negative = better without it)."""
        return self.loss - self.base_loss

    def to_dict(self) -> dict[str, Any]:
        return {"piece": self.piece, "how": self.how, "base_loss": self.base_loss,
                "loss": self.loss, "delta": self.delta}


def ablation_test(exploded: Exploded, prober: Any, pieces: Iterable[Any], *, how: str = "zero",
                  base_loss: float | None = None) -> list[TestResult]:
    """Take each piece out in turn, measure the loss, put it back.

    Every piece is restored before the next is tested, and the model is
    left exactly as it was found.
    """
    base = prober.loss() if base_loss is None else base_loss
    results = []
    for piece in pieces:
        with ablated(exploded, piece, how):
            loss = prober.loss()
        label = piece.name if hasattr(piece, "name") else str(piece)
        results.append(TestResult(label, how, base, loss))
    return results


@dataclass
class LayerVerdict:
    index: int
    numel: int
    influence: float | None = None       # block influence (tested)
    ablation_delta: float | None = None  # loss change without the layer (tested)
    redundancy_next: float = 0.0         # similarity to the next layer (measured)
    mean_rank_ratio: float = 0.0         # measured
    score: float = 0.0                   # higher = more needed
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def layer_report(exploded: Exploded, prober: Any = None, *, test: bool = True,
                 how: str = "zero", spectral: bool = False) -> list[LayerVerdict]:
    """Every mstem, measured (and tested when there is a prober), ranked.

    The score is the ablation delta when tested (what removing it costs),
    else the block influence, else the measured stand-in: low effective
    rank and high similarity to the next layer both read as less needed.
    """
    if not exploded.mstems:
        raise C4Error("this model has no layers to report on")
    verdicts = [LayerVerdict(m.index, m.numel) for m in exploded.mstems]
    for i, (mstem, verdict) in enumerate(zip(exploded.mstems, verdicts, strict=True)):
        if i + 1 < len(exploded.mstems):
            verdict.redundancy_next = redundancy(exploded, mstem, exploded.mstems[i + 1])
        if spectral:
            ratios = [measure(exploded, s).rank_ratio for s in mstem.stems if s.kind == "matrix"]
            verdict.mean_rank_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    if prober is not None:
        influence = block_influence(prober)
        for verdict, value in zip(verdicts, influence, strict=False):
            verdict.influence = value
        if test:
            base = prober.loss()
            for verdict, result in zip(verdicts, ablation_test(exploded, prober, exploded.mstems,
                                                               how=how, base_loss=base), strict=True):
                verdict.ablation_delta = result.delta
    for verdict in verdicts:
        if verdict.ablation_delta is not None:
            verdict.score = verdict.ablation_delta
            verdict.reasons.append(f"removing it changes the loss by {verdict.ablation_delta:+.4f}")
        elif verdict.influence is not None:
            verdict.score = verdict.influence
            verdict.reasons.append(f"block influence {verdict.influence:.4f}")
        else:
            verdict.score = (1.0 - verdict.redundancy_next) + verdict.mean_rank_ratio
            verdict.reasons.append("weights only: similarity to the next layer and rank")
        if verdict.redundancy_next > 0.9:
            verdict.reasons.append(f"{verdict.redundancy_next:.2f} alike the next layer")
    ranked = sorted(verdicts, key=lambda v: v.score)
    quarter = max(1, len(ranked) // 4)
    for position, verdict in enumerate(ranked):
        if verdict.ablation_delta is not None and verdict.ablation_delta < 0:
            verdict.verdict = "unneeded"      # the model is better without it
        elif position < quarter:
            verdict.verdict = "candidate to drop"
        elif position >= len(ranked) - quarter:
            verdict.verdict = "keep"
        else:
            verdict.verdict = "needed"
    return verdicts
