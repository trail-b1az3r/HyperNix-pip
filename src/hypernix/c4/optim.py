"""The c4 optimizers: back-propagate, separate, check, and only then update.

An ordinary optimizer turns every gradient into an update as soon as it
has one. c4's work is surgical — retraining the stems, Pdials or layers
:func:`~hypernix.c4.dial.train_priority` picked, healing a sown model —
and one bad section should not drag the rest with it. So
:class:`C4Optimizer` puts a step between the backward pass and the
update:

1. **separate** the gradients into sections: stems, mstems, or each
   stem's rows (Pdials);
2. **check** each section: a non-finite gradient skips that section (not
   the whole step); a gradient norm far above the section's own running
   norm is clipped back to it; a section that has had no gradient for
   ``dead_patience`` steps is frozen; unneeded dials are held where they
   are; sections outside ``only`` do not move at all;
3. **update** with the inner optimizer, then scale each section's change
   by its priority and put frozen and masked parameters back exactly.

The inner optimizer is one of three, built for this and not for
ordinary training:

``hybrid`` — :class:`C4HybridAdamW5S`
    AdamW's two moments (with bias correction and decoupled weight
    decay) whose step is damped by Pressure Cooker V5S's three-band
    oscillation score and bounded by a per-tensor trust ratio, with
    V5S's coordinate freeze for coordinates whose gradient has gone
    quiet. AdamW converges on few steps; V5S keeps a fine-tune from
    oscillating on them.
``v5s`` — :class:`C4CookerV5S`
    Pressure Cooker V5S with pressure diffusion off (diffusion blends
    neighbouring coordinates, which would leak an update across the
    Pdial boundaries c4 keeps separate), no global clip (sections are
    clipped individually), quicker coordinate freezing, and masked
    coordinates kept out of its momentum.
``adamw`` — ``torch.optim.AdamW``, for comparison.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..optimizers.pressure_cooker_v5 import _dequantize_momentum, _quantize_momentum
from ..optimizers.pressure_cooker_v5s import PressureCookerV5S, volumetric_oscillation_score
from .model import C4Error

__all__ = [
    "C4CookerV5S",
    "C4HybridAdamW5S",
    "C4Optimizer",
    "CheckPolicy",
    "Section",
    "make_inner",
]


class C4HybridAdamW5S(torch.optim.Optimizer):
    """AdamW moments, damped by V5S's oscillation score (see the module docstring)."""

    def __init__(self, params: Iterable[Any], *, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, vos_gain: float = 3.0,
                 bands: tuple[float, float, float] = (0.80, 0.95, 0.999),
                 band_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
                 trust_clip: tuple[float, float] | None = (0.05, 5.0),
                 freeze_threshold: float = 1e-8, freeze_patience: int = 16,
                 freeze_decay: float = 0.9) -> None:
        if lr <= 0:
            raise C4Error("lr must be positive")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.vos_gain = vos_gain
        self.bands = bands
        self.band_weights = band_weights
        self.trust_clip = trust_clip
        self.freeze_threshold = freeze_threshold
        self.freeze_patience = freeze_patience
        self.freeze_decay = freeze_decay

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                    state["cos"] = [0.0, 0.0, 0.0]
                    state["quiet"] = torch.zeros_like(p, dtype=torch.int16)
                    state["freeze"] = torch.ones_like(p, dtype=torch.float32)
                state["step"] += 1
                m, v = state["m"], state["v"]
                # The oscillation score compares this gradient with the momentum *before* it.
                if state["step"] > 1:
                    g_norm, m_norm = g.norm(), m.norm()
                    raw = float((g * m).sum() / (g_norm * m_norm).clamp(min=1e-30))
                    state["cos"] = [b * c + (1 - b) * raw for b, c in zip(self.bands, state["cos"], strict=True)]
                vos = volumetric_oscillation_score(*state["cos"], *self.band_weights)
                damp = 1.0 / (1.0 + max(vos, 0.0) * self.vos_gain)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** state["step"])
                v_hat = v / (1 - beta2 ** state["step"])
                update = m_hat / (v_hat.sqrt() + group["eps"])
                # V5S's coordinate freeze: quiet coordinates fade out.
                quiet = g.abs() < self.freeze_threshold
                state["quiet"] = torch.where(quiet, state["quiet"] + 1, torch.zeros_like(state["quiet"]))
                stale = state["quiet"] >= self.freeze_patience
                state["freeze"] = torch.where(stale, state["freeze"] * self.freeze_decay,
                                              torch.ones_like(state["freeze"]))
                update = update * state["freeze"]
                trust = 1.0
                if self.trust_clip is not None:
                    w_norm, u_norm = float(p.float().norm()), float(update.norm())
                    if w_norm > 0 and u_norm > 0:
                        trust = min(max(w_norm / u_norm, self.trust_clip[0]), self.trust_clip[1])
                lr = group["lr"] * damp * trust
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-lr)
                state["vos"] = vos
                state["damp"] = damp
        return loss


class C4CookerV5S(PressureCookerV5S):
    """Pressure Cooker V5S, changed for c4 (see the module docstring)."""

    def __init__(self, params: Iterable[Any], *, lr: float = 5e-5, **kwargs: Any) -> None:
        kwargs.setdefault("diffusion_factor", 0.0)
        kwargs.setdefault("grad_clip", None)
        kwargs.setdefault("freeze_patience", 8)
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(params, lr=lr, **kwargs)

    def _v5s_step_one(self, p: Any, g: Any, state: dict[str, Any], lr: float, wd: float) -> None:
        keep = state.get("c4_keep")
        if keep is not None:
            # Masked coordinates contribute nothing, now or through momentum later.
            m = _dequantize_momentum(state["m_q"], state["m_scale"]).view_as(p) * keep
            state["m_q"], state["m_scale"] = _quantize_momentum(m)
            g = g * keep
        super()._v5s_step_one(p, g, state, lr, wd)

    def _extra_scale(self, state: dict[str, Any], g_pred: Any, vos: float) -> float:
        return float(state.get("c4_scale", 1.0))


def make_inner(kind: str, params: Iterable[Any], *, lr: float = 1e-4, **kwargs: Any) -> torch.optim.Optimizer:
    kind = (kind or "hybrid").lower()
    if kind == "hybrid":
        return C4HybridAdamW5S(params, lr=lr, **kwargs)
    if kind in ("v5s", "pressure-cooker-v5s", "cooker"):
        return C4CookerV5S(params, lr=lr, **kwargs)
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    raise C4Error(f"the c4 optimizer's inner step is hybrid, v5s or adamw, not {kind!r}")


@dataclass
class Section:
    """A named group of parameters checked together; ``rows`` = each row its own Pdial."""

    name: str
    params: list[Any]
    rows: bool = False


@dataclass
class CheckPolicy:
    """How sections are checked before an update."""

    #: Clip a section whose gradient norm exceeds its running mean by this many std devs.
    spike_z: float = 4.0
    #: Steps of history before spike checks start.
    warmup: int = 5
    #: A section with gradient norm below this for `dead_patience` steps is frozen.
    dead_norm: float = 0.0
    dead_patience: int = 0
    #: Only these sections may move (None = all).
    only: set[str] | None = None
    #: Per-section multiplier on the update (1 = as the inner optimizer says).
    priorities: dict[str, float] = field(default_factory=dict)


class C4Optimizer:
    """Separate, check, then update. See the module docstring."""

    def __init__(self, sections: Sequence[Section], inner: torch.optim.Optimizer, *,
                 policy: CheckPolicy | None = None, masks: dict[Any, Any] | None = None) -> None:
        self.sections = list(sections)
        self.inner = inner
        self.policy = policy or CheckPolicy()
        #: param -> bool mask of dials held fixed (True = hold).
        self.masks = {p: m.to(p.device) for p, m in (masks or {}).items()}
        self._history: dict[str, list[float]] = {s.name: [] for s in self.sections}
        self._quiet: dict[str, int] = {s.name: 0 for s in self.sections}
        self.frozen: set[str] = set()
        self.steps = 0
        self.skipped: dict[str, int] = {}
        self.clipped: dict[str, int] = {}
        for p, mask in self.masks.items():
            if isinstance(inner, C4CookerV5S):
                inner.state[p]["c4_keep"] = (~mask).to(torch.float32)

    # -- construction ---------------------------------------------------

    @classmethod
    def for_exploded(cls, exploded: Any, module: Any, *, kind: str = "hybrid", lr: float = 1e-4,
                     granularity: str = "stem", only: Iterable[str] | None = None,
                     masks: dict[str, Any] | None = None, priorities: dict[str, float] | None = None,
                     policy: CheckPolicy | None = None, **inner_kwargs: Any) -> C4Optimizer:
        """Sections from an exploded model: ``stem``, ``mstem`` or ``pdial`` (rows)."""
        params = dict(module.named_parameters())
        sections: list[Section] = []
        if granularity == "mstem":
            used = set()
            for mstem in exploded.mstems:
                group = [params[s.name] for s in mstem.stems if s.name in params]
                used.update(s.name for s in mstem.stems)
                if group:
                    sections.append(Section(mstem.name, group))
            for name, p in params.items():
                if name not in used:
                    sections.append(Section(name, [p]))
        elif granularity in ("stem", "pdial"):
            for name, p in params.items():
                sections.append(Section(name, [p], rows=granularity == "pdial" and p.dim() >= 2))
        else:
            raise C4Error("sections are stem, mstem or pdial")
        if only is not None:
            wanted = {n.split("[")[0] for n in only}
            for name, p in params.items():
                p.requires_grad_(any(name == w or name.startswith(w) for w in wanted)
                                 or any(s.name in wanted and any(q is p for q in s.params) for s in sections))
        pol = policy or CheckPolicy()
        if only is not None:
            pol.only = {n.split("[")[0] for n in only}
        if priorities:
            pol.priorities.update(priorities)
        trainable = [p for p in params.values() if p.requires_grad]
        inner = make_inner(kind, trainable, lr=lr, **inner_kwargs)
        param_masks = {params[n]: m for n, m in (masks or {}).items() if n in params}
        return cls(sections, inner, policy=pol, masks=param_masks)

    # -- the step -------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def _allowed(self, section: Section) -> bool:
        if section.name in self.frozen:
            return False
        only = self.policy.only
        return only is None or section.name in only

    def _check(self, section: Section) -> float:
        """Check one section's gradient in place. Returns the multiplier its update gets."""
        grads = [p.grad for p in section.params if p.grad is not None]
        if not grads:
            return 0.0
        if not self._allowed(section):
            for g in grads:
                g.zero_()
            return 0.0
        if section.rows:
            return self._check_rows(section, grads)
        norm = math.sqrt(sum(float(g.float().pow(2).sum()) for g in grads))
        if not math.isfinite(norm):
            for g in grads:
                g.zero_()
            self.skipped[section.name] = self.skipped.get(section.name, 0) + 1
            return 0.0
        history = self._history[section.name]
        if len(history) >= self.policy.warmup:
            mean = sum(history) / len(history)
            std = math.sqrt(sum((h - mean) ** 2 for h in history) / len(history))
            ceiling = mean + self.policy.spike_z * max(std, 1e-12)
            if norm > ceiling > 0:
                for g in grads:
                    g.mul_(ceiling / norm)
                self.clipped[section.name] = self.clipped.get(section.name, 0) + 1
                norm = ceiling
        history.append(norm)
        del history[:-50]
        if self.policy.dead_patience:
            self._quiet[section.name] = self._quiet[section.name] + 1 if norm <= self.policy.dead_norm else 0
            if self._quiet[section.name] >= self.policy.dead_patience:
                self.frozen.add(section.name)
        return self.policy.priorities.get(section.name, 1.0)

    def _check_rows(self, section: Section, grads: list[Any]) -> float:
        """Row sections: each row (Pdial) is checked on its own, vectorised."""
        for g in grads:
            rows = g.reshape(g.shape[0], -1)
            bad = ~torch.isfinite(rows).all(dim=1)
            if bool(bad.any()):
                rows[bad] = 0.0
                key = f"{section.name}[rows]"
                self.skipped[key] = self.skipped.get(key, 0) + int(bad.sum())
            norms = rows.float().norm(dim=1)
            healthy = norms[norms > 0]
            if healthy.numel() > 4:
                median = float(healthy.median())
                ceiling = median * (1.0 + self.policy.spike_z)
                over = norms > ceiling
                if bool(over.any()):
                    rows[over] *= (ceiling / norms[over]).unsqueeze(1).to(rows.dtype)
                    key = f"{section.name}[rows]"
                    self.clipped[key] = self.clipped.get(key, 0) + int(over.sum())
        return self.policy.priorities.get(section.name, 1.0)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        scales = {}
        for section in self.sections:
            scale = self._check(section)
            for p in section.params:
                scales[p] = scale
        for p, mask in self.masks.items():
            if p.grad is not None:
                p.grad[mask] = 0.0
        # Keep a copy only of what the update must be scaled or undone on.
        before = {p: p.detach().clone() for p, s in scales.items()
                  if p.grad is not None and (s != 1.0 or p in self.masks)}
        before.update({p: p.detach().clone() for p in self.masks if p not in before})
        for p, _s in scales.items():
            if isinstance(self.inner, C4CookerV5S):
                self.inner.state[p]["c4_scale"] = 1.0
        self.inner.step()
        for p, old in before.items():
            scale = scales.get(p, 1.0)
            if scale != 1.0:
                p.copy_(old + (p - old) * scale)
            mask = self.masks.get(p)
            if mask is not None:
                p[mask] = old[mask]
        self.steps += 1
        return loss

    def report(self) -> dict[str, Any]:
        return {"steps": self.steps, "sections": len(self.sections), "frozen": sorted(self.frozen),
                "skipped": dict(self.skipped), "clipped": dict(self.clipped),
                "inner": type(self.inner).__name__}

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.inner.param_groups

    def state_dict(self) -> dict[str, Any]:
        return {"inner": self.inner.state_dict(), "frozen": sorted(self.frozen), "steps": self.steps}
