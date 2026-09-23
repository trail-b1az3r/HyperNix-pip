"""pressure_cooker — custom AdamW optimizer with device-tuned tiers.

v0.70.0 rewrite.  The base :class:`PressureCooker` is a pure-Python
implementation of the AdamW schedule with gradient clipping and
optional lookahead.  On top of that v0.70.0 ships specialised
tiers, split by the target device.

DEVICE TIERS:
* :class:`StovetopCooker`      — CPU tier 1.  Minimum-memory path
* :class:`ElectricCooker`      — CPU tier 2.  Multi-tensor optimized
* :class:`InductionCooker`     — GPU tier 1.  Fused AdamW + AMP
* :class:`ProCooker`           — GPU tier 2.  CUDA graphs capture
* :class:`UniversalCooker`     — Auto-selects a tier for you (see ``TIERS``)

Fused AdamW / ``foreach`` / GradScaler support is auto-detected from the
installed torch version at import time (see ``_HAS_FUSED_ADAMW`` /
``_HAS_FOREACH`` / ``_HAS_GRAD_SCALER`` below) rather than hand-declared
per class.

Quantization-aware training, the Workshop model-building framework
(``WorkshopFramework``), and the TTS/ASR pipeline classes
(``TTSEngine``, ``ASREngine``, etc.) all live elsewhere in the package —
QAT in :mod:`hypernix.pressure_cooker_v3` / ``_v4`` / ``_v5``, Workshop
and the TTS/ASR classes in :mod:`hypernix.workshop`. They are not part
of this module; do not import them from here.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# Version probe — some optimizations are only safe on recent torch.
# ---------------------------------------------------------------------------

_TORCH_VERSION: tuple[int, int] = tuple(  # type: ignore[assignment]
    int(p) for p in torch.__version__.split("+")[0].split(".")[:2]
)

_HAS_FUSED_ADAMW = _TORCH_VERSION >= (2, 0)
_HAS_FOREACH = _TORCH_VERSION >= (1, 12)
_HAS_GRAD_SCALER = hasattr(torch.cuda.amp, "GradScaler")  # 1.6+


class PressureCooker(Optimizer):
    """AdamW + warmup / plateau / cooldown + optional lookahead + v0.48
    mixed-precision / grad-accumulation / foreach / fused extras.

    All the v0.47 kwargs still work unchanged — the new features are
    opt-in via additional kwargs.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        peak_lr: float = 3e-4,
        warmup_steps: int = 200,
        plateau_steps: int = 1000,
        cooldown_steps: int = 200,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        lookahead_k: int = 0,
        lookahead_alpha: float = 0.5,
        # v0.48 additions ------------------------------------------
        grad_scaler: Any = None,
        grad_accum_steps: int = 1,
        foreach: bool | None = None,
        fused: bool | None = None,
        amsgrad: bool = False,
    ) -> None:
        if peak_lr <= 0:
            raise ValueError("peak_lr must be > 0")
        if warmup_steps < 0 or plateau_steps < 0 or cooldown_steps < 0:
            raise ValueError("schedule step counts must be >= 0")
        if not 0.0 <= lookahead_alpha <= 1.0:
            raise ValueError("lookahead_alpha must be in [0, 1]")
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        defaults = {
            "lr": 0.0,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = warmup_steps + plateau_steps + cooldown_steps
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.grad_scaler = grad_scaler
        self.grad_accum_steps = grad_accum_steps
        self.foreach = foreach
        self.fused = fused
        self.amsgrad = amsgrad
        self._step = 0
        self._accum_counter = 0

    # ------------------------------------------------------------------
    # LR schedule (unchanged — backward compatible)
    # ------------------------------------------------------------------

    def scheduled_lr(self, step: int | None = None) -> float:
        s = self._step if step is None else step
        if s < self.warmup_steps:
            return self.peak_lr * (s + 1) / max(1, self.warmup_steps)
        s -= self.warmup_steps
        if s < self.plateau_steps:
            return self.peak_lr
        s -= self.plateau_steps
        if self.cooldown_steps <= 0:
            return self.peak_lr
        if s >= self.cooldown_steps:
            return 0.0
        progress = s / self.cooldown_steps
        return self.peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Gradient accumulation: only the N-th call actually updates.
        self._accum_counter += 1
        if self._accum_counter < self.grad_accum_steps:
            return loss
        self._accum_counter = 0

        # GradScaler unscaling.  If the scaler reports an inf, skip
        # the update cleanly rather than corrupting the state.
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self)
            if _grad_has_inf(self):
                self.grad_scaler.update()
                return loss

        # Prefer the fused / foreach torch.optim.AdamW kernel when
        # explicitly requested, otherwise fall back to the pure-Python
        # per-parameter loop (keeps exact v0.47 semantics).
        if self.fused or (self.foreach is True and _HAS_FOREACH):
            self._adamw_multitensor()
        else:
            self._adamw_scalar()

        # Lookahead seal.
        if self.lookahead_k > 0:
            self._lookahead_update()

        if self.grad_scaler is not None:
            self.grad_scaler.update()
        self._step += 1
        return loss

    # ------------------------------------------------------------------
    # Inner AdamW implementations
    # ------------------------------------------------------------------

    def _adamw_scalar(self) -> None:
        """Pure-Python per-parameter loop.  v0.47 semantics."""
        lr = self.scheduled_lr(self._step)
        for group in self.param_groups:
            group["lr"] = lr
            self._adamw_scalar_for(
                [p for p in group["params"] if p.grad is not None],
                group,
            )

    def _adamw_scalar_for(
        self, params: list[torch.nn.Parameter], group: dict,
    ) -> None:
        """Per-group scalar AdamW.  Used both by :meth:`_adamw_scalar`
        and by :meth:`_adamw_multitensor` as a fallback when the
        private ``torch.optim._functional.adamw`` API is unavailable
        or has changed signature."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] = 0
                if self.lookahead_k > 0:
                    state["slow"] = p.detach().clone()
            # Tolerate ``state["step"]`` being a tensor or int.
            current = state["step"]
            current_value = (
                current.item() if isinstance(current, torch.Tensor) else current
            )
            state["step"] = current_value + 1
            step_t = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if wd != 0:
                p.mul_(1.0 - lr * wd)
            exp_avg.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
            bias1 = 1.0 - beta1 ** step_t
            bias2 = 1.0 - beta2 ** step_t
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
            step_size = lr / bias1
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _adamw_multitensor(self) -> None:
        """Dispatch to ``torch.optim.AdamW`` in foreach / fused mode so
        we get the fast fused CUDA kernel on modern torch.  Builds a
        one-shot inner optimizer to piggyback its kernel on our
        parameter groups + state."""
        lr = self.scheduled_lr(self._step)
        # Sync LR / betas / eps / weight_decay into each group so the
        # inner AdamW sees the schedule.
        for group in self.param_groups:
            group["lr"] = lr
        # Collect and forward state so the inner optimizer reuses
        # our exp_avg / exp_avg_sq buffers.
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            exp_avgs, exp_avg_sqs, state_steps = [], [], []
            for p in params:
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    if self.lookahead_k > 0:
                        state["slow"] = p.detach().clone()
                # state["step"] may be an int from the scalar path;
                # the multitensor path wants a tensor.
                if not isinstance(state["step"], torch.Tensor):
                    state["step"] = torch.tensor(
                        float(state["step"]), dtype=torch.float32, device=p.device,
                    )
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

            fused_ok = bool(self.fused) and _HAS_FUSED_ADAMW and all(
                p.is_cuda for p in params
            )
            # Pass 1 (v0.50): the private ``torch.optim._functional``
            # signature has shifted between torch 2.0 / 2.2 / 2.4 / 2.7.
            # If we can't import it or call it cleanly, fall back to
            # the scalar path rather than raising — the user gets the
            # right answer either way, just slower.
            functional_adamw = None
            try:
                from torch.optim._functional import adamw as functional_adamw
            except ImportError:
                pass
            if functional_adamw is None:
                self._adamw_scalar_for(params, group)
                continue
            try:
                functional_adamw(
                    params,
                    [p.grad for p in params],
                    exp_avgs,
                    exp_avg_sqs,
                    [],                              # max_exp_avg_sqs (amsgrad)
                    state_steps,
                    amsgrad=self.amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                    foreach=self.foreach is not False,
                    capturable=False,
                    differentiable=False,
                    fused=fused_ok,
                    grad_scale=None,
                    found_inf=None,
                )
            except TypeError:
                # Older / newer torch dropped or added kwargs — degrade
                # to the scalar path, no exception leaks to the caller.
                self._adamw_scalar_for(params, group)

    def _lookahead_update(self) -> None:
        k = self.lookahead_k
        alpha = self.lookahead_alpha
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "slow" not in state:
                    continue
                step_t = state["step"]
                step_value = step_t.item() if isinstance(step_t, torch.Tensor) else step_t
                if int(step_value) % k != 0:
                    continue
                slow = state["slow"]
                slow.add_(p - slow, alpha=alpha)
                p.copy_(slow)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def phase(self, step: int | None = None) -> str:
        s = self._step if step is None else step
        if s < self.warmup_steps:
            return "warmup"
        if s < self.warmup_steps + self.plateau_steps:
            return "plateau"
        if s < self.total_steps:
            return "cooldown"
        return "done"

    def describe(self) -> dict:
        return {
            "kind": type(self).__name__,
            "peak_lr": self.peak_lr,
            "warmup": self.warmup_steps,
            "plateau": self.plateau_steps,
            "cooldown": self.cooldown_steps,
            "lookahead_k": self.lookahead_k,
            "grad_accum_steps": self.grad_accum_steps,
            "foreach": self.foreach,
            "fused": self.fused,
            "amsgrad": self.amsgrad,
            "has_grad_scaler": self.grad_scaler is not None,
            "torch_version": _TORCH_VERSION,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(peak_lr={self.peak_lr}, "
            f"warmup={self.warmup_steps}, plateau={self.plateau_steps}, "
            f"cooldown={self.cooldown_steps}, "
            f"lookahead={f'k={self.lookahead_k}, alpha={self.lookahead_alpha}' if self.lookahead_k else 'off'}, "
            f"accum={self.grad_accum_steps}, "
            f"foreach={self.foreach}, fused={self.fused})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grad_has_inf(opt: Optimizer) -> bool:
    for group in opt.param_groups:
        for p in group["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
    return False


def _first_param_device(params: Iterable[torch.nn.Parameter]) -> torch.device | None:
    for p in params:
        if isinstance(p, dict):
            for inner in p.get("params", []):
                return inner.device
        else:
            return p.device
    return None


# ---------------------------------------------------------------------------
# CPU tier 1 — StovetopCooker
# ---------------------------------------------------------------------------

class StovetopCooker(PressureCooker):
    """CPU tier 1.  Minimum-memory path.

    Forces ``foreach=False`` and ``fused=False`` — no multi-tensor
    kernels, no GPU-specific paths.  Skips GradScaler (CPU fp16 is
    unstable, so mixed precision doesn't help on CPU anyway).
    Lookahead is available but off by default.
    """

    def __init__(self, params, **kwargs: Any) -> None:
        kwargs.setdefault("foreach", False)
        kwargs.setdefault("fused", False)
        kwargs.setdefault("grad_scaler", None)
        super().__init__(params, **kwargs)


# ---------------------------------------------------------------------------
# CPU tier 2 — ElectricCooker
# ---------------------------------------------------------------------------

class ElectricCooker(PressureCooker):
    """CPU tier 2.  Fast multi-tensor path.

    Uses ``foreach=True`` on torch ≥ 1.12 so parameter updates run
    as vectorised multi-tensor ops.  On torch < 1.12 falls back to
    Stovetop semantics automatically.  Good default for
    multi-core desktop CPUs with enough RAM.
    """

    def __init__(self, params, **kwargs: Any) -> None:
        kwargs.setdefault("foreach", _HAS_FOREACH)
        kwargs.setdefault("fused", False)
        kwargs.setdefault("grad_scaler", None)
        super().__init__(params, **kwargs)


# ---------------------------------------------------------------------------
# GPU tier 1 — InductionCooker
# ---------------------------------------------------------------------------

class InductionCooker(PressureCooker):
    """GPU tier 1.  ``foreach=True`` + ``fused=True`` AdamW on torch
    ≥ 2.0.  First-class ``torch.cuda.amp.GradScaler`` integration for
    fp16 runs — pass ``grad_scaler=torch.cuda.amp.GradScaler()`` and
    call ``scaler.scale(loss).backward()`` normally; the InductionCooker
    handles the unscale + inf-skip.
    """

    def __init__(self, params, **kwargs: Any) -> None:
        kwargs.setdefault("foreach", True)
        kwargs.setdefault("fused", _HAS_FUSED_ADAMW)
        super().__init__(params, **kwargs)


# ---------------------------------------------------------------------------
# GPU tier 2 — ProCooker (CUDA-graph capable)
# ---------------------------------------------------------------------------

class ProCooker(InductionCooker):
    """GPU tier 2.  InductionCooker plus optional CUDA graph capture.

    Call :meth:`warmup_graph(step_fn)` once with a representative
    training step to record a CUDA graph; subsequent ``step()`` calls
    replay it for a material speedup on small models / repetitive
    shapes.  Skip graph capture when batch shapes vary or the model
    has dynamic control flow.
    """

    _graph: Any = None
    _graph_step: Any = None

    def warmup_graph(self, step_fn) -> None:
        """Record a CUDA graph from a representative step.  Must be
        called on a CUDA device; raises ``RuntimeError`` otherwise."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device")
        # Warm up the allocator so graph capture doesn't race with
        # first-time CuDNN initialisation.
        for _ in range(3):
            step_fn()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._graph_step = step_fn()

    def replay_graph(self):
        """Replay the captured graph.  Returns whatever the original
        ``step_fn`` returned (typically the loss tensor, still live on
        device)."""
        if self._graph is None:
            raise RuntimeError("call warmup_graph(step_fn) first")
        self._graph.replay()
        return self._graph_step


# ---------------------------------------------------------------------------
# Universal selector
# ---------------------------------------------------------------------------

def _flatten_params(params) -> list[torch.nn.Parameter]:
    out = []
    for p in params:
        if isinstance(p, dict):
            out.extend(p.get("params", []))
        else:
            out.append(p)
    return out


def _is_pre_volta(device: torch.device) -> bool:
    """True for CUDA devices with compute capability < 7.0 — i.e.
    Pascal (sm_61, GTX 1080 / 1080 Ti / Titan Xp) and earlier.

    Patch (0.51.1): used by :class:`UniversalCooker.select` so a
    1080 user doesn't get the fused/foreach/CUDA-graph stack that
    silently requires sm_70+ and would crash with
    ``RuntimeError: fused=True requires CUDA capability >= 7.0``.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001
        return False
    return major < 7


class UniversalCooker:
    """Factory that returns the right tier for the model's device.

    Not an :class:`Optimizer` subclass — call
    ``UniversalCooker.select(params, **kwargs)`` (or
    :func:`universal_cooker`) to get the concrete instance.

    Patch (0.71.1): the default selection now prioritizes the V5 /
    V5S optimizer family (see :mod:`hypernix.pressure_cooker_v5` and
    :mod:`hypernix.pressure_cooker_v5s`) over the older device tiers
    below, since they're the actively-developed, fully custom
    optimizers and aren't restricted to CUDA the way ``ProCooker`` /
    ``InductionCooker`` are. Pass ``variant=`` to choose which one:

    * ``"v5s"``     (default) — :class:`PressureCookerV5S`, the newest
      and most specialised tier (3D oscillation resistance, pressure
      diffusion, low-power mode).
    * ``"v5"``      — :class:`PressureCookerV5` (ORCP core).
    * ``"v5-plus"`` — :class:`PressureCookerV5Plus` (ORCP-Ultra core;
      adds entropy scaling, resonance detection, QAT/MTP defaults).
    * ``"legacy"``  — the pre-0.71.1 behavior: ``ProCooker`` /
      ``InductionCooker`` on CUDA, ``ElectricCooker`` / ``StovetopCooker``
      on CPU, gated by ``prefer_speed`` exactly as before.

    Whichever V5-family variant is chosen, a CUDA device detected as
    pre-Volta (Pascal, sm_61/6.2 — e.g. a GTX 1080) automatically gets
    the matching ``Aged*`` tier instead (:class:`Agedcookerv5`,
    :class:`ULTRAagedcookerv5`, or :class:`Agedcookerv5s`), which
    enforces ``fused=False`` and other Pascal-safe defaults — the same
    auto-detection :func:`_is_pre_volta` already did for the legacy
    tiers, just extended to the new family.
    """

    _V5_VARIANTS = ("v5", "v5-plus", "v5s")

    @classmethod
    def select(
        cls, params, *, prefer_speed: bool = True, variant: str = "v5s", **kwargs: Any,
    ) -> Optimizer:
        listed = _flatten_params(list(params))
        dev = listed[0].device if listed else torch.device("cpu")

        if variant == "legacy":
            return cls._select_legacy(listed, dev, prefer_speed=prefer_speed, **kwargs)
        if variant not in cls._V5_VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of "
                f"{(*cls._V5_VARIANTS, 'legacy')}",
            )
        return cls._select_v5_family(listed, dev, variant=variant, **kwargs)

    @staticmethod
    def _select_legacy(listed, dev, *, prefer_speed: bool, **kwargs: Any) -> Optimizer:
        """Pre-0.71.1 selection, kept for anyone who explicitly opts out
        of the V5 / V5S family via ``variant='legacy'``."""
        if dev.type == "cuda":
            # Patch (0.51.1): Pascal (sm_61, e.g. GTX 1080) does not
            # support fused AdamW or CUDA graphs.  Force the safer
            # foreach-only InductionCooker variant with fused=False
            # so the optimizer actually runs on a 1080.  ProCooker is
            # gated on Volta+ (sm_70+, V100 / RTX 20-series and up).
            if _is_pre_volta(dev):
                kwargs.setdefault("fused", False)
                kwargs.setdefault("foreach", _HAS_FOREACH)
                return InductionCooker(listed, **kwargs)
            return ProCooker(listed, **kwargs) if prefer_speed else InductionCooker(listed, **kwargs)
        return ElectricCooker(listed, **kwargs) if prefer_speed else StovetopCooker(listed, **kwargs)

    @staticmethod
    def _select_v5_family(listed, dev, *, variant: str, **kwargs: Any) -> Optimizer:
        from hypernix.optimizers.pressure_cooker_v5 import (
            Agedcookerv5,
            PressureCookerV5,
            PressureCookerV5Plus,
            ULTRAagedcookerv5,
        )
        from hypernix.optimizers.pressure_cooker_v5s import Agedcookerv5s, PressureCookerV5S

        is_pascal = _is_pre_volta(dev)
        if variant == "v5":
            return Agedcookerv5(listed, **kwargs) if is_pascal else PressureCookerV5(listed, **kwargs)
        if variant == "v5-plus":
            return ULTRAagedcookerv5(listed, **kwargs) if is_pascal else PressureCookerV5Plus(listed, **kwargs)
        return Agedcookerv5s(listed, **kwargs) if is_pascal else PressureCookerV5S(listed, **kwargs)


# ---------------------------------------------------------------------------
# Registry + factory helpers
# ---------------------------------------------------------------------------

TIERS: dict[str, type[PressureCooker]] = {
    "pressure-cooker": PressureCooker,
    "stovetop": StovetopCooker,
    "electric": ElectricCooker,
    "induction": InductionCooker,
    "pro": ProCooker,
}


def pressure_cooker(
    params: Iterable[torch.nn.Parameter] | Iterable[dict],
    **kwargs: Any,
) -> PressureCooker:
    """Construct a :class:`PressureCooker` from keyword arguments.

    Backward-compatible with the v0.47 signature.  Pass ``tier=`` to
    pick one of the new variants by short name.
    """
    tier = kwargs.pop("tier", None)
    if tier is not None:
        key = tier.lower().replace("_", "-")
        if key not in TIERS:
            raise ValueError(
                f"unknown pressure cooker tier {tier!r}; valid: {sorted(TIERS)}",
            )
        return TIERS[key](params, **kwargs)
    return PressureCooker(params, **kwargs)


def stovetop_cooker(params, **kw: Any) -> StovetopCooker:
    return StovetopCooker(params, **kw)


def electric_cooker(params, **kw: Any) -> ElectricCooker:
    return ElectricCooker(params, **kw)


def induction_cooker(params, **kw: Any) -> InductionCooker:
    return InductionCooker(params, **kw)


def pro_cooker(params, **kw: Any) -> ProCooker:
    return ProCooker(params, **kw)


def universal_cooker(
    params, *, prefer_speed: bool = True, variant: str = "v5s", **kw: Any,
) -> Optimizer:
    """Return the best-fit cooker for the detected parameter device.

    Patch (0.71.1): defaults to the V5S optimizer tier (see
    :class:`UniversalCooker` for the full ``variant=`` options,
    including ``"legacy"`` to restore the pre-0.71.1 device-tier
    selection).
    """
    return UniversalCooker.select(params, prefer_speed=prefer_speed, variant=variant, **kw)


# Pressure Cooker v1 is deprecated (0.72.6). Constructing any of the
# classes above warns; importing this module does not.
from hypernix.optimizers.deprecation import deprecate as _deprecate  # noqa: E402

_deprecate(PressureCooker, "v1")
# UniversalCooker is deliberately not wrapped. It is a router: its default
# `variant="v5s"` returns the current V5 family, and only
# `variant="legacy"` reaches the V1 tiers — which are PressureCooker
# subclasses and warn on their own. Wrapping the router would warn people
# being sent to V5.
