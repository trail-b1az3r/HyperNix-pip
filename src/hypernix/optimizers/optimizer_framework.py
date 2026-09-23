"""hypernix.optimizer_framework — Structured optimizer utilities.

Provides a composable, schedule-aware base class for PyTorch optimizers
plus profiling helpers and a pure-Python fused AdamW step.

Added in v0.70.4b2.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

import torch

__all__ = [
    "GradStats",
    "OptimizerBase",
    "OptimizerProfiler",
    "ScheduleConfig",
    "StepProfile",
    "fused_adamw_step",
]


# ---------------------------------------------------------------------------
# ScheduleConfig
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScheduleConfig:
    """LR schedule: linear warmup → plateau → cosine cooldown."""

    lr: float = 3e-4
    warmup_steps: int = 200
    plateau_steps: int = 1000
    cooldown_steps: int = 200
    min_lr: float = 1e-6

    # Computed boundaries (set in __post_init__)
    _warmup_end: int = dataclasses.field(init=False, repr=False)
    _plateau_end: int = dataclasses.field(init=False, repr=False)
    _total_steps: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._warmup_end = self.warmup_steps
        self._plateau_end = self.warmup_steps + self.plateau_steps
        self._total_steps = self.warmup_steps + self.plateau_steps + self.cooldown_steps

    def validate(self) -> ScheduleConfig:
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.plateau_steps < 0:
            raise ValueError("plateau_steps must be >= 0")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        return self

    def phase_at_step(self, step: int) -> str:
        if step < self._warmup_end:
            return "warmup"
        if step < self._plateau_end:
            return "plateau"
        if step < self._total_steps:
            return "cooldown"
        return "done"

    def lr_at_step(self, step: int) -> float:
        phase = self.phase_at_step(step)
        if phase == "warmup":
            denom = max(self.warmup_steps, 1)
            return self.lr * (step + 1) / denom
        if phase == "plateau":
            return self.lr
        if phase == "cooldown":
            elapsed = step - self._plateau_end
            total = max(self.cooldown_steps, 1)
            # Cosine decay from lr → min_lr
            progress = elapsed / total
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.lr - self.min_lr) * cosine_decay
        # done
        return self.min_lr


# ---------------------------------------------------------------------------
# GradStats
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GradStats:
    """Statistics returned by :meth:`OptimizerBase.gradient_clip`."""

    total_norm: float
    clipped: bool
    clip_threshold: float | None = None


# ---------------------------------------------------------------------------
# StepProfile / OptimizerProfiler
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepProfile:
    """Timing / throughput measurements for a single optimizer step."""

    step: int
    elapsed_ms: float
    tokens: int | None = None

    @property
    def tokens_per_sec(self) -> float | None:
        if self.tokens is None or self.elapsed_ms <= 0:
            return None
        return self.tokens / (self.elapsed_ms / 1000.0)


class OptimizerProfiler:
    """Rolling-window step profiler."""

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._history: deque[StepProfile] = deque(maxlen=window)
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def end(self, step: int, tokens: int | None = None) -> StepProfile:
        t1 = time.perf_counter()
        elapsed_ms = (t1 - (self._t0 or t1)) * 1000.0
        profile = StepProfile(step=step, elapsed_ms=elapsed_ms, tokens=tokens)
        self._history.append(profile)
        self._t0 = None
        return profile

    @property
    def mean_step_ms(self) -> float:
        if not self._history:
            return 0.0
        return sum(p.elapsed_ms for p in self._history) / len(self._history)

    @property
    def mean_tokens_per_sec(self) -> float | None:
        rates = [p.tokens_per_sec for p in self._history if p.tokens_per_sec is not None]
        if not rates:
            return None
        return sum(rates) / len(rates)


# ---------------------------------------------------------------------------
# OptimizerBase
# ---------------------------------------------------------------------------

class OptimizerBase(torch.optim.Optimizer):
    """Schedule-aware base class for HyperNix optimizers.

    Subclasses must implement :meth:`step`.  The base class handles:
    * LR scheduling via :class:`ScheduleConfig`
    * Gradient norm / value clipping via :meth:`gradient_clip`
    * Optional step profiling via :class:`OptimizerProfiler`
    """

    def __init__(
        self,
        params: Iterable,
        defaults: dict[str, Any],
        schedule: ScheduleConfig | None = None,
        grad_clip: float | None = None,
        grad_clip_mode: str = "norm",  # "norm" | "value"
        enable_profiling: bool = False,
    ) -> None:
        schedule = schedule or ScheduleConfig()
        # Every param group gets an `lr`, seeded with the configured rate.
        # PyTorch's schedulers read `group["lr"]` at construction and raise
        # KeyError without it — so every OptimizerBase optimizer failed
        # the moment `NeoOven.train` wrapped it in CosineAnnealingLR, which
        # it always does. `_apply_lr_schedule` still sets the per-step value.
        defaults = {"lr": schedule.lr, **defaults}
        super().__init__(params, defaults)
        self._schedule = schedule
        self._grad_clip = grad_clip
        self._grad_clip_mode = grad_clip_mode
        self._global_step: int = 0
        self._profiler: OptimizerProfiler | None = (
            OptimizerProfiler() if enable_profiling else None
        )

    # -- LR helpers ----------------------------------------------------------

    def scheduled_lr(self, step: int | None = None) -> float:
        s = self._global_step if step is None else step
        return self._schedule.lr_at_step(s)

    def phase(self, step: int | None = None) -> str:
        s = self._global_step if step is None else step
        return self._schedule.phase_at_step(s)

    def _apply_lr_schedule(self) -> None:
        lr = self._schedule.lr_at_step(self._global_step)
        for pg in self.param_groups:
            pg["lr"] = lr

    # -- Gradient clipping ---------------------------------------------------

    def gradient_clip(self) -> GradStats:
        params_with_grad = [
            p for pg in self.param_groups for p in pg["params"] if p.grad is not None
        ]
        if not params_with_grad:
            return GradStats(total_norm=0.0, clipped=False,
                             clip_threshold=self._grad_clip)

        if self._grad_clip_mode == "value":
            total_norm = max(
                p.grad.abs().max().item() for p in params_with_grad
            )
            clipped = False
            if self._grad_clip is not None:
                for p in params_with_grad:
                    p.grad.clamp_(-self._grad_clip, self._grad_clip)
                clipped = total_norm > self._grad_clip
        else:  # "norm"
            total_norm = torch.nn.utils.clip_grad_norm_(
                params_with_grad,
                self._grad_clip if self._grad_clip is not None else float("inf"),
            ).item()
            clipped = (
                self._grad_clip is not None and total_norm > self._grad_clip
            )

        return GradStats(
            total_norm=total_norm,
            clipped=clipped,
            clip_threshold=self._grad_clip,
        )

    # -- Profiling helpers ---------------------------------------------------

    def profile_start(self) -> None:
        if self._profiler is not None:
            self._profiler.start()

    def profile_end(self, tokens: int | None = None) -> StepProfile | None:
        if self._profiler is None:
            return None
        return self._profiler.end(step=self._global_step, tokens=tokens)

    # -- Introspection -------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._global_step

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "lr": self._schedule.lr,
            "warmup_steps": self._schedule.warmup_steps,
            "plateau_steps": self._schedule.plateau_steps,
            "cooldown_steps": self._schedule.cooldown_steps,
            "grad_clip": self._grad_clip,
            "global_step": self._global_step,
            "phase": self.phase(),
            "scheduled_lr": self.scheduled_lr(),
        }

    def __repr__(self) -> str:
        d = self.describe()
        parts = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"{type(self).__name__}({parts})"

    # -- Abstract step (subclasses must implement) ---------------------------

    def step(self, closure=None):  # type: ignore[override]
        raise NotImplementedError("Subclasses must implement step()")


# ---------------------------------------------------------------------------
# fused_adamw_step (pure Python / CPU-friendly)
# ---------------------------------------------------------------------------

def fused_adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int,
) -> None:
    """In-place AdamW parameter update (CPU-safe, no CUDA fusion required).

    Updates *params* in place using the standard decoupled weight-decay
    AdamW rule (Loshchilov & Hutter 2019).

    Args:
        params: List of parameter tensors.
        grads: Corresponding gradient tensors.
        exp_avgs: First-moment (m) accumulators.
        exp_avg_sqs: Second-moment (v) accumulators.
        lr: Learning rate.
        betas: (beta1, beta2) exponential decay rates.
        eps: Numerical stability term.
        weight_decay: Decoupled weight-decay coefficient.
        step: Current step (1-indexed) for bias correction.
    """
    beta1, beta2 = betas
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    for p, grad, exp_avg, exp_avg_sq in zip(params, grads, exp_avgs, exp_avg_sqs, strict=True):
        # Decoupled weight decay
        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)

        # Moment updates (in-place)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Bias-corrected moments
        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2

        # Parameter update: p -= lr * m_hat / (sqrt(v_hat) + eps)
        denom = v_hat.sqrt().add_(eps)
        p.addcdiv_(m_hat, denom, value=-lr)
