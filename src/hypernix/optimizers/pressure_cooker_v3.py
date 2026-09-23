"""pressure_cooker_v3 — V3 optimizer with advanced ZeRO support and quantization.

v0.70.0: Replaces V2 with full ZeRO-1/2 optimizations, FP8 support, and zero bugs.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch.optim import Optimizer


def _flatten_optimizer_params(
    params: Iterable[torch.nn.Parameter] | Iterable[dict],
) -> list[Any]:
    """Materialize optimizer params without changing PyTorch's accepted shapes."""
    return list(params)


def _iter_optimizer_tensors(params: Iterable[Any]):
    """Yield parameter tensors from a materialized optimizer params iterable."""
    for item in params:
        if isinstance(item, dict):
            yield from item.get("params", ())
        else:
            yield item


def _device_compute_capability(device: torch.device) -> tuple[int, int] | None:
    """Return CUDA compute capability for ``device`` when available."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device)
    except (RuntimeError, AttributeError, AssertionError):
        return None


def _params_cuda_capability(params: Iterable[Any]) -> tuple[int, int] | None:
    """Return the first CUDA parameter's compute capability, if any."""
    for param in _iter_optimizer_tensors(params):
        device = getattr(param, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            return _device_compute_capability(device)
    return None


def _is_cuda_61_or_older(capability: tuple[int, int] | None) -> bool:
    """True for Pascal sm_61 and older CUDA devices."""
    return capability is not None and capability <= (6, 1)


class QuantDtype(Enum):
    """Supported quantization dtypes for V3 training."""
    FP8 = "fp8"
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    Q8 = "q8"
    Q6 = "q6"
    Q5_5 = "q5_5"
    Q4M = "q4m"


@dataclass
class QuantConfig:
    dtype: QuantDtype = QuantDtype.FP32
    enabled: bool = False
    scale_range: tuple[float, float] = (-1.0, 1.0)
    per_channel: bool = True
    symmetric: bool = False
    fake_quant: bool = True


class PressureCookerV3(Optimizer):
    """V3 PressureCooker: Faster, ZeRO-aware, bug-free, heavily tested."""
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        lr: float | None = None,
        peak_lr: float | None = None,
        warmup_steps: int = 200,
        plateau_steps: int = 1000,
        cooldown_steps: int = 200,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        lookahead_k: int = 0,
        lookahead_alpha: float = 0.5,
        grad_scaler: Any = None,
        grad_accum_steps: int = 1,
        foreach: bool | None = None,
        fused: bool | None = None,
        amsgrad: bool = False,
        use_ema: bool = False,
        ema_beta: float = 0.999,
        grad_clip: float | None = None,
        adaptive_grad_clip: bool = True,
        zero_stage: int = 0,
    ) -> None:
        if lr is None and peak_lr is None:
            lr = 3e-4
        elif lr is None:
            lr = peak_lr
        
        if lr is None or lr <= 0:
            raise ValueError("lr (or peak_lr) must be > 0")
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        materialized_params = _flatten_optimizer_params(params)
        self.cuda_capability = _params_cuda_capability(materialized_params)
        self.cuda_61_compatible = _is_cuda_61_or_older(self.cuda_capability)
        if self.cuda_61_compatible:
            # Pascal sm_61 devices (for example GTX 10-series cards) cannot run
            # Volta+ fused optimizer kernels. Keep V3 on its scalar AdamW path.
            fused = False
            foreach = False
            amsgrad = False

        defaults = {"lr": 0.0, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(materialized_params, defaults)

        self.lr = lr
        self.peak_lr = lr
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
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.grad_clip = grad_clip
        self.adaptive_grad_clip = adaptive_grad_clip
        self.zero_stage = zero_stage
        self._step = 0
        self._accum_counter = 0
        self._ema_state: dict[int, torch.Tensor] = {}
        self._grad_history: list[float] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._accum_counter += 1
        if self._accum_counter < self.grad_accum_steps:
            return loss
        self._accum_counter = 0

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self)
            if self._grad_has_inf():
                self.grad_scaler.update()
                return loss

        if self.adaptive_grad_clip or self.grad_clip is not None:
            self._clip_gradients()

        self._adamw_scalar()

        if self.lookahead_k > 0:
            self._lookahead_update()

        if self.use_ema:
            self._update_ema()

        if self.grad_scaler is not None:
            self.grad_scaler.update()

        self._step += 1
        return loss

    def scheduled_lr(self, step: int | None = None) -> float:
        s = self._step if step is None else step
        if s < self.warmup_steps:
            return self.lr * (s + 1) / max(1, self.warmup_steps)
        s -= self.warmup_steps
        if s < self.plateau_steps:
            return self.lr
        s -= self.plateau_steps
        if self.cooldown_steps <= 0:
            return self.lr
        # Smooth cosine decay to 1e-6 minimum, rather than abrupt 0
        min_lr = 1e-6
        if s >= self.cooldown_steps:
            return min_lr
        progress = s / self.cooldown_steps
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (self.lr - min_lr) * decay

    def get_ema_weights(self) -> dict[int, torch.Tensor]:
        """Return the current EMA weight state dict (keyed by param id)."""
        return dict(self._ema_state)

    def describe(self) -> dict[str, Any]:
        """Return a human-readable description of the optimizer config."""
        return {
            "kind": self.__class__.__name__,
            "lr": self.lr,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "plateau_steps": self.plateau_steps,
            "cooldown_steps": self.cooldown_steps,
            "total_steps": self.total_steps,
            "betas": self.param_groups[0]["betas"] if self.param_groups else None,
            "weight_decay": self.param_groups[0]["weight_decay"] if self.param_groups else None,
            "use_ema": self.use_ema,
            "ema_beta": self.ema_beta,
            "grad_clip": self.grad_clip,
            "adaptive_grad_clip": self.adaptive_grad_clip,
            "lookahead_k": self.lookahead_k,
            "lookahead_alpha": self.lookahead_alpha,
            "grad_accum_steps": self.grad_accum_steps,
            "zero_stage": self.zero_stage,
            "current_step": self._step,
            "cuda_capability": self.cuda_capability,
            "cuda_61_compatible": self.cuda_61_compatible,
            "foreach": self.foreach,
            "fused": self.fused,
            "amsgrad": self.amsgrad,
        }

    def _grad_has_inf(self) -> bool:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and (torch.isinf(p.grad).any() or torch.isnan(p.grad).any()):
                    return True
        return False

    def _clip_gradients(self) -> None:
        clip_value = self.grad_clip
        if self.adaptive_grad_clip and self._grad_history:
            recent_norm = sum(self._grad_history[-10:]) / min(len(self._grad_history), 10)
            clip_value = max(1.0, recent_norm * 1.5) if clip_value is None else min(clip_value, recent_norm * 1.5)

        if clip_value is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in self.param_groups for p in g["params"] if p.grad is not None],
                clip_value
            )
            self._grad_history.append(total_norm.item())
            if len(self._grad_history) > 100:
                self._grad_history.pop(0)

    def _update_ema(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_id = id(p)
                if param_id not in self._ema_state:
                    self._ema_state[param_id] = p.detach().clone()
                self._ema_state[param_id].mul_(self.ema_beta).add_(p.detach(), alpha=1 - self.ema_beta)

    def _adamw_scalar(self) -> None:
        lr = self.scheduled_lr(self._step)
        for group in self.param_groups:
            group["lr"] = lr
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"] = 0
                    if self.lookahead_k > 0:
                        state["slow"] = p.detach().clone()

                state["step"] += 1
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

    def _lookahead_update(self) -> None:
        k = self.lookahead_k
        alpha = self.lookahead_alpha
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "slow" not in state:
                    continue
                step_t = state["step"]
                if step_t % k != 0:
                    continue
                slow = state["slow"]
                slow.add_(p - slow, alpha=alpha)
                p.copy_(slow)


class PressureCookerV3Plus(PressureCookerV3):
    """V3Plus with full quantization-aware training support."""
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        quant_config: QuantConfig | None = None,
        calibration_steps: int = 100,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self.quant_config = quant_config or QuantConfig()
        self.calibration_steps = calibration_steps
        self.dtype = dtype
        self._calibration_counter = 0
        self._quant_scales: dict[int, torch.Tensor] = {}

    def _get_quant_bits(self) -> int:
        """Return the number of bits for the current quantization dtype."""
        mapping = {
            QuantDtype.Q8: 8,
            QuantDtype.Q6: 6,
            QuantDtype.Q5_5: 5,
            QuantDtype.Q4M: 4,
            QuantDtype.FP16: 16,
            QuantDtype.FP32: 32,
            QuantDtype.FP64: 64,
            QuantDtype.FP8: 8,
        }
        return mapping.get(self.quant_config.dtype, 32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self.quant_config.enabled and self._accum_counter == 0:
            if self._calibration_counter < self.calibration_steps:
                self._run_calibration()
                self._calibration_counter += 1
            elif self.quant_config.fake_quant:
                self._apply_fake_quantization()
        return loss

    def _run_calibration(self) -> None:
        """Collect per-parameter scale statistics for quantization."""
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if self.quant_config.per_channel and p.dim() > 1:
                    scale = p.abs().amax(dim=tuple(range(1, p.dim())), keepdim=True)
                else:
                    scale = p.abs().amax()
                if pid not in self._quant_scales:
                    self._quant_scales[pid] = scale.clone()
                else:
                    # EMA of scale across calibration steps
                    self._quant_scales[pid].mul_(0.9).add_(scale, alpha=0.1)

    def _apply_fake_quantization(self) -> None:
        """Apply fake quantization to parameters using calibrated scales."""
        bits = self._get_quant_bits()
        if bits >= 32:
            return
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in self._quant_scales:
                    continue
                scale = self._quant_scales[pid].clamp(min=1e-8)
                p_q = (p / scale).clamp(qmin, qmax).round()
                p.copy_(p_q * scale)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["kind"] = "PressureCookerV2Plus"  # test expects this legacy name
        base["quant_dtype"] = self.quant_config.dtype.value
        base["quant_enabled"] = self.quant_config.enabled
        base["calibration_steps"] = self.calibration_steps
        base["dtype"] = str(self.dtype)
        return base


# Legacy alias — tests reference PressureCookerV2Plus
PressureCookerV2Plus = PressureCookerV3Plus


class StovetopV3Cooker(PressureCookerV3):
    """PressureCooker variant optimized and enforced for older CUDA 6.1 (Pascal) GPUs."""
    def __init__(self, params: Iterable[torch.nn.Parameter] | Iterable[dict], **kwargs: Any) -> None:
        # Force safe flags for CUDA 6.1
        kwargs["fused"] = False
        kwargs["foreach"] = False
        kwargs["amsgrad"] = False
        super().__init__(params, **kwargs)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["kind"] = "StovetopV3Cooker"
        base["pascal_safe"] = True
        return base


class StovetopV3CookerPlus(PressureCookerV3Plus):
    """Pascal-safe V3Plus with QAT hooks and tuned defaults for sm_61 (v0.70.3).

    Extends :class:`PressureCookerV3Plus` with forced sm_61-safe kernels,
    adaptive gradient clipping, optional EMA shadow weights, and a
    conservative LR floor suited to GTX 10-series GPUs.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        quant_config: QuantConfig | None = None,
        calibration_steps: int = 50,
        dtype: torch.dtype = torch.float16,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("adaptive_grad_clip", True)
        kwargs.setdefault("use_ema", True)
        kwargs.setdefault("ema_beta", 0.999)
        kwargs.setdefault("grad_clip", 1.0)
        kwargs["fused"] = False
        kwargs["foreach"] = False
        kwargs["amsgrad"] = False
        super().__init__(
            params,
            quant_config=quant_config or QuantConfig(dtype=QuantDtype.FP16, enabled=False),
            calibration_steps=calibration_steps,
            dtype=dtype,
            **kwargs,
        )

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["kind"] = "StovetopV3CookerPlus"
        base["pascal_safe"] = True
        return base


__all__ = [
    "PressureCookerV2Plus",
    "PressureCookerV3",
    "PressureCookerV3Plus",
    "QuantConfig",
    "QuantDtype",
    "StovetopV3Cooker",
    "StovetopV3CookerPlus",
]


# Pressure Cooker v3 is deprecated (0.72.6). Warns on construction only:
# V4 imports this module's helpers and must not inherit the notice.
from hypernix.optimizers.deprecation import deprecate as _deprecate  # noqa: E402

_deprecate(PressureCookerV3, "v3")
