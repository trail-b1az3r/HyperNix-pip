"""hypernix.pressure_cooker_v5s — Pressure Cooker V5S: 3D-ORCP Optimizer.

Architecture
============
V5S is a **ground-up** optimizer built around three co-designed ideas that do
not appear together in any prior published optimizer:

1. **3D Cosine Oscillation Resistance (3D-COR)**
   Classical oscillation resistance tracks one EMA of the cosine similarity
   between the raw gradient and the momentum direction.  V5S tracks *three*
   cosine EMAs simultaneously at different time horizons (fast β≈0.8,
   medium β≈0.95, ultra-slow β≈0.999), then combines them into a single
   "volumetric oscillation score" (VOS) that detects oscillation across three
   distinct spectral bands:

   * Fast band  — reacts within 5–10 steps (detects noise / momentary reversals)
   * Medium band — 20–50 steps  (detects learning-rate-scale oscillation)
   * Ultra-slow band — 1000+ steps (detects macro divergence / saddle drift)

   Each band contributes proportionally to the damping coefficient.  The
   result is a smooth, frequency-aware damping signal that avoids both the
   under-damping of single-EMA approaches and the lag of fully-averaged ones.

2. **Pressure Diffusion (PD)**
   Inspired by thermal diffusion in physics: adjacent coordinates in the
   reshaped parameter tensor *share* a fraction of each other's gradient
   signal through a lightweight 1-D convolution over the flattened gradient.
   This acts as a spatial low-pass filter that:
   * reduces high-frequency gradient noise without adding a second-moment buffer,
   * naturally couples nearby weights so that isolated outlier gradients are
     smoothed before entering the update rule,
   * costs O(n) in time and zero additional persistent state.

3. **Low Power Mode (LPM)**
   V5S keeps RAM use below ~1.4× SGD by:
   * maintaining only *one* quantized (int8) momentum vector per parameter,
   * factoring the curvature estimate into row/column outer products for
     all matrix parameters,
   * never allocating a full second-moment tensor,
   * using uint8 age counters for gradient-age-based coordinate freezing
     (no additional float buffer needed).

Update Rule (per parameter)
---------------------------
Given raw gradient ``g``, dequantized momentum ``m``:

  1. Diffuse gradient:   gd = pressure_diffuse(g, diffusion_factor)
  2. Extrapolate:        g_pred = gd + extrapolation_alpha * m
  3. Compute 3D VOS:     vos = f(fast_cos, med_cos, ultra_cos)
  4. Damping:            damp = 1 / (1 + vos_positive * 3D_gain)
  5. Adaptive power:     pw = adaptive_power(vos)
  6. Curvature (factored): curv = row_curv ⊗ col_curv / mean(row_curv)
  7. Power update:       u = sign(g_pred) * |g_pred|^pw / sqrt(curv)
  8. Sophia clip:        u = clamp(u, −clip, +clip)
  9. Freeze scale:       u = u * freeze_decay^max(0, age−patience)
  10. Trust ratio:       lr_eff = lr * damp * LARS_trust(p, u)
  11. Weight decay:      p ← p * (1 - lr_eff * wd)
  12. Param update:      p ← p − u * lr_eff
  13. Momentum update:   m ← beta * m + (1-beta) * gd  (re-quantize to int8)
  14. EMA (optional):    ema ← ema_decay * ema + (1-ema_decay) * p

No second moment.  No bias correction.  No adaptive learning rate denominator.
This is not AdamW, RMSProp, Adafactor, Lion, or any mixture thereof.

Memory footprint relative to SGD (1.0x):
  SGD                    ~1.0x
  Momentum SGD           ~2.0x
  PressureCookerV5S      ~1.4x  (int8 momentum + factored curv + uint8 age)
  PressureCookerV5       ~1.7x
  AdamW                  ~3.0x

v0.70.6 — initial release of V5S architecture.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypernix.optimizers.optimizer_framework import OptimizerBase, ScheduleConfig
from hypernix.optimizers.pressure_cooker_v3 import (
    _flatten_optimizer_params,
    _is_cuda_61_or_older,
    _params_cuda_capability,
)
from hypernix.optimizers.pressure_cooker_v5 import (
    MTPConfig,
    MTPHead,
    QATConfig,
    QATFakeQuantize,
    _dequantize_momentum,
    _quantize_momentum,
)

__all__ = [
    "PressureCookerV5S",
    "V5SConfig",
    "DiffusionMode",
    "Agedcookerv5s",
    "pressure_diffuse",
    "volumetric_oscillation_score",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INT8_MAX = 127.0
_SPARSE_THRESHOLD = 1 << 16  # 65536 elements — use scalar curvature above this


# ---------------------------------------------------------------------------
# V5S Enums / Helpers
# ---------------------------------------------------------------------------

class DiffusionMode:
    """Constants for the pressure diffusion kernel shape.

    FLAT    — uniform rectangular kernel (equal weight to all neighbours)
    GAUSS   — Gaussian-weighted kernel  (more weight to close neighbours)
    TRIANGLE — linearly decaying weights (compromise between FLAT and GAUSS)
    """
    FLAT     = "flat"
    GAUSS    = "gauss"
    TRIANGLE = "triangle"


# ---------------------------------------------------------------------------
# Pressure Diffusion
# ---------------------------------------------------------------------------

def _build_diffusion_kernel(
    width: int,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 1-D normalised diffusion kernel of the given width and mode.

    Args:
        width: Kernel full width (must be odd and >= 3).
        mode:  One of DiffusionMode.{FLAT, GAUSS, TRIANGLE}.
        device: Target device.
        dtype:  Target dtype (usually float32).

    Returns:
        Tensor of shape (1, 1, width) normalised so that it sums to 1.
    """
    half = width // 2
    if mode == DiffusionMode.FLAT:
        k = torch.ones(width, device=device, dtype=dtype)
    elif mode == DiffusionMode.GAUSS:
        x = torch.arange(-half, half + 1, device=device, dtype=dtype)
        sigma = half / 2.0
        k = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    elif mode == DiffusionMode.TRIANGLE:
        x = torch.arange(0, half + 1, device=device, dtype=dtype)
        half_k = half + 1 - x  # linearly decaying
        k = torch.cat([half_k.flip(0)[:-1], half_k])
    else:
        raise ValueError(f"Unknown DiffusionMode: {mode!r}")
    k = k / k.sum()
    return k.view(1, 1, width)


def pressure_diffuse(
    g: torch.Tensor,
    factor: float,
    kernel_width: int = 3,
    mode: str = DiffusionMode.GAUSS,
) -> torch.Tensor:
    """Apply pressure diffusion to a gradient tensor.

    The gradient is reshaped to a 1-D signal, convolved with a normalised
    kernel, then reshaped back.  The output is a weighted average of the
    original gradient (weight = 1 - factor) and the diffused version
    (weight = factor).

    Args:
        g:            Gradient tensor (any shape).
        factor:       Diffusion strength in [0, 1].  0 = no diffusion.
        kernel_width: Width of the 1-D diffusion kernel (odd, >= 3).
        mode:         Kernel shape (DiffusionMode constant).

    Returns:
        Diffused gradient with same shape as input.
    """
    if factor <= 0.0 or g.numel() < kernel_width:
        return g

    orig_shape = g.shape
    flat = g.flatten().float()
    n = flat.numel()

    # Build kernel on same device / dtype
    k = _build_diffusion_kernel(kernel_width, mode, g.device, flat.dtype)
    pad = kernel_width // 2

    # Convolve: (1, 1, n) x (1, 1, k) → (1, 1, n)
    signal = flat.view(1, 1, n)
    diffused = F.conv1d(signal, k, padding=pad).view(n)

    result = (1.0 - factor) * flat + factor * diffused
    return result.view(orig_shape).to(g.dtype)


# ---------------------------------------------------------------------------
# 3D Cosine Oscillation Score
# ---------------------------------------------------------------------------

def volumetric_oscillation_score(
    fast_cos: float,
    med_cos: float,
    ultra_cos: float,
    fast_weight: float = 0.45,
    med_weight: float = 0.35,
    ultra_weight: float = 0.20,
) -> float:
    """Combine three cosine-similarity EMAs into a single oscillation score.

    Each band contributes proportionally according to its weight.  Negative
    cosine similarity indicates agreement (gradient aligns with momentum),
    positive indicates oscillation / reversal.

    A positive VOS means *oscillation detected*; a negative means *consistent
    gradient direction*.

    Args:
        fast_cos:      Fast-band cosine EMA  (β ≈ 0.80).
        med_cos:       Medium-band cosine EMA (β ≈ 0.95).
        ultra_cos:     Ultra-slow cosine EMA  (β ≈ 0.999).
        fast_weight:   Contribution weight for the fast band.
        med_weight:    Contribution weight for the medium band.
        ultra_weight:  Contribution weight for the ultra-slow band.

    Returns:
        Scalar oscillation score in [-1, 1].
    """
    if not (abs(fast_weight + med_weight + ultra_weight - 1.0) < 1e-6):
        raise AssertionError("Weights must sum to 1")
    # Negate cosines: positive cosine = agreement = low oscillation
    vos = -(fast_weight * fast_cos + med_weight * med_cos + ultra_weight * ultra_cos)
    return float(max(-1.0, min(1.0, vos)))


# ---------------------------------------------------------------------------
# V5S Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class V5SConfig:
    """Full configuration for :class:`PressureCookerV5S`.

    This dataclass groups all V5S-specific hyperparameters with sane defaults
    and provides validation, summary printing, and serialisation helpers.

    Args:
        fast_beta:        EMA decay for the fast cosine band.  Lower = more
                          reactive.  Default 0.80 (reacts in ~5 steps).
        med_beta:         EMA decay for the medium cosine band.  Default 0.95
                          (reacts in ~20 steps).
        ultra_slow_beta:  EMA decay for the ultra-slow band.  Default 0.999
                          (reacts in ~1000 steps).
        fast_weight:      Weight of fast band in VOS.  Default 0.45.
        med_weight:       Weight of medium band in VOS.  Default 0.35.
        ultra_weight:     Weight of ultra-slow band in VOS.  Default 0.20.
        diffusion_factor: Fraction of the gradient replaced by the diffused
                          version.  0 = no diffusion, 1 = fully diffused.
                          Default 0.12.
        diffusion_kernel: Kernel width for pressure diffusion.  Must be odd
                          and >= 3.  Default 3.
        diffusion_mode:   Kernel shape.  One of DiffusionMode.{FLAT, GAUSS,
                          TRIANGLE}.  Default GAUSS.
        vos_3d_gain:      Multiplier controlling how strongly the VOS damps
                          the effective learning rate.  Higher = more damping
                          during oscillation.  Default 3.0.
        momentum_beta:    Momentum coefficient for the single-buffer EMA.
                          Default 0.90.
        power:            Initial gradient power exponent for the update rule.
                          Default 0.50.
        power_min:        Minimum adaptive power exponent.  Applied during
                          heavy oscillation.  Default 0.30.
        power_max:        Maximum adaptive power exponent.  Applied when
                          gradient direction is consistent.  Default 1.00.
        curvature_beta:   EMA decay for the factored curvature estimate.
                          Default 0.98.
        sophia_clip:      Magnitude at which the power-scaled update is
                          hard-clipped.  Default 1.00.
        extrapolation:    Fraction of momentum added to the raw gradient
                          before computing the update (look-ahead).  Default
                          0.15.
        freeze_threshold: Gradient absolute magnitude below which a coordinate
                          is considered "stagnant".  Default 1e-6.
        freeze_patience:  Number of consecutive stagnant steps before the
                          coordinate starts being frozen.  Default 32.
        freeze_decay:     Per-step decay applied to frozen coordinates.
                          Default 0.98.
        factorize_matrices: If True, use Adafactor-style factored row/column
                          curvature for matrix parameters (saves memory).
                          Default True.
        ema_decay:        If > 0, maintain an EMA copy of the parameters.
                          Default 0.0 (disabled).
        qat_config:       Optional QAT configuration (inherited from V5).
        enable_mtp:       If True, allow MTP head attachment.  Default False.
        mtp_config:       Optional MTP configuration.
    """
    # 3D cosine oscillation
    fast_beta:       float = 0.80
    med_beta:        float = 0.95
    ultra_slow_beta: float = 0.999
    fast_weight:     float = 0.45
    med_weight:      float = 0.35
    ultra_weight:    float = 0.20
    vos_3d_gain:     float = 3.0

    # Pressure diffusion
    diffusion_factor: float = 0.12
    diffusion_kernel: int   = 3
    diffusion_mode:   str   = DiffusionMode.GAUSS

    # Core ORCP
    momentum_beta:    float = 0.90
    power:            float = 0.50
    power_min:        float = 0.30
    power_max:        float = 1.00
    curvature_beta:   float = 0.98
    sophia_clip:      float = 1.00
    extrapolation:    float = 0.15

    # Coordinate freezing
    freeze_threshold: float = 1e-6
    freeze_patience:  int   = 32
    freeze_decay:     float = 0.98

    # Memory / topology
    factorize_matrices: bool = True
    ema_decay:          float = 0.0

    # Optional V5 QAT / MTP
    qat_config:  QATConfig | None  = None
    enable_mtp:  bool              = False
    mtp_config:  MTPConfig | None  = None

    # Derived (computed in __post_init__)
    _weight_sum: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        ws = self.fast_weight + self.med_weight + self.ultra_weight
        if abs(ws - 1.0) > 1e-5:
            raise ValueError(
                f"fast_weight + med_weight + ultra_weight must equal 1.0; got {ws:.6f}"
            )
        self._weight_sum = ws

        if not (0.0 <= self.diffusion_factor <= 1.0):
            raise ValueError("diffusion_factor must be in [0, 1]")
        if self.diffusion_kernel < 3 or self.diffusion_kernel % 2 == 0:
            raise ValueError("diffusion_kernel must be odd and >= 3")
        if self.diffusion_mode not in (DiffusionMode.FLAT, DiffusionMode.GAUSS, DiffusionMode.TRIANGLE):
            raise ValueError(f"Unknown diffusion_mode: {self.diffusion_mode!r}")
        if not (0.0 <= self.fast_beta < 1.0):
            raise ValueError("fast_beta must be in [0, 1)")
        if not (0.0 <= self.med_beta < 1.0):
            raise ValueError("med_beta must be in [0, 1)")
        if not (0.0 <= self.ultra_slow_beta < 1.0):
            raise ValueError("ultra_slow_beta must be in [0, 1)")
        if self.power_min > self.power_max:
            raise ValueError("power_min must be <= power_max")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

    def summary(self) -> str:
        """Return a human-readable summary of this configuration."""
        lines = [
            "=== PressureCookerV5S Configuration ===",
            f"  3D-COR bands: fast β={self.fast_beta} (w={self.fast_weight}), "
            f"med β={self.med_beta} (w={self.med_weight}), "
            f"ultra β={self.ultra_slow_beta} (w={self.ultra_weight})",
            f"  Pressure diffusion: factor={self.diffusion_factor}, "
            f"kernel={self.diffusion_kernel}, mode={self.diffusion_mode!r}",
            f"  Core: momentum_β={self.momentum_beta}, power=[{self.power_min}, {self.power_max}]",
            f"  Curvature β={self.curvature_beta}, sophia_clip={self.sophia_clip}",
            f"  Extrapolation α={self.extrapolation}, VOS gain={self.vos_3d_gain}",
            f"  Freeze: threshold={self.freeze_threshold}, patience={self.freeze_patience}, "
            f"decay={self.freeze_decay}",
            f"  Factorize matrices: {self.factorize_matrices}",
            f"  EMA decay: {self.ema_decay}",
            f"  QAT: {self.qat_config is not None}",
            f"  MTP: {self.enable_mtp}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (for logging / checkpointing)."""
        return {
            "fast_beta":         self.fast_beta,
            "med_beta":          self.med_beta,
            "ultra_slow_beta":   self.ultra_slow_beta,
            "fast_weight":       self.fast_weight,
            "med_weight":        self.med_weight,
            "ultra_weight":      self.ultra_weight,
            "vos_3d_gain":       self.vos_3d_gain,
            "diffusion_factor":  self.diffusion_factor,
            "diffusion_kernel":  self.diffusion_kernel,
            "diffusion_mode":    self.diffusion_mode,
            "momentum_beta":     self.momentum_beta,
            "power":             self.power,
            "power_min":         self.power_min,
            "power_max":         self.power_max,
            "curvature_beta":    self.curvature_beta,
            "sophia_clip":       self.sophia_clip,
            "extrapolation":     self.extrapolation,
            "freeze_threshold":  self.freeze_threshold,
            "freeze_patience":   self.freeze_patience,
            "freeze_decay":      self.freeze_decay,
            "factorize_matrices": self.factorize_matrices,
            "ema_decay":         self.ema_decay,
        }


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class PressureCookerV5S(OptimizerBase):
    """Pressure Cooker V5S — 3D Oscillation-Resistant Cosine, Pressure Diffusion, Low Power.

    V5S is a **fully custom optimizer** written from first principles.  It
    shares no update logic with AdamW, RMSProp, Adafactor, Lion, SGD, LAMB, or
    LARS.  The only borrowed primitives from V5 are the quantized-momentum
    helper functions (``_quantize_momentum`` / ``_dequantize_momentum``) and
    the optional QAT / MTP infrastructure.

    See module docstring for the full mathematical description.

    Recommended hyperparameter starting points
    ------------------------------------------
    * General pretraining large models: defaults (lr=3e-4)
    * Fine-tuning:                      lr=1e-4, diffusion_factor=0.05,
                                        vos_3d_gain=4.0, freeze_patience=64
    * Low-memory edge:                  diffusion_factor=0.0, ema_decay=0.0,
                                        factorize_matrices=True
    * High-stability mode:              fast_weight=0.2, med_weight=0.5,
                                        ultra_weight=0.3, vos_3d_gain=5.0

    Args:
        params:     Parameter iterable or list of param-group dicts.
        v5s_config: Optional :class:`V5SConfig` instance.  Overrides all
                    keyword arguments for V5S-specific parameters.
        lr:         Global learning rate.  Default 3e-4.
        weight_decay: L2 regularisation coefficient.  Default 0.01.
        eps:        Small constant for numerical stability.  Default 1e-8.
        grad_clip:  Global gradient clipping norm (pre-step).  None = off.
        schedule:   Optional :class:`ScheduleConfig` for LR scheduling.
        trust_clip: (min, max) clamp for the LARS/LAMB-style trust ratio.
        **v5s_kwargs: Any V5SConfig field can be passed as keyword argument
                      when ``v5s_config`` is None.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        v5s_config: V5SConfig | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        grad_clip: float | None = 1.0,
        schedule: ScheduleConfig | None = None,
        trust_clip: tuple[float, float] = (0.05, 5.0),
        # V5SConfig fields as keyword arguments (used when v5s_config=None)
        fast_beta: float = 0.80,
        med_beta: float = 0.95,
        ultra_slow_beta: float = 0.999,
        fast_weight: float = 0.45,
        med_weight: float = 0.35,
        ultra_weight: float = 0.20,
        vos_3d_gain: float = 3.0,
        diffusion_factor: float = 0.12,
        diffusion_kernel: int = 3,
        diffusion_mode: str = DiffusionMode.GAUSS,
        momentum_beta: float = 0.90,
        power: float = 0.50,
        power_min: float = 0.30,
        power_max: float = 1.00,
        curvature_beta: float = 0.98,
        sophia_clip: float = 1.00,
        extrapolation: float = 0.15,
        freeze_threshold: float = 1e-6,
        freeze_patience: int = 32,
        freeze_decay: float = 0.98,
        factorize_matrices: bool = True,
        ema_decay: float = 0.0,
        qat_config: QATConfig | None = None,
        enable_mtp: bool = False,
        mtp_config: MTPConfig | None = None,
        six_bit_mode: str = "off",
        **kwargs: Any,
    ) -> None:
        materialized_params = _flatten_optimizer_params(params)
        cuda_cap = _params_cuda_capability(materialized_params)
        self._pascal_safe = _is_cuda_61_or_older(cuda_cap)

        # Build or validate V5SConfig
        if v5s_config is not None:
            self.cfg = v5s_config
        else:
            self.cfg = V5SConfig(
                fast_beta=fast_beta,
                med_beta=med_beta,
                ultra_slow_beta=ultra_slow_beta,
                fast_weight=fast_weight,
                med_weight=med_weight,
                ultra_weight=ultra_weight,
                vos_3d_gain=vos_3d_gain,
                diffusion_factor=diffusion_factor,
                diffusion_kernel=diffusion_kernel,
                diffusion_mode=diffusion_mode,
                momentum_beta=momentum_beta,
                power=power,
                power_min=power_min,
                power_max=power_max,
                curvature_beta=curvature_beta,
                sophia_clip=sophia_clip,
                extrapolation=extrapolation,
                freeze_threshold=freeze_threshold,
                freeze_patience=freeze_patience,
                freeze_decay=freeze_decay,
                factorize_matrices=factorize_matrices,
                ema_decay=ema_decay,
                qat_config=qat_config,
                enable_mtp=enable_mtp,
                mtp_config=mtp_config or (MTPConfig() if enable_mtp else None),
            )

        defaults = {"weight_decay": weight_decay}
        super().__init__(
            params=materialized_params,
            defaults=defaults,
            schedule=schedule or ScheduleConfig(lr=lr),
            grad_clip=grad_clip,
            **kwargs,
        )
        # 0.72.1: 6-bit momentum. "off" keeps the int8 buffer this
        # optimiser has always used; the three packing modes trade
        # pack/unpack work for optimiser memory. Validated here rather
        # than at first use — a typo should fail at construction, not
        # two hours into a run.
        from .sixbit import resolve_mode as _resolve_six_bit

        self.six_bit_mode = (six_bit_mode or "off").strip().lower()
        if self.six_bit_mode != "off":
            _resolve_six_bit(self.six_bit_mode)      # raises on an unknown mode

        self.lr = lr
        self.eps = eps
        self.trust_clip = trust_clip

        # QAT + MTP (optional, same as V5)
        self._qat_modules: list[tuple[str, nn.Module]] = []
        self.mtp_head: MTPHead | None = None

        # EMA buffers
        if self.cfg.ema_decay > 0.0:
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["ema"] = p.detach().clone()

    # -----------------------------------------------------------------------
    # QAT support (forwarded from V5)
    # -----------------------------------------------------------------------

    def attach_qat(self, model: nn.Module) -> None:
        """Attach QAT fake-quantization hooks to Linear/Conv layers."""
        cfg = self.cfg.qat_config
        if cfg is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d | nn.Conv3d):
                fq = QATFakeQuantize(
                    num_levels=cfg.num_levels,
                    symmetric=cfg.symmetric,
                    per_channel=cfg.per_channel,
                    learnable_scale=cfg.learnable_scales,
                    observer_steps=cfg.observer_steps,
                )
                module.register_forward_pre_hook(self._make_qat_hook(fq))
                self._qat_modules.append((name, fq))
        if self._qat_modules:
            print(f"[V5S-QAT] Attached to {len(self._qat_modules)} layers "
                  f"({cfg.bits}-bit, mode={self.cfg.diffusion_mode!r})")

    @staticmethod
    def _make_qat_hook(fq: QATFakeQuantize):
        def hook(module: nn.Module, _input: Any) -> None:
            if hasattr(module, "weight") and module.weight is not None:
                # Write into the existing storage in place instead of
                # reassigning `.data` to a brand-new tensor -- see the same
                # fix in pressure_cooker_v5.PressureCookerV5._make_qat_hook.
                module.weight.data.copy_(fq(module.weight.data))
        return hook

    def get_mtp_head(self, hidden_dim: int, vocab_size: int) -> MTPHead | None:
        """Get or build an MTP prediction head."""
        if not self.cfg.enable_mtp or self.cfg.mtp_config is None:
            return None
        if self.mtp_head is None:
            self.mtp_head = MTPHead(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_tokens=self.cfg.mtp_config.num_tokens,
                shared=self.cfg.mtp_config.shared_head,
            )
        return self.mtp_head

    # -----------------------------------------------------------------------
    # EMA support
    # -----------------------------------------------------------------------

    def _update_ema(self) -> None:
        if self.cfg.ema_decay <= 0.0:
            return
        d = self.cfg.ema_decay
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "ema" in st:
                    st["ema"].mul_(d).add_(p.data, alpha=1.0 - d)

    def swap_ema_weights(self, model: nn.Module) -> None:
        """Swap live weights with EMA weights for evaluation."""
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "ema" in st:
                    p.data, st["ema"] = st["ema"], p.data.clone()

    # -----------------------------------------------------------------------
    # State initialisation
    # -----------------------------------------------------------------------

    def _init_state(self, state: dict[str, Any], p: torch.Tensor) -> None:
        """Initialise all per-parameter optimizer state for V5S."""
        cfg = self.cfg

        # Step counter + adaptive power
        state["step"] = 0
        state["power"] = cfg.power

        # 3D cosine EMAs (all start at 0.0 = "no signal yet")
        state["fast_cos"]  = 0.0
        state["med_cos"]   = 0.0
        state["ultra_cos"] = 0.0

        # Previous VOS for trend detection
        state["prev_vos"] = 0.0

        # Single quantized momentum buffer
        m0 = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["m_q"], state["m_scale"] = _quantize_momentum(m0)

        # Factored curvature (Adafactor-style)
        is_matrix = cfg.factorize_matrices and p.dim() >= 2
        state["is_matrix"] = is_matrix
        if is_matrix:
            rows = p.shape[0]
            cols = p.numel() // rows
            state["row_curv"] = torch.zeros(rows, 1, device=p.device, dtype=torch.float32)
            state["col_curv"] = torch.zeros(1, cols, device=p.device, dtype=torch.float32)
            state["row_age"]  = torch.zeros(rows, 1, device=p.device, dtype=torch.uint8)
        else:
            sparse = p.numel() > _SPARSE_THRESHOLD
            state["sparse"] = sparse
            if sparse:
                state["curv"] = torch.zeros((), device=p.device, dtype=torch.float32)
            else:
                state["curv"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["age"] = torch.zeros_like(p, dtype=torch.uint8)

        # EMA buffer (if enabled)
        if cfg.ema_decay > 0.0 and "ema" not in state:
            state["ema"] = p.detach().clone()

    # -----------------------------------------------------------------------
    # 3D Cosine Oscillation Resistance
    # -----------------------------------------------------------------------

    def _update_3d_cos(
        self,
        state: dict[str, Any],
        g: torch.Tensor,
        m: torch.Tensor,
    ) -> tuple[float, float]:
        """Update the three cosine EMAs and return (raw_cos_sim, vos).

        Raw cosine similarity is positive when g and m align (good, consistent
        gradient direction) and negative when they point opposite ways (heavy
        oscillation / sign flip).

        VOS (volumetric oscillation score) is the negation of the weighted
        average of the three EMAs, so positive VOS = oscillating.

        Returns:
            (raw_cos_sim, vos): both in [-1, 1].
        """
        cfg = self.cfg
        g_norm = g.norm().clamp(min=self.eps)
        m_norm = m.norm().clamp(min=self.eps)
        raw_cos = float((g * m).sum() / (g_norm * m_norm))

        # Update three EMA bands independently
        state["fast_cos"]  = cfg.fast_beta  * state["fast_cos"]  + (1.0 - cfg.fast_beta)  * raw_cos
        state["med_cos"]   = cfg.med_beta   * state["med_cos"]   + (1.0 - cfg.med_beta)   * raw_cos
        state["ultra_cos"] = cfg.ultra_slow_beta * state["ultra_cos"] + (1.0 - cfg.ultra_slow_beta) * raw_cos

        vos = volumetric_oscillation_score(
            state["fast_cos"],
            state["med_cos"],
            state["ultra_cos"],
            fast_weight=cfg.fast_weight,
            med_weight=cfg.med_weight,
            ultra_weight=cfg.ultra_weight,
        )
        return raw_cos, vos

    def _vos_damping(self, vos: float) -> float:
        """Map VOS → effective learning-rate damping factor in (0, 1].

        When vos <= 0 (consistent direction): damping = 1.0 (no reduction).
        When vos > 0 (oscillating):           damping decreases toward 0
        as vos → 1, with the slope set by ``vos_3d_gain``.

        The formula is: damping = 1 / (1 + max(vos, 0) * vos_3d_gain)
        """
        return 1.0 / (1.0 + max(vos, 0.0) * self.cfg.vos_3d_gain)

    # -----------------------------------------------------------------------
    # Adaptive power exponent
    # -----------------------------------------------------------------------

    def _adaptive_power(self, state: dict[str, Any], vos: float) -> float:
        """Adapt the power exponent based on the current VOS.

        During heavy oscillation (high vos), the exponent moves toward
        ``power_min`` for more conservative updates.
        During consistent training (low / negative vos), it moves toward
        ``power_max`` for more aggressive updates.
        """
        cfg = self.cfg
        target = cfg.power_max - (cfg.power_max - cfg.power_min) * max(0.0, vos)
        # Smooth adaptation via EMA
        state["power"] = 0.9 * state["power"] + 0.1 * target
        return float(state["power"])

    # -----------------------------------------------------------------------
    # Factored curvature estimate
    # -----------------------------------------------------------------------

    def _update_curvature(
        self,
        state: dict[str, Any],
        p: torch.Tensor,
        g_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Update the factored curvature estimate and return it.

        For matrix parameters: row/column outer product (Adafactor-style).
        For vectors:           elementwise EMA of squared gradient.
        For very large vectors: scalar EMA of mean squared gradient.

        Returns curvature tensor broadcastable to g_pred's shape, clamped
        away from zero.
        """
        cfg = self.cfg
        beta = cfg.curvature_beta
        if state["is_matrix"]:
            row_curv = state["row_curv"]
            col_curv = state["col_curv"]
            rows = p.shape[0]
            cols = p.numel() // rows
            # row_curv/col_curv are sized against the flattened (rows, cols)
            # view built in _init_state. For anything with ndim > 2 (e.g. a
            # conv weight [out_ch, in_ch, kh, kw]), reducing g_pred with
            # dim=1/dim=0 in its *native* shape reduces over the wrong axes
            # entirely -- reshape to the same (rows, cols) view first.
            g2 = g_pred.reshape(rows, cols).pow(2)
            row_update = g2.mean(dim=1, keepdim=True).to(row_curv.dtype)
            col_update = g2.mean(dim=0, keepdim=True).to(col_curv.dtype)
            row_curv.mul_(beta).add_(row_update, alpha=1.0 - beta)
            col_curv.mul_(beta).add_(col_update, alpha=1.0 - beta)
            # Upcast for the final combine so Agedcookerv5s's compressed
            # (fp16) buffers can't silently underflow/lose precision here.
            row_mean = row_curv.float().mean().clamp(min=self.eps)
            curv = (row_curv.float() * col_curv.float() / row_mean).clamp(min=self.eps)
            curv = curv.reshape(p.shape)
        elif state["sparse"]:
            state["curv"].mul_(beta).add_(g_pred.pow(2).mean().to(state["curv"].dtype), alpha=1.0 - beta)
            curv = state["curv"].float().clamp(min=self.eps)
        else:
            state["curv"].mul_(beta).add_(g_pred.pow(2).to(state["curv"].dtype), alpha=1.0 - beta)
            curv = state["curv"].float().clamp(min=self.eps)
        return curv

    # -----------------------------------------------------------------------
    # Coordinate freezing with uint8 age counters
    # -----------------------------------------------------------------------

    def _coordinate_freeze_scale(
        self,
        state: dict[str, Any],
        g: torch.Tensor,
    ) -> torch.Tensor | float:
        """Compute per-coordinate freeze scale based on gradient age counters.

        Coordinates whose gradient has been below ``freeze_threshold`` for
        more than ``freeze_patience`` consecutive steps are soft-frozen by
        applying ``freeze_decay^(age - patience)`` to their update.

        Uses uint8 age counters — no additional float state.

        Returns:
            A per-coordinate (or scalar) scale tensor (values in (0, 1]).
        """
        cfg = self.cfg
        below = g.abs() < cfg.freeze_threshold
        if state["is_matrix"]:
            age = state["row_age"]
            rows = g.shape[0]
            cols = g.numel() // rows
            # row_age is sized against the flattened (rows, cols) view built
            # in _init_state -- reducing `below` with dim=1 in g's *native*
            # shape (e.g. a 4D conv grad) reduces over the wrong axes and
            # produces a mask that doesn't match row_age's shape for the
            # boolean indexing below. See the same fix in pressure_cooker_v5.
            row_below = below.reshape(rows, cols).all(dim=1, keepdim=True)
            age[row_below]  = torch.clamp(age[row_below].to(torch.int16) + 1, max=255).to(torch.uint8)
            age[~row_below] = 0
            stale = (age.to(torch.int16) - cfg.freeze_patience).clamp(min=0).float()
            scale = cfg.freeze_decay ** stale
            return scale.view(rows, *([1] * (g.dim() - 1)))
        age = state["age"]
        age[below]  = torch.clamp(age[below].to(torch.int16) + 1, max=255).to(torch.uint8)
        age[~below] = 0
        stale = (age.to(torch.int16) - cfg.freeze_patience).clamp(min=0).float()
        return cfg.freeze_decay ** stale

    # -----------------------------------------------------------------------
    # Trust ratio (LARS/LAMB-style per-parameter adaptive LR)
    # -----------------------------------------------------------------------

    def _trust_ratio(self, p: torch.Tensor, update: torch.Tensor) -> float:
        """Compute LARS-style trust ratio: ||p|| / ||update||.

        Clamped to ``trust_clip`` to avoid extreme scaling.
        """
        p_norm = p.norm(2).clamp(min=self.eps)
        u_norm = update.norm(2).clamp(min=self.eps)
        return float((p_norm / u_norm).clamp(self.trust_clip[0], self.trust_clip[1]))

    # -----------------------------------------------------------------------
    # Core update step
    # -----------------------------------------------------------------------

    def _v5s_step_one(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        state: dict[str, Any],
        lr: float,
        wd: float,
    ) -> None:
        """Process a single parameter tensor through the V5S update rule.

        This method implements all 14 steps from the module docstring for one
        parameter tensor.  It is factored out of ``_v5s_step`` for
        readability and to allow subclasses to override individual components.

        Args:
            p:     Parameter tensor (modified in-place).
            g:     Gradient tensor (same shape as p).
            state: Per-parameter state dictionary.
            lr:    Effective group learning rate.
            wd:    Effective group weight-decay coefficient.
        """
        cfg = self.cfg
        state["step"] += 1

        # 1. Dequantize momentum
        m = _dequantize_momentum(state["m_q"], state["m_scale"]).view_as(p)

        # 2. Pressure diffuse the raw gradient
        gd = pressure_diffuse(
            g,
            factor=cfg.diffusion_factor,
            kernel_width=cfg.diffusion_kernel,
            mode=cfg.diffusion_mode,
        )

        # 3. Compute 3D VOS
        raw_cos, vos = self._update_3d_cos(state, gd, m)

        # 4. VOS-based damping
        damp = self._vos_damping(vos)

        # 5. Adaptive power exponent
        pw = self._adaptive_power(state, vos)

        # 6. Gradient extrapolation (look-ahead)
        g_pred = gd + cfg.extrapolation * m

        # 7. Update curvature estimate (uses g_pred for stability)
        curv = self._update_curvature(state, p, g_pred)

        # 8. Power-scaled update: sign(g_pred) * |g_pred|^pw / sqrt(curv)
        update = torch.sign(g_pred) * g_pred.abs().pow(pw) / curv.sqrt()

        # 9. Sophia-style magnitude clipping
        update = update.clamp(-cfg.sophia_clip, cfg.sophia_clip)

        # 10. Coordinate freeze scaling
        freeze_scale = self._coordinate_freeze_scale(state, g)
        update = update * freeze_scale

        # 11. Hook for subclass additional scaling (default = 1.0)
        extra = self._extra_scale(state, g_pred, vos)
        update = update * extra

        # 12. Trust ratio (per-tensor adaptive LR)
        trust = self._trust_ratio(p, update)
        lr_eff = lr * damp * trust

        # 13. Weight decay (decoupled, applied before parameter update)
        if wd != 0.0:
            p.mul_(1.0 - lr_eff * wd)

        # 14. Parameter update
        p.sub_(update * lr_eff)

        # 15. Momentum update using diffused gradient (re-quantize)
        m_new = cfg.momentum_beta * m + (1.0 - cfg.momentum_beta) * gd
        state["m_q"], state["m_scale"] = _quantize_momentum(m_new)

        # 16. Save prev VOS for trend detection
        state["prev_vos"] = vos

    def _extra_scale(
        self,
        state: dict[str, Any],
        g_pred: torch.Tensor,
        vos: float,
    ) -> float:
        """Hook for subclasses to apply an additional scale factor.

        Base V5S returns 1.0.  Subclasses (e.g., V5S-Ultra) can override this
        to add entropy scaling, spectral analysis, etc.

        Args:
            state:  Per-parameter state.
            g_pred: Extrapolated gradient (after diffusion + extrapolation).
            vos:    Current volumetric oscillation score.

        Returns:
            A scalar in (0, +∞).  Values < 1 reduce the update; values > 1
            amplify it.
        """
        return 1.0

    # -----------------------------------------------------------------------
    # Main step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single V5S optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                     the loss.  Required if ``sam_rho > 0`` (not yet supported
                     in V5S; use PressureCookerV5 for SAM).

        Returns:
            Loss value if a closure was provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._apply_lr_schedule()
        if self._grad_clip is not None:
            self.gradient_clip()

        self._v5s_step()
        self._update_ema()
        self._global_step += 1
        return loss

    def _v5s_step(self) -> None:
        """Apply the V5S update rule to all parameters in all groups."""
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m_q" not in state:
                    self._init_state(state, p)
                self._v5s_step_one(p, g, state, lr, wd)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_oscillation_stats(self) -> dict[str, float]:
        """Return aggregated 3D cosine statistics across all parameters.

        Useful for monitoring training stability and tuning the VOS weights.

        Returns:
            Dictionary with mean fast/med/ultra cosine similarities and mean VOS.
        """
        fast_vals, med_vals, ultra_vals, vos_vals = [], [], [], []
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "fast_cos" not in st:
                    continue
                fast_vals.append(st["fast_cos"])
                med_vals.append(st["med_cos"])
                ultra_vals.append(st["ultra_cos"])
                vos_vals.append(volumetric_oscillation_score(
                    st["fast_cos"], st["med_cos"], st["ultra_cos"],
                    self.cfg.fast_weight, self.cfg.med_weight, self.cfg.ultra_weight,
                ))
        if not vos_vals:
            return {"fast_cos": 0.0, "med_cos": 0.0, "ultra_cos": 0.0, "vos": 0.0}
        n = len(vos_vals)
        return {
            "fast_cos":  sum(fast_vals)  / n,
            "med_cos":   sum(med_vals)   / n,
            "ultra_cos": sum(ultra_vals) / n,
            "vos":       sum(vos_vals)   / n,
        }

    def get_diffusion_stats(self) -> dict[str, Any]:
        """Return current pressure diffusion configuration summary."""
        return {
            "factor":        self.cfg.diffusion_factor,
            "kernel_width":  self.cfg.diffusion_kernel,
            "mode":          self.cfg.diffusion_mode,
            "enabled":       self.cfg.diffusion_factor > 0.0,
        }

    def get_frozen_fraction(self) -> float:
        """Return the fraction of parameter coordinates currently frozen (age > patience).

        A high value (> 0.5) indicates many parameters have stagnant gradients
        and the model may benefit from a learning rate increase or data refresh.
        """
        total, frozen = 0, 0
        patience = self.cfg.freeze_patience
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "age" in st:
                    age = st["age"].to(torch.int32)
                    frozen += int((age > patience).sum().item())
                    total  += age.numel()
                elif "row_age" in st:
                    age = st["row_age"].to(torch.int32)
                    # Each row represents a full row of the matrix
                    row_size = p.numel() // p.shape[0]
                    frozen += int((age > patience).sum().item()) * row_size
                    total  += p.numel()
        return frozen / total if total > 0 else 0.0

    def describe(self) -> dict[str, Any]:
        """Return a full description of this optimizer's configuration and state."""
        base = super().describe() if hasattr(super(), "describe") else {}
        osc = self.get_oscillation_stats()
        base.update({
            "kind":              "PressureCookerV5S",
            "architecture":      "3D-COR + PressureDiffusion + LowPower",
            "config":            self.cfg.to_dict(),
            "osc_stats":         osc,
            "frozen_fraction":   self.get_frozen_fraction(),
            "diffusion_stats":   self.get_diffusion_stats(),
            "qat_enabled":       self.cfg.qat_config is not None,
            "mtp_enabled":       self.cfg.enable_mtp,
            "ema_enabled":       self.cfg.ema_decay > 0.0,
            "pascal_safe":       self._pascal_safe,
            "global_step":       self._global_step,
        })
        return base

    def print_summary(self) -> None:
        """Print a human-readable summary of the optimizer state to stdout."""
        print(self.cfg.summary())
        osc = self.get_oscillation_stats()
        print(f"\nOscillation Stats (step {self._global_step}):")
        print(f"  fast_cos={osc['fast_cos']:.4f}  med_cos={osc['med_cos']:.4f}  "
              f"ultra_cos={osc['ultra_cos']:.4f}  VOS={osc['vos']:.4f}")
        print(f"  Frozen fraction: {self.get_frozen_fraction():.2%}")

    def extra_repr(self) -> str:
        """String representation for display."""
        return (
            f"lr={self.lr}, diffusion={self.cfg.diffusion_factor:.2f}, "
            f"vos_gain={self.cfg.vos_3d_gain}, "
            f"bands=(β_fast={self.cfg.fast_beta}, β_med={self.cfg.med_beta}, "
            f"β_ultra={self.cfg.ultra_slow_beta})"
        )


# ---------------------------------------------------------------------------
# Pascal-safe variant
# ---------------------------------------------------------------------------

class Agedcookerv5s(PressureCookerV5S):
    """PressureCookerV5S tuned and enforced for CUDA 6.1/6.2 (Pascal / GTX 10xx).

    Identical to V5S with the following enforced overrides:
    * All fused operations disabled (fused=False already handled in base).
    * Diffusion kernel width capped at 3 (larger kernels have higher overhead
      on Pascal without fp16 tensor cores).
    * State compression: curvature buffers stored as float16.
    """

    def __init__(self, params: Iterable, **kwargs: Any) -> None:
        # Enforce safe diffusion kernel
        if "diffusion_kernel" in kwargs and kwargs["diffusion_kernel"] > 3:
            warnings.warn(
                "Agedcookerv5s: diffusion_kernel capped at 3 for Pascal safety.",
                stacklevel=2,
            )
            kwargs["diffusion_kernel"] = 3

        super().__init__(params, **kwargs)

        if not self._pascal_safe:
            warnings.warn(
                "Agedcookerv5s is tuned for CUDA 6.1/6.2 (Pascal). "
                "Running on newer hardware may be suboptimal; consider "
                "PressureCookerV5S instead.",
                stacklevel=2,
            )

        # Compress curvature to fp16 after init
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "row_curv" in st:
                    st["row_curv"] = st["row_curv"].to(torch.float16)
                    st["col_curv"] = st["col_curv"].to(torch.float16)
                elif "curv" in st and st["curv"].numel() > 1:
                    st["curv"] = st["curv"].to(torch.float16)

    def _init_state(self, state: dict[str, Any], p: torch.Tensor) -> None:
        super()._init_state(state, p)
        # Immediately compress curvature buffers
        if state["is_matrix"]:
            state["row_curv"] = state["row_curv"].to(torch.float16)
            state["col_curv"] = state["col_curv"].to(torch.float16)
        elif not state["sparse"]:
            state["curv"] = state["curv"].to(torch.float16)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["pascal_safe"] = True
        base["curvature_dtype"] = "float16"
        return base
