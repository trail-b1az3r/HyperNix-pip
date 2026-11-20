"""hypernix.pressure_cooker_v7 — Pressure Cooker V7: the balanced optimizer.

V5 and V5S are memory-first: quantised momentum and factored curvature, at
the price of several small operations and host/device synchronisations per
parameter tensor per step. V6 is speed-first: one fp32 momentum buffer,
fused multi-tensor operations, one synchronisation per step, and no
adaptivity beyond a per-tensor trust ratio. V7 is built to sit between
them, around three measurements (25M parameters, 4 CPU threads; see
``hnx optimizer benchmark`` for your own):

* a **norm pass costs as much as an update pass** (6.8 ms against 5.6 ms).
  V6 computes two or three norms every step; that is about half its step.
* a **fused lerp** does the momentum update in one pass where a
  multiply-then-add takes two.
* **half-precision or int8 momentum costs 4-5x fp32 on a CPU**, which has
  no fast path for them, and is cheap on a GPU, which does.

So V7 keeps V6's architecture (few buffers, ``torch._foreach_*``, batched
synchronisation) and spends its budget differently:

**Statistics on an interval.** The per-tensor quantities that steer the
update -- parameter norm, momentum norm, the gradient's alignment with the
momentum -- change slowly. V7 measures them every ``stat_interval`` steps
(4 by default) and reuses the resulting coefficients in between. Only the
gradient norm is measured every step, because clipping and the non-finite
check need this step's value.

**Three timescales from one number.** V5S keeps three cosine EMAs per
tensor and reads each back to the host. V7 computes one cosine per tensor
(gradient against momentum) as a device vector, and keeps fast, medium and
slow EMAs of that vector: three floats per *tensor*, never a float read
per tensor. Their weighted sum is the **stability score** in [-1, 1].

**Adaptive pressure.** The step each tensor takes is scaled by a pressure
coefficient built from that score (a steady direction is pushed harder, an
oscillating one is damped), from the gradient norm against its own running
average (a spike is damped, which is the part of Adam's behaviour that
matters most and costs one float per tensor instead of one per element),
and from the reported loss when there is one. It is clamped, and it is a
vector on the device: applying it costs no synchronisation.

**Compressed momentum.** ``state_dtype`` is ``fp32``, ``bf16``, ``fp16``
or ``int8`` (one byte per parameter, with a per-tensor scale). ``auto``
picks from the device, the model's size and the memory that is free, and
says why. On a CPU it picks fp32 unless memory is short, because of the
third measurement above.

**Optional, and off unless they earn it:** factored row/column statistics
for matrices (per-row and per-column scaling from two small vectors, never
a full second-moment tensor); pressure diffusion (a three-tap smoothing of
the gradient, applied in proportion to measured oscillation and skipped
when there is none); an EMA shadow of the weights.

One synchronisation per step
----------------------------
With ``skip_on_nonfinite`` (the default) V7 reads one small vector back
per step: whether the gradients are finite, and the diagnostics
(stability, pressure, gradient norm). That one read replaces V6's
``tolist()`` and its separate clipping read. Turn it off and a step
performs none.

Update rule (per bucket of same-device, same-dtype tensors)
-----------------------------------------------------------
::

    g      gradient (optionally diffused, optionally row/column scaled)
    n_g    = ||g|| per tensor                       every step
    c      = clip * spike damping                    per tensor, on device
    m      = beta_t * m + (1 - beta) * c * g        compressed momentum
    every stat_interval steps:
        cos    = <g, m> / (n_g ||m||)
        score  = w_f ema_f(cos) + w_m ema_m(cos) + w_s ema_s(cos)
        trust  = clamp(||p|| / ||m||, trust_clip)
        press  = clamp((1 + gain * score) * loss_factor, pressure_clip)
    p      = p * (1 - lr * wd)
    p      = p - lr / (1 - beta^t) * trust * press * m

What it is not: there is no per-element second moment, so V7 is not Adam.
A tensor whose elements have very different gradient scales is better
served by ``factored=True`` or by AdamW. V7 is not claimed to be better
than AdamW in general; the benchmark is free to show where AdamW wins.

The family (see the classes below) is this one implementation configured
five ways: Balanced (the default), Mini, Q, S and M.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from hypernix.optimizers.optimizer_framework import OptimizerBase, ScheduleConfig
from hypernix.optimizers.pressure_cooker_v3 import _flatten_optimizer_params

__all__ = [
    "PressureCookerV7",
    "V7Balanced",
    "PressureCookerV7Mini",
    "V7Mini",
    "PressureCookerV7Q",
    "V7Q",
    "PressureCookerV7S",
    "V7S",
    "PressureCookerV7M",
    "V7M",
    "V7_PRESETS",
    "V7_FORMAT",
    "STATE_DTYPES",
    "resolve_state_dtype",
    "MINI_MIN_PARAMS",
    "MINI_MAX_PARAMS",
]

#: Version of the V7 state layout, written into every checkpoint.
V7_FORMAT = 1
STATE_DTYPES = ("fp32", "bf16", "fp16", "int8")
#: V7Mini's band: strictly more than the first, strictly fewer than the second.
MINI_MIN_PARAMS = 39_000_000
MINI_MAX_PARAMS = 750_000_000

_HAS_FOREACH = all(hasattr(torch, name) for name in (
    "_foreach_mul_", "_foreach_add_", "_foreach_addcmul_", "_foreach_norm", "_foreach_lerp_"))
_TORCH_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
                "int8": torch.int8}

#: The named configurations. A preset sets defaults; an explicit argument wins.
V7_PRESETS: dict[str, dict[str, Any]] = {
    # Speed, memory and convergence together.
    "balanced": dict(state_dtype="auto", stat_interval=4, stability=True, adaptive_pressure=True,
                     trust_ratio=True, diffusion=True, diffusion_factor=0.05, factored=True,
                     ema_decay=None, skip_on_nonfinite=True, lazy_weight_decay=False),
    # The least machinery that is still V7: no diffusion, statistics rarely.
    "fast": dict(state_dtype="fp32", stat_interval=8, stability=True, adaptive_pressure=True,
                 trust_ratio=True, diffusion=False, diffusion_factor=0.0, factored=False,
                 ema_decay=None, skip_on_nonfinite=True, lazy_weight_decay=False),
    # The smallest state: one byte of momentum per parameter, factored statistics.
    "memory": dict(state_dtype="int8", stat_interval=4, stability=True, adaptive_pressure=True,
                   trust_ratio=True, diffusion=False, diffusion_factor=0.0, factored=True,
                   ema_decay=None, skip_on_nonfinite=True, lazy_weight_decay=False),
    # More of the stability machinery, measured more often.
    "stable": dict(state_dtype="auto", stat_interval=2, stability=True, adaptive_pressure=True,
                   trust_ratio=True, diffusion=True, diffusion_factor=0.15, factored=True,
                   ema_decay=None, skip_on_nonfinite=True, lazy_weight_decay=False,
                   stability_gain=0.5, momentum_damping=0.5),
}


def resolve_state_dtype(n_params: int, device: torch.device | str, *,
                        param_dtype: torch.dtype = torch.float32,
                        free_bytes: int | None = None) -> tuple[str, str]:
    """``(state dtype, why)`` for ``state_dtype="auto"``.

    The rule, in order:

    1. Parameters that are themselves half precision keep their own dtype.
    2. If fp32 momentum would take more than a quarter of the free memory
       on CUDA, or more than half of it on a CPU, int8 (one byte per
       parameter). A CPU gets the higher bar because int8 costs it about
       four times the step time; measured at 749M parameters, a quarter
       switched a CPU with 9 GiB free to int8 for 3 GiB of momentum.
    3. On CUDA, bf16 where the device has it, else fp16: half the bytes of
       fp32, in arithmetic a GPU does natively.
    4. On a CPU, fp32: half precision and int8 arithmetic measure 4-5x
       slower there, so the bytes are not worth the time unless rule 2 says
       memory is short.
    """
    device = torch.device(device)
    if param_dtype in (torch.float16, torch.bfloat16):
        name = "fp16" if param_dtype == torch.float16 else "bf16"
        return name, f"the parameters are {name}, so the momentum is too"
    fp32_bytes = int(n_params) * 4
    if free_bytes is None:
        free_bytes = _free_bytes(device)
    share, words = (4, "a quarter") if device.type == "cuda" else (2, "half")
    if free_bytes and fp32_bytes > free_bytes // share:
        return "int8", (f"fp32 momentum ({fp32_bytes / 2**30:.2f} GiB) would take more than "
                        f"{words} of the {free_bytes / 2**30:.2f} GiB free, so int8 momentum")
    if device.type == "cuda":
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001 - an old torch or driver: assume not
            bf16 = False
        if bf16:
            return "bf16", "a CUDA device with bf16: half the bytes of fp32, natively"
        return "fp16", "a CUDA device without bf16: fp16 momentum, half the bytes of fp32"
    return "fp32", ("a CPU with memory to spare: half precision and int8 arithmetic are "
                    "4-5x slower than fp32 here")


def _free_bytes(device: torch.device) -> int:
    try:
        if device.type == "cuda":
            return int(torch.cuda.mem_get_info(device)[0])
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - unknown is "not short of memory"
        return 0


class _Bucket:
    """The tensors of one param group that share a device and a dtype.

    Everything per-tensor that steers the update lives here as a vector of
    length ``len(params)`` on the device, so it is updated and applied
    without a host read per tensor.
    """

    __slots__ = ("params", "group", "key", "cos_fast", "cos_med", "cos_slow", "gnorm_ema",
                 "coef", "score", "pressure", "trust", "decay_acc", "has_factored",
                 "unfactored")

    def __init__(self, params: list[torch.Tensor], group: dict[str, Any], key: tuple) -> None:
        self.params = params
        self.group = group
        self.key = key
        n = len(params)
        device = params[0].device
        self.cos_fast = torch.zeros(n, device=device)
        self.cos_med = torch.zeros(n, device=device)
        self.cos_slow = torch.zeros(n, device=device)
        self.gnorm_ema = torch.zeros(n, device=device)
        self.score = torch.zeros(n, device=device)
        self.pressure = torch.ones(n, device=device)
        self.trust = torch.ones(n, device=device)
        #: trust * pressure, the coefficient reused between statistics steps.
        self.coef = torch.ones(n, device=device)
        self.decay_acc = 1.0
        self.has_factored = False
        #: 1 where the tensor has no factored statistics (and so gets the trust ratio).
        self.unfactored = torch.ones(n, device=device)

    VECTORS = ("cos_fast", "cos_med", "cos_slow", "gnorm_ema", "score", "pressure", "trust", "coef")


class PressureCookerV7(OptimizerBase):
    """Pressure Cooker V7 (Balanced): see the module docstring for the design.

    Args:
        params: Parameters or param-group dicts.
        preset: ``"balanced"`` (default), ``"fast"``, ``"memory"`` or
            ``"stable"``: a set of defaults. Any argument given explicitly
            wins over the preset.
        schedule: A :class:`ScheduleConfig`; built from ``lr`` when omitted.
        lr: Peak learning rate.
        momentum_beta: Decay of the momentum buffer.
        weight_decay: Decoupled weight decay.
        state_dtype: ``"auto"``, ``"fp32"``, ``"bf16"``, ``"fp16"`` or
            ``"int8"``: how the momentum is stored.
        stat_interval: Steps between measurements of the per-tensor
            statistics (trust ratio, stability). 1 measures every step.
        stability: Track the three-timescale stability score.
        stability_gain: How strongly the score moves the pressure.
        momentum_damping: How strongly oscillation shortens the momentum's
            memory (0 turns it off).
        adaptive_pressure: Scale each tensor's step by its pressure.
        pressure_clip: ``(min, max)`` for the pressure.
        spike_power: Exponent of the gradient-spike damping (0 turns it off).
        trust_ratio: Per-tensor ``clamp(||p|| / ||m||)`` scaling, as V6.
        trust_clip: ``(min, max)`` for the trust ratio.
        diffusion: Smooth the gradient of matrices with a three-tap kernel,
            in proportion to measured oscillation.
        diffusion_factor: The most diffusion ever applied, in [0, 1].
        diffusion_kernel: The side weight of the three-tap kernel (the
            centre gets the rest). 0.25 is a triangle; V5S's Gaussian is 0.107.
        factored: Row and column statistics for matrices: per-row and
            per-column scaling of the gradient from two small vectors.
        factored_beta: Decay of those statistics.
        factored_min: The smallest dimension a matrix must have to be factored.
        ema_decay: Keep an EMA shadow of the weights with this decay
            (``None``: no shadow). It is a full extra copy of the weights.
        grad_clip: Global gradient-norm clip, folded into the momentum
            update (no separate pass). ``None`` turns it off.
        grad_scaler: A ``GradScaler`` for fp16 autocast training.
        grad_accum_steps: Apply an update every Nth call to :meth:`step`.
        skip_on_nonfinite: Skip the update, leaving all state untouched,
            when any gradient is NaN or infinite. Costs the step's one
            host read, which also carries the diagnostics.
        lazy_weight_decay: Apply the accumulated decay every
            ``stat_interval`` steps instead of every step (one pass fewer).
            The difference is that updates made in between are decayed
            together with the weights; it is of order ``lr * wd`` per step.
        foreach: Use ``torch._foreach_*``. Off, or unavailable, runs the
            same arithmetic one tensor at a time.
        eps: Floor for denominators.
    """

    variant = "balanced"
    lineage = "V7"

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        preset: str | None = None,
        schedule: ScheduleConfig | None = None,
        lr: float = 3e-4,
        momentum_beta: float = 0.9,
        weight_decay: float = 0.01,
        state_dtype: str | None = None,
        stat_interval: int | None = None,
        stability: bool | None = None,
        stability_gain: float | None = None,
        momentum_damping: float | None = None,
        stability_betas: tuple[float, float, float] = (0.6, 0.9, 0.99),
        stability_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
        adaptive_pressure: bool | None = None,
        pressure_clip: tuple[float, float] = (0.25, 2.0),
        spike_power: float = 0.5,
        trust_ratio: bool | None = None,
        trust_clip: tuple[float, float] = (0.05, 5.0),
        diffusion: bool | None = None,
        diffusion_factor: float | None = None,
        diffusion_kernel: float = 0.25,
        factored: bool | None = None,
        factored_beta: float = 0.99,
        factored_min: int = 16,
        ema_decay: float | None = None,
        grad_clip: float | None = 1.0,
        grad_scaler: Any = None,
        grad_accum_steps: int = 1,
        skip_on_nonfinite: bool | None = None,
        lazy_weight_decay: bool | None = None,
        foreach: bool = True,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        preset_name = (preset or self._default_preset()).lower()
        if preset_name not in V7_PRESETS:
            raise ValueError(f"preset must be one of {', '.join(V7_PRESETS)}, not {preset!r}")
        chosen = {**V7_PRESETS[preset_name], **self._variant_defaults()}

        def pick(name: str, given: Any) -> Any:
            return chosen[name] if given is None else given

        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if not 0.0 <= momentum_beta < 1.0:
            raise ValueError("momentum_beta must be in [0, 1)")
        materialized = _flatten_optimizer_params(params)
        super().__init__(
            params=materialized, defaults={"weight_decay": weight_decay},
            schedule=schedule or ScheduleConfig(lr=lr),
            # Clipping is done here, folded into the momentum update.
            grad_clip=None, **kwargs,
        )
        self.preset = preset_name
        self.lr = lr
        self.momentum_beta = float(momentum_beta)
        self.weight_decay = weight_decay
        self.stat_interval = max(1, int(pick("stat_interval", stat_interval)))
        self.stability = bool(pick("stability", stability))
        self.stability_gain = float(chosen.get("stability_gain", 0.3)
                                    if stability_gain is None else stability_gain)
        self.momentum_damping = float(chosen.get("momentum_damping", 0.25)
                                      if momentum_damping is None else momentum_damping)
        self.stability_betas = tuple(float(b) for b in stability_betas)
        self.stability_weights = tuple(float(w) for w in stability_weights)
        if len(self.stability_betas) != 3 or len(self.stability_weights) != 3 \
                or abs(sum(self.stability_weights) - 1.0) > 1e-6:
            raise ValueError("stability_betas and stability_weights are three values each, "
                             "and the weights sum to 1")
        self.adaptive_pressure = bool(pick("adaptive_pressure", adaptive_pressure))
        self.pressure_clip = (float(pressure_clip[0]), float(pressure_clip[1]))
        self.spike_power = float(spike_power)
        self.trust_ratio = bool(pick("trust_ratio", trust_ratio))
        self.trust_clip = (float(trust_clip[0]), float(trust_clip[1]))
        self.diffusion_factor = float(pick("diffusion_factor", diffusion_factor))
        self.diffusion = bool(pick("diffusion", diffusion)) and self.diffusion_factor > 0
        if not 0.0 <= self.diffusion_factor <= 1.0:
            raise ValueError("diffusion_factor must be in [0, 1]")
        self.diffusion_kernel = float(diffusion_kernel)
        self.factored = bool(pick("factored", factored))
        self.factored_beta = float(factored_beta)
        self.factored_min = int(factored_min)
        self.ema_decay = chosen.get("ema_decay") if ema_decay is None else ema_decay
        self.clip_norm = grad_clip
        self.grad_scaler = grad_scaler
        self.grad_accum_steps = grad_accum_steps
        self.skip_on_nonfinite = bool(pick("skip_on_nonfinite", skip_on_nonfinite))
        self.lazy_weight_decay = bool(pick("lazy_weight_decay", lazy_weight_decay))
        self.eps = eps
        self.foreach = bool(foreach and _HAS_FOREACH)
        if foreach and not _HAS_FOREACH:
            warnings.warn(f"torch {torch.__version__} lacks the torch._foreach_* operations V7 "
                          "uses; it will run one tensor at a time. Correct, not fused.",
                          stacklevel=2)

        tensors = [p for group in self.param_groups for p in group["params"]]
        self.n_params = sum(p.numel() for p in tensors)
        requested = pick("state_dtype", state_dtype).lower()
        if requested == "auto":
            first = tensors[0] if tensors else torch.zeros(1)
            self.state_dtype, self.state_dtype_reason = resolve_state_dtype(
                self.n_params, first.device, param_dtype=first.dtype)
        elif requested in STATE_DTYPES:
            self.state_dtype, self.state_dtype_reason = requested, "asked for"
        else:
            raise ValueError(f"state_dtype must be auto or one of {', '.join(STATE_DTYPES)}, "
                             f"not {state_dtype!r}")
        self.state_dtype_requested = requested

        self._buckets: list[_Bucket] = []
        self._bucket_signature: tuple = ()
        self._accum_counter = 0
        self._skipped_steps = 0
        self._loss_fast: float | None = None
        self._loss_slow: float | None = None
        self._loss_factor = 1.0
        self._osc_host = 0.0          # the last oscillation read back; gates diffusion
        self._clip_total: torch.Tensor | None = None
        self._pending_bucket_state: dict[tuple, Any] | None = None
        #: What the step's one host read brought back (empty until the first).
        self.last_diagnostics: dict[str, float] = {}

    # -- configuration hooks for the family ---------------------------------

    def _default_preset(self) -> str:
        return "balanced"

    def _variant_defaults(self) -> dict[str, Any]:
        return {}

    # -- loss feedback -------------------------------------------------------

    def report_loss(self, loss: float) -> None:
        """Tell the optimizer the training loss (optional).

        A loss that is rising against its own slow average lowers the
        pressure of every tensor until it stops rising. Pass the float you
        already have for logging: this performs no device read of its own.
        """
        value = float(loss)
        if not math.isfinite(value):
            return
        if self._loss_fast is None or self._loss_slow is None:
            self._loss_fast = self._loss_slow = value
            return
        self._loss_fast = 0.7 * self._loss_fast + 0.3 * value
        self._loss_slow = 0.98 * self._loss_slow + 0.02 * value
        rising = self._loss_fast > self._loss_slow * 1.02 + 1e-12
        target = 0.5 if rising else 1.0
        self._loss_factor += 0.2 * (target - self._loss_factor)

    # -- state ---------------------------------------------------------------

    def _new_momentum(self, p: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(p, dtype=self._momentum_dtype(p),
                                memory_format=torch.preserve_format)

    def _momentum_dtype(self, p: torch.Tensor) -> torch.dtype:
        if self.state_dtype == "fp32":
            return p.dtype            # "full precision" is the parameter's own
        return _TORCH_DTYPE[self.state_dtype]

    def _factorable(self, p: torch.Tensor) -> bool:
        return (self.factored and p.dim() >= 2 and p.shape[0] >= self.factored_min
                and p.numel() // p.shape[0] >= self.factored_min)

    def _init_state(self, state: dict[str, Any], p: torch.Tensor) -> None:
        state["momentum"] = self._new_momentum(p)
        if self.state_dtype == "int8":
            state["scale"] = torch.zeros((), device=p.device, dtype=torch.float32)
        if self._factorable(p):
            state["row"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
            state["col"] = torch.zeros(p.numel() // p.shape[0], device=p.device,
                                       dtype=torch.float32)
        if self.ema_decay is not None:
            state["ema"] = p.detach().clone()

    def _collect(self) -> list[_Bucket]:
        """The buckets for the parameters that have a gradient this step."""
        signature = tuple(id(p) for group in self.param_groups for p in group["params"]
                          if p.grad is not None)
        if signature == self._bucket_signature and self._buckets:
            return self._buckets
        members: dict[tuple, tuple[list[torch.Tensor], dict[str, Any]]] = {}
        for index, group in enumerate(self.param_groups):
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("PressureCookerV7 does not support sparse gradients; "
                                       "use torch.optim.SparseAdam for that parameter")
                if "momentum" not in self.state[p]:
                    self._init_state(self.state[p], p)
                key = (index, str(p.device), str(p.dtype))
                members.setdefault(key, ([], group))[0].append(p)
        rebuilt = [_Bucket(tensors, group, key) for key, (tensors, group) in members.items()]
        for bucket in rebuilt:
            flags = ["row" in self.state[p] for p in bucket.params]
            bucket.has_factored = any(flags)
            bucket.unfactored = torch.tensor([0.0 if f else 1.0 for f in flags],
                                             device=bucket.params[0].device)
        previous = {b.key: b for b in self._buckets}
        for bucket in rebuilt:
            old = previous.get(bucket.key)
            if old is not None and len(old.params) == len(bucket.params):
                for name in _Bucket.VECTORS:        # the same tensors: keep their statistics
                    setattr(bucket, name, getattr(old, name))
                bucket.decay_acc = old.decay_acc
        self._buckets = rebuilt
        self._bucket_signature = signature
        self._restore_bucket_state()
        return self._buckets

    # -- the gradient, before it reaches the momentum --------------------------

    def _diffuse(self, g: torch.Tensor, strength: float) -> torch.Tensor:
        """One conv with a blended three-tap kernel: no separate lerp pass."""
        side = strength * self.diffusion_kernel
        kernel = torch.tensor([side, 1.0 - 2.0 * side, side], device=g.device,
                              dtype=g.dtype).view(1, 1, 3)
        return F.conv1d(g.reshape(1, 1, -1), kernel, padding=1).view(g.shape)

    def _refresh_factors(self, g: torch.Tensor, state: dict[str, Any]) -> None:
        """Update a matrix's row and column statistics from this gradient, and
        the two scaling vectors derived from them.

        Adafactor's factorisation of the second moment: ``v_ij ~ r_i c_j /
        mean(r)``. V7 uses it to scale the gradient by ``1 / sqrt(v_ij)``,
        which is ``rs_i * cs_j`` with the two vectors kept here.
        """
        sq = g.reshape(g.shape[0], -1).float().square()
        row, col = state["row"], state["col"]
        beta = self.factored_beta if state.get("seeded") else 0.0
        row.mul_(beta).add_(sq.mean(dim=1), alpha=1.0 - beta)
        col.mul_(beta).add_(sq.mean(dim=0), alpha=1.0 - beta)
        state["seeded"] = True
        state["rs"] = (row / row.mean().clamp_min(self.eps)).clamp_min(self.eps).rsqrt()
        state["cs"] = col.clamp_min(self.eps).rsqrt()

    def _prepare_grads(self, bucket: _Bucket) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = [p.grad for p in bucket.params]  # type: ignore[misc]
        strength = self.diffusion_factor * min(1.0, max(0.0, self._osc_host)) \
            if self.diffusion else 0.0
        if strength <= 1e-4:
            return grads                              # the common case: untouched
        return [self._diffuse(g, strength) if g.dim() >= 2 else g for g in grads]

    # -- one bucket's step -------------------------------------------------------

    def _foreach_ok(self, tensors: list[torch.Tensor]) -> bool:
        return self.foreach and len(tensors) > 0

    def _norms(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if self._foreach_ok(tensors):
            return torch.stack(list(torch._foreach_norm(tensors))).float()
        return torch.stack([t.norm(2) for t in tensors]).float()

    def _update_momentum(self, bucket: _Bucket, grads: list[torch.Tensor],
                         gscale: torch.Tensor | None, beta_t: torch.Tensor | None
                         ) -> list[torch.Tensor]:
        """m = beta_t * m + (1 - beta) * gscale * g~. Returns float views of m.

        ``g~`` is the gradient, scaled by its row and column factors for the
        tensors that have them.
        """
        beta = self.momentum_beta
        states = [self.state[p] for p in bucket.params]
        moms = [s["momentum"] for s in states]
        compressed = self.state_dtype != "fp32" and moms[0].dtype != grads[0].dtype
        if compressed or bucket.has_factored:
            return self._update_momentum_each(states, grads, gscale, beta_t, compressed)
        if gscale is None and beta_t is None and self._foreach_ok(moms):
            torch._foreach_lerp_(moms, grads, 1.0 - beta)          # one fused pass
            return moms
        scales = list(gscale.to(moms[0].dtype).unbind(0)) if gscale is not None else None
        betas = list(beta_t.to(moms[0].dtype).unbind(0)) if beta_t is not None else None
        if self._foreach_ok(moms):
            if betas is None:
                torch._foreach_mul_(moms, beta)
            else:
                torch._foreach_mul_(moms, betas)
            if scales is None:
                torch._foreach_add_(moms, grads, alpha=1.0 - beta)
            else:
                torch._foreach_addcmul_(moms, grads, scales, value=1.0 - beta)
            return moms
        return self._update_momentum_each(states, grads, gscale, beta_t, False)

    def _update_momentum_each(self, states: list[dict[str, Any]], grads: list[torch.Tensor],
                              gscale: torch.Tensor | None, beta_t: torch.Tensor | None,
                              compressed: bool) -> list[torch.Tensor]:
        """One tensor at a time: compressed momentum, and factored tensors.

        Compressed momentum is decoded to the gradient's dtype, updated and
        stored back; the decoded tensor is returned for the rest of the step
        and then dropped (temporary, never state). A factored tensor's
        gradient is scaled by its row factors into one temporary and by its
        column factors inside the fused multiply-add, so no second temporary
        is built.
        """
        beta = self.momentum_beta
        out = []
        for i, (state, g) in enumerate(zip(states, grads, strict=True)):
            stored = state["momentum"]
            if not compressed:
                work = stored
            elif self.state_dtype == "int8":
                work = stored.to(g.dtype).mul_(state["scale"].to(g.dtype))
            else:
                work = stored.to(g.dtype)
            work.mul_(beta if beta_t is None else beta_t[i].to(work.dtype))
            weight = (1.0 - beta) if gscale is None else (1.0 - beta) * gscale[i]
            if "rs" in state:
                rows = (state["rs"] * weight).to(g.dtype)
                scaled = g.reshape(g.shape[0], -1) * rows[:, None]
                work.view(g.shape[0], -1).addcmul_(scaled, state["cs"].to(g.dtype)[None, :])
            elif gscale is None:
                work.add_(g, alpha=1.0 - beta)
            else:
                work.addcmul_(g, gscale[i].to(g.dtype), value=1.0 - beta)
            if compressed and self.state_dtype == "int8":
                scale = work.abs().amax().float().clamp_min(1e-30) / 127.0
                stored.copy_((work / scale.to(g.dtype)).round_().clamp_(-127, 127))
                state["scale"].copy_(scale)
            elif compressed:
                stored.copy_(work)
            out.append(work)
        return out

    def _measure(self, bucket: _Bucket, grads: list[torch.Tensor], moms: list[torch.Tensor],
                 gnorm: torch.Tensor) -> None:
        """Refresh trust, stability and pressure for this bucket. Device only."""
        eps = self.eps
        mnorm = self._norms(moms)
        if self.trust_ratio:
            pnorm = self._norms(bucket.params)
            trust = (pnorm / mnorm.clamp_min(eps)).clamp(*self.trust_clip)
            # A factored tensor's step is already normalised by its statistics;
            # a trust ratio on top of that measured worse, so it gets none.
            bucket.trust = torch.where(bucket.unfactored > 0, trust, torch.ones_like(trust))
        else:
            bucket.trust = torch.ones_like(mnorm)
        if self.stability:
            dots = torch.stack([torch.dot(g.reshape(-1).float(), m.reshape(-1).float())
                                for g, m in zip(grads, moms, strict=True)])
            cos = (dots / (gnorm * mnorm).clamp_min(eps)).clamp(-1.0, 1.0)
            cos = torch.where(torch.isfinite(cos), cos, torch.zeros_like(cos))
            for name, beta in zip(("cos_fast", "cos_med", "cos_slow"), self.stability_betas,
                                  strict=True):
                ema = getattr(bucket, name)
                ema.mul_(beta).add_(cos, alpha=1.0 - beta)
            wf, wm, ws = self.stability_weights
            bucket.score = (wf * bucket.cos_fast + wm * bucket.cos_med
                            + ws * bucket.cos_slow).clamp(-1.0, 1.0)
        if self.adaptive_pressure:
            pressure = 1.0 + self.stability_gain * bucket.score
            bucket.pressure = (pressure * self._loss_factor).clamp(*self.pressure_clip)
        else:
            bucket.pressure = torch.ones_like(mnorm)
        bucket.coef = bucket.trust * bucket.pressure

    def _step_bucket(self, bucket: _Bucket, measure: bool,
                     raw_gnorm: torch.Tensor | None) -> list[torch.Tensor]:
        """Returns the device scalars this bucket contributes to the host read.

        *raw_gnorm* is the per-tensor norm of the raw gradients when the step
        already measured it (for the clip and the non-finite check), so it is
        not measured twice.
        """
        group = bucket.group
        lr, wd = group["lr"], group["weight_decay"]
        params = bucket.params
        grads = self._prepare_grads(bucket)
        eps = self.eps

        need_gnorm = (self.clip_norm is not None
                      or (self.adaptive_pressure and self.spike_power > 0) or measure)
        changed = any(g is not p.grad for g, p in zip(grads, params, strict=True))
        if not need_gnorm:
            gnorm = None
        elif raw_gnorm is not None and not changed:
            gnorm = raw_gnorm
        else:
            gnorm = self._norms(grads)

        gscale = None
        if gnorm is not None and (self.clip_norm is not None
                                  or (self.adaptive_pressure and self.spike_power > 0)):
            gscale = torch.ones_like(gnorm)
            if self.adaptive_pressure and self.spike_power > 0:
                seeded = torch.where(bucket.gnorm_ema > 0, bucket.gnorm_ema, gnorm)
                ratio = (gnorm / seeded.clamp_min(eps)).clamp_min(1.0)
                gscale = gscale * ratio.pow(-self.spike_power)
                bucket.gnorm_ema = torch.where(
                    torch.isfinite(gnorm), 0.9 * seeded + 0.1 * gnorm, bucket.gnorm_ema)
            if self.clip_norm is not None and self._clip_total is not None:
                gscale = gscale * (self.clip_norm / self._clip_total.clamp_min(eps)).clamp(max=1.0)

        beta_t = None
        if self.stability and self.momentum_damping > 0:
            osc = (-bucket.score).clamp_min(0.0)
            beta_t = self.momentum_beta * (1.0 - self.momentum_damping * osc)

        if bucket.has_factored and (measure or self._global_step == 0):
            for p, g in zip(params, grads, strict=True):
                state = self.state[p]
                if "row" in state:
                    self._refresh_factors(g, state)
        moms = self._update_momentum(bucket, grads, gscale, beta_t)
        if measure:
            self._measure(bucket, grads, moms, gnorm if gnorm is not None else self._norms(grads))

        # Decoupled weight decay.
        if wd != 0.0:
            factor = 1.0 - lr * wd
            if self.lazy_weight_decay:
                bucket.decay_acc *= factor
                if measure:
                    factor, bucket.decay_acc = bucket.decay_acc, 1.0
                else:
                    factor = 1.0
            if factor != 1.0:
                if self._foreach_ok(params):
                    torch._foreach_mul_(params, factor)
                else:
                    for p in params:
                        p.mul_(factor)

        # Bias correction folded into the step size, as V6 does.
        correction = 1.0 - self.momentum_beta ** (self._global_step + 1)
        eff_lr = lr / max(correction, eps)
        plain = not self.trust_ratio and not self.adaptive_pressure
        if plain:
            if self._foreach_ok(params) and moms[0].dtype == params[0].dtype:
                torch._foreach_add_(params, moms, alpha=-eff_lr)
            else:
                for p, m in zip(params, moms, strict=True):
                    p.add_(m.to(p.dtype), alpha=-eff_lr)
        else:
            coefs = list(bucket.coef.to(params[0].dtype).unbind(0))
            if self._foreach_ok(params) and moms[0].dtype == params[0].dtype:
                torch._foreach_addcmul_(params, moms, coefs, value=-eff_lr)
            else:
                for p, m, c in zip(params, moms, coefs, strict=True):
                    p.addcmul_(m.to(p.dtype), c, value=-eff_lr)

        if self.ema_decay is not None:
            emas = [self.state[p]["ema"] for p in params]
            if self._foreach_ok(emas):
                torch._foreach_lerp_(emas, params, 1.0 - self.ema_decay)
            else:
                for e, p in zip(emas, params, strict=True):
                    e.lerp_(p, 1.0 - self.ema_decay)

        return [bucket.score.mean(), bucket.pressure.mean(), bucket.trust.mean()]

    # -- the step ------------------------------------------------------------------

    def _apply_update(self, buckets: list[_Bucket], measure: bool,
                      raw_gnorms: list[torch.Tensor] | None) -> list[torch.Tensor]:
        """Hook: V7V routes this through a compiled function."""
        stats = [self._step_bucket(bucket, measure, raw_gnorms[i] if raw_gnorms else None)
                 for i, bucket in enumerate(buckets)]
        return [torch.stack([s[i] for s in stats]).mean() for i in range(3)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._apply_lr_schedule()
        self._accum_counter += 1
        if self._accum_counter < self.grad_accum_steps:
            return loss
        self._accum_counter = 0
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self)

        buckets = self._collect()
        if not buckets:
            if self.grad_scaler is not None:
                self.grad_scaler.update()
            return loss

        check = self.skip_on_nonfinite or self.grad_scaler is not None
        raw_gnorms: list[torch.Tensor] | None = None
        self._clip_total = None
        if check or self.clip_norm is not None:
            # One norm pass over the raw gradients serves the non-finite check,
            # the clip and (unless diffusion or factoring changes them) the
            # per-tensor statistics.
            raw_gnorms = [self._norms([p.grad for p in b.params])  # type: ignore[misc]
                          for b in buckets]
            self._clip_total = torch.stack([n.square().sum() for n in raw_gnorms]).sum().sqrt()
        if check and self._clip_total is not None:
            # The step's one host read, before anything is modified.
            if not math.isfinite(float(self._clip_total)):
                self._skipped_steps += 1
                if self.grad_scaler is not None:
                    self.grad_scaler.update()
                return loss

        measure = self._global_step % self.stat_interval == 0
        score, pressure, trust = self._apply_update(buckets, measure, raw_gnorms)

        if check and measure and self._clip_total is not None:
            # Diagnostics, and the oscillation that gates diffusion: read on
            # the steps that measured (in between nothing here has changed),
            # and only where a host read is already being paid for.
            host = torch.stack([score, pressure, trust, self._clip_total.float()]).tolist()
            self.last_diagnostics = {"stability_score": host[0], "pressure": host[1],
                                     "trust_ratio": host[2], "grad_norm": host[3]}
            self._osc_host = min(1.0, 4.0 * max(0.0, -host[0]))

        if self.grad_scaler is not None:
            self.grad_scaler.update()
        self._global_step += 1
        return loss

    # -- memory, diagnostics, checkpoints --------------------------------------------

    def state_bytes(self) -> int:
        """Bytes of persistent optimizer state, counted from the tensors themselves."""
        total = 0
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        for bucket in self._buckets:
            for name in _Bucket.VECTORS:
                vector = getattr(bucket, name)
                total += vector.numel() * vector.element_size()
        return total

    def ema_parameters(self) -> list[torch.Tensor]:
        """The EMA shadow, in parameter order (empty without ``ema_decay``)."""
        return [self.state[p]["ema"] for group in self.param_groups for p in group["params"]
                if "ema" in self.state.get(p, {})]

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({
            "kind": type(self).__name__, "variant": self.variant, "lineage": self.lineage,
            "preset": self.preset, "v7_format": V7_FORMAT,
            "state_dtype": self.state_dtype, "state_dtype_requested": self.state_dtype_requested,
            "state_dtype_reason": self.state_dtype_reason,
            "state_bytes": self.state_bytes(), "parameters": self.n_params,
            "state_bytes_per_parameter": round(self.state_bytes() / max(1, self.n_params), 4),
            "momentum_beta": self.momentum_beta, "stat_interval": self.stat_interval,
            "stability": self.stability, "adaptive_pressure": self.adaptive_pressure,
            "trust_ratio": self.trust_ratio, "trust_clip": self.trust_clip,
            "diffusion": self.diffusion, "diffusion_factor": self.diffusion_factor,
            "factored": self.factored, "ema": self.ema_decay is not None,
            "grad_clip": self.clip_norm, "grad_accum_steps": self.grad_accum_steps,
            "grad_scaler": self.grad_scaler is not None,
            "skip_on_nonfinite": self.skip_on_nonfinite,
            "lazy_weight_decay": self.lazy_weight_decay, "foreach": self.foreach,
            "host_reads_per_step": 1 if (self.skip_on_nonfinite or self.grad_scaler) else 0,
            "skipped_steps": self._skipped_steps, "loss_factor": round(self._loss_factor, 4),
            "kernel": self._kernel_name(),
            **self.last_diagnostics,
        })
        return base

    def _kernel_name(self) -> str:
        return "foreach" if self.foreach else "per-tensor"

    def state_dict(self) -> dict[str, Any]:
        out = super().state_dict()
        out["v7"] = {
            "format": V7_FORMAT, "variant": self.variant, "preset": self.preset,
            "state_dtype": self.state_dtype, "global_step": self._global_step,
            "loss": [self._loss_fast, self._loss_slow, self._loss_factor],
            "oscillation": self._osc_host, "skipped_steps": self._skipped_steps,
            "buckets": [{"key": list(b.key), "n": len(b.params), "decay_acc": b.decay_acc,
                         **{name: getattr(b, name).detach().cpu() for name in _Bucket.VECTORS}}
                        for b in self._buckets],
        }
        return out

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a V7 checkpoint, converting the momentum if its dtype differs.

        A checkpoint from another optimizer is refused: its state means
        something else, and reading it as V7's would train from garbage.
        """
        meta = state_dict.get("v7")
        if not isinstance(meta, dict) or "format" not in meta:
            raise ValueError("this is not a Pressure Cooker V7 checkpoint (no 'v7' entry). "
                             "State from V5, V5+, V5S, V6 or AdamW cannot be read as V7's; "
                             "start V7's state fresh.")
        if int(meta["format"]) > V7_FORMAT:
            raise ValueError(f"this checkpoint is V7 format {meta['format']}; this HyperNix "
                             f"reads up to {V7_FORMAT}")
        plain = {k: v for k, v in state_dict.items() if k != "v7"}
        super().load_state_dict(plain)
        saved = str(meta.get("state_dtype", "fp32"))
        self._convert_loaded_state(saved)
        self._global_step = int(meta.get("global_step", 0))
        fast, slow, factor = meta.get("loss", [None, None, 1.0])
        self._loss_fast, self._loss_slow, self._loss_factor = fast, slow, float(factor)
        self._osc_host = float(meta.get("oscillation", 0.0))
        self._skipped_steps = int(meta.get("skipped_steps", 0))
        self._accum_counter = 0
        self._buckets, self._bucket_signature = [], ()
        self._pending_bucket_state = {tuple(b["key"]): b for b in meta.get("buckets", [])}

    def _convert_loaded_state(self, saved: str) -> None:
        """Bring loaded momentum to this optimizer's ``state_dtype``."""
        lossy = STATE_DTYPES.index(self.state_dtype) > STATE_DTYPES.index(saved) \
            if saved in STATE_DTYPES else False
        if saved != self.state_dtype:
            message = (f"V7 checkpoint momentum is {saved}; this optimizer stores "
                       f"{self.state_dtype}. Converting.")
            if lossy:
                message += (" The conversion loses precision: the first steps may differ "
                            "from an uninterrupted run.")
            warnings.warn(message, stacklevel=3)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "momentum" not in state:
                    continue
                stored = state["momentum"]
                if saved == "int8":
                    value = stored.float() * state.get("scale", torch.ones(())).float()
                else:
                    value = stored.float()
                value = value.to(p.device)
                if self.state_dtype == "int8":
                    scale = value.abs().amax().clamp_min(1e-30) / 127.0
                    state["momentum"] = (value / scale).round_().clamp_(-127, 127).to(torch.int8)
                    state["scale"] = scale.float()
                else:
                    state["momentum"] = value.to(self._momentum_dtype(p))
                    state.pop("scale", None)
                # Optional state follows this optimizer's configuration, not the file's.
                if self._factorable(p):
                    if "row" not in state:
                        state["row"] = torch.zeros(p.shape[0], device=p.device)
                        state["col"] = torch.zeros(p.numel() // p.shape[0], device=p.device)
                else:
                    state.pop("row", None)
                    state.pop("col", None)
                if self.ema_decay is not None and "ema" not in state:
                    state["ema"] = p.detach().clone()
                elif self.ema_decay is None:
                    state.pop("ema", None)

    def _restore_bucket_state(self) -> None:
        pending = self._pending_bucket_state
        if not pending:
            return
        for bucket in self._buckets:
            saved = pending.get(bucket.key)
            if saved is None or int(saved.get("n", -1)) != len(bucket.params):
                continue                # a different set of tensors: start its statistics fresh
            device = bucket.params[0].device
            for name in _Bucket.VECTORS:
                if name in saved:
                    setattr(bucket, name, saved[name].to(device).float().clone())
            bucket.decay_acc = float(saved.get("decay_acc", 1.0))
        self._pending_bucket_state = None


#: The balanced default under its family name.
V7Balanced = PressureCookerV7


class PressureCookerV7Mini(PressureCookerV7):
    """V7Mini: for models of more than 39M and fewer than 750M parameters.

    In that range a step is many medium-sized tensors, and what the
    optimizer adds is passes over them and per-step bookkeeping. Mini is
    the V7 update with the cheapest bookkeeping that keeps its behaviour:
    statistics every 8 steps, no diffusion, lazy weight decay. Whether that is faster on your model is measured by
    ``hnx optimizer benchmark``, not assumed.

    It works outside the band; :attr:`band_note` says its advantage is not
    established there.
    """

    variant = "mini"

    def _variant_defaults(self) -> dict[str, Any]:
        return dict(stat_interval=8, diffusion=False, diffusion_factor=0.0, factored=True,
                    lazy_weight_decay=True)

    @property
    def in_band(self) -> bool:
        return MINI_MIN_PARAMS < self.n_params < MINI_MAX_PARAMS

    @property
    def band_note(self) -> str:
        if self.in_band:
            return ""
        return (f"V7Mini is designed for more than {MINI_MIN_PARAMS:,} and fewer than "
                f"{MINI_MAX_PARAMS:,} trainable parameters; this model has {self.n_params:,}, "
                "where its advantage is not established")

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["in_band"] = self.in_band
        base["band_note"] = self.band_note
        return base


class PressureCookerV7Q(PressureCookerV7):
    """V7Q: throughput first.

    The fewest passes the V7 update can be done in: a fused lerp for the
    momentum, statistics (the trust ratio) every 16 steps, lazy weight
    decay, fp32 state, and by default no gradient clipping and no
    non-finite check -- so a step is three passes and no host read.

    What that gives up, plainly: no stability tracking, no adaptive
    pressure, no spike damping, and a bad batch is not caught (turn
    ``skip_on_nonfinite`` or ``grad_clip`` back on and pay one norm pass
    for it). The trust ratio is the only adaptivity, and it is up to 16
    steps stale. Expect to tune the learning rate more than with Balanced.
    """

    variant = "q"

    def _default_preset(self) -> str:
        return "fast"

    def _variant_defaults(self) -> dict[str, Any]:
        return dict(state_dtype="fp32", stat_interval=16, stability=False,
                    adaptive_pressure=False, diffusion=False, diffusion_factor=0.0,
                    factored=False, skip_on_nonfinite=False, lazy_weight_decay=True)

    def __init__(self, params: Any, *, grad_clip: float | None = None, **kwargs: Any) -> None:
        super().__init__(params, grad_clip=grad_clip, **kwargs)


class PressureCookerV7S(PressureCookerV7):
    """V7S: the V5S lineage on the V7 engine.

    Kept from V5S, with the same meaning:

    * **int8 momentum** with a per-tensor scale;
    * **factored curvature**: row and column statistics for matrices, no
      full second-moment tensor;
    * **three-timescale cosine oscillation tracking**, with V5S's band
      weights (0.45, 0.35, 0.20) and decays (0.8, 0.95, 0.999);
    * **pressure diffusion** with V5S's Gaussian three-tap kernel.

    Changed, and why:

    * V5S reads three cosines and a trust ratio to the host for every
      tensor on every step. V7S keeps them as device vectors and measures
      them every second step: one host read per step for the whole model.
    * V5S diffuses every gradient at a fixed strength. V7S scales the
      strength by the measured oscillation and skips the pass when there is
      none.
    * V5S's power update (``sign(g)|g|^p``), its Sophia-style element clip
      and its uint8 gradient-age freezing are **not** carried over. They
      need per-element work or an extra buffer that the V7 engine does not
      have, and V7's adaptive pressure and spike damping cover the same
      ground at tensor granularity. This is a difference in the
      mathematics, not only in the implementation: V7S is not a drop-in
      replacement that reproduces V5S's trajectory.
    """

    variant = "s"
    lineage = "V5S"

    def _default_preset(self) -> str:
        return "stable"

    def _variant_defaults(self) -> dict[str, Any]:
        return dict(state_dtype="int8", factored=True, diffusion=True, diffusion_factor=0.15,
                    stat_interval=2)

    def __init__(self, params: Any, *,
                 stability_betas: tuple[float, float, float] = (0.8, 0.95, 0.999),
                 stability_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
                 diffusion_kernel: float = 0.107, **kwargs: Any) -> None:
        super().__init__(params, stability_betas=stability_betas,
                         stability_weights=stability_weights,
                         diffusion_kernel=diffusion_kernel, **kwargs)


class PressureCookerV7M(PressureCookerV7):
    """V7M: the V5 and V5+ lineage on the V7 engine.

    The optimizer side of V5/V5+ is here: quantised momentum, factored
    curvature, a trust region (the per-tensor trust ratio and pressure),
    gradient-norm monitoring (``last_diagnostics``), and an optional EMA
    shadow of the weights.

    The *training* features of V5/V5+ -- quantisation-aware training,
    multi-token prediction, model preparation, sensitivity analysis, mixed
    precision per layer -- change the model or the loss, so they are not
    part of ``step()`` and nothing turns them on by default. They are
    explicit calls in :mod:`hypernix.optimizers.v7m_features`; a V7M that
    never calls them pays nothing for them.

    A V5 or V5+ checkpoint is not a V7M checkpoint and is refused by
    :meth:`load_state_dict`.
    """

    variant = "m"
    lineage = "V5/V5+"

    def _default_preset(self) -> str:
        return "memory"

    def _variant_defaults(self) -> dict[str, Any]:
        return dict(state_dtype="int8", factored=True, stat_interval=4)


V7Mini = PressureCookerV7Mini
V7Q = PressureCookerV7Q
V7S = PressureCookerV7S
V7M = PressureCookerV7M
