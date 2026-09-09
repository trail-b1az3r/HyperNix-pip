"""hypernix.system.pascal — auto-tuning for Pascal, sm_61.

The GTX 1080, 1080 Ti, P40, P4 and P100 are still the cheapest way to get
a lot of VRAM, and they are all wrong in the same specific ways. This
module encodes those ways so the rest of HyperNix does not have to
rediscover them per run.

What is actually different about sm_61
--------------------------------------
**No tensor cores.** Volta introduced them. On Pascal, FP16 arithmetic
runs on the same FP32 units, so half precision buys memory bandwidth and
capacity — not throughput. Anything that promises a Pascal speedup from
FP16 compute is measuring something else.

**FP16 arithmetic is a quarter speed on the consumer cards.** GP104
(1080), GP102 (1080 Ti, P40) and GP106 have a 1:64 FP16:FP32 rate,
because NVIDIA fused off the fast path. GP100 (P100) is the exception at
2:1 — it is the only Pascal chip where FP16 compute is genuinely faster.
:data:`PASCAL_GPUS` records which is which, and the tuner uses it: on a
1080 the correct answer is "store in FP16, compute in FP32", and on a
P100 it is not.

**No BF16.** BF16's whole appeal is FP32's exponent range at half the
width, which is exactly what prevents the overflow that makes FP16
training produce NaN. Pascal does not have it. That absence is why this
module exists at all: everything in :class:`FP16Guard` is working around
not having the format that would make the problem go away.

**Memory bandwidth is the binding constraint.** A 1080 Ti has 11.3
TFLOPS FP32 against 484 GB/s. That ratio makes it arithmetic-rich and
bandwidth-poor relative to a modern card, so the tuning that helps is
about moving less data, not doing less maths.

Nothing here requires torch. :func:`detect` reads what it can and returns
a description; the tuner is a pure function of that description, so it is
testable against every card in the table without owning one.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PascalGPU",
    "PASCAL_GPUS",
    "SM61",
    "GPUInfo",
    "detect",
    "identify",
    "is_pascal",
    "PascalTuning",
    "autotune",
    "FP16Guard",
    "KernelTuning",
    "kernel_tuning_for",
]

#: Pascal's consumer/datacentre compute capability. GP100 is 6.0; every
#: other Pascal chip is 6.1. Both are handled; the differences that
#: matter are in :data:`PASCAL_GPUS`, not in the capability number.
SM61 = (6, 1)
SM60 = (6, 0)


@dataclass(frozen=True)
class PascalGPU:
    """One Pascal card, and the numbers that change the tuning."""

    name: str
    chip: str
    compute: tuple[int, int]
    vram_gb: float
    bandwidth_gb_s: float
    fp32_tflops: float
    #: FP16:FP32 arithmetic throughput ratio. 1/64 on the consumer chips,
    #: 2.0 on GP100. The single most misunderstood Pascal number.
    fp16_ratio: float
    ecc: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def fp16_is_fast(self) -> bool:
        """True only on GP100. Everywhere else FP16 is a storage format."""
        return self.fp16_ratio >= 1.0

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of bandwidth — how compute-rich the card is.

        A modern card sits far higher. A low number here is why the
        Pascal tuning is about moving less data rather than doing less
        arithmetic.
        """
        return self.fp32_tflops * 1e12 / (self.bandwidth_gb_s * 1e9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chip": self.chip,
            "compute": f"{self.compute[0]}.{self.compute[1]}",
            "vram_gb": self.vram_gb,
            "bandwidth_gb_s": self.bandwidth_gb_s,
            "fp32_tflops": self.fp32_tflops,
            "fp16_ratio": self.fp16_ratio,
            "fp16_is_fast": self.fp16_is_fast,
            "arithmetic_intensity": round(self.arithmetic_intensity, 2),
            "ecc": self.ecc,
        }


PASCAL_GPUS: dict[str, PascalGPU] = {
    g.name: g
    for g in (
        PascalGPU("GTX 1080", "GP104", SM61, 8.0, 320.0, 8.87, 1 / 64,
                  aliases=("geforce gtx 1080", "gtx1080", "1080")),
        PascalGPU("GTX 1080 Ti", "GP102", SM61, 11.0, 484.0, 11.34, 1 / 64,
                  aliases=("geforce gtx 1080 ti", "gtx1080ti", "1080ti", "1080 ti")),
        PascalGPU("GTX 1070", "GP104", SM61, 8.0, 256.0, 6.46, 1 / 64,
                  aliases=("geforce gtx 1070", "gtx1070", "1070")),
        PascalGPU("GTX 1060", "GP106", SM61, 6.0, 192.0, 4.38, 1 / 64,
                  aliases=("geforce gtx 1060", "gtx1060", "1060")),
        PascalGPU("Tesla P40", "GP102", SM61, 24.0, 346.0, 11.76, 1 / 64, ecc=True,
                  aliases=("p40", "tesla p40")),
        PascalGPU("Tesla P4", "GP104", SM61, 8.0, 192.0, 5.44, 1 / 64, ecc=True,
                  aliases=("p4", "tesla p4")),
        PascalGPU("Tesla P100", "GP100", SM60, 16.0, 732.0, 9.53, 2.0, ecc=True,
                  aliases=("p100", "tesla p100", "p100-pcie", "p100-sxm2")),
        PascalGPU("Quadro P6000", "GP102", SM61, 24.0, 432.0, 12.63, 1 / 64, ecc=True,
                  aliases=("p6000", "quadro p6000")),
    )
}

_GPU_ALIASES: dict[str, str] = {}
for _g in PASCAL_GPUS.values():
    _GPU_ALIASES[_g.name.lower()] = _g.name
    for _a in _g.aliases:
        _GPU_ALIASES[_a] = _g.name


def identify(name: str) -> PascalGPU | None:
    """Match a reported GPU name against the table, or ``None``.

    Matching is by normalised substring rather than equality because the
    reported name varies by driver, vendor and platform — "NVIDIA GeForce
    GTX 1080 Ti" and "GeForce GTX 1080 Ti" are the same card.
    """
    if not name:
        return None
    lowered = name.lower().replace("nvidia", "").replace("geforce", "").strip()
    if lowered in _GPU_ALIASES:
        return PASCAL_GPUS[_GPU_ALIASES[lowered]]
    # Longest alias first so "1080 ti" wins over "1080".
    for alias in sorted(_GPU_ALIASES, key=len, reverse=True):
        if alias in lowered:
            return PASCAL_GPUS[_GPU_ALIASES[alias]]
    return None


def is_pascal(compute: tuple[int, int] | None) -> bool:
    return compute is not None and compute[0] == 6


@dataclass
class GPUInfo:
    """What we could learn about the GPU actually present."""

    name: str = ""
    compute: tuple[int, int] | None = None
    vram_gb: float = 0.0
    driver: str = ""
    source: str = "none"          # "torch" | "nvidia-smi" | "none"
    matched: PascalGPU | None = None

    @property
    def is_pascal(self) -> bool:
        return is_pascal(self.compute)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compute": f"{self.compute[0]}.{self.compute[1]}" if self.compute else None,
            "vram_gb": round(self.vram_gb, 2),
            "driver": self.driver,
            "source": self.source,
            "is_pascal": self.is_pascal,
            "matched": self.matched.to_dict() if self.matched else None,
        }


def detect(index: int = 0) -> GPUInfo:
    """Describe the GPU at *index*. Never raises.

    torch first because it reports the compute capability directly;
    ``nvidia-smi`` second because it works without torch installed, which
    is the case on a machine that is only running the quantiser.
    """
    info = GPUInfo()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > index:
            props = torch.cuda.get_device_properties(index)
            info.name = props.name
            info.compute = (props.major, props.minor)
            info.vram_gb = props.total_memory / (1024 ** 3)
            info.source = "torch"
            info.matched = identify(info.name)
            return info
    except Exception:  # noqa: BLE001 - detection is best-effort by contract
        logger.debug("pascal.detect: torch path unavailable", exc_info=True)

    # The vendor-tool path goes through hypernix.system.gpus rather than
    # shelling out here. It was a second, slightly different nvidia-smi
    # parser -- and the two disagreed about "[N/A]", which this one read
    # as 0.0 VRAM.
    #
    # This module is about *Pascal*, so the answer is still NVIDIA-shaped:
    # `compute` is a CUDA compute capability and `identify` matches NVIDIA
    # product names. An AMD card detected here gets its name, VRAM and
    # driver filled in and no compute capability, which is honest -- gfx
    # targets are not (major, minor) and pretending otherwise would make
    # the FP16 guard below reason about a number that means something
    # else.
    try:
        from . import gpus as _gpus

        cards = _gpus.detect()
        card = next((c for c in cards if c.index == index), None)
        if card is not None:
            info.name = card.name
            info.vram_gb = (
                card.memory_total_mb / 1024 if card.memory_total_mb else 0.0
            )
            info.driver = card.driver
            info.source = f"{card.vendor.value}-smi"
            if card.vendor is _gpus.Vendor.NVIDIA and "." in card.compute:
                major, minor = card.compute.split(".", 1)
                try:
                    info.compute = (int(major), int(minor))
                except ValueError:
                    logger.debug("pascal.detect: odd compute_cap %r", card.compute)
            info.matched = identify(info.name)
            return info
    except Exception:  # noqa: BLE001 - detection is best-effort by contract
        logger.debug("pascal.detect: gpus path failed", exc_info=True)
    return info


# ---------------------------------------------------------------------------
# The FP16 guard
# ---------------------------------------------------------------------------


@dataclass
class FP16Guard:
    """Keeps FP16 training on Pascal from producing NaN.

    FP16's exponent range tops out at 65504. On a card with BF16 you
    would simply use BF16 and stop thinking about it. Pascal has no BF16,
    so the options are FP32 everywhere (half the effective VRAM) or FP16
    storage with the overflow handled explicitly. This is the second one.

    Four mechanisms, each earning its place:

    * **Loss scaling** multiplies the loss before backward so small
      gradients survive FP16's *lower* limit, then divides them out. The
      scale is dynamic: too high overflows, too low underflows, and the
      right value changes during training.
    * **Backoff on overflow.** When a gradient comes back non-finite the
      step is skipped and the scale halved. Skipping is not optional — a
      NaN that reaches the weights is permanent.
    * **Growth after a quiet period.** After ``growth_interval`` clean
      steps the scale doubles, so an early backoff does not leave the
      run underflowing for the next ten thousand steps.
    * **A hard FP32 fallback.** After ``max_consecutive_overflows``
      backoffs in a row, the guard gives up on FP16 and says so. A run
      that overflows constantly is not being saved by loss scaling; it is
      being slowed down by it, and the honest answer is FP32.

    All of it is pure arithmetic over a float and two counters, so it is
    testable without a GPU — which is the point, because the failure
    being prevented takes an hour of training to reproduce otherwise.
    """

    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    min_scale: float = 1.0
    max_scale: float = 2.0 ** 24
    max_consecutive_overflows: int = 20

    scale: float = field(init=False)
    good_steps: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)
    overflow_steps: int = field(default=0, init=False)
    consecutive_overflows: int = field(default=0, init=False)
    fell_back_to_fp32: bool = field(default=False, init=False)
    fallback_reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not 0 < self.backoff_factor < 1:
            raise ValueError("backoff_factor must be between 0 and 1")
        if self.growth_factor <= 1:
            raise ValueError("growth_factor must be greater than 1")
        self.scale = float(self.init_scale)

    def step(self, *, overflow: bool) -> bool:
        """Record one optimiser step. Returns True if it should be applied.

        ``overflow`` is whatever the caller uses to decide the gradients
        were not finite — ``torch.isfinite(...).all()`` inverted, or a
        found-inf tensor from a grad scaler. The guard does not look at
        tensors itself, which is what keeps it torch-free.
        """
        self.total_steps += 1
        if self.fell_back_to_fp32:
            # Once in FP32 there is nothing to scale; every step applies.
            self.good_steps += 1
            return True

        if overflow:
            self.overflow_steps += 1
            self.consecutive_overflows += 1
            self.good_steps = 0
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            if self.consecutive_overflows >= self.max_consecutive_overflows:
                self.fell_back_to_fp32 = True
                self.fallback_reason = (
                    f"{self.consecutive_overflows} consecutive FP16 overflows. Loss scaling "
                    "cannot rescue a run that overflows every step — the dynamic range simply "
                    "is not there. Falling back to FP32 compute. On Pascal this is the correct "
                    "answer: there is no BF16 to switch to, and FP16 arithmetic is not faster "
                    "on this chip anyway."
                )
                logger.warning("pascal.FP16Guard: %s", self.fallback_reason)
            return False

        self.consecutive_overflows = 0
        self.good_steps += 1
        if self.good_steps >= self.growth_interval:
            self.scale = min(self.max_scale, self.scale * self.growth_factor)
            self.good_steps = 0
        return True

    @property
    def overflow_rate(self) -> float:
        return self.overflow_steps / self.total_steps if self.total_steps else 0.0

    @property
    def healthy(self) -> bool:
        """A run overflowing more than 5% of steps is not really working."""
        return not self.fell_back_to_fp32 and self.overflow_rate <= 0.05

    def state(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "total_steps": self.total_steps,
            "overflow_steps": self.overflow_steps,
            "overflow_rate": round(self.overflow_rate, 5),
            "consecutive_overflows": self.consecutive_overflows,
            "fell_back_to_fp32": self.fell_back_to_fp32,
            "fallback_reason": self.fallback_reason,
            "healthy": self.healthy,
        }


# ---------------------------------------------------------------------------
# Kernel tuning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelTuning:
    """Launch parameters for a Pascal chip.

    Pascal SMs hold 2048 resident threads and 96 KB of shared memory per
    SM (64 KB on GP100). A block of 256 threads divides both cleanly,
    which is why it is the default here rather than the 1024 that looks
    better on paper: 1024-thread blocks quantise occupancy badly on a
    chip with 2048 slots and make register pressure the binding limit.
    """

    block_size: int
    max_registers: int
    shared_memory_kb: int
    unroll: int
    vector_width: int
    l2_cache_kb: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "max_registers": self.max_registers,
            "shared_memory_kb": self.shared_memory_kb,
            "unroll": self.unroll,
            "vector_width": self.vector_width,
            "l2_cache_kb": self.l2_cache_kb,
            "rationale": self.rationale,
        }

    def nvcc_flags(self) -> list[str]:
        """Flags for compiling a custom kernel for this chip."""
        arch = "sm_60" if self.shared_memory_kb == 64 else "sm_61"
        return [
            f"-arch={arch}",
            f"-maxrregcount={self.max_registers}",
            "-use_fast_math",
            "--ptxas-options=-v",
        ]


def kernel_tuning_for(gpu: PascalGPU | None) -> KernelTuning:
    """Launch parameters for *gpu*, or a safe Pascal default."""
    if gpu is not None and gpu.chip == "GP100":
        return KernelTuning(
            block_size=256, max_registers=64, shared_memory_kb=64, unroll=4,
            vector_width=2, l2_cache_kb=4096,
            rationale=(
                "GP100: 64 KB shared memory per SM and HBM2 at 732 GB/s. Bandwidth is "
                "not the constraint here the way it is on GDDR5X parts, so the unroll "
                "stays moderate and registers are capped to keep two blocks resident."
            ),
        )
    if gpu is not None and gpu.chip == "GP102":
        return KernelTuning(
            block_size=256, max_registers=72, shared_memory_kb=96, unroll=8,
            vector_width=4, l2_cache_kb=3072,
            rationale=(
                "GP102 (1080 Ti / P40 / P6000): the widest Pascal memory bus at 384-bit. "
                "float4 loads and an 8x unroll keep enough requests in flight to saturate "
                "it; 72 registers still allows 2 blocks/SM at 256 threads."
            ),
        )
    return KernelTuning(
        block_size=256, max_registers=64, shared_memory_kb=96, unroll=4,
        vector_width=4, l2_cache_kb=2048,
        rationale=(
            "GP104/GP106 default: 256-bit or narrower bus. float4 loads still pay, but a "
            "shorter unroll keeps the register file free enough for 2-3 resident blocks."
        ),
    )


# ---------------------------------------------------------------------------
# The tuner
# ---------------------------------------------------------------------------


@dataclass
class PascalTuning:
    """A complete tuning recommendation, with reasons attached.

    Every field has a matching entry in :attr:`reasons`. A tuner that
    hands back numbers without saying why produces cargo cult; the
    reasons are what let someone disagree with it on purpose.
    """

    gpu: GPUInfo
    compute_dtype: str
    storage_dtype: str
    use_amp: bool
    loss_scaling: bool
    micro_batch: int
    gradient_accumulation: int
    gradient_checkpointing: bool
    optimizer: str
    six_bit_mode: str
    attention: str
    tf32: bool
    channels_last: bool
    allowed_quant_formats: list[str]
    kernel: KernelTuning
    reasons: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.gradient_accumulation

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu.to_dict(),
            "compute_dtype": self.compute_dtype,
            "storage_dtype": self.storage_dtype,
            "use_amp": self.use_amp,
            "loss_scaling": self.loss_scaling,
            "micro_batch": self.micro_batch,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch": self.effective_batch,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optimizer": self.optimizer,
            "six_bit_mode": self.six_bit_mode,
            "attention": self.attention,
            "tf32": self.tf32,
            "channels_last": self.channels_last,
            "allowed_quant_formats": list(self.allowed_quant_formats),
            "kernel": self.kernel.to_dict(),
            "reasons": dict(self.reasons),
            "warnings": list(self.warnings),
        }


def autotune(
    gpu: GPUInfo | None = None,
    *,
    parameters: int = 0,
    sequence_length: int = 2048,
    target_effective_batch: int = 32,
) -> PascalTuning:
    """Produce a tuning for the detected (or supplied) GPU.

    A pure function of *gpu* so it can be tested against every card in
    :data:`PASCAL_GPUS` without owning one.
    """
    info = gpu if gpu is not None else detect()
    card = info.matched
    reasons: dict[str, str] = {}
    warnings: list[str] = []

    if not info.is_pascal:
        warnings.append(
            "This is not a Pascal GPU. The tuning below is Pascal-shaped and will be "
            "conservative on newer hardware — use BF16 and the stock settings instead."
        )

    # -- dtypes ---------------------------------------------------------
    fp16_fast = bool(card and card.fp16_is_fast)
    if fp16_fast:
        compute_dtype, storage_dtype, use_amp = "fp16", "fp16", True
        reasons["compute_dtype"] = (
            f"{card.chip} is the one Pascal chip with a fast FP16 path "
            f"({card.fp16_ratio:.0f}:1 versus FP32), so FP16 compute is a real speedup here."
        )
    else:
        compute_dtype, storage_dtype, use_amp = "fp32", "fp16", True
        reasons["compute_dtype"] = (
            "FP16 arithmetic runs at 1:64 of FP32 on this chip — NVIDIA fused off the fast "
            "path. FP16 is worth using as a *storage* format for the bandwidth and capacity, "
            "but computing in it would be 64x slower, so compute stays FP32."
        )
    reasons["storage_dtype"] = (
        "Weights and activations stored in FP16 halve the bytes moved, and bandwidth is the "
        "binding constraint on this card."
    )
    reasons["use_amp"] = (
        "Mixed precision with an FP32 master copy: the accumulation that would overflow in "
        "FP16 happens in FP32."
    )

    loss_scaling = storage_dtype == "fp16" or compute_dtype == "fp16"
    reasons["loss_scaling"] = (
        "Pascal has no BF16. FP16's exponent range is what produces training NaNs, and "
        "dynamic loss scaling is the only mitigation available on this hardware. "
        "See FP16Guard."
    )

    # -- batch sizing ---------------------------------------------------
    vram = info.vram_gb or (card.vram_gb if card else 8.0)
    # Rough activation cost per sequence per micro-batch element, in GB.
    # Deliberately crude: the point is to land in the right order of
    # magnitude and let gradient accumulation carry the rest.
    per_sample_gb = max(0.05, (sequence_length / 2048) * 0.35)
    weight_gb = (parameters * 2 / 1e9) if parameters else vram * 0.25
    optimizer_gb = weight_gb * 1.7          # PressureCookerV5 budget
    headroom = max(1.0, vram - weight_gb - optimizer_gb - 1.0)
    micro_batch = max(1, int(headroom / per_sample_gb))
    micro_batch = min(micro_batch, 16)      # beyond this, bandwidth not capacity binds
    gradient_accumulation = max(1, math.ceil(target_effective_batch / micro_batch))
    reasons["micro_batch"] = (
        f"{vram:.0f} GB VRAM minus ~{weight_gb:.1f} GB weights and ~{optimizer_gb:.1f} GB "
        f"optimiser state leaves ~{headroom:.1f} GB for activations, at roughly "
        f"{per_sample_gb:.2f} GB per sample at sequence length {sequence_length}."
    )
    reasons["gradient_accumulation"] = (
        f"{micro_batch} x {gradient_accumulation} reaches an effective batch of "
        f"{micro_batch * gradient_accumulation}. Accumulation is free in memory and cheap in "
        "time here, which is the right trade on a capacity-limited card."
    )

    gradient_checkpointing = micro_batch <= 2 or (parameters and parameters > 3e9)
    reasons["gradient_checkpointing"] = (
        "On at this size: recomputing activations costs roughly 30% more time and saves far "
        "more memory than that, which is the correct trade when capacity is what is scarce."
        if gradient_checkpointing
        else "Off: there is enough headroom for a reasonable micro-batch without it."
    )

    # -- optimiser and packing -----------------------------------------
    optimizer = "Agedcookerv5" if info.is_pascal else "PressureCookerV5"
    reasons["optimizer"] = (
        "Agedcookerv5 is the Pascal-constrained variant of PressureCookerV5 — quantised "
        "momentum and factored curvature, sized for a card without tensor cores."
        if info.is_pascal
        else "PressureCookerV5: quantised momentum plus factored curvature, ~1.7x SGD memory."
    )
    six_bit_mode = "aligned"
    reasons["six_bit_mode"] = (
        "'aligned' rather than 'packed': the optimiser step on Pascal is bandwidth-bound, "
        "not compute-bound, but unpacking 4-values-in-3-bytes costs shifts on a chip with no "
        "spare integer throughput. One value per byte wastes 25% of the buffer and is faster "
        "here. Switch to 'packed' if VRAM is the binding limit rather than time."
    )

    attention = "sdpa-math"
    reasons["attention"] = (
        "FlashAttention needs sm_75 or newer. On Pascal the math fallback is the only correct "
        "path; anything claiming Flash on a 1080 is silently falling back anyway."
    )

    tf32 = False
    reasons["tf32"] = "TF32 is an Ampere tensor-core format. Pascal has neither."
    channels_last = False
    reasons["channels_last"] = (
        "channels_last pays off with tensor cores and cuDNN kernels that want NHWC. On Pascal "
        "it usually costs a transpose for nothing."
    )

    # -- what this card can actually run --------------------------------
    from ..quant.formats import list_formats

    allowed = [f.name for f in list_formats(compute_capability=info.compute or SM61)]
    reasons["allowed_quant_formats"] = (
        "Filtered by compute capability: FP8 needs sm_89 and FP4 needs sm_100, so neither is "
        "offered here. These are missing instructions, not slow paths."
    )

    if card and card.ecc:
        warnings.append(
            f"{card.name} has ECC memory, which costs roughly 6% of both bandwidth and "
            "usable VRAM when enabled. `nvidia-smi -e 0` disables it if this is a workstation "
            "rather than a server."
        )
    if card and card.vram_gb <= 6:
        warnings.append(
            f"{card.vram_gb:.0f} GB is tight for training anything past ~1B parameters. "
            "Expect to rely on gradient checkpointing and a micro-batch of 1."
        )
    if not card and info.is_pascal:
        warnings.append(
            f"Pascal-class GPU {info.name!r} is not in the tuning table; using GP104 defaults. "
            "The dtype and attention choices are right for any sm_61 part."
        )

    return PascalTuning(
        gpu=info,
        compute_dtype=compute_dtype,
        storage_dtype=storage_dtype,
        use_amp=use_amp,
        loss_scaling=loss_scaling,
        micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation,
        gradient_checkpointing=bool(gradient_checkpointing),
        optimizer=optimizer,
        six_bit_mode=six_bit_mode,
        attention=attention,
        tf32=tf32,
        channels_last=channels_last,
        allowed_quant_formats=allowed,
        kernel=kernel_tuning_for(card),
        reasons=reasons,
        warnings=warnings,
    )
