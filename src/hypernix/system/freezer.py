"""freezer — VRAM manager for training and inference.

Three variants sit behind the same interface:

* :class:`OldFreezer`   — tuned for 8–10 GB cards.  Conservative batch
                          and context defaults, bf16 / fp16 preferred,
                          aggressive cache-empty on every step.
* :class:`NewFreezer`   — tuned for 11 GB+ cards.  Larger batches, keeps
                          activations on-device, fp32 by default.
* :class:`FlashFreezer` — wraps any freezer and adds OOM-safety.  On a
                          ``torch.cuda.OutOfMemoryError`` it empties the
                          cache, waits for free VRAM to climb back to a
                          threshold, and retries.  Configurable backoff
                          and optional "slow" mode that halves the batch
                          size every retry so progress continues even on
                          a contested card.

QAT Support (v0.70.5):
  * QAT-aware batch sizing (accounts for fake quantization memory)
  * QAT model preparation helpers
  * Per-bit-width VRAM multiplier profiles

Typical use::

    from hypernix import freezer

    fz = freezer.auto_freezer()            # picks Old or New by VRAM
    fz = freezer.flash_freezer(base=fz)    # add OOM-safety
    bs  = fz.suggest_batch_size(hint=8)
    fz.guard(lambda: model(batch))         # retries on OOM

    # With QAT
    bs = fz.suggest_qat_batch_size(bits=6, hint=8)
    fz.prepare_for_qat(model, bits=6)      # attach fake quant hooks

Everything degrades cleanly on CPU-only systems: ``suggest_batch_size``
returns the hint unchanged, ``guard`` just calls the function, and the
waits are no-ops.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

T = TypeVar("T")


# ---------------------------------------------------------------------------
# VRAM probing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VRAMBudget:
    """Snapshot of device memory at one point in time.

    All values are in bytes.  ``free`` and ``total`` are 0 on CPU-only
    systems so callers can do ``if budget.total:`` as a quick gate.
    """
    device: str
    total: int = 0
    free: int = 0

    @property
    def total_gb(self) -> float:
        return self.total / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return (self.total - self.free) / (1024 ** 3)


def probe_vram(device_index: int = 0) -> VRAMBudget:
    """Return current VRAM state for the given CUDA device, or a zeroed
    budget on CPU-only systems."""
    if not torch.cuda.is_available():
        return VRAMBudget(device="cpu")
    try:
        free, total = torch.cuda.mem_get_info(device_index)
    except (RuntimeError, AttributeError):
        # Older torch or driver without mem_get_info — fall back to
        # properties.total_memory and leave free=0 so callers treat the
        # device as "don't know".
        props = torch.cuda.get_device_properties(device_index)
        return VRAMBudget(device=f"cuda:{device_index}", total=props.total_memory, free=0)
    return VRAMBudget(device=f"cuda:{device_index}", total=total, free=free)


# ---------------------------------------------------------------------------
# Compute capability (Pascal / Ampere / Hopper / …) detection
# ---------------------------------------------------------------------------

#: Compute capability for Pascal consumer cards (GTX 1060/1070/1080/Ti, Titan X/Xp).
PASCAL_CC: tuple[int, int] = (6, 1)

#: Compute capabilities that do NOT have native bf16 — Volta (7,0), Turing
#: (7,5), Pascal (6,x) and older. Ampere (8,0+) introduced bf16. bf16 on
#: these is either unsupported or emulated in software.
_NO_NATIVE_BF16 = frozenset({(6, 0), (6, 1), (6, 2), (7, 0), (7, 5)})


def compute_capability(device_index: int = 0) -> tuple[int, int] | None:
    """Return ``(major, minor)`` for a CUDA device, or ``None`` on CPU-only hosts."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device_index)
    except (RuntimeError, AttributeError):
        return None


def is_pascal(device_index: int = 0) -> bool:
    """True if the device is a Pascal chip (sm_60, sm_61, sm_62)."""
    cc = compute_capability(device_index)
    return cc is not None and cc[0] == 6


def pascal_safe_dtype(device_index: int = 0) -> torch.dtype:
    """Pick a dtype that trains stably on the detected device.

    * ``torch.bfloat16`` — Ampere (sm_80) and newer.
    * ``torch.float16``  — Pascal / Volta / Turing (sm_6x / 7x).
    * ``torch.float32``  — CPU-only hosts (PyTorch CPU fp16 matmul is
                           emulated and overflows quickly in training).

    bf16 on Pascal is either unsupported or falls back to a slow software
    path; fp16 has native tensor-core-free fast-math on Pascal (sm_61).
    """
    cc = compute_capability(device_index)
    if cc is None:
        # No CUDA device — fp16 on CPU is unstable for training.
        return torch.float32
    if cc in _NO_NATIVE_BF16 or cc < (8, 0):
        return torch.float16
    if not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def pascal_mode_hints(device_index: int = 0) -> dict[str, object]:
    """Return a dict of recommended settings for Pascal (sm_61) GPUs.

    Nothing here is *required* — it's a cheat sheet for callers building
    their own training loop.  Keys:

    ``dtype``            fp16 (not bf16 — Pascal has no native bf16).
    ``use_sdpa``         False; the PyTorch 2 fused SDPA kernels assume
                         Ampere+ tensor cores.
    ``use_compile``      False; ``torch.compile`` breaks often on sm_61
                         due to Triton kernel assumptions.
    ``tf32``             False; Pascal has no TF32 support.
    ``matmul_precision`` "highest" — the TF32 knob is inert anyway.
    ``install_hint``     One-line pip command for the CUDA 11.8 wheel,
                         which is the last official PyTorch line still
                         compiled for sm_61.
    """
    return {
        "dtype": torch.float16,
        "use_sdpa": False,
        "use_compile": False,
        "tf32": False,
        "matmul_precision": "highest",
        "install_hint": (
            "pip install --index-url "
            "https://download.pytorch.org/whl/cu118 torch"
        ),
    }


# ---------------------------------------------------------------------------
# Base freezer
# ---------------------------------------------------------------------------

class Freezer:
    """Base VRAM manager.  Concrete subclasses set the tuning knobs."""

    #: Preferred torch dtype for forward/backward.
    preferred_dtype: torch.dtype = torch.float32
    #: Base batch size for a training step.
    base_batch_size: int = 2
    #: Base context length.
    base_context_length: int = 1024
    #: If True, call ``torch.cuda.empty_cache`` after every training step.
    empty_cache_each_step: bool = False
    #: Human-readable label used in logs.
    name: str = "Freezer"

    def budget(self) -> VRAMBudget:
        return probe_vram()

    def suggest_batch_size(self, hint: int | None = None) -> int:
        """Return a batch size appropriate for this freezer.

        If ``hint`` is given it's capped at ``self.base_batch_size`` for
        conservative freezers and returned unchanged otherwise.
        """
        if hint is None:
            return self.base_batch_size
        return min(hint, self.base_batch_size) if self._caps_hint else hint

    _caps_hint: bool = False  # Old caps; New doesn't.

    def suggest_context_length(self, hint: int | None = None) -> int:
        if hint is None:
            return self.base_context_length
        return min(hint, self.base_context_length) if self._caps_hint else hint

    def guard(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call ``fn(*args, **kwargs)``.  Base implementation is a plain call;
        :class:`FlashFreezer` overrides this with OOM-safe retry."""
        return fn(*args, **kwargs)

    def __repr__(self) -> str:
        b = self.budget()
        return (
            f"<{self.name} device={b.device} total={b.total_gb:.1f}GB "
            f"free={b.free_gb:.1f}GB bs={self.base_batch_size} "
            f"ctx={self.base_context_length}>"
        )

    # ------------------------------------------------------------------
    # QAT (Quantization-Aware Training) support
    # ------------------------------------------------------------------

    #: VRAM multiplier for QAT training at different bit widths.
    #: Fake quantization stores both original and quantized values,
    #: plus running statistics, increasing memory usage.
    _QAT_VRAM_MULTIPLIERS: dict[int, float] = {
        4: 1.15,   # Q4: minimal overhead
        5: 1.20,   # Q5: low overhead
        6: 1.25,   # Q6: moderate overhead
        8: 1.35,   # Q8: highest overhead (more precise stats)
    }

    def suggest_qat_batch_size(
        self,
        bits: int = 6,
        hint: int | None = None,
    ) -> int:
        """Suggest a QAT-aware batch size.

        QAT training uses more VRAM due to fake quantization buffers
        and running statistics. This method adjusts the batch size
        downward to account for the overhead.

        Args:
            bits: QAT bit width (4, 5, 6, or 8).
            hint: Preferred batch size (will be adjusted for QAT).

        Returns:
            QAT-safe batch size.
        """
        base_bs = self.suggest_batch_size(hint)
        multiplier = self._QAT_VRAM_MULTIPLIERS.get(bits, 1.25)
        # Reduce batch size proportionally to VRAM multiplier
        qat_bs = max(1, int(base_bs / multiplier))
        return qat_bs

    def prepare_for_qat(
        self,
        model: Any,
        bits: int = 6,
        per_layer: bool = True,
    ) -> None:
        """Prepare a model for QAT training in the freezer context.

        This is a lightweight wrapper that logs QAT preparation and
        adjusts freezer settings for QAT mode. The actual QAT hook
        attachment is handled by PressureCookerV5.attach_qat().

        Args:
            model: The model to prepare.
            bits: QAT bit width.
            per_layer: Use per-layer quantization scales.
        """
        self._qat_bits = bits
        self._qat_per_layer = per_layer
        # Adjust batch size for QAT if not already set
        if hasattr(self, "current_batch_size"):
            self.current_batch_size = self.suggest_qat_batch_size(
                bits, self.current_batch_size
            )

    def qat_vram_overhead(self, bits: int = 6) -> float:
        """Return the VRAM multiplier for QAT at the given bit width.

        Args:
            bits: QAT bit width.

        Returns:
            VRAM multiplier (1.0 = no overhead).
        """
        return self._QAT_VRAM_MULTIPLIERS.get(bits, 1.25)

    def is_qat_active(self) -> bool:
        """Check if QAT mode is currently active."""
        return hasattr(self, "_qat_bits")


# ---------------------------------------------------------------------------
# OldFreezer: 8–10 GB cards
# ---------------------------------------------------------------------------

class OldFreezer(Freezer):
    """For 8–10 GB GPUs (GTX 1080, RTX 2060/2070, RTX 3060-8GB, …).

    Picks dtype from :func:`pascal_safe_dtype`, so on a GTX 1080 (sm_61)
    you get native fp16 rather than emulated bf16.
    """

    preferred_dtype = pascal_safe_dtype()
    base_batch_size = 1
    base_context_length = 512
    empty_cache_each_step = True
    name = "OldFreezer"
    _caps_hint = True


# ---------------------------------------------------------------------------
# NewFreezer: 11 GB+
# ---------------------------------------------------------------------------

class NewFreezer(Freezer):
    """For 11 GB+ GPUs (RTX 2080 Ti, RTX 3080 12GB, 3090, 4090, H100, …)."""

    preferred_dtype = torch.float32
    base_batch_size = 8
    base_context_length = 2048
    empty_cache_each_step = False
    name = "NewFreezer"
    _caps_hint = False  # hint wins on big cards


# ---------------------------------------------------------------------------
# FlashFreezer: wraps a base freezer with OOM-safe retry
# ---------------------------------------------------------------------------

class FlashFreezer(Freezer):
    """Adaptive wrapper that slows or pauses to avoid OOM.

    On every ``guard(fn)`` call it runs ``fn``.  If a
    ``torch.cuda.OutOfMemoryError`` fires:

    1. empty the allocator cache,
    2. wait until ``probe_vram().free_gb`` climbs back above
       ``min_free_gb`` (or a fixed backoff timeout, whichever happens
       first),
    3. retry — up to ``max_retries`` times.

    If ``slow`` is True, each retry halves the effective batch size via
    the ``current_batch_size`` attribute, which your training loop should
    consult in place of a hard-coded batch size.  Progress still happens;
    it's just slower.
    """

    name = "FlashFreezer"

    def __init__(
        self,
        base: Freezer | None = None,
        *,
        max_retries: int = 5,
        backoff_s: float = 2.0,
        min_free_gb: float = 0.5,
        slow: bool = True,
    ) -> None:
        self.base = base or auto_freezer()
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.min_free_gb = min_free_gb
        self.slow = slow
        # Mirror the base's tuning so suggest_* work transparently.
        self.preferred_dtype = self.base.preferred_dtype
        self.base_batch_size = self.base.base_batch_size
        self.base_context_length = self.base.base_context_length
        self.empty_cache_each_step = self.base.empty_cache_each_step
        self._caps_hint = self.base._caps_hint
        self.current_batch_size = self.base_batch_size

    def wait_for(self, min_free_gb: float | None = None, timeout_s: float | None = None) -> bool:
        """Block until ``probe_vram().free_gb >= min_free_gb`` or timeout.

        Returns True if the threshold was reached, False on timeout.
        CPU-only systems return True immediately.
        """
        need = self.min_free_gb if min_free_gb is None else min_free_gb
        if not torch.cuda.is_available():
            return True
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if probe_vram().free_gb >= need:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(min(self.backoff_s, 1.0))

    def guard(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn``; on OOM empty the cache, wait, and retry."""
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.slow and self.current_batch_size > 1:
                    self.current_batch_size = max(1, self.current_batch_size // 2)
                sleep_for = self.backoff_s * (2 ** attempt)
                time.sleep(min(sleep_for, 60.0))
                self.wait_for(self.min_free_gb, timeout_s=sleep_for)
        if last_exc is None:
            raise AssertionError('last_exc is not None')
        raise last_exc


# ---------------------------------------------------------------------------
# Auto-picker
# ---------------------------------------------------------------------------

def auto_freezer(threshold_gb: float = 11.0) -> Freezer:
    """Return :class:`NewFreezer` if total VRAM ≥ ``threshold_gb``, else
    :class:`OldFreezer`.  On a CPU-only host returns :class:`OldFreezer`
    (its small defaults won't hurt; they just produce tiny batches)."""
    b = probe_vram()
    if b.total_gb >= threshold_gb:
        return NewFreezer()
    return OldFreezer()


def old_freezer() -> OldFreezer:
    return OldFreezer()


def new_freezer() -> NewFreezer:
    return NewFreezer()


def flash_freezer(**kwargs: Any) -> FlashFreezer:
    return FlashFreezer(**kwargs)


# ---------------------------------------------------------------------------
# CPU presets
# ---------------------------------------------------------------------------
# Per-CPU tuning hints used by the smoke alarms and by callers that
# want sane BLAS / OpenMP defaults for a known chip.  Throughput numbers
# are deliberately conservative — they're rough heuristics for
# step-time estimation, not benchmark results.

@dataclass(frozen=True)
class CPUPreset:
    name: str
    cores: int
    threads: int
    base_clock_ghz: float
    avx_levels: tuple[str, ...]
    recommended_threads: int
    #: Approximate fp32 GFLOPS per thread for the kernels hypernix
    #: cares about (matmul-heavy AdamW training).  Multiply by
    #: ``recommended_threads`` for a whole-CPU figure.
    gflops_per_thread: float
    notes: str = ""


def _cpu(name, cores, threads, ghz, avx, threads_rec, gflops, notes=""):
    return CPUPreset(
        name=name, cores=cores, threads=threads, base_clock_ghz=ghz,
        avx_levels=tuple(avx), recommended_threads=threads_rec,
        gflops_per_thread=gflops, notes=notes,
    )


#: Lookup table — short name -> :class:`CPUPreset`.  Names are
#: lowercased and ``-`` / ``_`` are equivalent at lookup time.
CPU_PRESETS: dict[str, CPUPreset] = {
    # ---- Intel 7th gen (Kaby Lake) ----
    "i7-7660u": _cpu("Intel Core i7-7660U", 2, 4, 2.5, ["AVX", "AVX2"], 2,
                     14.0, "15W ULV ultrabook (e.g. 13\" MBP, XPS 13)"),
    "i7-7700hq": _cpu("Intel Core i7-7700HQ", 4, 8, 2.8, ["AVX", "AVX2"], 4,
                      18.0, "45W mobile gaming/workstation"),
    "i7-7700k": _cpu("Intel Core i7-7700K", 4, 8, 4.2, ["AVX", "AVX2"], 4,
                     24.0, "91W desktop"),
    # ---- 11th gen ----
    "i7-11700k": _cpu("Intel Core i7-11700K", 8, 16, 3.6, ["AVX2", "AVX-512"], 8,
                      30.0, "Rocket Lake desktop, AVX-512"),
    "i7-11800h": _cpu("Intel Core i7-11800H", 8, 16, 2.3, ["AVX2"], 8,
                      24.0, "Tiger Lake-H mobile"),
    # ---- 12th gen (Alder Lake hybrid: P + E cores) ----
    "i7-12700k": _cpu("Intel Core i7-12700K", 12, 20, 3.6, ["AVX2"], 8,
                      30.0, "8 P-cores + 4 E-cores"),
    "i7-12700h": _cpu("Intel Core i7-12700H", 14, 20, 2.3, ["AVX2"], 10,
                      26.0, "6 P + 8 E mobile"),
    # ---- 13th gen ----
    "i7-13700k": _cpu("Intel Core i7-13700K", 16, 24, 3.4, ["AVX2"], 12,
                      32.0, "8 P + 8 E desktop"),
    "i7-13700h": _cpu("Intel Core i7-13700H", 14, 20, 2.4, ["AVX2"], 10,
                      28.0, "6 P + 8 E mobile"),
    # ---- 14th gen (Raptor Lake-R) ----
    "i7-14700k": _cpu("Intel Core i7-14700K", 20, 28, 3.4, ["AVX2"], 14,
                      34.0, "8 P + 12 E desktop"),
    "i7-14700hx": _cpu("Intel Core i7-14700HX", 20, 28, 2.1, ["AVX2"], 14,
                       30.0, "8 P + 12 E mobile-HX"),
    # ---- Core Ultra Series 1 (Meteor Lake, last-gen) ----
    "core-ultra-7-155h": _cpu("Intel Core Ultra 7 155H", 16, 22, 1.4,
                              ["AVX2", "AVX-VNNI"], 12, 28.0,
                              "Meteor Lake, 6P + 8E + 2LP-E + NPU"),
    "core-ultra-7-165h": _cpu("Intel Core Ultra 7 165H", 16, 22, 1.4,
                              ["AVX2", "AVX-VNNI"], 12, 30.0,
                              "Meteor Lake, refresh of 155H"),
    "core-ultra-7-258v": _cpu("Intel Core Ultra 7 258V", 8, 8, 2.2,
                              ["AVX2", "AVX-VNNI"], 6, 24.0,
                              "Lunar Lake, 4P + 4LP-E"),
    # ---- Core Ultra Series 2 (Arrow Lake, newest gen) ----
    "core-ultra-7-265k": _cpu("Intel Core Ultra 7 265K", 20, 20, 3.9,
                              ["AVX2", "AVX-VNNI", "AVX10"], 14, 36.0,
                              "Arrow Lake desktop, no SMT"),
    "core-ultra-9-285k": _cpu("Intel Core Ultra 9 285K", 24, 24, 3.7,
                              ["AVX2", "AVX-VNNI", "AVX10"], 16, 38.0,
                              "Arrow Lake top SKU, no SMT"),
}

# ---------------------------------------------------------------------------
# v0.47: i5 and i9 family expansion (7th, 11th, 12th, 13th, 14th gen) +
# Core Ultra 5 / Ultra 9 Series 1 and 2.  Added in one block so the
# git diff is readable.
# ---------------------------------------------------------------------------
CPU_PRESETS.update({
    # ---- 7th gen i5 (Kaby Lake) ----
    "i5-7200u": _cpu("Intel Core i5-7200U", 2, 4, 2.5, ["AVX", "AVX2"], 2,
                     12.0, "15W ULV ultrabook"),
    "i5-7300hq": _cpu("Intel Core i5-7300HQ", 4, 4, 2.5, ["AVX", "AVX2"], 4,
                      14.0, "45W mobile, no SMT"),
    "i5-7400": _cpu("Intel Core i5-7400", 4, 4, 3.0, ["AVX", "AVX2"], 4,
                    16.0, "65W desktop"),
    "i5-7600k": _cpu("Intel Core i5-7600K", 4, 4, 3.8, ["AVX", "AVX2"], 4,
                     20.0, "91W desktop, unlocked"),
    # ---- 7th gen i9 (Skylake-X HEDT — same family as 7700K era) ----
    "i9-7900x": _cpu("Intel Core i9-7900X", 10, 20, 3.3, ["AVX", "AVX2", "AVX-512"],
                     10, 28.0, "HEDT, 140W"),
    "i9-7980xe": _cpu("Intel Core i9-7980XE", 18, 36, 2.6,
                      ["AVX", "AVX2", "AVX-512"], 18, 30.0,
                      "HEDT flagship, 165W"),
    # ---- 11th gen i5 (Rocket Lake) ----
    "i5-11400": _cpu("Intel Core i5-11400", 6, 12, 2.6, ["AVX2", "AVX-512"], 6,
                     22.0, "65W desktop"),
    "i5-11600k": _cpu("Intel Core i5-11600K", 6, 12, 3.9, ["AVX2", "AVX-512"], 6,
                      26.0, "125W desktop, unlocked"),
    "i5-11320h": _cpu("Intel Core i5-11320H", 4, 8, 3.2, ["AVX2"], 4,
                      18.0, "35W mobile"),
    # ---- 11th gen i9 ----
    "i9-11900k": _cpu("Intel Core i9-11900K", 8, 16, 3.5, ["AVX2", "AVX-512"],
                      8, 32.0, "125W desktop, unlocked"),
    # ---- 12th gen i5 (Alder Lake) ----
    "i5-12400": _cpu("Intel Core i5-12400", 6, 12, 2.5, ["AVX2"], 6,
                     22.0, "6 P-cores, no E-cores"),
    "i5-12500": _cpu("Intel Core i5-12500", 6, 12, 3.0, ["AVX2"], 6,
                     24.0, ""),
    "i5-12600k": _cpu("Intel Core i5-12600K", 10, 16, 3.7, ["AVX2"], 8,
                      28.0, "6 P + 4 E desktop"),
    # ---- 12th gen i9 ----
    "i9-12900k": _cpu("Intel Core i9-12900K", 16, 24, 3.2, ["AVX2"], 12,
                      34.0, "8 P + 8 E desktop"),
    "i9-12900hx": _cpu("Intel Core i9-12900HX", 16, 24, 2.3, ["AVX2"], 12,
                       30.0, "8 P + 8 E mobile-HX"),
    # ---- 13th gen i5 (Raptor Lake) ----
    "i5-13400": _cpu("Intel Core i5-13400", 10, 16, 2.5, ["AVX2"], 8,
                     24.0, "6 P + 4 E"),
    "i5-13500": _cpu("Intel Core i5-13500", 14, 20, 2.5, ["AVX2"], 10,
                     26.0, "6 P + 8 E"),
    "i5-13600k": _cpu("Intel Core i5-13600K", 14, 20, 3.5, ["AVX2"], 10,
                      30.0, "6 P + 8 E desktop"),
    # ---- 13th gen i9 ----
    "i9-13900k": _cpu("Intel Core i9-13900K", 24, 32, 3.0, ["AVX2"], 16,
                      36.0, "8 P + 16 E desktop"),
    "i9-13900hx": _cpu("Intel Core i9-13900HX", 24, 32, 2.2, ["AVX2"], 16,
                       32.0, "8 P + 16 E mobile"),
    # ---- 14th gen i5 (Raptor Lake-R) ----
    "i5-14400": _cpu("Intel Core i5-14400", 10, 16, 2.5, ["AVX2"], 8,
                     26.0, ""),
    "i5-14500": _cpu("Intel Core i5-14500", 14, 20, 2.6, ["AVX2"], 10,
                     28.0, ""),
    "i5-14600k": _cpu("Intel Core i5-14600K", 14, 20, 3.5, ["AVX2"], 10,
                      30.0, "6 P + 8 E desktop"),
    # ---- 14th gen i9 ----
    "i9-14900k": _cpu("Intel Core i9-14900K", 24, 32, 3.2, ["AVX2"], 16,
                      38.0, "8 P + 16 E desktop"),
    "i9-14900ks": _cpu("Intel Core i9-14900KS", 24, 32, 3.2, ["AVX2"], 16,
                       40.0, "binned 6.2 GHz boost"),
    "i9-14900hx": _cpu("Intel Core i9-14900HX", 24, 32, 2.2, ["AVX2"], 16,
                       34.0, "8 P + 16 E mobile"),
    # ---- Core Ultra 5 Series 1 (Meteor Lake) ----
    "core-ultra-5-125h": _cpu("Intel Core Ultra 5 125H", 14, 18, 1.2,
                              ["AVX2", "AVX-VNNI"], 10, 24.0,
                              "4 P + 8 E + 2 LP-E, NPU"),
    "core-ultra-5-135h": _cpu("Intel Core Ultra 5 135H", 14, 18, 1.2,
                              ["AVX2", "AVX-VNNI"], 10, 26.0,
                              "refresh of 125H"),
    "core-ultra-5-228v": _cpu("Intel Core Ultra 5 228V", 8, 8, 2.0,
                              ["AVX2", "AVX-VNNI"], 6, 22.0,
                              "Lunar Lake"),
    # ---- Core Ultra 5 Series 2 (Arrow Lake) ----
    "core-ultra-5-225k": _cpu("Intel Core Ultra 5 225K", 10, 10, 3.3,
                              ["AVX2", "AVX-VNNI", "AVX10"], 8, 28.0,
                              "Arrow Lake desktop, no SMT"),
    "core-ultra-5-235k": _cpu("Intel Core Ultra 5 235K", 14, 14, 3.4,
                              ["AVX2", "AVX-VNNI", "AVX10"], 10, 30.0,
                              "Arrow Lake desktop, no SMT"),
    # ---- Core Ultra 9 Series 1 ----
    "core-ultra-9-185h": _cpu("Intel Core Ultra 9 185H", 16, 22, 2.3,
                              ["AVX2", "AVX-VNNI"], 12, 32.0,
                              "Meteor Lake top SKU"),
})

# ---------------------------------------------------------------------------
# AMD Ryzen desktop, Zen 3 / 4 / 5 (5000 / 7000 / 9000 series). The module
# docstring/README referenced "Ryzen" support before any Ryzen entries
# actually existed here — this closes that gap. Zen 3 is AVX2-only; Zen 4
# added full-width AVX-512 with none of the heavy frequency throttling
# Intel's earlier AVX-512 implementations show under sustained load (see
# the Zen 4/5 gflops_per_thread values below relative to same-clock AVX-512
# Intel entries above), and Zen 5 adds a further real IPC uplift on top of
# that at comparable clocks. recommended_threads = physical core count for
# all of these (Ryzen desktop has no hybrid P/E-core split to account for,
# unlike the Alder-Lake-and-later Intel entries above).
# ---------------------------------------------------------------------------
CPU_PRESETS.update({
    # ---- Ryzen 5000 series (Zen 3, AM4, AVX2 only) ----
    "ryzen-5-5600x": _cpu("AMD Ryzen 5 5600X", 6, 12, 3.7, ["AVX", "AVX2"], 6,
                          22.0, "65W desktop"),
    "ryzen-7-5800x": _cpu("AMD Ryzen 7 5800X", 8, 16, 3.8, ["AVX", "AVX2"], 8,
                          26.0, "105W desktop"),
    "ryzen-9-5900x": _cpu("AMD Ryzen 9 5900X", 12, 24, 3.7, ["AVX", "AVX2"], 12,
                          28.0, "105W desktop, 2 CCDs"),
    "ryzen-9-5950x": _cpu("AMD Ryzen 9 5950X", 16, 32, 3.4, ["AVX", "AVX2"], 16,
                          30.0, "105W desktop flagship, 2 CCDs"),
    # ---- Ryzen 7000 series (Zen 4, AM5, AVX-512) ----
    "ryzen-5-7600x": _cpu("AMD Ryzen 5 7600X", 6, 12, 4.7, ["AVX2", "AVX-512"], 6,
                          30.0, "105W desktop"),
    "ryzen-7-7700x": _cpu("AMD Ryzen 7 7700X", 8, 16, 4.5, ["AVX2", "AVX-512"], 8,
                          34.0, "105W desktop"),
    "ryzen-9-7900x": _cpu("AMD Ryzen 9 7900X", 12, 24, 4.7, ["AVX2", "AVX-512"], 12,
                          38.0, "170W desktop, 2 CCDs"),
    "ryzen-9-7950x": _cpu("AMD Ryzen 9 7950X", 16, 32, 4.5, ["AVX2", "AVX-512"], 16,
                          40.0, "170W desktop flagship, 2 CCDs"),
    # ---- Ryzen 9000 series (Zen 5, AM5, AVX-512) ----
    "ryzen-5-9600x": _cpu("AMD Ryzen 5 9600X", 6, 12, 3.9, ["AVX2", "AVX-512"], 6,
                          32.0, "65W desktop"),
    "ryzen-7-9700x": _cpu("AMD Ryzen 7 9700X", 8, 16, 3.8, ["AVX2", "AVX-512"], 8,
                          36.0, "65W desktop"),
    "ryzen-9-9900x": _cpu("AMD Ryzen 9 9900X", 12, 24, 4.4, ["AVX2", "AVX-512"], 12,
                          40.0, "120W desktop, 2 CCDs"),
    "ryzen-9-9950x": _cpu("AMD Ryzen 9 9950X", 16, 32, 4.3, ["AVX2", "AVX-512"], 16,
                          42.0, "170W desktop flagship, 2 CCDs"),
})


def _cpu_key(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


# 0.52.5: generational-family aliases.  Real users type
# ``i7_7th_gen`` / ``i9-12th-gen`` etc. instead of the exact SKU,
# so resolve those to a representative canonical entry rather than
# returning None.  Pick the mobile flagship per generation since
# that's the most common laptop SKU.
_CPU_GENERATIONAL_ALIASES: dict[str, str] = {
    # i3 / i5 / i7 / i9 by generation
    "i7-7th-gen": "i7-7700hq",
    "i7-8th-gen": "i7-7700hq",   # closest CPU_PRESETS entry; fall-back family
    "i7-9th-gen": "i7-7700hq",
    "i7-10th-gen": "i7-7700hq",
    "i7-11th-gen": "i7-11800h",
    "i7-12th-gen": "i7-12700h",
    "i7-13th-gen": "i7-13700h",
    "i7-14th-gen": "i7-14700hx",
    "i9-11th-gen": "i9-11900k",
    "i9-12th-gen": "i9-12900k",
    "i9-13th-gen": "i9-13900k",
    "i9-14th-gen": "i9-14900k",
    "i5-11th-gen": "i5-11600k",
    "i5-12th-gen": "i5-12600k",
    "i5-13th-gen": "i5-13600k",
    "i5-14th-gen": "i5-14600k",
    # Core Ultra families (Series 1 + 2)
    "ultra-5": "core-ultra-5-125h",
    "ultra-7": "core-ultra-7-155h",
    "ultra-9": "core-ultra-9-185h",
    "core-ultra": "core-ultra-7-155h",
    "core-ultra-1": "core-ultra-7-155h",
    "core-ultra-2": "core-ultra-7-265k",
    # Ryzen 5 / 7 / 9 by series (desktop flagship-adjacent SKU per series,
    # mirroring the i7/i9 "Nth-gen" aliases above)
    "ryzen-5000": "ryzen-9-5900x",
    "ryzen-7000": "ryzen-9-7900x",
    "ryzen-9000": "ryzen-9-9900x",
    "ryzen-5-5000": "ryzen-5-5600x",
    "ryzen-7-5000": "ryzen-7-5800x",
    "ryzen-9-5000": "ryzen-9-5900x",
    "ryzen-5-7000": "ryzen-5-7600x",
    "ryzen-7-7000": "ryzen-7-7700x",
    "ryzen-9-7000": "ryzen-9-7900x",
    "ryzen-5-9000": "ryzen-5-9600x",
    "ryzen-7-9000": "ryzen-7-9700x",
    "ryzen-9-9000": "ryzen-9-9900x",
}


def cpu_preset(name: str) -> CPUPreset | None:
    """Look up a CPU preset by short name (case- and dash-insensitive).

    0.52.5: also resolves generational-family aliases like
    ``"i7_7th_gen"`` / ``"i9-12th-gen"`` / ``"core-ultra"`` to a
    representative SKU when the exact key isn't in
    :data:`CPU_PRESETS`.
    """
    key = _cpu_key(name)
    direct = CPU_PRESETS.get(key)
    if direct is not None:
        return direct
    aliased = _CPU_GENERATIONAL_ALIASES.get(key)
    if aliased is not None:
        return CPU_PRESETS.get(aliased)
    return None


# ---------------------------------------------------------------------------
# GPU presets
# ---------------------------------------------------------------------------
# Per-GPU tuning hints used by the smoke alarms and by callers that
# want a known-good starting point without probing torch.cuda.

@dataclass(frozen=True)
class GPUPreset:
    name: str
    vram_gb: float
    compute_capability: tuple[int, int]
    preferred_dtype: torch.dtype
    #: Approximate memory bandwidth in GB/s — the dominant signal for
    #: training throughput on transformer workloads.
    bandwidth_gb_s: float
    #: "Old" or "New" — which freezer class fits this card.  Anything
    #: with < 11 GB VRAM lands on Old; the Pascal/Volta/Turing fp16-only
    #: cards stay on Old even when they have 12 GB.
    freezer_class: str
    notes: str = ""


def _gpu(name, vram, cc, dtype, bw, fz_class, notes=""):
    return GPUPreset(
        name=name, vram_gb=vram, compute_capability=cc, preferred_dtype=dtype,
        bandwidth_gb_s=bw, freezer_class=fz_class, notes=notes,
    )


GPU_PRESETS: dict[str, GPUPreset] = {
    # ---- Hopper (sm_90) data-center ----
    "h100": _gpu("NVIDIA H100 80GB", 80.0, (9, 0), torch.bfloat16, 3350.0, "New",
                 "PCIe / SXM5 80GB"),
    "h100-94": _gpu("NVIDIA H100 NVL 94GB", 94.0, (9, 0), torch.bfloat16, 3900.0, "New",
                    "NVL variant, 94GB HBM3"),
    "h200": _gpu("NVIDIA H200 141GB", 141.0, (9, 0), torch.bfloat16, 4800.0, "New",
                 "HBM3e, 141GB"),
    # ---- Ampere workstation (sm_86) ----
    "rtx-a4500": _gpu("NVIDIA RTX A4500", 20.0, (8, 6), torch.bfloat16, 640.0, "New",
                      "Ampere workstation"),
    "rtx-a5000": _gpu("NVIDIA RTX A5000", 24.0, (8, 6), torch.bfloat16, 768.0, "New", ""),
    "rtx-a5500": _gpu("NVIDIA RTX A5500", 24.0, (8, 6), torch.bfloat16, 768.0, "New", ""),
    "rtx-a6000": _gpu("NVIDIA RTX A6000", 48.0, (8, 6), torch.bfloat16, 768.0, "New",
                      "Ampere workstation flagship"),
    # ---- RTX PRO (Ada Lovelace, sm_89) ----
    "rtx-pro-4000-ada": _gpu("NVIDIA RTX PRO 4000 Ada", 20.0, (8, 9),
                             torch.bfloat16, 360.0, "New", ""),
    "rtx-pro-5000-ada": _gpu("NVIDIA RTX PRO 5000 Ada", 32.0, (8, 9),
                             torch.bfloat16, 576.0, "New", ""),
    "rtx-pro-6000-ada": _gpu("NVIDIA RTX PRO 6000 Ada", 48.0, (8, 9),
                             torch.bfloat16, 960.0, "New",
                             "Ada workstation flagship"),
    # ---- RTX PRO Blackwell (sm_120) ----
    "rtx-pro-6000-blackwell": _gpu("NVIDIA RTX PRO 6000 Blackwell", 96.0, (12, 0),
                                   torch.bfloat16, 1792.0, "New",
                                   "Blackwell workstation, 96GB"),
    # ---- Ada Lovelace consumer (sm_89) ----
    "rtx-4070-ti-super": _gpu("NVIDIA GeForce RTX 4070 Ti Super", 16.0, (8, 9),
                              torch.bfloat16, 672.0, "New",
                              "16GB GDDR6X"),
    "rtx-4080-super": _gpu("NVIDIA GeForce RTX 4080 Super", 16.0, (8, 9),
                           torch.bfloat16, 736.0, "New", ""),
    # ---- Turing consumer (sm_75 — no native bf16) ----
    "gtx-1660-ti": _gpu("NVIDIA GeForce GTX 1660 Ti", 6.0, (7, 5),
                        torch.float16, 288.0, "Old",
                        "Turing without RT cores; 6GB caps batch hard"),
    "rtx-2080": _gpu("NVIDIA GeForce RTX 2080", 8.0, (7, 5),
                     torch.float16, 448.0, "Old", ""),
    "rtx-2080-super": _gpu("NVIDIA GeForce RTX 2080 Super", 8.0, (7, 5),
                           torch.float16, 496.0, "Old", ""),
    "rtx-2080-ti": _gpu("NVIDIA GeForce RTX 2080 Ti", 11.0, (7, 5),
                        torch.float16, 616.0, "New",
                        "11GB lands on the New side of the 11GB threshold"),
    # ---- Ampere consumer (sm_86) ----
    "rtx-3080-ti": _gpu("NVIDIA GeForce RTX 3080 Ti", 12.0, (8, 6),
                        torch.bfloat16, 912.0, "New",
                        "12GB GDDR6X; bf16 native"),
    # ---- Pascal (kept for the GTX 1080 playbook) ----
    "gtx-1080": _gpu("NVIDIA GeForce GTX 1080", 8.0, (6, 1),
                     torch.float16, 320.0, "Old",
                     "Pascal sm_61 — see wiki/Pascal.md"),
    "gtx-1080-ti": _gpu("NVIDIA GeForce GTX 1080 Ti", 11.0, (6, 1),
                        torch.float16, 484.0, "New",
                        "Pascal sm_61, 11GB"),
}

# ---------------------------------------------------------------------------
# v0.47: GPU preset expansion — fuller GTX 10 / RTX 20 / 30 / 40 / 50
# coverage + Apple M-series unified memory + AMD Instinct + consumer
# Radeon.  For Apple cards the "VRAM" field is unified memory;
# freezer_class is set by whether the box typically has ≥11 GB free.
# ---------------------------------------------------------------------------
GPU_PRESETS.update({
    # ---- Remaining Pascal (sm_61) ----
    "gtx-1050": _gpu("NVIDIA GeForce GTX 1050", 2.0, (6, 1),
                     torch.float16, 112.0, "Old",
                     "2GB, desktop / laptop low-end"),
    "gtx-1050-ti": _gpu("NVIDIA GeForce GTX 1050 Ti", 4.0, (6, 1),
                        torch.float16, 112.0, "Old", "4GB entry"),
    "gtx-1060": _gpu("NVIDIA GeForce GTX 1060 6GB", 6.0, (6, 1),
                     torch.float16, 192.0, "Old", ""),
    "gtx-1070": _gpu("NVIDIA GeForce GTX 1070", 8.0, (6, 1),
                     torch.float16, 256.0, "Old", ""),
    "gtx-1070-ti": _gpu("NVIDIA GeForce GTX 1070 Ti", 8.0, (6, 1),
                        torch.float16, 256.0, "Old", ""),

    # ---- Remaining Turing (sm_75) ----
    "gtx-1650": _gpu("NVIDIA GeForce GTX 1650", 4.0, (7, 5),
                     torch.float16, 128.0, "Old", ""),
    "gtx-1650-super": _gpu("NVIDIA GeForce GTX 1650 SUPER", 4.0, (7, 5),
                           torch.float16, 192.0, "Old", ""),
    "gtx-1660": _gpu("NVIDIA GeForce GTX 1660", 6.0, (7, 5),
                     torch.float16, 192.0, "Old", ""),
    "gtx-1660-super": _gpu("NVIDIA GeForce GTX 1660 SUPER", 6.0, (7, 5),
                           torch.float16, 336.0, "Old", ""),
    "rtx-2060": _gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5),
                     torch.float16, 336.0, "Old", ""),
    "rtx-2060-super": _gpu("NVIDIA GeForce RTX 2060 SUPER", 8.0, (7, 5),
                           torch.float16, 448.0, "Old", ""),
    "rtx-2070": _gpu("NVIDIA GeForce RTX 2070", 8.0, (7, 5),
                     torch.float16, 448.0, "Old", ""),
    "rtx-2070-super": _gpu("NVIDIA GeForce RTX 2070 SUPER", 8.0, (7, 5),
                           torch.float16, 448.0, "Old", ""),

    # ---- Ampere consumer (sm_86) ----
    "rtx-3050": _gpu("NVIDIA GeForce RTX 3050 8GB", 8.0, (8, 6),
                     torch.bfloat16, 224.0, "Old", ""),
    "rtx-3060": _gpu("NVIDIA GeForce RTX 3060 12GB", 12.0, (8, 6),
                     torch.bfloat16, 360.0, "New",
                     "12GB GDDR6 — the 8GB variant has only 8GB"),
    "rtx-3060-ti": _gpu("NVIDIA GeForce RTX 3060 Ti", 8.0, (8, 6),
                        torch.bfloat16, 448.0, "Old", ""),
    "rtx-3070": _gpu("NVIDIA GeForce RTX 3070", 8.0, (8, 6),
                     torch.bfloat16, 448.0, "Old", ""),
    "rtx-3070-ti": _gpu("NVIDIA GeForce RTX 3070 Ti", 8.0, (8, 6),
                        torch.bfloat16, 608.0, "Old", ""),
    "rtx-3080": _gpu("NVIDIA GeForce RTX 3080 10GB", 10.0, (8, 6),
                     torch.bfloat16, 760.0, "Old",
                     "10GB — OldFreezer territory"),
    "rtx-3090": _gpu("NVIDIA GeForce RTX 3090", 24.0, (8, 6),
                     torch.bfloat16, 936.0, "New", ""),
    "rtx-3090-ti": _gpu("NVIDIA GeForce RTX 3090 Ti", 24.0, (8, 6),
                        torch.bfloat16, 1008.0, "New", ""),

    # ---- Ada Lovelace consumer (sm_89) ----
    "rtx-4060": _gpu("NVIDIA GeForce RTX 4060", 8.0, (8, 9),
                     torch.bfloat16, 272.0, "Old", ""),
    "rtx-4060-ti-8g": _gpu("NVIDIA GeForce RTX 4060 Ti 8GB", 8.0, (8, 9),
                           torch.bfloat16, 288.0, "Old", ""),
    "rtx-4060-ti-16g": _gpu("NVIDIA GeForce RTX 4060 Ti 16GB", 16.0, (8, 9),
                            torch.bfloat16, 288.0, "New",
                            "16GB variant"),
    "rtx-4070": _gpu("NVIDIA GeForce RTX 4070", 12.0, (8, 9),
                     torch.bfloat16, 504.0, "New", ""),
    "rtx-4070-ti": _gpu("NVIDIA GeForce RTX 4070 Ti", 12.0, (8, 9),
                        torch.bfloat16, 504.0, "New", ""),
    "rtx-4080": _gpu("NVIDIA GeForce RTX 4080", 16.0, (8, 9),
                     torch.bfloat16, 717.0, "New", ""),
    "rtx-4090": _gpu("NVIDIA GeForce RTX 4090", 24.0, (8, 9),
                     torch.bfloat16, 1008.0, "New", ""),

    # ---- Blackwell consumer (sm_120) ----
    "rtx-5070": _gpu("NVIDIA GeForce RTX 5070", 12.0, (12, 0),
                     torch.bfloat16, 672.0, "New", ""),
    "rtx-5070-ti": _gpu("NVIDIA GeForce RTX 5070 Ti", 16.0, (12, 0),
                        torch.bfloat16, 896.0, "New", ""),
    "rtx-5080": _gpu("NVIDIA GeForce RTX 5080", 16.0, (12, 0),
                     torch.bfloat16, 960.0, "New", ""),
    "rtx-5090": _gpu("NVIDIA GeForce RTX 5090", 32.0, (12, 0),
                     torch.bfloat16, 1792.0, "New", ""),

    # ---- Apple Silicon / MPS ----
    # Compute capability is the conventional (0, 0) sentinel for MPS
    # (not a CUDA device); VRAM is *unified memory* on Apple silicon,
    # so "free" is fluid.  Bandwidth is the memory bus.
    "apple-m1": _gpu("Apple M1 8GB", 8.0, (0, 0), torch.float16,
                     68.0, "Old", "MPS, unified memory"),
    "apple-m1-pro": _gpu("Apple M1 Pro 16GB", 16.0, (0, 0), torch.float16,
                         200.0, "New", "MPS"),
    "apple-m1-max": _gpu("Apple M1 Max 32GB", 32.0, (0, 0), torch.float16,
                         400.0, "New", "MPS"),
    "apple-m1-ultra": _gpu("Apple M1 Ultra 64GB", 64.0, (0, 0), torch.float16,
                           800.0, "New", "MPS, 2-chip"),
    "apple-m2": _gpu("Apple M2 8GB", 8.0, (0, 0), torch.float16,
                     100.0, "Old", "MPS"),
    "apple-m2-pro": _gpu("Apple M2 Pro 16GB", 16.0, (0, 0), torch.float16,
                         200.0, "New", "MPS"),
    "apple-m2-max": _gpu("Apple M2 Max 32GB", 32.0, (0, 0), torch.float16,
                         400.0, "New", "MPS"),
    "apple-m3": _gpu("Apple M3", 8.0, (0, 0), torch.float16,
                     100.0, "Old", "MPS"),
    "apple-m3-pro": _gpu("Apple M3 Pro", 18.0, (0, 0), torch.float16,
                         150.0, "New", "MPS, reduced bandwidth vs M2 Pro"),
    "apple-m3-max": _gpu("Apple M3 Max", 36.0, (0, 0), torch.float16,
                         400.0, "New", "MPS"),
    "apple-m4": _gpu("Apple M4", 16.0, (0, 0), torch.float16,
                     120.0, "New", "MPS"),
    "apple-m4-pro": _gpu("Apple M4 Pro", 24.0, (0, 0), torch.float16,
                         273.0, "New", "MPS"),
    "apple-m4-max": _gpu("Apple M4 Max", 48.0, (0, 0), torch.float16,
                         546.0, "New", "MPS"),

    # ---- AMD Radeon (ROCm / DirectML) ----
    # compute_capability (0, 0) sentinel — these aren't CUDA devices.
    # freezer_class is set by raw VRAM.
    "radeon-rx-6800-xt": _gpu("AMD Radeon RX 6800 XT", 16.0, (0, 0),
                              torch.float16, 512.0, "New",
                              "RDNA 2, 16GB"),
    "radeon-rx-6900-xt": _gpu("AMD Radeon RX 6900 XT", 16.0, (0, 0),
                              torch.float16, 512.0, "New", "RDNA 2"),
    "radeon-rx-7900-xt": _gpu("AMD Radeon RX 7900 XT", 20.0, (0, 0),
                              torch.float16, 800.0, "New", "RDNA 3"),
    "radeon-rx-7900-xtx": _gpu("AMD Radeon RX 7900 XTX", 24.0, (0, 0),
                               torch.float16, 960.0, "New", "RDNA 3 flagship"),
    # AMD Instinct (data-center, CDNA)
    "instinct-mi250x": _gpu("AMD Instinct MI250X", 128.0, (0, 0),
                            torch.bfloat16, 3277.0, "New",
                            "CDNA 2, 2×64GB HBM"),
    "instinct-mi300x": _gpu("AMD Instinct MI300X", 192.0, (0, 0),
                            torch.bfloat16, 5300.0, "New",
                            "CDNA 3, 192GB HBM3"),
})


def _gpu_key(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


def gpu_preset(name: str) -> GPUPreset | None:
    """Look up a GPU preset by short name (case- and dash-insensitive)."""
    return GPU_PRESETS.get(_gpu_key(name))
