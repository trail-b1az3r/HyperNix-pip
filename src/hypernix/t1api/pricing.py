"""t1api.pricing — what a model costs to serve, estimated from the file.

``hypernix-t1 index`` reads everything about a model except what to
charge for it, because that is a policy decision. But "a policy decision"
is not much help to somebody with eleven GGUFs and a registry full of
zeros, and a price of zero on a 70B is not a policy, it is an unanswered
question that bills the operator.

So this estimates one, from five things the file and the machine already
know:

1. **File size.** What has to be read off disk and held in memory.
2. **Quantisation.** Bits per weight decides how much of that size is
   model and how much is overhead, and a 2-bit model is cheaper to run
   than a 16-bit one of the same footprint.
3. **Parameters.** The work per token, and the thing that actually scales
   cost. Read from the tensor table, not the filename.
4. **The GPU.** A model that fits in VRAM is an order of magnitude
   cheaper per token than one paging through system RAM, and whether it
   fits is a fact about *this* machine.
5. **Active vs total parameters.** A mixture-of-experts model with 235B
   total and 22B active does 22B of work per token and needs 235B of
   memory. Pricing it as a 235B is wrong by 10x, and pricing it as a 22B
   ignores the hardware it demands.

What this is not
----------------
It is not a market price and it does not know what anybody else charges.
It is a cost-shaped number an operator can start from — and the report
says what went into it, so somebody who disagrees can see *where* they
disagree rather than being handed a figure.

Every estimate carries its assumptions. A model whose active parameter
count could not be read is priced as dense and says so, because a
silently-wrong price is worse than an obviously uncertain one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PriceEstimate",
    "estimate",
    "estimate_for_indexed",
    "GPU_HOURLY_USD",
    "CPU_HOURLY_USD",
]

#: What an hour of GPU is worth, in USD. A mid-range rented card — the
#: figure an operator is most likely to be implicitly paying, whether
#: they rent one or are amortising one they bought. Overridable, and the
#: report names it so a wrong assumption is visible rather than baked in.
GPU_HOURLY_USD = 0.60

#: The same for a machine serving from system RAM and CPU. Much cheaper
#: per hour and much slower per token, which is the whole point: the
#: ratio between these two is what makes "does it fit in VRAM" the
#: biggest single factor here.
CPU_HOURLY_USD = 0.08

#: Tokens per second, per billion *active* parameters, on a GPU with the
#: model resident. A rough constant, and rough is the right precision:
#: the difference between 40 and 60 tokens/s does not change what anybody
#: charges, and the difference between GPU and CPU does.
GPU_TOKENS_PER_SECOND_PER_B = 320.0

#: The same off the GPU. An order of magnitude down, which is the number
#: that matters.
CPU_TOKENS_PER_SECOND_PER_B = 22.0

#: Reading the prompt is cheaper per token than writing the reply —
#: prefill is parallel across the sequence, decode is one token at a
#: time. Roughly a fifth, and the ratio is why input and output are
#: priced separately at all.
INPUT_COST_RATIO = 0.2

#: Never price below this. A model that is nearly free to serve still is
#: not free, and a registry full of 0.0 cannot express "cheap" — it is
#: indistinguishable from "nobody set this".
FLOOR_PER_1K = 0.00002


@dataclass
class PriceEstimate:
    """A price, and everything that went into it."""

    input_price_per_1k: float
    output_price_per_1k: float
    currency: str = "USD"

    #: The inputs, so a disagreement can be located rather than just felt.
    total_parameters_b: float = 0.0
    active_parameters_b: float = 0.0
    file_bytes: int = 0
    bits_per_weight: float = 0.0
    quant: str = ""
    #: What it is expected to run on, given this machine.
    placement: str = ""
    estimated_tokens_per_second: float = 0.0
    vram_bytes: int = 0
    #: Where a number was assumed rather than measured. A model priced as
    #: dense because its expert count could not be read says so here.
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_price_per_1k": round(self.input_price_per_1k, 6),
            "output_price_per_1k": round(self.output_price_per_1k, 6),
            "currency": self.currency,
            "total_parameters_b": round(self.total_parameters_b, 3),
            "active_parameters_b": round(self.active_parameters_b, 3),
            "file_bytes": self.file_bytes,
            "bits_per_weight": round(self.bits_per_weight, 3),
            "quant": self.quant,
            "placement": self.placement,
            "estimated_tokens_per_second": round(self.estimated_tokens_per_second, 1),
            "vram_bytes": self.vram_bytes,
            "assumptions": list(self.assumptions),
        }

    def describe(self) -> str:
        lines = [
            f"  ${self.input_price_per_1k:.5f} in / "
            f"${self.output_price_per_1k:.5f} out per 1k tokens",
            f"  {self.total_parameters_b:.1f}B parameters"
            + (
                f" ({self.active_parameters_b:.1f}B active)"
                if self.active_parameters_b
                and self.active_parameters_b != self.total_parameters_b
                else ""
            )
            + (f" at {self.quant}" if self.quant else ""),
            f"  runs on {self.placement} at ~{self.estimated_tokens_per_second:.0f} tok/s",
        ]
        for note in self.assumptions:
            lines.append(f"  assumed: {note}")
        return "\n".join(lines)


def _vram_bytes() -> int:
    """Total VRAM on this machine, or 0 when there is none to speak of."""
    try:
        from ..system import gpus

        cards = gpus.detect()
    except Exception as exc:  # noqa: BLE001 - no vendor tool is guaranteed
        logger.debug("pricing: could not detect GPUs: %s", exc)
        return 0
    total = 0
    for card in cards:
        megabytes = getattr(card, "memory_total_mb", None)
        if megabytes:
            total += int(megabytes) * 1024 * 1024
    return total


def estimate(
    *,
    file_bytes: int,
    total_parameters_b: float,
    active_parameters_b: float = 0.0,
    bits_per_weight: float = 0.0,
    quant: str = "",
    vram_bytes: int | None = None,
    gpu_hourly: float = GPU_HOURLY_USD,
    cpu_hourly: float = CPU_HOURLY_USD,
) -> PriceEstimate:
    """Estimate what serving this model costs, per 1k tokens.

    *active_parameters_b* is the MoE case and the one most worth getting
    right: a model with 235B total and 22B active does 22B of work per
    token and needs 235B of memory. Pricing it by total is wrong by an
    order of magnitude; pricing it by active ignores the hardware it
    demands. Speed comes from active, placement comes from the file size.
    """
    assumptions: list[str] = []

    if total_parameters_b <= 0:
        # Derive from the file. Rough, and better than pricing a 70B as
        # though it were free.
        if file_bytes and bits_per_weight:
            total_parameters_b = (file_bytes * 8) / bits_per_weight / 1e9
            assumptions.append(
                "parameter count derived from file size and bits-per-weight; "
                "the tensor table would be exact"
            )
        else:
            total_parameters_b = 7.0
            assumptions.append(
                "no parameter count and no bits-per-weight, so 7B was assumed"
            )

    if active_parameters_b <= 0:
        active_parameters_b = total_parameters_b
        assumptions.append(
            "priced as a dense model; if this is a mixture-of-experts its "
            "active parameter count would make it cheaper"
        )

    if not bits_per_weight and file_bytes and total_parameters_b:
        bits_per_weight = (file_bytes * 8) / (total_parameters_b * 1e9)

    if vram_bytes is None:
        vram_bytes = _vram_bytes()

    # Placement. The single biggest factor, which is why it is decided
    # from the file's real size rather than from a parameter count: what
    # has to fit is the file plus working memory, and 20% for the KV
    # cache and activations is the usual shape of that.
    needed = int(file_bytes * 1.2) if file_bytes else 0
    if vram_bytes and needed and needed <= vram_bytes:
        placement = "gpu"
        hourly = gpu_hourly
        rate = GPU_TOKENS_PER_SECOND_PER_B
    elif vram_bytes and needed and needed <= vram_bytes * 2:
        # Partially offloaded: some layers on the card, the rest in RAM.
        # Between the two, nearer the slow end, because the slowest layer
        # sets the pace.
        placement = "split gpu/cpu"
        hourly = (gpu_hourly + cpu_hourly) / 2
        rate = (GPU_TOKENS_PER_SECOND_PER_B + CPU_TOKENS_PER_SECOND_PER_B * 3) / 4
    else:
        placement = "cpu"
        hourly = cpu_hourly
        rate = CPU_TOKENS_PER_SECOND_PER_B
        if not vram_bytes:
            assumptions.append(
                "no GPU detected on this machine, so CPU serving was assumed"
            )

    # Tokens per second, from the work actually done per token.
    tokens_per_second = max(0.5, rate / max(0.5, active_parameters_b))
    seconds_per_1k = 1000.0 / tokens_per_second
    output_cost = (seconds_per_1k / 3600.0) * hourly
    output_cost = max(FLOOR_PER_1K, output_cost)
    input_cost = max(FLOOR_PER_1K, output_cost * INPUT_COST_RATIO)

    return PriceEstimate(
        input_price_per_1k=input_cost,
        output_price_per_1k=output_cost,
        total_parameters_b=total_parameters_b,
        active_parameters_b=active_parameters_b,
        file_bytes=file_bytes,
        bits_per_weight=bits_per_weight,
        quant=quant,
        placement=placement,
        estimated_tokens_per_second=tokens_per_second,
        vram_bytes=vram_bytes or 0,
        assumptions=assumptions,
    )


def estimate_for_indexed(model: Any, *, vram_bytes: int | None = None) -> PriceEstimate:
    """Estimate for an :class:`~hypernix.t1api.modelindex.IndexedModel`.

    The bridge between what the indexer read off the file and what the
    registry needs written into it.
    """
    return estimate(
        file_bytes=int(getattr(model, "file_bytes", 0) or 0),
        total_parameters_b=float(getattr(model, "parameters_b", 0.0) or 0.0),
        active_parameters_b=float(getattr(model, "active_parameters_b", 0.0) or 0.0),
        bits_per_weight=float(getattr(model, "bits_per_weight", 0.0) or 0.0),
        quant=str(getattr(model, "tier", "") or ""),
        vram_bytes=vram_bytes,
    )
