"""hypernix.quant.hyprslug — quantise a GGUF without llama.cpp.

Also answers to **doomslug**, **doomslugthedestroyer** and **dstd**.

Everything that quantised in this package used to shell out to
``llama-quantize``. That has two problems. The small one is that a
machine which has not built llama.cpp cannot quantise at all. The large
one is that the sub-bit tiers — ``IQ0.9_L``, ``IQ0.75_M``,
``IQ0.5_XXXL`` — are HyperNix types that ``llama-quantize`` has never
heard of, so for those it was never going to be the answer, and what
:mod:`hypernix.quant.steamroller` actually did was copy the staging file
and write a sidecar claiming a tier it had not applied.

hyprslug is the quantiser those tiers needed. It reads a GGUF with
:mod:`hypernix.quant.gguf`, packs each eligible tensor with
:mod:`hypernix.quant.subbit`, and writes a real GGUF whose tensors are
genuinely at the advertised bitrate. No binary, no build, no llama.cpp
on the machine at any point.

What it will and will not touch
-------------------------------
Not every tensor should be crushed. Normalisation weights, biases and
anything one-dimensional are a rounding error of the file size and a
large fraction of the damage, so they are copied at source precision —
the same call every serious quantiser makes, for the same reason.
Token embeddings and the output head are configurable because the right
answer depends on the model: on a small model they dominate the file, on
a large one they do not.

Honesty
-------
A tensor whose element count does not divide into 256 cannot be packed
and is copied instead, and the result says how many were copied. A run
that quietly left half a model at F16 while reporting "IQ0.5" would be
the same failure this module exists to fix.
"""
from __future__ import annotations

import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import llamaquants
from .gguf import GGMLType, GGUFError, GGUFFile, GGUFTensor, GGUFWriter
from .imatrix import expand_for_tensor
from .lowbit import CODECS, LowBitError
from .lowbit import quantize_array as lowbit_quantize
from .subbit import BLOCK_SIZE, PACKINGS, SubBitError, quantize_tensor

logger = logging.getLogger(__name__)

__all__ = [
    "HyprslugError",
    "TIER_TYPES",
    "RECIPES",
    "Recipe",
    "QuantizeReport",
    "quantize_gguf",
    "resolve_recipe",
    "tier_for_packing",
    "ALIASES",
    "RECIPE_ALIASES",
]

#: The names this tool answers to. doomslug is the original; the longer
#: forms are kept because people type them.
ALIASES = ("hyprslug", "doomslug", "doomslugthedestroyer", "dstd")


class HyprslugError(Exception):
    """Quantisation could not proceed."""


#: Steamroller tier -> (GGML type id, subbit packing name).
TIER_TYPES: dict[str, tuple[int, str]] = {
    # Sign-and-scale, from hypernix.quant.subbit. Ordered widest first.
    "INT1": (int(GGMLType.HNX_INT1), "int1_binary"),
    "IQ0.9_L": (int(GGMLType.HNX_IQ0_9), "sign_scale_l"),
    "IQ0.75_M": (int(GGMLType.HNX_IQ0_75), "pair_code_m"),
    "IQ0.5_XXXL": (int(GGMLType.HNX_IQ0_5), "quad_code_xxxl"),
    "IQ0.25_UXL": (int(GGMLType.HNX_IQ0_25), "quarter_code_uxl"),
    # Every sign kept plus per-sub-block magnitude. The only tier here
    # above INT1, and the only one whose rate buys structure rather than
    # more signs -- there are no more signs to buy once all 256 are
    # stored.
    "HNX_1375BIT": (int(GGMLType.HNX_1375), "hnx_1375bit"),
    # Fixed codebook, from hypernix.quant.lowbit. The packing name is the
    # codec name; :func:`_encoder_for` tells the two families apart by
    # looking the name up rather than by parsing it.
    "INT4": (int(GGMLType.HNX_INT4), "INT4"),
    "FP2": (int(GGMLType.HNX_FP2), "FP2"),
}


#: Tensors whose name contains one of these is *never* packed, whatever
#: the recipe says: one-dimensional weights are a rounding error of the
#: file size and a large fraction of the damage.
_ALWAYS_COPY = ("_norm", "norm.", ".bias")


@dataclass(frozen=True)
class Recipe:
    """Which block format each tensor gets, for one named quantisation.

    llama.cpp's "Q4_K_M" is not a block format — Q4_K is. The suffix
    names a *mix*: most tensors at Q4_K, the ones the model leans on
    hardest a step wider. Keeping the two apart is why this is a table
    of policies over :mod:`hypernix.quant.llamaquants` rather than ten
    more encoders.

    ``overrides`` is checked in order, first match wins, against the
    lower-cased tensor name.
    """

    name: str
    base: str
    summary: str
    overrides: tuple[tuple[str, str], ...] = ()
    output: str = ""
    embeddings: str = ""

    def format_for(self, tensor_name: str) -> str:
        """The block format *tensor_name* should be written in."""
        lowered = tensor_name.lower()
        if self.output and (lowered.startswith("output.") or lowered == "output.weight"):
            return self.output
        if self.embeddings and ("token_embd" in lowered or "tok_embeddings" in lowered):
            return self.embeddings
        for fragment, fmt in self.overrides:
            if fragment in lowered:
                return fmt
        return self.base

    @property
    def bits_per_weight(self) -> float:
        return llamaquants.FORMATS[self.base].bits_per_weight


# The tensors llama.cpp widens in a "_M" mix. attn_v and ffn_down are
# the two the perplexity numbers move most on, which is why upstream
# spends the extra bits there and nowhere else.
_M_WIDENS = ("attn_v", "ffn_down")
_L_WIDENS = ("attn_v", "ffn_down", "attn_k")


def _mix(name: str, base: str, wider: str, widens: tuple[str, ...], summary: str) -> Recipe:
    return Recipe(
        name=name,
        base=base,
        summary=summary,
        overrides=tuple((fragment, wider) for fragment in widens),
        output="Q6_K",
    )


#: Every quantisation hyprslug can write, by name.
#:
#: The plain names are single block formats. The ``_S``/``_M``/``_L``
#: names are mixes, and they are *our* reading of what upstream does
#: rather than a byte-for-byte reproduction of its table: llama.cpp
#: picks per layer index as well as per tensor role, and a file that
#: claimed to match it exactly would be claiming something nobody has
#: checked. What is exact is the block encoding of every tensor.
RECIPES: dict[str, Recipe] = {}


def _register(recipe: Recipe) -> None:
    RECIPES[recipe.name] = recipe


for _plain_name in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
    _register(Recipe(
        _plain_name, _plain_name,
        f"{llamaquants.FORMATS[_plain_name].bits_per_weight:.2f} bits per weight, "
        f"every eligible tensor.",
    ))

_register(_mix("Q2_K_S", "Q2_K", "Q3_K", _M_WIDENS,
               "2-bit k-quant, small. The narrowest upstream type worth running."))
_register(_mix("Q3_K_S", "Q3_K", "Q3_K", (),
               "3-bit k-quant, small. Uniform Q3_K with a Q6_K head."))
_register(_mix("Q3_K_M", "Q3_K", "Q4_K", _M_WIDENS,
               "3-bit k-quant, medium."))
_register(_mix("Q3_K_L", "Q3_K", "Q5_K", _L_WIDENS,
               "3-bit k-quant, large. The staging tier every descent passes through."))
_register(_mix("Q4_K_S", "Q4_K", "Q4_K", (),
               "4-bit k-quant, small. Uniform Q4_K with a Q6_K head."))
_register(_mix("Q4_K_M", "Q4_K", "Q6_K", _M_WIDENS,
               "4-bit k-quant, medium. The usual default."))
_register(_mix("Q5_K_S", "Q5_K", "Q5_K", (),
               "5-bit k-quant, small."))
_register(_mix("Q5_K_M", "Q5_K", "Q6_K", _M_WIDENS,
               "5-bit k-quant, medium."))

#: Source types this can read element-wise without help.
#:
#: A quantised source is read through :mod:`hypernix.quant.llamaquants`
#: instead, which is what makes requantisation possible at all: a
#: Q8_0 GGUF is the only copy of the model most people have, and
#: "quantise from the unquantised weights" is advice they cannot take.
_UNQUANTIZED = {
    int(GGMLType.F32): ("<f", 4),
    int(GGMLType.F16): ("<e", 2),
    int(GGMLType.BF16): (None, 2),
}


#: HyperNix extension GGML type -> the codec that decodes it. Built from
#: :data:`TIER_TYPES` so a tier added there is readable here without a
#: second table to keep in step.
def _extension_packing(ggml_type: int) -> str:
    return next(
        (packing for _tier, (kind, packing) in TIER_TYPES.items()
         if kind == int(ggml_type)),
        "",
    )


def _readable(ggml_type: int) -> bool:
    """Whether hyprslug can turn this type back into floats.

    The HyperNix extension types belong here as much as the upstream
    ones. Leaving them out meant a sub-bit GGUF was not a valid *source*:
    :func:`_should_quantize` declined every tensor with "source type 200
    is one hyprslug cannot read" and the run copied them verbatim --
    reporting success while producing a file that still carried type 200.
    Every caller trying to convert a sub-bit model to something stock
    llama.cpp reads got back a file that stock llama.cpp still refuses.
    """
    kind = int(ggml_type)
    return (
        kind in _UNQUANTIZED
        or llamaquants.is_supported(kind)
        or bool(_extension_packing(kind))
    )


def _bits_per_weight(packing: str) -> float:
    """The rate a packing writes, whichever family it belongs to."""
    if packing in PACKINGS:
        return PACKINGS[packing].bits_per_weight
    if packing in CODECS:
        return CODECS[packing].bits_per_weight
    raise HyprslugError(f"No packing named {packing!r}")


def tier_for_packing(packing: str) -> str:
    """The tier name a packing belongs to."""
    for tier, (_, name) in TIER_TYPES.items():
        if name == packing:
            return tier
    raise HyprslugError(f"No tier uses packing {packing!r}")


#: Spellings that name a recipe without resembling it.
#:
#: ``Q4M`` is the one people actually type, and squashing separators does
#: not get there from ``Q4_K_M`` -- the ``K`` is missing, not the
#: underscore. Left unmapped it falls through to "unknown target", which
#: is a confusing way to reject the most common request there is.
RECIPE_ALIASES: dict[str, str] = {
    "Q4M": "Q4_K_M",
    "Q3M": "Q3_K_M",
    "Q5M": "Q5_K_M",
    "Q4S": "Q4_K_S",
    "Q3S": "Q3_K_S",
    "Q5S": "Q5_K_S",
    "Q2S": "Q2_K_S",
    "Q3L": "Q3_K_L",
}


def resolve_recipe(tier: str) -> Recipe | None:
    """The :class:`Recipe` for *tier*, or ``None`` if it is a sub-bit tier.

    Case- and separator-insensitive, because ``q4_k_m``, ``Q4_K_M`` and
    ``q4km`` are all the same request and refusing two of them helps
    nobody. :data:`RECIPE_ALIASES` covers the spellings that drop a
    letter rather than a separator.
    """
    key = (tier or "").strip().upper().replace("-", "_")
    if key in RECIPES:
        return RECIPES[key]
    squashed = key.replace("_", "")
    for name, recipe in RECIPES.items():
        if name.replace("_", "") == squashed:
            return recipe
    aliased = RECIPE_ALIASES.get(squashed)
    return RECIPES[aliased] if aliased else None


def all_targets() -> list[str]:
    """Every name :func:`quantize_gguf` accepts, widest first."""
    return sorted(RECIPES, key=lambda n: -RECIPES[n].bits_per_weight) + list(TIER_TYPES)


def _decode_floats(data: bytes, ggml_type: int) -> list[float]:
    """Tensor bytes to a list of floats, quantised source or not."""
    spec = _UNQUANTIZED.get(int(ggml_type))
    if spec is None:
        if llamaquants.is_supported(int(ggml_type)):
            # Requantising loses whatever the first pass lost -- that is
            # unavoidable and it is not silent: the report says the source
            # was already quantised.
            return [float(v) for v in llamaquants.dequantize_array(data, int(ggml_type))]
        packing = _extension_packing(ggml_type)
        if packing:
            # A HyperNix extension source. Going *up* from one of these
            # recovers nothing the packing threw away -- a 0.9-bit tensor
            # re-encoded as Q4_K_M is a Q4_K_M copy of a 0.9-bit model,
            # not a recovered one -- but it is what makes the file
            # readable by a stock loader at all, and the report's
            # requantized_from field says where it came from.
            if packing in CODECS:
                from .lowbit import dequantize_array as _low_dequantize

                return [float(v) for v in _low_dequantize(data, packing)]
            from .subbit import dequantize_array as _sub_dequantize

            return [float(v) for v in _sub_dequantize(data, packing)]
        raise HyprslugError(
            f"hyprslug reads F32, F16, BF16, the llama.cpp block types and the "
            f"HyperNix extension types; this tensor is type {ggml_type}, which "
            f"is none of them."
        )
    fmt, width = spec
    count = len(data) // width
    if fmt is None:
        # BF16 is the top 16 bits of an F32, so widening is a shift, not
        # a conversion — and unlike F16 it cannot overflow doing it.
        return [
            struct.unpack("<f", b"\x00\x00" + data[i * 2:i * 2 + 2])[0]
            for i in range(count)
        ]
    return list(struct.unpack(f"<{count}{fmt[1]}", data[: count * width]))


def _is_embedding(name: str) -> bool:
    return "token_embd" in name or "tok_embeddings" in name


def _is_output_head(name: str) -> bool:
    return name.startswith("output.") or name == "output.weight"


def _should_quantize(
    tensor: GGUFTensor,
    *,
    block: int,
    quantize_embeddings: bool,
    quantize_output: bool,
) -> tuple[bool, str]:
    """Whether to pack *tensor*, and why not when not."""
    name = tensor.name.lower()
    if len(tensor.shape) < 2:
        return False, "1-D (norm or bias): all of the damage, none of the size"
    if any(fragment in name for fragment in _ALWAYS_COPY):
        return False, "a norm or bias: all of the damage, none of the size"
    if not _readable(int(tensor.ggml_type)):
        return False, f"source type {tensor.ggml_type} is one hyprslug cannot read"
    # The *row* length, not the element count. GGML quantises row by
    # row, so the constraint is on ne[0] -- and shape[0] is ne[0],
    # read straight from the file in GGUF's own order.
    #
    # Checking the total instead let a real model through and produced a
    # file llama.cpp refuses to load:
    #
    #     gguf_init_from_reader: tensor 'blk.0.ssm_conv1d.weight' of
    #     type 202 (IQ0.5_XXXL) has 4 elements per row, not a multiple
    #     of block size (256)
    #
    # An SSM convolution weight is [4, N]: four elements per row, and a
    # total of 4N that divides into 256 whenever N does. The total-count
    # check passed, the tensor was packed, and the model would not load.
    #
    # The row check subsumes the total: if ne[0] divides into the block
    # size then so does the product. It also happens to be why packing
    # the flattened array is safe -- with ne[0] a multiple of the block,
    # no block ever straddles two rows.
    row = int(tensor.shape[0])
    if row % block:
        return False, (
            f"{row} elements per row do not divide into {block}-element blocks"
        )
    if not quantize_embeddings and _is_embedding(name):
        return False, "token embeddings (pass quantize_embeddings=True to include)"
    if not quantize_output and _is_output_head(name):
        return False, "output head (pass quantize_output=True to include)"
    return True, ""


@dataclass
class QuantizeReport:
    """What a run actually did — including what it declined to do."""

    tier: str = ""
    packing: str = ""
    formats: dict[str, int] = field(default_factory=dict)
    requantized_from: dict[str, int] = field(default_factory=dict)
    source_bytes: int = 0
    output_bytes: int = 0
    tensors_total: int = 0
    tensors_quantized: int = 0
    tensors_copied: int = 0
    elements_quantized: int = 0
    elements_copied: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def compression(self) -> float:
        return (self.source_bytes / self.output_bytes) if self.output_bytes else 0.0

    @property
    def quantized_fraction(self) -> float:
        total = self.elements_quantized + self.elements_copied
        return (self.elements_quantized / total) if total else 0.0

    @property
    def elements_total(self) -> int:
        return self.elements_quantized + self.elements_copied

    @property
    def effective_bits_per_weight(self) -> float:
        """What the *file* costs per weight, not what the tier packs at.

        The number the tier is named for describes the tensors it packs.
        It says nothing about the ones left alone -- and on a model with
        a large vocabulary those are most of the file. A Qwen3-class 2B
        has a 151,936-token vocabulary, so `token_embd` and `output` are
        622M of its 2.03B parameters; left at BF16 they are 1.24 GB
        before a single packed tensor is written, and `IQ0.5_XXXL`
        produces a 1.4 GB file at 5.3 bits per weight. Picking a
        different tier barely moves it.

        Reporting only the tier's rate is how somebody spends an hour
        quantising and gets a file ten times the size they asked for,
        with nothing anywhere saying why.
        """
        return (self.output_bytes * 8 / self.elements_total) if self.elements_total else 0.0

    @property
    def tier_bits_per_weight(self) -> float:
        """The rate the packing writes, for the tensors it touched."""
        return _bits_per_weight(self.packing) if self.packing else 0.0

    @property
    def name_is_misleading(self) -> bool:
        """True when the file costs far more per weight than its tier.

        1.5x is deliberately generous: norms and biases are always
        copied and always small, so a little overshoot is the design
        working. Ten times over is the embedding table, and worth
        shouting about.
        """
        tier = self.tier_bits_per_weight
        return bool(tier) and self.effective_bits_per_weight > tier * 1.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "packing": self.packing,
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "compression": round(self.compression, 3),
            "tensors_total": self.tensors_total,
            "tensors_quantized": self.tensors_quantized,
            "tensors_copied": self.tensors_copied,
            "formats": dict(sorted(self.formats.items())),
            "requantized_from": dict(sorted(self.requantized_from.items())),
            "quantized_fraction": round(self.quantized_fraction, 4),
            "effective_bits_per_weight": round(self.effective_bits_per_weight, 3),
            "tier_bits_per_weight": round(self.tier_bits_per_weight, 3),
            "name_is_misleading": self.name_is_misleading,
            "skipped": [{"tensor": n, "reason": r} for n, r in self.skipped],
            "seconds": round(self.seconds, 2),
        }

    def describe(self) -> str:
        lines = [
            f"{self.tier}  ({self.packing})",
            f"  {self.source_bytes / 1e6:.1f} MB -> {self.output_bytes / 1e6:.1f} MB "
            f"({self.compression:.1f}x)",
            f"  {self.tensors_quantized}/{self.tensors_total} tensors packed, "
            f"{self.quantized_fraction * 100:.1f}% of weights",
            f"  {self.effective_bits_per_weight:.2f} bits/weight over the whole file "
            f"({self.tier_bits_per_weight:.3f} where it packed)",
        ]
        if self.name_is_misleading:
            over = self.effective_bits_per_weight / self.tier_bits_per_weight
            lines.append("")
            lines.append(
                f"  ! This file costs {over:.0f}x what the tier name suggests."
            )
            lines.append(
                "    The tensors left at source precision are most of it -- on a"
            )
            lines.append(
                "    large vocabulary, token_embd and output alone can be a third"
            )
            lines.append(
                "    of the parameters and nearly all of the bytes."
            )
            lines.append("    To get the size the tier is named for:")
            lines.append("      --quantize-embeddings --quantize-output")
            lines.append("")
        if len(self.formats) > 1:
            mix = ", ".join(f"{fmt} x{count}" for fmt, count in sorted(self.formats.items()))
            lines.append(f"  mix: {mix}")
        if self.requantized_from:
            # Requantising compounds the first pass's error. Whether that
            # matters is the operator's call; whether they get to make it
            # is not.
            was = ", ".join(
                f"{fmt} x{count}" for fmt, count in sorted(self.requantized_from.items())
            )
            lines.append(f"  ! requantised from an already-quantised source: {was}")
        if self.tensors_copied:
            lines.append(f"  {self.tensors_copied} copied at source precision:")
            for name, reason in self.skipped[:5]:
                lines.append(f"    {name}: {reason}")
            if len(self.skipped) > 5:
                lines.append(f"    ... and {len(self.skipped) - 5} more")
        return "\n".join(lines)


def load_imatrix(path: str | Path) -> dict[str, list[float]]:
    """Read an importance matrix, keyed by tensor name.

    Both formats: llama.cpp's binary ``.imatrix`` — the one people share
    — and JSON. :mod:`hypernix.quant.imatrix` decides which by content
    rather than by suffix, because people rename these files.
    """
    from .imatrix import Imatrix, ImatrixError

    try:
        return Imatrix.load(path).to_simple_dict()
    except ImatrixError as exc:
        raise HyprslugError(f"Could not read imatrix {path}: {exc}") from exc


def quantize_gguf(
    source: str | Path,
    destination: str | Path,
    tier: str,
    *,
    imatrix: str | Path | dict[str, list[float]] | None = None,
    quantize_embeddings: bool | None = None,
    quantize_output: bool | None = None,
    progress: Callable[[dict], None] | None = None,
) -> QuantizeReport:
    """Quantise *source* to *tier*, writing *destination*.

    *tier* is either a llama.cpp quantisation — a block format like
    ``Q4_K`` or a mix like ``Q4_K_M`` — or one of the HyperNix sub-bit
    tiers in :data:`TIER_TYPES`. Returns a :class:`QuantizeReport`
    describing what was packed and what was not. Nothing here invokes
    llama.cpp, at any point, for any tier.

    *quantize_embeddings* and *quantize_output* default to what the
    target implies: a llama.cpp mix quantises both (with the head a step
    wider, as upstream does), and a sub-bit tier leaves both alone,
    because at half a bit the embedding table is the model.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    recipe = resolve_recipe(tier)
    packing = ""
    ggml_type = 0
    if recipe is None:
        if tier not in TIER_TYPES:
            raise HyprslugError(
                f"Unknown target {tier!r}. hyprslug writes: {', '.join(all_targets())}"
            )
        ggml_type, packing = TIER_TYPES[tier]
        if packing not in PACKINGS and packing not in CODECS:
            raise HyprslugError(
                f"Tier {tier} names packing {packing!r}, which does not exist."
            )
    if quantize_embeddings is None:
        quantize_embeddings = recipe is not None
    if quantize_output is None:
        quantize_output = recipe is not None
    if not source_path.exists():
        raise HyprslugError(f"No such model: {source_path}")

    weights_by_tensor: dict[str, list[float]] = {}
    if isinstance(imatrix, dict):
        weights_by_tensor = imatrix
    elif imatrix is not None:
        weights_by_tensor = load_imatrix(imatrix)

    started = time.time()
    try:
        model = GGUFFile.read(source_path)
    except GGUFError as exc:
        raise HyprslugError(f"{source_path}: {exc}") from exc

    report = QuantizeReport(
        tier=recipe.name if recipe else tier,
        packing=packing,
        source_bytes=source_path.stat().st_size,
        tensors_total=len(model.tensors),
    )

    writer = GGUFWriter(destination_path, alignment=model.alignment)
    writer.copy_metadata_from(model)
    # Recorded in the file itself, not a sidecar. A sidecar can be lost in
    # a copy, and then nothing about the model says what was done to it.
    writer.set_metadata("hypernix.quantiser", "hyprslug")
    writer.set_metadata("hypernix.tier", report.tier)
    writer.set_metadata("hypernix.imatrix", bool(weights_by_tensor))
    if recipe is None:
        writer.set_metadata("hypernix.packing", packing)
        writer.set_metadata("hypernix.sub_bit", True)
        writer.set_metadata(
            "general.file_type_description",
            f"HyperNix {tier} ({_bits_per_weight(packing):.3f} bpw)",
        )
    else:
        writer.set_metadata("hypernix.sub_bit", False)
        writer.set_metadata("hypernix.base_format", recipe.base)
        writer.set_metadata(
            "general.file_type_description",
            f"{recipe.name} ({recipe.bits_per_weight:.2f} bpw base) via hyprslug",
        )

    # (original tensor, declared tensor, block format or "" for a copy).
    plan: list[tuple[GGUFTensor, GGUFTensor, str]] = []
    for tensor in model.tensors:
        chosen = recipe.format_for(tensor.name) if recipe else ""
        block = llamaquants.FORMATS[chosen].block if recipe else BLOCK_SIZE
        do_it, reason = _should_quantize(
            tensor,
            block=block,
            quantize_embeddings=quantize_embeddings,
            quantize_output=quantize_output,
        )
        if not do_it:
            chosen = ""
        if do_it and recipe:
            target_type = llamaquants.FORMATS[chosen].ggml_type
        elif do_it:
            target_type = ggml_type
        else:
            target_type = tensor.ggml_type
        declared = writer.add_tensor(tensor.name, tensor.shape, target_type)
        plan.append((tensor, declared, chosen if recipe else ("sub-bit" if do_it else "")))
        if do_it:
            report.tensors_quantized += 1
            report.elements_quantized += tensor.elements
            label = chosen if recipe else tier
            report.formats[label] = report.formats.get(label, 0) + 1
            if int(tensor.ggml_type) not in _UNQUANTIZED:
                was = GGMLType(int(tensor.ggml_type)).name
                report.requantized_from[was] = report.requantized_from.get(was, 0) + 1
        else:
            report.tensors_copied += 1
            report.elements_copied += tensor.elements
            report.skipped.append((tensor.name, reason))

    by_name = {declared.name: (original, fmt) for original, declared, fmt in plan}
    done = 0

    def _data_for(declared: GGUFTensor) -> bytes:
        nonlocal done
        original, fmt = by_name[declared.name]
        raw = model.tensor_bytes(original)
        done += 1
        if progress is not None:
            try:
                progress({
                    "event": "tensor",
                    "name": declared.name,
                    "index": done,
                    "total": len(plan),
                    "quantized": bool(fmt),
                    "format": fmt,
                })
            except Exception:  # noqa: BLE001 - a listener must not fail the run
                logger.debug("hyprslug: progress callback raised", exc_info=True)
        if not fmt:
            return raw
        values = _decode_floats(raw, original.ggml_type)
        importance = weights_by_tensor.get(declared.name)
        if importance is not None:
            # An imatrix carries one number per *input channel*; a
            # quantiser wants one per weight, and a GGUF weight tensor is
            # rows of n_input elements, so the vector tiles. Where the two
            # cannot be reconciled the imatrix is a different model's, and
            # applying it would weight the wrong positions -- worse than
            # not applying it, so say so and carry on without it.
            expanded = expand_for_tensor(importance, len(values))
            if expanded is None:
                logger.warning(
                    "hyprslug: imatrix for %s has %d entries, which does not divide "
                    "the tensor's %d; ignoring it",
                    declared.name, len(importance), len(values),
                )
            importance = expanded
        if fmt == "sub-bit":
            try:
                if packing in CODECS:
                    # A fixed codebook carries its own magnitude, so an
                    # imatrix has nothing to decide here -- there is no
                    # scale to steer and no sign to drop. Passing one
                    # would be accepting an argument and ignoring it.
                    return lowbit_quantize(values, packing)
                return quantize_tensor(values, packing, importance)
            except (SubBitError, LowBitError) as exc:
                raise HyprslugError(f"{declared.name}: {exc}") from exc
        try:
            return llamaquants.quantize_array(values, fmt, importance)
        except llamaquants.LlamaQuantError as exc:
            raise HyprslugError(f"{declared.name}: {exc}") from exc

    try:
        writer.write(_data_for)
    except (GGUFError, OSError) as exc:
        raise HyprslugError(f"Could not write {destination_path}: {exc}") from exc

    report.output_bytes = destination_path.stat().st_size
    report.seconds = time.time() - started
    if progress is not None:
        try:
            progress({"event": "done", **report.to_dict()})
        except Exception:  # noqa: BLE001
            logger.debug("hyprslug: progress callback raised", exc_info=True)
    return report
