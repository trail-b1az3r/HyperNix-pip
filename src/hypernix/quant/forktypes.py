"""The GGML types only a llama.cpp fork knows, and which fork runs them.

The hnx llama mod (ggml-hnx's patch, :mod:`hypernix.quant.runtime_bridge`)
reads stock types and HyperNix's own (200-211, 224-225). A GGUF quantised
with ik_llama.cpp's IQ*_K, IQ*_KS, the trellis IQ*_KT or its row-repacked
*_R4/_R8 types loads in neither: stock llama.cpp and the patched build
both stop at "invalid ggml type". These are the types
:mod:`hypernix.models.multilama` can run, read from each fork's
``ggml/include/ggml.h`` (ik_llama.cpp ``main`` and PrismML-Eng/llama.cpp
``master``, October 2026), so the runner can tell what a file needs and
start the fork that has it instead of failing.

**Ids that mean two things.** ik_llama.cpp numbers its repacked types
202-233, and HyperNix took 200-211 first: 202, 206, 208, 210 and 211 are
IQ0.5_XXXL, FP2, INT8, INT3 and FP8 here and Q4_0_R8, Q5_0_R4, Q8_0_R8,
Q2_K_R4 and Q3_K_R4 there. Those ids cannot be renumbered (they are in
every file hyprslug has written), so :func:`which_runtime` settles them
by what else is in the file: an ik_llama.cpp file carries ik-only types
alongside, and otherwise the ids are read as HyperNix's, as they always
have been here (``--backend multilama-ik`` says otherwise). HyperNix's types
added since (224, 225) sit in a gap ik_llama.cpp leaves free.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AMBIGUOUS",
    "IK_TYPES",
    "UPSTREAM_COUNT",
    "Verdict",
    "fork_type_name",
    "which_runtime",
]

#: Every id below this is upstream llama.cpp's (GGML_TYPE_COUNT = 43).
UPSTREAM_COUNT = 43

#: ik_llama.cpp's types that upstream does not have, by id.
IK_TYPES: dict[int, str] = {
    97: "Q8_0_X4", 98: "Q8_1_X4", 99: "Q8_2_X4",
    133: "Q6_0", 134: "IQ1_BN", 135: "IQ2_BN", 136: "Q8_K64",
    137: "IQ2_K", 138: "IQ3_K", 139: "IQ4_K", 140: "IQ5_K", 141: "IQ6_K",
    142: "PQ2_0", 143: "PTQ1_0", 144: "IQ4_KS", 145: "IQ2_KS", 146: "IQ4_KSS",
    147: "Q8_K16", 148: "Q8_K32", 149: "Q8_KR8", 150: "Q8_K128", 151: "Q8_KV",
    152: "IQ5_KS", 153: "IQ2_KT", 154: "IQ3_KT", 155: "IQ4_KT", 156: "IQ3_KS",
    157: "IQ2_KL", 158: "IQ1_KT", 159: "Q1_0_G128_R8", 160: "PQ2_0_R8", 161: "PTQ1_0_R8",
    202: "Q4_0_R8", 206: "Q5_0_R4", 208: "Q8_0_R8", 210: "Q2_K_R4", 211: "Q3_K_R4",
    212: "Q4_K_R4", 213: "Q5_K_R4", 214: "Q6_K_R4", 216: "IQ2_XXS_R4", 217: "IQ2_XS_R4",
    218: "IQ3_XXS_R4", 219: "IQ1_S_R4", 220: "IQ4_NL_R4", 221: "IQ3_S_R4", 222: "IQ2_S_R4",
    223: "IQ4_XS_R8", 229: "IQ1_M_R4", 230: "BF16_R16", 233: "Q6_0_R4",
    335: "IQ2_BN_R4", 337: "IQ2_K_R4", 338: "IQ3_K_R4", 339: "IQ4_K_R4", 340: "IQ5_K_R4",
    344: "IQ4_KS_R4", 345: "IQ4_KS_R16", 346: "IQ3_KS_R16", 352: "IQ5_KS_R4",
    353: "MXFP4_R8", 397: "Q8_K_R16", 398: "Q8_KV_R8", 399: "Q8_K_R8",
}

#: HyperNix's own ids, as :data:`hypernix.quant.ggufcheck.LLAMA_CPP_REGISTERED_TYPES`.
_HNX = frozenset({200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 224, 225})

#: Ids both HyperNix and ik_llama.cpp use, for different types.
AMBIGUOUS = frozenset(_HNX & set(IK_TYPES))

#: Architectures only a fork's loader knows.
_FORK_ARCHITECTURES = {"nanbeige": "nanbeige"}


def fork_type_name(ggml_type: int) -> str:
    """ik_llama.cpp's name for *ggml_type*, or ``""``."""
    return IK_TYPES.get(int(ggml_type), "")


@dataclass(frozen=True)
class Verdict:
    """What a GGUF needs: ``hnx`` (the patched build), ``stock``, or a
    multilama fork (``ik``, ``nanbeige``), and why."""

    runtime: str
    fork_types: tuple[str, ...] = ()
    why: str = ""

    @property
    def needs_fork(self) -> bool:
        return self.runtime not in ("hnx", "stock")


def which_runtime(types: Iterable[int], metadata: Mapping[str, Any] | None = None) -> Verdict:
    """Which runtime can load a GGUF with these tensor *types*."""
    metadata = metadata or {}
    found = {int(t) for t in types}
    architecture = str(metadata.get("general.architecture") or "").lower()
    if architecture in _FORK_ARCHITECTURES:
        fork = _FORK_ARCHITECTURES[architecture]
        return Verdict(fork, why=f"general.architecture={architecture} is only in the {fork} fork")
    written_by_hyprslug = any(str(key).startswith("hypernix.") for key in metadata)
    hnx = found & _HNX
    ik_only = {t for t in found if t in IK_TYPES and t not in AMBIGUOUS}
    if ik_only:
        names = tuple(sorted(IK_TYPES[t] for t in found if t in IK_TYPES))
        if hnx - AMBIGUOUS:
            return Verdict("", names, "it mixes HyperNix types with ik_llama.cpp's; nothing loads both")
        return Verdict("ik", names, f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} "
                                     f"ik_llama.cpp's")
    if hnx:
        # The shared ids alone are read as HyperNix's, as they always
        # have been here; `--backend multilama-ik` sends such a file to
        # ik_llama.cpp instead.
        why = ("HyperNix types: the patched build (hnx llama mod)" if written_by_hyprslug or hnx - AMBIGUOUS
               else "ids HyperNix and ik_llama.cpp share, read as HyperNix's "
                    "(--backend multilama-ik if it is an ik_llama.cpp repack)")
        return Verdict("hnx", why=why)
    unknown = sorted(t for t in found if t >= UPSTREAM_COUNT)
    if unknown:
        return Verdict("", why=f"type ids {unknown} are not known to any runtime here")
    return Verdict("stock", why="only upstream llama.cpp types")
