"""hyperlink.ondevice — will this model actually run on that phone?

HyperLink can download a GGUF from Hugging Face and run it with no
server involved. The hard part is not the running; it is answering
"will this one work?" *before* a 4 GB download, and being right.

Getting it wrong is expensive in both directions. Too optimistic and the
phone downloads for twenty minutes on cellular and is then killed by the
OS partway through the first reply. Too pessimistic and a model that
would have run fine is hidden from a list.

Three things make the obvious answer wrong
------------------------------------------

**Total RAM is not the budget.** This is the big one. iOS does not let an
app use the RAM the marketing page advertises: each process gets a
jetsam limit, and exceeding it is not a swap or a slowdown, it is an
immediate kill with no exception to catch. On an 8 GB iPhone that limit
is commonly in the 2-3 GB range for a normal app. A 7B model at Q4_K_M
is about 4.4 GB of weights, so "8 GB device, 4.4 GB model, fits" is
wrong by a factor that ends the app. The number that matters is
``os_proc_available_memory()``, which is what this module wants in
:attr:`DeviceBudget.available_bytes`. The
``com.apple.developer.kernel.increased-memory-limit`` entitlement raises
the ceiling, and Apple grants it selectively; :attr:`has_increased_limit`
is recorded for reporting, never used to inflate an estimate.

**The KV cache is not small.** It scales with context length, and at
long contexts it can exceed the weights. A 7B model with 32 layers and
8 KV heads at 128 head-dim costs about 0.5 MB per token at fp16, so a
32k context is roughly 16 GB — an order of magnitude more than the
quantised weights. Any planner that sizes only the weights will be
badly wrong exactly when someone tries to use the long context they
chose the model for. :func:`largest_context` exists because the useful
question is usually not "does it fit" but "how much context can I have".

**On a phone, GPU offload does not reduce memory.** On a desktop with a
discrete card, moving layers to the GPU moves them out of system RAM.
Apple silicon has unified memory: a Metal buffer and a malloc come from
the same pool and count against the same limit. Offloading to Metal is
worth doing for speed and does nothing whatever for fit, and a planner
that subtracts offloaded layers from the memory estimate will approve
models that cannot run.

A note on the Neural Engine
---------------------------
The ANE cannot run a GGUF model. It is reachable only through Core ML,
and llama.cpp has no Core ML backend for LLM inference — its Apple
backend is Metal, with Accelerate on the CPU path. Converting a model to
Core ML is a different format, a different toolchain, and a different
file; it is not a switch. :class:`ComputeBackend` therefore offers what
actually exists, and :data:`ANE_EXPLANATION` is the honest answer to give
someone looking for the option.

Accuracy
--------
These are estimates with real error bars, not measurements. Weight sizes
are within a few percent because they come from the quantisation
arithmetic. The KV cache is exact given the right shape metadata. The
runtime overhead term is the rough one: it covers llama.cpp's compute
buffers, the tokeniser, and the app itself, and is modelled as a floor
plus a per-context term. :data:`SAFETY_MARGIN` keeps a "runs" verdict
away from the cliff, because the cost of being 10% optimistic here is a
process kill.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..quant.formats import FORMATS, QuantFormat

__all__ = [
    "ANE_EXPLANATION",
    "BASE_OVERHEAD_BYTES",
    "FORMATS",
    "OVERHEAD_BYTES_PER_CONTEXT_TOKEN",
    "TIGHT_MARGIN",
    "ComputeBackend",
    "DeviceBudget",
    "FitPlan",
    "GGUFCandidate",
    "ModelShape",
    "SAFETY_MARGIN",
    "SIZE_SAFETY_FACTOR",
    "SUPPLEMENTARY_GGUF_BITS",
    "Verdict",
    "choose_candidate",
    "kv_cache_bytes",
    "largest_context",
    "plan",
    "quant_bits",
    "quant_from_filename",
    "runtime_overhead_bytes",
]

MIB = 1024 * 1024
GIB = 1024 * MIB

#: Fraction of the available budget a plan is allowed to use before it
#: stops calling itself "runs". Being 10% optimistic on a desktop means
#: swapping; here it means the OS kills the process mid-sentence.
SAFETY_MARGIN = 0.85

#: Below this, a plan is called "tight" rather than refused: it will
#: probably work and may be killed under memory pressure from elsewhere.
TIGHT_MARGIN = 0.95

#: llama.cpp's fixed costs plus the app itself: tokeniser, sampler,
#: graph, and the SwiftUI process that hosts it.
BASE_OVERHEAD_BYTES = 320 * MIB

#: Compute buffers scale with the batch and the context. This is a
#: deliberately coarse per-token figure covering logits, the graph's
#: scratch space and Metal's staging buffers.
OVERHEAD_BYTES_PER_CONTEXT_TOKEN = 2048

#: Effective bits per weight for GGUF quantisations that appear on
#: Hugging Face but are not in ``hypernix.quant.formats.FORMATS``.
#:
#: FORMATS is a curated registry of formats worth recommending; this is
#: the wider set of names that actually turn up in a file listing.
#: ``Q4_K_S`` alone is on thousands of repositories, and a planner that
#: cannot size it has to refuse or guess.
SUPPLEMENTARY_GGUF_BITS: dict[str, float] = {
    "Q2_K": 3.35,
    "Q3_K_S": 3.50,
    "Q3_K_M": 3.91,
    "Q4_0": 4.55,
    "Q4_1": 5.06,
    "Q4_K_S": 4.58,
    "Q5_0": 5.54,
    "Q5_1": 6.06,
    "Q5_K_S": 5.52,
    "IQ1_S": 1.56,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.50,
    "IQ3_XXS": 3.06,
    "IQ3_XS": 3.30,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    "IQ4_NL": 4.50,
    "F16": 16.0,
    "FP16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
    "FP32": 32.0,
}

#: The precision llama.cpp keeps the output/embedding tensors at inside
#: a k-quant file, regardless of the name on the tin. This is why a
#: "4-bit" 1B model is 0.81 GB and not 0.75 GB: at that size the
#: embedding table is a fifth of the parameters, so the exception
#: dominates the rule.
EMBEDDING_TENSOR_BITS = 6.56

#: Applied after the tensor arithmetic. Covers what the model above
#: still does not: per-tensor scale metadata, the GGUF header and
#: tokeniser, and llama.cpp's freedom to vary the mixed-precision policy
#: between releases. Erring high is not symmetric with erring low --
#: over-estimating hides a model that would have run, under-estimating
#: gets the process killed after a multi-gigabyte download.
SIZE_SAFETY_FACTOR = 1.02

#: What a phone can actually do with a GGUF.
ANE_EXPLANATION = (
    "The Neural Engine cannot run GGUF models. It is reachable only "
    "through Core ML, and llama.cpp's Apple backend is Metal (GPU) with "
    "Accelerate on the CPU path — there is no Core ML backend for LLM "
    "inference. Running on the ANE would mean converting the model to "
    "Core ML, which is a different file in a different format, not a "
    "setting. Metal is the acceleration that exists here, and on Apple "
    "silicon it is fast."
)


class ComputeBackend:
    """Backends that exist for GGUF on an iPhone.

    Strings rather than an enum because they cross a JSON boundary to
    Swift and back, and a value from a newer server must survive the
    round trip rather than fail a decode.
    """

    #: Accelerate/AMX on the performance cores.
    CPU = "cpu"
    #: The GPU, via ggml-metal. Shares system memory with everything
    #: else, so it buys speed and not headroom.
    METAL = "metal"
    #: Metal for most layers, CPU for the rest. What a model that nearly
    #: fits in the GPU's working set wants.
    HYBRID = "hybrid"

    ALL = frozenset({CPU, METAL, HYBRID})


class Verdict:
    RUNS = "runs"
    TIGHT = "tight"
    NO = "no"


@dataclass(frozen=True)
class DeviceBudget:
    """What this process may actually use, in bytes.

    ``available_bytes`` must come from ``os_proc_available_memory()``,
    not ``ProcessInfo.physicalMemory``. The second is the device's RAM
    and is not the question — see the module docstring.
    """

    available_bytes: int
    total_ram_bytes: int = 0
    has_increased_limit: bool = False
    #: Performance cores the user has allowed. 0 means "decide for me".
    cpu_cores: int = 0
    device_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_bytes": self.available_bytes,
            "total_ram_bytes": self.total_ram_bytes,
            "has_increased_limit": self.has_increased_limit,
            "cpu_cores": self.cpu_cores,
            "device_model": self.device_model,
        }


@dataclass(frozen=True)
class ModelShape:
    """Everything needed to size a model, from its GGUF metadata.

    ``kv_heads`` is the *key/value* head count, which for a
    grouped-query model is far smaller than the attention head count and
    is what the cache is actually sized by. Using the attention head
    count instead overestimates the cache by the GQA ratio — 4x on Llama
    3, 8x on some others — and turns a model that runs into one the
    planner refuses.
    """

    parameters: int
    quant: str
    layers: int
    kv_heads: int
    head_dim: int
    train_context: int = 0
    file_bytes: int = 0
    name: str = ""
    #: From the GGUF metadata. Optional because a search listing does
    #: not carry them; supplying them makes the size estimate markedly
    #: better for small models, where the embedding table dominates.
    vocab_size: int = 0
    embedding_dim: int = 0
    tied_embeddings: bool = False

    @property
    def quant_format(self) -> QuantFormat | None:
        return _lookup_quant(self.quant)

    def weights_bytes(self) -> int:
        """Bytes of weights, from the file where possible.

        The file size is the truth when it is known. Falling back to
        arithmetic is for a candidate seen in a search listing that has
        not been downloaded — which is exactly when the answer matters,
        since the point is to decide *before* the download.

        The arithmetic is not simply bits times parameters. llama.cpp
        keeps the embedding and output tensors at a higher precision
        than the file's name suggests, and for a small model those
        tensors are a large fraction of it: Llama-3.2-1B has a 128k
        vocabulary over 2048 dimensions, which is 21% of its
        parameters. Sizing that model at a flat 4.83 bits under-counts
        by 8%, and under-counting is the direction that gets the process
        killed. Given a vocabulary and dimension, the exception is
        priced separately; without them a conservative factor stands in.
        """
        if self.file_bytes > 0:
            return self.file_bytes
        bits = quant_bits(self.quant)
        if bits <= 0 or self.parameters <= 0:
            return 0

        if self.vocab_size > 0 and self.embedding_dim > 0 and bits < EMBEDDING_TENSOR_BITS:
            table = self.vocab_size * self.embedding_dim
            # An untied model stores the output projection separately;
            # a tied one reuses the embedding table for both.
            special = table * (1 if self.tied_embeddings else 2)
            special = min(special, self.parameters)
            rest = self.parameters - special
            total_bits = special * EMBEDDING_TENSOR_BITS + rest * bits
        else:
            total_bits = self.parameters * bits
        return int(total_bits / 8 * SIZE_SAFETY_FACTOR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "quant": self.quant,
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "train_context": self.train_context,
            "file_bytes": self.file_bytes,
        }


@dataclass
class FitPlan:
    verdict: str
    context: int
    weights_bytes: int
    kv_bytes: int
    overhead_bytes: int
    total_bytes: int
    available_bytes: int
    backend: str = ComputeBackend.METAL
    gpu_layers: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def headroom_bytes(self) -> int:
        return self.available_bytes - self.total_bytes

    @property
    def utilisation(self) -> float:
        if self.available_bytes <= 0:
            return float("inf")
        return self.total_bytes / self.available_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "context": self.context,
            "weights_bytes": self.weights_bytes,
            "kv_bytes": self.kv_bytes,
            "overhead_bytes": self.overhead_bytes,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "headroom_bytes": self.headroom_bytes,
            "utilisation": round(self.utilisation, 4),
            "backend": self.backend,
            "gpu_layers": self.gpu_layers,
            "warnings": self.warnings,
            "notes": self.notes,
        }


def _lookup_quant(name: str) -> QuantFormat | None:
    """Find a format by name or alias, case-insensitively."""
    if not name:
        return None
    wanted = name.strip().lower()
    for fmt in FORMATS.values():
        if fmt.name.lower() == wanted or wanted in {a.lower() for a in fmt.aliases}:
            return fmt
    return None


def quant_bits(name: str) -> float:
    """Effective bits per weight for a GGUF quantisation name.

    Checks the curated registry first, then the supplementary table of
    names that only ever appear in file listings. Returns 0.0 for a name
    neither knows, which callers must treat as "cannot size this"
    rather than as zero bytes.
    """
    fmt = _lookup_quant(name)
    if fmt is not None:
        return fmt.effective_bits
    return SUPPLEMENTARY_GGUF_BITS.get((name or "").strip().upper(), 0.0)


# Longest first, so `Q4_K_M` is not matched as `Q4_K`. A shorter alias
# winning would label a file with a format whose bits-per-weight is
# wrong, and the whole estimate follows from that number.
_QUANT_NAMES = sorted(
    {f.name for f in FORMATS.values()}
    | {a.upper() for f in FORMATS.values() for a in f.aliases}
    | set(SUPPLEMENTARY_GGUF_BITS),
    key=len,
    reverse=True,
)
_QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in _QUANT_NAMES) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def quant_from_filename(filename: str) -> str:
    """The quantisation a GGUF filename advertises, or "".

    Hugging Face GGUF repositories name files by convention rather than
    by rule — ``Meta-Llama-3-8B-Instruct.Q4_K_M.gguf``,
    ``llama-3-8b-instruct-q4_k_m.gguf``, ``model-IQ2_M.gguf`` — so this
    matches the known format names as whole tokens rather than trying to
    parse a structure that does not exist.
    """
    match = _QUANT_IN_NAME.search(filename or "")
    if match is None:
        return ""
    fmt = _lookup_quant(match.group(1))
    return fmt.name if fmt else match.group(1).upper()


def _bits_for_candidate(quant: str) -> float:
    return quant_bits(quant)


@dataclass(frozen=True)
class GGUFCandidate:
    """One downloadable file from a Hugging Face search result."""

    repo_id: str
    filename: str
    size_bytes: int = 0
    quant: str = ""
    sha256: str = ""

    @property
    def resolved_quant(self) -> str:
        return self.quant or quant_from_filename(self.filename)

    @property
    def download_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "quant": self.resolved_quant,
            "sha256": self.sha256,
            "download_url": self.download_url,
        }


def kv_cache_bytes(shape: ModelShape, context: int, *, kv_bits: int = 16) -> int:
    """Bytes the key/value cache needs for *context* tokens.

    ``2`` for the two caches, K and V. Sized by the *key/value* head
    count, not the attention head count: a grouped-query model shares
    each KV head across several attention heads, so using the larger
    number overestimates by the GQA ratio.

    ``kv_bits`` of 8 is llama.cpp's quantised cache, which roughly halves
    this at a small quality cost and is often what makes a long context
    possible at all on a phone.
    """
    if context <= 0 or shape.layers <= 0 or shape.kv_heads <= 0 or shape.head_dim <= 0:
        return 0
    per_token = 2 * shape.layers * shape.kv_heads * shape.head_dim * (kv_bits / 8)
    return int(per_token * context)


def runtime_overhead_bytes(context: int) -> int:
    """llama.cpp's compute buffers plus the app hosting it."""
    return BASE_OVERHEAD_BYTES + max(0, context) * OVERHEAD_BYTES_PER_CONTEXT_TOKEN


def plan(
    shape: ModelShape,
    budget: DeviceBudget,
    *,
    context: int = 4096,
    kv_bits: int = 16,
    backend: str = ComputeBackend.METAL,
) -> FitPlan:
    """Decide whether *shape* runs on *budget*, and say why not."""
    if backend not in ComputeBackend.ALL:
        backend = ComputeBackend.METAL

    context = max(0, int(context))
    weights = shape.weights_bytes()
    kv = kv_cache_bytes(shape, context, kv_bits=kv_bits)
    overhead = runtime_overhead_bytes(context)
    total = weights + kv + overhead

    warnings: list[str] = []
    notes: list[str] = []

    if budget.available_bytes <= 0:
        warnings.append(
            "The device did not report an available-memory figure, so this "
            "is unchecked. Ask os_proc_available_memory() rather than "
            "assuming the device's RAM."
        )

    # The comparison people expect, and the reason it is not used.
    if budget.total_ram_bytes and budget.available_bytes:
        if total <= budget.total_ram_bytes and total > budget.available_bytes:
            notes.append(
                f"This model fits the device's {_gib(budget.total_ram_bytes)} of "
                f"RAM and does not fit this app's "
                f"{_gib(budget.available_bytes)} limit. iOS gives each process a "
                f"jetsam limit well below total RAM, and exceeding it is an "
                f"immediate kill rather than a slowdown."
            )
            if not budget.has_increased_limit:
                notes.append(
                    "The com.apple.developer.kernel.increased-memory-limit "
                    "entitlement raises that ceiling; it is not enabled here."
                )

    if weights <= 0:
        warnings.append(
            "No size for this model — neither a file size nor a recognised "
            "quantisation. The estimate below counts only the cache and "
            "overhead and is not a fit answer."
        )

    if shape.train_context and context > shape.train_context:
        notes.append(
            f"Asking for {context} tokens of context on a model trained for "
            f"{shape.train_context}. It will run, and quality past the "
            f"trained length degrades in a way the size estimate cannot see."
        )

    if backend in (ComputeBackend.METAL, ComputeBackend.HYBRID):
        notes.append(
            "Apple silicon has unified memory, so moving layers to Metal "
            "does not reduce this total — it buys speed, not headroom."
        )

    # Half the weights, not all of them. Once the cache is that large,
    # shortening the context or halving the cache saves more than
    # dropping a quantisation tier does -- and it costs no quality in
    # the weights. Waiting for the cache to strictly exceed the weights
    # withholds the advice through the whole range where it is the best
    # move available.
    if weights > 0 and kv >= weights / 2:
        comparison = "larger than" if kv > weights else "already comparable to"
        notes.append(
            f"The KV cache ({_gib(kv)}) is {comparison} the weights "
            f"({_gib(weights)}) at this context length. Halving the context, "
            f"or an 8-bit cache, saves more than a smaller quantisation would "
            f"and costs nothing in the weights."
        )

    limit = budget.available_bytes * SAFETY_MARGIN
    tight_limit = budget.available_bytes * TIGHT_MARGIN
    if budget.available_bytes <= 0:
        verdict = Verdict.NO
    elif total <= limit:
        verdict = Verdict.RUNS
    elif total <= tight_limit:
        verdict = Verdict.TIGHT
        warnings.append(
            "This leaves very little headroom. It will probably run and may "
            "be killed if anything else on the phone wants memory."
        )
    else:
        verdict = Verdict.NO

    gpu_layers = shape.layers if backend in (ComputeBackend.METAL, ComputeBackend.HYBRID) else 0
    return FitPlan(
        verdict=verdict, context=context, weights_bytes=weights, kv_bytes=kv,
        overhead_bytes=overhead, total_bytes=total,
        available_bytes=budget.available_bytes, backend=backend,
        gpu_layers=gpu_layers, warnings=warnings, notes=notes,
    )


def largest_context(
    shape: ModelShape,
    budget: DeviceBudget,
    *,
    kv_bits: int = 16,
    ceiling: int = 131072,
    granularity: int = 256,
) -> int:
    """The longest context that still earns a "runs" verdict.

    Usually the more useful question than "does it fit": a model that
    does not fit at 32k frequently runs comfortably at 4k, and refusing
    it outright hides a model that would have been fine.

    Rounded down to *granularity* so the answer is a number someone
    would choose, and because presenting 6,143 as a context length
    implies a precision the estimate does not have.
    """
    if budget.available_bytes <= 0 or shape.weights_bytes() <= 0:
        return 0
    # Even at zero context the weights and floor may not fit.
    if plan(shape, budget, context=0, kv_bits=kv_bits).verdict == Verdict.NO:
        return 0

    low, high, best = 0, max(granularity, int(ceiling)), 0
    while low <= high:
        mid = (low + high) // 2
        if plan(shape, budget, context=mid, kv_bits=kv_bits).verdict == Verdict.RUNS:
            best, low = mid, mid + granularity
        else:
            high = mid - granularity
        if high - low < granularity and best:
            break
    if shape.train_context:
        best = min(best, shape.train_context)
    return (best // granularity) * granularity


def choose_candidate(
    candidates: list[GGUFCandidate],
    shape_for: Any,
    budget: DeviceBudget,
    *,
    context: int = 4096,
) -> GGUFCandidate | None:
    """The highest-quality candidate that runs.

    Quality is ordered by effective bits per weight: given two files
    that both fit, the wider one is the better model. Falling back to
    the smallest file when nothing fits would download something that
    cannot run, so this returns None instead and the caller says so.
    """
    runnable: list[tuple[float, GGUFCandidate]] = []
    for candidate in candidates:
        shape = shape_for(candidate)
        if shape is None:
            continue
        if plan(shape, budget, context=context).verdict != Verdict.NO:
            runnable.append((quant_bits(candidate.resolved_quant), candidate))
    if not runnable:
        return None
    runnable.sort(key=lambda pair: pair[0], reverse=True)
    return runnable[0][1]


def _gib(value: int) -> str:
    if value >= GIB:
        return f"{value / GIB:.1f} GiB"
    return f"{value / MIB:.0f} MiB"
