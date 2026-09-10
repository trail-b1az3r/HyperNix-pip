"""hypernix.quant.dflash2 — a draft model, inside the model it drafts for.

Speculative decoding runs two models. A small fast one proposes the next
few tokens; the large one checks all of them in a single forward pass and
keeps the longest prefix it agrees with. Every token that survives cost
one draft step instead of one full step, and — this is the part that
makes it safe rather than a quality trade — the output is *identical* to
what the large model would have produced on its own. It is a speed
change, not a behaviour change.

The reason nobody does it is logistics. It needs two files that share a
tokenizer, and the small one has to come from somewhere. So the
speed-up sits behind "find or train a compatible draft model", which for
most people is where it stops.

Dflash2 removes that step. :func:`attach` derives a draft from the base
model — a subset of its layers, quantised hard — and writes it into the
*same GGUF*, under a namespaced tensor prefix with its own metadata
block. One file, one download, one path to pass around; the runtime
picks the draft up if it can use it and reads straight past it if it
cannot.

What the draft actually is
--------------------------
Layers, dropped and requantised. Not trained, not distilled: this runs
in the time a quantisation takes, on a machine with no GPU, from nothing
but the base model. What you get for that is a draft whose token
proposals agree with the base often enough to be worth the arithmetic —
and :func:`speculate` measures exactly how often, because a draft that
agrees 15% of the time makes generation *slower* and the only honest way
to know is to count.

First and last layers are always kept. A pruned model that lost its
first block does not produce merely worse tokens, it produces tokens
from a different distribution entirely, and its proposals are rejected
at a rate that makes the whole exercise negative.

Compatibility
-------------
The extra tensors are namespaced under ``dflash2.`` and every
``dflash2.*`` metadata key sits outside the ``general.``/``<arch>.``
namespaces upstream uses. A llama.cpp that has never heard of Dflash2
loads the file, finds every tensor it expects, ignores the ones it does
not, and runs the base model exactly as before. :func:`extract` writes
the embedded draft out as its own GGUF for a runtime that wants
``--model-draft`` as a separate path — the one file still carries both.
"""
from __future__ import annotations

import logging
import re
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import llamaquants
from .gguf import GGMLType, GGUFError, GGUFFile, GGUFTensor, GGUFWriter

logger = logging.getLogger(__name__)

__all__ = [
    "Dflash2Error",
    "PREFIX",
    "VERSION",
    "DraftPlan",
    "AttachReport",
    "SpeculationResult",
    "plan_draft",
    "attach",
    "has_draft",
    "read_draft_info",
    "extract",
    "strip",
    "speculate",
]

#: Every tensor the draft owns starts with this. Chosen so a stock
#: loader's tensor lookups miss it and its own lookups cannot collide.
PREFIX = "dflash2."

#: The metadata generation. Bumped only when the layout changes in a way
#: a reader has to know about.
VERSION = 2

#: Metadata keys, all outside the namespaces upstream uses.
KEY_PRESENT = "dflash2.present"
KEY_VERSION = "dflash2.version"
KEY_BLOCK_COUNT = "dflash2.block_count"
KEY_LAYER_MAP = "dflash2.layer_map"
KEY_QUANT = "dflash2.quant"
KEY_DRAFT_TOKENS = "dflash2.draft_tokens"
KEY_SHARED = "dflash2.shared_tensors"
KEY_SOURCE_BLOCKS = "dflash2.source_block_count"

#: How many tokens the draft proposes per round unless told otherwise.
#: Four is the usual sweet spot: past it, one rejection throws away more
#: work than the extra acceptances win back.
DEFAULT_DRAFT_TOKENS = 4

_BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")

#: Tensors that are not part of any block, and so are shared rather than
#: rebuilt: the draft has to speak the same vocabulary as the model it
#: drafts for, or its proposals are not even in the right alphabet.
_SHAREABLE = ("token_embd.weight", "output.weight", "output_norm.weight")


class Dflash2Error(Exception):
    """A draft could not be derived, attached, read or extracted."""


# ---------------------------------------------------------------------------
# Choosing the layers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftPlan:
    """Which of the base's layers the draft keeps, and at what width."""

    layers: tuple[int, ...]
    quant: str
    source_blocks: int
    shared: tuple[str, ...] = ()
    draft_tokens: int = DEFAULT_DRAFT_TOKENS

    @property
    def depth_ratio(self) -> float:
        return len(self.layers) / self.source_blocks if self.source_blocks else 0.0

    def describe(self) -> str:
        return (
            f"dflash2 draft: {len(self.layers)}/{self.source_blocks} layers "
            f"({self.depth_ratio * 100:.0f}%) at {self.quant}, "
            f"proposing {self.draft_tokens} tokens per round\n"
            f"  keeping layers: {', '.join(str(i) for i in self.layers)}"
        )


def _count_blocks(model: GGUFFile) -> int:
    highest = -1
    for tensor in model.tensors:
        match = _BLOCK.match(tensor.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def plan_draft(
    model: GGUFFile,
    *,
    layers: Sequence[int] | None = None,
    depth: float = 0.25,
    quant: str = "Q4_0",
    draft_tokens: int = DEFAULT_DRAFT_TOKENS,
    share_embeddings: bool = True,
) -> DraftPlan:
    """Decide the draft's shape without writing anything.

    *depth* is the fraction of the base's layers to keep. The first and
    last are always among them: a pruned model missing its first block
    does not produce slightly worse tokens, it produces tokens from a
    different distribution, and every one of its proposals is rejected.
    """
    total = _count_blocks(model)
    if total < 2:
        raise Dflash2Error(
            f"This model has {total} transformer block(s). There is nothing to "
            "draft from — a draft is the base model with layers removed."
        )
    if quant not in llamaquants.FORMATS:
        raise Dflash2Error(
            f"Unknown draft quantisation {quant!r}. Available: "
            f"{', '.join(llamaquants.FORMATS)}"
        )
    if draft_tokens < 1:
        raise Dflash2Error("draft_tokens must be at least 1.")

    if layers is not None:
        chosen = sorted({int(index) for index in layers})
        if not chosen:
            raise Dflash2Error("An explicit layer list cannot be empty.")
        outside = [index for index in chosen if not 0 <= index < total]
        if outside:
            raise Dflash2Error(
                f"Layer(s) {outside} are outside this model's 0..{total - 1}."
            )
    else:
        if not 0 < depth <= 1:
            raise Dflash2Error(f"depth must be in (0, 1], got {depth}.")
        keep = max(2, round(total * depth))
        keep = min(keep, total)
        # First and last pinned, the rest spread evenly through the middle.
        # Integer arithmetic rather than round(): rounding half-to-even
        # made consecutive steps land on the same layer, so asking for
        # every layer of an 8-block model returned six of them.
        chosen = [0, total - 1]
        middle = keep - 2
        if middle > 0:
            span = total - 2
            for step in range(middle):
                index = 1 + (step * span + span // 2) // middle
                chosen.append(min(max(index, 1), total - 2))
        chosen = sorted(set(chosen))

    shared = tuple(
        name for name in _SHAREABLE
        if share_embeddings and any(t.name == name for t in model.tensors)
    )
    return DraftPlan(
        layers=tuple(chosen),
        quant=quant,
        source_blocks=total,
        shared=shared,
        draft_tokens=int(draft_tokens),
    )


# ---------------------------------------------------------------------------
# Attaching
# ---------------------------------------------------------------------------


@dataclass
class AttachReport:
    """What :func:`attach` produced, and what it cost."""

    plan: DraftPlan | None = None
    base_bytes: int = 0
    output_bytes: int = 0
    draft_bytes: int = 0
    draft_tensors: int = 0
    copied_tensors: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def overhead(self) -> float:
        """Draft size as a fraction of the base. The price of the speed-up."""
        return (self.draft_bytes / self.base_bytes) if self.base_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": list(self.plan.layers) if self.plan else [],
            "source_blocks": self.plan.source_blocks if self.plan else 0,
            "quant": self.plan.quant if self.plan else "",
            "draft_tokens": self.plan.draft_tokens if self.plan else 0,
            "shared": list(self.plan.shared) if self.plan else [],
            "base_bytes": self.base_bytes,
            "output_bytes": self.output_bytes,
            "draft_bytes": self.draft_bytes,
            "draft_tensors": self.draft_tensors,
            "copied_tensors": self.copied_tensors,
            "overhead": round(self.overhead, 4),
            "skipped": [{"tensor": n, "reason": r} for n, r in self.skipped],
        }

    def describe(self) -> str:
        lines = []
        if self.plan:
            lines.append(self.plan.describe())
        lines.append(
            f"  {self.base_bytes / 1e6:.1f} MB -> {self.output_bytes / 1e6:.1f} MB "
            f"(+{self.overhead * 100:.1f}% for the draft)"
        )
        lines.append(
            f"  {self.draft_tensors} draft tensor(s) beside "
            f"{self.copied_tensors} base tensor(s)"
        )
        if self.plan and self.plan.shared:
            lines.append(
                f"  sharing the base's {', '.join(self.plan.shared)} "
                f"(the draft must speak the same vocabulary)"
            )
        if self.skipped:
            lines.append(f"  {len(self.skipped)} draft tensor(s) copied unquantised:")
            for name, reason in self.skipped[:4]:
                lines.append(f"    {name}: {reason}")
        return "\n".join(lines)


_UNQUANTIZED = {int(GGMLType.F32), int(GGMLType.F16), int(GGMLType.BF16)}


def _decode(raw: bytes, ggml_type: int) -> list[float] | None:
    """Tensor bytes to floats, or None when the type is unreadable."""
    kind = int(ggml_type)
    if kind == int(GGMLType.F32):
        return list(struct.unpack(f"<{len(raw) // 4}f", raw[: len(raw) // 4 * 4]))
    if kind == int(GGMLType.F16):
        return list(struct.unpack(f"<{len(raw) // 2}e", raw[: len(raw) // 2 * 2]))
    if kind == int(GGMLType.BF16):
        return [
            struct.unpack("<f", b"\x00\x00" + raw[i * 2:i * 2 + 2])[0]
            for i in range(len(raw) // 2)
        ]
    if llamaquants.is_supported(kind):
        return [float(v) for v in llamaquants.dequantize_array(raw, kind)]
    return None


def attach(
    base: str | Path,
    destination: str | Path,
    *,
    layers: Sequence[int] | None = None,
    depth: float = 0.25,
    quant: str = "Q4_0",
    draft_tokens: int = DEFAULT_DRAFT_TOKENS,
    share_embeddings: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> AttachReport:
    """Derive a draft from *base* and write both into *destination*.

    The base's tensors are copied through byte for byte — attaching a
    draft must not change the model anyone was already running.
    """
    base_path = Path(base)
    out_path = Path(destination)
    if not base_path.exists():
        raise Dflash2Error(f"No such model: {base_path}")
    try:
        model = GGUFFile.read(base_path)
    except GGUFError as exc:
        raise Dflash2Error(f"{base_path}: {exc}") from exc

    if any(tensor.name.startswith(PREFIX) for tensor in model.tensors):
        raise Dflash2Error(
            f"{base_path} already carries a Dflash2 draft. Use `dflash2 strip` "
            "first if you want to replace it — attaching a second one would "
            "leave two drafts and no way to say which is current."
        )

    plan = plan_draft(
        model,
        layers=layers,
        depth=depth,
        quant=quant,
        draft_tokens=draft_tokens,
        share_embeddings=share_embeddings,
    )
    report = AttachReport(plan=plan, base_bytes=base_path.stat().st_size)

    block_size = llamaquants.FORMATS[plan.quant].block

    writer = GGUFWriter(out_path, alignment=model.alignment)
    writer.copy_metadata_from(model)
    writer.set_metadata(KEY_PRESENT, True)
    writer.set_metadata(KEY_VERSION, VERSION)
    writer.set_metadata(KEY_BLOCK_COUNT, len(plan.layers))
    writer.set_metadata(KEY_SOURCE_BLOCKS, plan.source_blocks)
    writer.set_metadata(KEY_LAYER_MAP, [int(i) for i in plan.layers])
    writer.set_metadata(KEY_QUANT, plan.quant)
    writer.set_metadata(KEY_DRAFT_TOKENS, plan.draft_tokens)
    writer.set_metadata(KEY_SHARED, list(plan.shared))

    # Base tensors first, unchanged and in their original order, so the
    # file a stock loader reads is the file it read before.
    sources: dict[str, tuple[GGUFTensor, str]] = {}
    for tensor in model.tensors:
        writer.add_tensor(tensor.name, tensor.shape, tensor.ggml_type)
        sources[tensor.name] = (tensor, "")
        report.copied_tensors += 1

    for draft_index, source_index in enumerate(plan.layers):
        prefix = f"blk.{source_index}."
        for tensor in model.tensors:
            if not tensor.name.startswith(prefix):
                continue
            suffix = tensor.name[len(prefix):]
            draft_name = f"{PREFIX}blk.{draft_index}.{suffix}"
            quantisable = (
                len(tensor.shape) >= 2
                # ne[0], the row length -- GGML quantises row by row.
                # The element count was the wrong test and shipped a
                # model llama.cpp refuses: an SSM convolution weight is
                # [4, N], four per row, total 4N which divides into 256
                # whenever N does. See _should_quantize in hyprslug.py.
                and int(tensor.shape[0]) % block_size == 0
                and (
                    int(tensor.ggml_type) in _UNQUANTIZED
                    or llamaquants.is_supported(int(tensor.ggml_type))
                )
            )
            if quantisable:
                target_type = llamaquants.FORMATS[plan.quant].ggml_type
                chosen = plan.quant
            else:
                target_type = tensor.ggml_type
                chosen = ""
                report.skipped.append((
                    draft_name,
                    "1-D or not divisible: a norm is a rounding error of the size",
                ))
            declared = writer.add_tensor(draft_name, tensor.shape, target_type)
            sources[draft_name] = (tensor, chosen)
            report.draft_tensors += 1
            report.draft_bytes += declared.nbytes

    done = 0
    total = len(sources)

    def _data_for(declared: GGUFTensor) -> bytes:
        nonlocal done
        original, chosen = sources[declared.name]
        raw = model.tensor_bytes(original)
        done += 1
        if progress is not None:
            try:
                progress({
                    "event": "tensor",
                    "name": declared.name,
                    "index": done,
                    "total": total,
                    "draft": declared.name.startswith(PREFIX),
                })
            except Exception:  # noqa: BLE001 - a listener must not fail the run
                logger.debug("dflash2: progress callback raised", exc_info=True)
        if not chosen:
            return raw
        values = _decode(raw, original.ggml_type)
        if values is None:  # pragma: no cover - guarded by `quantisable`
            return raw
        return llamaquants.quantize_array(values, chosen)

    try:
        writer.write(_data_for)
    except (GGUFError, OSError, llamaquants.LlamaQuantError) as exc:
        raise Dflash2Error(f"Could not write {out_path}: {exc}") from exc

    report.output_bytes = out_path.stat().st_size
    if progress is not None:
        try:
            progress({"event": "done", **report.to_dict()})
        except Exception:  # noqa: BLE001
            logger.debug("dflash2: progress callback raised", exc_info=True)
    return report


# ---------------------------------------------------------------------------
# Reading one back
# ---------------------------------------------------------------------------


def read_draft_info(path: str | Path) -> dict[str, Any]:
    """What draft (if any) *path* carries.

    ``{"present": False}`` for a model without one — not an error. Most
    GGUFs do not have a draft and asking is the normal way to find out.
    """
    try:
        model = GGUFFile.read(path)
    except GGUFError as exc:
        raise Dflash2Error(f"{path}: {exc}") from exc
    return _info_from(model)


def _info_from(model: GGUFFile) -> dict[str, Any]:
    draft_tensors = [t for t in model.tensors if t.name.startswith(PREFIX)]
    if not model.metadata.get(KEY_PRESENT) and not draft_tensors:
        return {"present": False}
    if not draft_tensors:
        # Metadata says yes and there are no tensors. Better to say the
        # file is inconsistent than to report a draft nothing can run.
        raise Dflash2Error(
            "This file claims a Dflash2 draft in its metadata and carries no "
            f"{PREFIX} tensors. It has been rewritten by something that dropped "
            "them; the metadata is describing a draft that is not there."
        )
    return {
        "present": True,
        "version": int(model.metadata.get(KEY_VERSION, 0)),
        "block_count": int(model.metadata.get(KEY_BLOCK_COUNT, 0)),
        "source_block_count": int(model.metadata.get(KEY_SOURCE_BLOCKS, 0)),
        "layer_map": [int(i) for i in (model.metadata.get(KEY_LAYER_MAP) or [])],
        "quant": model.metadata.get(KEY_QUANT, ""),
        "draft_tokens": int(model.metadata.get(KEY_DRAFT_TOKENS, DEFAULT_DRAFT_TOKENS)),
        "shared": list(model.metadata.get(KEY_SHARED) or []),
        "tensors": len(draft_tensors),
        "bytes": sum(t.nbytes for t in draft_tensors),
    }


def has_draft(path: str | Path) -> bool:
    """True when *path* carries a Dflash2 draft."""
    try:
        return bool(read_draft_info(path).get("present"))
    except Dflash2Error:
        return False


def extract(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write the embedded draft out as a GGUF of its own.

    For a runtime that wants ``--model-draft <path>`` rather than one
    file: the single artifact still carries both, and this materialises
    the half that runtime needs. The block count in the architecture
    metadata is rewritten to the draft's own — a draft that claims the
    base's layer count describes tensors it does not have, and a loader
    fails on the first missing block rather than on the metadata.
    """
    try:
        model = GGUFFile.read(source)
    except GGUFError as exc:
        raise Dflash2Error(f"{source}: {exc}") from exc
    info = _info_from(model)
    if not info["present"]:
        raise Dflash2Error(f"{source} carries no Dflash2 draft to extract.")

    out_path = Path(destination)
    writer = GGUFWriter(out_path, alignment=model.alignment)
    writer.copy_metadata_from(model)
    for key in (KEY_PRESENT, KEY_VERSION, KEY_BLOCK_COUNT, KEY_LAYER_MAP,
                KEY_QUANT, KEY_DRAFT_TOKENS, KEY_SHARED, KEY_SOURCE_BLOCKS):
        writer.metadata.pop(key, None)
        writer.metadata_types.pop(key, None)

    architecture = model.metadata.get("general.architecture", "")
    if architecture:
        writer.set_metadata(f"{architecture}.block_count", info["block_count"])
    writer.set_metadata("general.name",
                        f"{model.metadata.get('general.name', 'model')} (dflash2 draft)")
    writer.set_metadata("hypernix.dflash2_draft", True)

    sources: dict[str, GGUFTensor] = {}
    for tensor in model.tensors:
        if tensor.name.startswith(PREFIX):
            name = tensor.name[len(PREFIX):]
        elif tensor.name in info["shared"]:
            name = tensor.name
        else:
            continue
        writer.add_tensor(name, tensor.shape, tensor.ggml_type)
        sources[name] = tensor

    writer.write(lambda declared: model.tensor_bytes(sources[declared.name]))
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "tensors": len(sources),
        "block_count": info["block_count"],
    }


def strip(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write *source* without its draft — the base model, as it was."""
    try:
        model = GGUFFile.read(source)
    except GGUFError as exc:
        raise Dflash2Error(f"{source}: {exc}") from exc

    out_path = Path(destination)
    writer = GGUFWriter(out_path, alignment=model.alignment)
    writer.copy_metadata_from(model)
    for key in (KEY_PRESENT, KEY_VERSION, KEY_BLOCK_COUNT, KEY_LAYER_MAP,
                KEY_QUANT, KEY_DRAFT_TOKENS, KEY_SHARED, KEY_SOURCE_BLOCKS):
        writer.metadata.pop(key, None)
        writer.metadata_types.pop(key, None)

    sources: dict[str, GGUFTensor] = {}
    for tensor in model.tensors:
        if tensor.name.startswith(PREFIX):
            continue
        writer.add_tensor(tensor.name, tensor.shape, tensor.ggml_type)
        sources[tensor.name] = tensor
    writer.write(lambda declared: model.tensor_bytes(sources[declared.name]))
    return {"path": str(out_path), "bytes": out_path.stat().st_size,
            "tensors": len(sources)}


# ---------------------------------------------------------------------------
# Using one
# ---------------------------------------------------------------------------


@dataclass
class SpeculationResult:
    """What a speculative run produced, and whether it was worth it."""

    tokens: list[int] = field(default_factory=list)
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0
    target_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.proposed) if self.proposed else 0.0

    @property
    def tokens_per_target_call(self) -> float:
        """The number that decides whether this helped.

        One means the draft bought nothing and cost its own runtime. Below
        the ratio of draft cost to target cost, it made generation slower.
        """
        return (len(self.tokens) / self.target_calls) if self.target_calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rounds": self.rounds,
            "target_calls": self.target_calls,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_target_call": round(self.tokens_per_target_call, 3),
        }

    def describe(self) -> str:
        return (
            f"{len(self.tokens)} token(s) in {self.target_calls} target call(s) "
            f"({self.tokens_per_target_call:.2f} per call)\n"
            f"  draft proposed {self.proposed}, {self.accepted} accepted "
            f"({self.acceptance_rate * 100:.0f}%)"
        )


def speculate(
    prefix: Sequence[int],
    propose: Callable[[list[int], int], Sequence[int]],
    verify: Callable[[list[int], Sequence[int]], Sequence[int]],
    *,
    draft_tokens: int = DEFAULT_DRAFT_TOKENS,
    max_new_tokens: int = 64,
    stop: Sequence[int] = (),
) -> SpeculationResult:
    """Generate with a draft proposing and the target verifying.

    *propose* is ``(context, k) -> up to k token ids`` from the draft.
    *verify* is ``(context, proposal) -> len(proposal) + 1 token ids``:
    the target's own choice at each position of the proposal, plus one
    more for the position after it.

    The tokens returned are **exactly** those greedy decoding with the
    target alone would produce. That is the guarantee that makes this a
    speed change rather than a quality trade: a proposal is kept only
    where the target independently chose the same token, and the first
    place they differ, the target's token wins and the rest of the
    proposal is thrown away. A bad draft costs time; it cannot cost
    correctness.
    """
    if draft_tokens < 1:
        raise Dflash2Error("draft_tokens must be at least 1.")
    result = SpeculationResult()
    context = list(prefix)
    stops = set(int(token) for token in stop)

    while len(result.tokens) < max_new_tokens:
        want = min(draft_tokens, max_new_tokens - len(result.tokens))
        proposal = [int(token) for token in propose(context, want)][:want]
        result.rounds += 1
        result.proposed += len(proposal)

        verified = [int(token) for token in verify(context, proposal)]
        result.target_calls += 1
        if len(verified) < len(proposal) + 1:
            raise Dflash2Error(
                f"verify() returned {len(verified)} token(s) for a proposal of "
                f"{len(proposal)}; it must return one per position plus one more "
                "for the position after the proposal."
            )

        matched = 0
        for offer, truth in zip(proposal, verified, strict=False):
            if offer != truth:
                break
            matched += 1
        result.accepted += matched

        # Everything the target agreed with, plus its own token at the
        # first disagreement (or the bonus token when it agreed with all
        # of them). That last one is why a round always advances even
        # when the draft is useless.
        emitted = verified[: matched + 1]
        for token in emitted:
            if len(result.tokens) >= max_new_tokens:
                break
            result.tokens.append(token)
            context.append(token)
            if token in stops:
                return result
        if not emitted:  # pragma: no cover - verify() guaranteed one above
            break
    return result
