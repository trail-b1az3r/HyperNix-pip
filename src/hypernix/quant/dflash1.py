"""hypernix.quant.dflash1 — a draft model, as its own file.

:mod:`hypernix.quant.dflash2` writes a speculative-decoding draft *into*
the model it drafts for. Dflash1 is the generation before that, and it is
kept rather than superseded because the one-file trick is not free: it
needs a runtime that knows to look under the ``dflash2.`` prefix. Every
runtime that does speculative decoding at all already knows how to take a
second path.

    llama-server -m model.gguf --model-draft model.draft.gguf
    llama-cli    -m model.gguf --model-draft model.draft.gguf --draft 4

So Dflash1 derives the same draft by the same method and writes a
complete, ordinary GGUF: its own metadata, its own tokenizer, its own
blocks numbered from zero. Nothing about the file says "draft" except a
few ``dflash1.*`` keys recording where it came from. llama.cpp opens it
as a model, because it is one.

When to use which
-----------------
============================  ========
Situation                     Use
============================  ========
llama.cpp, ollama, LM Studio  dflash1
One file to distribute        dflash2
A runtime with ``--model-draft``  dflash1
The HyperNix runtime          either
============================  ========

Neither is better. ``dflash2 extract`` turns an embedded draft into this
format, and :func:`attach_from` goes the other way, so a draft derived
once can be shipped either way without deriving it again.

What the draft is, and what it is not
--------------------------------------
Layers, dropped and requantised — see :mod:`hypernix.quant.dflash2` for
the reasoning, which is the same reasoning. It is not trained and it is
not distilled. It runs in the time a quantisation takes, on a machine
with no GPU, from nothing but the base model. The question that decides
whether it was worth doing is the acceptance rate, and
:func:`hypernix.quant.dflash2.speculate` measures it.

The one thing this file must get right
---------------------------------------
The tokenizer. A draft proposes token *ids*, and the base model verifies
them as ids; if the two files disagree about which id is which string,
every proposal is rejected and generation gets slower with no other
symptom. So the whole tokenizer metadata block is copied across
unmodified, and :func:`derive` refuses to write a draft from a base that
has none rather than producing a file that looks fine and is useless.
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
from .dflash2 import DEFAULT_DRAFT_TOKENS, DraftPlan, plan_draft
from .gguf import GGMLType, GGUFError, GGUFFile, GGUFTensor, GGUFWriter

logger = logging.getLogger(__name__)

__all__ = [
    "Dflash1Error",
    "VERSION",
    "DeriveReport",
    "derive",
    "is_draft",
    "read_draft_info",
]

#: The metadata generation, bumped only when the layout changes.
VERSION = 1

KEY_PRESENT = "dflash1.present"
KEY_VERSION = "dflash1.version"
KEY_QUANT = "dflash1.quant"
KEY_LAYER_MAP = "dflash1.layer_map"
KEY_SOURCE_BLOCKS = "dflash1.source_block_count"
KEY_DRAFT_TOKENS = "dflash1.draft_tokens"
KEY_BASE_NAME = "dflash1.base_name"

_BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")

#: Metadata keys whose value counts blocks and therefore has to be
#: rewritten: the draft has fewer than the base did, and a file that says
#: 32 while carrying 8 is one llama.cpp allocates 32 layers for and then
#: fails to find tensors for.
_BLOCK_COUNT_SUFFIX = ".block_count"

#: Types that can be read element-wise here. Anything quantised goes
#: through :mod:`hypernix.quant.llamaquants` instead.
_UNQUANTIZED = {int(GGMLType.F32), int(GGMLType.F16), int(GGMLType.BF16)}


class Dflash1Error(Exception):
    """A standalone draft could not be derived or written."""


@dataclass
class DeriveReport:
    """What :func:`derive` produced."""

    plan: DraftPlan | None = None
    base_bytes: int = 0
    draft_bytes: int = 0
    draft_tensors: int = 0
    copied_tensors: int = 0
    shared_tensors: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """Draft size as a fraction of the base. Smaller is faster."""
        return (self.draft_bytes / self.base_bytes) if self.base_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": list(self.plan.layers) if self.plan else [],
            "source_blocks": self.plan.source_blocks if self.plan else 0,
            "quant": self.plan.quant if self.plan else "",
            "draft_tokens": self.plan.draft_tokens if self.plan else 0,
            "base_bytes": self.base_bytes,
            "draft_bytes": self.draft_bytes,
            "draft_tensors": self.draft_tensors,
            "copied_tensors": self.copied_tensors,
            "shared_tensors": self.shared_tensors,
            "ratio": round(self.ratio, 4),
            "skipped": [{"tensor": n, "reason": r} for n, r in self.skipped],
        }

    def describe(self) -> str:
        lines = []
        if self.plan:
            lines.append(
                f"dflash1 draft: {len(self.plan.layers)}/{self.plan.source_blocks} "
                f"layers at {self.plan.quant}, proposing "
                f"{self.plan.draft_tokens} tokens per round"
            )
            lines.append(
                f"  keeping layers: "
                f"{', '.join(str(i) for i in self.plan.layers)}"
            )
        lines.append(
            f"  {self.base_bytes / 1e6:.1f} MB base -> "
            f"{self.draft_bytes / 1e6:.1f} MB draft ({self.ratio * 100:.1f}%)"
        )
        lines.append(
            f"  {self.draft_tensors} block tensor(s) + "
            f"{self.shared_tensors} vocabulary tensor(s)"
        )
        if self.skipped:
            lines.append(
                f"  {len(self.skipped)} copied at source precision:"
            )
            for name, reason in self.skipped[:4]:
                lines.append(f"    {name}: {reason}")
            if len(self.skipped) > 4:
                lines.append(f"    ... and {len(self.skipped) - 4} more")
        return "\n".join(lines)


def is_draft(model: GGUFFile) -> bool:
    """Whether *model* was written by this module."""
    return bool(model.metadata.get(KEY_PRESENT))


def read_draft_info(path: str | Path) -> dict[str, Any]:
    """The ``dflash1.*`` keys of *path*, or ``{}`` if it is not a draft."""
    try:
        model = GGUFFile.read(Path(path))
    except GGUFError as exc:
        raise Dflash1Error(f"{path}: {exc}") from exc
    if not is_draft(model):
        return {}
    return {
        key[len("dflash1."):]: value
        for key, value in model.metadata.items()
        if key.startswith("dflash1.")
    }


def _architecture(model: GGUFFile) -> str:
    arch = model.metadata.get("general.architecture")
    return str(arch) if isinstance(arch, str) else ""


def _has_tokenizer(model: GGUFFile) -> bool:
    return any(key.startswith("tokenizer.") for key in model.metadata)


def _decode(raw: bytes, ggml_type: int) -> list[float] | None:
    """Tensor bytes to floats, or ``None`` when the type is unreadable."""
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


def derive(
    base: str | Path,
    destination: str | Path,
    *,
    layers: Sequence[int] | None = None,
    depth: float = 0.25,
    quant: str = "Q4_0",
    draft_tokens: int = DEFAULT_DRAFT_TOKENS,
    require_tokenizer: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> DeriveReport:
    """Write a standalone draft GGUF for *base* at *destination*.

    The draft keeps ``depth`` of the base's layers (first and last always
    among them), renumbered from zero, quantised to *quant*, with the
    base's vocabulary tensors and the whole of its tokenizer metadata
    copied across unchanged.

    *require_tokenizer* exists so a test fixture can be drafted; leave it
    on for anything real. A draft whose tokenizer disagrees with the base
    has every proposal rejected, which does not error — it just makes
    generation slower than not using a draft at all, with no symptom that
    points at the cause.
    """
    base_path = Path(base)
    out_path = Path(destination)
    if not base_path.exists():
        raise Dflash1Error(f"No such model: {base_path}")
    try:
        model = GGUFFile.read(base_path)
    except GGUFError as exc:
        raise Dflash1Error(f"{base_path}: {exc}") from exc

    if require_tokenizer and not _has_tokenizer(model):
        raise Dflash1Error(
            f"{base_path} carries no tokenizer metadata, so a draft derived "
            "from it would propose token ids the base model reads as "
            "different tokens. Every proposal would be rejected and "
            "generation would be slower, silently. Pass "
            "require_tokenizer=False only if you know the runtime supplies "
            "the vocabulary from somewhere else."
        )

    plan = plan_draft(
        model,
        layers=layers,
        depth=depth,
        quant=quant,
        draft_tokens=draft_tokens,
        share_embeddings=True,
    )
    report = DeriveReport(plan=plan, base_bytes=base_path.stat().st_size)
    block_size = llamaquants.FORMATS[plan.quant].block

    writer = GGUFWriter(out_path, alignment=model.alignment)
    writer.copy_metadata_from(model)

    # The draft has fewer blocks than the base, and llama.cpp allocates
    # from this key before it looks for a single tensor. Left at the
    # base's value the file does not load at all -- "missing tensor
    # blk.8.attn_q.weight" from a draft that legitimately has eight.
    arch = _architecture(model)
    rewritten = []
    for key in list(writer.metadata):
        if key.endswith(_BLOCK_COUNT_SUFFIX) and (not arch or key.startswith(arch)):
            writer.set_metadata(key, len(plan.layers))
            rewritten.append(key)
    if not rewritten:
        logger.warning(
            "dflash1: %s has no *%s key, so nothing recorded the base's depth "
            "and nothing records the draft's. The file is still correct; a "
            "loader that infers depth from the tensor table will read it.",
            base_path, _BLOCK_COUNT_SUFFIX,
        )

    if isinstance(model.metadata.get("general.name"), str):
        writer.set_metadata(
            "general.name", f"{model.metadata['general.name']} (dflash1 draft)"
        )
    writer.set_metadata(KEY_PRESENT, True)
    writer.set_metadata(KEY_VERSION, VERSION)
    writer.set_metadata(KEY_QUANT, plan.quant)
    writer.set_metadata(KEY_LAYER_MAP, [int(i) for i in plan.layers])
    writer.set_metadata(KEY_SOURCE_BLOCKS, plan.source_blocks)
    writer.set_metadata(KEY_DRAFT_TOKENS, plan.draft_tokens)
    writer.set_metadata(KEY_BASE_NAME, base_path.name)
    writer.set_metadata(
        "general.file_type_description",
        f"dflash1 draft ({len(plan.layers)}/{plan.source_blocks} layers, "
        f"{plan.quant}) via hyprslug",
    )

    sources: dict[str, tuple[GGUFTensor, str]] = {}

    # The vocabulary tensors, byte for byte. A draft that re-quantised
    # its embedding table would be speaking a slightly different
    # language from the model checking its work.
    for name in plan.shared:
        tensor = model.get(name)
        if tensor is None:  # pragma: no cover - plan_draft only lists present ones
            continue
        writer.add_tensor(tensor.name, tensor.shape, tensor.ggml_type)
        sources[tensor.name] = (tensor, "")
        report.shared_tensors += 1

    for draft_index, source_index in enumerate(plan.layers):
        prefix = f"blk.{source_index}."
        for tensor in model.tensors:
            if not tensor.name.startswith(prefix):
                continue
            # Renumbered, and *not* namespaced: this is a model in its
            # own right, so its ninth block is blk.8, not
            # dflash1.blk.8. That is the whole difference from dflash2.
            draft_name = f"blk.{draft_index}.{tensor.name[len(prefix):]}"
            quantisable = (
                len(tensor.shape) >= 2
                # ne[0], the row length: GGML quantises row by row. See
                # _should_quantize in hyprslug.py for the model this
                # check was added for.
                and int(tensor.shape[0]) % block_size == 0
                and (
                    int(tensor.ggml_type) in _UNQUANTIZED
                    or llamaquants.is_supported(int(tensor.ggml_type))
                )
            )
            if quantisable:
                target_type = llamaquants.FORMATS[plan.quant].ggml_type
                chosen = plan.quant
                report.draft_tensors += 1
            else:
                target_type = tensor.ggml_type
                chosen = ""
                report.copied_tensors += 1
                report.skipped.append((
                    draft_name,
                    "1-D or not divisible: a norm is a rounding error of the size",
                ))
            writer.add_tensor(draft_name, tensor.shape, target_type)
            sources[draft_name] = (tensor, chosen)

    if not sources:
        raise Dflash1Error(
            f"{base_path} has no block tensors named blk.N.*, so there is "
            "nothing to derive a draft from."
        )

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
                    "quantized": bool(chosen),
                })
            except Exception:  # noqa: BLE001 - a listener must not fail the run
                logger.debug("dflash1: progress callback raised", exc_info=True)
        if not chosen:
            return raw
        values = _decode(raw, original.ggml_type)
        if values is None:  # pragma: no cover - guarded by `quantisable`
            return raw
        return llamaquants.quantize_array(values, chosen)

    try:
        writer.write(_data_for)
    except (GGUFError, OSError, llamaquants.LlamaQuantError) as exc:
        raise Dflash1Error(f"Could not write {out_path}: {exc}") from exc

    report.draft_bytes = out_path.stat().st_size
    if progress is not None:
        try:
            progress({"event": "done", **report.to_dict()})
        except Exception:  # noqa: BLE001
            logger.debug("dflash1: progress callback raised", exc_info=True)
    return report
