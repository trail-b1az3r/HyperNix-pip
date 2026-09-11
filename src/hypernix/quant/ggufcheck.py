"""Whether a GGUF on disk is one llama.cpp will actually load.

Until 0.72.4.post16 ``hyprslug`` decided whether a tensor could carry a
block-quantised type by dividing its *element total* by the block size.
GGML quantises row by row, so the real constraint is on ``ne[0]`` -- the
row length -- and the two differ for exactly the tensors a transformer
has a few of and a hybrid has many of::

    gguf_init_from_reader: tensor 'blk.0.ssm_conv1d.weight' of type 202
    (IQ0.5_XXXL) has 4 elements per row, not a multiple of block size
    (256)

That is a *load-time* refusal: the file is written, sits on disk at the
size it advertises, and fails only when something tries to open it. A
2B model is a long quantisation run to discover that at the end of.

So this module answers the question directly, from the tensor table
alone -- no dequantisation, no full read -- and repairs a file that
answers wrong. :func:`check_gguf` is what ``--check`` runs;
:func:`repair_gguf` rewrites the offending tensors back to F32.

Repair is not the same as re-quantising. The values it writes are the
ones the sub-bit packer produced, widened back to F32: whatever the
quantiser lost is already lost. It gets a file to load without a second
run over the base model, and :func:`repair_gguf` says so in its report
rather than letting the smaller file imply otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gguf import (
    GGMLType,
    GGUFError,
    GGUFFile,
    GGUFTensor,
    GGUFWriter,
    type_block_size,
)

__all__ = [
    "BadTensor",
    "CheckReport",
    "RepairReport",
    "block_elements",
    "LLAMA_CPP_REGISTERED_TYPES",
    "check_gguf",
    "llama_cpp_can_load_type",
    "repair_gguf",
    "row_length",
]


#: Elements per block. ``gguf.type_block_size`` already is this; it is
#: re-exported rather than reimplemented so the two cannot drift, and
#: named here because the row check reads better with it in scope.
block_elements = type_block_size


#: The HyperNix type ids a patched llama.cpp actually registers.
#:
#: ``native/ggml-hnx/tools/patch_llamacpp.py`` adds five enum members
#: (200-204) and pins ``GGML_TYPE_COUNT`` to 205, so 205 and 206 are
#: past the end of both trait tables and gguf.cpp rejects them on the
#: type check before it ever reaches a tensor.
#:
#: That matters because :mod:`hypernix.quant.hyprslug` offers seven
#: tiers, not five. INT4 and FP2 quantise, write a well-formed GGUF and
#: run under :mod:`hypernix.models.hnxrun` -- and cannot be opened by
#: any llama.cpp build, patched or not. Nothing said so until this
#: table existed, so somebody could spend an hour quantising to a tier
#: that llama-server can never load.
#:
#: ``tests/test_ggufcheck.py`` parses the patch script's own enum text
#: and asserts this agrees with it, so adding a type on the C side
#: without updating this is a test failure rather than a surprise.
LLAMA_CPP_REGISTERED_TYPES: frozenset[int] = frozenset({200, 201, 202, 203, 204})


def llama_cpp_can_load_type(ggml_type: int) -> bool:
    """Whether a patched llama.cpp has a trait-table entry for this id.

    Upstream ids (below 200) are llama.cpp's own and always readable;
    only the HyperNix range needs the patch.
    """
    kind = int(ggml_type)
    return kind < 200 or kind in LLAMA_CPP_REGISTERED_TYPES


def row_length(tensor: GGUFTensor) -> int:
    """``ne[0]``: the fastest-moving dimension, as GGUF stores it.

    :class:`GGUFTensor` keeps the dimensions in file order, so this is a
    lookup rather than a convention that could be got backwards -- which
    is the mistake this whole module exists because of.
    """
    return int(tensor.shape[0]) if tensor.shape else 0


def _type_name(ggml_type: int) -> str:
    """The name a reader prints for this id.

    For a HyperNix type that is the *tier* name -- ``IQ0.5_XXXL``, which
    is what the patched llama.cpp puts in its error -- and not the
    Python enum's spelling, ``HNX_IQ0_5``. The two differ, and printing
    the second while claiming to quote the first is how somebody
    searching for their error message fails to find this tool.
    """
    from .hyprslug import TIER_TYPES

    kind = int(ggml_type)
    for tier, (type_id, _packing) in TIER_TYPES.items():
        if int(type_id) == kind:
            return tier
    try:
        return GGMLType(kind).name
    except ValueError:
        return f"type {kind}"


@dataclass
class BadTensor:
    """One tensor llama.cpp will refuse, and the words it will use."""

    name: str
    ggml_type: int
    row: int
    block: int
    shape: tuple[int, ...] = ()

    @property
    def type_name(self) -> str:
        return _type_name(self.ggml_type)

    @property
    def llama_cpp_message(self) -> str:
        """Verbatim, so searching the log lands here.

        Matching upstream's wording exactly is deliberate: somebody with
        this error in front of them should be able to paste it and find
        the tool that explains it.
        """
        return (
            f"tensor '{self.name}' of type {self.ggml_type} ({self.type_name}) "
            f"has {self.row} elements per row, not a multiple of block size "
            f"({self.block})"
        )


@dataclass
class CheckReport:
    path: str = ""
    tensors: int = 0
    bad: list[BadTensor] = field(default_factory=list)
    #: Types present in the file, by name, with a count each.
    types: dict[str, int] = field(default_factory=dict)
    #: ``hypernix.tier`` as the file's own metadata states it.
    tier: str = ""
    #: Problems with the container rather than a tensor's type: a file
    #: that stops before its tensor data does, a duplicate name, a
    #: misaligned offset. These are what an interrupted quantise or a
    #: full disk leaves behind, and llama.cpp refuses them with a
    #: generic "failed to load model" that names no tensor at all.
    structural: list[str] = field(default_factory=list)
    #: Types in the file that no llama.cpp build registers, by name.
    #: A different failure from :attr:`bad` with a different remedy:
    #: the row problem is one tensor and repairable, this is the whole
    #: file and is not.
    unregistered: list[str] = field(default_factory=list)

    @property
    def loadable(self) -> bool:
        """Whether *llama.cpp* will open it."""
        return not self.bad and not self.unregistered and not self.structural

    @property
    def runs_under_hnxrun(self) -> bool:
        """HyperNix's own runtime knows every tier it writes, so an
        unregistered type is only a llama.cpp limitation -- worth saying,
        because the file is not junk.

        A structural fault is different: bytes that were never written
        cannot be read by any runtime, so it disqualifies this too.
        """
        return not self.bad and not self.structural

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "loadable": self.loadable,
            "tensors": self.tensors,
            "tier": self.tier,
            "types": self.types,
            "unregistered": self.unregistered,
            "structural": self.structural,
            "runs_under_hnxrun": self.runs_under_hnxrun,
            "bad": [
                {
                    "name": entry.name,
                    "type": entry.ggml_type,
                    "type_name": entry.type_name,
                    "shape": list(entry.shape),
                    "elements_per_row": entry.row,
                    "block": entry.block,
                    "llama_cpp": entry.llama_cpp_message,
                }
                for entry in self.bad
            ],
        }

    def describe(self) -> str:
        lines = [f"{self.path}", f"  tensors  {self.tensors}"]
        if self.tier:
            lines.append(f"  tier     {self.tier}")
        if self.types:
            spread = ", ".join(
                f"{name} x{count}" for name, count in sorted(self.types.items())
            )
            lines.append(f"  types    {spread}")
        if self.structural:
            lines.append("  The file itself is wrong, before any tensor's type:")
            for problem in self.structural[:6]:
                lines.append(f"    {problem}")
            if len(self.structural) > 6:
                lines.append(f"    ... and {len(self.structural) - 6} more")
            lines.append("")
            lines.append(
                "  A quantise that was interrupted, or a disk that filled up,"
            )
            lines.append(
                "  leaves exactly this. llama.cpp reports it as a bare 'failed"
            )
            lines.append(
                "  to load model' naming no tensor. Re-run the quantisation."
            )
        if self.unregistered:
            spread = ", ".join(sorted(self.unregistered))
            lines.append(f"  No llama.cpp build registers: {spread}")
            lines.append(
                "  These tiers run under HyperNix's own runtime (hnx generate,"
            )
            lines.append(
                "  hnx chat) and nothing else: the ggml patch registers ids"
            )
            lines.append(
                "  200-204 and pins GGML_TYPE_COUNT to 205, so gguf.cpp rejects"
            )
            lines.append(
                "  them on the type check. Re-quantise to IQ0.9_L, IQ0.75_M,"
            )
            lines.append(
                "  IQ0.5_XXXL, IQ0.25_UXL or INT1 to use llama.cpp or llama-server."
            )
        if self.loadable:
            lines.append("  llama.cpp will load this file.")
            return "\n".join(lines)
        if not self.bad:
            return "\n".join(lines)
        lines.append(
            f"  {len(self.bad)} tensor(s) llama.cpp will refuse, the first being:"
        )
        for entry in self.bad[:5]:
            lines.append(f"    {entry.llama_cpp_message}")
        if len(self.bad) > 5:
            lines.append(f"    ... and {len(self.bad) - 5} more")
        lines.append("")
        lines.append(
            "  Written by hyprslug before 0.72.4.post16, which checked the"
        )
        lines.append(
            "  element total instead of the row length. Re-quantise from the"
        )
        lines.append(
            "  base model for the best result, or --repair-to for a file that"
        )
        lines.append(
            "  loads without one (it widens those tensors back to F32; it"
        )
        lines.append("  cannot recover what the quantiser already discarded).")
        return "\n".join(lines)



def _check_container(model: GGUFFile, path: Path, report: CheckReport) -> None:
    """The checks that are about the file, not about a tensor's type.

    llama.cpp validates these in ``gguf_init_from_reader`` before it
    looks at a single block, and it reports every one of them as the same
    bare "failed to load model" with no tensor named -- so a file that
    fails here gives its owner nothing at all to go on.

    Truncation is first because it is much the most likely: a quantise
    that was interrupted, or a disk that filled part-way through writing
    a multi-gigabyte tensor, leaves a file whose header promises more
    than the file holds.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return

    # Tensor offsets are relative to the data region; the reader
    # already resolved where that starts.
    data_start = int(model.data_start)
    seen: dict[str, int] = {}
    for index, tensor in enumerate(model.tensors):
        if tensor.name in seen:
            report.structural.append(
                f"'{tensor.name}' appears twice (entries {seen[tensor.name]} and {index})"
            )
        seen[tensor.name] = index

        if any(int(dim) <= 0 for dim in tensor.shape):
            report.structural.append(
                f"'{tensor.name}' has a zero or negative dimension: {tuple(tensor.shape)}"
            )
        if len(tensor.shape) > 4:
            report.structural.append(
                f"'{tensor.name}' has {len(tensor.shape)} dimensions; GGML allows 4"
            )
        if tensor.nbytes:
            end = data_start + int(tensor.offset) + int(tensor.nbytes)
            if end > size:
                short = end - size
                report.structural.append(
                    f"'{tensor.name}' needs {short:,} bytes past the end of the "
                    f"file (the file is {size:,} bytes; the table asks for "
                    f"{end:,}) -- the file is truncated"
                )


def check_gguf(path: str | Path) -> CheckReport:
    """Read *path*'s tensor table and report what llama.cpp would refuse.

    The tensor *table*, not the tensor data: this opens the header and
    stops, so checking a 40 GB model costs the same as checking a small
    one.
    """
    model = GGUFFile.read(path)
    report = CheckReport(
        path=str(path),
        tensors=len(model.tensors),
        tier=str(model.metadata.get("hypernix.tier", "")),
    )
    _check_container(model, Path(path), report)
    for tensor in model.tensors:
        name = _type_name(tensor.ggml_type)
        report.types[name] = report.types.get(name, 0) + 1
        try:
            block = block_elements(tensor.ggml_type)
        except GGUFError:
            # A type this build does not know. Not our call to make: a
            # reader that knows it will judge it, and one that does not
            # will refuse it for that reason instead.
            continue
        if not llama_cpp_can_load_type(tensor.ggml_type) and name not in report.unregistered:
            report.unregistered.append(name)
        row = row_length(tensor)
        if block > 1 and row % block:
            report.bad.append(
                BadTensor(
                    name=tensor.name,
                    ggml_type=int(tensor.ggml_type),
                    row=row,
                    block=block,
                    shape=tuple(tensor.shape),
                )
            )
    return report


@dataclass
class RepairReport:
    source: str = ""
    output: str = ""
    repaired: list[str] = field(default_factory=list)
    copied: int = 0
    source_bytes: int = 0
    output_bytes: int = 0

    @property
    def grew_by(self) -> int:
        return self.output_bytes - self.source_bytes

    def describe(self) -> str:
        lines = [
            f"repaired {len(self.repaired)} tensor(s) -> F32, copied {self.copied}",
            f"  {self.source}  {self.source_bytes:,} bytes",
            f"  {self.output}  {self.output_bytes:,} bytes "
            f"({self.grew_by:+,})",
        ]
        for name in self.repaired[:10]:
            lines.append(f"    {name}")
        if len(self.repaired) > 10:
            lines.append(f"    ... and {len(self.repaired) - 10} more")
        lines.append(
            "  These carry the values the sub-bit packer produced, widened"
        )
        lines.append(
            "  back to F32 -- the file loads, but the quantiser's loss is"
        )
        lines.append(
            "  already in them. Re-quantising from the base model is better."
        )
        return "\n".join(lines)


def repair_gguf(source: str | Path, output: str | Path) -> RepairReport:
    """Rewrite *source* to *output* with every unloadable tensor as F32.

    Everything else is copied through byte for byte, metadata included,
    so a repaired file differs from its input in exactly the tensors
    that made it unloadable.
    """
    import struct

    import numpy as np

    from ..models.hnxrun import _dequantize

    source_path, output_path = Path(source), Path(output)
    if output_path.resolve() == source_path.resolve():
        raise GGUFError(
            "Repair writes a new file; give --repair-to a path that is not "
            "the model being repaired."
        )
    model = GGUFFile.read(source_path)
    report = RepairReport(source=str(source_path), output=str(output_path))
    broken = {entry.name for entry in check_gguf(source_path).bad}

    writer = GGUFWriter(output_path)
    writer.copy_metadata_from(model)
    payload: dict[str, bytes] = {}
    for tensor in model.tensors:
        raw = model.tensor_bytes(tensor)
        if tensor.name in broken:
            values = _dequantize(raw, tensor.ggml_type, tensor.elements)
            flat = np.asarray(values, dtype="<f4").reshape(-1)[: tensor.elements]
            payload[tensor.name] = struct.pack(f"<{flat.size}f", *flat.tolist())
            writer.add_tensor(tensor.name, tensor.shape, int(GGMLType.F32))
            report.repaired.append(tensor.name)
        else:
            payload[tensor.name] = raw
            writer.add_tensor(tensor.name, tensor.shape, int(tensor.ggml_type))
            report.copied += 1
    writer.write(lambda tensor: payload[tensor.name])

    report.source_bytes = source_path.stat().st_size
    report.output_bytes = output_path.stat().st_size
    return report
