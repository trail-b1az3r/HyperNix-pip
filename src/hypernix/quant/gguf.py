"""hypernix.quant.gguf — read and write GGUF without llama.cpp.

Every quantiser in this package used to shell out to ``llama-quantize``,
which means a machine that has not built llama.cpp cannot quantise at
all — and the sub-bit tiers in :mod:`hypernix.quant.steamroller` are
HyperNix types that ``llama-quantize`` has never heard of, so for those
it could never have been the answer in the first place.

This is the file format, and nothing else. It parses a GGUF v2/v3
header, its metadata key-value block and its tensor table, gives you the
tensor bytes, and writes a well-formed GGUF back. It does not know what a
model is, does not load one, and has no opinion about quantisation —
:mod:`hypernix.quant.subbit` is where that lives.

Two things it takes seriously
-----------------------------
**Alignment.** GGUF pads the tensor-data region to ``general.alignment``
(32 by default) and every tensor offset is relative to the start of that
region. Getting this wrong produces a file that opens, reports sensible
shapes, and returns garbage from the second tensor onward — which is
much worse than a file that fails to open.

**Unknown types survive.** A reader that dropped metadata keys it did not
recognise would silently strip a model's chat template, its rope scaling
and its tokenizer on every round trip. Unknown *values* are preserved
verbatim; only unknown *type ids* are an error, because those cannot be
copied without knowing their length.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

__all__ = [
    "GGUFError",
    "GGUFValueType",
    "GGMLType",
    "GGUFTensor",
    "GGUFFile",
    "GGUF_MAGIC",
    "DEFAULT_ALIGNMENT",
    "type_block_size",
    "type_size_bytes",
    "tensor_nbytes",
    "tensor_nbytes_unchecked",
]

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
_SUPPORTED_VERSIONS = (2, 3)


class GGUFError(Exception):
    """The file is not GGUF, or is GGUF this module cannot represent."""


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMAT = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.BOOL: "<?",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}


class GGMLType(IntEnum):
    """Tensor element types, upstream plus this package's own.

    The HyperNix sub-bit types start at 200, far above anything llama.cpp
    has allocated, so a stock loader hits an unknown type id and refuses
    the file by name instead of reading a 0.5-bit tensor as Q4_K and
    producing noise. Refusing loudly is the whole point of picking a
    number that cannot collide.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30

    # --- HyperNix extension types -------------------------------------
    # Deliberately outside anything upstream has allocated, so a stock
    # loader refuses the file by name rather than reading a 0.5-bit
    # tensor as Q4_K and producing noise.
    HNX_IQ0_9 = 200
    HNX_IQ0_75 = 201
    HNX_IQ0_5 = 202
    HNX_IQ0_25 = 203
    HNX_INT1 = 204
    HNX_INT4 = 205
    HNX_FP2 = 206
    HNX_1375 = 207


#: ``type -> (elements per block, bytes per block)``.
#:
#: Only the types this package reads or writes. A type absent here can
#: still be *copied* — the tensor table carries its byte length — but its
#: element count cannot be checked, so :func:`tensor_nbytes` refuses
#: rather than guessing.
_BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.F64: (1, 8),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 40),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
    GGMLType.IQ4_NL: (32, 18),
    GGMLType.IQ4_XS: (256, 136),
    # HyperNix sub-bit. One block is 256 weights; the byte counts are what
    # the packings in hypernix.quant.subbit actually emit, and the tests
    # assert the two agree — a table that drifts from the packer produces
    # a file whose offsets are all wrong from the first tensor.
    GGMLType.HNX_IQ0_9: (256, 30),
    GGMLType.HNX_IQ0_75: (256, 26),
    GGMLType.HNX_IQ0_5: (256, 18),
    GGMLType.HNX_IQ0_25: (256, 8),
    # The fixed-codebook types from hypernix.quant.lowbit, plus INT1,
    # which is subbit's k == g case rather than a codec of its own.
    GGMLType.HNX_INT1: (256, 34),
    GGMLType.HNX_INT4: (256, 130),
    GGMLType.HNX_FP2: (256, 66),
    # Every sign (32 B) + sixteen 5-bit sub-block magnitudes
    # (10 B) + the FP16 scale (2 B). 44 * 8 / 256 = 1.375 exactly.
    GGMLType.HNX_1375: (256, 44),
}


def type_block_size(ggml_type: int) -> int:
    """Elements per block for *ggml_type*."""
    shape = _BLOCK_SHAPE.get(int(ggml_type))
    if shape is None:
        raise GGUFError(f"Unknown GGML type id {ggml_type}; block size not known.")
    return shape[0]


def type_size_bytes(ggml_type: int) -> int:
    """Bytes per block for *ggml_type*."""
    shape = _BLOCK_SHAPE.get(int(ggml_type))
    if shape is None:
        raise GGUFError(f"Unknown GGML type id {ggml_type}; block size not known.")
    return shape[1]


def tensor_nbytes(ggml_type: int, shape: tuple[int, ...]) -> int:
    """Bytes one tensor of *shape* occupies in *ggml_type*.

    Refuses a shape GGML cannot represent in this type, which is the
    last place a mistake upstream can still be caught for free: every
    write goes through here to lay the file out, so a tensor that gets
    past this is a tensor in a file somebody will try to load.

    The constraint is on ``ne[0]``, the row length. GGML quantises row
    by row, so a block never straddles two rows and it is the *row* that
    has to divide -- not the element total, which is the weaker test
    this used to apply. A ``[4, 4096]`` SSM convolution weight has
    16,384 elements, divides cleanly by 256, and is four elements per
    row: it passed here, was written as a sub-bit type, and llama.cpp
    refused the finished file with

        tensor 'blk.0.ssm_conv1d.weight' of type 202 (IQ0.5_XXXL) has 4
        elements per row, not a multiple of block size (256)

    The row test subsumes the total, since a product is divisible by
    anything its first factor is.
    """
    block = type_block_size(ggml_type)
    row = int(shape[0]) if shape else _element_count(shape)
    if block > 1 and row % block:
        raise GGUFError(
            f"A tensor of {row} elements per row does not divide into "
            f"{block}-element blocks for type {ggml_type}. GGML quantises row "
            f"by row, so ne[0] is the dimension that has to divide."
        )
    return tensor_nbytes_unchecked(ggml_type, shape)


def _element_count(shape: tuple[int, ...]) -> int:
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return elements


def tensor_nbytes_unchecked(ggml_type: int, shape: tuple[int, ...]) -> int:
    """:func:`tensor_nbytes` without the row check, for *reading*.

    A reader must not refuse what a writer should not have produced.
    Files with the row bug exist -- that is the whole reason
    :mod:`hypernix.quant.ggufcheck` exists -- and a reader that raised on
    them would make them undiagnosable and unrepairable by the very tool
    written to fix them.

    The arithmetic is what the old writer used, so it reproduces the
    byte layout of such a file exactly; for any legal tensor the two
    functions agree, since a row that divides means a total that does.
    """
    block, size = _BLOCK_SHAPE.get(int(ggml_type), (None, None))
    if block is None:
        raise GGUFError(f"Unknown GGML type id {ggml_type}; size not known.")
    return (_element_count(shape) // block) * size


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _read_exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise GGUFError(f"Truncated file: wanted {count} bytes, got {len(data)}.")
    return data


def _read_scalar(stream: BinaryIO, value_type: GGUFValueType) -> Any:
    fmt = _SCALAR_FORMAT.get(value_type)
    if fmt is None:
        raise GGUFError(f"{value_type!r} is not a scalar.")
    return struct.unpack(fmt, _read_exact(stream, struct.calcsize(fmt)))[0]


def _read_string(stream: BinaryIO) -> str:
    length = struct.unpack("<Q", _read_exact(stream, 8))[0]
    if length > (1 << 30):
        raise GGUFError(f"Implausible string length {length}; file is probably not GGUF.")
    return _read_exact(stream, length).decode("utf-8", errors="replace")


def _read_value(stream: BinaryIO) -> Any:
    raw_type = struct.unpack("<I", _read_exact(stream, 4))[0]
    return _read_typed_value(stream, raw_type)


def _read_typed_value(stream: BinaryIO, raw_type: int) -> Any:
    try:
        value_type = GGUFValueType(raw_type)
    except ValueError as exc:
        raise GGUFError(f"Unknown GGUF value type {raw_type}.") from exc
    if value_type is GGUFValueType.STRING:
        return _read_string(stream)
    if value_type is GGUFValueType.ARRAY:
        item_type = struct.unpack("<I", _read_exact(stream, 4))[0]
        count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        return [_read_typed_value(stream, item_type) for _ in range(count)]
    return _read_scalar(stream, value_type)


@dataclass
class GGUFTensor:
    """One tensor's table entry, and where its bytes are."""

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int          # relative to the start of the tensor-data region
    nbytes: int = 0      # filled in on read; recomputed on write

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


@dataclass
class GGUFFile:
    """A parsed GGUF: metadata, tensor table, and access to tensor bytes."""

    path: Path | None = None
    version: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Value type per metadata key, so a round trip re-emits what it read
    #: rather than re-deriving a type from the Python object and turning
    #: every int into an INT32.
    metadata_types: dict[str, tuple[int, int | None]] = field(default_factory=dict)
    tensors: list[GGUFTensor] = field(default_factory=list)
    alignment: int = DEFAULT_ALIGNMENT
    data_start: int = 0

    @classmethod
    def read(cls, path: str | Path) -> GGUFFile:
        """Parse the header, metadata and tensor table of *path*.

        Tensor *data* is not read — a quantiser streams it one tensor at
        a time, and a 70B model does not fit in memory twice.
        """
        path = Path(path)
        with path.open("rb") as stream:
            magic = _read_exact(stream, 4)
            if magic != GGUF_MAGIC:
                raise GGUFError(
                    f"{path} does not start with the GGUF magic (got {magic!r})."
                )
            version = struct.unpack("<I", _read_exact(stream, 4))[0]
            if version not in _SUPPORTED_VERSIONS:
                raise GGUFError(
                    f"GGUF version {version} is not supported "
                    f"(this reads {', '.join(map(str, _SUPPORTED_VERSIONS))})."
                )
            tensor_count = struct.unpack("<Q", _read_exact(stream, 8))[0]
            kv_count = struct.unpack("<Q", _read_exact(stream, 8))[0]

            metadata: dict[str, Any] = {}
            metadata_types: dict[str, tuple[int, int | None]] = {}
            for _ in range(kv_count):
                key = _read_string(stream)
                raw_type = struct.unpack("<I", _read_exact(stream, 4))[0]
                if raw_type == int(GGUFValueType.ARRAY):
                    item_type = struct.unpack("<I", _read_exact(stream, 4))[0]
                    count = struct.unpack("<Q", _read_exact(stream, 8))[0]
                    metadata[key] = [
                        _read_typed_value(stream, item_type) for _ in range(count)
                    ]
                    metadata_types[key] = (raw_type, item_type)
                else:
                    metadata[key] = _read_typed_value(stream, raw_type)
                    metadata_types[key] = (raw_type, None)

            tensors: list[GGUFTensor] = []
            for _ in range(tensor_count):
                name = _read_string(stream)
                dims = struct.unpack("<I", _read_exact(stream, 4))[0]
                shape = tuple(
                    struct.unpack("<Q", _read_exact(stream, 8))[0] for _ in range(dims)
                )
                ggml_type = struct.unpack("<I", _read_exact(stream, 4))[0]
                offset = struct.unpack("<Q", _read_exact(stream, 8))[0]
                tensors.append(
                    GGUFTensor(name=name, shape=shape, ggml_type=ggml_type, offset=offset)
                )

            alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT)
            here = stream.tell()
            data_start = here + (-here % alignment)

        for tensor in tensors:
            try:
                tensor.nbytes = tensor_nbytes_unchecked(
                    tensor.ggml_type, tensor.shape
                )
            except GGUFError:
                # An unknown type can still be copied byte for byte; the
                # length comes from the next tensor's offset instead.
                tensor.nbytes = 0
        _fill_unknown_lengths(tensors, path, data_start)

        return cls(
            path=path,
            version=version,
            metadata=metadata,
            metadata_types=metadata_types,
            tensors=tensors,
            alignment=alignment,
            data_start=data_start,
        )

    def tensor_bytes(self, tensor: GGUFTensor) -> bytes:
        """The raw bytes of one tensor, read on demand."""
        if self.path is None:
            raise GGUFError("This GGUFFile was not read from a path.")
        with self.path.open("rb") as stream:
            stream.seek(self.data_start + tensor.offset)
            return _read_exact(stream, tensor.nbytes)

    def get(self, name: str) -> GGUFTensor | None:
        return next((t for t in self.tensors if t.name == name), None)

    @property
    def total_elements(self) -> int:
        return sum(t.elements for t in self.tensors)


def _fill_unknown_lengths(
    tensors: list[GGUFTensor], path: Path, data_start: int
) -> None:
    """Length of an unknown-typed tensor, from where the next one starts."""
    if not tensors:
        return
    ordered = sorted(tensors, key=lambda t: t.offset)
    file_size = path.stat().st_size
    for index, tensor in enumerate(ordered):
        if tensor.nbytes:
            continue
        if index + 1 < len(ordered):
            tensor.nbytes = ordered[index + 1].offset - tensor.offset
        else:
            tensor.nbytes = max(0, file_size - data_start - tensor.offset)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _write_string(stream: BinaryIO, text: str) -> None:
    encoded = text.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)


def _infer_type(value: Any) -> tuple[int, int | None]:
    """A GGUF type for a Python value, when the original type is unknown."""
    if isinstance(value, bool):
        return int(GGUFValueType.BOOL), None
    if isinstance(value, int):
        return int(GGUFValueType.INT64 if abs(value) > 2**31 - 1 else GGUFValueType.INT32), None
    if isinstance(value, float):
        return int(GGUFValueType.FLOAT32), None
    if isinstance(value, str):
        return int(GGUFValueType.STRING), None
    if isinstance(value, (list, tuple)):
        if not value:
            return int(GGUFValueType.ARRAY), int(GGUFValueType.INT32)
        item_type, _ = _infer_type(value[0])
        return int(GGUFValueType.ARRAY), item_type
    raise GGUFError(f"Cannot represent {type(value).__name__} in GGUF metadata.")


def _write_typed_value(stream: BinaryIO, raw_type: int, value: Any) -> None:
    value_type = GGUFValueType(raw_type)
    if value_type is GGUFValueType.STRING:
        _write_string(stream, str(value))
        return
    fmt = _SCALAR_FORMAT[value_type]
    if value_type is GGUFValueType.BOOL:
        value = bool(value)
    elif value_type in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        value = float(value)
    else:
        value = int(value)
    stream.write(struct.pack(fmt, value))


class GGUFWriter:
    """Write a GGUF file, tensor data streamed in whatever order suits.

    Two passes are unavoidable: the tensor table records each tensor's
    offset, and an offset is not known until every tensor before it has
    been sized. So tensors are declared first (name, shape, type), the
    header is written, and the data follows in declaration order.
    """

    def __init__(self, path: str | Path, *, alignment: int = DEFAULT_ALIGNMENT) -> None:
        self.path = Path(path)
        self.alignment = alignment
        self.metadata: dict[str, Any] = {}
        self.metadata_types: dict[str, tuple[int, int | None]] = {}
        self._tensors: list[GGUFTensor] = []

    def set_metadata(
        self, key: str, value: Any, *, type_hint: tuple[int, int | None] | None = None
    ) -> None:
        self.metadata[key] = value
        self.metadata_types[key] = type_hint or _infer_type(value)

    def copy_metadata_from(self, source: GGUFFile) -> None:
        """Carry every key across, types included.

        Not a filtered copy: dropping keys a reader does not recognise is
        how a round trip silently strips a chat template, a rope scaling
        factor or a tokenizer.
        """
        for key, value in source.metadata.items():
            self.metadata[key] = value
            self.metadata_types[key] = source.metadata_types.get(key) or _infer_type(value)

    def add_tensor(self, name: str, shape: tuple[int, ...], ggml_type: int) -> GGUFTensor:
        tensor = GGUFTensor(
            name=name,
            shape=tuple(int(d) for d in shape),
            ggml_type=int(ggml_type),
            offset=0,
        )
        tensor.nbytes = tensor_nbytes(tensor.ggml_type, tensor.shape)
        self._tensors.append(tensor)
        return tensor

    def _layout(self) -> None:
        offset = 0
        for tensor in self._tensors:
            tensor.offset = offset
            offset += tensor.nbytes
            offset += -offset % self.alignment

    def write(self, data_for: Any) -> None:
        """Write the file. ``data_for(tensor) -> bytes`` supplies each tensor.

        Called one tensor at a time and in declaration order, so a
        caller can quantise on demand instead of holding a whole model.
        """
        self.metadata.setdefault("general.alignment", self.alignment)
        self.metadata_types.setdefault(
            "general.alignment", (int(GGUFValueType.UINT32), None)
        )
        self._layout()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as stream:
            stream.write(GGUF_MAGIC)
            stream.write(struct.pack("<I", 3))
            stream.write(struct.pack("<Q", len(self._tensors)))
            stream.write(struct.pack("<Q", len(self.metadata)))

            for key, value in self.metadata.items():
                _write_string(stream, key)
                raw_type, item_type = self.metadata_types.get(key) or _infer_type(value)
                stream.write(struct.pack("<I", raw_type))
                if raw_type == int(GGUFValueType.ARRAY):
                    item_type = item_type if item_type is not None else _infer_type(value)[1]
                    stream.write(struct.pack("<I", int(item_type)))
                    stream.write(struct.pack("<Q", len(value)))
                    for item in value:
                        _write_typed_value(stream, int(item_type), item)
                else:
                    _write_typed_value(stream, raw_type, value)

            for tensor in self._tensors:
                _write_string(stream, tensor.name)
                stream.write(struct.pack("<I", len(tensor.shape)))
                for dim in tensor.shape:
                    stream.write(struct.pack("<Q", int(dim)))
                stream.write(struct.pack("<I", int(tensor.ggml_type)))
                stream.write(struct.pack("<Q", int(tensor.offset)))

            here = stream.tell()
            stream.write(b"\x00" * (-here % self.alignment))

            data_start = stream.tell()
            for tensor in self._tensors:
                stream.seek(data_start + tensor.offset)
                payload = data_for(tensor)
                if len(payload) != tensor.nbytes:
                    raise GGUFError(
                        f"{tensor.name}: supplied {len(payload)} bytes, "
                        f"the table says {tensor.nbytes}."
                    )
                stream.write(payload)
            # The last tensor's padding, so the file length matches the
            # layout a reader will compute from the table.
            end = stream.tell()
            stream.write(b"\x00" * (-end % self.alignment))
