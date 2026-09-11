"""hypernix.models.hnxrun — run a GGUF, including the sub-bit ones.

The IQ0.x tiers have been real quantisations since 0.72.3 pt 2: the
tensors genuinely carry 0.56 bits per weight and the file is a
well-formed GGUF. What they have never been is *runnable*. The type ids
are at 200 and above, which no llama.cpp knows and which even the
reference ``gguf`` Python reader rejects outright::

    ValueError: np.uint32(202) is not a valid GGMLQuantizationType

So :mod:`hypernix.models.ggufrun` refused them by name, and a person who
quantised a model to IQ0.5 got a file that was correct, small, and
useless. "It is a real quantisation" is not much comfort when nothing
will load it.

This is the loader and the forward pass that make it a model. It reads
any GGUF this package can write — F32/F16/BF16, the llama.cpp block
types via :mod:`hypernix.quant.llamaquants`, and the HyperNix sub-bit
types via :mod:`hypernix.quant.subbit` — dequantises every tensor to
torch, and runs the llama-family architecture over it.

What it is and is not
---------------------
It is a *reference* implementation: correctness first, in torch, with no
kernels and no fused anything. It will not beat llama.cpp and is not
trying to; the point is that a 0.5-bit model has somewhere to run at
all. For an upstream quant type, llama.cpp remains the right answer and
:mod:`hypernix.models.ggufrun` still sends it there.

The RoPE convention
-------------------
llama.cpp's converter permutes Q and K on the way into a GGUF so that
rotary embedding applies to *adjacent pairs* rather than to split
halves. That permutation is why reading these tensors back and applying
the half-split form — which is what every Hugging Face implementation
does — produces a model that loads, runs, and generates confident
nonsense. Adjacent pairs is the convention here, matching the file.

Attention is causal with a KV cache; grouped-query attention is handled
by repeating the KV heads. Both are checked against a full recompute in
the tests, because a cache that drifts from the uncached path is the
classic way this goes subtly wrong.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "HnxRunError",
    "HnxEnvironmentError",
    "ModelConfig",
    "LoadedModel",
    "PackedWeight",
    "load_model",
    "generate_tokens",
    "continue_text",
    "generate_text",
    "describe",
]


class HnxRunError(RuntimeError):
    """The model could not be loaded or run, with a reason worth reading."""


class HnxEnvironmentError(HnxRunError):
    """This *machine* cannot run it -- torch is missing or the device is not.

    A subclass rather than a sibling so every existing ``except
    HnxRunError`` still catches it, but callers that report a failure
    against a filename can tell the two apart. "model.gguf: PyTorch is
    installed but cannot load libcusparseLt.so.0" blames a file for a
    broken install, and sends the reader to re-download the model.
    """


# ---------------------------------------------------------------------------
# Configuration, read from the file rather than guessed
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """The architecture, as the GGUF describes itself."""

    architecture: str = "llama"
    block_count: int = 0
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    feed_forward_length: int = 0
    context_length: int = 2048
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_dimension_count: int = 0
    vocab_size: int = 0

    @property
    def head_dim(self) -> int:
        return self.embedding_length // self.head_count if self.head_count else 0

    @property
    def kv_groups(self) -> int:
        """How many query heads share each key/value head."""
        return self.head_count // self.head_count_kv if self.head_count_kv else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "block_count": self.block_count,
            "embedding_length": self.embedding_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "head_dim": self.head_dim,
            "feed_forward_length": self.feed_forward_length,
            "context_length": self.context_length,
            "rope_theta": self.rope_theta,
            "vocab_size": self.vocab_size,
        }


#: Architectures whose tensor names and block shape this forward pass
#: matches. Refusing a name we do not know beats running the llama graph
#: over a model that is not one and reporting the result as generation.
SUPPORTED_ARCHITECTURES = ("llama", "mistral", "qwen2", "hypernix")


def _metadata_int(meta: dict, *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in meta:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                continue
    return default


def _metadata_float(meta: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                continue
    return default


def read_config(metadata: dict, tensors: dict) -> ModelConfig:
    """The architecture from the file's own metadata.

    Falls back to the tensor shapes for anything the metadata omits —
    a converter that dropped ``feed_forward_length`` still produced a
    model whose ``ffn_up`` says how wide the MLP is, and refusing to run
    it over a missing key nobody reads would be pedantry.
    """
    arch = str(metadata.get("general.architecture", "llama"))
    prefix = arch
    config = ModelConfig(
        architecture=arch,
        block_count=_metadata_int(metadata, f"{prefix}.block_count"),
        embedding_length=_metadata_int(metadata, f"{prefix}.embedding_length"),
        head_count=_metadata_int(metadata, f"{prefix}.attention.head_count"),
        head_count_kv=_metadata_int(metadata, f"{prefix}.attention.head_count_kv"),
        feed_forward_length=_metadata_int(metadata, f"{prefix}.feed_forward_length"),
        context_length=_metadata_int(metadata, f"{prefix}.context_length", default=2048),
        rms_eps=_metadata_float(
            metadata, f"{prefix}.attention.layer_norm_rms_epsilon", default=1e-5
        ),
        rope_theta=_metadata_float(metadata, f"{prefix}.rope.freq_base", default=10000.0),
        rope_dimension_count=_metadata_int(metadata, f"{prefix}.rope.dimension_count"),
    )

    embd = tensors.get("token_embd.weight")
    if embd is not None:
        config.vocab_size = int(embd.shape[0])
        if not config.embedding_length:
            config.embedding_length = int(embd.shape[1])
    # Everything below reads .shape, which a PackedWeight has too -- the
    # config must not depend on whether the weights were materialised.
    if not config.block_count:
        config.block_count = 1 + max(
            (
                int(name.split(".")[1])
                for name in tensors
                if name.startswith("blk.") and name.split(".")[1].isdigit()
            ),
            default=-1,
        )
    if not config.head_count:
        config.head_count = 1
    if not config.head_count_kv:
        kv = tensors.get("blk.0.attn_k.weight")
        if kv is not None and config.head_dim:
            config.head_count_kv = max(1, int(kv.shape[0]) // config.head_dim)
        else:
            config.head_count_kv = config.head_count
    if not config.feed_forward_length:
        up = tensors.get("blk.0.ffn_up.weight")
        if up is not None:
            config.feed_forward_length = int(up.shape[0])
    if not config.rope_dimension_count:
        config.rope_dimension_count = config.head_dim
    return config


# ---------------------------------------------------------------------------
# Reading the weights, whatever they are packed as
# ---------------------------------------------------------------------------


#: Ceiling on the transient buffer a packed matmul unpacks into.
CHUNK_BYTES = 8 << 20

#: HNX sign-and-scale GGML type -> the packing name subbit.py knows it
#: by. These are the ones the folded matmul applies to, because they are
#: the ones with dropped signs to reconstruct.
_SUB_BIT_PACKINGS = {
    200: "sign_scale_l",
    201: "pair_code_m",
    202: "quad_code_xxxl",
    203: "quarter_code_uxl",
    204: "int1_binary",
    207: "hnx_1375bit",
}

#: HNX fixed-codebook GGML type -> the codec name lowbit.py knows it by.
#: These carry magnitude, so there is nothing to fold: every weight has
#: its own code and the matmul is the ordinary one.
_LOW_BIT_CODECS = {205: "INT4", 206: "FP2"}


def _dequantize(raw: bytes, ggml_type: int, elements: int):
    """Tensor bytes to a flat float32 numpy array, for every type we write."""
    import numpy as np

    from ..quant import llamaquants
    from ..quant.gguf import GGMLType
    from ..quant.subbit import dequantize_array as subbit_dequantize

    kind = int(ggml_type)
    if kind == int(GGMLType.F32):
        return np.frombuffer(raw, dtype="<f4", count=elements).astype(np.float32)
    if kind == int(GGMLType.F16):
        return np.frombuffer(raw, dtype="<f2", count=elements).astype(np.float32)
    if kind == int(GGMLType.BF16):
        # The top 16 bits of an F32, so widening is a shift.
        upper = np.frombuffer(raw, dtype="<u2", count=elements).astype(np.uint32)
        return (upper << 16).view(np.float32).astype(np.float32)
    if kind == int(GGMLType.F64):
        return np.frombuffer(raw, dtype="<f8", count=elements).astype(np.float32)
    if llamaquants.is_supported(kind):
        # Already a numpy array, and already float32: no per-element
        # Python anywhere on this path.
        return llamaquants.dequantize_array(raw, kind)[:elements]
    if kind in _SUB_BIT_PACKINGS:
        return subbit_dequantize(raw, _SUB_BIT_PACKINGS[kind])[:elements]
    if kind in _LOW_BIT_CODECS:
        from ..quant.lowbit import dequantize_array as lowbit_dequantize

        return lowbit_dequantize(raw, _LOW_BIT_CODECS[kind])[:elements]
    raise HnxRunError(
        f"GGML type {kind} is one this runtime cannot decode. It reads F32, F16, "
        f"BF16, the llama.cpp block types, and the HyperNix extension types."
    )


def _block_geometry(ggml_type: int) -> tuple[int, int]:
    """``(elements per block, bytes per block)`` for a quantised type."""
    from ..quant import llamaquants
    from ..quant.subbit import BLOCK_SIZE, packed_block_bytes

    kind = int(ggml_type)
    if kind in _SUB_BIT_PACKINGS:
        return BLOCK_SIZE, packed_block_bytes(_SUB_BIT_PACKINGS[kind])
    if kind in _LOW_BIT_CODECS:
        from ..quant.lowbit import BLOCK_SIZE as LOW_BLOCK
        from ..quant.lowbit import packed_block_bytes as low_block_bytes

        return LOW_BLOCK, low_block_bytes(_LOW_BIT_CODECS[kind])
    fmt = llamaquants.BY_TYPE.get(kind)
    if fmt is None:
        raise HnxRunError(f"GGML type {kind} has no block geometry here.")
    return fmt.block, fmt.block_bytes


class PackedWeight:
    """A quantised weight that stays packed in memory.

    The whole point of a 0.5-bit model is that it is small. Dequantising
    it to float32 at load time gives that away entirely -- 0.56 bits on
    disk becomes 32 bits resident, which is *larger* than the F16 model
    it was made from and makes the tier pointless for the machines it
    exists to serve.

    So the packed bytes are what is held, and rows are unpacked a group
    at a time into a small buffer that is thrown away. Resident cost is
    the file's own size plus one group; the arithmetic is done in float32
    as it must be, but only ever on a slice.

    Slicing granularity
    -------------------
    The packed stream is blocks over the *flattened* tensor, so a slice
    has to land on a block boundary. One row is not always a whole number
    of blocks -- a 64-wide layer packed in 256-element blocks puts four
    rows in one block -- so the unit is the smallest run of rows that is:
    ``block // gcd(block, columns)``. For any real model that is one row;
    for a narrow one it is a handful, and either way the arithmetic is
    exact rather than approximately aligned.

    The fold
    --------
    For the sub-bit types the unpacking never reaches one float per
    weight. A dropped position repeats its group's last stored sign, so
    the input can be folded to match -- see :meth:`_fold_input` -- and
    the dot product then runs against the ``kept`` signs alone. At
    ``IQ0.5_XXXL`` that is half the arithmetic and none of the expansion,
    and the expansion was the larger half. It changes the sum's
    association, so the folded result agrees with the materialised one to
    float32 rounding rather than exactly.

    The cost is time: every forward pass re-unpacks. That is the trade a
    reference runtime should make by default, and :func:`load_model`
    takes ``materialize=True`` for anyone who would rather spend the
    memory, or ``cache_bytes`` for a budget between the two.
    """

    __slots__ = ("raw", "ggml_type", "shape", "device", "_block", "_block_bytes",
                 "_rows_per_group", "_group_bytes", "_pinned", "_packing",
                 "_device_packed", "_decode_here")

    def __init__(self, raw: bytes, ggml_type: int, shape: tuple[int, ...],
                 device: str, *, decode_on_device: bool | None = None):
        from math import gcd

        self._pinned = None
        self._device_packed = None
        self.raw = raw
        self.ggml_type = int(ggml_type)
        self.shape = tuple(int(d) for d in shape)
        self.device = device
        self._block, self._block_bytes = _block_geometry(ggml_type)

        columns = self.shape[-1]
        rows = self.shape[0]
        if (rows * columns) % self._block:
            raise HnxRunError(
                f"A tensor of {rows * columns} elements is not a whole number of "
                f"{self._block}-element blocks, so it cannot have been packed."
            )
        self._rows_per_group = self._block // gcd(self._block, columns)
        self._group_bytes = (
            self._rows_per_group * columns // self._block
        ) * self._block_bytes
        self._packing = self._folding_packing(columns)
        self._decode_here = decode_on_device

    def _folding_packing(self, columns: int) -> str | None:
        """The packing name if this weight can take the folded matmul.

        Only sub-bit types can, and only when a sign group sits inside a
        single row: the fold rewrites ``x`` per group, so a group that
        straddles a row boundary would mix two rows' inputs. ``columns``
        is a multiple of 8 in every real model, so this is a guard rather
        than a limitation -- but it fails quietly and wrongly if left
        unchecked, which is the kind that ships.
        """
        from ..quant.subbit import PACKINGS

        name = _SUB_BIT_PACKINGS.get(self.ggml_type)
        if name is None:
            return None
        return name if columns % PACKINGS[name].group == 0 else None

    @property
    def on_accelerator(self) -> bool:
        """Whether this weight lives somewhere that is not the CPU."""
        return not str(self.device).startswith("cpu")

    def _decodes_on_device(self) -> bool:
        """Whether the packed bytes can be decoded where they sit.

        Only the HyperNix extension types can. The llama.cpp block types
        still decode through numpy and upload the result, which is the
        arrangement this exists to avoid -- but ``hypernix generate``
        routes an upstream quant to llama.cpp anyway, so the slow path is
        one nobody takes to run a model.
        """
        from .hnxtorch import supports

        if not supports(self.ggml_type):
            return False
        if self._decode_here is not None:
            # Forced on or off. On is how the accelerator path gets
            # tested on a machine with no accelerator: it is the same
            # code CUDA runs, and asserting it against the numpy decoder
            # is the only check of it that does not need a GPU present.
            return bool(self._decode_here)
        return self.on_accelerator

    def packed_on_device(self):
        """The packed bytes as a uint8 tensor on this weight's device.

        Uploaded once and kept. This is the whole reason a sub-bit model
        is worth putting on a GPU: the packed form is the small one, so
        PCIe carries 0.9 bits per weight at load rather than 32 bits per
        weight per forward pass.
        """
        if self._device_packed is None:
            from .hnxtorch import to_device_bytes

            self._device_packed = to_device_bytes(self.raw, self.device)
        return self._device_packed

    @property
    def nbytes(self) -> int:
        """What this weight costs in memory, pinned or not."""
        if self._pinned is not None:
            return self._pinned.numel() * self._pinned.element_size()
        return len(self.raw)

    @property
    def device_bytes(self) -> int:
        """What this weight occupies in the device's own memory.

        Reported separately from :attr:`nbytes` rather than folded into
        it: the host copy is still held, and a single number covering
        both would be a memory figure that matches neither the RAM the
        process uses nor the VRAM the card reports.

        Non-zero on CPU only when the torch decoder has been forced on,
        which is a testing mode -- there it is a real second allocation
        bought for nothing, and saying zero would hide it.
        """
        if self._pinned is not None and self.on_accelerator:
            return self._pinned.numel() * self._pinned.element_size()
        if self._device_packed is None:
            return 0
        return self._device_packed.numel()

    @property
    def dense_bytes(self) -> int:
        """What pinning this weight would cost."""
        total = 4
        for dim in self.shape:
            total *= int(dim)
        return total

    @property
    def pinned(self) -> bool:
        return self._pinned is not None

    def pin(self) -> None:
        """Decode once and keep it, trading the memory for the time.

        Every forward pass re-unpacks an unpinned weight. Pinning the
        ones a budget can afford is the dial between "costs what the file
        costs" and "runs at float32 speed" -- and because the traversal
        order is fixed, pinning the largest tensors saves the most decode
        work per byte spent.
        """
        if self._pinned is None:
            self._pinned = self.rows_slice(0, self.rows)

    def unpin(self) -> None:
        self._pinned = None

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def columns(self) -> int:
        return self.shape[-1]

    def _decode_groups(self, first_group: int, last_group: int):
        """Rows from whole groups ``[first_group, last_group)``, as float32."""
        import torch

        start = first_group * self._group_bytes
        stop = last_group * self._group_bytes
        if self._decodes_on_device():
            from .hnxtorch import decode

            flat = decode(self.packed_on_device()[start:stop], self.ggml_type)
            return flat.reshape(-1, self.columns)
        chunk = self.raw[start:stop]
        count = (last_group - first_group) * self._rows_per_group * self.columns
        # No .copy(): every decoder returns a freshly allocated array, so
        # torch can take it as is. Copying here doubled the traffic of the
        # hottest loop in packed mode for nothing.
        flat = _dequantize(chunk, self.ggml_type, count)
        return torch.from_numpy(flat.reshape(-1, self.columns)).to(self.device)

    def rows_slice(self, start: int, stop: int):
        """Rows ``[start, stop)``, unpacking only the groups they touch."""
        if self._pinned is not None:
            return self._pinned[start:stop]
        first = start // self._rows_per_group
        last = -(-stop // self._rows_per_group)  # ceiling division
        decoded = self._decode_groups(first, last)
        offset = start - first * self._rows_per_group
        return decoded[offset:offset + (stop - start)]

    def select(self, indices):
        """The rows *indices* names, unpacked and nothing else.

        This is what makes an embedding lookup affordable: the table is
        usually the largest tensor in the file, and a prompt touches a
        handful of its rows.
        """
        import torch

        wanted = [int(i) for i in indices]
        decoded = {row: self.rows_slice(row, row + 1)[0] for row in sorted(set(wanted))}
        return torch.stack([decoded[row] for row in wanted])

    def to_dense(self):
        """The whole tensor as float32, for a caller that wants it.

        Defeats the point of holding it packed, so it is a request rather
        than the default: inspection, comparison, a debugger.
        """
        return self.rows_slice(0, self.rows)

    def _chunk_step(self, width: int) -> int:
        """Rows per chunk, so the transient buffer stays near CHUNK_BYTES.

        Bounded by bytes rather than rows so peak memory is a property of
        this runtime rather than of how wide someone's FFN happens to be.
        Bigger than a megabyte because each chunk costs a fixed amount of
        numpy call overhead, and eight is still nothing next to the
        float32 model this exists to avoid holding.
        """
        rows = max(1, CHUNK_BYTES // (max(width, 1) * 4))
        return max(self._rows_per_group, rows - rows % self._rows_per_group)

    def _folded_signs(self, start: int, stop: int):
        """Rows ``[start, stop)`` as *stored* signs, ``(rows, columns * k // g)``.

        Never widened to one value per weight. That expansion is the
        largest allocation on the packed path and the fold makes it
        unnecessary.
        """
        import torch

        from ..quant.subbit import stored_signs

        first = start // self._rows_per_group
        last = -(-stop // self._rows_per_group)  # ceiling division
        rows = (last - first) * self._rows_per_group
        window = slice(first * self._group_bytes, last * self._group_bytes)
        if self._decodes_on_device():
            from .hnxtorch import stored_signs as device_stored_signs

            decoded = device_stored_signs(
                self.packed_on_device()[window], self._packing
            ).reshape(rows, -1)
        else:
            decoded = torch.from_numpy(
                stored_signs(self.raw[window], self._packing).reshape(rows, -1)
            ).to(self.device)
        offset = start - first * self._rows_per_group
        return decoded[offset:offset + (stop - start)]

    def _fold_input(self, x):
        """``x``, with each group's last stored position absorbing the rest.

        A dropped position repeats its group's last stored sign, so for a
        group of ``g`` weights of which ``k`` signs survive::

            sum_j sign[j] * x[j] == sum_{j<k-1} sign[j] * x[j]
                                    + sign[k-1] * sum_{j>=k-1} x[j]

        Fold the input that way and the dot product can run against the
        stored signs alone -- ``k/g`` of the weights, so half the arithmetic
        at ``IQ0.5_XXXL`` and none of the expansion. The result is the same
        sum with its terms gathered differently, so it agrees with the
        materialised path to float32 rounding rather than exactly.
        """
        import torch

        from ..quant.subbit import PACKINGS

        spec = PACKINGS[self._packing]
        grouped = x.reshape(*x.shape[:-1], self.columns // spec.group, spec.group)
        folded = torch.cat(
            (
                grouped[..., :spec.kept - 1],
                grouped[..., spec.kept - 1:].sum(-1, keepdim=True),
            ),
            dim=-1,
        )
        return folded.reshape(*x.shape[:-1], -1)

    def matmul_t(self, x, *, chunk_rows: int = 0):
        """``x @ self.T``, unpacking the weight a chunk of rows at a time."""
        import torch

        if self._pinned is not None:
            return x @ self._pinned.T

        folding = self._packing is not None
        if folding:
            from ..quant.subbit import PACKINGS

            spec = PACKINGS[self._packing]
            width = self.columns * spec.kept // spec.group
            operand = self._fold_input(x)
        else:
            width = self.columns
            operand = x

        if chunk_rows <= 0:
            step = self._chunk_step(width)
        else:
            step = max(
                self._rows_per_group, chunk_rows - chunk_rows % self._rows_per_group
            )

        out = torch.empty(
            (*x.shape[:-1], self.rows), dtype=torch.float32, device=self.device
        )
        for start in range(0, self.rows, step):
            stop = min(start + step, self.rows)
            rows = (
                self._folded_signs(start, stop)
                if folding
                else self.rows_slice(start, stop)
            )
            out[..., start:stop] = operand @ rows.T
        return out


def _linear(x, weight, *, chunk_rows: int = 0):
    """``x @ weight.T`` whether the weight is packed or materialised."""
    if isinstance(weight, PackedWeight):
        return weight.matmul_t(x, chunk_rows=chunk_rows)
    return x @ weight.T


def _embed(weight, ids):
    if isinstance(weight, PackedWeight):
        return weight.select(ids)
    return weight[ids]


def _weight_shape(weight) -> tuple[int, ...]:
    return tuple(weight.shape)


def _weight_bytes(weight) -> int:
    if isinstance(weight, PackedWeight):
        return weight.nbytes
    return weight.numel() * weight.element_size()


def _weight_elements(weight) -> int:
    total = 1
    for dim in _weight_shape(weight):
        total *= int(dim)
    return total


@dataclass
class LoadedModel:
    """A GGUF, dequantised and ready to run."""

    config: ModelConfig
    tensors: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    path: str = ""
    #: ``{ggml type name: tensor count}`` — what the file was packed as.
    packed_as: dict = field(default_factory=dict)
    tokenizer: Any = None
    #: Bytes on disk, for comparison with what the load actually cost.
    file_bytes: int = 0
    #: Where the weights are. "cpu", "cuda:0", "mps", ...
    device: str = "cpu"

    @property
    def device_bytes(self) -> int:
        """What this model occupies on an accelerator.

        Separate from :attr:`resident_bytes`, which counts host memory.
        On a GPU the packed bytes are held in both places -- the file's
        copy on the host and the uploaded copy on the card -- and one
        number covering both would match neither what the process uses
        nor what nvidia-smi reports.
        """
        total = 0
        for weight in self.tensors.values():
            if isinstance(weight, PackedWeight):
                total += weight.device_bytes
            elif hasattr(weight, "device") and str(weight.device) != "cpu":
                total += weight.numel() * weight.element_size()
        return total

    @property
    def sub_bit(self) -> bool:
        return any(name.startswith("HNX_") for name in self.packed_as)

    @property
    def resident_bytes(self) -> int:
        """What the weights occupy in memory, packed or not."""
        return sum(_weight_bytes(w) for w in self.tensors.values())

    @property
    def weight_count(self) -> int:
        return sum(_weight_elements(w) for w in self.tensors.values())

    @property
    def resident_bits_per_weight(self) -> float:
        """The number that says whether this is still a sub-bit model.

        On disk is the easy half. A runtime that dequantises to float32
        at load time turns 0.56 bits into 32 and hands back none of the
        saving that made the tier worth having.
        """
        count = self.weight_count
        return (self.resident_bytes * 8 / count) if count else 0.0

    @property
    def packed_in_memory(self) -> int:
        """How many tensors are still in their on-disk form."""
        return sum(
            1
            for w in self.tensors.values()
            if isinstance(w, PackedWeight) and not w.pinned
        )

    @property
    def pinned_in_memory(self) -> int:
        """How many packed weights were decoded once and kept."""
        return sum(
            1 for w in self.tensors.values() if isinstance(w, PackedWeight) and w.pinned
        )

    def describe(self) -> str:
        mix = ", ".join(f"{k} x{v}" for k, v in sorted(self.packed_as.items()))
        lines = [
            f"{self.config.architecture}: {self.config.block_count} blocks, "
            f"{self.config.embedding_length} wide, "
            f"{self.config.head_count} heads "
            f"({self.config.head_count_kv} kv), vocab {self.config.vocab_size}",
            f"  packed as: {mix}",
            f"  resident : {self.resident_bytes / 1e6:.2f} MB "
            f"({self.resident_bits_per_weight:.3f} bits/weight), "
            f"{self.packed_in_memory}/{len(self.tensors)} tensors still packed"
            + (f", {self.pinned_in_memory} pinned" if self.pinned_in_memory else ""),
        ]
        if str(self.device) != "cpu":
            lines.append(
                f"  device   : {self.device}, {self.device_bytes / 1e6:.2f} MB "
                f"resident there (packed)"
            )
        if self.file_bytes:
            lines.append(
                f"  on disk  : {self.file_bytes / 1e6:.2f} MB "
                f"({self.resident_bytes / self.file_bytes:.2f}x resident)"
            )
        return "\n".join(lines)


#: Types that are already float and have nothing to unpack.
_FLOAT_TYPES = (0, 1, 28, 30)  # F32, F16, F64, BF16


def _spend_cache_budget(tensors: dict, budget: int) -> int:
    """Pin as many weights as *budget* affords, largest first.

    Largest first because every forward pass touches every tensor once,
    so there is no locality to exploit and no cache-replacement policy
    that helps: the only question is how much decode work a byte of
    budget buys, and a big tensor buys the most.
    """
    if budget <= 0:
        return 0
    candidates = sorted(
        (
            (weight.dense_bytes, name)
            for name, weight in tensors.items()
            if isinstance(weight, PackedWeight)
        ),
        reverse=True,
    )
    spent = 0
    for size, name in candidates:
        if spent + size > budget:
            continue
        tensors[name].pin()
        spent += size
    return spent


def load_model(
    path: str | Path,
    *,
    device: str = "cpu",
    materialize: bool = False,
    cache_bytes: int = 0,
) -> LoadedModel:
    """Read a GGUF, keeping quantised tensors packed in memory.

    A quantised tensor stays in its on-disk form and is unpacked a chunk
    of rows at a time inside each matmul, so a 0.5-bit model costs about
    what the file costs. That is the whole point of the tier: dequantising
    at load time would turn 0.56 bits into 32 and hand back every byte
    the quantisation saved.

    ``materialize=True`` dequantises everything to float32 up front
    instead — faster per token, and the memory profile of the model it
    was made from. Float tensors and one-dimensional weights (norms) are
    always materialised; they are a rounding error of the size either
    way.

    ``cache_bytes`` is the dial between the two. Weights are pinned
    largest-first until the budget is spent, so a machine with room for
    half the model in float32 spends it on the half that costs the most
    to unpack, and one with no room to spare keeps every byte packed.
    :attr:`LoadedModel.resident_bits_per_weight` reports where that
    landed rather than leaving it to be guessed.

    ``device`` takes ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``,
    ``"mps"``, ``"xpu"``, or a :class:`~hypernix.models.hnxdevice.Device`.
    On an accelerator the *packed* bytes are uploaded once and decoded
    there, which is the arrangement worth having: the packed form is the
    small one, so a 7B at ``IQ0.9_L`` puts about 800 MB on the card
    instead of the 28 GB its float16 weights would need. A named device
    that cannot run raises with the reason and the remedy rather than
    quietly falling back to the CPU — see
    :func:`hypernix.models.hnxdevice.select`.
    """
    import numpy as np

    from ..quant.gguf import GGMLType, GGUFError, GGUFFile
    from .hnxdevice import DeviceError, import_torch, select

    # Before anything else, and through import_torch rather than a bare
    # `import torch`: a broken CUDA install fails here with a linker
    # error naming a shared object, and this is the one place that can
    # turn that into a sentence about torch.
    try:
        torch = import_torch()
        resolved = select(str(device))
    except DeviceError as exc:
        raise HnxEnvironmentError(str(exc)) from exc
    device = resolved.torch_device

    model_path = Path(path)
    if not model_path.exists():
        raise HnxRunError(f"No such model: {model_path}")
    try:
        gguf_file = GGUFFile.read(model_path)
    except GGUFError as exc:
        raise HnxRunError(f"{model_path}: {exc}") from exc

    tensors: dict[str, Any] = {}
    packed: dict[str, int] = {}
    for tensor in gguf_file.tensors:
        raw = gguf_file.tensor_bytes(tensor)
        # GGUF stores shape fastest-dimension-first: (n_input, n_output)
        # for a weight matrix. Reversing gives the (out, in) that a
        # linear layer wants, and getting it backwards produces a model
        # that loads cleanly and multiplies the wrong way round.
        shape = tuple(int(d) for d in reversed(tensor.shape))
        kind = int(tensor.ggml_type)

        keep_packed = (
            not materialize and kind not in _FLOAT_TYPES and len(shape) > 1
        )
        if keep_packed:
            try:
                tensors[tensor.name] = PackedWeight(raw, kind, shape, device)
            except HnxRunError:
                # A tensor whose rows are not whole blocks cannot be
                # sliced; materialise that one rather than refusing the
                # model over a shape no real converter produces.
                keep_packed = False
        if not keep_packed:
            flat = _dequantize(raw, kind, tensor.elements)
            array = np.asarray(flat, dtype=np.float32).reshape(shape)
            tensors[tensor.name] = torch.from_numpy(array.copy()).to(device)

        try:
            name = GGMLType(kind).name
        except ValueError:  # pragma: no cover - guarded by _dequantize
            name = str(kind)
        packed[name] = packed.get(name, 0) + 1

    config = read_config(gguf_file.metadata, tensors)
    if config.architecture not in SUPPORTED_ARCHITECTURES:
        raise HnxRunError(
            f"This runtime implements the llama-family graph; the file says its "
            f"architecture is {config.architecture!r}. Running the wrong graph over "
            f"a model produces confident nonsense rather than an error, so it is "
            f"refused here. Known: {', '.join(SUPPORTED_ARCHITECTURES)}."
        )
    missing = [
        name
        for name in ("token_embd.weight",)
        if name not in tensors
    ]
    if missing:
        raise HnxRunError(
            f"{model_path} is missing {', '.join(missing)}, so there is no model to "
            f"run. It has {len(tensors)} tensor(s); this looks like a fragment or a "
            f"file that is not a language model."
        )

    if cache_bytes > 0:
        _spend_cache_budget(tensors, int(cache_bytes))

    from .hnxtokenizer import tokenizer_from_metadata

    return LoadedModel(
        config=config,
        tensors=tensors,
        metadata=dict(gguf_file.metadata),
        path=str(model_path),
        packed_as=packed,
        tokenizer=tokenizer_from_metadata(gguf_file.metadata),
        file_bytes=model_path.stat().st_size,
        device=device,
    )


def describe(path: str | Path) -> dict[str, Any]:
    """Architecture and packing, without running anything."""
    loaded = load_model(path)
    return {
        "path": str(path),
        **loaded.config.to_dict(),
        "packed_as": dict(sorted(loaded.packed_as.items())),
        "sub_bit": loaded.sub_bit,
        "has_tokenizer": loaded.tokenizer is not None,
        "file_bytes": loaded.file_bytes,
        "resident_bytes": loaded.resident_bytes,
        "resident_bits_per_weight": round(loaded.resident_bits_per_weight, 4),
        "tensors_still_packed": loaded.packed_in_memory,
    }


# ---------------------------------------------------------------------------
# The forward pass
# ---------------------------------------------------------------------------


def _rms_norm(x, weight, eps: float):
    import torch

    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _rope(x, positions, theta: float, rope_dims: int):
    """Rotary embedding over *adjacent pairs*, as the GGUF expects.

    ``x`` is ``(heads, seq, head_dim)``. llama.cpp's converter permutes
    Q and K precisely so that this — not the half-split form every
    Hugging Face implementation uses — is the right rotation for the
    stored weights.
    """
    import torch

    head_dim = x.shape[-1]
    dims = min(rope_dims or head_dim, head_dim)
    dims -= dims % 2
    if dims <= 0:
        return x

    rotated = x[..., :dims]
    passthrough = x[..., dims:]

    # device=x.device, not the default. Without it this table is built on
    # the CPU and multiplied by positions that live wherever the model
    # does, which is fine on a CPU run and a hard error on every
    # accelerator -- the whole forward pass dies with "found at least two
    # devices". Nothing here reads a device from a global, so the tensor
    # being rotated is the only authority on where the arithmetic goes.
    inverse = 1.0 / (
        theta
        ** (
            torch.arange(0, dims, 2, dtype=torch.float32, device=x.device)
            / dims
        )
    )
    angles = positions.to(torch.float32)[:, None] * inverse[None, :]
    cos = torch.cos(angles)[None, :, :]
    sin = torch.sin(angles)[None, :, :]

    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    out = torch.empty_like(rotated)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    if passthrough.shape[-1]:
        return torch.cat([out, passthrough], dim=-1)
    return out


def _repeat_kv(x, groups: int):
    """Grouped-query attention: one KV head serves *groups* query heads."""
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=0)


class _Cache:
    """Per-layer key/value history. Plain lists; correctness over speed."""

    def __init__(self, layers: int) -> None:
        self.keys: list[Any] = [None] * layers
        self.values: list[Any] = [None] * layers

    def __len__(self) -> int:
        first = self.keys[0] if self.keys else None
        return 0 if first is None else int(first.shape[1])


def forward(
    model: LoadedModel,
    token_ids,
    *,
    cache: _Cache | None = None,
    start_position: int = 0,
):
    """Logits for the last position of *token_ids*.

    Returns ``(logits, cache)``. Pass the returned cache back with the
    next token to continue without recomputing the prefix.
    """
    import torch

    config = model.config
    weights = model.tensors
    embedding = weights["token_embd.weight"]
    device = getattr(embedding, "device", "cpu")

    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).reshape(-1)
    if ids.numel() == 0:
        raise HnxRunError("Nothing to run: the token sequence is empty.")
    if int(ids.max()) >= config.vocab_size or int(ids.min()) < 0:
        raise HnxRunError(
            f"Token id out of range: this model's vocabulary is 0..{config.vocab_size - 1}."
        )

    hidden = _embed(weights["token_embd.weight"], ids)
    seq = ids.shape[0]
    positions = torch.arange(start_position, start_position + seq, device=device)
    if cache is None:
        cache = _Cache(config.block_count)

    head_dim = config.head_dim
    scale = 1.0 / math.sqrt(head_dim) if head_dim else 1.0

    def need(name: str):
        tensor = weights.get(name)
        if tensor is None:
            raise HnxRunError(f"{model.path} has no {name}; the model is incomplete.")
        return tensor

    for layer in range(config.block_count):
        prefix = f"blk.{layer}."
        normed = _rms_norm(hidden, need(prefix + "attn_norm.weight"), config.rms_eps)

        q = _linear(normed, need(prefix + "attn_q.weight"))
        k = _linear(normed, need(prefix + "attn_k.weight"))
        v = _linear(normed, need(prefix + "attn_v.weight"))

        q = q.view(seq, config.head_count, head_dim).transpose(0, 1)
        k = k.view(seq, config.head_count_kv, head_dim).transpose(0, 1)
        v = v.view(seq, config.head_count_kv, head_dim).transpose(0, 1)

        q = _rope(q, positions, config.rope_theta, config.rope_dimension_count)
        k = _rope(k, positions, config.rope_theta, config.rope_dimension_count)

        if cache.keys[layer] is not None:
            k = torch.cat([cache.keys[layer], k], dim=1)
            v = torch.cat([cache.values[layer], v], dim=1)
        cache.keys[layer] = k
        cache.values[layer] = v

        keys = _repeat_kv(k, config.kv_groups)
        values = _repeat_kv(v, config.kv_groups)

        scores = (q @ keys.transpose(1, 2)) * scale
        total = keys.shape[1]
        # Causal mask over absolute positions, so a cached prefix is
        # visible and the future never is -- including when this call
        # carries several new tokens at once.
        query_pos = positions[:, None]
        key_pos = torch.arange(total, device=device)[None, :]
        scores = scores.masked_fill((key_pos > query_pos)[None, :, :], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = (attention @ values).transpose(0, 1).reshape(seq, -1)

        hidden = hidden + _linear(context, need(prefix + "attn_output.weight"))

        ffn_norm = weights.get(prefix + "ffn_norm.weight")
        normed = (
            _rms_norm(hidden, ffn_norm, config.rms_eps) if ffn_norm is not None else hidden
        )
        gate = _linear(normed, need(prefix + "ffn_gate.weight"))
        up = _linear(normed, need(prefix + "ffn_up.weight"))
        hidden = hidden + _linear(
            torch.nn.functional.silu(gate) * up, need(prefix + "ffn_down.weight")
        )

    output_norm = weights.get("output_norm.weight")
    if output_norm is not None:
        hidden = _rms_norm(hidden, output_norm, config.rms_eps)

    # A model with no output head ties it to the embedding, which is the
    # common arrangement for small models and not an error.
    head = weights.get("output.weight", weights["token_embd.weight"])
    return _linear(hidden, head), cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_tokens(
    model: LoadedModel,
    prompt_ids,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
    stop_ids=(),
) -> list[int]:
    """Continue *prompt_ids*. ``temperature=0`` is greedy and deterministic."""
    import torch

    if max_new_tokens < 0:
        raise HnxRunError("max_new_tokens cannot be negative.")
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    ids = [int(t) for t in prompt_ids]
    if not ids:
        raise HnxRunError("Nothing to continue: the prompt has no tokens.")

    logits, cache = forward(model, ids)
    produced: list[int] = []
    stops = {int(t) for t in stop_ids}

    for _ in range(max_new_tokens):
        last = logits[-1]
        if temperature <= 0:
            token = int(torch.argmax(last))
        else:
            scaled = last / float(temperature)
            if top_k and top_k < scaled.numel():
                cut = torch.topk(scaled, int(top_k)).values[-1]
                scaled = scaled.masked_fill(scaled < cut, float("-inf"))
            probabilities = torch.softmax(scaled, dim=-1)
            # The draw happens on the CPU whatever the model runs on. The
            # generator above is a CPU one, and torch refuses a generator
            # whose device differs from the tensor's, so a device-resident
            # probability vector would raise here rather than sample. It
            # also means a seed picks the same sequence of draws on every
            # backend, which a per-device generator would not.
            token = int(
                torch.multinomial(probabilities.cpu(), 1, generator=generator)
            )
        produced.append(token)
        if token in stops:
            break
        logits, cache = forward(
            model, [token], cache=cache, start_position=len(ids) + len(produced) - 1
        )
    return produced


def continue_text(
    model: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
) -> str:
    """Continue *prompt* with a model that is already loaded.

    The text-in, text-out half of :func:`generate_tokens`, taking the
    model rather than a path so a caller holding one open -- a chat REPL,
    a server -- pays the load once. :func:`generate_text` is this plus
    the load, for a caller with only a filename.

    Needs the GGUF to have carried its tokenizer, which every real
    conversion does. Without one there is no way to turn text into the
    ids this model was trained on, and guessing an encoding would produce
    output that looks like a broken model rather than a missing
    tokenizer.
    """
    if model.tokenizer is None:
        raise HnxRunError(
            f"{model.path or 'This model'} carries no tokenizer metadata, so text "
            f"cannot be turned into tokens for it. Use generate_tokens() with ids "
            f"you already have, or convert the model with a tool that writes "
            f"tokenizer.ggml.*."
        )
    prompt_ids = model.tokenizer.encode(prompt)
    if not prompt_ids:
        raise HnxRunError("The prompt tokenised to nothing.")
    produced = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        stop_ids=model.tokenizer.stop_ids,
    )
    return model.tokenizer.decode(produced)


def generate_text(
    path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
    device: str = "cpu",
    cache_bytes: int = 0,
) -> str:
    """Load *path* and continue *prompt*.

    One shot: the model is loaded, used and dropped. Anything that
    generates more than once should :func:`load_model` and call
    :func:`continue_text`, because a GGUF load is seconds to minutes and
    paying it per message makes a REPL unusable.
    """
    model = load_model(path, device=device, cache_bytes=cache_bytes)
    return continue_text(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
