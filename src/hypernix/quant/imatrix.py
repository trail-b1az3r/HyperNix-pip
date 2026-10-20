"""hypernix.quant.imatrix — make an importance matrix, and read anyone's.

An importance matrix is one number per *input channel* of every matmul
in the model: the mean of that channel's activation squared, measured
over calibration text. It is not a property of the weights. Two models
with identical weights and different training data want different
imatrices, and no amount of staring at a weight tensor produces one —
which is why "derive it from the weights" is not an option this module
offers, however convenient that would be.

What it buys you is where the quantiser spends its error. Below about
four bits a block cannot represent everything in it, and the difference
between a usable Q3_K and an unusable one is almost entirely which
channels it decided to protect. For the HyperNix sub-bit tiers it is not
a refinement at all but the whole mechanism: at 0.5 bits there is no
magnitude left to allocate, and all an imatrix can do — all there *is*
to do — is decide which signs survive.

How it is measured
------------------
Forward hooks on every linear layer, accumulating ``sum(x**2)`` per
input feature across calibration tokens. That is what llama.cpp's
``imatrix`` tool does, and doing the same thing means the numbers mean
the same thing: an imatrix from here works in ``llama-quantize``, and
one from the community works in :mod:`hypernix.quant.hyprslug`.

Both file formats are read and written. llama.cpp's binary ``.imatrix``
is the one people share; the JSON is the one you can look at when the
answer is wrong and you need to know why.

The honest limit
----------------
This needs to run the model, which means it needs the model in a
framework that can run it — a Hugging Face checkpoint, not a GGUF. A
GGUF forward pass is an inference engine, and this package does not
contain one. :func:`collect_from_pretrained` says so plainly rather than
returning something weight-derived and calling it an imatrix.
"""
from __future__ import annotations

import json
import logging
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypernix.security.hfpin import hub_revision

logger = logging.getLogger(__name__)

__all__ = [
    "ImatrixError",
    "ImatrixEntry",
    "Imatrix",
    "gguf_tensor_name",
    "collect",
    "collect_from_pretrained",
    "expand_for_tensor",
]


class ImatrixError(Exception):
    """An imatrix could not be measured, read or written."""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

#: Hugging Face submodule suffix -> GGUF tensor stem.
#:
#: An imatrix is keyed by GGUF tensor name because that is what the
#: quantiser has in its hand. Emitting torch's own module names would
#: produce a file that looks right, loads fine, and matches nothing.
_SUFFIX_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "attention.wq": "attn_q",
    "attention.wk": "attn_k",
    "attention.wv": "attn_v",
    "attention.wo": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "feed_forward.w1": "ffn_gate",
    "feed_forward.w3": "ffn_up",
    "feed_forward.w2": "ffn_down",
}

_LAYER = re.compile(r"(?:^|\.)(?:model\.)?(?:layers|h|blocks)\.(\d+)\.(.+)$")


def gguf_tensor_name(module_name: str) -> str:
    """The GGUF tensor a torch submodule's weight becomes, or ``""``.

    ``model.layers.7.self_attn.q_proj`` -> ``blk.7.attn_q.weight``.
    Returns empty for a module with no GGUF counterpart, because a
    guessed name is worse than a missing one: it silently weights the
    wrong tensor.
    """
    name = module_name.strip()
    if name in ("lm_head", "model.lm_head", "output"):
        return "output.weight"
    match = _LAYER.search(name)
    if not match:
        return ""
    index, suffix = match.group(1), match.group(2)
    stem = _SUFFIX_TO_GGUF.get(suffix)
    if stem is None:
        return ""
    return f"blk.{index}.{stem}.weight"


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


@dataclass
class ImatrixEntry:
    """One tensor's importance: the sum of ``x**2`` per input channel."""

    name: str
    sums: list[float]
    calls: int = 0

    @property
    def means(self) -> list[float]:
        """The per-channel mean, which is what a consumer wants."""
        if self.calls <= 0:
            return list(self.sums)
        return [value / self.calls for value in self.sums]

    def merge(self, other: ImatrixEntry) -> None:
        if len(other.sums) != len(self.sums):
            raise ImatrixError(
                f"{self.name}: cannot merge {len(other.sums)} channels into "
                f"{len(self.sums)} — these are different models."
            )
        self.sums = [a + b for a, b in zip(self.sums, other.sums, strict=True)]
        self.calls += other.calls


@dataclass
class Imatrix:
    """Every tensor's importance, plus what it was measured on."""

    entries: dict[str, ImatrixEntry] = field(default_factory=dict)
    dataset: str = ""
    tokens: int = 0
    #: Modules that were measured but map to no GGUF tensor. Kept so the
    #: report can say what was left out instead of quietly dropping it.
    unmapped: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def add(self, name: str, sums: Sequence[float], calls: int) -> None:
        entry = ImatrixEntry(name, [float(v) for v in sums], int(calls))
        if name in self.entries:
            self.entries[name].merge(entry)
        else:
            self.entries[name] = entry

    def to_simple_dict(self) -> dict[str, list[float]]:
        """``{tensor: per-channel means}`` — what hyprslug consumes."""
        return {name: entry.means for name, entry in sorted(self.entries.items())}

    def describe(self) -> str:
        lines = [
            f"imatrix: {len(self.entries)} tensor(s), {self.tokens} calibration token(s)"
        ]
        if self.dataset:
            lines.append(f"  dataset: {self.dataset}")
        if self.unmapped:
            lines.append(
                f"  {len(self.unmapped)} module(s) measured but not mapped to a GGUF "
                f"tensor: {', '.join(self.unmapped[:4])}"
                + (" ..." if len(self.unmapped) > 4 else "")
            )
        thin = [n for n, e in self.entries.items() if e.calls < 1]
        if thin:
            lines.append(f"  ! {len(thin)} tensor(s) saw no calibration data at all")
        return "\n".join(lines)

    # -- writing ---------------------------------------------------------

    def save_json(self, path: str | Path, *, simple: bool = False) -> Path:
        """Write JSON. ``simple=True`` writes just ``{tensor: [means]}``.

        The full form keeps the call counts and the dataset name, so a
        second run can be merged into it. The simple form is what
        hyprslug reads directly, and what a person can diff.
        """
        path = Path(path)
        if simple:
            payload: dict[str, Any] = self.to_simple_dict()
        else:
            payload = {
                "version": 1,
                "dataset": self.dataset,
                "tokens": self.tokens,
                "unmapped": self.unmapped,
                "entries": {
                    name: {"values": entry.means, "calls": entry.calls}
                    for name, entry in sorted(self.entries.items())
                },
            }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_binary(self, path: str | Path) -> Path:
        """Write llama.cpp's ``.imatrix``, so ``llama-quantize`` can read it.

        Values are stored as sums with the call count beside them —
        which is what llama.cpp writes and what its loader divides by. A
        file of means with ``ncall`` set would be divided a second time
        and every weight would be wrong by the same factor, which is
        exactly the kind of error that looks like a bad calibration set.
        """
        path = Path(path)
        with path.open("wb") as handle:
            handle.write(struct.pack("<i", len(self.entries)))
            for name, entry in sorted(self.entries.items()):
                raw = name.encode("utf-8")
                handle.write(struct.pack("<i", len(raw)))
                handle.write(raw)
                handle.write(struct.pack("<i", max(entry.calls, 0)))
                handle.write(struct.pack("<i", len(entry.sums)))
                if entry.sums:
                    handle.write(struct.pack(f"<{len(entry.sums)}f", *entry.sums))
            last_call = max((e.calls for e in self.entries.values()), default=0)
            handle.write(struct.pack("<i", last_call))
            dataset = self.dataset.encode("utf-8")
            handle.write(struct.pack("<i", len(dataset)))
            handle.write(dataset)
        return path

    # -- reading ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Imatrix:
        """Read either format, deciding by content rather than by suffix.

        People rename these files. A ``.imatrix`` that is really JSON,
        or a ``.json`` that is really llama.cpp's binary, should both
        work rather than fail with a parse error about the wrong format.
        """
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ImatrixError(f"Could not read {path}: {exc}") from exc
        if not raw:
            raise ImatrixError(f"{path} is empty.")
        stripped = raw.lstrip()
        if stripped[:1] in (b"{", b"["):
            return cls._from_json(raw, path)
        return cls._from_binary(raw, path)

    @classmethod
    def _from_json(cls, raw: bytes, path: Path) -> Imatrix:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ImatrixError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ImatrixError(f"{path} is not a mapping of tensor name to weights.")

        matrix = cls()
        if "entries" in data and isinstance(data["entries"], dict):
            matrix.dataset = str(data.get("dataset") or "")
            matrix.tokens = int(data.get("tokens") or 0)
            matrix.unmapped = [str(x) for x in (data.get("unmapped") or [])]
            for name, entry in data["entries"].items():
                values = entry.get("values") if isinstance(entry, dict) else entry
                calls = int(entry.get("calls", 1)) if isinstance(entry, dict) else 1
                matrix.entries[str(name)] = ImatrixEntry(
                    str(name), [float(v) * max(calls, 1) for v in values], max(calls, 1)
                )
            return matrix
        for name, values in data.items():
            if not isinstance(values, (list, tuple)):
                raise ImatrixError(
                    f"{path}: {name} maps to {type(values).__name__}, not a list."
                )
            matrix.entries[str(name)] = ImatrixEntry(
                str(name), [float(v) for v in values], 1
            )
        return matrix

    @classmethod
    def _from_binary(cls, raw: bytes, path: Path) -> Imatrix:
        matrix = cls()
        offset = 0

        def _int() -> int:
            nonlocal offset
            if offset + 4 > len(raw):
                raise ImatrixError(f"{path} ends mid-record; it is truncated.")
            value = struct.unpack_from("<i", raw, offset)[0]
            offset += 4
            return value

        count = _int()
        if count < 0 or count > 1_000_000:
            raise ImatrixError(
                f"{path} claims {count} entries, which is not a llama.cpp imatrix "
                f"(nor JSON). Pass a .imatrix or a .json."
            )
        for _ in range(count):
            length = _int()
            if length < 0 or offset + length > len(raw):
                raise ImatrixError(f"{path} has a name length that runs past the end.")
            name = raw[offset:offset + length].decode("utf-8", errors="replace")
            offset += length
            calls = _int()
            nval = _int()
            if nval < 0 or offset + nval * 4 > len(raw):
                raise ImatrixError(f"{path}: {name} claims {nval} values it does not have.")
            values = list(struct.unpack_from(f"<{nval}f", raw, offset)) if nval else []
            offset += nval * 4
            matrix.entries[name] = ImatrixEntry(name, values, max(calls, 1))

        if offset + 4 <= len(raw):
            offset += 4  # m_last_call, already implied by each entry's ncall
            if offset + 4 <= len(raw):
                length = struct.unpack_from("<i", raw, offset)[0]
                offset += 4
                if 0 <= length <= len(raw) - offset:
                    matrix.dataset = raw[offset:offset + length].decode(
                        "utf-8", errors="replace"
                    )
        return matrix


# ---------------------------------------------------------------------------
# Using one
# ---------------------------------------------------------------------------


def expand_for_tensor(values: Sequence[float], elements: int) -> list[float] | None:
    """Per-channel importance repeated to one weight per element.

    An imatrix has one number per *input channel*; a quantiser wants one
    per weight. A GGUF weight tensor is rows of ``n_input`` elements, so
    the vector tiles. Returns ``None`` when the two cannot be reconciled
    — a mismatched imatrix is a different model's, and applying it
    anyway would weight the wrong positions, which is worse than not
    applying it at all.
    """
    width = len(values)
    if width == 0 or elements <= 0:
        return None
    if width == elements:
        return [float(v) for v in values]
    if elements % width:
        return None
    return [float(v) for v in values] * (elements // width)


# ---------------------------------------------------------------------------
# Measuring one
# ---------------------------------------------------------------------------


def _linear_modules(model) -> list[tuple[str, Any]]:
    import torch

    found = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            found.append((name, module))
    return found


def collect(
    model,
    tokenizer,
    texts: Iterable[str],
    *,
    chunk_tokens: int = 512,
    max_chunks: int | None = None,
    device: str | None = None,
    dataset: str = "",
    progress=None,
) -> Imatrix:
    """Measure an imatrix by running *texts* through *model*.

    *model* is any torch module whose linear layers are ``nn.Linear``;
    *tokenizer* anything with ``encode``. Text is concatenated and cut
    into ``chunk_tokens``-token chunks, because a matmul's channel
    statistics are a property of the activations and short ragged inputs
    give the padding a vote.
    """
    import torch

    modules = _linear_modules(model)
    if not modules:
        raise ImatrixError(
            "This model has no torch.nn.Linear layers to hook. An imatrix is "
            "measured from activations; there is nothing here to measure."
        )

    matrix = Imatrix(dataset=dataset)
    sums: dict[str, torch.Tensor] = {}
    calls: dict[str, int] = {}
    unmapped: set[str] = set()

    def _hook(module_name: str):
        target = gguf_tensor_name(module_name)

        def _fn(_module, inputs):
            if not inputs:
                return
            activations = inputs[0]
            if not isinstance(activations, torch.Tensor):
                return
            flat = activations.detach().reshape(-1, activations.shape[-1]).float()
            squared = (flat * flat).sum(dim=0).cpu()
            key = target or module_name
            if not target:
                unmapped.add(module_name)
            if key in sums:
                sums[key] += squared
            else:
                sums[key] = squared
            calls[key] = calls.get(key, 0) + flat.shape[0]

        return _fn

    handles = [
        module.register_forward_pre_hook(_hook(name)) for name, module in modules
    ]

    seen_tokens = 0
    try:
        was_training = model.training
        model.eval()
        buffer: list[int] = []
        chunks_done = 0
        with torch.no_grad():
            for text in texts:
                buffer.extend(tokenizer.encode(text))
                while len(buffer) >= chunk_tokens:
                    if max_chunks is not None and chunks_done >= max_chunks:
                        buffer = []
                        break
                    chunk, buffer = buffer[:chunk_tokens], buffer[chunk_tokens:]
                    ids = torch.tensor([chunk], dtype=torch.long)
                    if device:
                        ids = ids.to(device)
                    model(ids)
                    seen_tokens += len(chunk)
                    chunks_done += 1
                    if progress is not None:
                        try:
                            progress({
                                "event": "chunk",
                                "chunks": chunks_done,
                                "tokens": seen_tokens,
                            })
                        except Exception:  # noqa: BLE001 - a listener must not fail a run
                            logger.debug("imatrix: progress callback raised", exc_info=True)
                if max_chunks is not None and chunks_done >= max_chunks:
                    break
        if was_training:
            model.train()
    finally:
        for handle in handles:
            handle.remove()

    if seen_tokens == 0:
        raise ImatrixError(
            f"No calibration text reached the model: the input tokenised to fewer "
            f"than one {chunk_tokens}-token chunk. Pass more text, or a smaller "
            f"--chunk-tokens."
        )

    for name, tensor in sums.items():
        matrix.add(name, tensor.tolist(), calls.get(name, 1))
    matrix.tokens = seen_tokens
    matrix.unmapped = sorted(unmapped)
    return matrix


def collect_from_pretrained(
    model_path: str | Path,
    texts: Iterable[str],
    **kwargs,
) -> Imatrix:
    """Load a Hugging Face checkpoint and measure an imatrix from it.

    A GGUF cannot be used here: measuring an imatrix means running the
    model, running it means a framework that can, and this package does
    not carry an inference engine. Quantise *from* the checkpoint you
    measured on and the names line up.
    """
    path = str(model_path)
    if path.endswith(".gguf"):
        raise ImatrixError(
            "An imatrix is measured from activations, which means running the "
            "model — and a GGUF needs an inference engine to run. Point this at "
            "the Hugging Face checkpoint the GGUF was converted from; the tensor "
            "names match, so the result applies to either."
        )
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the [train] extra
        raise ImatrixError(
            "Measuring an imatrix needs transformers: pip install 'hypernix[train]'"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, revision=hub_revision(path))
        model = AutoModelForCausalLM.from_pretrained(path, revision=hub_revision(path))
    except Exception as exc:  # noqa: BLE001 - transformers raises many things
        raise ImatrixError(f"Could not load {path}: {exc}") from exc
    kwargs.setdefault("dataset", "")
    return collect(model, tokenizer, texts, **kwargs)
