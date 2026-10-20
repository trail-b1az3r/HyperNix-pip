"""hypernix.c4.explode — a model blown apart into the pieces c4 works on.

Five kinds of piece, from largest to smallest:

``mstem``
    A layer with its connections: every stem of one transformer block
    (attention, feed-forward, their norms). What c4 drops, keeps or
    transplants when it works on depth.
``stem``
    One connection and its parameters: one weight tensor, such as a
    block's ``ffn_up`` (hidden → intermediate) or its ``attn_norm``.
    Tensors outside every layer (embeddings, the head, the final norm)
    are stems with no mstem.
``Pdial``
    Three or more parameters measured, tested or trained together: a row
    of a stem (one output neuron's incoming weights), a column (one
    input's outgoing weights), an attention head's slice, or a block.
``mdial``
    Two parameters: a pair, for asking whether two weights matter more
    together than apart.
``dial``
    One parameter.

Nothing is copied to explode a model: pieces are addresses into the
loaded tensors, and the dials of a billion-parameter model are counted
and generated on demand rather than listed. An address is a string —
``"blk.3.ffn_up.weight[12, 40]"`` is a dial,
``"blk.3.ffn_up.weight[12, 40]+[12, 41]"`` an mdial,
``"blk.3.ffn_up.weight[12, :]"`` a row Pdial — which :func:`address`
reads back.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import C4Error, C4Model, layer_of, role_of

__all__ = [
    "Dial",
    "Exploded",
    "MDial",
    "MStem",
    "PDial",
    "Stem",
    "address",
    "explode",
]


@dataclass(frozen=True)
class Stem:
    """One connection: a tensor, its role and its place."""

    name: str
    shape: tuple[int, ...]
    layer: int
    role: str

    @property
    def numel(self) -> int:
        total = 1
        for d in self.shape:
            total *= d
        return total

    @property
    def kind(self) -> str:
        """``matrix`` for a connection between two widths, ``vector`` for a norm or bias."""
        return "matrix" if len(self.shape) >= 2 else "vector"

    @property
    def fan_out(self) -> int:
        return self.shape[0]

    @property
    def fan_in(self) -> int:
        return self.shape[1] if len(self.shape) > 1 else 1


@dataclass
class MStem:
    """A layer and its connections."""

    index: int
    stems: list[Stem] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"layer.{self.index}"

    @property
    def numel(self) -> int:
        return sum(s.numel for s in self.stems)

    def stem(self, role: str) -> Stem | None:
        return next((s for s in self.stems if s.role == role), None)


@dataclass(frozen=True)
class Dial:
    """One parameter."""

    stem: str
    index: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.stem}[{', '.join(map(str, self.index))}]"


@dataclass(frozen=True)
class MDial:
    """Two parameters."""

    stem: str
    first: tuple[int, ...]
    second: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise C4Error("an mdial is two different parameters")

    @property
    def dials(self) -> tuple[Dial, Dial]:
        return Dial(self.stem, self.first), Dial(self.stem, self.second)

    def __str__(self) -> str:
        return (f"{self.stem}[{', '.join(map(str, self.first))}]"
                f"+[{', '.join(map(str, self.second))}]")


@dataclass(frozen=True)
class PDial:
    """Three or more parameters: a slice of one stem.

    ``axis`` is ``row`` (an output neuron), ``col`` (an input), ``head``
    (an attention head's rows), ``block`` (a run of flat indices), or
    ``all``. ``start``/``stop`` bound it on that axis.
    """

    stem: str
    axis: str
    start: int
    stop: int

    def selector(self, shape: tuple[int, ...]) -> Any:
        """The index expression that picks this Pdial out of its tensor."""
        if self.axis in ("row", "head"):
            return (slice(self.start, self.stop),)
        if self.axis == "col":
            return (slice(None), slice(self.start, self.stop))
        if self.axis in ("block", "all"):
            return slice(self.start, self.stop)          # into the flattened view
        raise C4Error(f"unknown Pdial axis {self.axis!r}")

    def size(self, shape: tuple[int, ...]) -> int:
        width = self.stop - self.start
        if self.axis in ("row", "head"):
            return width * (shape[1] if len(shape) > 1 else 1)
        if self.axis == "col":
            return width * shape[0]
        return width

    def __str__(self) -> str:
        if self.axis in ("row", "head"):
            inner = f"{self.start}:{self.stop}, :" if self.stop - self.start > 1 else f"{self.start}, :"
            return f"{self.stem}[{inner}]"
        if self.axis == "col":
            inner = f"{self.start}:{self.stop}" if self.stop - self.start > 1 else f"{self.start}"
            return f"{self.stem}[:, {inner}]"
        return f"{self.stem}{{{self.start}:{self.stop}}}"


@dataclass
class Exploded:
    """A model's pieces, and ways to walk them."""

    model: C4Model
    mstems: list[MStem]
    stems: dict[str, Stem]

    # -- what there is --------------------------------------------------

    @property
    def loose_stems(self) -> list[Stem]:
        """Stems outside every layer: embeddings, the head, the final norm."""
        return [s for s in self.stems.values() if s.layer < 0]

    def counts(self) -> dict[str, int]:
        dials = sum(s.numel for s in self.stems.values())
        return {
            "mstems": len(self.mstems),
            "stems": len(self.stems),
            "row_pdials": sum(s.fan_out for s in self.stems.values() if s.kind == "matrix"),
            "mdials": dials // 2,
            "dials": dials,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.model.path), "format": self.model.format, "device": self.model.device,
            "layers": len(self.mstems), **self.counts(),
            "roles": sorted({s.role for s in self.stems.values()}),
        }

    def tensor(self, stem: str | Stem) -> Any:
        name = stem.name if isinstance(stem, Stem) else stem
        if name not in self.model.tensors:
            raise C4Error(f"no stem {name!r} in this model")
        return self.model.tensors[name]

    # -- walking the small pieces --------------------------------------

    def pdials(self, stem: str | Stem, *, by: str = "row", size: int = 1) -> Iterator[PDial]:
        """The Pdials of a stem: rows, columns, heads or flat blocks of *size*.

        A row or column of a vector stem is not three parameters, so a
        vector is walked as blocks.
        """
        info = self.stems[stem.name if isinstance(stem, Stem) else stem]
        if by == "head":
            heads = int(self.model.config.get("num_attention_heads") or 0)
            if not heads or info.fan_out % heads:
                raise C4Error(f"{info.name} does not divide into attention heads")
            size = info.fan_out // heads
            by = "row"
            axis = "head"
        else:
            axis = by
        if info.kind == "vector" and by in ("row", "col"):
            by = axis = "block"
            size = max(3, size)
        extent = {"row": info.fan_out, "col": info.fan_in}.get(by, info.numel)
        if by not in ("row", "col", "block"):
            raise C4Error(f"walk Pdials by row, col, head or block, not {by!r}")
        step = max(1, size)
        for start in range(0, extent, step):
            stop = min(extent, start + step)
            dial = PDial(info.name, axis, start, stop)
            if dial.size(info.shape) >= 3:
                yield dial

    def mdials(self, stem: str | Stem, *, along: str = "row", limit: int | None = None) -> Iterator[MDial]:
        """Neighbouring pairs in a stem: along a row (j, j+1) or a column (i, i+1)."""
        info = self.stems[stem.name if isinstance(stem, Stem) else stem]
        made = 0
        if info.kind == "vector":
            for i in range(0, info.numel - 1, 2):
                yield MDial(info.name, (i,), (i + 1,))
                made += 1
                if limit is not None and made >= limit:
                    return
            return
        rows, cols = info.fan_out, info.fan_in
        if along == "row":
            for i in range(rows):
                for j in range(0, cols - 1, 2):
                    yield MDial(info.name, (i, j), (i, j + 1))
                    made += 1
                    if limit is not None and made >= limit:
                        return
        else:
            for j in range(cols):
                for i in range(0, rows - 1, 2):
                    yield MDial(info.name, (i, j), (i + 1, j))
                    made += 1
                    if limit is not None and made >= limit:
                        return

    def dials(self, stem: str | Stem, *, limit: int | None = None) -> Iterator[Dial]:
        info = self.stems[stem.name if isinstance(stem, Stem) else stem]
        import itertools

        ranges = [range(d) for d in info.shape] or [range(1)]
        for made, index in enumerate(itertools.product(*ranges)):
            if limit is not None and made >= limit:
                return
            yield Dial(info.name, tuple(index))

    # -- reading values --------------------------------------------------

    def value(self, piece: Dial | MDial | PDial | Stem | str) -> Any:
        """The current values of a piece (a view where PyTorch can give one)."""
        if isinstance(piece, str):
            piece = address(piece, self)
        if isinstance(piece, Stem):
            return self.tensor(piece)
        tensor = self.tensor(piece.stem)
        if isinstance(piece, Dial):
            return tensor[piece.index]
        if isinstance(piece, MDial):
            import torch

            return torch.stack([tensor[piece.first], tensor[piece.second]])
        selector = piece.selector(tuple(tensor.shape))
        if piece.axis in ("block", "all"):
            return tensor.reshape(-1)[selector]
        return tensor[selector]

    # -- saving ---------------------------------------------------------

    def save(self, folder: str | Path, *, weights: bool = True) -> Path:
        """Write the exploded model to *folder*: a manifest, and with
        *weights* one safetensors file per mstem (and one for the loose
        stems), so a piece can be loaded without the rest."""
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "c4": 1, **self.summary(), "config": self.model.config,
            "mstems": [{"index": m.index, "stems": [s.name for s in m.stems], "numel": m.numel}
                       for m in self.mstems],
            "stems": {s.name: {"shape": list(s.shape), "layer": s.layer, "role": s.role}
                      for s in self.stems.values()},
        }
        if weights:
            from safetensors.torch import save_file

            def dump(names: list[str], file: Path) -> None:
                save_file({n: self.model.tensors[n].detach().contiguous().cpu() for n in names}, str(file))

            for mstem in self.mstems:
                dump([s.name for s in mstem.stems], out / f"mstem-{mstem.index:04d}.safetensors")
            loose = [s.name for s in self.loose_stems]
            if loose:
                dump(loose, out / "loose.safetensors")
        (out / "c4-explode.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        return out


def explode(model: C4Model | str | Path, **load_kwargs: Any) -> Exploded:
    """Blow *model* (a :class:`C4Model`, or a path to load) into its pieces."""
    if not isinstance(model, C4Model):
        from .model import load

        model = load(model, **load_kwargs)
    stems: dict[str, Stem] = {}
    by_layer: dict[int, MStem] = {}
    for name, tensor in model.tensors.items():
        if not getattr(tensor, "is_floating_point", lambda: False)():
            continue
        stem = Stem(name, tuple(int(d) for d in tensor.shape), layer_of(name), role_of(name))
        stems[name] = stem
        if stem.layer >= 0:
            by_layer.setdefault(stem.layer, MStem(stem.layer)).stems.append(stem)
    mstems = [by_layer[i] for i in sorted(by_layer)]
    return Exploded(model=model, mstems=mstems, stems=stems)


_ADDRESS = re.compile(r"^(?P<stem>[^\[\]{}]+)(?P<rest>.*)$")


def _index(text: str) -> list[str]:
    return [p.strip() for p in text.split(",")]


def address(text: str, exploded: Exploded | None = None) -> Dial | MDial | PDial | Stem:
    """Read an address string back into a piece (see the module docstring)."""
    match = _ADDRESS.match(text.strip())
    if not match:
        raise C4Error(f"not a c4 address: {text!r}")
    stem, rest = match.group("stem").strip(), match.group("rest").strip()
    if exploded is not None and stem not in exploded.stems:
        raise C4Error(f"no stem {stem!r} in this model")
    if not rest:
        if exploded is None:
            raise C4Error("a bare stem address needs the exploded model to resolve")
        return exploded.stems[stem]
    if rest.startswith("{"):
        start, stop = rest.strip("{}").split(":")
        return PDial(stem, "block", int(start), int(stop))
    parts = re.findall(r"\[([^\]]*)\]", rest)
    if len(parts) == 2:
        return MDial(stem, tuple(int(v) for v in _index(parts[0])), tuple(int(v) for v in _index(parts[1])))
    if len(parts) != 1:
        raise C4Error(f"not a c4 address: {text!r}")
    fields = _index(parts[0])
    if all(f.lstrip("-").isdigit() for f in fields):
        return Dial(stem, tuple(int(v) for v in fields))
    if len(fields) == 2 and fields[1] == ":":
        start, stop = (fields[0].split(":") + [None])[:2] if ":" in fields[0] else (fields[0], None)
        start_i = int(start)
        return PDial(stem, "row", start_i, int(stop) if stop else start_i + 1)
    if len(fields) == 2 and fields[0] == ":":
        start, stop = (fields[1].split(":") + [None])[:2] if ":" in fields[1] else (fields[1], None)
        start_i = int(start)
        return PDial(stem, "col", start_i, int(stop) if stop else start_i + 1)
    raise C4Error(f"not a c4 address: {text!r}")
