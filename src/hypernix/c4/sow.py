"""hypernix.c4.sow — a new model, grown from what a reap gathered.

:func:`sow` takes a :class:`~hypernix.c4.reap.Harvest` and its donor
models and writes a model made of the cherry-picked pieces:

1. the embeddings, head and final norm from the harvest's ``loose_from``
   donor;
2. each kept layer, from whichever donor the harvest names, renumbered
   to its new depth;
3. the unneeded dials zeroed;
4. feed-forward channels pruned where the harvest says (rows of
   ``ffn_gate``/``ffn_up``, columns of ``ffn_down``), with the config's
   ``intermediate_size`` changed to match;
5. ``config.json`` rewritten (``num_hidden_layers``, and any per-layer
   list such as ``layer_types`` re-indexed), tokenizer files copied from
   the donor.

It writes safetensors (or ``.pt``, or a GGUF for a GGUF donor), loads
the result back to check it is whole, and, given a prober, reports the
calibration loss before and after. ``heal`` then trains the new model
briefly with the c4 optimizer to recover what the cut cost.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .explode import Exploded, explode
from .model import C4Error, layer_of, role_of
from .reap import Harvest
from .restore import write

__all__ = ["SowReport", "heal", "renumber", "sow"]

_LAYER_IN_NAME = re.compile(r"((?:^|\.)(?:layers|blk|h|block|blocks)\.)(\d+)(\.)")


def renumber(name: str, new_index: int) -> str:
    """*name* with its layer index replaced by *new_index*."""
    return _LAYER_IN_NAME.sub(lambda m: f"{m.group(1)}{new_index}{m.group(3)}", name, count=1)


@dataclass
class SowReport:
    path: str
    layers: int
    parameters_before: int
    parameters_after: int
    zeroed_dials: int
    ffn_channels: int | None
    loss_before: float | None = None
    loss_after: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _donors(harvest: Harvest, explodeds: Sequence[Exploded] | None, device: str | None) -> list[Exploded]:
    if explodeds is not None:
        if len(explodeds) != len(harvest.donors):
            raise C4Error(f"the harvest has {len(harvest.donors)} donors; {len(explodeds)} were given")
        return list(explodeds)
    return [explode(path, device=device or "auto") for path in harvest.donors]


def _reindex_config(config: dict[str, Any], old_layers: int, picks: list[int]) -> dict[str, Any]:
    out = dict(config)
    for key, value in config.items():
        if isinstance(value, list) and len(value) == old_layers and old_layers > 1:
            out[key] = [value[i] for i in picks]
    for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if key in out:
            out[key] = len(picks)
    return out


def sow(harvest: Harvest, out: str | Path, *, explodeds: Sequence[Exploded] | None = None,
        format: str | None = None, device: str | None = "cpu", prober: Any = None,
        check: bool = True) -> SowReport:
    """Write the model *harvest* describes to *out*."""
    import torch

    donors = _donors(harvest, explodeds, device)
    base = donors[harvest.loose_from]
    fmt = format or ("gguf" if base.model.format == "gguf" else "safetensors")
    state: dict[str, Any] = {}
    loose_zeroed = 0

    for name, tensor in base.model.tensors.items():
        if layer_of(name) < 0:
            state[name] = tensor.detach().clone()
            mask = harvest.masks.get((harvest.loose_from, name))
            if mask is not None:
                state[name][mask.to(tensor.device)] = 0.0
                loose_zeroed += int(mask.sum())

    zeroed = loose_zeroed
    for position, (donor_index, layer) in enumerate(harvest.layers):
        donor = donors[donor_index]
        if layer >= len(donor.mstems):
            raise C4Error(f"donor {donor_index} has no layer {layer}")
        keep = harvest.channels.get(position)
        for stem in donor.mstems[layer].stems:
            value = donor.tensor(stem).detach().clone()
            mask = harvest.masks.get((donor_index, stem.name))
            if mask is not None:
                value[mask.to(value.device)] = 0.0
                zeroed += int(mask.sum())
            if keep is not None:
                index = torch.tensor(keep, dtype=torch.long, device=value.device)
                if stem.role in ("ffn_gate", "ffn_up"):
                    value = value.index_select(0, index)
                elif stem.role == "ffn_down":
                    value = value.index_select(1, index)
            state[renumber(stem.name, position)] = value

    config = _reindex_config(base.model.config, len(base.mstems), [layer for _, layer in harvest.layers])
    channels = len(next(iter(harvest.channels.values()))) if harvest.channels else None
    if channels is not None:
        if len({len(c) for c in harvest.channels.values()}) != 1 or len(harvest.channels) != len(harvest.layers):
            raise C4Error("every kept layer must keep the same number of feed-forward channels")
        for key in ("intermediate_size", "n_inner", "ffn_dim"):
            if key in config:
                config[key] = channels
    report = SowReport(
        path=str(out), layers=len(harvest.layers),
        parameters_before=base.model.parameters,
        parameters_after=int(sum(t.numel() for t in state.values())),
        zeroed_dials=zeroed, ffn_channels=channels,
    )
    if prober is not None:
        report.loss_before = prober.loss()
    if fmt == "gguf":
        model = base.model
        metadata = dict(model.metadata)
        arch = str(metadata.get("general.architecture") or "llama")
        metadata[f"{arch}.block_count"] = len(harvest.layers)
        if channels is not None:
            metadata[f"{arch}.feed_forward_length"] = channels
        model_copy = type(model)(path=model.path, format="gguf", config=config, tensors=state,
                                 device=model.device, metadata=metadata)
        written = write(model_copy, out, format="gguf", config=config)
    else:
        written = write(base.model, out, format=fmt, tensors=state, config=config)
    report.path = str(written)
    if check:
        from .model import load

        again = load(written, device=device or "cpu")
        if len({layer_of(n) for n in again.tensors} - {-1}) != len(harvest.layers):
            raise C4Error(f"{written} did not come back with {len(harvest.layers)} layers")
        report.notes.append("loaded back whole")
        if prober is not None:
            from .probe import Prober

            try:
                new_prober = Prober(again, prober.calibration)
                report.loss_after = new_prober.loss()
            except C4Error as exc:
                report.notes.append(f"could not run it to compare: {exc}")
    roles = sorted({role_of(n) for n in state})
    report.notes.append(f"roles: {', '.join(roles)}")
    return report


def heal(path: str | Path, calibration: Any, *, steps: int = 50, optimizer: str = "hybrid",
         lr: float = 1e-4, device: str | None = "auto", out: str | Path | None = None,
         sections: str = "stem") -> dict[str, Any]:
    """Train a sown model for a few steps with the c4 optimizer, then save it.

    Cutting layers or channels costs some loss; a short heal recovers
    much of it. Every step goes through :class:`~hypernix.c4.optim.C4Optimizer`,
    so a section whose gradient goes bad is skipped rather than stepped.
    """
    from .model import load
    from .optim import C4Optimizer
    from .probe import Prober

    model = load(path, device=device)
    prober = Prober(model, calibration)
    exploded = explode(model)
    module = prober.module
    module.train()
    opt = C4Optimizer.for_exploded(exploded, module, kind=optimizer, lr=lr, granularity=sections)
    before = prober.loss()
    batches = calibration.batches
    for step in range(steps):
        batch = batches[step % len(batches)].to(model.device)
        opt.zero_grad(set_to_none=True)
        loss = module(input_ids=batch, labels=batch).loss
        loss.backward()
        opt.step()
    module.eval()
    after = prober.loss()
    written = write(model, out or path, tensors=dict(module.state_dict()))
    return {"path": str(written), "loss_before": before, "loss_after": after, "steps": steps,
            "skipped_sections": opt.report()["skipped"]}
