"""hypernix.v7m_features — V7M's optional training features, as explicit calls.

Pressure Cooker V5 and V5+ bundled training-time transformations with the
optimizer: constructing a V5+ gave you 6-bit quantisation-aware training
whether you asked or not, and ``prepare_model`` attached hooks that rewrote
weights. V7M separates the two. The optimizer (:class:`PressureCookerV7M`)
only optimises. Everything that changes the *model* or the *loss* is here,
is called by you, returns something that says what it did, and can be
undone:

==========================  ===================================================
:func:`attach_qat`          fake-quantise the weights of chosen layers on every
                            forward pass. Changes what the model computes.
:func:`quantization_sensitivity`  measure, layer by layer, how much the loss
                            rises when that layer alone is quantised. Changes
                            nothing: weights are restored.
:func:`mixed_precision_plan`  turn those measurements into bits per layer.
                            Pure arithmetic.
:func:`make_mtp_head`,      multi-token prediction: an extra head and an extra
:func:`mtp_loss`            loss term. Changes the objective.
:func:`ema_weights`         a context manager that evaluates with the
                            optimizer's EMA shadow and puts the live weights
                            back.
==========================  ===================================================

Costs, plainly. QAT adds a quantise pass over each attached layer's weights
per forward. MTP adds a head (parameters, memory, forward time) and makes
the loss a different loss. The EMA shadow is a full extra copy of the
weights. None of them is on unless you call it, and a V7M that calls none
runs exactly the V7 update and pays nothing for them.

QAT weights are written through a straight-through estimator: the forward
pass sees quantised weights, the stored weights stay full precision, so a
checkpoint is an ordinary checkpoint and removing the hooks gives the
unquantised model back.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from hypernix.optimizers.pressure_cooker_v5 import (
    MTPConfig,
    MTPHead,
    QATConfig,
    compute_quantization_params,
    fake_quantize_tensor,
)

__all__ = [
    "QATHandle",
    "attach_qat",
    "quantization_sensitivity",
    "mixed_precision_plan",
    "make_mtp_head",
    "mtp_loss",
    "ema_weights",
    "QATConfig",
    "MTPConfig",
    "QUANTIZABLE",
    "SENSITIVE",
]

#: Layers whose weights QAT may quantise.
QUANTIZABLE = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
#: Layers left alone: quantising them costs accuracy out of proportion.
SENSITIVE = (nn.Embedding, nn.LayerNorm)


def _quantized(weight: torch.Tensor, bits: int, config: QATConfig) -> torch.Tensor:
    levels = 2 ** int(bits)
    scale, zero = compute_quantization_params(weight.detach(), levels, config.symmetric,
                                              config.per_channel)
    if config.symmetric:
        # Symmetric: centre the grid on zero.
        zero = torch.full_like(scale, float(levels // 2))
    return fake_quantize_tensor(weight, scale, zero, levels, config.symmetric)


@dataclass
class QATHandle:
    """What :func:`attach_qat` did, and how to undo it."""

    layers: dict[str, int] = field(default_factory=dict)      # layer name -> bits
    skipped: list[str] = field(default_factory=list)
    _undo: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def remove(self) -> None:
        """Take the fake quantisation off again: the model computes with its
        stored full-precision weights, as before."""
        for undo in self._undo:
            undo()
        self._undo.clear()

    @property
    def active(self) -> bool:
        return bool(self._undo)


def attach_qat(model: nn.Module, config: QATConfig | None = None, *,
               bits: dict[str, int] | None = None, skip: tuple[str, ...] = ()) -> QATHandle:
    """Fake-quantise the weights of *model*'s linear and conv layers.

    Each attached layer computes with ``quantise(weight)`` on the forward
    pass and passes gradients straight through to the full-precision
    weight. *bits* gives a width per layer name (from
    :func:`mixed_precision_plan`); layers not named use ``config.bits``.
    Layer names in *skip* are left alone.

    This changes what the model computes. It is never called for you.
    """
    import torch.nn.utils.parametrize as parametrize

    settings = config or QATConfig()
    handle = QATHandle()
    per_layer = bits or {}

    class _Fake(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.width = width

        def forward(self, weight: torch.Tensor) -> torch.Tensor:
            return _quantized(weight, self.width, settings)

    def undo_for(module: nn.Module) -> Callable[[], None]:
        def undo() -> None:
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
        return undo

    for name, module in model.named_modules():
        if not isinstance(module, QUANTIZABLE) or getattr(module, "weight", None) is None:
            continue
        if name in skip:
            handle.skipped.append(name)
            continue
        width = int(per_layer.get(name, settings.bits))
        if width >= 16:                        # "leave this layer in full precision"
            handle.skipped.append(name)
            continue
        parametrize.register_parametrization(module, "weight", _Fake(width), unsafe=True)
        handle.layers[name] = width
        handle._undo.append(undo_for(module))
    return handle


@torch.no_grad()
def quantization_sensitivity(model: nn.Module, loss_fn: Callable[[], torch.Tensor | float], *,
                             bits: tuple[int, ...] = (4, 6, 8),
                             config: QATConfig | None = None) -> dict[str, dict[int, float]]:
    """How much the loss rises when one layer alone is quantised.

    *loss_fn* evaluates the model on a fixed batch and returns the loss
    (it is called once per layer per width, plus once for the baseline, so
    make it cheap). Returns ``{layer: {bits: loss - baseline}}``. Every
    weight is restored before this returns; nothing is left quantised.
    """
    config = config or QATConfig()
    was_training = model.training
    model.eval()
    try:
        baseline = float(loss_fn())
        out: dict[str, dict[int, float]] = {}
        for name, module in model.named_modules():
            if not isinstance(module, QUANTIZABLE) or getattr(module, "weight", None) is None:
                continue
            original = module.weight.detach().clone()
            out[name] = {}
            try:
                for width in bits:
                    module.weight.copy_(_quantized(original, width, config))
                    out[name][int(width)] = float(loss_fn()) - baseline
            finally:
                module.weight.copy_(original)
        return out
    finally:
        model.train(was_training)


def mixed_precision_plan(sensitivity: dict[str, dict[int, float]], *,
                         tolerance: float = 0.01,
                         full_precision_above: float | None = None) -> dict[str, int]:
    """Bits per layer: the narrowest measured width whose loss rise is
    within *tolerance*; the widest measured one when none is.

    With *full_precision_above*, a layer whose rise at its widest measured
    width still exceeds that is given 16, which :func:`attach_qat` reads as
    "do not quantise this layer".
    """
    plan: dict[str, int] = {}
    for name, by_bits in sensitivity.items():
        widths = sorted(by_bits)
        chosen = next((w for w in widths if by_bits[w] <= tolerance), widths[-1])
        if full_precision_above is not None and by_bits[widths[-1]] > full_precision_above:
            chosen = 16
        plan[name] = chosen
    return plan


def make_mtp_head(hidden_dim: int, vocab_size: int, config: MTPConfig | None = None) -> MTPHead:
    """A multi-token prediction head. Its parameters are yours to add to the
    optimizer; V7M does not add them for you."""
    config = config or MTPConfig()
    return MTPHead(hidden_dim=hidden_dim, vocab_size=vocab_size, num_tokens=config.num_tokens,
                   shared=config.shared_head)


def mtp_loss(head: MTPHead, hidden: torch.Tensor, targets: torch.Tensor,
             config: MTPConfig | None = None) -> torch.Tensor:
    """The extra loss term: token ``t + k`` predicted from position ``t``,
    for ``k = 1 .. num_tokens``, weighted by ``config.lambda_weight``.

    *hidden* is ``(batch, seq, hidden)``; *targets* is ``(batch, seq)`` with
    the token that follows each position (the usual shifted labels). Add
    the result to your own loss: this changes the training objective.
    """
    config = config or MTPConfig()
    logits = head(hidden, sequential=config.sequential)
    total = hidden.new_zeros(())
    used = 0
    for k, logit in enumerate(logits):
        if targets.shape[1] <= k:
            break
        # Position t predicts the target k places further on.
        pred = logit[:, : logit.shape[1] - k]
        gold = targets[:, k:]
        total = total + nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                                    gold.reshape(-1))
        used += 1
    return config.lambda_weight * total / max(1, used)


@contextlib.contextmanager
def ema_weights(optimizer: Any) -> Iterator[None]:
    """Evaluate with the optimizer's EMA shadow; put the live weights back after.

    Needs an optimizer built with ``ema_decay``. Raises if it has no shadow,
    rather than silently evaluating the live weights.
    """
    pairs = [(p, optimizer.state[p]["ema"]) for group in optimizer.param_groups
             for p in group["params"] if "ema" in optimizer.state.get(p, {})]
    if not pairs:
        raise RuntimeError("this optimizer keeps no EMA shadow; construct it with ema_decay=...")
    saved = [p.detach().clone() for p, _ in pairs]
    with torch.no_grad():
        for p, ema in pairs:
            p.copy_(ema)
    try:
        yield
    finally:
        with torch.no_grad():
            for (p, _), live in zip(pairs, saved, strict=True):
                p.copy_(live)
