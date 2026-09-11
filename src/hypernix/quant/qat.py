"""hypernix.quant.qat — training a model that survives its own quantiser.

Post-training quantisation to a sub-bit tier measures out as you would
expect: IQ0.5_XXXL keeps 75% of signs and correlates +0.40 with the
weights it came from. Sixty percent of the information is gone and the
model was never told it was going to happen.

Quantisation-aware training tells it. The forward pass uses the
*quantised* weights, so every loss the model sees is the loss it will
have at inference; the backward pass updates the underlying float
weights through a straight-through estimator. The model is free to move
weights across the decision boundaries the packer will apply -- which is
the entire mechanism, and the reason QAT recovers accuracy that no
post-training method can.

What has to be true for this to work
------------------------------------
**The fake quantiser must be the real one.** If training simulates a
slightly different packing from the one ``hyprslug`` writes, the model
is optimised for a quantiser that never ships, and the result is
*worse* than not training at all -- it has spent its capacity adapting
to the wrong boundaries. So :func:`fake_quantize` is checked against
:mod:`hypernix.quant.subbit` element for element, and
``tests/test_qat.py`` fails if they ever part company.

That is not a hypothetical worry in this package. The row-length bug
shipped because the writer and the reader shared a misconception and
agreed with each other; a QAT that drifts from its packer is the same
mistake with a training run attached to it.

Why the arithmetic is written twice
-----------------------------------
``subbit.py`` packs to *bytes*, one block at a time, in Python. A
training step needs the same numbers as a torch tensor, on whatever
device the model is on, differentiably, thousands of times per epoch.
Going through the byte packer would be several orders of magnitude too
slow and would break the graph. So the torch path here re-derives the
same values without ever producing a byte -- and then a test proves the
two agree.

Usage
-----
::

    from hypernix.quant.qat import prepare_qat

    model, report = prepare_qat(model, tier="HNX_1375BIT")
    ...                       # train as usual
    model = finalize_qat(model)   # fold the fake quant away

The wrapped modules quantise their weight on every forward pass, so a
checkpoint saved mid-training holds float weights that are *known to
quantise well* -- which is what ``hyprslug`` should then be pointed at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .subbit import BLOCK_SIZE, PACKINGS, SubBitError

__all__ = [
    "QATError",
    "QATConfig",
    "QATReport",
    "fake_quantize",
    "block_scales",
    "clamp_weight_",
    "QATLinear",
    "prepare_qat",
    "finalize_qat",
    "quantization_error",
]


class QATError(RuntimeError):
    """QAT could not be set up, or was asked for something impossible."""


@dataclass
class QATConfig:
    """What to simulate, and where.

    ``tier`` names a HyperNix tier; the packing behind it is what gets
    simulated. ``skip`` is matched against module names -- the embedding
    and the output head are excluded by default for the same reason
    ``hyprslug`` leaves them in float, and because putting a 151,936-row
    table through a fake quantiser every step is most of a training
    budget.
    """

    tier: str = "HNX_1375BIT"
    skip: tuple[str, ...] = ("embed", "token_embd", "lm_head", "output")
    #: Minimum in-features for a layer to be worth quantising. A tensor
    #: whose rows are shorter than one block cannot be packed at all --
    #: the same ne[0] rule the file format applies.
    min_features: int = BLOCK_SIZE
    #: Hold every weight within this many block scales, applied before
    #: each forward pass.
    #:
    #: **Not optional in practice.** A straight-through estimator puts no
    #: pressure on a weight's magnitude -- only its sign reaches the
    #: output -- so weights drift outward, the block scale is their mean
    #: absolute value and drifts with them, and reconstruction gets
    #: steadily worse. Measured on a 256-wide distillation task, 600
    #: steps:
    #:
    #: =============  ==============  ==========  ==============
    #: tier           train in float  QAT, no clamp  QAT + clamp
    #: =============  ==============  ==========  ==============
    #: HNX_1375BIT            0.0352      0.0533          0.0231
    #: INT1                   0.0372      0.0537          0.0195
    #: =============  ==============  ==========  ==============
    #:
    #: Without it QAT was *worse than not doing QAT* in every run. With
    #: it, 1.5x to 1.9x better. Set to 0 to disable and get the textbook
    #: straight-through behaviour.
    clamp: float = 1.5

    def packing(self) -> str:
        from .hyprslug import TIER_TYPES

        entry = TIER_TYPES.get(self.tier)
        if entry is None:
            raise QATError(
                f"Unknown tier {self.tier!r}. QAT simulates the HyperNix "
                f"tiers: {', '.join(TIER_TYPES)}"
            )
        packing = entry[1]
        if packing not in PACKINGS:
            raise QATError(
                f"Tier {self.tier} uses packing {packing!r}, which is a fixed "
                f"codebook rather than a sign-and-scale packing. QAT covers "
                f"the packings in hypernix.quant.subbit."
            )
        return packing


@dataclass
class QATReport:
    """Which layers were wrapped, and which were left alone."""

    tier: str = ""
    packing: str = ""
    wrapped: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def parameters_quantised(self) -> int:
        return self._counted

    _counted: int = 0

    def describe(self) -> str:
        lines = [
            f"QAT: {self.tier} ({self.packing})",
            f"  {len(self.wrapped)} layers simulated, "
            f"{self._counted / 1e6:.1f}M parameters",
        ]
        for name, reason in self.skipped[:6]:
            lines.append(f"    left alone: {name} -- {reason}")
        if len(self.skipped) > 6:
            lines.append(f"    ... and {len(self.skipped) - 6} more")
        if not self.wrapped:
            lines.append(
                "  ! nothing was wrapped, so this run is ordinary training"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The differentiable quantiser
# ---------------------------------------------------------------------------

def _as_stored(scale):
    """The scale as the file will hold it: FP16.

    Skipping this was a real difference, not a rounding nicety. The
    packer writes ``struct.pack('<e', scale)`` and the decoder reads it
    back, so every weight in a block is a multiple of an FP16 number.
    Training against the float32 scale optimises for values the format
    cannot represent -- a ~3e-4 relative offset on every weight, in the
    one place where matching the deployed quantiser is the whole point.

    ``.half().float()`` is round-to-nearest-even, which is what
    ``struct.pack`` does too.
    """
    return scale.half().float()


def _sub_magnitudes(absolute, spec, importance=None):
    """Per-sub-block magnitude, as the packer computes it.

    Mirrors ``subbit._quantize_with_magnitude``: the reference is the
    largest sub-block mean, not the block mean, so the indices span the
    codebook and nothing clamps.
    """
    import torch

    blocks, _ = absolute.shape
    sub = absolute.view(blocks, spec.sub_blocks, spec.sub_size)
    if importance is None:
        means = sub.mean(dim=2)
    else:
        weights = importance.view(blocks, spec.sub_blocks, spec.sub_size).clamp_min(0)
        total = weights.sum(dim=2)
        means = torch.where(
            total > 0, (sub * weights).sum(dim=2) / total.clamp_min(1e-30),
            torch.zeros_like(total),
        )
    scale = means.max(dim=1, keepdim=True).values
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    safe = scale.clamp_min(1e-30)
    # round-half-away-from-zero, matching Python's round() on the
    # positive values this only ever sees -- torch.round is
    # round-half-to-even and differs on exact .5, which lands often
    # because these are means of small integers' worth of weights.
    exact = means / safe * spec.levels
    index = torch.floor(exact + 0.5).clamp_(0, spec.levels)
    # The index is chosen against the full-precision scale, exactly as
    # the packer does, and then combined with the scale *as stored* --
    # FP16, see _as_stored.
    #
    # `(scale * index) / levels`, in that order, because that is what the
    # decoder computes. `(index / levels) * scale` is the same number in
    # real arithmetic and a different one in float32, and the test that
    # compares against the packer's own bytes is not fooled by it.
    return (_as_stored(scale) * index) / spec.levels, scale


def fake_quantize(weight, packing: str, importance=None):
    """Quantise and immediately dequantise, as a differentiable tensor.

    The value returned is what the packed weight decodes to -- the same
    number ``dequantize_array`` produces for the bytes
    ``quantize_tensor`` would write. No bytes are produced.

    Gradients pass straight through to *weight* (the straight-through
    estimator). Rounding and sign extraction have zero derivative almost
    everywhere, so without this there is nothing to train on; with it the
    float weight moves and the quantised value follows in steps.
    """
    import torch

    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")

    flat = weight.reshape(-1)
    if flat.numel() % BLOCK_SIZE:
        raise QATError(
            f"{flat.numel()} weights is not a whole number of "
            f"{BLOCK_SIZE}-weight blocks; a layer this shape cannot be packed."
        )
    blocks = flat.view(-1, BLOCK_SIZE).float()
    absolute = blocks.abs()
    weights = None if importance is None else importance.reshape(-1).view(-1, BLOCK_SIZE)

    # The packer's sign convention: >= 0 is positive, so a stored zero
    # decodes to +scale rather than 0. Reproducing that exactly matters
    # more than it looks -- a QAT that treats zero as negative trains the
    # model to place weights on the wrong side of a boundary the packer
    # resolves the other way.
    signs = torch.where(blocks >= 0, 1.0, -1.0)

    if spec.has_sub_magnitude:
        magnitude, _scale = _sub_magnitudes(absolute, spec, weights)
        out = signs * magnitude.repeat_interleave(spec.sub_size, dim=1)
    else:
        if weights is None:
            scale = absolute.mean(dim=1, keepdim=True)
        else:
            positive = weights.clamp_min(0)
            total = positive.sum(dim=1, keepdim=True)
            scale = torch.where(
                total > 0, (absolute * positive).sum(dim=1, keepdim=True)
                / total.clamp_min(1e-30), torch.zeros_like(total),
            )
        scale = _as_stored(torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0))
        if spec.kept == spec.group:
            out = signs * scale
        else:
            # Positions whose sign was dropped repeat the last stored one
            # in their group -- the decoder's rule, reproduced here.
            grouped = signs.view(-1, spec.codes_per_block, spec.group)
            last = grouped[:, :, spec.kept - 1:spec.kept]
            kept = grouped[:, :, : spec.kept]
            filled = last.expand(-1, -1, spec.group - spec.kept)
            out = torch.cat([kept, filled], dim=2).reshape(-1, BLOCK_SIZE) * scale

    quantised = out.reshape(weight.shape).to(weight.dtype)

    # Straight-through: forward is the quantised value, backward is the
    # identity.
    #
    # The usual spelling is `w + (q - w).detach()`, and it is wrong here
    # in two ways. In floating point `w + (q - w)` is not exactly `q` --
    # each step rounds -- so the forward value drifted an ulp off the
    # packer, and a test comparing against the real bytes caught it. And
    # for a non-finite weight it is `inf + (-inf)`, which is NaN, where
    # the packer degrades the whole block to zeros (see subbit._finite).
    #
    # `q.detach() + (w - w.detach())` adds an exact zero instead, so the
    # forward value is `q` bit for bit; nan_to_num on that zero keeps a
    # poisoned weight from spreading, and costs it its gradient, which
    # is the right answer for a weight that is already inf.
    passthrough = torch.nan_to_num(
        weight - weight.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    return quantised.detach() + passthrough


def quantization_error(weight, packing: str) -> dict[str, float]:
    """How far this weight is from its quantised self, right now.

    Useful during training: the whole point of QAT is that this number
    comes down as the model learns to sit where the packer can represent
    it, and a run where it does not is a run that is not working.
    """
    import torch

    with torch.no_grad():
        flat = weight.reshape(-1).float()
        back = fake_quantize(weight.detach(), packing).reshape(-1).float()
        delta = flat - back
        denominator = flat.norm().clamp_min(1e-30)
        signs_kept = ((flat >= 0) == (back >= 0)).float().mean()
        return {
            "rmse": float(delta.pow(2).mean().sqrt()),
            "relative": float(delta.norm() / denominator),
            "signs_kept": float(signs_kept),
        }


# ---------------------------------------------------------------------------
# Wrapping a model
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:                       # pragma: no cover
        raise QATError(
            "QAT needs PyTorch. Install the training extra: pip install "
            "hypernix[training]"
        ) from exc
    return torch, nn


def _make_qat_linear():
    """Built lazily so importing this module does not import torch.

    ``hypernix.quant`` is imported by the quantiser, which runs on
    machines that have no torch and no reason to. A class defined at
    module scope would drag it in for all of them.
    """
    torch, nn = _torch()

    class QATLinear(nn.Module):
        """``nn.Linear`` whose weight is quantised on every forward pass.

        The float weight stays the parameter -- that is what the
        optimiser updates and what a checkpoint holds. What the layer
        *computes with* is the quantised value, so the loss the model
        minimises is the loss it will have after ``hyprslug`` writes the
        file.
        """

        def __init__(self, linear, packing: str, clamp: float = 1.5):
            super().__init__()
            self.packing = packing
            self.clamp = float(clamp)
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            self.weight = linear.weight
            self.bias = linear.bias
            #: Set False to train a few steps in plain float -- useful as
            #: a warm-up, since a randomly initialised model quantises to
            #: noise and the first steps can be wasted fighting it.
            self.quantize: bool = True

        def forward(self, x):
            if not self.quantize:
                return nn.functional.linear(x, self.weight, self.bias)
            if self.clamp > 0:
                clamp_weight_(self.weight, self.packing, self.clamp)
            return nn.functional.linear(
                x, fake_quantize(self.weight, self.packing), self.bias
            )

        def to_linear(self):
            """A plain ``nn.Linear`` holding the *float* weight.

            Not the quantised one: the float weight is the thing worth
            keeping, because it is what ``hyprslug`` should quantise and
            it carries more than the packed form can. Folding the fake
            quant in here would throw away the training.
            """
            out = nn.Linear(self.in_features, self.out_features,
                            bias=self.bias is not None)
            with torch.no_grad():
                out.weight.copy_(self.weight)
                if self.bias is not None:
                    out.bias.copy_(self.bias)
            return out

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, packing={self.packing}"
            )

    return QATLinear


_QAT_LINEAR: Any = None


def QATLinear(linear, packing: str, clamp: float = 1.5):   # noqa: N802 - a class
    """Wrap one ``nn.Linear`` for quantisation-aware training."""
    global _QAT_LINEAR
    if _QAT_LINEAR is None:
        _QAT_LINEAR = _make_qat_linear()
    return _QAT_LINEAR(linear, packing, clamp)


def block_scales(weight, packing: str):
    """Each block's scale, as the packer computes it.

    Exposed because :func:`clamp_weight_` needs it and because a training
    loop watching these is watching the thing that goes wrong: they
    should sit still, and a run where they climb is a run drifting away
    from the quantiser.
    """
    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")
    blocks = weight.detach().reshape(-1, BLOCK_SIZE).float().abs()
    if spec.has_sub_magnitude:
        means = blocks.view(-1, spec.sub_blocks, spec.sub_size).mean(dim=2)
        return means.max(dim=1, keepdim=True).values
    return blocks.mean(dim=1, keepdim=True)


def clamp_weight_(weight, packing: str, limit: float = 1.5) -> None:
    """Hold *weight* within ``limit`` block scales, in place.

    The counterweight to the straight-through estimator: STE reports a
    gradient for a weight whose sign the output already has, so nothing
    stops it moving further out, and the block scale follows it. See
    :attr:`QATConfig.clamp` for the measurements.
    """
    import torch

    if limit <= 0:
        return
    with torch.no_grad():
        scales = block_scales(weight, packing)
        bound = (limit * scales).expand(-1, BLOCK_SIZE).reshape(weight.shape)
        weight.clamp_(-bound, bound)


def _eligible(name: str, module, config: QATConfig) -> tuple[bool, str]:
    lowered = name.lower()
    for fragment in config.skip:
        if fragment in lowered:
            return False, f"name matches {fragment!r}"
    if module.in_features < config.min_features:
        return False, (
            f"{module.in_features} inputs per row is under one "
            f"{BLOCK_SIZE}-weight block"
        )
    if module.in_features % BLOCK_SIZE:
        # The file format's rule, applied here so training cannot
        # simulate a layer the quantiser will decline to pack.
        return False, (
            f"{module.in_features} elements per row do not divide into "
            f"{BLOCK_SIZE}-element blocks"
        )
    return True, ""


def prepare_qat(model, tier: str = "HNX_1375BIT", *, config: QATConfig | None = None):
    """Swap eligible ``nn.Linear`` layers for quantisation-aware ones.

    Returns ``(model, report)``. The model is modified in place and also
    returned, so either style reads correctly.

    Layers are skipped for the same reasons ``hyprslug`` skips tensors --
    embeddings, the output head, and any row shorter than a block or not
    a multiple of one. Training a layer the quantiser will leave in float
    would teach the model to compensate for damage that never happens.
    """
    _, nn = _torch()
    settings = config or QATConfig(tier=tier)
    if config is None and tier:
        settings.tier = tier
    packing = settings.packing()
    report = QATReport(tier=settings.tier, packing=packing)

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            ok, reason = _eligible(full, child, settings)
            if not ok:
                report.skipped.append((full, reason))
                continue
            setattr(parent, child_name, QATLinear(child, packing, settings.clamp))
            report.wrapped.append(full)
            report._counted += child.weight.numel()
    return model, report


def finalize_qat(model):
    """Unwrap every QAT layer, leaving plain ``nn.Linear`` with the float
    weights the training produced.

    Quantise the result with ``hyprslug`` as usual -- that is the point:
    the weights now sit where the packer can represent them, so the same
    tier costs less accuracy than it would have on a model that never
    saw it coming.
    """
    _, nn = _torch()
    if _QAT_LINEAR is None:
        return model
    for _parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, _QAT_LINEAR):
                setattr(parent, child_name, child.to_linear())
    return model
