"""deep_fryer — randomise a percentage of model weights.

A deep fryer drops parameters into hot oil: some come out crispy,
some come out ruined.  Here, it perturbs a fraction of a model's
weights with Gaussian noise — useful for **regularisation** (light
frying, small σ, small fraction) and **robustness testing** (heavy
frying, bigger σ, larger fraction).

Two tiers:

* :class:`LightFry`  — t1.  Perturbs ~1-5% of parameters with small
                         Gaussian noise.  Use during training as a
                         regulariser, or between epochs to knock the
                         model off a local minimum.
* :class:`HeavyFry`  — t2.  Perturbs 20-50% of parameters with
                         larger noise, plus random zeroing.  Use for
                         robustness testing, for generating "bad
                         model" negatives to train a judge against,
                         or as input to a :class:`hypernix.salt_shaker`
                         chain.

Both work in place on a ``torch.nn.Module``.  Call :meth:`fry` to
perturb, :meth:`save_pristine` to snapshot the untouched weights
first, and :meth:`un_fry` to restore them.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn as nn


@dataclass
class Fryer:
    """Base deep-fryer.  Subclasses set noise intensity + sparsity."""

    model: nn.Module
    #: Fraction of tensor elements (per parameter) touched by the fryer.
    fraction: float = 0.02
    #: Gaussian std-dev of the additive noise, relative to the
    #: parameter's own std.
    noise_std: float = 0.1
    #: Optional name regex / substring patterns; when non-empty, only
    #: parameters whose name matches at least one entry are fried.
    patterns: tuple[str, ...] = ()
    seed: int = 0
    name: ClassVar[str] = "Fryer"
    _pristine: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def save_pristine(self) -> int:
        """Store the untouched state dict so :meth:`un_fry` can restore
        it.  Returns the number of tensors snapshotted."""
        self._pristine = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        return len(self._pristine)

    def un_fry(self) -> int:
        """Restore the last pristine snapshot.  Returns the number of
        tensors restored; 0 if no snapshot exists."""
        if not self._pristine:
            return 0
        with torch.no_grad():
            for name, tensor in self.model.state_dict().items():
                if name in self._pristine:
                    tensor.copy_(self._pristine[name])
        return len(self._pristine)

    # ------------------------------------------------------------------
    # Frying
    # ------------------------------------------------------------------

    def _matches(self, name: str) -> bool:
        if not self.patterns:
            return True
        return any(p in name for p in self.patterns)

    @torch.no_grad()
    def fry(self) -> dict[str, int]:
        """Perturb a ``fraction`` of each matching parameter's elements
        with ``N(0, noise_std * param.std())``.

        Returns a dict ``{param_name: n_elements_perturbed}`` for
        provenance.  Zero-perturbation params (all-zero tensors) are
        left untouched since the Gaussian would be degenerate.
        """
        # Pass 1 (v0.50): use a torch.Generator seeded from
        # ``self.seed`` so the noise is reproducible regardless of
        # the global torch RNG state.  Previously two callers with
        # the same seed but different global RNG states got
        # different noise.
        rng_py = random.Random(self.seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
        touched: dict[str, int] = {}
        for pname, p in self.model.named_parameters():
            if not p.requires_grad and not self._should_fry_frozen():
                continue
            if not self._matches(pname):
                continue
            flat = p.view(-1)
            n = flat.numel()
            if n == 0:
                continue
            k = max(1, int(n * self.fraction))
            idx = torch.tensor(
                rng_py.sample(range(n), k), device=flat.device, dtype=torch.long,
            )
            std = float(flat.std().item())
            if std == 0.0:
                continue
            # Per-parameter generator on the parameter's device so the
            # noise is reproducible on both CPU and CUDA.
            torch_rng = torch.Generator(device=flat.device)
            torch_rng.manual_seed(self.seed + sum(map(ord, pname)))
            noise = torch.randn(
                k, device=flat.device, dtype=flat.dtype, generator=torch_rng,
            ) * (self.noise_std * std)
            flat[idx] += noise
            self._apply_extra(flat, idx, rng_py)
            touched[pname] = k
        return touched

    def _should_fry_frozen(self) -> bool:
        return False

    def _apply_extra(
        self, flat: torch.Tensor, idx: torch.Tensor, rng: random.Random,
    ) -> None:
        """Hook for subclasses that want to do more than add noise."""
        return


# ---------------------------------------------------------------------------
# t1 — LightFry
# ---------------------------------------------------------------------------

@dataclass
class LightFry(Fryer):
    """Mild regularising perturbation.  Defaults: 2% of elements,
    0.1× param-std Gaussian noise."""

    fraction: float = 0.02
    noise_std: float = 0.1
    name: ClassVar[str] = "LightFry"


# ---------------------------------------------------------------------------
# t2 — HeavyFry
# ---------------------------------------------------------------------------

@dataclass
class HeavyFry(Fryer):
    """Severe perturbation.  Defaults: 30% of elements, 0.5× param-std
    Gaussian noise, plus :attr:`zero_rate` fraction of the touched
    elements reset to exactly zero.  Use for adversarial-robustness
    training datasets or for generating deliberately bad-model
    negatives to train a judge against."""

    fraction: float = 0.3
    noise_std: float = 0.5
    zero_rate: float = 0.1
    name: ClassVar[str] = "HeavyFry"

    def _should_fry_frozen(self) -> bool:
        return True

    def _apply_extra(
        self, flat: torch.Tensor, idx: torch.Tensor, rng: random.Random,
    ) -> None:
        if self.zero_rate <= 0:
            return
        n_zero = int(idx.numel() * self.zero_rate)
        if n_zero <= 0:
            return
        # Pick n_zero of the already-perturbed indices and zero them.
        zero_sample = torch.tensor(
            rng.sample(range(idx.numel()), n_zero),
            device=idx.device, dtype=torch.long,
        )
        flat[idx[zero_sample]] = 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TIERS: dict[str, type[Fryer]] = {
    "light-fry": LightFry,
    "heavy-fry": HeavyFry,
}


def deep_fryer(
    tier: str, model: nn.Module, *,
    fraction: float | None = None,
    noise_std: float | None = None,
    patterns: Iterable[str] = (),
    seed: int = 0,
    **extra: Any,
) -> Fryer:
    """Construct a deep-fryer tier by short name."""
    key = tier.lower().replace("_", "-")
    if key not in TIERS:
        raise ValueError(
            f"unknown fryer tier {tier!r}; valid: {sorted(TIERS)}",
        )
    cls = TIERS[key]
    kwargs: dict[str, Any] = {"model": model, "patterns": tuple(patterns), "seed": seed}
    if fraction is not None:
        kwargs["fraction"] = fraction
    if noise_std is not None:
        kwargs["noise_std"] = noise_std
    kwargs.update(extra)
    return cls(**kwargs)
