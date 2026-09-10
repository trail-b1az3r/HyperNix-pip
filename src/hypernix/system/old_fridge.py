"""old_fridge — cold storage for weights.

Memory-management helpers sitting between a fresh snapshot and the
training loop. Nothing here is novel; it's the small handful of things
you write once per project and then forget about:

* :func:`freeze` — set ``requires_grad=False`` on parameters matched by
  name patterns (glob or substring). Useful for training just the
  LM head on top of frozen trunk weights.
* :func:`unfreeze` — inverse.
* :func:`parameter_stats` — counts of total / trainable / frozen params
  and the memory footprint.
* :func:`offload_to_cpu` — move named submodules to CPU (the rest stays
  on the active device).
* :func:`chill_cache` — empty the CUDA cache and run ``gc.collect``.
* :func:`unwrap_model` — peel DDP / FSDP / DataParallel wrappers.

Everything tolerates being called on CPU-only models; the CUDA-only
primitives gate themselves behind ``torch.cuda.is_available()``.
"""
from __future__ import annotations

from hypernix.system.deprecation import deprecated_module

# Announced before the heavy imports below, so the notice reaches the
# operator immediately rather than after torch has finished loading.
deprecated_module(
    "hypernix.system.old_fridge",
    instead="hypernix.models.neo_oven",
    since="0.71.5a2",
)

import fnmatch
import gc
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ParamStats:
    total: int
    trainable: int
    frozen: int
    bytes: int

    @property
    def megabytes(self) -> float:
        return self.bytes / (1024 * 1024)


def _match_any(name: str, patterns: tuple[str, ...]) -> bool:
    """Return True if ``name`` matches any glob / substring pattern."""
    for p in patterns:
        if fnmatch.fnmatch(name, p) or p in name:
            return True
    return False


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP / FSDP / DataParallel wrappers for parameter access."""
    current = model
    seen: set[int] = set()
    while True:
        oid = id(current)
        if oid in seen:
            break
        seen.add(oid)
        if hasattr(current, "module"):
            current = current.module  # type: ignore[assignment]
            continue
        # FSDP v1/v2 and torch.distributed wrappers
        unwrapped = getattr(current, "_fsdp_wrapped_module", None)
        if unwrapped is not None:
            current = unwrapped
            continue
        unwrapped = getattr(current, "_orig_mod", None)
        if unwrapped is not None:
            current = unwrapped
            continue
        break
    return current

def freeze(model: nn.Module, patterns: Iterable[str] = ("embed_tokens",)) -> int:
    """Set ``requires_grad=False`` on parameters whose names match ``patterns``.

    Returns the number of frozen parameters. Default freezes just the
    token embedding — a common starting point when fine-tuning a judge
    or classifier head.
    """
    model = _unwrap_model(model)
    pats = tuple(patterns)
    n = 0
    for name, param in model.named_parameters():
        if _match_any(name, pats):
            if param.requires_grad:
                param.requires_grad = False
                n += param.numel()
    return n


def unfreeze(model: nn.Module, patterns: Iterable[str] = ("*",)) -> int:
    """Inverse of :func:`freeze`. Default unfreezes everything."""
    model = _unwrap_model(model)
    pats = tuple(patterns)
    n = 0
    for name, param in model.named_parameters():
        if _match_any(name, pats):
            if not param.requires_grad:
                param.requires_grad = True
                n += param.numel()
    return n


def parameter_stats(model: nn.Module) -> ParamStats:
    model = _unwrap_model(model)
    total = trainable = frozen = 0
    nbytes = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        nbytes += n * p.element_size()
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    return ParamStats(total=total, trainable=trainable, frozen=frozen, bytes=nbytes)


def offload_to_cpu(model: nn.Module, patterns: Iterable[str] = ("embed_tokens",)) -> int:
    """Move submodules matched by ``patterns`` to CPU. Returns how many moved.

    Use sparingly — repeatedly shuttling tensors between CPU and GPU is
    slow. Intended for the "huge embeddings + tiny active set" pattern
    where the embedding table is cold during most of training.
    """
    model = _unwrap_model(model)
    pats = tuple(patterns)
    moved = 0
    for name, module in model.named_modules():
        if name and _match_any(name, pats):
            module.to("cpu")
            moved += 1
    return moved


def chill_cache() -> None:
    """Free unreferenced tensors from the allocator cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner module, unwrapping DDP / FSDP / DataParallel."""
    return _unwrap_model(model)
