"""Run Llama-architecture models on the llama-essir Hub kernel.

``llama-essir`` is this project's kernel on the Hugging Face Kernel Hub
(``ray0rf1re/llama-essir``, built from ``native/llama-essir``): Triton
RMSNorm, rotary position embeddings and the SwiGLU gate. This module
points the ``kernels`` library at it, so a Transformers Llama -- or any
model built from the same blocks -- swaps those layers for the kernel's::

    from transformers import AutoModelForCausalLM
    from hypernix.hub_kernels import kernelize_llama

    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="bfloat16").cuda()
    kernelize_llama(model)

Trust is scoped: the kernel's publisher is not on the ``kernels`` trusted
list, so loading it needs consent, and this module gives it to exactly
``ray0rf1re/llama-essir`` -- not to the publisher's other repositories,
and not to anything else a mapping might name.

Needs ``pip install kernels`` (and Transformers for a Transformers model).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    import torch.nn as nn

__all__ = [
    "DEVICES",
    "REPO_ID",
    "VERSION",
    "KernelsUnavailable",
    "kernel_mapping",
    "kernelize_llama",
    "load",
]

#: The Hub repository and the major version this code is written against.
#: A new major version is a new API; bumping this is a code change.
REPO_ID = "ray0rf1re/llama-essir"
VERSION = 1

#: Device types the kernel has GPU code for. Elsewhere the layers keep
#: their own forward: ``kernels.kernelize`` only targets accelerators.
DEVICES = ("cuda", "rocm")

#: Transformers' hook name -> (kind, name in the kernel).
_HOOKS: Mapping[str, tuple[str, str]] = {
    "RMSNorm": ("layer", "RMSNorm"),
    "rotary_pos_emb": ("func", "apply_rotary_transformers"),
    "SiluAndMul": ("layer", "SiluAndMul"),
}


class KernelsUnavailable(ImportError):
    """The ``kernels`` library is not installed."""


def _kernels() -> ModuleType:
    try:
        import kernels
    except ImportError as exc:
        raise KernelsUnavailable(
            "Hub kernels need the `kernels` library: pip install kernels"
        ) from exc
    return kernels


def load(*, version: int = VERSION, local_path: str | Path | None = None) -> ModuleType:
    """The kernel module itself: ``rms_norm``, ``rotary``,
    ``silu_and_mul``, ``apply_rotary_transformers`` and ``layers``.

    *local_path* loads a local build instead of the Hub copy (the
    directory holding ``build/``; ``native/llama-essir/local_build.py``
    makes one).
    """
    kernels = _kernels()
    if local_path is not None:
        return kernels.get_local_kernel(Path(local_path))
    return kernels.get_kernel(REPO_ID, version=version, trust_remote_code=[REPO_ID])


def kernel_mapping(
    *, version: int = VERSION, local_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """The ``kernels`` layer mapping for the three Llama hooks, per device.

    Pass it to ``kernels.use_kernel_mapping`` to combine it with other
    mappings; :func:`kernelize_llama` does that for you.
    """
    kernels = _kernels()
    mapping: dict[str, dict[str, Any]] = {}
    for hook, (kind, name) in _HOOKS.items():
        per_device = {}
        for device in DEVICES:
            per_device[device] = _repository(kernels, kind, name, version, local_path)
        mapping[hook] = per_device
    return mapping


def _repository(kernels: ModuleType, kind: str, name: str, version: int,
                local_path: str | Path | None) -> Any:
    if local_path is not None:
        path = Path(local_path)
        if kind == "layer":
            return kernels.LocalLayerRepository(path, layer_name=name)
        return kernels.LocalFuncRepository(path, func_name=name)
    if kind == "layer":
        return kernels.LayerRepository(repo_id=REPO_ID, layer_name=name, version=version,
                                       trust_remote_code=[REPO_ID])
    return kernels.FuncRepository(repo_id=REPO_ID, func_name=name, version=version,
                                  trust_remote_code=[REPO_ID])


def kernelize_llama(
    model: nn.Module,
    *,
    mode: str = "inference",
    device: str | None = None,
    version: int = VERSION,
    local_path: str | Path | None = None,
    use_fallback: bool = True,
) -> nn.Module:
    """Swap *model*'s RMSNorm, rotary embedding and SwiGLU gate for the
    llama-essir kernels, in place; returns the model.

    *mode* is ``"inference"`` or ``"training"``, optionally with
    ``"+compile"`` for a model you will ``torch.compile``. The kernels have
    no backward pass, so in training mode the layers keep their own
    forward (``use_fallback=False`` raises instead). *device* is the
    device type to load for (``"cuda"`` or ``"rocm"``); by default it is
    read from the model's parameters.
    """
    kernels = _kernels()
    with kernels.use_kernel_mapping(kernel_mapping(version=version, local_path=local_path)):
        return kernels.kernelize(model, mode=_mode(kernels, mode), device=device,
                                 use_fallback=use_fallback)


def _mode(kernels: ModuleType, mode: str) -> Any:
    parts = {part.strip().lower() for part in mode.split("+")}
    base = parts - {"compile"}
    if base == {"inference"}:
        chosen = kernels.Mode.INFERENCE
    elif base == {"training"}:
        chosen = kernels.Mode.TRAINING
    else:
        raise ValueError(f"mode must be 'inference' or 'training' (optionally '+compile'), not {mode!r}")
    if "compile" in parts:
        chosen = chosen | kernels.Mode.TORCH_COMPILE
    return chosen
