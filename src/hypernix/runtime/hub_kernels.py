"""Run Llama-architecture models on the llama-easy Hub kernel.

``llama-easy`` is this project's kernel on the Hugging Face Kernel Hub
(``ray0rf1re/llama-easy``, built from ``native/llama-easy``): Triton
RMSNorm, rotary position embeddings and the SwiGLU gate. This module
points the ``kernels`` library at it, so a Transformers Llama -- or any
model built from the same blocks -- swaps those layers for the kernel's::

    from hypernix.hub_kernels import from_pretrained

    model = from_pretrained("<any Llama-architecture model>", device_map="cuda")

or, for a model already loaded, ``kernelize_llama(model)``.

Trust is scoped: the kernel's publisher is not on the ``kernels`` trusted
list, so loading it needs consent, and this module gives it to exactly
``ray0rf1re/llama-easy`` -- not to the publisher's other repositories,
and not to anything else a mapping might name.

Needs ``pip install kernels`` (and Transformers for a Transformers model).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "DEVICES",
    "REPO_ID",
    "VERSION",
    "KernelsUnavailable",
    "from_pretrained",
    "kernel_config",
    "kernel_mapping",
    "kernelize_llama",
    "load",
]

#: The Hub repository and the major version this code is written against.
#: A new major version is a new API; bumping this is a code change.
REPO_ID = "ray0rf1re/llama-easy"
VERSION = 1

#: Device types the kernel has GPU code for. Elsewhere the layers keep
#: their own forward: ``kernels.kernelize`` only targets accelerators.
DEVICES = ("cuda", "rocm")

#: Transformers' hook name -> the layer in the kernel.
_HOOKS: Mapping[str, str] = {
    "RMSNorm": "RMSNorm",
    "rotary_pos_emb": "ApplyRotary",
    "SiluAndMul": "SiluAndMul",
}


#: The hooks a ``KernelConfig`` names: the two every Llama carries.
#: Transformers refuses a ``KernelConfig`` that names a hook the model
#: lacks, and Llama has no ``SiluAndMul``.
_CONFIG_LAYERS: Mapping[str, str] = {hook: _HOOKS[hook] for hook in ("RMSNorm", "rotary_pos_emb")}


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
    directory holding ``build/``; ``native/llama-easy/local_build.py``
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
    for hook, layer in _HOOKS.items():
        mapping[hook] = {device: _repository(kernels, layer, version, local_path) for device in DEVICES}
    return mapping


def _repository(kernels: ModuleType, layer: str, version: int, local_path: str | Path | None) -> Any:
    if local_path is not None:
        return kernels.LocalLayerRepository(Path(local_path), layer_name=layer)
    return kernels.LayerRepository(repo_id=REPO_ID, layer_name=layer, version=version,
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
    llama-easy kernels, in place; returns the model.

    *mode* is ``"inference"`` or ``"training"``, optionally with
    ``"+compile"`` for a model you will ``torch.compile``. The kernels have
    no backward pass, so in training mode the layers keep their own
    forward (``use_fallback=False`` raises instead). *device* is the
    device type to load for (``"cuda"`` or ``"rocm"``); by default it is
    read from the model's parameters.
    """
    kernels = _kernels()
    # inherit_mapping=False: only this kernel. Inheriting would also apply
    # whatever global mapping is registered -- Transformers registers its
    # own -- and download kernels for hooks nobody asked to replace.
    with kernels.use_kernel_mapping(kernel_mapping(version=version, local_path=local_path),
                                    inherit_mapping=False):
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


def kernel_config(*, version: int = VERSION, local_path: str | Path | None = None) -> Any:
    """A Transformers ``KernelConfig`` for this kernel, for
    ``from_pretrained(..., kernel_config=...)``: Transformers then swaps
    the layers in itself as it loads the model, on the model's device.
    """
    try:
        from transformers import KernelConfig
    except ImportError as exc:
        raise KernelsUnavailable("kernel_config needs Transformers 5 or newer: pip install -U transformers") from exc
    _kernels()
    if local_path is not None:
        root = Path(local_path).resolve()
        return KernelConfig({hook: f"{root}:{layer}" for hook, layer in _CONFIG_LAYERS.items()},
                            use_local_kernel=True)
    # Transformers takes a bool here, and applies it to this one repository.
    return KernelConfig({
        hook: (f"{REPO_ID}:{layer}", {"version": version, "trust_remote_code": True})
        for hook, layer in _CONFIG_LAYERS.items()
    })


def from_pretrained(
    model_id: str | Path,
    *,
    version: int = VERSION,
    local_path: str | Path | None = None,
    **kwargs: Any,
) -> nn.Module:
    """``AutoModelForCausalLM.from_pretrained``, then the kernel swapped in.

    The easiest way in: ``from_pretrained("<Llama model>", device_map="cuda")``.
    Every other keyword goes to Transformers (``torch_dtype``,
    ``device_map``, ``revision`` ...). A model that lands on the CPU is
    returned unchanged -- the kernel is GPU code, and ``kernelize`` does
    not target the CPU -- and the log says so.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    device = _device_type(model)
    if device not in DEVICES:
        logger.info("llama-easy: %s is on %s, so it keeps its own layers.", model_id, device)
        return model
    return kernelize_llama(model, device=device, version=version, local_path=local_path)


def _device_type(model: nn.Module) -> str:
    try:
        param = next(model.parameters())
    except StopIteration:
        return "cpu"
    import torch

    if param.device.type == "cuda" and torch.version.hip is not None:
        return "rocm"
    return param.device.type
