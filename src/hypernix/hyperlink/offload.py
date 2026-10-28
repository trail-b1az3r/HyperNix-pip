"""CPU offloading for the HyperNix runners (0.72.8rc1).

A model that does not fit in VRAM used to have two choices: fewer layers
on the GPU (llama.cpp's ``-ngl``), or -- for a native HyperNix model
served by :mod:`.brewed_server` -- the whole model on one device, which
meant the CPU for anything bigger than the card. This is the rest:

* :class:`Offload` -- the llama.cpp side. ``cpu_moe`` keeps a
  mixture-of-experts model's expert weights in system RAM
  (``--cpu-moe`` / ``--n-cpu-moe N``), which is most of such a model and
  the part only a few of which run per token, so everything else can
  stay on the GPU. ``kv_on_cpu`` keeps the KV cache in RAM
  (``--no-kv-offload``), trading speed for a long context on a small
  card.
* :func:`split_layers` -- the PyTorch side. The first *gpu_layers*
  blocks of a Brewer model on the GPU with the embeddings and the head,
  the rest in RAM; activations cross once each way per forward. The
  same meaning ``gpu_layers`` has for llama.cpp.
* :func:`fit_layers` -- how many blocks fit in the free VRAM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Offload", "fit_layers", "split_layers"]

#: Kept free on the GPU beyond the weights: activations, the attention
#: scratch, and whatever else is running.
VRAM_HEADROOM = 0.15


@dataclass(frozen=True)
class Offload:
    #: Experts of a mixture-of-experts model in system RAM: 0 none, -1
    #: every layer's, N the first N layers' (the rest stay on the GPU).
    cpu_moe: int = 0
    #: The KV cache in system RAM rather than VRAM.
    kv_on_cpu: bool = False

    @property
    def any(self) -> bool:
        return bool(self.cpu_moe) or self.kv_on_cpu

    def llama_argv(self, fork: str = "") -> list[str]:
        """llama-server's flags for this. KoboldCpp has no equivalents."""
        if not self.any:
            return []
        if fork == "kobold":
            raise ValueError("KoboldCpp cannot keep experts or the KV cache on the CPU; "
                             "use the llama.cpp backend for CPU offloading.")
        argv: list[str] = []
        if self.cpu_moe < 0:
            argv.append("--cpu-moe")
        elif self.cpu_moe > 0:
            argv += ["--n-cpu-moe", str(int(self.cpu_moe))]
        if self.kv_on_cpu:
            argv.append("--no-kv-offload")
        return argv

    def describe(self) -> str:
        parts = []
        if self.cpu_moe < 0:
            parts.append("every expert in RAM")
        elif self.cpu_moe > 0:
            parts.append(f"experts of the first {self.cpu_moe} layers in RAM")
        if self.kv_on_cpu:
            parts.append("the KV cache in RAM")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _block_bytes(module: Any) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def fit_layers(model: Any, free_bytes: int, *, headroom: float = VRAM_HEADROOM) -> int:
    """How many of *model*'s blocks fit in *free_bytes* of VRAM, with the
    embeddings and the head (which go on the GPU first) and *headroom*
    kept free. ``len(blocks)`` when all of it fits."""
    inner = getattr(model, "inner", model)
    budget = int(free_bytes * (1.0 - headroom))
    fixed = _block_bytes(inner.embed) + _block_bytes(inner.norm_out)
    if inner.lm_head.weight is not inner.embed.embed.weight:
        fixed += _block_bytes(inner.lm_head)
    budget -= fixed
    count = 0
    for block in inner.blocks:
        size = _block_bytes(block)
        if size > budget:
            break
        budget -= size
        count += 1
    return count


def _move(value: Any, device: Any) -> Any:
    import torch

    return value.to(device) if isinstance(value, torch.Tensor) else value


def split_layers(model: Any, gpu_layers: int, *, device: str = "cuda", dtype: Any = None) -> dict[str, Any]:
    """Put the first *gpu_layers* blocks of a Brewer model (or its adapter)
    on *device* and the rest on the CPU, with the embeddings, final norm and
    head on *device*. Inputs are moved to each part's device by forward
    pre-hooks, so the model is called exactly as before.

    ``gpu_layers`` of 0 is all on the CPU; at least ``len(blocks)`` is all
    on *device*. Returns where everything went.
    """
    import torch

    inner = getattr(model, "inner", model)
    blocks = list(inner.blocks)
    total = len(blocks)
    on_gpu = max(0, min(int(gpu_layers), total))
    gpu, cpu = torch.device(device), torch.device("cpu")
    head = gpu if on_gpu else cpu

    def place(module: Any, where: Any) -> None:
        module.to(where, dtype=dtype) if dtype is not None else module.to(where)
        module.register_forward_pre_hook(
            lambda _m, args, kwargs, where=where: (
                tuple(_move(a, where) for a in args),
                {k: _move(v, where) for k, v in kwargs.items()}),
            with_kwargs=True,
        )

    # The head shares its weight with the embedding when the model ties
    # them, so the two must be on one device.
    place(inner.embed, head)
    for index, block in enumerate(blocks):
        place(block, gpu if index < on_gpu else cpu)
    place(inner.norm_out, head)
    place(inner.lm_head, head)
    return {"gpu_layers": on_gpu, "total_layers": total, "device": str(gpu) if on_gpu else "cpu",
            "gpu_bytes": sum(_block_bytes(b) for b in blocks[:on_gpu]),
            "cpu_bytes": sum(_block_bytes(b) for b in blocks[on_gpu:])}
