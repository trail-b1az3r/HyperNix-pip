"""strife.gpu — whole layers on a GPU, through PyTorch.

``gpu_layers=N`` puts the weight matrices of the first *N* blocks on a
device; the rest stay on the CPU. A layer's weights are decoded once to
half precision, moved to the device, and its matmuls run there. Everything
else -- attention arithmetic, norms, the KV cache -- stays in host memory,
so this is partial *tensor* offload: the heavy multiplies on the card, the
state on the host.

What this is, plainly:

* It needs PyTorch with a working CUDA device. Without one the request is
  answered with ``gpu_layers_used: 0`` and the reason.
* A card PyTorch was not built for is refused by name. Pascal (a GTX 1080,
  compute capability 6.1) is the common case: current PyTorch wheels do
  not include its kernels, and trying anyway fails inside CUDA.
* How many layers fit is decided from the card's **free** memory, less a
  reserve, never its total. A model that does not fit entirely is split,
  and the plan says how.
* The device path is exercised in the tests on PyTorch's CPU device
  (``device="cpu"``), which runs the same code. **It has not been run on a
  real GPU by its author**: the machine it was written on has none.
  ``hnx strife benchmark`` on your card is the evidence for your card.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import StrifeError

__all__ = ["DeviceWeights", "choose_device", "layer_bytes", "layers_that_fit",
           "RESERVE_BYTES", "ALL_LAYERS"]

#: VRAM left alone: the CUDA context, the allocator's slack, and whatever
#: the desktop needs.
RESERVE_BYTES = 768 << 20
#: ``gpu_layers`` meaning "as many as fit".
ALL_LAYERS = -1
_LAYER = re.compile(r"^blk\.(\d+)\.")


def layer_of(name: str) -> int | None:
    match = _LAYER.match(name)
    return int(match.group(1)) if match else None


def layer_bytes(gguf: Any, *, bytes_per_weight: int = 2) -> dict[int, int]:
    """Device memory each block's matrices need once decoded (half precision by default)."""
    out: dict[int, int] = {}
    for tensor in gguf.tensors:
        index = layer_of(tensor.name)
        if index is None or len(tensor.shape) < 2:
            continue
        elements = 1
        for dim in tensor.shape:
            elements *= int(dim)
        out[index] = out.get(index, 0) + elements * bytes_per_weight
    return out


def layers_that_fit(per_layer: dict[int, int], wanted: int, free_bytes: int,
                    reserve: int = RESERVE_BYTES) -> tuple[int, int]:
    """``(layers, bytes)``: the first layers, in order, that fit in what is free."""
    budget = max(0, int(free_bytes) - int(reserve))
    limit = len(per_layer) if wanted == ALL_LAYERS else max(0, min(int(wanted), len(per_layer)))
    used = count = 0
    for index in sorted(per_layer)[:limit]:
        if used + per_layer[index] > budget:
            break
        used += per_layer[index]
        count += 1
    return count, used


@dataclass
class DeviceChoice:
    device: str = ""                 # "" when nothing can be used
    name: str = ""
    free_bytes: int = 0
    total_bytes: int = 0
    half: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.device)


def choose_device(wanted: str | None = None) -> DeviceChoice:
    """The device ``gpu_layers`` would use, or why there is none.

    *wanted* is ``None``/``"auto"`` (the first CUDA device), ``"cuda:N"``, or
    ``"cpu"`` -- PyTorch's CPU device, which exists so the device path can
    be tested and compared on a machine without a GPU.
    """
    choice = DeviceChoice()
    try:
        import torch
    except ImportError:
        choice.reasons.append("PyTorch is not installed, so no layer can go on a GPU "
                              "(pip install torch)")
        return choice
    wanted = (wanted or "auto").lower()
    if wanted == "cpu":
        import psutil

        memory = psutil.virtual_memory()
        return DeviceChoice(device="cpu", name="PyTorch CPU device (testing)",
                            free_bytes=int(memory.available), total_bytes=int(memory.total),
                            half=False)
    if not torch.cuda.is_available():
        choice.reasons.append("PyTorch finds no CUDA device, so every layer runs on the CPU")
        return choice
    index = 0
    if wanted.startswith("cuda:"):
        try:
            index = int(wanted.split(":", 1)[1])
        except ValueError:
            choice.reasons.append(f"{wanted!r} is not a device")
            return choice
    if not 0 <= index < torch.cuda.device_count():
        choice.reasons.append(f"there is no cuda:{index} (PyTorch sees {torch.cuda.device_count()})")
        return choice
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    built_for = {arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")}
    if built_for and f"sm_{major}{minor}" not in built_for:
        choice.reasons.append(
            f"{name} has compute capability {major}.{minor}, which this PyTorch "
            f"({torch.__version__}) was not built for (it has {', '.join(sorted(built_for))}); "
            "every layer runs on the CPU. A PyTorch build that includes it would be used.")
        return choice
    try:
        free, total = torch.cuda.mem_get_info(index)
    except RuntimeError as exc:
        choice.reasons.append(f"cuda:{index} could not report its memory: {exc}")
        return choice
    return DeviceChoice(device=f"cuda:{index}", name=name, free_bytes=int(free),
                        total_bytes=int(total), half=True)


class DeviceWeights:
    """The matrices of the offloaded layers, on the device."""

    def __init__(self, device: str, *, half: bool = True) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if half else torch.float32
        self._tensors: dict[str, Any] = {}
        self.bytes = 0
        self.matmuls = 0

    def has(self, name: str) -> bool:
        return name in self._tensors

    def put(self, name: str, array: np.ndarray) -> None:
        """Move a decoded ``(rows, cols)`` float32 weight to the device."""
        torch = self._torch
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self.device, self.dtype)
        except RuntimeError as exc:           # out of memory, most often
            raise StrifeError("strife.gpu.load_failed",
                              f"{name} could not be placed on {self.device}: {exc}",
                              remedy="Ask for fewer --gpu-layers.") from exc
        self._tensors[name] = tensor
        self.bytes += tensor.element_size() * tensor.nelement()

    def matmul(self, x: np.ndarray, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        """``x @ W.T`` on the device; the result comes back as float32."""
        torch = self._torch
        weight = self._tensors[name]
        if rows is not None:
            weight = weight[rows[0]:rows[1]]
        with torch.inference_mode():
            xd = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device,
                                                                                self.dtype)
            out = (xd @ weight.T).to(torch.float32).cpu().numpy()
        self.matmuls += 1
        return out

    def peak_bytes(self) -> int:
        if self.device.type == "cuda":
            return int(self._torch.cuda.max_memory_allocated(self.device))
        return self.bytes

    def stats(self) -> dict[str, Any]:
        return {"device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "tensors": len(self._tensors), "bytes": self.bytes,
                "peak_bytes": self.peak_bytes(), "matmuls": self.matmuls}

    def close(self) -> None:
        self._tensors.clear()
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()
