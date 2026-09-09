"""hypernix.models.hnxdevice — which accelerators can actually run this.

    hypernix devices
    hypernix generate --model-dir m.gguf --device cuda
    hypernix hyprslug-headers serve m.gguf --device auto

Picking a device is three questions, and conflating them is how a runtime
ends up raising a CUDA error at the first matmul instead of a sentence at
startup:

1. Is the backend **built into this torch**? A wheel carries a fixed list
   of GPU architectures and nothing outside it will run.
2. Is there a **device present** for it?
3. Is that device's **architecture in the wheel's list**?

The third is the one that bites, and it is the reason this module exists.

The Pascal problem, concretely
------------------------------
A GTX 1080 is compute capability **6.1**. Recent torch wheels do not
build for it::

    >>> torch.__version__
    '2.13.0+cu130'
    >>> torch.cuda.get_arch_list()
    ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']

Install that on a 1080 and ``torch.cuda.is_available()`` returns **True**.
Everything looks fine until the first kernel launch, which fails with
``no kernel image is available for execution on the device`` — an error
that reads like a broken driver and is actually a wheel that was never
built for the card.

So :func:`probe` compares the device's capability against
``get_arch_list()`` and says which wheel to install instead. CUDA 11.8
and 12.1–12.6 wheels carry ``sm_61``; from roughly cu128 onward they do
not.

Half precision is a separate trap on the same card. GP102/104 run FP16 at
**1/64** of FP32 throughput — it is present, so a naive "use fp16 on
CUDA" rule finds it and makes the model sixty times slower. Anything
below ``sm_70`` gets float32 here, and :func:`default_dtype` says so.

Vulkan
------
There is no useful Vulkan path *through torch*. The backend exists in the
source tree, is not built into any released wheel, and implements a small
set of vision ops rather than what a transformer needs. Reporting it as
"available" because an import succeeds would be a lie with a long
debugging tail.

The Vulkan path that works is **llama.cpp's**, which is what LM Studio
ships and uses on AMD, Intel and older NVIDIA cards. A HyperNix sub-bit
model cannot go there as it stands — no llama.cpp has the kernels — but
:func:`hypernix.quant.hyprslug_headers.wrap` converts one into a type
that can. So ``--device vulkan`` answers with that command rather than
with an exception, which is the difference between a dead end and a
route.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Device",
    "DeviceError",
    "probe",
    "select",
    "describe",
    "default_dtype",
    "capability_of",
    "wheel_hint",
    "VULKAN_ADVICE",
]


class DeviceError(RuntimeError):
    """A device was asked for that cannot run this."""


#: Shared-object names that mean "torch is installed but its CUDA
#: runtime is not", mapped to the wheel that carries each. A CUDA torch
#: pulls these in as separate ``nvidia-*`` packages; install it where one
#: cannot be resolved -- a slim image, a pruned virtualenv, a uv sync
#: that dropped an optional dependency -- and ``import torch`` dies with
#: a bare linker error naming a file nobody has heard of.
_CUDA_RUNTIME_LIBS = {
    "libcusparseLt": "nvidia-cusparselt-cu12",
    "libcusparse": "nvidia-cusparse-cu12",
    "libcublas": "nvidia-cublas-cu12",
    "libcublasLt": "nvidia-cublas-cu12",
    "libcudart": "nvidia-cuda-runtime-cu12",
    "libcudnn": "nvidia-cudnn-cu12",
    "libcufft": "nvidia-cufft-cu12",
    "libcurand": "nvidia-curand-cu12",
    "libcusolver": "nvidia-cusolver-cu12",
    "libnccl": "nvidia-nccl-cu12",
    "libnvJitLink": "nvidia-nvjitlink-cu12",
    "libnvrtc": "nvidia-cuda-nvrtc-cu12",
}


def _torch_unavailable_reason(exc: ImportError) -> str:
    """One line for a device row, for when torch will not import.

    "torch is not installed" was the only thing the probes used to say,
    and it is wrong in the case people actually hit: torch *is*
    installed, and a shared object it links against is not. Reporting the
    absent package for a present one sends the reader to reinstall the
    thing they already have.
    """
    text = str(exc)
    if "No module named" in text and "torch" in text:
        return "torch is not installed."
    missing = text.split(":", 1)[0].strip()
    if missing.startswith("lib") and ".so" in missing:
        return (f"torch is installed but cannot load {missing}, one of the "
                f"NVIDIA runtime wheels a CUDA build needs.")
    return f"torch is installed but will not import: {text}"


def import_torch():
    """``import torch``, or a sentence saying why it could not be done.

    The runtime imports torch lazily and in several places, so without
    this every one of them can end a command with a linker traceback::

        ImportError: libcusparseLt.so.0: cannot open shared object file:
        No such file or directory

    That names a file the reader has never installed, in a package they
    did not know torch depended on, from a command that was about to
    load a GGUF. Nothing in it says the actual situation: a CUDA build of
    torch is installed and one of its NVIDIA runtime wheels is missing,
    and a machine that only wants to run a model on the CPU never needed
    that build at all.
    """
    try:
        import torch
    except ImportError as exc:
        raise DeviceError(_torch_import_advice(exc)) from exc
    return torch


def _torch_import_advice(exc: ImportError) -> str:
    text = str(exc)
    if "No module named" in text and "torch" in text:
        return (
            "PyTorch is not installed, and the sub-bit runtime needs it to "
            "run a model.\n"
            "    pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "is the smallest install that works; drop the --index-url for "
            "the CUDA build. `hypernix devices` then reports what it found."
        )

    missing = text.split(":", 1)[0].strip()
    stem = missing.split(".so")[0]
    package = _CUDA_RUNTIME_LIBS.get(stem)
    if not (missing.startswith("lib") and ".so" in missing):
        return (
            f"PyTorch is installed but will not import: {text}\n"
            f"Run `python -c \'import torch\'` to see it directly. The "
            f"sub-bit runtime cannot load a model until that succeeds."
        )

    remedy = (
        f"    pip install {package}\n" if package else
        "    reinstall torch so its CUDA runtime wheels come with it\n"
    )
    return (
        f"PyTorch is installed but cannot load {missing}, so importing it "
        f"fails before this command can start.\n"
        f"\n"
        f"That library ships in a separate NVIDIA wheel that a CUDA build of "
        f"torch depends on"
        + (f" ({package})" if package else "")
        + ". It is missing here, which usually means the CUDA build was "
        "installed into an environment that did not pull its runtime "
        "dependencies.\n"
        "\n"
        "Two ways out. If this machine has no NVIDIA GPU, or you only want "
        "to run on the CPU, the CPU build is smaller and has none of these "
        "dependencies:\n"
        "    pip install --force-reinstall torch --index-url "
        "https://download.pytorch.org/whl/cpu\n"
        "To keep CUDA, install the missing runtime:\n"
        + remedy +
        "\n"
        "`hypernix devices` reports what torch can actually use once it "
        "imports."
    )


#: Compute capability -> the cards people call it. Only what changes a
#: decision: whether the wheel builds for it, and whether FP16 is real.
_ARCH_NAMES = {
    (3, 5): "Kepler (K40, GTX 780)",
    (3, 7): "Kepler (K80)",
    (5, 0): "Maxwell (GTX 750 Ti, M10)",
    (5, 2): "Maxwell (GTX 970/980, M60)",
    (6, 0): "Pascal (P100)",
    (6, 1): "Pascal (GTX 1060/1070/1080, Titan Xp, P40)",
    (7, 0): "Volta (V100)",
    (7, 5): "Turing (RTX 2080, T4)",
    (8, 0): "Ampere (A100)",
    (8, 6): "Ampere (RTX 3090, A10)",
    (8, 9): "Ada (RTX 4090, L40)",
    (9, 0): "Hopper (H100)",
    (10, 0): "Blackwell (B200)",
    (12, 0): "Blackwell (RTX 50xx)",
}

#: The first capability with tensor cores and full-rate FP16. Below this,
#: half precision exists and is a trap: GP102/104 run it at 1/64.
_FP16_FROM = (7, 0)

VULKAN_ADVICE = (
    "PyTorch has no usable Vulkan backend — it is not built into any released "
    "wheel and implements vision ops, not a transformer. The Vulkan path that "
    "works is llama.cpp's, which is what LM Studio uses on AMD, Intel and "
    "older NVIDIA cards. Convert the model to a type it can read:\n"
    "    hypernix hyprslug-headers wrap MODEL.gguf -o compat.gguf\n"
    "and load compat.gguf in LM Studio with the Vulkan runtime selected. "
    "That is a stock-quantised copy, not a sub-bit model — see "
    "wiki/HyprSlug-Headers.md. To keep the tier, run it on CPU or CUDA here "
    "and reach it over HTTP with `hyprslug-headers serve`."
)


@dataclass
class Device:
    """One candidate, and whether it can actually run a model."""

    name: str
    kind: str
    label: str = ""
    usable: bool = False
    #: Why not, when not. Empty when usable.
    reason: str = ""
    #: ``(major, minor)`` for CUDA, otherwise ``None``.
    capability: tuple[int, int] | None = None
    total_memory: int = 0
    #: What to do about it, when there is something to do.
    remedy: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def torch_device(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "label": self.label,
            "usable": self.usable,
            "reason": self.reason,
            "capability": (
                f"{self.capability[0]}.{self.capability[1]}"
                if self.capability else None
            ),
            "total_memory": self.total_memory,
            "remedy": self.remedy,
            **self.extra,
        }

    def describe(self) -> str:
        mark = "ok " if self.usable else "no "
        memory = f"  {self.total_memory / 1e9:.1f} GB" if self.total_memory else ""
        line = f"  {mark} {self.name:10} {self.label}{memory}"
        if not self.usable and self.reason:
            line += f"\n        {self.reason}"
        if self.remedy:
            line += f"\n        {self.remedy}"
        return line


def capability_of(index: int = 0) -> tuple[int, int] | None:
    """``(major, minor)`` of CUDA device *index*, or ``None``."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability(index))  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - probing must never raise
        return None


def _arch_list() -> list[str]:
    try:
        import torch

        return list(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        return []


def _built_capabilities() -> list[tuple[int, int]]:
    """The capabilities this torch wheel actually has kernels for."""
    out: list[tuple[int, int]] = []
    for entry in _arch_list():
        digits = entry.removeprefix("sm_").removeprefix("compute_")
        if digits.isdigit() and len(digits) >= 2:
            out.append((int(digits[:-1]), int(digits[-1])))
    return sorted(set(out))


def _runs_on(capability: tuple[int, int], built: list[tuple[int, int]]) -> bool:
    """Whether a wheel built for *built* can launch on *capability*.

    Exact match, or a lower-minor build of the same major — CUDA is
    binary-compatible upward within a major version, so an ``sm_80``
    cubin runs on ``sm_86``. It is *not* compatible downward, which is
    the direction that matters here.
    """
    if not built:
        return False
    if capability in built:
        return True
    same_major = [c for c in built if c[0] == capability[0] and c[1] <= capability[1]]
    return bool(same_major)


def wheel_hint(capability: tuple[int, int]) -> str:
    """Which torch wheel to install for a card this build cannot reach."""
    major, minor = capability
    if (major, minor) < (5, 0):
        return (
            "No current PyTorch supports this card. The last that did was "
            "torch 1.13 with CUDA 11.7, which this package no longer targets."
        )
    if (major, minor) <= (6, 1):
        return (
            "Pascal and Maxwell need a CUDA 11.8 or 12.1-12.6 wheel; builds "
            "from about cu128 onward drop sm_50 through sm_61. Install:\n"
            "    pip install torch --index-url "
            "https://download.pytorch.org/whl/cu118"
        )
    if (major, minor) < (7, 5):
        return (
            "Volta needs a CUDA 11.8 or 12.x wheel. Install:\n"
            "    pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121"
        )
    return (
        "This card is newer than the wheel. Install a build that names it:\n"
        "    pip install --upgrade torch --index-url "
        "https://download.pytorch.org/whl/cu128"
    )


def default_dtype(device: Device | str):
    """The compute dtype to use on *device*.

    float32 everywhere below ``sm_70``, and that is the whole point of
    the function. FP16 is *present* on Pascal and runs at 1/64 rate on
    GP102/104, so a rule as reasonable-looking as "half on CUDA, float on
    CPU" makes a GTX 1080 dramatically slower while appearing to
    optimise it.
    """
    import torch

    if isinstance(device, str):
        device = _one(device)
    if device.kind != "cuda" or device.capability is None:
        return torch.float32
    return torch.float16 if device.capability >= _FP16_FROM else torch.float32


def _cpu_device() -> Device:
    import platform

    machine = platform.machine()
    threads = os.cpu_count() or 1
    return Device(
        name="cpu",
        kind="cpu",
        label=f"{machine}, {threads} threads",
        usable=True,
        extra={"threads": threads, "machine": machine},
    )


def _cuda_devices() -> list[Device]:
    try:
        import torch
    except ImportError as exc:
        return [Device("cuda", "cuda", "CUDA",
                       reason=_torch_unavailable_reason(exc),
                       remedy='Run `hypernix devices` for the full diagnosis, or install the CPU build: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu')]

    if not torch.version.cuda and not torch.version.hip:
        return [Device(
            "cuda", "cuda", "CUDA",
            reason=f"This torch ({torch.__version__}) is a CPU-only build.",
            remedy="pip install torch --index-url "
                   "https://download.pytorch.org/whl/cu121",
        )]

    if not torch.cuda.is_available():
        return [Device(
            "cuda", "cuda",
            f"CUDA {torch.version.cuda or torch.version.hip}",
            reason="No CUDA device is visible. Either there is no GPU, the "
                   "driver is not loaded, or CUDA_VISIBLE_DEVICES hides it.",
        )]

    built = _built_capabilities()
    found: list[Device] = []
    for index in range(torch.cuda.device_count()):
        capability = capability_of(index)
        try:
            properties = torch.cuda.get_device_properties(index)
            name = properties.name
            memory = int(properties.total_memory)
        except Exception:  # noqa: BLE001
            name, memory = f"cuda:{index}", 0

        architecture = _ARCH_NAMES.get(capability, "")
        label = f"{name} (sm_{capability[0]}{capability[1]}"
        label += f", {architecture})" if architecture else ")"

        device = Device(
            name=f"cuda:{index}", kind="cuda", label=label,
            capability=capability, total_memory=memory,
            extra={"arch_list": built and [f"sm_{a}{b}" for a, b in built]},
        )
        if capability is None:
            device.reason = "Could not read this device's compute capability."
        elif _runs_on(capability, built):
            device.usable = True
            if capability < _FP16_FROM:
                device.remedy = (
                    f"float32 only: sm_{capability[0]}{capability[1]} runs FP16 "
                    f"at a fraction of FP32 rate, so half precision would make "
                    f"this slower, not faster."
                )
        else:
            names = ", ".join(f"sm_{a}{b}" for a, b in built) or "nothing"
            device.reason = (
                f"This torch ({torch.__version__}) has no kernels for "
                f"sm_{capability[0]}{capability[1]}. It was built for {names}. "
                f"torch.cuda.is_available() is True and the first kernel launch "
                f"would fail with 'no kernel image is available for execution "
                f"on the device'."
            )
            device.remedy = wheel_hint(capability)
        found.append(device)
    return found


def _mps_device() -> Device:
    try:
        import torch

        backend = getattr(torch.backends, "mps", None)
        if backend is None:
            return Device("mps", "mps", "Apple Metal",
                          reason="This torch has no MPS backend.")
        if not backend.is_available():
            reason = ("Not an Apple Silicon Mac, or this torch was not built "
                      "with MPS.")
            if hasattr(backend, "is_built") and not backend.is_built():
                reason = "This torch was not built with MPS."
            return Device("mps", "mps", "Apple Metal", reason=reason)
        return Device("mps", "mps", "Apple Metal", usable=True)
    except ImportError as exc:
        return Device("mps", "mps", "Apple Metal",
                      reason=_torch_unavailable_reason(exc), remedy='Run `hypernix devices` for the full diagnosis, or install the CPU build: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu')


def _xpu_device() -> Device:
    try:
        import torch

        backend = getattr(torch, "xpu", None)
        if backend is None or not backend.is_available():
            return Device(
                "xpu", "xpu", "Intel GPU (oneAPI)",
                reason="No Intel GPU backend in this torch, or no device.",
                remedy="Intel Arc and Data Center GPUs need a torch built with "
                       "XPU support (intel-extension-for-pytorch, or torch 2.5+ "
                       "from the Intel index).",
            )
        return Device("xpu", "xpu", "Intel GPU (oneAPI)", usable=True)
    except ImportError as exc:
        return Device("xpu", "xpu", "Intel GPU",
                      reason=_torch_unavailable_reason(exc), remedy='Run `hypernix devices` for the full diagnosis, or install the CPU build: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu')


def _vulkan_device() -> Device:
    """Always unusable *through torch*, with the route that does work.

    Reporting Vulkan as available because an import succeeded would be a
    lie with a long debugging tail. See the module docstring.
    """
    return Device(
        "vulkan", "vulkan", "Vulkan (via llama.cpp, not torch)",
        reason="PyTorch has no usable Vulkan backend for transformer inference.",
        remedy=VULKAN_ADVICE,
    )


def probe() -> list[Device]:
    """Every candidate backend, usable or not, with the reason.

    Unusable entries are kept deliberately. "CUDA is not in the list" and
    "CUDA is there but this wheel has no kernels for your card" are
    different problems with different fixes, and a probe that only
    returned what works could not tell them apart.
    """
    found: list[Device] = []
    found.extend(_cuda_devices())
    found.append(_mps_device())
    found.append(_xpu_device())
    found.append(_vulkan_device())
    found.append(_cpu_device())
    return found


def _one(name: str) -> Device:
    for device in probe():
        if device.name == name or device.kind == name:
            return device
    return Device(name, name.split(":")[0], reason=f"Unknown device {name!r}.")


#: Preference order for ``--device auto``. CUDA first because it is the
#: only one that changes the answer by an order of magnitude; CPU last
#: because it always works.
_AUTO_ORDER = ("cuda", "xpu", "mps", "cpu")


def select(preference: str = "auto") -> Device:
    """Resolve a ``--device`` value to something that will actually run.

    ``auto`` takes the first usable backend in preference order and falls
    back to CPU, which cannot fail. A named device that is present but
    unusable raises with the reason *and the remedy* rather than being
    silently downgraded — someone who typed ``--device cuda`` wants to
    know why they did not get it.
    """
    wanted = (preference or "auto").strip().lower()
    devices = probe()

    if wanted in ("", "auto"):
        for kind in _AUTO_ORDER:
            for device in devices:
                if device.kind == kind and device.usable:
                    return device
        return _cpu_device()

    if wanted == "vulkan":
        raise DeviceError(VULKAN_ADVICE)

    matches = [d for d in devices if d.name == wanted or d.kind == wanted]
    if not matches:
        names = ", ".join(sorted({d.kind for d in devices}))
        raise DeviceError(f"Unknown device {preference!r}. Try one of: {names}, auto.")

    usable = next((d for d in matches if d.usable), None)
    if usable is not None:
        return usable

    first = matches[0]
    message = f"{first.name} cannot run this: {first.reason}"
    if first.remedy:
        message += f"\n\n{first.remedy}"
    raise DeviceError(message)


def describe(devices: list[Device] | None = None) -> str:
    """The probe as a block of text, for ``hypernix devices``."""
    devices = devices if devices is not None else probe()
    lines = ["Devices this runtime can use:"]
    lines += [device.describe() for device in devices]
    usable = [d for d in devices if d.usable]
    lines.append("")
    if usable:
        chosen = select("auto")
        lines.append(f"  --device auto would pick: {chosen.name} ({chosen.label})")
    else:
        lines.append("  Nothing usable but CPU.")

    # What the vendor tools see, which is a different question from what
    # torch can use. A card that is physically present and unusable here
    # -- no ROCm build of torch, a wheel without kernels for it -- shows
    # up in one list and not the other, and seeing both together is what
    # tells you which of those you are looking at.
    from ..system import gpus as _gpus

    hardware = _gpus.detect()
    if hardware:
        lines.append("")
        lines.append("Hardware the vendor tools report:")
        lines.append(_gpus.describe(hardware))
        torch_kinds = {d.kind for d in devices if d.usable}
        unusable = [
            card for card in hardware
            if card.framework not in torch_kinds and card.framework != "cpu"
        ]
        if unusable:
            names = ", ".join(f"{c.vendor.value}:{c.index}" for c in unusable)
            lines.append("")
            lines.append(
                f"  {names} is present but this torch cannot use it. "
                f"That is a torch build question, not a driver one."
            )
    return "\n".join(lines)
