"""neuron.nets — small networks, built from a description.

Everything this package trains is small on purpose. A policy that steers
a robot arm or reacts to a game frame has to answer inside a control
loop, and a model that takes 40ms is not a policy, it is a pause. These
are thousands to low-millions of parameters: they train on a CPU and
they run in well under a millisecond.

Two builders, because the inputs come in two shapes
---------------------------------------------------
:func:`mlp` for a vector of numbers — joint angles, positions,
velocities, a handful of sensor readings. :func:`small_cnn` for an
image, which is the "basic and quick image analysis" case: a few
convolutions with stride, global pooling, done. Not a ResNet. If you
need a ResNet you should fine-tune a pretrained one, and this package is
not what you want.

Why a spec rather than hand-written modules
-------------------------------------------
Every trainer here has to be able to say "give me a network for this
environment" without the caller restating shapes that the environment
already knows. :func:`for_env` does that, and it is also what makes a
saved policy reloadable: :func:`save` writes the spec next to the
weights, so :func:`load` rebuilds the architecture instead of requiring
the caller to remember it. A checkpoint you cannot load without the
script that made it is a checkpoint you will lose.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .envs import Env, Space

__all__ = [
    "NetSpec",
    "mlp",
    "small_cnn",
    "build",
    "for_env",
    "save",
    "load",
    "greedy",
    "stochastic",
    "count_parameters",
]


@dataclass
class NetSpec:
    """Everything needed to rebuild a network. Saved beside the weights."""

    kind: str = "mlp"                      # "mlp" | "cnn"
    inputs: int = 0                        # mlp: input features
    outputs: int = 0                       # action count / class count
    hidden: tuple[int, ...] = (64, 64)
    #: cnn only — (channels, height, width)
    image: tuple[int, int, int] = (0, 0, 0)
    channels: tuple[int, ...] = (16, 32, 64)
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetSpec:
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        for key in ("hidden", "channels", "image"):
            if key in clean and clean[key] is not None:
                clean[key] = tuple(clean[key])
        return cls(**clean)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"unknown activation {name!r}; try one of "
            f"{', '.join(sorted(_ACTIVATIONS))}"
        ) from None


def mlp(
    inputs: int,
    outputs: int,
    hidden: tuple[int, ...] = (64, 64),
    activation: str = "relu",
) -> nn.Sequential:
    """A plain stack. The right answer for anything that is not an image."""
    if inputs <= 0 or outputs <= 0:
        raise ValueError("an MLP needs a positive input and output size")
    layers: list[nn.Module] = []
    width = inputs
    for size in hidden:
        layers.append(nn.Linear(width, size))
        layers.append(_activation(activation))
        width = size
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x.flatten(1)


def small_cnn(
    image: tuple[int, int, int],
    outputs: int,
    channels: tuple[int, ...] = (16, 32, 64),
    activation: str = "relu",
) -> nn.Sequential:
    """Strided convolutions into global average pooling.

    Global pooling rather than a flatten into a big linear layer, so the
    parameter count does not explode with input size and the same spec
    survives a change of camera resolution. That matters here: the
    robotics and game cases both tend to change frame size once, late,
    after somebody has already collected data.
    """
    depth, height, width = image
    if depth <= 0 or height <= 0 or width <= 0:
        raise ValueError("an image needs positive (channels, height, width)")
    if outputs <= 0:
        raise ValueError("a CNN needs a positive output size")

    layers: list[nn.Module] = []
    previous = depth
    for size in channels:
        layers.append(nn.Conv2d(previous, size, kernel_size=3, stride=2, padding=1))
        layers.append(_activation(activation))
        previous = size
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(_Flatten())
    layers.append(nn.Linear(previous, outputs))
    return nn.Sequential(*layers)


def build(spec: NetSpec) -> nn.Module:
    """A network from its description."""
    if spec.kind == "mlp":
        return mlp(spec.inputs, spec.outputs, tuple(spec.hidden), spec.activation)
    if spec.kind == "cnn":
        return small_cnn(
            tuple(spec.image), spec.outputs, tuple(spec.channels), spec.activation
        )
    raise ValueError(f"unknown network kind {spec.kind!r}; try 'mlp' or 'cnn'")


def for_env(
    env: Env, hidden: tuple[int, ...] = (64, 64)
) -> tuple[nn.Module, NetSpec]:
    """A network shaped for *env*, and the spec that rebuilds it.

    Reads both spaces off the environment so no caller has to restate
    what the environment already knows — a duplicated shape is a shape
    that goes stale the first time the observation changes.
    """
    observation: Space = env.observation_space
    action: Space = env.action_space
    if not action.discrete:
        raise ValueError(
            "for_env builds discrete-action policies; a continuous action "
            "space needs a head that outputs a distribution, which this "
            "package does not do yet"
        )
    if observation.discrete:
        raise ValueError(
            "a discrete observation space needs an embedding rather than an "
            "MLP; wrap it in a one-hot observation first"
        )
    if len(observation.shape) == 3:
        spec = NetSpec(
            kind="cnn", image=tuple(observation.shape), outputs=action.n
        )
    else:
        spec = NetSpec(
            kind="mlp", inputs=observation.size, outputs=action.n, hidden=hidden
        )
    return build(spec), spec


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def save(model: nn.Module, spec: NetSpec, path: str | Path) -> Path:
    """Weights and architecture together, in one file.

    Together on purpose. `torch.save(model.state_dict())` alone produces
    a file that cannot be loaded without the code that built it, and
    that code is usually a notebook cell that is already gone.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"spec": spec.to_dict(), "state": model.state_dict(), "format": 1},
        target,
    )
    return target


def load(path: str | Path) -> tuple[nn.Module, NetSpec]:
    """Rebuild what :func:`save` wrote."""
    target = Path(path)
    # weights_only=True: a checkpoint is data, and torch.load defaults to
    # unpickling arbitrary objects. A policy file downloaded from
    # anywhere is exactly the case where that matters.
    blob = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "spec" not in blob or "state" not in blob:
        raise ValueError(
            f"{target} is not a neuron checkpoint — it has no spec beside "
            f"its weights. A bare state_dict cannot be rebuilt without the "
            f"code that made it; re-save it with neuron.nets.save."
        )
    spec = NetSpec.from_dict(blob["spec"])
    model = build(spec)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, spec


def save_json(spec: NetSpec, path: str | Path) -> Path:
    """The spec alone, readable. For inspecting what a run produced."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Turning a network into a policy
# ---------------------------------------------------------------------------


def _as_batch(observation) -> torch.Tensor:
    array = np.asarray(observation, dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def greedy(model: nn.Module):
    """Always the highest-scoring action. What you evaluate and ship.

    Separate from :func:`stochastic` rather than a flag, because using
    the wrong one is a silent, plausible-looking failure: a sampled
    policy evaluates worse than it is, and a greedy policy explores
    nothing and learns nothing.
    """

    def policy(observation) -> int:
        model.eval()
        with torch.no_grad():
            return int(model(_as_batch(observation)).argmax(dim=-1).item())

    return policy


def stochastic(model: nn.Module, *, generator: torch.Generator | None = None):
    """Sample from the softmax. What you explore with while training."""

    def policy(observation) -> int:
        model.eval()
        with torch.no_grad():
            logits = model(_as_batch(observation))
            probabilities = torch.softmax(logits, dim=-1)
            return int(
                torch.multinomial(probabilities[0], 1, generator=generator).item()
            )

    return policy


def count_parameters(model: nn.Module) -> int:
    """How big this actually is. Printed by every trainer for a reason."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
