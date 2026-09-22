"""hypernix.neuron — train small networks that act, not just predict.

The rest of this project trains language models. `neuron` is for the
other kind of job people keep asking it for: something that plays a
game, tells one picture from another, or drives a robot arm — where the
model is small, the data is something you generate rather than download,
and "is it any good" means "does it work when you run it", not a loss
number.

Four ways to train, because these problems arrive in different shapes
--------------------------------------------------------------------
**Supervised** (:mod:`~hypernix.neuron.supervised`) when you have
labels. Image recognition usually is this, and it is the one everybody
already knows.

**Imitation** (:mod:`~hypernix.neuron.imitation`) when you can *do* the
task but cannot describe it. Play twenty rounds yourself, record what
you pressed, and clone it. For game automation this beats reinforcement
learning on the only axis that matters at the start: it works in
minutes instead of hours, from data you can actually produce.

**Reinforcement** (:mod:`~hypernix.neuron.rl`) when there is no right
answer to copy, only outcomes that are better or worse. Expensive and
fiddly, and the honest default is to reach for it *second* — after
behaviour cloning has shown the problem is learnable at all.

**Evaluation** (:mod:`~hypernix.neuron.evaluate`) is not an afterthought
here and does not live inside the trainers. A policy that scores well on
held-out demonstrations and then walks into a wall is the normal outcome
of imitation learning, not an unusual one, so measuring accuracy and
measuring *return from actually running it* are separate calls that
report separate numbers.

Why there is an environment protocol and two toy environments
-------------------------------------------------------------
:mod:`~hypernix.neuron.envs` defines the smallest interface a thing has
to have to be trained against — ``reset``, ``step``, and a description
of its spaces — and ships two environments implementing it. They are
there so every trainer in this package has something deterministic to be
tested against, on any machine, with no simulator installed and no
display. An RL implementation whose tests all mock the environment is an
RL implementation nobody has run.

The protocol is deliberately the Gymnasium shape (``reset() ->
(obs, info)``, ``step() -> (obs, reward, terminated, truncated, info)``)
so a real Gymnasium environment, a robot driver, or a screen-capture
game wrapper drops in without an adapter.

Sizes
-----
Everything here builds small networks on purpose: an MLP or a compact
CNN, thousands to low millions of parameters. That is what these tasks
need, it trains on a CPU, and it runs inside a control loop fast enough
to matter. If you want a large vision model, fine-tune one — this is for
the part where you need an answer in four milliseconds.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "data",
    "envs",
    "evaluate",
    "imitation",
    "nets",
    "rl",
    "supervised",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(__all__)
