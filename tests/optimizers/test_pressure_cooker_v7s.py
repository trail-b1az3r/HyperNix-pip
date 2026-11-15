"""V7S: the V5S lineage on the V7 engine."""
from __future__ import annotations

import torch
from _v7_helpers import make_model, train

from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2
from hypernix.optimizers.pressure_cooker_v5s import PressureCookerV5S
from hypernix.optimizers.pressure_cooker_v7 import PressureCookerV7S


def test_keeps_the_v5s_ingredients():
    model = make_model()
    optimizer = PressureCookerV7S(model.parameters(), lr=3e-3)
    losses = train(model, optimizer, steps=40)
    assert losses[-1] < losses[0]
    described = optimizer.describe()
    assert described["variant"] == "s" and described["lineage"] == "V5S"
    assert described["state_dtype"] == "int8" and described["factored"] is True
    assert described["diffusion"] is True and described["stability"] is True
    assert described["stat_interval"] == 2
    assert optimizer.stability_weights == (0.45, 0.35, 0.20)
    assert optimizer.stability_betas == (0.8, 0.95, 0.999)
    weight = next(model.parameters())
    assert optimizer.state[weight]["momentum"].dtype == torch.int8
    assert "row" in optimizer.state[weight] and "scale" in optimizer.state[weight]


def test_state_is_about_one_byte_per_parameter():
    model = make_model()
    optimizer = PressureCookerV7S(model.parameters())
    train(model, optimizer, steps=2)
    n = sum(p.numel() for p in model.parameters())
    assert optimizer.state_bytes() < 1.5 * n


def test_a_v5s_checkpoint_is_refused():
    import pytest

    model = make_model()
    v5s = PressureCookerV5S(model.parameters())
    train(model, v5s, steps=2, report=False)
    with pytest.raises(ValueError, match="not a Pressure Cooker V7 checkpoint"):
        PressureCookerV7S(model.parameters()).load_state_dict(v5s.state_dict())


def test_stability_objective_selects_it():
    chosen = OptimizerFrameworkV2.auto_select(objective="stability")
    assert chosen.key == "v7s"
