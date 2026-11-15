"""V7Q: throughput first, and plain about what that costs."""
from __future__ import annotations

import torch
import torch.nn as nn
from _v7_helpers import flat, make_data, make_model, train

from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2
from hypernix.optimizers.pressure_cooker_v7 import PressureCookerV7Q


def test_defaults_are_the_cheapest_path():
    model = make_model()
    optimizer = PressureCookerV7Q(model.parameters(), lr=1e-2)
    losses = train(model, optimizer, steps=40)
    assert losses[-1] < losses[0]
    described = optimizer.describe()
    assert described["variant"] == "q" and described["state_dtype"] == "fp32"
    assert described["stability"] is False and described["adaptive_pressure"] is False
    assert described["factored"] is False and described["diffusion"] is False
    assert described["grad_clip"] is None and described["skip_on_nonfinite"] is False
    assert described["host_reads_per_step"] == 0 and described["stat_interval"] == 16
    assert described["lazy_weight_decay"] is True


def test_state_is_one_buffer_per_parameter():
    model = make_model()
    optimizer = PressureCookerV7Q(model.parameters())
    train(model, optimizer, steps=2)
    n = sum(p.numel() for p in model.parameters())
    assert all(set(s) == {"momentum"} for s in optimizer.state.values())
    assert 4 * n <= optimizer.state_bytes() < 4.2 * n


def test_safety_can_be_turned_back_on():
    model = make_model()
    optimizer = PressureCookerV7Q(model.parameters(), lr=1e-2, skip_on_nonfinite=True,
                                  grad_clip=1.0)
    train(model, optimizer, steps=3)
    before = flat(model).clone()
    x, y = make_data()
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    next(model.parameters()).grad.fill_(float("inf"))
    optimizer.step()
    assert torch.equal(flat(model), before)
    assert optimizer.describe()["host_reads_per_step"] == 1


def test_throughput_selection_needs_a_measurement_to_claim_one():
    unmeasured = OptimizerFrameworkV2.auto_select(objective="throughput")
    assert unmeasured.key == "v7q" and unmeasured.reason_code == "throughput_unmeasured"
    faster = OptimizerFrameworkV2.auto_select(
        objective="throughput", measured={"v7q": {"ms_per_step": 1.0},
                                          "v6": {"ms_per_step": 2.0}})
    assert faster.key == "v7q" and faster.reason_code == "throughput_measured"
    slower = OptimizerFrameworkV2.auto_select(
        objective="throughput", measured={"v7q": {"ms_per_step": 3.0},
                                          "v6": {"ms_per_step": 2.0}})
    assert slower.key == "v6"
    assert any(c["key"] == "v7q" for c in slower.considered)
