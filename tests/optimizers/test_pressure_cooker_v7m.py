"""V7M: the V5/V5+ lineage. QAT and MTP are explicit, and off unless called."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from _v7_helpers import flat, make_data, make_model, train

from hypernix.optimizers import v7m_features
from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2
from hypernix.optimizers.pressure_cooker_v5 import MTPConfig, QATConfig
from hypernix.optimizers.pressure_cooker_v7 import PressureCookerV7M


def test_defaults_carry_no_training_features():
    model = make_model()
    before = [type(m) for m in model.modules()]
    optimizer = PressureCookerV7M(model.parameters(), lr=3e-3)
    losses = train(model, optimizer, steps=40)
    assert losses[-1] < losses[0]
    described = optimizer.describe()
    assert described["variant"] == "m" and described["lineage"] == "V5/V5+"
    assert described["state_dtype"] == "int8" and described["factored"] is True
    assert described["ema"] is False
    # Nothing was attached to the model: no parametrizations, no extra modules.
    assert [type(m) for m in model.modules()] == before
    assert not any(nn.utils.parametrize.is_parametrized(m) for m in model.modules())


def test_qat_is_attached_explicitly_and_removed_cleanly():
    model = make_model()
    x, _ = make_data()
    plain = model(x).detach().clone()
    handle = v7m_features.attach_qat(model, QATConfig())
    assert handle.active
    quantized = model(x).detach()
    assert not torch.allclose(plain, quantized)
    optimizer = PressureCookerV7M(model.parameters(), lr=3e-3)
    losses = train(model, optimizer, steps=20)      # gradients reach the real weights
    assert losses[-1] < losses[0]
    handle.remove()
    assert not handle.active
    assert not any(nn.utils.parametrize.is_parametrized(m) for m in model.modules())
    assert torch.isfinite(flat(model)).all()


def test_sensitivity_and_the_mixed_precision_plan():
    model = make_model()
    x, y = make_data()

    def loss() -> torch.Tensor:
        return nn.functional.cross_entropy(model(x), y)

    weights_before = flat(model).clone()
    sensitivity = v7m_features.quantization_sensitivity(model, loss, bits=(4, 8))
    assert torch.equal(flat(model), weights_before)        # the model is put back
    assert sensitivity and all(set(v) == {4, 8} for v in sensitivity.values())
    plan = v7m_features.mixed_precision_plan(sensitivity)
    assert set(plan) == set(sensitivity) and set(plan.values()) <= {4, 8}


def test_mtp_head_and_loss_are_explicit():
    head = v7m_features.make_mtp_head(16, 32, MTPConfig())
    hidden = torch.randn(2, 12, 16)
    targets = torch.randint(0, 32, (2, 12))
    loss = v7m_features.mtp_loss(head, hidden, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())


def test_ema_weights_swap_in_and_back():
    model = make_model()
    optimizer = PressureCookerV7M(model.parameters(), lr=3e-3, ema_decay=0.9)
    train(model, optimizer, steps=15)
    live = flat(model).clone()
    with v7m_features.ema_weights(optimizer):
        assert not torch.equal(flat(model), live)
    assert torch.equal(flat(model), live)
    without = PressureCookerV7M(make_model().parameters())
    with pytest.raises(RuntimeError, match="no EMA shadow"), v7m_features.ema_weights(without):
        pass


def test_quantization_objective_selects_it():
    chosen = OptimizerFrameworkV2.auto_select(objective="quantization")
    assert chosen.key == "v7m" and chosen.reason_code == "v5_lineage_features"
