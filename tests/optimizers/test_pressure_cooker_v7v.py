"""Pressure Cooker V7V: the accelerated kernels are optional, the fallback is not."""
from __future__ import annotations

import pytest
import torch
from _v7_helpers import flat, make_model, train

from hypernix.optimizers.pressure_cooker_v7 import PressureCookerV7
from hypernix.optimizers.pressure_cooker_v7v import V7V, PressureCookerV7V


def test_cpu_falls_back_to_foreach_and_says_why():
    model = make_model()
    optimizer = PressureCookerV7V(model.parameters(), lr=3e-3)
    losses = train(model, optimizer, steps=20)
    assert losses[-1] < losses[0]
    described = optimizer.describe()
    assert described["kernel"] == "foreach"
    assert described["compiled"] is False and described["cuda_graph_captured"] is False
    reasons = " ".join(described["fallback_reasons"])
    assert "torch.compile" in reasons and "CUDA graph" in reasons
    assert V7V is PressureCookerV7V


def test_fallback_matches_v7_exactly():
    ends = []
    for cls in (PressureCookerV7, PressureCookerV7V):
        model = make_model()
        train(model, cls(model.parameters(), lr=3e-3), steps=20)
        ends.append(flat(model))
    assert torch.equal(ends[0], ends[1])


def test_compile_off_is_recorded():
    model = make_model()
    optimizer = PressureCookerV7V(model.parameters(), compile=False)
    assert any("turned off" in r for r in optimizer.fallback_reasons)


def test_cuda_graph_is_refused_on_cpu_with_a_reason():
    model = make_model()
    optimizer = PressureCookerV7V(model.parameters())
    with pytest.raises(RuntimeError, match="CUDA graphs need CUDA"):
        optimizer.warmup_graph(lambda: None)
    with pytest.raises(RuntimeError, match="warmup_graph"):
        optimizer.replay_graph()


def test_a_failing_compiled_step_falls_back_and_keeps_training():
    model = make_model()
    optimizer = PressureCookerV7V(model.parameters(), lr=3e-3, compile=False)

    def broken(*args, **kwargs):
        raise RuntimeError("no compiler here")

    optimizer._compiled = broken
    with pytest.warns(UserWarning, match="compiled V7V step failed"):
        losses = train(model, optimizer, steps=10)
    assert losses[-1] < losses[0]
    assert optimizer.describe()["kernel"] == "foreach"
    assert any("failed at run time" in r for r in optimizer.fallback_reasons)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_refuses_per_step_decisions():
    model = make_model().cuda()
    optimizer = PressureCookerV7V(model.parameters(), compile=False)
    with pytest.raises(RuntimeError, match="freeze decisions"):
        optimizer.warmup_graph(lambda: None)
