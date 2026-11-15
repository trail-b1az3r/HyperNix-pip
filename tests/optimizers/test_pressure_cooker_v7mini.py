"""V7Mini: its band is strict, and is never assumed from a guess."""
from __future__ import annotations

import pytest
from _v7_helpers import make_model, train

from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2
from hypernix.optimizers.pressure_cooker_v7 import (
    MINI_MAX_PARAMS,
    MINI_MIN_PARAMS,
    PressureCookerV7Mini,
)


def test_band_constants():
    assert MINI_MIN_PARAMS == 39_000_000 and MINI_MAX_PARAMS == 750_000_000


def test_trains_outside_the_band_and_says_so():
    model = make_model()
    optimizer = PressureCookerV7Mini(model.parameters(), lr=3e-3)
    losses = train(model, optimizer, steps=30)
    assert losses[-1] < losses[0]
    described = optimizer.describe()
    assert described["variant"] == "mini" and described["in_band"] is False
    assert "not established" in described["band_note"]
    assert described["stat_interval"] == 8 and described["diffusion"] is False
    assert described["lazy_weight_decay"] is True


@pytest.mark.parametrize("count,inside", [
    (MINI_MIN_PARAMS - 1, False), (MINI_MIN_PARAMS, False), (MINI_MIN_PARAMS + 1, True),
    (100_000_000, True), (MINI_MAX_PARAMS - 1, True), (MINI_MAX_PARAMS, False),
    (MINI_MAX_PARAMS + 1, False)])
def test_auto_selection_uses_strict_boundaries(count, inside):
    chosen = OptimizerFrameworkV2.auto_select(parameter_count=count,
                                              parameter_count_verified=True)
    assert (chosen.key == "v7mini") is inside
    if inside:
        assert chosen.reason_code == "mini_band"
    else:
        assert chosen.key == "v7"


@pytest.mark.parametrize("count", [None, 100_000_000])
def test_an_unverified_or_unknown_count_never_selects_mini(count):
    chosen = OptimizerFrameworkV2.auto_select(parameter_count=count,
                                              parameter_count_verified=False)
    assert chosen.key == "v7"
    assert chosen.parameter_count_verified is False


def test_an_explicit_request_is_honoured_outside_the_band():
    chosen = OptimizerFrameworkV2.auto_select(requested="v7mini", parameter_count=1_000,
                                              parameter_count_verified=True)
    assert chosen.key == "v7mini" and chosen.explicit is True
    assert any("not established" in n or "band" in n for n in chosen.notes)
