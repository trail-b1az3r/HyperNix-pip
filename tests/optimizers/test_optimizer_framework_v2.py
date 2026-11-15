"""OptimizerFrameworkV2: one interface over every optimizer, and honest selection."""
from __future__ import annotations

import json

import pytest
import torch
from _v7_helpers import make_model, train

from hypernix.optimizers.optimizer_framework_v2 import (
    FEATURES,
    OBJECTIVES,
    OPTIMIZERS,
    OptimizerFrameworkV2,
)

CPU_KEYS = [k for k, spec in OPTIMIZERS.items() if not spec.requires_cuda]


def test_registry_covers_adamw_and_every_generation():
    assert {"adamw", "v1", "v2", "v3", "v4", "v5", "v5plus", "v5s", "v6", "v6v", "v7", "v7v",
            "v7mini", "v7q", "v7s", "v7m"} <= set(OPTIMIZERS)
    assert OptimizerFrameworkV2.spec("V7 Balanced").key == "v7"
    assert OptimizerFrameworkV2.spec("v5+").key == "v5plus"
    with pytest.raises(ValueError, match="unknown optimizer"):
        OptimizerFrameworkV2.spec("v99")


@pytest.mark.parametrize("key", CPU_KEYS)
def test_every_optimizer_is_created_and_steps(key):
    model = make_model()
    optimizer = OptimizerFrameworkV2.create(key, model.parameters(), lr=1e-3)
    losses = train(model, optimizer, steps=5)
    assert all(torch.isfinite(torch.tensor(losses)))
    info = OptimizerFrameworkV2.diagnostics(optimizer)
    json.dumps(info)
    assert info["state_bytes"] > 0


def test_existing_optimizers_keep_their_own_classes():
    from hypernix.optimizers.pressure_cooker_v5 import PressureCookerV5
    from hypernix.optimizers.pressure_cooker_v6 import PressureCookerV6

    model = make_model()
    assert type(OptimizerFrameworkV2.create("v5", model.parameters())) is PressureCookerV5
    assert type(OptimizerFrameworkV2.create("v6", model.parameters())) is PressureCookerV6
    assert type(OptimizerFrameworkV2.create("adamw", model.parameters())) is torch.optim.AdamW


def test_an_optimizer_that_needs_cuda_is_refused_not_replaced():
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises((ValueError, RuntimeError), match="CUDA"):
        OptimizerFrameworkV2.create("v6v", make_model().parameters())
    with pytest.raises(ValueError, match="nothing was substituted"):
        OptimizerFrameworkV2.auto_select(requested="v6v", device="cpu")


def test_default_selection_is_v7_balanced_with_a_reason():
    chosen = OptimizerFrameworkV2.auto_select()
    assert chosen.key == "v7" and chosen.reason_code == "balanced" and chosen.reason
    assert chosen.explicit is False
    json.dumps(chosen.to_dict())


@pytest.mark.parametrize("objective,key", [
    ("balanced", "v7"), ("throughput", "v7q"), ("memory", "v7"), ("stability", "v7s"),
    ("quantization", "v7m"), ("compatibility", "adamw")])
def test_objectives(objective, key):
    assert objective in OBJECTIVES
    chosen = OptimizerFrameworkV2.auto_select(objective=objective)
    assert chosen.key == key and chosen.objective == objective
    if objective == "memory":
        assert chosen.options.get("preset") == "memory"


def test_unknown_objective_and_feature_are_refused():
    with pytest.raises(ValueError, match="objective"):
        OptimizerFrameworkV2.auto_select(objective="vibes")
    with pytest.raises(ValueError, match="feature"):
        OptimizerFrameworkV2.auto_select(required_features=["telepathy"])


def test_explicit_choice_is_honoured_or_refused_never_swapped():
    chosen = OptimizerFrameworkV2.auto_select(requested="v5s", objective="throughput")
    assert chosen.key == "v5s" and chosen.explicit is True
    feature = next(f for f in FEATURES
                   if f not in OPTIMIZERS["adamw"].features)
    with pytest.raises(ValueError, match="nothing was substituted"):
        OptimizerFrameworkV2.auto_select(requested="adamw", required_features=[feature])


def test_required_features_steer_the_choice():
    for feature in FEATURES:
        try:
            chosen = OptimizerFrameworkV2.auto_select(required_features=[feature])
        except ValueError:
            continue                       # no CPU optimizer has it: refused, with a reason
        assert feature in OPTIMIZERS[chosen.key].features


def test_selection_counts_parameters_itself_when_given_them():
    model = make_model()
    chosen = OptimizerFrameworkV2.auto_select(model.parameters())
    assert chosen.parameter_count == sum(p.numel() for p in model.parameters())
    assert chosen.parameter_count_verified is True and chosen.key == "v7"


def test_selection_is_deterministic():
    kwargs = dict(parameter_count=200_000_000, parameter_count_verified=True,
                  objective="balanced")
    assert OptimizerFrameworkV2.auto_select(**kwargs).to_dict() == \
        OptimizerFrameworkV2.auto_select(**kwargs).to_dict()


def test_build_creates_what_was_selected():
    model = make_model()
    chosen = OptimizerFrameworkV2.auto_select(model.parameters(), objective="memory")
    optimizer = OptimizerFrameworkV2.build(chosen, model.parameters(), lr=1e-3)
    assert chosen.key == "v7" and optimizer.describe()["preset"] == "memory"
    assert OptimizerFrameworkV2.diagnostics(optimizer)["selection"]["reason_code"] == "memory"


def test_table_marks_ratings_as_targets_until_measured():
    rows = OptimizerFrameworkV2.table()
    assert {r["key"] for r in rows} == set(OPTIMIZERS)
    assert all(r["ratings_are"] == "targets" and r["measured"] == {} for r in rows)
    measured = OptimizerFrameworkV2.table({"v7": {"ms_per_step": 1.5}})
    assert next(r for r in measured if r["key"] == "v7")["measured"] == {"ms_per_step": 1.5}
