"""Pressure Cooker V7 (Balanced): the update, its state and its checkpoints."""
from __future__ import annotations

import copy
import math
import warnings

import pytest
import torch
import torch.nn as nn
from _v7_helpers import flat, make_data, make_model, train

from hypernix.optimizers.pressure_cooker_v5 import PressureCookerV5
from hypernix.optimizers.pressure_cooker_v7 import (
    STATE_DTYPES,
    V7_PRESETS,
    PressureCookerV7,
    resolve_state_dtype,
)


def test_trains_and_is_deterministic():
    runs = []
    for _ in range(2):
        model = make_model()
        losses = train(model, PressureCookerV7(model.parameters(), lr=1e-2), steps=100)
        runs.append((losses, flat(model)))
    assert runs[0][0][-1] < 0.7 * runs[0][0][0]
    assert runs[0][0] == runs[1][0]
    assert torch.equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("preset", sorted(V7_PRESETS))
def test_every_preset_trains(preset):
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), preset=preset, lr=3e-3)
    losses = train(model, optimizer)
    assert math.isfinite(losses[-1]) and losses[-1] < losses[0]
    assert optimizer.describe()["preset"] == preset


def test_unknown_preset_and_bad_arguments_are_refused():
    model = make_model()
    with pytest.raises(ValueError, match="preset"):
        PressureCookerV7(model.parameters(), preset="turbo")
    with pytest.raises(ValueError, match="state_dtype"):
        PressureCookerV7(model.parameters(), state_dtype="fp8")
    with pytest.raises(ValueError, match="grad_accum_steps"):
        PressureCookerV7(model.parameters(), grad_accum_steps=0)
    with pytest.raises(ValueError, match="diffusion_factor"):
        PressureCookerV7(model.parameters(), diffusion_factor=1.5)


@pytest.mark.parametrize("dtype", STATE_DTYPES)
def test_each_state_dtype_trains_and_is_stored_as_asked(dtype):
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), state_dtype=dtype, lr=3e-3)
    losses = train(model, optimizer)
    assert losses[-1] < losses[0]
    want = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
            "int8": torch.int8}[dtype]
    weight = next(model.parameters())
    assert optimizer.state[weight]["momentum"].dtype == want
    assert optimizer.describe()["state_dtype"] == dtype


def test_compressed_state_is_smaller_and_counted_from_tensors():
    sizes = {}
    for dtype in STATE_DTYPES:
        model = make_model()
        optimizer = PressureCookerV7(model.parameters(), state_dtype=dtype, factored=False)
        train(model, optimizer, steps=2)
        sizes[dtype] = optimizer.state_bytes()
        counted = sum(v.numel() * v.element_size() for s in optimizer.state.values()
                      for v in s.values() if isinstance(v, torch.Tensor))
        assert sizes[dtype] >= counted
    assert sizes["int8"] < sizes["bf16"] < sizes["fp32"]
    assert sizes["fp16"] == sizes["bf16"]


def test_auto_state_dtype_states_its_reason():
    name, why = resolve_state_dtype(10_000_000, "cpu")
    assert name == "fp32" and why
    name, why = resolve_state_dtype(10_000_000, "cpu", free_bytes=1_000_000)
    assert name != "fp32" and "free" in why
    model = make_model()
    described = PressureCookerV7(model.parameters()).describe()
    assert described["state_dtype_requested"] == "auto"
    assert described["state_dtype"] in STATE_DTYPES and described["state_dtype_reason"]


def test_foreach_and_per_tensor_paths_agree():
    results = []
    for foreach in (True, False):
        model = make_model()
        optimizer = PressureCookerV7(model.parameters(), lr=3e-3, foreach=foreach)
        train(model, optimizer, steps=25)
        results.append(flat(model))
        assert optimizer.describe()["kernel"] == ("foreach" if foreach else "per-tensor")
    # The two paths do the same arithmetic in a different order.
    assert torch.allclose(results[0], results[1], rtol=1e-4, atol=1e-5)


def test_factored_state_exists_only_for_matrices():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), factored=True)
    train(model, optimizer, steps=3)
    for p in model.parameters():
        state = optimizer.state[p]
        assert ("row" in state) == (p.dim() >= 2 and min(p.shape) >= 16)
        if "row" in state:
            assert state["row"].numel() == p.shape[0] and state["col"].numel() == p.shape[1]
    plain = make_model()
    off = PressureCookerV7(plain.parameters(), factored=False)
    train(plain, off, steps=3)
    assert not any("row" in s for s in off.state.values())


def test_diffusion_changes_the_update_only_when_on():
    ends = {}
    for name, kwargs in {"off": dict(diffusion=False),
                         "on": dict(diffusion=True, diffusion_factor=0.5),
                         "zero": dict(diffusion=True, diffusion_factor=0.0)}.items():
        model = make_model()
        train(model, PressureCookerV7(model.parameters(), lr=3e-3, factored=False, **kwargs),
              steps=20)
        ends[name] = flat(model)
    assert torch.equal(ends["off"], ends["zero"])
    assert torch.isfinite(ends["on"]).all()


def test_trust_ratio_stays_inside_its_clip():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3, factored=False,
                                 trust_ratio=True, trust_clip=(0.05, 5.0), stat_interval=1)
    train(model, optimizer, steps=20)
    for bucket in optimizer._buckets:
        assert float(bucket.trust.min()) >= 0.05 - 1e-6
        assert float(bucket.trust.max()) <= 5.0 + 1e-6


def test_adaptive_pressure_stays_inside_its_clip_and_follows_loss():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3, pressure_clip=(0.25, 2.0),
                                 stat_interval=1)
    train(model, optimizer, steps=20)
    for bucket in optimizer._buckets:
        assert 0.25 - 1e-6 <= float(bucket.pressure.min())
        assert float(bucket.pressure.max()) <= 2.0 + 1e-6
    before = optimizer.describe()["loss_factor"]
    for value in (5.0, 50.0, 500.0):          # the loss blowing up
        optimizer.report_loss(value)
    assert optimizer.describe()["loss_factor"] < before
    optimizer.report_loss(float("nan"))        # ignored, not propagated
    assert math.isfinite(optimizer.describe()["loss_factor"])


def test_non_finite_gradients_skip_the_step_and_leave_state_alone():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3)
    train(model, optimizer, steps=5)
    before = flat(model).clone()
    momentum = [optimizer.state[p]["momentum"].clone() for p in model.parameters()]
    x, y = make_data()
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    next(model.parameters()).grad[0, 0] = float("nan")
    optimizer.step()
    assert torch.equal(flat(model), before)
    for p, m in zip(model.parameters(), momentum, strict=True):
        assert torch.equal(optimizer.state[p]["momentum"], m)
    assert optimizer.describe()["skipped_steps"] == 1
    assert train(model, optimizer, steps=5)[-1] < 10       # and it carries on


def test_gradient_accumulation_steps_once_per_window():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3, grad_accum_steps=4)
    start = flat(model).clone()
    x, y = make_data()
    for i in range(4):
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
        if i < 3:
            assert torch.equal(flat(model), start)
    assert not torch.equal(flat(model), start)
    assert optimizer.describe()["grad_accum_steps"] == 4


def test_gradient_clipping_bounds_the_step():
    moved = {}
    for clip in (1e-3, None):
        model = make_model()
        optimizer = PressureCookerV7(model.parameters(), lr=1e-2, grad_clip=clip,
                                     factored=False, trust_ratio=False, stability=False,
                                     adaptive_pressure=False, weight_decay=0.0)
        start = flat(model).clone()
        train(model, optimizer, steps=1, report=False)
        moved[clip] = float((flat(model) - start).norm())
    assert moved[1e-3] < 0.05 * moved[None]


def test_grad_scaler_is_driven_and_mixed_dtypes_step():
    model = make_model()
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3, grad_scaler=scaler)
    assert optimizer.describe()["grad_scaler"] is True
    assert train(model, optimizer, steps=10)[-1] < 10
    # Parameters of two dtypes land in separate buckets and both move.
    a = nn.Parameter(torch.randn(24, 24))
    b = nn.Parameter(torch.randn(24, 24, dtype=torch.float64))
    mixed = PressureCookerV7([a, b], lr=1e-2)
    a0, b0 = a.detach().clone(), b.detach().clone()
    for _ in range(3):
        mixed.zero_grad()
        (a.square().sum() + b.square().sum().float()).backward()
        mixed.step()
    assert not torch.equal(a, a0) and not torch.equal(b, b0)
    assert len(mixed._buckets) == 2


def test_parameter_groups_keep_their_own_weight_decay():
    model = make_model()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    optimizer = PressureCookerV7([{"params": decay, "weight_decay": 0.1},
                                  {"params": rest, "weight_decay": 0.0}], lr=3e-3)
    assert train(model, optimizer, steps=10)[-1] < 10
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.1, 0.0]


def test_sparse_gradients_are_refused_with_a_reason():
    embed = nn.Embedding(50, 16, sparse=True)
    optimizer = PressureCookerV7(embed.parameters())
    embed(torch.tensor([1, 2, 3])).sum().backward()
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()


def test_parameters_without_gradients_are_left_alone():
    model = make_model()
    frozen = next(model.parameters())
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3)
    x, y = make_data()
    nn.functional.cross_entropy(model(x), y).backward()
    frozen.grad = None
    before = frozen.detach().clone()
    optimizer.step()
    assert torch.equal(frozen, before)


def test_checkpoint_resumes_the_same_trajectory():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3)
    train(model, optimizer, steps=12)
    saved_model = {k: v.clone() for k, v in model.state_dict().items()}
    # Like torch's own optimizers, state_dict() refers to the live tensors.
    saved = copy.deepcopy(optimizer.state_dict())
    assert saved["v7"]["format"] == 1 and saved["v7"]["global_step"] == 12
    straight = train(model, optimizer, steps=8)

    again = make_model(seed=5)
    again.load_state_dict(saved_model)
    resumed = PressureCookerV7(again.parameters(), lr=3e-3)
    resumed.load_state_dict(saved)
    assert train(again, resumed, steps=8) == pytest.approx(straight, rel=1e-5)


def test_checkpoint_converts_between_state_dtypes_with_a_warning():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3, state_dtype="fp32")
    train(model, optimizer, steps=6)
    saved = optimizer.state_dict()
    target = PressureCookerV7(model.parameters(), lr=3e-3, state_dtype="int8")
    with pytest.warns(UserWarning, match="loses precision"):
        target.load_state_dict(saved)
    weight = next(model.parameters())
    assert target.state[weight]["momentum"].dtype == torch.int8
    restored = target.state[weight]["momentum"].float() * target.state[weight]["scale"]
    assert torch.allclose(restored, optimizer.state[weight]["momentum"],
                          atol=float(optimizer.state[weight]["momentum"].abs().max()) / 100)
    # Widening is announced too, without the precision warning.
    back = PressureCookerV7(model.parameters(), lr=3e-3, state_dtype="fp32")
    with pytest.warns(UserWarning, match="Converting") as caught:
        back.load_state_dict(target.state_dict())
    assert not any("loses precision" in str(w.message) for w in caught)
    assert train(model, back, steps=3)[-1] < 10
    # The same dtype loads silently.
    same = PressureCookerV7(model.parameters(), lr=3e-3, state_dtype="fp32")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same.load_state_dict(saved)


def test_state_from_another_optimizer_is_refused():
    model = make_model()
    v5 = PressureCookerV5(model.parameters())
    train(model, v5, steps=2, report=False)
    v7 = PressureCookerV7(model.parameters())
    with pytest.raises(ValueError, match="not a Pressure Cooker V7 checkpoint"):
        v7.load_state_dict(v5.state_dict())
    with pytest.raises(ValueError, match="not a Pressure Cooker V7 checkpoint"):
        v7.load_state_dict(torch.optim.AdamW(model.parameters()).state_dict())
    newer = v7.state_dict()
    newer["v7"]["format"] = 99
    with pytest.raises(ValueError, match="format 99"):
        v7.load_state_dict(newer)


def test_ema_shadow_tracks_the_weights():
    model = make_model()
    optimizer = PressureCookerV7(model.parameters(), lr=3e-3, ema_decay=0.9)
    assert PressureCookerV7(make_model().parameters()).ema_parameters() == []
    train(model, optimizer, steps=20)
    shadow = optimizer.ema_parameters()
    assert len(shadow) == len(list(model.parameters()))
    gap = max(float((s - p).abs().max()) for s, p in zip(shadow, model.parameters(), strict=True))
    assert 0 < gap < 1.0


def test_describe_is_plain_data():
    import json

    model = make_model()
    optimizer = PressureCookerV7(model.parameters())
    train(model, optimizer, steps=5)
    described = optimizer.describe()
    json.dumps(described)
    assert described["variant"] == "balanced" and described["parameters"] > 0
    assert described["state_bytes"] == optimizer.state_bytes()
    assert described["host_reads_per_step"] in (0, 1)
