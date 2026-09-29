"""Tests for v0.48 pressure_cooker rewrite:

* 4 new tiers (Stovetop / Electric / Induction / Pro) with device-tuned defaults.
* Universal selector that picks by param device.
* Grad-accumulation (only the N-th step actually updates).
* GradScaler integration (skip on inf; advance state on finite).
* Backward-compat with the v0.47 signature.
* LR schedule / phase labels unchanged.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _tiny() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(8, 8)


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

def test_v047_signature_still_works() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = _tiny()
    opt = PressureCooker(
        m.parameters(),
        peak_lr=1.0, warmup_steps=2, plateau_steps=3, cooldown_steps=2,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
        lookahead_k=0, lookahead_alpha=0.5,
    )
    # LR schedule identical to v0.47:
    assert opt.scheduled_lr(0) == pytest.approx(0.5)
    assert opt.scheduled_lr(1) == pytest.approx(1.0)
    assert opt.scheduled_lr(2) == 1.0                # plateau
    assert opt.scheduled_lr(5) == 1.0                # still plateau
    assert opt.scheduled_lr(7) == 0.0                # done
    assert opt.phase(0) == "warmup"
    assert opt.phase(3) == "plateau"
    assert opt.phase(5) == "cooldown"
    assert opt.phase(7) == "done"


def test_step_updates_params_basic() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = _tiny()
    before = m.weight.detach().clone()
    opt = PressureCooker(
        m.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=5, cooldown_steps=0,
    )
    m(torch.randn(1, 8)).sum().backward()
    opt.step()
    assert not torch.equal(before, m.weight)


# ---------------------------------------------------------------------------
# The four new tiers
# ---------------------------------------------------------------------------

def test_stovetop_forces_foreach_false() -> None:
    from hypernix.pressure_cooker import StovetopCooker

    opt = StovetopCooker(_tiny().parameters(), peak_lr=0.1)
    assert opt.foreach is False
    assert opt.fused is False
    assert opt.grad_scaler is None


def test_electric_enables_foreach_when_available() -> None:
    from hypernix.pressure_cooker import _HAS_FOREACH, ElectricCooker

    opt = ElectricCooker(_tiny().parameters(), peak_lr=0.1)
    assert opt.foreach is _HAS_FOREACH
    assert opt.fused is False


def test_induction_enables_fused_when_available() -> None:
    from hypernix.pressure_cooker import _HAS_FUSED_ADAMW, InductionCooker

    opt = InductionCooker(_tiny().parameters(), peak_lr=0.1)
    assert opt.foreach is True
    assert opt.fused is _HAS_FUSED_ADAMW


def test_pro_inherits_induction_defaults() -> None:
    from hypernix.pressure_cooker import InductionCooker, ProCooker

    opt = ProCooker(_tiny().parameters(), peak_lr=0.1)
    assert isinstance(opt, InductionCooker)
    assert opt.foreach is True


def test_pro_cooker_graph_warmup_requires_cuda() -> None:
    from hypernix.pressure_cooker import ProCooker

    opt = ProCooker(_tiny().parameters(), peak_lr=0.1)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            opt.warmup_graph(lambda: torch.tensor(0.0))
    else:
        pytest.skip("CUDA available — graph warmup is a runtime concern")


def test_pro_cooker_replay_without_warmup_raises() -> None:
    from hypernix.pressure_cooker import ProCooker

    opt = ProCooker(_tiny().parameters(), peak_lr=0.1)
    with pytest.raises(RuntimeError, match="warmup_graph"):
        opt.replay_graph()


# ---------------------------------------------------------------------------
# Universal selector
# ---------------------------------------------------------------------------

def test_universal_cooker_picks_electric_on_cpu() -> None:
    from hypernix.pressure_cooker import ElectricCooker, universal_cooker

    opt = universal_cooker(_tiny().parameters(), peak_lr=0.1, prefer_speed=True, variant="legacy")
    assert isinstance(opt, ElectricCooker)


def test_universal_cooker_prefer_safety_picks_stovetop_on_cpu() -> None:
    from hypernix.pressure_cooker import StovetopCooker, universal_cooker

    opt = universal_cooker(_tiny().parameters(), peak_lr=0.1, prefer_speed=False, variant="legacy")
    assert isinstance(opt, StovetopCooker)


# ---------------------------------------------------------------------------
# Grad accumulation
# ---------------------------------------------------------------------------

def test_grad_accum_only_updates_on_nth_call() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = _tiny()
    before = m.weight.detach().clone()
    opt = PressureCooker(
        m.parameters(), peak_lr=0.5, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0,
        grad_accum_steps=3,
    )
    # Two "virtual" steps: weights should not move.
    for _ in range(2):
        m(torch.randn(1, 8)).sum().backward()
        opt.step()
    assert torch.equal(before, m.weight)
    # Third call fires the real update.
    m(torch.randn(1, 8)).sum().backward()
    opt.step()
    assert not torch.equal(before, m.weight)


def test_grad_accum_rejects_bad_value() -> None:
    from hypernix.pressure_cooker import PressureCooker

    with pytest.raises(ValueError, match="grad_accum_steps"):
        PressureCooker(_tiny().parameters(), peak_lr=0.1, grad_accum_steps=0)


# ---------------------------------------------------------------------------
# GradScaler integration
# ---------------------------------------------------------------------------

class _FakeScaler:
    """Stands in for ``torch.cuda.amp.GradScaler`` so we can exercise
    both the unscale + inf-skip path and the update-on-finite path
    without needing a CUDA device."""

    def __init__(self, inf: bool = False) -> None:
        self.unscale_calls = 0
        self.update_calls = 0
        self.inf = inf

    def unscale_(self, optimizer) -> None:
        self.unscale_calls += 1
        if self.inf:
            # Poison one grad so _grad_has_inf triggers.
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad = torch.full_like(p.grad, float("inf"))
                        return

    def update(self) -> None:
        self.update_calls += 1


def test_grad_scaler_skips_update_on_inf() -> None:
    from hypernix.pressure_cooker import InductionCooker

    m = _tiny()
    scaler = _FakeScaler(inf=True)
    opt = InductionCooker(
        m.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0,
        grad_scaler=scaler,
    )
    before = m.weight.detach().clone()
    m(torch.randn(1, 8)).sum().backward()
    opt.step()

    assert scaler.unscale_calls == 1
    assert scaler.update_calls == 1                # still advances scaler
    assert torch.equal(before, m.weight)           # no weight update


def test_grad_scaler_step_runs_when_finite() -> None:
    from hypernix.pressure_cooker import InductionCooker

    m = _tiny()
    scaler = _FakeScaler(inf=False)
    opt = InductionCooker(
        m.parameters(), peak_lr=0.5, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0,
        grad_scaler=scaler,
    )
    before = m.weight.detach().clone()
    m(torch.randn(1, 8)).sum().backward()
    opt.step()

    assert scaler.unscale_calls == 1
    assert scaler.update_calls == 1
    assert not torch.equal(before, m.weight)


# ---------------------------------------------------------------------------
# Foreach / fused path parity
# ---------------------------------------------------------------------------

def test_scalar_and_foreach_paths_produce_similar_updates() -> None:
    """Both the pure-Python and multi-tensor paths should move the
    weights by comparable amounts on the first step."""
    from hypernix.pressure_cooker import PressureCooker

    torch.manual_seed(0)
    ma = nn.Linear(8, 8)
    torch.manual_seed(0)
    mb = nn.Linear(8, 8)
    assert torch.equal(ma.weight, mb.weight)

    opt_a = PressureCooker(
        ma.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=5, cooldown_steps=0, foreach=False, fused=False,
    )
    opt_b = PressureCooker(
        mb.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=5, cooldown_steps=0, foreach=True, fused=False,
    )

    x = torch.randn(4, 8)
    ma(x).sum().backward()
    mb(x).sum().backward()

    opt_a.step()
    opt_b.step()

    # Both paths should move the weight by the same amount within fp
    # rounding.  On CPU there's no fused kernel so parity is exact.
    torch.testing.assert_close(ma.weight, mb.weight, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

def test_tiers_registry_lists_all_five() -> None:
    from hypernix.pressure_cooker import TIERS

    assert set(TIERS) == {
        "pressure-cooker", "stovetop", "electric", "induction", "pro",
    }


def test_factory_by_tier_name() -> None:
    from hypernix.pressure_cooker import (
        ElectricCooker,
        InductionCooker,
        ProCooker,
        StovetopCooker,
        pressure_cooker,
    )

    m = _tiny()
    assert isinstance(
        pressure_cooker(m.parameters(), tier="stovetop"), StovetopCooker,
    )
    assert isinstance(
        pressure_cooker(m.parameters(), tier="electric"), ElectricCooker,
    )
    assert isinstance(
        pressure_cooker(m.parameters(), tier="induction"), InductionCooker,
    )
    assert isinstance(
        pressure_cooker(m.parameters(), tier="pro"), ProCooker,
    )
    # Unknown tier -> ValueError.
    with pytest.raises(ValueError, match="unknown pressure cooker tier"):
        pressure_cooker(m.parameters(), tier="pressure")


def test_describe_reports_kind_and_knobs() -> None:
    from hypernix.pressure_cooker import StovetopCooker

    opt = StovetopCooker(_tiny().parameters(), peak_lr=0.1, grad_accum_steps=4)
    d = opt.describe()
    assert d["kind"] == "StovetopCooker"
    assert d["foreach"] is False
    assert d["grad_accum_steps"] == 4


# ---------------------------------------------------------------------------
# Lookahead still works after rewrite
# ---------------------------------------------------------------------------

def test_lookahead_slow_weights_populated() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = _tiny()
    opt = PressureCooker(
        m.parameters(), peak_lr=0.5, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0,
        lookahead_k=2, lookahead_alpha=0.5,
    )
    for _ in range(4):
        m(torch.randn(1, 8)).sum().backward()
        opt.step()

    states = list(opt.state.values())
    assert any("slow" in s for s in states)
