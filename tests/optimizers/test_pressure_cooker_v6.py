"""Tests for PressureCookerV6 (SSTM core) and PressureCookerV6V (CUDA wrapper).

V6V's CUDA-only paths (construction on real CUDA parameters, CUDA graph
capture/replay) can only be exercised on a machine with a CUDA device and
are skipped here otherwise via ``pytest.mark.skipif``. Everything that
doesn't strictly require CUDA -- the constructor's device-check guard, the
inherited SSTM update rule, the ``torch.compile`` wrap-and-fallback logic --
is tested on whatever device the suite runs on.
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from hypernix.pressure_cooker_v6 import PressureCookerV6
from hypernix.pressure_cooker_v6v import PressureCookerV6V


def _tiny() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))


def _train_one(opt_factory, steps: int = 40, lr_for_data: float = 1.0):
    torch.manual_seed(0)
    model = _tiny()
    opt = opt_factory(model)
    torch.manual_seed(1)
    x = torch.randn(8, 8)
    target = torch.randn(8, 8)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, opt, losses


# ---------------------------------------------------------------------------
# PressureCookerV6 -- basic construction / config
# ---------------------------------------------------------------------------

class TestPressureCookerV6Init:
    def test_default_init(self):
        m = _tiny()
        opt = PressureCookerV6(m.parameters())
        assert opt.lr == 3e-4
        assert opt.momentum_beta == 0.9
        assert opt.trust_ratio is True
        assert opt.nesterov is False

    def test_custom_hparams(self):
        m = _tiny()
        opt = PressureCookerV6(
            m.parameters(), lr=1e-2, momentum_beta=0.8, weight_decay=0.05,
            nesterov=True, trust_ratio=False, grad_clip=None,
        )
        assert opt.lr == 1e-2
        assert opt.momentum_beta == 0.8
        assert opt.nesterov is True
        assert opt.trust_ratio is False

    def test_describe_reports_kind(self):
        m = _tiny()
        opt = PressureCookerV6(m.parameters())
        d = opt.describe()
        assert d["kind"] == "PressureCookerV6"
        assert "trust_clip" in d
        assert "foreach" in d


# ---------------------------------------------------------------------------
# PressureCookerV6 -- optimization actually works
# ---------------------------------------------------------------------------

class TestPressureCookerV6Training:
    def test_loss_decreases(self):
        _, _, losses = _train_one(
            lambda m: PressureCookerV6(m.parameters(), lr=1e-2, grad_clip=None)
        )
        assert losses[-1] < losses[0]

    def test_loss_decreases_with_nesterov(self):
        _, _, losses = _train_one(
            lambda m: PressureCookerV6(
                m.parameters(), lr=1e-2, nesterov=True, grad_clip=None
            )
        )
        assert losses[-1] < losses[0]

    def test_loss_decreases_without_trust_ratio(self):
        _, _, losses = _train_one(
            lambda m: PressureCookerV6(
                m.parameters(), lr=1e-2, trust_ratio=False, grad_clip=None
            )
        )
        assert losses[-1] < losses[0]

    def test_loss_decreases_without_foreach(self):
        _, _, losses = _train_one(
            lambda m: PressureCookerV6(
                m.parameters(), lr=1e-2, foreach=False, grad_clip=None
            )
        )
        assert losses[-1] < losses[0]

    def test_foreach_and_non_foreach_paths_agree(self):
        """The fused foreach path and the plain-loop fallback should take
        the exact same numerical steps -- same seeds, same trajectory."""
        _, _, losses_fused = _train_one(
            lambda m: PressureCookerV6(
                m.parameters(), lr=1e-2, grad_clip=None, foreach=True
            )
        )
        _, _, losses_loop = _train_one(
            lambda m: PressureCookerV6(
                m.parameters(), lr=1e-2, grad_clip=None, foreach=False
            )
        )
        for a, b in zip(losses_fused, losses_loop, strict=True):
            assert a == pytest.approx(b, rel=1e-5)

    def test_weight_decay_shrinks_weights_with_zero_grad_signal(self):
        torch.manual_seed(0)
        model = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        opt = PressureCookerV6(
            model.parameters(), lr=1e-1, weight_decay=0.5,
            momentum_beta=0.0, trust_ratio=False, grad_clip=None,
        )
        model.weight.grad = torch.zeros_like(model.weight)
        opt.step()
        # Weight decay applies even with a zero gradient.
        assert model.weight.mean().item() < 1.0

    def test_grad_clip_runs(self):
        _, opt, losses = _train_one(
            lambda m: PressureCookerV6(m.parameters(), lr=1e-2, grad_clip=1.0)
        )
        assert losses[-1] < losses[0]

    def test_multiple_param_groups(self):
        torch.manual_seed(0)
        model = _tiny()
        params = list(model.parameters())
        opt = PressureCookerV6(
            [
                {"params": params[:1], "lr": 1e-2},
                {"params": params[1:], "lr": 5e-3},
            ],
            grad_clip=None,
        )
        x = torch.randn(8, 8)
        target = torch.randn(8, 8)
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            loss = (model(x) - target).pow(2).mean()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)

    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        model = _tiny()
        opt = PressureCookerV6(model.parameters(), lr=1e-2, grad_clip=None)
        x = torch.randn(8, 8)
        target = torch.randn(8, 8)
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            (model(x) - target).pow(2).mean().backward()
            opt.step()
        sd = opt.state_dict()
        assert len(sd["state"]) > 0
        for s in sd["state"].values():
            assert "momentum" in s


# ---------------------------------------------------------------------------
# Grad accumulation
# ---------------------------------------------------------------------------

class TestGradAccumulation:
    def test_rejects_bad_value(self):
        m = _tiny()
        with pytest.raises(ValueError, match="grad_accum_steps"):
            PressureCookerV6(m.parameters(), lr=1e-2, grad_accum_steps=0)

    def test_only_updates_on_nth_call(self):
        torch.manual_seed(0)
        m = _tiny()
        before = m[0].weight.detach().clone()
        opt = PressureCookerV6(
            m.parameters(), lr=0.5, grad_accum_steps=3, grad_clip=None,
        )
        for _ in range(2):
            m(torch.randn(4, 8)).sum().backward()
            opt.step()
        assert torch.equal(before, m[0].weight)
        m(torch.randn(4, 8)).sum().backward()
        opt.step()
        assert not torch.equal(before, m[0].weight)

    def test_global_step_only_advances_on_real_update(self):
        torch.manual_seed(0)
        m = _tiny()
        opt = PressureCookerV6(m.parameters(), lr=1e-2, grad_accum_steps=3, grad_clip=None)
        for _ in range(2):
            m(torch.randn(4, 8)).sum().backward()
            opt.step()
        assert opt._global_step == 0
        m(torch.randn(4, 8)).sum().backward()
        opt.step()
        assert opt._global_step == 1


# ---------------------------------------------------------------------------
# GradScaler / non-finite handling
# ---------------------------------------------------------------------------

class _FakeScaler:
    """Stands in for ``torch.cuda.amp.GradScaler`` so this can be exercised
    without a CUDA device -- mirrors the one in
    tests/optimizers/test_pressure_cooker_v048.py exactly (same interface, same
    contract: unscale_ + poison-on-inf, update() just counts calls)."""

    def __init__(self, inf: bool = False) -> None:
        self.unscale_calls = 0
        self.update_calls = 0
        self.inf = inf

    def unscale_(self, optimizer) -> None:
        self.unscale_calls += 1
        if self.inf:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad = torch.full_like(p.grad, float("inf"))
                        return

    def update(self) -> None:
        self.update_calls += 1


class TestGradScalerIntegration:
    def test_skips_update_on_inf(self):
        torch.manual_seed(0)
        m = _tiny()
        scaler = _FakeScaler(inf=True)
        opt = PressureCookerV6(m.parameters(), lr=0.5, grad_scaler=scaler, grad_clip=None)
        before = m[0].weight.detach().clone()
        m(torch.randn(4, 8)).sum().backward()
        opt.step()

        assert scaler.unscale_calls == 1
        assert scaler.update_calls == 1  # scaler still advances
        assert torch.equal(before, m[0].weight)  # no weight update
        assert opt._skipped_steps == 1
        assert opt._global_step == 0  # not counted as a real step

    def test_runs_when_finite(self):
        torch.manual_seed(0)
        m = _tiny()
        scaler = _FakeScaler(inf=False)
        opt = PressureCookerV6(m.parameters(), lr=0.5, grad_scaler=scaler, grad_clip=None)
        before = m[0].weight.detach().clone()
        m(torch.randn(4, 8)).sum().backward()
        opt.step()

        assert scaler.unscale_calls == 1
        assert scaler.update_calls == 1
        assert not torch.equal(before, m[0].weight)
        assert opt._skipped_steps == 0
        assert opt._global_step == 1

    def test_momentum_state_untouched_on_skip(self):
        """A skipped step must not mutate the momentum buffer -- that's
        the whole point of skipping instead of stepping with garbage."""
        torch.manual_seed(0)
        m = _tiny()
        scaler = _FakeScaler(inf=False)
        opt = PressureCookerV6(m.parameters(), lr=0.5, grad_scaler=scaler, grad_clip=None)
        m(torch.randn(4, 8)).sum().backward()
        opt.step()  # finite -- establishes real momentum state
        momentum_before = {
            id(p): s["momentum"].clone() for p, s in opt.state.items()
        }

        scaler.inf = True
        m(torch.randn(4, 8)).sum().backward()
        opt.step()  # should skip cleanly

        for p, s in opt.state.items():
            assert torch.equal(s["momentum"], momentum_before[id(p)])


class TestSkipOnNonfinite:
    def test_disabled_by_default(self):
        torch.manual_seed(0)
        m = _tiny()
        opt = PressureCookerV6(m.parameters(), lr=0.5, grad_clip=None)
        assert opt.skip_on_nonfinite is False
        before = m[0].weight.detach().clone()
        m(torch.randn(4, 8)).sum().backward()
        m[0].weight.grad.fill_(float("inf"))
        opt.step()
        # Without skip_on_nonfinite, V6 doesn't check -- the (broken)
        # update still applies. This documents current behavior rather
        # than asserting a specific NaN pattern.
        assert not torch.equal(before, m[0].weight)

    def test_skips_bad_batch_when_enabled(self):
        torch.manual_seed(0)
        m = _tiny()
        opt = PressureCookerV6(
            m.parameters(), lr=0.5, skip_on_nonfinite=True, grad_clip=None,
        )
        before = m[0].weight.detach().clone()
        m(torch.randn(4, 8)).sum().backward()
        m[0].weight.grad.fill_(float("nan"))
        opt.step()
        assert torch.equal(before, m[0].weight)
        assert opt._skipped_steps == 1


# ---------------------------------------------------------------------------
# Realistic, heterogeneous-shape model (not just a toy MLP)
# ---------------------------------------------------------------------------

class _TinyTransformerBlock(nn.Module):
    """Small but structurally real: embedding + multi-head attention +
    LayerNorm + MLP, with 1-D, 2-D, and embedding-table parameter shapes
    mixed in the same optimizer -- the kind of heterogeneity a toy MLP
    doesn't exercise but real transformer training always has."""

    def __init__(self, vocab=64, d_model=32, n_heads=4, d_ff=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens):
        x = self.embed(tokens)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return self.head(x)


class TestRealisticModel:
    def test_transformer_block_trains_with_v6(self):
        torch.manual_seed(0)
        model = _TinyTransformerBlock()
        opt = PressureCookerV6(model.parameters(), lr=3e-3, grad_clip=1.0)

        torch.manual_seed(1)
        tokens = torch.randint(0, 64, (4, 16))
        targets = torch.randint(0, 64, (4, 16))
        losses = []
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]
        # Confirms V6 actually handled the mix of embedding tables,
        # LayerNorm 1-D affine params, attention in/out projections, and
        # ordinary 2-D linear weights in a single param group without
        # shape/dtype-specific special-casing breaking.
        n_tensors_with_state = len(opt.state)
        n_params = sum(1 for _ in model.parameters())
        assert n_tensors_with_state == n_params

    def test_transformer_block_trains_with_v6_grad_accum(self):
        torch.manual_seed(0)
        model = _TinyTransformerBlock()
        opt = PressureCookerV6(
            model.parameters(), lr=3e-3, grad_clip=1.0, grad_accum_steps=2,
        )
        torch.manual_seed(1)
        tokens = torch.randint(0, 64, (4, 16))
        targets = torch.randint(0, 64, (4, 16))
        losses = []
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Comparative sanity: state footprint should be exactly half of AdamW's
# (one fp32 buffer vs two), for identically-shaped params.
# ---------------------------------------------------------------------------

def _state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.nelement() * v.element_size()
    return total


def test_optimizer_state_is_half_of_adamw():
    torch.manual_seed(0)
    model_a = nn.Linear(64, 64)
    torch.manual_seed(0)
    model_b = nn.Linear(64, 64)

    adamw = AdamW(model_a.parameters(), lr=1e-3)
    v6 = PressureCookerV6(model_b.parameters(), lr=1e-3, grad_clip=None)

    x = torch.randn(4, 64)
    for model, opt in ((model_a, adamw), (model_b, v6)):
        opt.zero_grad(set_to_none=True)
        model(x).pow(2).mean().backward()
        opt.step()

    adamw_bytes = _state_bytes(adamw)
    v6_bytes = _state_bytes(v6)
    # Not an exact 2:1 -- modern torch's AdamW also stores a tiny 0-dim
    # 'step' tensor per parameter alongside exp_avg/exp_avg_sq, so the
    # real ratio is *slightly* better than half. Loose tolerance here
    # because the point under test is "roughly one buffer vs two", not
    # bit-exact parity with AdamW's internal bookkeeping tensors -- see
    # scripts/measure_optimizer_memory.py for the precise, per-hidden-size
    # measured numbers actually cited in the docs.
    assert v6_bytes == pytest.approx(adamw_bytes * 0.5, rel=0.01)


# ---------------------------------------------------------------------------
# PressureCookerV6V
# ---------------------------------------------------------------------------

class TestPressureCookerV6V:
    def test_raises_without_cuda_params(self):
        m = _tiny()
        with pytest.raises(RuntimeError, match="CUDA"):
            PressureCookerV6V(m.parameters(), lr=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_constructs_on_cuda(self):
        m = _tiny().cuda()
        opt = PressureCookerV6V(m.parameters(), lr=1e-3)
        assert opt.describe()["kind"] == "PressureCookerV6V"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_trains_on_cuda(self):
        torch.manual_seed(0)
        model = _tiny().cuda()
        opt = PressureCookerV6V(model.parameters(), lr=1e-2, grad_clip=None)
        x = torch.randn(8, 8, device="cuda")
        target = torch.randn(8, 8, device="cuda")
        first_loss = None
        for i in range(20):
            opt.zero_grad(set_to_none=True)
            loss = (model(x) - target).pow(2).mean()
            loss.backward()
            opt.step()
            if i == 0:
                first_loss = loss.item()
        assert loss.item() < first_loss

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_warmup_and_replay_graph(self):
        torch.manual_seed(0)
        model = _tiny().cuda()
        opt = PressureCookerV6V(model.parameters(), lr=1e-2, grad_clip=None, compile=False)
        x = torch.randn(8, 8, device="cuda")
        target = torch.randn(8, 8, device="cuda")

        def step_fn():
            opt.zero_grad(set_to_none=True)
            loss = (model(x) - target).pow(2).mean()
            loss.backward()
            opt.step()
            return loss

        opt.warmup_graph(step_fn)
        for _ in range(5):
            loss = opt.replay_graph()
        assert torch.isfinite(loss)

    def test_warmup_graph_requires_cuda(self, monkeypatch):
        """Even a (test-only) constructed instance should refuse graph
        capture without a real CUDA device."""
        import hypernix.pressure_cooker_v6v as v6v_mod

        monkeypatch.setattr(v6v_mod, "_has_cuda_param", lambda params: True)
        m = _tiny()
        opt = PressureCookerV6V(m.parameters(), lr=1e-3, compile=False)
        with pytest.raises(RuntimeError, match="CUDA"):
            opt.warmup_graph(lambda: None)

    def test_compile_fallback_is_graceful_when_unavailable(self, monkeypatch):
        """If torch.compile isn't available, V6V should warn and keep
        working via the eager path instead of raising."""
        import hypernix.pressure_cooker_v6v as v6v_mod

        monkeypatch.setattr(v6v_mod, "_has_cuda_param", lambda params: True)
        monkeypatch.delattr(torch, "compile", raising=False)
        m = _tiny()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = PressureCookerV6V(m.parameters(), lr=1e-2, grad_clip=None)
        assert opt._compiled_sstm_step is None
        assert any("torch.compile" in str(w.message) for w in caught)

        x = torch.randn(8, 8)
        target = torch.randn(8, 8)
        opt.zero_grad(set_to_none=True)
        loss = (m(x) - target).pow(2).mean()
        loss.backward()
        opt.step()  # should not raise even without torch.compile
        assert torch.isfinite(loss)
