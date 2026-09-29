"""Tests for v0.47: deep_fryer, cake_pan, and the CPU/GPU preset expansion."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# deep_fryer
# ---------------------------------------------------------------------------

def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Linear(16, 32),
        nn.Linear(32, 16),
    )
    return m


def test_light_fry_changes_then_restores() -> None:
    from hypernix import deep_fryer

    m = _tiny_model()
    orig = {k: v.detach().clone() for k, v in m.state_dict().items()}

    fz = deep_fryer.LightFry(model=m, fraction=0.5, noise_std=0.5, seed=0)
    fz.save_pristine()
    touched = fz.fry()

    assert touched                                    # something was changed
    assert not torch.equal(orig["0.weight"], m.state_dict()["0.weight"])

    fz.un_fry()
    for k, v in orig.items():
        assert torch.equal(v, m.state_dict()[k])


def test_heavy_fry_zeroes_some_elements() -> None:
    from hypernix import deep_fryer

    m = _tiny_model()
    fz = deep_fryer.HeavyFry(
        model=m, fraction=1.0, noise_std=0.1,
        zero_rate=1.0, seed=0,
    )
    fz.fry()
    # With fraction=1.0 + zero_rate=1.0 every touched element is zeroed;
    # touched == all elements, so every weight is zero.
    for name, p in m.named_parameters():
        assert torch.count_nonzero(p) == 0, name


def test_deep_fryer_patterns_filter() -> None:
    from hypernix import deep_fryer

    m = _tiny_model()
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}

    # Only touch "0.weight" (the first Linear's weight).
    fz = deep_fryer.LightFry(
        model=m, fraction=1.0, noise_std=1.0,
        patterns=("0.weight",), seed=0,
    )
    fz.fry()
    after = m.state_dict()
    assert not torch.equal(before["0.weight"], after["0.weight"])
    assert torch.equal(before["0.bias"], after["0.bias"])
    assert torch.equal(before["1.weight"], after["1.weight"])


def test_deep_fryer_factory_and_unknown() -> None:
    from hypernix import deep_fryer

    m = _tiny_model()
    fz = deep_fryer.deep_fryer("light-fry", m, fraction=0.1)
    assert isinstance(fz, deep_fryer.LightFry)
    assert fz.fraction == 0.1

    with pytest.raises(ValueError, match="unknown fryer tier"):
        deep_fryer.deep_fryer("nuked", m)


def test_deep_fryer_rejects_bad_args() -> None:
    from hypernix import deep_fryer

    m = _tiny_model()
    with pytest.raises(ValueError, match="fraction"):
        deep_fryer.LightFry(model=m, fraction=1.5)
    with pytest.raises(ValueError, match="noise_std"):
        deep_fryer.LightFry(model=m, noise_std=-0.1)


def test_un_fry_without_snapshot_is_noop() -> None:
    from hypernix import deep_fryer

    m = _tiny_model()
    fz = deep_fryer.LightFry(model=m)
    assert fz.un_fry() == 0


# ---------------------------------------------------------------------------
# cake_pan
# ---------------------------------------------------------------------------

def test_cake_pan_bake_returns_loss() -> None:
    from hypernix import cake_pan

    m = _tiny_model()
    pan = cake_pan.cake_pan(model=m, step_timeout_s=0, snapshot_every=0)
    loss = pan.bake(lambda: torch.tensor(0.5))
    assert loss.item() == 0.5
    assert pan.step_count == 1


def test_cake_pan_rolls_back_on_nan() -> None:
    from hypernix import cake_pan
    from hypernix.cake_pan import BakeOff

    m = _tiny_model()
    orig_w = m[0].weight.detach().clone()

    pan = cake_pan.cake_pan(model=m, step_timeout_s=0, snapshot_every=0)
    pan.save_pristine()

    # Corrupt the model's weight in place, then fake a NaN loss.
    with torch.no_grad():
        m[0].weight.zero_().add_(99.0)

    with pytest.raises(BakeOff, match="NaN"):
        pan.bake(lambda: torch.tensor(float("nan")))

    # After roll_back, the 99.0 corruption should be gone.
    assert torch.equal(orig_w, m[0].weight)


def test_cake_pan_grad_nan_detection() -> None:
    from hypernix import cake_pan
    from hypernix.cake_pan import BakeOff

    m = _tiny_model()
    pan = cake_pan.cake_pan(
        model=m, step_timeout_s=0, snapshot_every=0, check_grads=True,
    )
    pan.save_pristine()

    def step():
        # Run a forward / backward that yields a finite loss but
        # poison the grad manually afterwards.  Finite loss would
        # normally pass; the grad scan should still catch it.
        x = torch.zeros(1, 16)
        y = m(x).sum()
        y.backward()
        m[0].weight.grad.fill_(float("inf"))
        return y

    with pytest.raises(BakeOff, match="grad"):
        pan.bake(step)


def test_cake_pan_snapshot_writes_file(tmp_path) -> None:
    from hypernix import cake_pan

    m = _tiny_model()
    snap = tmp_path / "ckpt.pt"
    pan = cake_pan.cake_pan(
        model=m, step_timeout_s=0,
        snapshot_every=2, snapshot_path=snap,
    )
    pan.save_pristine()

    pan.bake(lambda: torch.tensor(0.1))
    assert not snap.exists()
    pan.bake(lambda: torch.tensor(0.1))
    assert snap.exists()


def test_cake_pan_oven_retries_and_counts() -> None:
    from hypernix import cake_pan

    m = _tiny_model()
    pan = cake_pan.cake_pan(model=m, step_timeout_s=0, snapshot_every=0)
    pan.save_pristine()

    bad_batches = 0
    bake_offs: list = []

    def step(batch):
        nonlocal bad_batches
        if batch == "bad":
            bad_batches += 1
            return torch.tensor(float("nan"))
        return torch.tensor(0.0)

    good = pan.oven(
        ["good", "bad", "good"],
        step,
        on_bake_off=lambda exc: bake_offs.append(exc),
        max_retries_per_batch=1,
    )
    # Two good batches succeed; the "bad" batch tries twice then is
    # skipped.
    assert good == 2
    assert len(bake_offs) >= 1


# ---------------------------------------------------------------------------
# CPU / GPU preset expansion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    # New i5 presets
    "i5-7200u", "i5-7300hq", "i5-7600k",
    "i5-11400", "i5-11600k",
    "i5-12400", "i5-12600k",
    "i5-13400", "i5-13500", "i5-13600k",
    "i5-14600k",
    # New i9 presets
    "i9-7900x", "i9-7980xe",
    "i9-11900k", "i9-12900k", "i9-12900hx",
    "i9-13900k", "i9-13900hx",
    "i9-14900k", "i9-14900ks", "i9-14900hx",
    # New Core Ultra 5 / Ultra 9
    "core-ultra-5-125h", "core-ultra-5-135h", "core-ultra-5-225k",
    "core-ultra-5-235k", "core-ultra-9-185h",
])
def test_new_cpu_presets(name: str) -> None:
    from hypernix.freezer import cpu_preset

    p = cpu_preset(name)
    assert p is not None
    assert p.cores >= 1
    assert p.threads >= p.cores
    assert p.gflops_per_thread > 0


def test_cpu_preset_count_roughly_quadrupled() -> None:
    from hypernix.freezer import CPU_PRESETS

    # v0.43 shipped 16; v0.47 adds ~32 more.
    assert len(CPU_PRESETS) >= 40


@pytest.mark.parametrize("name,vram", [
    # Pascal remainders
    ("gtx-1050", 2.0), ("gtx-1050-ti", 4.0), ("gtx-1060", 6.0),
    ("gtx-1070", 8.0), ("gtx-1070-ti", 8.0),
    # Turing
    ("gtx-1650", 4.0), ("gtx-1660", 6.0), ("gtx-1660-super", 6.0),
    ("rtx-2060", 6.0), ("rtx-2070", 8.0), ("rtx-2070-super", 8.0),
    # Ampere consumer
    ("rtx-3050", 8.0), ("rtx-3060", 12.0), ("rtx-3060-ti", 8.0),
    ("rtx-3070", 8.0), ("rtx-3080", 10.0),
    ("rtx-3090", 24.0), ("rtx-3090-ti", 24.0),
    # Ada
    ("rtx-4060", 8.0), ("rtx-4060-ti-16g", 16.0), ("rtx-4070", 12.0),
    ("rtx-4070-ti", 12.0), ("rtx-4080", 16.0), ("rtx-4090", 24.0),
    # Blackwell consumer
    ("rtx-5070", 12.0), ("rtx-5070-ti", 16.0),
    ("rtx-5080", 16.0), ("rtx-5090", 32.0),
    # Apple
    ("apple-m1", 8.0), ("apple-m2-max", 32.0), ("apple-m4-max", 48.0),
    # AMD
    ("radeon-rx-7900-xtx", 24.0), ("instinct-mi300x", 192.0),
])
def test_new_gpu_presets(name: str, vram: float) -> None:
    from hypernix.freezer import gpu_preset

    p = gpu_preset(name)
    assert p is not None
    assert p.vram_gb == vram


def test_gpu_preset_count_roughly_tripled() -> None:
    from hypernix.freezer import GPU_PRESETS

    # v0.43 shipped 20; v0.47 adds ~40 more.
    assert len(GPU_PRESETS) >= 55


def test_apple_silicon_has_compute_cap_zero() -> None:
    """MPS isn't a CUDA device; Apple presets use (0, 0) sentinel."""
    from hypernix.freezer import gpu_preset

    assert gpu_preset("apple-m1").compute_capability == (0, 0)
    assert gpu_preset("apple-m4-max").compute_capability == (0, 0)


def test_amd_instinct_has_compute_cap_zero() -> None:
    from hypernix.freezer import gpu_preset

    assert gpu_preset("instinct-mi300x").compute_capability == (0, 0)


def test_rtx_5090_is_new_class() -> None:
    from hypernix.freezer import gpu_preset

    assert gpu_preset("rtx-5090").freezer_class == "New"
    assert gpu_preset("rtx-5090").vram_gb == 32.0


def test_v047_exports() -> None:
    import hypernix

    assert hypernix.deep_fryer is not None
    assert hypernix.cake_pan is not None
