"""Tests for hypernix.freezer — VRAM manager (Old / New / Flash)."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

# ---------------------------------------------------------------------------
# probe_vram / VRAMBudget
# ---------------------------------------------------------------------------

def test_probe_vram_cpu_only_returns_zeroed_budget() -> None:
    from hypernix.freezer import probe_vram

    with patch.object(torch.cuda, "is_available", return_value=False):
        b = probe_vram()
    assert b.device == "cpu"
    assert b.total == 0
    assert b.free == 0
    assert b.total_gb == 0.0


def test_probe_vram_cuda_reads_mem_get_info() -> None:
    from hypernix.freezer import probe_vram

    fake_total = 12 * 1024 ** 3
    fake_free = 9 * 1024 ** 3
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "mem_get_info", return_value=(fake_free, fake_total)),
    ):
        b = probe_vram()
    assert b.device == "cuda:0"
    assert b.total == fake_total
    assert b.free == fake_free
    assert b.total_gb == pytest.approx(12.0)
    assert b.free_gb == pytest.approx(9.0)
    assert b.used_gb == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# OldFreezer / NewFreezer tuning
# ---------------------------------------------------------------------------

def test_old_freezer_defaults_are_conservative() -> None:
    from hypernix.freezer import OldFreezer

    fz = OldFreezer()
    assert fz.base_batch_size == 1
    assert fz.base_context_length == 512
    assert fz.empty_cache_each_step is True
    # fp16 on real Pascal/Volta/Turing GPUs; bf16 on Ampere+; fp32 on CPU.
    assert fz.preferred_dtype in (torch.bfloat16, torch.float16, torch.float32)


def test_new_freezer_defaults_are_generous() -> None:
    from hypernix.freezer import NewFreezer

    fz = NewFreezer()
    assert fz.base_batch_size == 8
    assert fz.base_context_length == 2048
    assert fz.empty_cache_each_step is False
    assert fz.preferred_dtype == torch.float32


def test_old_freezer_caps_hint() -> None:
    """On an 8-10GB card, a caller-supplied hint is clamped to base_batch_size."""
    from hypernix.freezer import OldFreezer

    fz = OldFreezer()
    assert fz.suggest_batch_size(hint=32) == 1
    assert fz.suggest_batch_size(hint=1) == 1
    assert fz.suggest_context_length(hint=4096) == 512


def test_new_freezer_respects_hint() -> None:
    """On an 11GB+ card, the caller's hint wins."""
    from hypernix.freezer import NewFreezer

    fz = NewFreezer()
    assert fz.suggest_batch_size(hint=32) == 32
    assert fz.suggest_batch_size(hint=None) == 8
    assert fz.suggest_context_length(hint=4096) == 4096


# ---------------------------------------------------------------------------
# auto_freezer
# ---------------------------------------------------------------------------

def test_auto_freezer_picks_old_below_threshold() -> None:
    from hypernix.freezer import OldFreezer, VRAMBudget, auto_freezer

    with patch("hypernix.freezer.probe_vram",
               return_value=VRAMBudget(device="cuda:0", total=8 * 1024 ** 3, free=6 * 1024 ** 3)):
        fz = auto_freezer()
    assert isinstance(fz, OldFreezer)


def test_auto_freezer_picks_new_above_threshold() -> None:
    from hypernix.freezer import NewFreezer, VRAMBudget, auto_freezer

    with patch("hypernix.freezer.probe_vram",
               return_value=VRAMBudget(device="cuda:0", total=24 * 1024 ** 3, free=20 * 1024 ** 3)):
        fz = auto_freezer()
    assert isinstance(fz, NewFreezer)


def test_auto_freezer_cpu_falls_back_to_old() -> None:
    from hypernix.freezer import OldFreezer, auto_freezer

    with patch.object(torch.cuda, "is_available", return_value=False):
        fz = auto_freezer()
    assert isinstance(fz, OldFreezer)


# ---------------------------------------------------------------------------
# FlashFreezer — OOM retry
# ---------------------------------------------------------------------------

def test_flash_freezer_passes_through_on_success() -> None:
    from hypernix.freezer import FlashFreezer, OldFreezer

    fz = FlashFreezer(base=OldFreezer(), max_retries=2, backoff_s=0.0)
    assert fz.guard(lambda: 7) == 7


def test_flash_freezer_retries_on_oom() -> None:
    from hypernix.freezer import FlashFreezer, NewFreezer

    fz = FlashFreezer(base=NewFreezer(), max_retries=3, backoff_s=0.0, slow=True)

    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"

    # Skip the wait_for loop entirely while testing.
    with patch.object(FlashFreezer, "wait_for", return_value=True):
        result = fz.guard(flaky)
    assert result == "ok"
    assert calls["n"] == 3


def test_flash_freezer_slow_halves_current_batch_size() -> None:
    from hypernix.freezer import FlashFreezer, NewFreezer

    fz = FlashFreezer(base=NewFreezer(), max_retries=5, backoff_s=0.0, slow=True)
    assert fz.current_batch_size == 8

    calls = {"n": 0}

    def fail_twice() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "done"

    with patch.object(FlashFreezer, "wait_for", return_value=True):
        fz.guard(fail_twice)
    # Halved twice: 8 -> 4 -> 2.
    assert fz.current_batch_size == 2


def test_flash_freezer_reraises_after_max_retries() -> None:
    from hypernix.freezer import FlashFreezer, OldFreezer

    fz = FlashFreezer(base=OldFreezer(), max_retries=2, backoff_s=0.0)

    def always_oom() -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with (
        patch.object(FlashFreezer, "wait_for", return_value=True),
        pytest.raises(torch.cuda.OutOfMemoryError),
    ):
        fz.guard(always_oom)


def test_flash_freezer_wait_for_returns_true_on_cpu() -> None:
    from hypernix.freezer import FlashFreezer, OldFreezer

    fz = FlashFreezer(base=OldFreezer(), backoff_s=0.0)
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert fz.wait_for(min_free_gb=100.0) is True


def test_flash_freezer_mirrors_base_tuning() -> None:
    from hypernix.freezer import FlashFreezer, OldFreezer

    base = OldFreezer()
    fz = FlashFreezer(base=base, backoff_s=0.0)
    assert fz.preferred_dtype == base.preferred_dtype
    assert fz.base_batch_size == base.base_batch_size
    assert fz.base_context_length == base.base_context_length
    assert fz._caps_hint == base._caps_hint


# ---------------------------------------------------------------------------
# Public factory shortcuts
# ---------------------------------------------------------------------------

def test_factory_shortcuts() -> None:
    from hypernix.freezer import (
        FlashFreezer,
        NewFreezer,
        OldFreezer,
        flash_freezer,
        new_freezer,
        old_freezer,
    )

    assert isinstance(old_freezer(), OldFreezer)
    assert isinstance(new_freezer(), NewFreezer)
    assert isinstance(flash_freezer(base=OldFreezer(), backoff_s=0.0), FlashFreezer)


def test_repr_includes_device_and_gb() -> None:
    from hypernix.freezer import OldFreezer, VRAMBudget

    fz = OldFreezer()
    with patch("hypernix.freezer.probe_vram",
               return_value=VRAMBudget(device="cuda:0", total=10 * 1024 ** 3, free=8 * 1024 ** 3)):
        s = repr(fz)
    assert "OldFreezer" in s
    assert "cuda:0" in s
    assert "10.0GB" in s


def test_freezer_exposed_on_package() -> None:
    import hypernix

    assert hypernix.freezer is not None
    assert hypernix.freezer.auto_freezer is not None


# ---------------------------------------------------------------------------
# CUDA 6.1 (Pascal) helpers
# ---------------------------------------------------------------------------

def test_compute_capability_none_on_cpu() -> None:
    from hypernix.freezer import compute_capability

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert compute_capability() is None


def test_is_pascal_true_for_sm61() -> None:
    from hypernix.freezer import is_pascal

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_capability", return_value=(6, 1)),
    ):
        assert is_pascal() is True


def test_is_pascal_false_for_ampere() -> None:
    from hypernix.freezer import is_pascal

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    ):
        assert is_pascal() is False


def test_pascal_safe_dtype_on_pascal_returns_fp16() -> None:
    from hypernix.freezer import pascal_safe_dtype

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_capability", return_value=(6, 1)),
    ):
        assert pascal_safe_dtype() == torch.float16


def test_pascal_safe_dtype_on_turing_returns_fp16() -> None:
    from hypernix.freezer import pascal_safe_dtype

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)),
    ):
        assert pascal_safe_dtype() == torch.float16


def test_pascal_safe_dtype_on_ampere_returns_bf16_when_supported() -> None:
    from hypernix.freezer import pascal_safe_dtype

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        patch.object(torch.cuda, "is_bf16_supported", return_value=True),
    ):
        assert pascal_safe_dtype() == torch.bfloat16


def test_pascal_safe_dtype_on_cpu_returns_fp32() -> None:
    from hypernix.freezer import pascal_safe_dtype

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert pascal_safe_dtype() == torch.float32


def test_pascal_mode_hints_shape() -> None:
    from hypernix.freezer import pascal_mode_hints

    hints = pascal_mode_hints()
    assert hints["dtype"] == torch.float16
    assert hints["use_sdpa"] is False
    assert hints["use_compile"] is False
    assert hints["tf32"] is False
    assert "cu118" in hints["install_hint"]
