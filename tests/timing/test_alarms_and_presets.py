"""Tests for v0.43 CPU/GPU presets and the smoke_alarm module."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import torch

# ---------------------------------------------------------------------------
# CPU presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "i7-7660u", "i7-7700hq", "i7-7700k",
    "i7-11700k", "i7-11800h",
    "i7-12700k", "i7-12700h",
    "i7-13700k", "i7-13700h",
    "i7-14700k", "i7-14700hx",
    "core-ultra-7-155h", "core-ultra-7-165h", "core-ultra-7-258v",
    "core-ultra-7-265k", "core-ultra-9-285k",
])
def test_cpu_preset_present(name: str) -> None:
    from hypernix.freezer import cpu_preset

    p = cpu_preset(name)
    assert p is not None
    assert p.cores >= 1
    assert p.threads >= p.cores
    assert p.recommended_threads >= 1
    assert p.gflops_per_thread > 0


def test_cpu_preset_lookup_is_case_and_dash_insensitive() -> None:
    from hypernix.freezer import cpu_preset

    a = cpu_preset("i7-7700hq")
    b = cpu_preset("I7_7700HQ")
    c = cpu_preset("i7 7700hq")
    assert a is b is c


def test_cpu_preset_unknown_returns_none() -> None:
    from hypernix.freezer import cpu_preset

    assert cpu_preset("does-not-exist") is None


def test_i7_7660u_specs_match_intel() -> None:
    from hypernix.freezer import cpu_preset

    p = cpu_preset("i7-7660u")
    assert p.cores == 2
    assert p.threads == 4
    assert "AVX2" in p.avx_levels


def test_core_ultra_2_has_avx10() -> None:
    """Arrow Lake (Series 2) is the first to ship AVX10 in client SKUs."""
    from hypernix.freezer import cpu_preset

    assert "AVX10" in cpu_preset("core-ultra-7-265k").avx_levels
    assert "AVX10" in cpu_preset("core-ultra-9-285k").avx_levels
    # Series 1 / Meteor Lake doesn't:
    assert "AVX10" not in cpu_preset("core-ultra-7-155h").avx_levels


# ---------------------------------------------------------------------------
# GPU presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected_vram", [
    ("h100", 80.0), ("h100-94", 94.0), ("h200", 141.0),
    ("rtx-a4500", 20.0), ("rtx-a5000", 24.0), ("rtx-a5500", 24.0),
    ("rtx-a6000", 48.0),
    ("rtx-pro-4000-ada", 20.0), ("rtx-pro-5000-ada", 32.0),
    ("rtx-pro-6000-ada", 48.0), ("rtx-pro-6000-blackwell", 96.0),
    ("rtx-4070-ti-super", 16.0), ("rtx-4080-super", 16.0),
    ("gtx-1660-ti", 6.0),
    ("rtx-2080", 8.0), ("rtx-2080-super", 8.0), ("rtx-2080-ti", 11.0),
    ("rtx-3080-ti", 12.0),
])
def test_gpu_preset_vram(name: str, expected_vram: float) -> None:
    from hypernix.freezer import gpu_preset

    p = gpu_preset(name)
    assert p is not None
    assert p.vram_gb == expected_vram


def test_gpu_preset_compute_capability() -> None:
    from hypernix.freezer import gpu_preset

    assert gpu_preset("h100").compute_capability == (9, 0)
    assert gpu_preset("h200").compute_capability == (9, 0)
    assert gpu_preset("rtx-a6000").compute_capability == (8, 6)
    assert gpu_preset("rtx-pro-6000-ada").compute_capability == (8, 9)
    assert gpu_preset("rtx-pro-6000-blackwell").compute_capability == (12, 0)
    assert gpu_preset("rtx-4080-super").compute_capability == (8, 9)
    assert gpu_preset("gtx-1660-ti").compute_capability == (7, 5)
    assert gpu_preset("rtx-2080-ti").compute_capability == (7, 5)
    assert gpu_preset("rtx-3080-ti").compute_capability == (8, 6)


def test_turing_uses_fp16_not_bf16() -> None:
    from hypernix.freezer import gpu_preset

    # sm_75 has no native bf16:
    assert gpu_preset("rtx-2080-ti").preferred_dtype == torch.float16
    assert gpu_preset("rtx-2080-super").preferred_dtype == torch.float16
    assert gpu_preset("gtx-1660-ti").preferred_dtype == torch.float16


def test_freezer_class_assignment() -> None:
    from hypernix.freezer import gpu_preset

    # >= 11GB cards land on New, smaller cards on Old, regardless of arch.
    assert gpu_preset("rtx-2080-ti").freezer_class == "New"  # 11GB
    assert gpu_preset("gtx-1660-ti").freezer_class == "Old"  # 6GB
    assert gpu_preset("rtx-4070-ti-super").freezer_class == "New"  # 16GB
    assert gpu_preset("h100").freezer_class == "New"


# ---------------------------------------------------------------------------
# RadsAlarm (lightest)
# ---------------------------------------------------------------------------

def test_rads_alarm_basic_budget() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(
        time_budget_seconds=3600.0, model_params=100_000_000,
        context_length=1024, batch_size=1,
    )
    b = a.budget()
    assert b.estimated_step_seconds == pytest.approx(1.0)
    # 3600s * 0.9 / 1.0 = 3240
    assert b.recommended_steps == 3240
    assert "RadsAlarm" in b.notes


def test_rads_alarm_scales_linearly_with_params() -> None:
    from hypernix import smoke_alarm

    base = smoke_alarm.rads_alarm(time_budget_seconds=3600.0, model_params=100_000_000)
    big = smoke_alarm.rads_alarm(time_budget_seconds=3600.0, model_params=500_000_000)
    assert big.estimate_step_seconds() == pytest.approx(5 * base.estimate_step_seconds())


def test_rads_alarm_safety_margin_reduces_steps() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(
        time_budget_seconds=1000.0, safety_margin=0.0, model_params=100_000_000,
    )
    b = smoke_alarm.rads_alarm(
        time_budget_seconds=1000.0, safety_margin=0.5, model_params=100_000_000,
    )
    assert a.recommended_steps() == 1000
    assert b.recommended_steps() == 500


# ---------------------------------------------------------------------------
# GasAlarm (mid)
# ---------------------------------------------------------------------------

def test_gas_alarm_h100_faster_than_baseline() -> None:
    from hypernix import smoke_alarm

    rads = smoke_alarm.rads_alarm(time_budget_seconds=3600.0, model_params=100_000_000)
    gas = smoke_alarm.gas_alarm(time_budget_seconds=3600.0, model_params=100_000_000,
                                gpu_name="h100")
    assert gas.estimate_step_seconds() < rads.estimate_step_seconds() / 4


def test_gas_alarm_pascal_slower_than_baseline() -> None:
    from hypernix import smoke_alarm

    gas = smoke_alarm.gas_alarm(time_budget_seconds=3600.0, model_params=100_000_000,
                                gpu_name="gtx-1080")
    # 320 GB/s vs. 700 GB/s baseline -> ~2.2× slower.
    assert gas.estimate_step_seconds() > 1.5
    assert gas.estimate_step_seconds() < 3.5


def test_gas_alarm_cpu_only_is_much_slower() -> None:
    from hypernix import smoke_alarm

    gas = smoke_alarm.gas_alarm(time_budget_seconds=3600.0, model_params=100_000_000,
                                cpu_name="i7-7700hq")
    # 20× CPU-vs-GPU penalty plus GFLOPS scaling -> definitely > 10s.
    assert gas.estimate_step_seconds() > 10.0


def test_gas_alarm_unknown_gpu_falls_back_to_baseline() -> None:
    from hypernix import smoke_alarm

    gas = smoke_alarm.gas_alarm(time_budget_seconds=3600.0, model_params=100_000_000,
                                gpu_name="not-a-real-gpu")
    rads = smoke_alarm.rads_alarm(time_budget_seconds=3600.0, model_params=100_000_000)
    assert gas.estimate_step_seconds() == pytest.approx(rads.estimate_step_seconds())


# ---------------------------------------------------------------------------
# ModernAlarm (warmup-measured)
# ---------------------------------------------------------------------------

def test_modern_alarm_uses_measured_time() -> None:
    from hypernix import smoke_alarm

    def fake_step() -> None:
        time.sleep(0.01)  # 10ms per step

    a = smoke_alarm.modern_alarm(time_budget_seconds=10.0, step_fn=fake_step,
                                 warmup_steps=3)
    # Should measure ~0.01s per step. The upper bound is deliberately loose
    # (50x the nominal 10ms) rather than tight (previously 0.1s / 10x):
    # median-of-3 wall-clock timing is noisy on shared/virtualized CI
    # runners, especially Windows, where a single slow scheduler tick can
    # push a 10ms sleep past 100ms and flake a tight bound. We still assert
    # well below the generic-fallback value of 1.0s (see
    # test_modern_alarm_falls_back_before_warmup) so a regression that
    # silently falls back to the default estimate is still caught.
    assert 0.005 < a.estimate_step_seconds() < 0.5
    # Calculate expected steps based on actual measured time.
    # With 10s budget and 0.9 safety factor, expected steps = 9 / measured_step_seconds
    expected_steps = 9.0 / a.estimate_step_seconds()
    # Allow 20% tolerance around the calculated expectation
    assert expected_steps * 0.8 < a.recommended_steps() < expected_steps * 1.2


def test_modern_alarm_falls_back_before_warmup() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.ModernAlarm(time_budget_seconds=10.0, model_params=100_000_000)
    # No warmup yet → uses generic estimate.
    assert a.measured_step_seconds is None
    assert a.estimate_step_seconds() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AutoAlarm (selector)
# ---------------------------------------------------------------------------

def test_auto_alarm_with_warmup_picks_modern() -> None:
    from hypernix import smoke_alarm

    def step() -> None:
        time.sleep(0.005)

    a = smoke_alarm.auto_alarm(
        time_budget_seconds=10.0, warmup_step_fn=step, detect_hardware=False,
    )
    assert isinstance(a, smoke_alarm.ModernAlarm)
    assert a.measured_step_seconds is not None


# Regression: GasAlarm(preset="i7-7700hq") used to raise
# "unexpected keyword argument 'preset'".  Every constructor / factory
# now accepts ``preset=`` as a one-string shortcut that resolves
# against GPU_PRESETS first, then CPU_PRESETS.


def test_gas_alarm_accepts_preset_cpu() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.GasAlarm(time_budget_seconds=3600.0, preset="i7-7700hq")
    assert a.cpu is not None
    assert a.cpu.name.startswith("Intel Core i7-7700HQ")
    assert a.gpu is None


def test_gas_alarm_accepts_preset_gpu() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.GasAlarm(time_budget_seconds=3600.0, preset="rtx-3080-ti")
    assert a.gpu is not None
    assert a.gpu.name.startswith("NVIDIA GeForce RTX 3080 Ti")
    assert a.cpu is None


def test_gas_alarm_factory_accepts_preset() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.gas_alarm(time_budget_seconds=3600.0, preset="h100")
    assert a.gpu is not None
    assert "H100" in a.gpu.name


def test_auto_alarm_accepts_preset() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.auto_alarm(
        time_budget_seconds=3600.0, preset="i7-7700hq",
        detect_hardware=False,
    )
    assert isinstance(a, smoke_alarm.GasAlarm)
    assert a.cpu is not None


def test_gas_alarm_unknown_preset_lists_valid() -> None:
    import pytest

    from hypernix import smoke_alarm

    with pytest.raises(ValueError, match="unknown preset"):
        smoke_alarm.GasAlarm(time_budget_seconds=3600.0, preset="not-real-cpu")


def test_preset_does_not_override_explicit_cpu() -> None:
    """Explicit ``cpu=`` wins over a conflicting ``preset=`` hint."""
    from hypernix import smoke_alarm
    from hypernix.freezer import cpu_preset

    explicit = cpu_preset("i7-14700k")
    a = smoke_alarm.GasAlarm(
        time_budget_seconds=3600.0, cpu=explicit, preset="i7-7660u",
    )
    assert a.cpu is explicit


def test_auto_alarm_with_gpu_name_picks_gas() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.auto_alarm(
        time_budget_seconds=3600.0, gpu_name="h100", detect_hardware=False,
    )
    assert isinstance(a, smoke_alarm.GasAlarm)
    assert a.gpu is not None
    assert a.gpu.name.startswith("NVIDIA H100")


def test_auto_alarm_no_inputs_picks_rads() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.auto_alarm(time_budget_seconds=3600.0, detect_hardware=False)
    assert isinstance(a, smoke_alarm.RadsAlarm)


# ---------------------------------------------------------------------------
# Mid-run check + storage warning
# ---------------------------------------------------------------------------

def test_alarm_check_on_pace() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(time_budget_seconds=1000.0, model_params=100_000_000)
    # 100 steps after 100s with ~1s/step -> exactly on pace.
    status = a.check(elapsed_seconds=100.0, completed_steps=100)
    assert status.on_pace is True
    assert status.expected_steps == 100


def test_alarm_check_behind_pace() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(time_budget_seconds=1000.0, model_params=100_000_000)
    # Half the expected steps -> definitely behind.
    status = a.check(elapsed_seconds=100.0, completed_steps=20)
    assert status.on_pace is False


def test_storage_warning_triggers() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(
        time_budget_seconds=10000.0, model_params=100_000_000,
        available_storage_gb=1.0,
    )
    # 10 saves * 0.5GB each = 5GB, > 1GB available.
    msg = a.storage_warning(save_every=1000, snapshot_size_gb=0.5)
    assert "storage warning" in msg


def test_storage_warning_silent_when_no_budget() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(
        time_budget_seconds=10000.0, model_params=100_000_000,
        available_storage_gb=None,
    )
    assert a.storage_warning(save_every=1000, snapshot_size_gb=10.0) == ""


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def test_detect_gpu_preset_cpu_returns_none() -> None:
    from hypernix.smoke_alarm import detect_gpu_preset

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert detect_gpu_preset() is None


def test_detect_gpu_preset_finds_h100() -> None:
    from hypernix.smoke_alarm import detect_gpu_preset

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA H100 80GB HBM3"),
    ):
        p = detect_gpu_preset()
    assert p is not None
    assert "H100" in p.name


def test_detect_gpu_preset_unknown_returns_none() -> None:
    from hypernix.smoke_alarm import detect_gpu_preset

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="Some Generic GPU"),
    ):
        assert detect_gpu_preset() is None


# ---------------------------------------------------------------------------
# Aliases the user requested
# ---------------------------------------------------------------------------

def test_radioactive_alarm_alias() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.radioactive_alarm(time_budget_seconds=3600.0)
    assert isinstance(a, smoke_alarm.RadsAlarm)
    assert smoke_alarm.rad_alarm is smoke_alarm.rads_alarm


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_v043_exports_present() -> None:
    import hypernix

    assert hypernix.smoke_alarm is not None
    assert hypernix.cpu_preset is not None
    assert hypernix.gpu_preset is not None
    assert hypernix.CPU_PRESETS is not None
    assert hypernix.GPU_PRESETS is not None
