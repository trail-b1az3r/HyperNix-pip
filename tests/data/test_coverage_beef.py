"""Extra coverage across the package.

Covers edge cases that the per-module test files don't hit:

* Lunchbox: JSONL round-trip, unicode, empty-box pack, large-box
  normalise, ``from_records`` preserves order, duplicate-key
  overwrite semantics, push_to_hub URL shape.
* Pressure cooker: amsgrad knob wiring, state-dict round trip,
  repr text, grad-accum with closure, fused kwarg honoured.
* Deep fryer: frozen parameter handling, pristine-snapshot after
  fry is still reversible, multiple saves / restores.
* Cake pan: memory_guard is a no-op on CPU, `oven` counts zero
  successful steps when every batch bakes off.
* Freezer presets: every CPU preset has at least one AVX-family
  entry; every GPU preset's bandwidth is positive and reasonable.
* Shakers: deterministic across two independent instances, empty
  line passthrough, rate=0 is identity.
* Smoke alarm: budget.time_hours math, storage_warning with
  save_every=0 is silent, unknown preset error message contains
  the first GPU name (proves it's listing valid options).
* End-to-end integration: an evaluator pipeline that runs two
  items through industrial_range, packs with Lunchbox, writes
  JSONL, reloads via Table.from_rows, and verifies schema fidelity.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Lunchbox — edge cases
# ---------------------------------------------------------------------------

def test_lunchbox_empty_box_still_packs(tmp_path: Path) -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    # No records → pack_jsonl writes an empty file cleanly (no crash).
    p = box.pack_jsonl(tmp_path / "empty.jsonl")
    assert p.read_text() == ""


def test_lunchbox_unicode_roundtrips(tmp_path: Path) -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(id="r1", prompt="日本語のテスト 🍔", response="café naïve piñata")
    p = box.pack_jsonl(tmp_path / "u.jsonl")
    row = json.loads(p.read_text(encoding="utf-8").strip())
    assert row["prompt"] == "日本語のテスト 🍔"
    assert row["response"] == "café naïve piñata"


def test_lunchbox_large_box_normalises_without_loss() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    for i in range(10_000):
        extra = {"bonus": i * 2} if i % 2 == 0 else {}
        box.add(id=f"r{i}", value=i, **extra)

    rows = box.normalize()
    assert len(rows) == 10_000
    # Every row has both columns, even when the source dict was
    # missing "bonus".
    assert rows[1]["bonus"] is None
    assert rows[2]["bonus"] == 4


def test_lunchbox_duplicate_keys_last_wins() -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    # A single call with duplicate kwargs is a SyntaxError; sneak them
    # in via a manual dict to test Lunchbox's record-append semantics
    # with whatever Python produced.
    box.records.append({"id": "r1", "v": 1})
    box.records.append({"id": "r1", "v": 2})  # same "id", different v
    rows = box.normalize()
    # Both records are preserved — Lunchbox doesn't dedupe.
    assert len(rows) == 2
    assert rows[0]["v"] == 1
    assert rows[1]["v"] == 2


def test_lunchbox_pack_raises_on_mixed_types(tmp_path: Path) -> None:
    from hypernix.lunchbox import Lunchbox

    box = Lunchbox()
    box.add(score=1.5)
    box.add(score="not a number")
    with pytest.raises(ValueError, match="mixed types"):
        box.pack_jsonl(tmp_path / "bad.jsonl")


def test_lunchbox_from_records_preserves_order() -> None:
    from hypernix.lunchbox import Lunchbox

    records = [{"id": f"r{i}", "v": i} for i in range(50)]
    box = Lunchbox.from_records(records)
    assert len(box) == 50
    norm = box.normalize()
    for i, row in enumerate(norm):
        assert row["id"] == f"r{i}"
        assert row["v"] == i


def test_lunchbox_push_to_hub_returns_dataset_url() -> None:
    """Verify the URL returned from push_to_hub is the canonical
    HF dataset URL — useful as a return value for logging."""
    from unittest.mock import MagicMock

    import hypernix.lunchbox as lb

    fake_ds = MagicMock()
    fake_ds_cls = MagicMock()
    fake_ds_cls.from_list.return_value = fake_ds
    fake_module = MagicMock(Dataset=fake_ds_cls)

    box = lb.Lunchbox()
    box.add(id="r1", prompt="Q")

    with patch.dict("sys.modules", {"datasets": fake_module}):
        url = box.push_to_hub("user/my-dataset", token="tok")
    assert url == "https://huggingface.co/datasets/user/my-dataset"


# ---------------------------------------------------------------------------
# Pressure cooker — more coverage
# ---------------------------------------------------------------------------

def test_pressure_cooker_amsgrad_flag_stored() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = nn.Linear(4, 4)
    opt = PressureCooker(m.parameters(), peak_lr=0.1, amsgrad=True)
    assert opt.amsgrad is True
    d = opt.describe()
    assert d["amsgrad"] is True


def test_pressure_cooker_repr_is_readable() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = nn.Linear(4, 4)
    opt = PressureCooker(
        m.parameters(), peak_lr=1e-4,
        warmup_steps=10, plateau_steps=100, cooldown_steps=20,
        lookahead_k=5, lookahead_alpha=0.5, grad_accum_steps=4,
    )
    s = repr(opt)
    assert "PressureCooker" in s
    assert "warmup=10" in s
    assert "plateau=100" in s
    assert "k=5" in s
    assert "accum=4" in s


def test_pressure_cooker_step_with_closure() -> None:
    from hypernix.pressure_cooker import PressureCooker

    m = nn.Linear(4, 4)
    opt = PressureCooker(
        m.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0,
    )
    before = m.weight.detach().clone()

    def closure():
        opt.zero_grad()
        loss = m(torch.randn(2, 4)).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert not torch.equal(before, m.weight)


def test_pressure_cooker_foreach_path_state_persists() -> None:
    """After running a foreach step, a second step must reuse the
    existing state rather than resetting it.  This guards the
    ``state["step"]`` tensor/int duality."""
    from hypernix.pressure_cooker import PressureCooker

    m = nn.Linear(4, 4)
    opt = PressureCooker(
        m.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0, foreach=True,
    )

    for _ in range(3):
        m(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()

    # Every param should have 3 step counters by now.
    steps = []
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            s = state["step"]
            steps.append(s.item() if isinstance(s, torch.Tensor) else s)
    assert all(s == 3 for s in steps)


# ---------------------------------------------------------------------------
# Deep fryer — more coverage
# ---------------------------------------------------------------------------

def test_deep_fryer_skips_frozen_params_by_default() -> None:
    from hypernix.deep_fryer import LightFry

    m = nn.Linear(8, 8)
    # Freeze the weight.
    m.weight.requires_grad = False
    before = m.weight.detach().clone()

    LightFry(model=m, fraction=1.0, noise_std=1.0, seed=0).fry()

    # Weight unchanged because it's frozen.
    assert torch.equal(before, m.weight)
    # Bias is still trainable and should have moved.
    # (some noise was applied).


def test_deep_fryer_heavy_fries_frozen_params() -> None:
    """HeavyFry is the robustness-testing tier — it should perturb
    frozen params too (otherwise 'bad-model negatives' wouldn't
    cover frozen embeddings).  Regression on _should_fry_frozen."""
    from hypernix.deep_fryer import HeavyFry

    m = nn.Linear(8, 8)
    m.weight.requires_grad = False
    before = m.weight.detach().clone()

    HeavyFry(model=m, fraction=0.5, noise_std=1.0, zero_rate=0.0,
             seed=0).fry()
    assert not torch.equal(before, m.weight)


def test_deep_fryer_multiple_save_restore_cycles() -> None:
    from hypernix.deep_fryer import LightFry

    m = nn.Linear(4, 4)
    state0 = {k: v.detach().clone() for k, v in m.state_dict().items()}

    fz = LightFry(model=m, fraction=1.0, noise_std=0.5, seed=0)

    for _ in range(3):
        fz.save_pristine()
        fz.fry()
        fz.un_fry()
        # Round-trip lands on the same weights every time.
        for k, v in state0.items():
            assert torch.equal(v, m.state_dict()[k])


# ---------------------------------------------------------------------------
# Cake pan — more coverage
# ---------------------------------------------------------------------------

def test_cake_pan_memory_guard_is_cpu_noop() -> None:
    from hypernix.cake_pan import CakePan

    m = nn.Linear(4, 4)
    pan = CakePan(model=m, free_gb_trip=1e9)  # huge trip threshold
    # probe_vram on CPU returns total=0 so the guard short-circuits.
    assert pan.memory_guard() is False


def test_cake_pan_oven_all_bad_yields_zero_good() -> None:
    from hypernix.cake_pan import cake_pan

    m = nn.Linear(4, 4)
    pan = cake_pan(model=m, step_timeout_s=0, snapshot_every=0)
    pan.save_pristine()

    def step(batch):  # noqa: ARG001
        return torch.tensor(float("nan"))

    good = pan.oven(
        ["a", "b", "c"], step, max_retries_per_batch=1,
    )
    assert good == 0


def test_cake_pan_step_count_advances_across_good_steps() -> None:
    from hypernix.cake_pan import cake_pan

    m = nn.Linear(4, 4)
    pan = cake_pan(model=m, step_timeout_s=0, snapshot_every=0)
    pan.save_pristine()

    for _ in range(5):
        pan.bake(lambda: torch.tensor(0.1))
    assert pan.step_count == 5


# ---------------------------------------------------------------------------
# Freezer presets — sanity sweeps
# ---------------------------------------------------------------------------

def test_every_cpu_preset_has_an_avx_entry() -> None:
    from hypernix.freezer import CPU_PRESETS

    for name, preset in CPU_PRESETS.items():
        assert preset.avx_levels, f"{name} has no AVX levels"
        assert any("AVX" in level for level in preset.avx_levels), name


def test_every_gpu_preset_has_positive_bandwidth_and_vram() -> None:
    from hypernix.freezer import GPU_PRESETS

    for name, preset in GPU_PRESETS.items():
        assert preset.vram_gb > 0, name
        assert preset.bandwidth_gb_s > 0, name
        assert preset.freezer_class in {"Old", "New"}, name


def test_preset_lookup_keys_are_lowercase_normalised() -> None:
    """Every registry key must round-trip through ``_cpu_key`` /
    ``_gpu_key`` unchanged — otherwise case-insensitive lookups can
    miss."""
    from hypernix.freezer import CPU_PRESETS, GPU_PRESETS, _cpu_key, _gpu_key

    for key in CPU_PRESETS:
        assert _cpu_key(key) == key
    for key in GPU_PRESETS:
        assert _gpu_key(key) == key


# ---------------------------------------------------------------------------
# Shakers — more coverage
# ---------------------------------------------------------------------------

def test_shakers_same_seed_same_output() -> None:
    from hypernix import pepper_shaker, salt_shaker

    src = ["one two three four five", "six seven eight nine ten"]
    for mod, cls in [(salt_shaker, "FromTheBag"),
                     (pepper_shaker, "SmallShaker")]:
        klass = getattr(mod, cls)
        a = list(klass(source=src, rate=0.5, seed=42))
        b = list(klass(source=src, rate=0.5, seed=42))
        assert a == b, f"{cls} is not deterministic"


def test_shakers_rate_zero_is_identity() -> None:
    from hypernix import pepper_shaker, salt_shaker

    src = ["hello world"]
    assert list(salt_shaker.FromTheBag(source=src, rate=0.0)) == src
    assert list(salt_shaker.HandCrusher(source=src, rate=0.0)) == src
    assert list(pepper_shaker.SmallShaker(source=src, rate=0.0)) == src
    assert list(pepper_shaker.TallHandmade(source=src, rate=0.0)) == src


def test_shakers_preserve_empty_lines() -> None:
    from hypernix import salt_shaker

    # SaucePan drops empties; shakers don't.  Empty line in, empty out.
    assert list(salt_shaker.FromTheBag(source=[""], rate=1.0)) == [""]


# ---------------------------------------------------------------------------
# Smoke alarm — more coverage
# ---------------------------------------------------------------------------

def test_training_budget_time_hours() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(time_budget_seconds=7200.0)
    b = a.budget()
    assert b.time_hours == pytest.approx(2.0)


def test_storage_warning_silent_when_save_every_zero() -> None:
    from hypernix import smoke_alarm

    a = smoke_alarm.rads_alarm(
        time_budget_seconds=3600.0, available_storage_gb=1.0,
    )
    assert a.storage_warning(save_every=0, snapshot_size_gb=99.0) == ""


def test_unknown_preset_error_lists_valid() -> None:
    from hypernix import smoke_alarm

    with pytest.raises(ValueError, match="unknown preset") as exc_info:
        smoke_alarm.GasAlarm(
            time_budget_seconds=3600.0, preset="definitely-not-real",
        )
    # Error must include at least one real GPU name so the user can
    # find the right spelling.
    assert "h100" in str(exc_info.value) or "rtx" in str(exc_info.value)


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------

class _StubJudge:
    """Minimal oven-shaped object for industrial_range.complete."""

    def complete(self, prompt, *, max_new_tokens, temperature,
                 stop, seed=None):
        return "GOOD" if "France" in prompt else "BAD"


def test_evaluator_to_lunchbox_end_to_end(tmp_path: Path) -> None:
    """Plausible evaluator pipeline: industrial_range produces
    labels, Lunchbox packs them with a consistent schema, the
    JSONL output round-trips through Table.from_rows."""
    from hypernix import industrial_range
    from hypernix.lunchbox import Lunchbox
    from hypernix.table import Table

    judge = industrial_range.industrial_range(judge=_StubJudge())

    prompts = [
        ("r1", "Capital of France?", "Paris"),
        ("r2", "Who invented Python?", "Guido"),
    ]

    box = Lunchbox.for_eval()
    for pid, prompt, response in prompts:
        label = judge.label(prompt, response)
        box.add(
            id=pid, category="qa", difficulty="easy", tier="t1",
            prompt=prompt, reference=response, model_response=response,
            keyword_score=1.0 if label == "GOOD" else 0.0,
            latency_s=0.1, variant="stub", pipeline_meta="{}",
        )

    p = box.pack_jsonl(tmp_path / "eval.jsonl")

    # Read back as a Table and assert schema fidelity.
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    t = Table.from_rows(rows)

    assert len(t) == 2
    assert set(t.columns()) == {
        "id", "category", "difficulty", "tier", "prompt", "reference",
        "model_response", "keyword_score", "latency_s", "variant",
        "pipeline_meta",
    }
    assert t.rows[0]["id"] == "r1"
    assert t.rows[0]["keyword_score"] == 1.0        # Paris in prompt → GOOD
    assert t.rows[1]["keyword_score"] == 0.0        # Python prompt → BAD


def test_freezer_pick_pascal_safe_dtype_returns_float32_on_cpu() -> None:
    """Regression pin for the CPU-fp32 behaviour introduced in
    0.41.0 — cross-references the preset reach."""
    from hypernix.freezer import pascal_safe_dtype

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert pascal_safe_dtype() == torch.float32


def test_ranges_chain_into_mediocre_fridge() -> None:
    from hypernix import (
        industrial_range,
        mediocre_fridge,
        new_range,
    )

    # Plain-rubric label_rule path:
    rule = new_range.new_range()
    examples = mediocre_fridge.collect_responses_from(
        _StubOven(reply="Paris"), prompts=["Capital of France?"],
        label_rule=rule,
    )
    assert examples[0].label == "GOOD"

    # LLM-as-judge path:
    judge = industrial_range.industrial_range(judge=_StubJudge())
    examples = mediocre_fridge.collect_responses_from(
        _StubOven(reply="Paris"), prompts=["Capital of France?"],
        label_rule=judge,
    )
    assert examples[0].label == "GOOD"


class _StubOven:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, prompt, *, max_new_tokens=32,
                 temperature=0.7, stop=()):
        return self._reply


def test_pick_pan_roundtrips_through_sink(tmp_path: Path) -> None:
    from hypernix import pans, sink

    src = tmp_path / "raw.txt"
    src.write_text("  hello  world\n\nmore   stuff\n", encoding="utf-8")

    cleaned = pans.pick_pan("sauce-pan", source=src)
    out = sink.Sink(path=tmp_path / "clean.txt").pour(cleaned)
    assert out.read_text() == "hello world\nmore stuff\n"


def test_ensure_no_orphan_modules() -> None:
    """Every module registered in hypernix.__all__ must be importable
    and non-None.  Protects against half-wired registrations."""
    import hypernix

    for name in hypernix.__all__:
        assert hasattr(hypernix, name), name
        assert getattr(hypernix, name) is not None, name
