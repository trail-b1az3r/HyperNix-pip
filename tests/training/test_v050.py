"""Tests for v0.50.0:

* New modules: whisk, cutting_board, apron, recipe_book.
* Pass 1 bug fixes: pressure_cooker private-API graceful fallback,
  deep_fryer torch-RNG reproducibility, food_processor SliceBlade
  overlap validation, industrial_range pairwise tie parsing.
* Pass 2 bug fixes: instant_pot missing-dataset error message,
  microwave config.json gate, cake_pan timeout-handler rollback.
* Pass 3: regression tests pinning the pre-fix behaviour was
  actually wrong and the post-fix behaviour is right.
"""
from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# whisk
# ---------------------------------------------------------------------------

def _state(values: list[float]) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(values, dtype=torch.float32)}


def test_whisk_swa_uniform_mean() -> None:
    from hypernix import whisk

    out = whisk.swa_average([
        _state([1.0, 2.0]),
        _state([3.0, 4.0]),
        _state([5.0, 6.0]),
    ])
    torch.testing.assert_close(out["w"], torch.tensor([3.0, 4.0]))


def test_whisk_ema_weights_later_more() -> None:
    from hypernix import whisk

    out = whisk.ema(
        [_state([0.0]), _state([1.0])], decay=0.5,
    )
    # weights = [0.5, 1.0] / 1.5 = [1/3, 2/3]
    # mean = 0/3 + 2/3 = 2/3
    torch.testing.assert_close(out["w"], torch.tensor([2.0 / 3.0]))


def test_whisk_ema_decay_zero_returns_last() -> None:
    from hypernix import whisk

    out = whisk.ema(
        [_state([10.0]), _state([20.0])], decay=0.0,
    )
    torch.testing.assert_close(out["w"], torch.tensor([20.0]))


def test_whisk_geometric_mean() -> None:
    from hypernix import whisk

    out = whisk.geometric_mean([
        _state([4.0]),
        _state([9.0]),
    ])
    torch.testing.assert_close(out["w"], torch.tensor([6.0]), rtol=1e-3, atol=1e-3)


def test_whisk_strict_mismatch_raises() -> None:
    from hypernix import whisk

    a = _state([1.0])
    b = {"w2": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="mismatched keys"):
        whisk.swa_average([a, b], strict=True)


def test_whisk_non_strict_intersects() -> None:
    from hypernix import whisk

    a = {"w": torch.tensor([1.0]), "extra": torch.tensor([99.0])}
    b = {"w": torch.tensor([3.0])}
    out = whisk.swa_average([a, b], strict=False)
    assert "extra" not in out
    torch.testing.assert_close(out["w"], torch.tensor([2.0]))


def test_whisk_integer_tensors_taken_from_first() -> None:
    """Integer buffers (e.g. token-ID lookups) shouldn't be averaged."""
    from hypernix import whisk

    a = {"ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    b = {"ids": torch.tensor([99, 99, 99], dtype=torch.long)}
    out = whisk.swa_average([a, b])
    assert torch.equal(out["ids"], torch.tensor([1, 2, 3], dtype=torch.long))


def test_whisk_factory_dispatches_modes() -> None:
    from hypernix import whisk

    items = [_state([0.0]), _state([2.0])]
    assert whisk.whisk(items, mode="swa")["w"].item() == 1.0
    assert whisk.whisk(items, mode="ema", decay=0.0)["w"].item() == 2.0
    with pytest.raises(ValueError, match="unknown whisk mode"):
        whisk.whisk(items, mode="not-real")


def test_whisk_load_from_path(tmp_path: Path) -> None:
    from hypernix import whisk

    p1 = tmp_path / "ckpt-1.pt"
    p2 = tmp_path / "ckpt-2.pt"
    torch.save(_state([1.0]), p1)
    torch.save(_state([3.0]), p2)
    out = whisk.swa_average([p1, p2])
    torch.testing.assert_close(out["w"], torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# cutting_board
# ---------------------------------------------------------------------------

def test_cutting_board_default_split_ratios() -> None:
    from hypernix.cutting_board import cutting_board

    rows = [f"row {i}" for i in range(100)]
    slices = cutting_board(rows, train=0.8, val=0.1, test=0.1, seed=0)
    assert len(slices["train"]) == 80
    assert len(slices["val"]) == 10
    assert len(slices["test"]) == 10
    # No duplicates across slices.
    assert set(slices["train"]) | set(slices["val"]) | set(slices["test"]) == set(rows)


def test_cutting_board_renormalises_ratios() -> None:
    from hypernix.cutting_board import cutting_board

    rows = list(range(20))
    slices = cutting_board([str(r) for r in rows], train=4, val=1, test=0, seed=0)
    # 4:1:0 → train_r=0.8, val_r=0.2, test_r=0.0
    assert len(slices["train"]) == 16
    assert len(slices["val"]) == 4
    assert len(slices["test"]) == 0


def test_cutting_board_deterministic_with_seed() -> None:
    from hypernix.cutting_board import cutting_board

    rows = [str(i) for i in range(50)]
    a = cutting_board(rows, seed=7)
    b = cutting_board(rows, seed=7)
    assert a == b


def test_cutting_board_writes_files(tmp_path: Path) -> None:
    from hypernix.cutting_board import CuttingBoard

    rows = [f"line {i}" for i in range(20)]
    paths = CuttingBoard(seed=0).slice_to_files(rows, tmp_path)
    assert paths["train"].exists() and paths["val"].exists() and paths["test"].exists()
    assert sum(1 for _ in paths["train"].read_text().splitlines()) == 16


def test_stratified_split_preserves_class_ratio() -> None:
    from hypernix.cutting_board import stratified_split

    records = (
        [{"id": i, "label": "GOOD"} for i in range(80)]
        + [{"id": 100 + i, "label": "BAD"} for i in range(20)]
    )
    s = stratified_split(records, train=0.8, val=0.1, test=0.1, seed=0)
    # Original distribution: 80% GOOD / 20% BAD.  Stratified split
    # should give each slice the same ratio (within rounding).
    for split in ("train", "val", "test"):
        if not s[split]:
            continue
        good = sum(1 for r in s[split] if r["label"] == "GOOD")
        ratio = good / len(s[split])
        assert 0.7 < ratio < 0.9, f"{split} ratio {ratio} not ~0.8"


def test_cutting_board_no_shuffle_preserves_order() -> None:
    from hypernix.cutting_board import CuttingBoard

    rows = [f"r{i}" for i in range(10)]
    slices = CuttingBoard(shuffle=False, seed=0).slice(rows)
    # No shuffle means train is the first 80% of input order.
    assert slices["train"] == rows[:8]


def test_cutting_board_rejects_negative_ratio() -> None:
    from hypernix.cutting_board import CuttingBoard

    with pytest.raises(ValueError, match=">="):
        CuttingBoard(train_ratio=-0.1)


# ---------------------------------------------------------------------------
# apron
# ---------------------------------------------------------------------------

def test_apron_restores_pre_call_state() -> None:
    """Bug fix: apron(seed=...) must snapshot BEFORE seeding so the
    restore lands the caller back where they started, not at the
    seeded checkpoint."""
    from hypernix.apron import apron

    random.seed(0)
    a = random.random()
    b_before = random.random()
    # Reset to the same starting point.
    random.seed(0)
    a_again = random.random()
    assert a_again == a
    with apron(seed=42):
        # Some inner work consumes the seeded RNG.
        random.random()
        random.random()
    # After the apron exits, the next .random() should match the
    # b_before value — the pre-apron state was restored.
    assert random.random() == b_before


def test_apron_no_seed_is_pure_save_restore() -> None:
    from hypernix.apron import apron

    random.seed(0)
    seq_no_apron = [random.random() for _ in range(3)]

    random.seed(0)
    out: list[float] = []
    out.append(random.random())
    with apron():
        # Consume the RNG inside; should not affect outside.
        for _ in range(10):
            random.random()
    out.append(random.random())
    out.append(random.random())
    assert out == seq_no_apron


def test_apron_torch_rng_restored() -> None:
    from hypernix.apron import apron

    torch.manual_seed(0)
    torch.randn(1)                              # discarded — sets the baseline
    b_after = torch.randn(1).item()

    torch.manual_seed(0)
    torch.randn(1)
    with apron(seed=99):
        for _ in range(10):
            torch.randn(1)
    assert torch.randn(1).item() == pytest.approx(b_after)


def test_apron_object_form() -> None:
    from hypernix.apron import Apron

    random.seed(0)
    captured = Apron.snapshot()
    random.random()
    random.random()
    captured.restore()
    # Restored to seed-0 starting state.
    assert random.random() == pytest.approx(0.8444218515250481)


# ---------------------------------------------------------------------------
# recipe_book
# ---------------------------------------------------------------------------

def test_recipe_book_add_get_remove() -> None:
    from hypernix.recipe_book import RecipeBook

    book = RecipeBook()
    book.add("test", {"steps": 100})
    assert "test" in book
    assert book.get("test")["steps"] == 100
    book.remove("test")
    assert "test" not in book


def test_recipe_book_get_unknown_raises() -> None:
    from hypernix.recipe_book import RecipeBook

    book = RecipeBook()
    with pytest.raises(KeyError, match="unknown recipe"):
        book.get("not-defined")


def test_recipe_book_add_rejects_non_dict() -> None:
    from hypernix.recipe_book import RecipeBook

    with pytest.raises(TypeError, match="dict"):
        RecipeBook().add("name", "not a dict")  # type: ignore[arg-type]


def test_recipe_book_save_load_roundtrip(tmp_path: Path) -> None:
    from hypernix.recipe_book import RecipeBook

    book = RecipeBook()
    book.add("a", {"steps": 100, "lr": 1e-3})
    book.add("b", {"steps": 200})
    p = book.save(tmp_path / "recipes.json")

    other = RecipeBook.load(p)
    assert sorted(other.names()) == ["a", "b"]
    assert other.get("a")["lr"] == 1e-3


def test_recipe_book_from_builtins_has_known_names() -> None:
    from hypernix.recipe_book import RecipeBook

    book = RecipeBook.from_builtins()
    assert "evaluator-quick" in book
    assert "ftune-pascal" in book
    assert "nightly-coldbrew" in book


def test_recipe_book_cook_dispatches_to_instant_pot(tmp_path: Path) -> None:
    """Use the recipe_book.cook() dispatch for an instant_pot recipe.
    instant_pot.brew is patched so we just verify the recipe was
    forwarded with the overrides applied."""
    import hypernix.recipe_book as rb_mod

    book = rb_mod.RecipeBook()
    book.add("simple", {
        "kind": "instant_pot",
        "dataset": "x.txt", "out_dir": "y", "steps": 100,
    })

    fake_brew = MagicMock(return_value=tmp_path)
    with patch("hypernix.instant_pot.brew", fake_brew):
        book.cook("simple", steps=200)

    args = fake_brew.call_args.args[0]
    assert args["dataset"] == "x.txt"
    assert args["steps"] == 200
    assert "kind" not in args


def test_recipe_book_cook_unknown_kind() -> None:
    from hypernix.recipe_book import RecipeBook

    book = RecipeBook()
    book.add("weird", {"kind": "unsupported_kitchen_appliance"})
    with pytest.raises(ValueError, match="unknown recipe kind"):
        book.cook("weird")


# ---------------------------------------------------------------------------
# Pass 1 bug-fix regressions
# ---------------------------------------------------------------------------

def test_food_processor_slice_blade_rejects_overlap_ge_slice(tmp_path: Path) -> None:
    """Pass 1: SliceBlade with overlap_chars >= slice_chars used to
    silently emit duplicate windows.  Now it raises."""
    from hypernix import food_processor as fp

    src = tmp_path / "x.txt"
    src.write_text("a" * 100, encoding="utf-8")

    with pytest.raises(ValueError, match="overlap_chars"):
        list(fp.SliceBlade(source=src, slice_chars=10, overlap_chars=10))
    with pytest.raises(ValueError, match="overlap_chars"):
        list(fp.SliceBlade(source=src, slice_chars=10, overlap_chars=15))
    with pytest.raises(ValueError, match=">= 0"):
        list(fp.SliceBlade(source=src, slice_chars=10, overlap_chars=-1))


def test_industrial_range_pairwise_tie_anywhere() -> None:
    """Pass 1: 'I think it's a tie' used to parse as B (the 't' in
    'it' triggered the leading-T check after stripping then 'B' was
    matched).  Now any 'tie' / 'tied' / 'equal' anywhere wins."""
    from hypernix.industrial_range import IndustrialRange

    assert IndustrialRange._parse_pairwise("Tie") == "T"
    assert IndustrialRange._parse_pairwise("It's a tie") == "T"
    assert IndustrialRange._parse_pairwise("they are tied") == "T"
    assert IndustrialRange._parse_pairwise("they look equal to me") == "T"
    # Plain A / B still work.
    assert IndustrialRange._parse_pairwise("A is better") == "A"
    assert IndustrialRange._parse_pairwise("B wins") == "B"


def test_deep_fryer_reproducible_across_global_rng_states() -> None:
    """Pass 1: deep_fryer used to pull noise from the global torch
    RNG, so two calls with the same seed but different global states
    produced different noise.  Now each parameter has a torch.Generator
    seeded from self.seed + name."""
    from hypernix.deep_fryer import LightFry

    def setup():
        torch.manual_seed(0)
        return nn.Linear(8, 8)

    m_a = setup()
    LightFry(model=m_a, fraction=0.5, noise_std=1.0, seed=42).fry()
    snap_a = m_a.weight.detach().clone()

    m_b = setup()
    # Mess with the global torch RNG between setup and fry — old
    # implementation would now produce different results.
    torch.randn(1000)
    LightFry(model=m_b, fraction=0.5, noise_std=1.0, seed=42).fry()
    snap_b = m_b.weight.detach().clone()

    torch.testing.assert_close(snap_a, snap_b, rtol=1e-5, atol=1e-5)


def test_pressure_cooker_falls_back_when_private_api_missing() -> None:
    """Pass 1: when torch.optim._functional.adamw is unavailable or
    has the wrong signature, the foreach path should fall back to the
    scalar path silently rather than raising."""
    from hypernix.pressure_cooker import PressureCooker

    m = nn.Linear(4, 4)
    opt = PressureCooker(
        m.parameters(), peak_lr=0.1, warmup_steps=0,
        plateau_steps=10, cooldown_steps=0, foreach=True,
    )
    before = m.weight.detach().clone()

    # Patch the private import to raise — simulates a torch where
    # the symbol moved or got renamed.
    fake_optim_functional = MagicMock()
    fake_optim_functional.adamw.side_effect = TypeError(
        "simulated signature change",
    )
    with patch.dict(
        "sys.modules", {"torch.optim._functional": fake_optim_functional},
    ):
        m(torch.randn(1, 4)).sum().backward()
        opt.step()

    # Weights still moved — fallback worked.
    assert not torch.equal(before, m.weight)


# ---------------------------------------------------------------------------
# Pass 2 bug-fix regressions
# ---------------------------------------------------------------------------

def test_instant_pot_fast_fails_on_missing_dataset(tmp_path: Path) -> None:
    """Pass 2: instant_pot.brew now raises FileNotFoundError with the
    actual path when the dataset doesn't exist, instead of letting
    train() raise a deeper error twenty stack frames down."""
    from hypernix import instant_pot

    with pytest.raises(FileNotFoundError, match="does not exist"):
        instant_pot.brew({
            "dataset": str(tmp_path / "missing.txt"),
            "out_dir": str(tmp_path / "out"),
        })


def test_microwave_treats_dir_without_config_json_as_repo(tmp_path: Path) -> None:
    """Pass 2: a string that happens to match a same-named local
    directory is no longer treated as a snapshot path unless that
    directory actually contains a config.json.  Otherwise short-name
    lookups silently shadowed."""
    from hypernix import microwave

    # Make a directory that exists but isn't a snapshot.
    not_a_snap = tmp_path / "junk_dir_named_like_a_repo"
    not_a_snap.mkdir()

    # Patch old_oven.preheat so we can inspect what microwave passed.
    fake_preheat = MagicMock()
    with patch("hypernix.old_oven.preheat", fake_preheat):
        microwave._preheat(
            str(not_a_snap), device="cpu", dtype="float32", quiet=True,
        )
    kwargs = fake_preheat.call_args.kwargs
    # Without a config.json, microwave treats the string as a repo id
    # — so local_dir stays None.
    assert kwargs["local_dir"] is None
    assert kwargs["repo_id"] == str(not_a_snap)


def test_microwave_recognises_real_snapshot(tmp_path: Path) -> None:
    """Companion to the previous test: an existing dir that *does*
    contain a config.json is correctly treated as a snapshot path."""
    from hypernix import microwave

    real = tmp_path / "real-snap"
    real.mkdir()
    (real / "config.json").write_text("{}")

    fake_preheat = MagicMock()
    with patch("hypernix.old_oven.preheat", fake_preheat):
        microwave._preheat(
            str(real), device="cpu", dtype="float32", quiet=True,
        )
    kwargs = fake_preheat.call_args.kwargs
    assert kwargs["local_dir"] == real


def test_cake_pan_timeout_handler_rolls_back() -> None:
    """Pass 2: when the SIGALRM fires mid-step, the model state can
    be partly updated.  The timeout handler now calls roll_back()
    before raising BakeOff so corrupted state doesn't survive."""
    import signal

    from hypernix import cake_pan
    from hypernix.cake_pan import BakeOff

    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM only on POSIX")

    m = nn.Linear(4, 4)
    pan = cake_pan.cake_pan(model=m, step_timeout_s=1, snapshot_every=0)
    pan.save_pristine()
    orig = m.weight.detach().clone()

    def slow_step():
        # Corrupt the weight, then sleep past the timeout.
        with torch.no_grad():
            m.weight.zero_().add_(99.0)
        import time
        time.sleep(3.0)
        return torch.tensor(0.0)

    with pytest.raises(BakeOff, match="exceeded"):
        pan.bake(slow_step)

    # The corrupting "+99" was rolled back by the timeout handler.
    assert torch.equal(orig, m.weight)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_v050_modules_in_package() -> None:
    import hypernix

    for name in ("whisk", "cutting_board", "apron", "recipe_book"):
        assert getattr(hypernix, name) is not None, name
