"""Tests for v0.44 kitchen modules:

    pans / microwave / table / sink / instant_pot / coffee_maker /
    pressure_cooker.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# pans (5 tiers)
# ---------------------------------------------------------------------------

def test_frying_pan_passes_lines_through(tmp_path: Path) -> None:
    from hypernix import pans

    src = tmp_path / "a.txt"
    src.write_text("line 1   \nline 2\n", encoding="utf-8")
    out = list(pans.FryingPan(src))
    assert out == ["line 1", "line 2"]


def test_sauce_pan_collapses_whitespace_and_drops_empties(tmp_path: Path) -> None:
    from hypernix import pans

    src = tmp_path / "a.txt"
    src.write_text("  hello    world  \n\n\nanother line\n   \n", encoding="utf-8")
    out = list(pans.SaucePan(src))
    assert out == ["hello world", "another line"]


def test_skillet_chat_wraps_each_line(tmp_path: Path) -> None:
    from hypernix import pans

    p = pans.Skillet(source=["hello", "there"])
    out = list(p)
    assert out == ["<USER> hello\n<ASSISTANT>", "<USER> there\n<ASSISTANT>"]


def test_skillet_instruct_mode(tmp_path: Path) -> None:
    from hypernix import pans

    p = pans.Skillet(source=["do a thing"], mode="instruct")
    assert list(p) == ["### Instruction: do a thing\n### Response:"]


def test_grill_pan_dedupes_and_min_length(tmp_path: Path) -> None:
    from hypernix import pans

    lines = [
        "too short",        # < min_chars=8 is false since len==9, keep
        "short",            # <8, drop
        "exact same line",
        "exact same line",  # dup
        "another long line",
    ]
    p = pans.GrillPan(source=lines, min_chars=8)
    out = list(p)
    # "too short" (9 chars) kept; "short" (5) dropped; dup dropped.
    assert "too short" in out
    assert "exact same line" in out
    assert "another long line" in out
    assert out.count("exact same line") == 1
    assert "short" not in out


def test_wok_shuffles_and_optionally_reverses() -> None:
    from hypernix import pans

    source = [f"line {i}" for i in range(50)]
    p = pans.Wok(source=source, seed=0, reverse_ratio=0.0)
    shuffled = list(p)
    assert sorted(shuffled) == sorted(source)
    assert shuffled != source  # very unlikely to be identical

    # With reverse_ratio=1.0 every line flips word order.
    p2 = pans.Wok(source=["a b c"], seed=0, reverse_ratio=1.0)
    assert list(p2) == ["c b a"]


def test_pick_pan_factory() -> None:
    from hypernix import pans

    for name in ["frying-pan", "sauce-pan", "skillet", "grill-pan", "wok"]:
        p = pans.pick_pan(name, source=["x"])
        assert isinstance(p, pans.Pan)


# Regression: second positional arg used to silently bind to `name`
# (inherited from Pan.__init__), so ``Skillet(src, "instruct")`` would
# set ``name="instruct"`` and leave ``mode="chat"``.  `name` is now a
# ClassVar, so positional arg #2 is the first real field on each pan.

def test_skillet_second_positional_is_mode_not_name() -> None:
    from hypernix import pans

    s = pans.Skillet(["hi"], "instruct")
    assert s.mode == "instruct"
    assert s.name == "Skillet"   # unchanged — name is a ClassVar


def test_grill_pan_second_positional_is_min_chars() -> None:
    from hypernix import pans

    g = pans.GrillPan(["hello", "world", "hi", "hello"], 4)
    assert g.min_chars == 4
    # "hi" (2 chars) dropped; second "hello" dropped as duplicate.
    assert list(g) == ["hello", "world"]


def test_grill_pan_seen_is_not_in_init_signature() -> None:
    import inspect

    from hypernix import pans

    params = inspect.signature(pans.GrillPan).parameters
    assert "_seen" not in params


def test_pick_pan_unknown_tier_gives_useful_error() -> None:
    import pytest

    from hypernix import pans

    with pytest.raises(ValueError, match="unknown pan tier"):
        pans.pick_pan("microwave-pan", source=["x"])


def test_pick_pan_bad_kwarg_lists_valid_ones() -> None:
    import pytest

    from hypernix import pans

    with pytest.raises(ValueError, match="Skillet rejected"):
        pans.pick_pan("skillet", source=["x"], min_chars=16)


# Regression: v0.45 shipped without a context_length / max_chars knob,
# so a training script calling ``FryingPan(context_length=256)`` got
# a bare TypeError.  Every pan now accepts both as keyword-only fields.

def test_frying_pan_accepts_context_length() -> None:
    from hypernix import pans

    long_line = "x" * 10_000
    out = list(pans.FryingPan(source=[long_line], context_length=256))
    # context_length=256 → max_chars = 256 * 4 = 1024.
    assert len(out[0]) == 1024


def test_pan_max_chars_truncates() -> None:
    from hypernix import pans

    long_line = "x" * 10_000
    out = list(pans.SaucePan(source=[long_line], max_chars=500))
    assert len(out[0]) == 500


def test_pan_context_length_wins_over_max_chars() -> None:
    from hypernix import pans

    long_line = "x" * 10_000
    out = list(pans.FryingPan(
        source=[long_line], max_chars=99, context_length=50,
    ))
    # ctx=50 → 50 * 4 = 200, not 99.
    assert len(out[0]) == 200


def test_pan_length_cap_is_noop_when_unset() -> None:
    from hypernix import pans

    line = "x" * 10_000
    out = list(pans.FryingPan(source=[line]))
    assert len(out[0]) == 10_000


def test_wok_honors_context_length() -> None:
    from hypernix import pans

    lines = ["x" * 10_000, "y" * 10_000]
    out = list(pans.Wok(source=lines, seed=0, context_length=64))
    # 64 * 4 = 256
    assert all(len(s) <= 256 for s in out)


def test_pick_pan_forwards_context_length() -> None:
    from hypernix import pans

    src = ["abcd " * 1000]
    pan = pans.pick_pan("grill-pan", source=src, context_length=64)
    assert all(len(s) <= 256 for s in pan)


# ---------------------------------------------------------------------------
# microwave
# ---------------------------------------------------------------------------

def test_microwave_zap_on_local_snapshot(tmp_path: Path) -> None:
    from hypernix import HyperNixConfig, init_from_scratch, microwave

    cfg = HyperNixConfig(
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16,
    )
    snap = tmp_path / "snap"
    init_from_scratch(str(snap), cfg, tokenizer_source=None, seed=0)

    out = microwave.zap(snap, "hello", max_new_tokens=2, temperature=0.0,
                       device="cpu")
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def test_table_head_and_columns() -> None:
    from hypernix.table import Table

    t = Table.from_rows([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert len(t) == 2
    assert t.columns() == ["a", "b"]
    assert t.head(1) == [{"a": 1, "b": 2}]


def test_table_filter_and_select() -> None:
    from hypernix.table import Table

    t = Table.from_rows([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}])
    filt = t.filter(lambda r: r["x"] > 1)
    assert len(filt) == 2
    assert filt.select("y").rows == [{"y": "b"}, {"y": "c"}]


def test_table_from_training_log(tmp_path: Path) -> None:
    from hypernix.table import Table

    log = tmp_path / "train.log"
    log.write_text(
        "[hypernix.train] step 10/100  loss=2.3456  ppl=10.4\n"
        "[hypernix.train] step 20/100  loss=1.9870  ppl=7.30\n",
        encoding="utf-8",
    )
    t = Table.from_training_log(log)
    assert t.rows == [
        {"step": 10, "loss": 2.3456},
        {"step": 20, "loss": 1.987},
    ]


def test_table_from_judge_corpus(tmp_path: Path) -> None:
    from hypernix import mediocre_fridge
    from hypernix.table import Table

    corpus = tmp_path / "judge.txt"
    mediocre_fridge.synthesize_judge_corpus(n=8, out_path=corpus, seed=0)
    t = Table.from_judge_corpus(corpus)
    assert len(t) == 8
    assert set(t.columns()) == {"prompt", "response", "label"}
    assert all(r["label"] in {"GOOD", "BAD"} for r in t.rows)


def test_table_show_renders_without_crashing() -> None:
    from hypernix.table import Table

    t = Table.from_rows([{"step": 1, "loss": 2.5}, {"step": 2, "loss": 2.0}])
    s = t.show(n=10)
    assert "step" in s and "loss" in s
    # Empty table:
    assert Table([]).show() == "(empty table)"


# ---------------------------------------------------------------------------
# sink
# ---------------------------------------------------------------------------

def test_sink_write_appends(tmp_path: Path) -> None:
    from hypernix.sink import Sink

    s = Sink(path=tmp_path / "out.txt")
    s.write("line one")
    s.write("line two\n")   # already has newline
    assert (tmp_path / "out.txt").read_text() == "line one\nline two\n"


def test_sink_dedupe_skips_duplicates(tmp_path: Path) -> None:
    from hypernix.sink import Sink

    s = Sink(path=tmp_path / "u.txt", dedupe=True)
    assert s.write("hello") is True
    assert s.write("hello") is False
    assert s.write("world") is True
    assert (tmp_path / "u.txt").read_text() == "hello\nworld\n"


def test_sink_rotates_when_over_budget(tmp_path: Path) -> None:
    from hypernix.sink import Sink

    s = Sink(path=tmp_path / "r.txt", rotate_bytes=20)
    for i in range(5):
        s.write(f"block {i} extra")     # ~14 bytes each
    # At least one rotated file should exist.
    assert (tmp_path / "r.txt").exists()
    assert any(p.name.startswith("r.txt.") for p in tmp_path.iterdir())


def test_sink_pour_from_pan(tmp_path: Path) -> None:
    from hypernix.pans import SaucePan
    from hypernix.sink import Sink

    src = tmp_path / "in.txt"
    src.write_text("  line one \n\n  line  two  \n", encoding="utf-8")
    out = Sink(path=tmp_path / "out.txt").pour(SaucePan(src))
    assert out.read_text() == "line one\nline two\n"


# ---------------------------------------------------------------------------
# pressure_cooker
# ---------------------------------------------------------------------------

def test_pressure_cooker_lr_schedule() -> None:
    from hypernix import pressure_cooker

    p = torch.nn.Parameter(torch.zeros(2))
    opt = pressure_cooker.pressure_cooker(
        [p], peak_lr=1.0,
        warmup_steps=10, plateau_steps=10, cooldown_steps=10,
    )
    # Warmup: linear 0 -> 1.
    assert opt.scheduled_lr(0) == pytest.approx(0.1)
    assert opt.scheduled_lr(9) == pytest.approx(1.0)
    # Plateau: constant 1.
    assert opt.scheduled_lr(10) == 1.0
    assert opt.scheduled_lr(19) == 1.0
    # Cooldown: cosine 1 -> 0.
    assert opt.scheduled_lr(20) == pytest.approx(1.0)
    assert 0 < opt.scheduled_lr(25) < 1.0
    assert opt.scheduled_lr(30) == 0.0


def test_pressure_cooker_phase() -> None:
    from hypernix import pressure_cooker

    p = torch.nn.Parameter(torch.zeros(2))
    opt = pressure_cooker.pressure_cooker(
        [p], peak_lr=1.0, warmup_steps=5, plateau_steps=5, cooldown_steps=5,
    )
    assert opt.phase(0) == "warmup"
    assert opt.phase(5) == "plateau"
    assert opt.phase(10) == "cooldown"
    assert opt.phase(15) == "done"


def test_pressure_cooker_step_updates_params() -> None:
    from hypernix import pressure_cooker

    p = torch.nn.Parameter(torch.ones(4, 4))
    before = p.detach().clone()
    opt = pressure_cooker.pressure_cooker(
        [p], peak_lr=0.1, warmup_steps=0, plateau_steps=5, cooldown_steps=0,
    )
    loss = p.sum()
    loss.backward()
    opt.step()
    assert not torch.equal(p.detach(), before)


def test_pressure_cooker_lookahead_smooths() -> None:
    from hypernix import pressure_cooker

    p = torch.nn.Parameter(torch.zeros(4))
    opt = pressure_cooker.pressure_cooker(
        [p], peak_lr=1.0,
        warmup_steps=0, plateau_steps=10, cooldown_steps=0,
        lookahead_k=2, lookahead_alpha=0.5,
    )
    # After a few steps, state[p] must have the slow-weight tensor.
    for _ in range(4):
        p.grad = torch.ones_like(p)
        opt.step()
    state = opt.state[p]
    assert "slow" in state


def test_pressure_cooker_rejects_invalid_args() -> None:
    from hypernix import pressure_cooker

    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        pressure_cooker.pressure_cooker([p], peak_lr=0.0)
    with pytest.raises(ValueError):
        pressure_cooker.pressure_cooker([p], warmup_steps=-1)
    with pytest.raises(ValueError):
        pressure_cooker.pressure_cooker([p], lookahead_alpha=1.5)


# ---------------------------------------------------------------------------
# coffee_maker
# ---------------------------------------------------------------------------

def test_coffee_maker_runs_n_cycles() -> None:
    from hypernix import coffee_maker

    calls = {"n": 0}

    def brew():
        calls["n"] += 1
        return calls["n"]

    maker = coffee_maker.coffee_maker(brew, interval_seconds=0.0)
    history = maker.run(cycles=3)
    assert len(history) == 3
    assert all(h.ok for h in history)
    assert [h.result for h in history] == [1, 2, 3]


def test_coffee_maker_captures_exceptions() -> None:
    from hypernix import coffee_maker

    def flaky():
        raise RuntimeError("boom")

    maker = coffee_maker.coffee_maker(flaky, interval_seconds=0.0)
    history = maker.run(cycles=2)
    assert len(history) == 2
    assert all(not h.ok for h in history)
    assert all("RuntimeError" in h.error for h in history)


def test_coffee_maker_summary() -> None:
    from hypernix import coffee_maker

    n = {"i": 0}

    def brew():
        n["i"] += 1
        if n["i"] == 2:
            raise ValueError("nope")

    maker = coffee_maker.coffee_maker(brew, interval_seconds=0.0)
    maker.run(cycles=3)
    summary = maker.summary()
    assert summary["cycles"] == 3
    assert summary["failed"] == 1


def test_coffee_maker_stop_is_cooperative() -> None:
    from hypernix import coffee_maker

    # A brew that asks to stop mid-run.
    maker: coffee_maker.CoffeeMaker | None = None

    def brew():
        assert maker is not None
        if maker.history and len(maker.history) >= 1:
            maker.stop()
        return "coffee"

    maker = coffee_maker.coffee_maker(brew, interval_seconds=0.0)
    history = maker.run(cycles=10)
    # Should stop after the second brew's stop() takes effect.
    assert len(history) < 10
    assert len(history) >= 1


# ---------------------------------------------------------------------------
# instant_pot
# ---------------------------------------------------------------------------

def test_instant_pot_brew_happy_path(tmp_path: Path) -> None:
    from hypernix import HyperNixConfig, init_from_scratch, instant_pot

    cfg = HyperNixConfig(
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16,
    )
    snap = tmp_path / "snap"
    init_from_scratch(str(snap), cfg, tokenizer_source=None, seed=0)

    dataset = tmp_path / "corpus.txt"
    dataset.write_text(("abcdefghij" * 32 + "\n") * 8, encoding="utf-8")

    trained = instant_pot.brew({
        "local_dir": str(snap),
        "dataset": str(dataset),
        "out_dir": str(tmp_path / "trained"),
        "steps": 2,
        "batch_size": 1,
        "context_length": 16,
        "log_every": 1,
        "save_every": 0,
        "device": "cpu",
        "dtype": "float32",
        "seed": 0,
        "quiet": True,
    })
    assert Path(trained).exists()
    assert (Path(trained) / "config.json").exists()


def test_instant_pot_rejects_missing_keys() -> None:
    from hypernix import instant_pot

    with pytest.raises(KeyError, match="dataset"):
        instant_pot.brew({"out_dir": "/tmp/x"})
    with pytest.raises(KeyError, match="out_dir"):
        instant_pot.brew({"dataset": "/tmp/y"})


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_v044_exports() -> None:
    import hypernix

    for mod in ["pans", "microwave", "table", "sink", "instant_pot",
                "coffee_maker", "pressure_cooker"]:
        assert getattr(hypernix, mod) is not None, mod
