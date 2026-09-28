"""Tests for hypernix.tupperware."""
from __future__ import annotations

from pathlib import Path

import pytest

from hypernix.tupperware import RoundPlan, Tupperware, TupperwareConfig


def test_plan_returns_correct_round_count() -> None:
    box = Tupperware(TupperwareConfig(num_rounds=4))
    plans = box.plan(num_tokens=40_000, param_count=80_000_000)
    assert len(plans) == 4
    assert all(isinstance(p, RoundPlan) for p in plans)


def test_lr_decays_per_round() -> None:
    box = Tupperware(TupperwareConfig(num_rounds=3, base_lr=1e-3))
    plans = box.plan(num_tokens=10_000, total_steps=300)
    assert plans[0].lr > plans[1].lr > plans[2].lr


def test_split_file_writes_rounds(tmp_path: Path) -> None:
    src = tmp_path / "data.txt"
    src.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    out = tmp_path / "rounds"
    box = Tupperware(TupperwareConfig(num_rounds=2))
    paths = box.split_file(src, out)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    merged = "".join(p.read_text(encoding="utf-8") for p in paths)
    assert "line1" in merged and "line5" in merged


def test_eval_flags() -> None:
    box = Tupperware(TupperwareConfig(num_rounds=3, eval_each_round=True, eval_final_only=True))
    plans = box.plan(num_tokens=9000, total_steps=300)
    assert sum(p.eval_after for p in plans) == 1
    assert plans[-1].eval_after is True


def test_run_rounds_calls_eval(tmp_path: Path) -> None:
    box = Tupperware(TupperwareConfig(num_rounds=2, eval_each_round=True))
    plans = box.plan(num_tokens=1000, total_steps=100)
    paths = [tmp_path / f"r{i}.txt" for i in range(2)]
    for p in paths:
        p.write_text("data\n", encoding="utf-8")

    eval_calls: list[int] = []

    def train_fn(path: Path, plan: RoundPlan) -> str:
        return f"trained-{plan.round_index}"

    def eval_fn(path: Path, plan: RoundPlan, train_out: str) -> dict:
        eval_calls.append(plan.round_index)
        return {"loss": 1.0}

    results = box.run_rounds(plans, paths, train_fn, eval_fn)
    assert len(results) == 2
    assert len(eval_calls) == 2
    assert results[0]["eval"]["loss"] == 1.0


def test_invalid_num_rounds() -> None:
    with pytest.raises(ValueError):
        Tupperware(TupperwareConfig(num_rounds=0))
