"""Tests for v0.45 kitchen expansion:

* microwave tiers (defrost / low_zap / zap / high_zap / chat_zap + reheat)
* coffee_maker: +FrenchPressMaker, PercolatorMaker, ColdBrewMaker
* espresso_maker (4 tiers)
* blender / toaster / food_processor (4 tiers each)
* smoker (4 training-quality tiers)
* CLI `brew` subcommand
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cli_subprocess_env() -> dict[str, str]:
    """Near-hermetic environment for `hypernix.cli` subprocess tests.

    Deliberately minimal so these tests exercise the CLI's own env
    handling (e.g. HYPERNIX_AUTO_INSTALL=0) rather than whatever happens
    to be set in the surrounding shell/CI runner.

    On Windows, though, a handful of OS-level variables aren't optional:
    without SystemRoot (and friends) Winsock can't load its service
    provider table, so even `import asyncio` -- pulled in transitively by
    `torch` -> `unittest.mock` -> `asyncio` -- fails with
    "WinError 10106: The requested service provider could not be loaded
    or initialized" before any hypernix code runs. A hardcoded POSIX PATH
    also leaves Windows unable to resolve system DLLs. Carry over just
    enough of the real environment to let the interpreter and its
    dependencies boot on whatever platform we're on.
    """
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        "HYPERNIX_AUTO_INSTALL": "0",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
    }
    if sys.platform == "win32":
        for key in (
            "SystemRoot", "windir", "SystemDrive", "COMSPEC",
            "TEMP", "TMP", "USERPROFILE", "PATHEXT",
            "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
        ):
            if key in os.environ:
                env[key] = os.environ[key]
        system_root = env.get("SystemRoot", r"C:\Windows")
        env["PATH"] = os.pathsep.join([
            str(Path(sys.executable).parent),
            str(Path(sys.executable).parent / "Scripts"),
            f"{system_root}\\System32",
            system_root,
            f"{system_root}\\System32\\Wbem",
        ])
        env.setdefault("USERPROFILE", str(Path.home()))
    return env

# ---------------------------------------------------------------------------
# microwave — new tiers
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_snapshot(tmp_path: Path) -> Path:
    from hypernix import HyperNixConfig, init_from_scratch

    cfg = HyperNixConfig(
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16,
    )
    snap = tmp_path / "snap"
    init_from_scratch(str(snap), cfg, tokenizer_source=None, seed=0)
    return snap


def test_microwave_defrost_returns_oven(tiny_snapshot: Path) -> None:
    from hypernix import microwave

    oven = microwave.defrost(tiny_snapshot, device="cpu")
    assert hasattr(oven, "complete")


def test_microwave_low_zap_deterministic(tiny_snapshot: Path) -> None:
    from hypernix import microwave

    a = microwave.low_zap(tiny_snapshot, "hi", device="cpu")
    b = microwave.low_zap(tiny_snapshot, "hi", device="cpu")
    assert a == b  # temp=0, top_k=1 ⇒ identical


def test_microwave_high_zap_works(tiny_snapshot: Path) -> None:
    from hypernix import microwave

    out = microwave.high_zap(
        tiny_snapshot, "hi", max_new_tokens=2, device="cpu",
    )
    assert isinstance(out, str)


def test_microwave_reheat_extends_prior(tiny_snapshot: Path) -> None:
    from hypernix import microwave

    oven = microwave.defrost(tiny_snapshot, device="cpu")
    out = microwave.reheat(oven, "prior text ", " and more",
                           max_new_tokens=2, temperature=0.0)
    assert isinstance(out, str)


def test_microwave_tiers_registry() -> None:
    from hypernix import microwave

    assert set(microwave.TIERS) == {"defrost", "low", "standard", "high", "chat"}


# ---------------------------------------------------------------------------
# coffee_maker — new tiers + new type
# ---------------------------------------------------------------------------

def test_french_press_runs_batch() -> None:
    from hypernix import coffee_maker

    def fn1(): return "a"
    def fn2(): return "b"
    def fn3(): return "c"
    p = coffee_maker.french_press([fn1, fn2, fn3])
    out = p.plunge()
    assert [b.result for b in out] == ["a", "b", "c"]


def test_french_press_captures_exceptions() -> None:
    from hypernix import coffee_maker

    def boom():
        raise RuntimeError("no")

    p = coffee_maker.french_press([lambda: "ok", boom, lambda: "ok2"])
    hist = p.plunge()
    assert [b.ok for b in hist] == [True, False, True]


def test_percolator_iterates() -> None:
    from hypernix import coffee_maker

    def refine(prev):
        return (prev or "") + "a"

    p = coffee_maker.percolator(refine, seed_input="", max_cycles=3)
    final = p.percolate()
    assert final == "aaa"
    assert len(p.history) == 3


def test_percolator_convergence_short_circuits() -> None:
    from hypernix import coffee_maker

    def refine(prev):
        return (prev or "") + "x"

    p = coffee_maker.percolator(
        refine, seed_input="",
        max_cycles=10,
        convergence=lambda old, new: len(new) >= 3,
    )
    final = p.percolate()
    assert final == "xxx"
    assert len(p.history) == 3


def test_cold_brew_persists_checkpoint(tmp_path: Path) -> None:
    from hypernix import coffee_maker

    ck = tmp_path / "ck.json"

    def phase_fn(state, phase):
        state["phase"] = phase
        return state

    cb = coffee_maker.cold_brew(phase_fn, phases=3, checkpoint_path=ck)
    final = cb.brew()
    assert final == {"phase": 2}
    assert ck.exists()
    on_disk = json.loads(ck.read_text(encoding="utf-8"))
    assert on_disk["next_phase"] == 3


def test_cold_brew_resumes_from_checkpoint(tmp_path: Path) -> None:
    from hypernix import coffee_maker

    ck = tmp_path / "ck.json"
    ck.write_text(json.dumps({"state": {"phase": 1}, "next_phase": 2}), encoding="utf-8")
    calls = []

    def phase_fn(state, phase):
        calls.append(phase)
        state["phase"] = phase
        return state

    cb = coffee_maker.cold_brew(phase_fn, phases=4, checkpoint_path=ck)
    cb.brew()
    # Phases 2 and 3 only; earlier phases are skipped thanks to the
    # persisted next_phase=2.
    assert calls == [2, 3]


# ---------------------------------------------------------------------------
# espresso_maker
# ---------------------------------------------------------------------------

class _FakeOven:
    def __init__(self, reply: str = "42") -> None:
        self._reply = reply
        self.calls = 0

    def complete(self, prompt, *, max_new_tokens, temperature, top_k, top_p,
                 stop, seed):
        self.calls += 1
        return self._reply


def test_ristretto_one_sample_per_prompt() -> None:
    from hypernix import espresso_maker

    oven = _FakeOven()
    r = espresso_maker.ristretto(oven)
    shots = r.pull(["q1", "q2", "q3"])
    assert len(shots) == 3
    assert oven.calls == 3


def test_double_shot_takes_two_samples_per_prompt() -> None:
    from hypernix import espresso_maker

    oven = _FakeOven()
    d = espresso_maker.double_shot(oven, scorer=lambda p, o, r: len(o))
    shots = d.pull(["q1", "q2"])
    assert oven.calls == 4
    assert len(shots) == 2


def test_lungo_scorer_picks_best() -> None:
    from hypernix import espresso_maker

    class ReplyOven:
        def __init__(self):
            self.i = 0
            self.replies = ["short", "longer reply here"]

        def complete(self, *a, **k):
            r = self.replies[self.i % 2]
            self.i += 1
            return r

    oven = ReplyOven()
    lu = espresso_maker.lungo(oven, scorer=lambda p, o, r: len(o),
                              samples_per_prompt=2)
    shots = lu.pull(["q"])
    assert shots[0].output == "longer reply here"


def test_espresso_tiers_factory() -> None:
    from hypernix import espresso_maker

    for t in ["ristretto", "single-shot", "double-shot", "lungo"]:
        m = espresso_maker.espresso_maker(t, oven=_FakeOven())
        assert m.name


# ---------------------------------------------------------------------------
# blender
# ---------------------------------------------------------------------------

def test_hand_blender_concatenates() -> None:
    from hypernix import blender

    out = list(blender.HandBlender(sources=[["a", "b"], ["c", "d"]]))
    assert out == ["a", "b", "c", "d"]


def test_personal_blender_round_robin() -> None:
    from hypernix import blender

    out = list(blender.PersonalBlender(sources=[["a", "b", "c"], ["X", "Y"]]))
    assert out == ["a", "X", "b", "Y", "c"]


def test_countertop_blender_weighted() -> None:
    from hypernix import blender

    # Sample only the first chunk so neither source runs out first
    # and skews the ratio.  At 90/10 weights we expect ~90% "a".
    b = blender.CountertopBlender(
        sources=[["a"] * 10000, ["b"] * 10000],
        weights=[0.9, 0.1], seed=0,
    )
    sample = [next(iter(b)) for _ in range(500)]
    # Collect a fresh 500-element stream head; easier via the iterator.
    it = iter(b)
    sample = [next(it) for _ in range(500)]
    ratio = sample.count("a") / len(sample)
    assert ratio > 0.8


def test_high_power_blender_shuffles() -> None:
    from hypernix import blender

    out = list(blender.HighPowerBlender(
        sources=[[f"a{i}" for i in range(20)], [f"b{i}" for i in range(20)]],
        seed=0,
    ))
    assert len(out) == 40
    # Very unlikely still in block-order.
    assert out[:20] != [f"a{i}" for i in range(20)]


def test_blender_factory() -> None:
    from hypernix import blender

    for name in ["hand-blender", "personal-blender", "countertop-blender",
                 "high-power-blender"]:
        b = blender.blender(name, sources=[["x"]])
        assert b.sources


# ---------------------------------------------------------------------------
# toaster
# ---------------------------------------------------------------------------

def test_two_slice_toaster_pairs_lines() -> None:
    from hypernix import toaster

    out = list(toaster.TwoSliceToaster(source=["Q one", "A one", "Q two", "A two"]))
    assert out == ["Q: Q one\nA: A one", "Q: Q two\nA: A two"]


def test_four_slice_toaster_groups_by_four() -> None:
    from hypernix import toaster

    src = ["u1", "a1", "u2", "a2", "u3"]
    out = list(toaster.FourSliceToaster(source=src))
    assert len(out) == 1
    assert "<USER> u1" in out[0] and "<ASSISTANT> a2" in out[0]


def test_conveyor_toaster_template_per_line() -> None:
    from hypernix import toaster

    out = list(toaster.ConveyorToaster(
        source=["hello", "there"], template="<T>{line}</T>",
    ))
    assert out == ["<T>hello</T>", "<T>there</T>"]


def test_toaster_oven_wraps_documents(tmp_path: Path) -> None:
    from hypernix import toaster

    src = tmp_path / "d.txt"
    src.write_text("para one\nline two\n\npara two\n", encoding="utf-8")
    out = list(toaster.ToasterOven(source=src))
    assert len(out) == 2
    assert out[0].startswith("<DOCUMENT>") and out[0].endswith("</DOCUMENT>")


# ---------------------------------------------------------------------------
# food_processor
# ---------------------------------------------------------------------------

def test_chop_blade_splits_on_blank_line(tmp_path: Path) -> None:
    from hypernix import food_processor

    src = tmp_path / "doc.txt"
    src.write_text("block 1\nmore\n\nblock 2\n\nblock 3\n", encoding="utf-8")
    out = list(food_processor.ChopBlade(source=src))
    assert out == ["block 1\nmore", "block 2", "block 3"]


def test_slice_blade_fixed_length(tmp_path: Path) -> None:
    from hypernix import food_processor

    src = tmp_path / "long.txt"
    src.write_text("abcdefghij" * 5, encoding="utf-8")  # 50 chars
    out = list(food_processor.SliceBlade(source=src, slice_chars=20, overlap_chars=0))
    assert len(out) == 3                    # 20 + 20 + 10
    assert all(len(s) <= 20 for s in out)


def test_slice_blade_overlap(tmp_path: Path) -> None:
    from hypernix import food_processor

    src = tmp_path / "long.txt"
    src.write_text("abcdefghij" * 3, encoding="utf-8")  # 30 chars
    out = list(food_processor.SliceBlade(source=src, slice_chars=15, overlap_chars=5))
    # Step = 10, slices at 0, 10, 20 => 3 slices.
    assert len(out) == 3


def test_shred_blade_windows(tmp_path: Path) -> None:
    from hypernix import food_processor

    src = tmp_path / "w.txt"
    src.write_text(" ".join(f"tok{i}" for i in range(10)), encoding="utf-8")
    out = list(food_processor.ShredBlade(source=src, window_tokens=4, stride_tokens=2))
    # Tokens 0..3, 2..5, 4..7, 6..9, 8..9  => 5 windows at most
    assert 4 <= len(out) <= 6
    assert out[0].split()[:2] == ["tok0", "tok1"]


def test_puree_blade_collapses_whole_file(tmp_path: Path) -> None:
    from hypernix import food_processor

    src = tmp_path / "p.txt"
    src.write_text("hello   world\n\nmore text\n", encoding="utf-8")
    out = list(food_processor.PureeBlade(source=src))
    assert out == ["hello world more text"]


def test_food_processor_factory(tmp_path: Path) -> None:
    from hypernix import food_processor

    src = tmp_path / "x.txt"
    src.write_text("x\n", encoding="utf-8")
    for name in ["chop", "slice", "shred", "puree"]:
        fp = food_processor.food_processor(name, source=src)
        list(fp)


# ---------------------------------------------------------------------------
# smoker
# ---------------------------------------------------------------------------

def test_useable_smoker_forwards_kwargs(tiny_snapshot: Path, tmp_path: Path) -> None:
    from hypernix import old_oven, smoker

    oven = old_oven.preheat(local_dir=tiny_snapshot, device="cpu")
    ds = tmp_path / "corpus.txt"
    ds.write_text(("abcdefghij" * 32 + "\n") * 8, encoding="utf-8")

    s = smoker.useable_smoker(oven=oven, steps=2, batch_size=1,
                              context_length=16, log_every=1, quiet=True)
    out = s.smoke(ds, tmp_path / "trained")
    assert out.exists()
    assert (out / "config.json").exists()


def test_good_smoker_records_schedule(tiny_snapshot: Path, tmp_path: Path) -> None:
    from hypernix import old_oven, smoker

    oven = old_oven.preheat(local_dir=tiny_snapshot, device="cpu")
    ds = tmp_path / "corpus.txt"
    ds.write_text(("abcdefghij" * 32 + "\n") * 8, encoding="utf-8")

    s = smoker.good_smoker(oven=oven, steps=4, batch_size=1,
                           context_length=16, log_every=1, quiet=True)
    s.smoke(ds, tmp_path / "trained")
    assert s.history[-1]["tier"] == "GoodSmoker"
    assert "effective_lr" in s.history[-1]


def test_smoker_factory_tier_names() -> None:
    from hypernix import smoker

    class Dummy:
        pass

    for name in ["useable", "good", "commercial", "high-quality"]:
        s = smoker.smoker(name, oven=Dummy())
        assert s.name


# ---------------------------------------------------------------------------
# CLI brew subcommand
# ---------------------------------------------------------------------------

def test_cli_brew_runs_from_json(tiny_snapshot: Path, tmp_path: Path) -> None:
    recipe = {
        "local_dir": str(tiny_snapshot),
        "dataset": str(tmp_path / "c.txt"),
        "out_dir": str(tmp_path / "brewed"),
        "steps": 2, "batch_size": 1, "context_length": 16,
        "log_every": 1, "save_every": 0, "device": "cpu", "dtype": "float32",
        "seed": 0, "quiet": True,
    }
    Path(recipe["dataset"]).write_text(
        ("abcdefghij" * 32 + "\n") * 8, encoding="utf-8",
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    env = _cli_subprocess_env()
    cp = subprocess.run(
        [sys.executable, "-m", "hypernix.cli", "brew", str(recipe_path)],
        env=env, capture_output=True, text=True, timeout=180, check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert (tmp_path / "brewed" / "config.json").exists()


def test_cli_brew_rejects_bad_override(tmp_path: Path) -> None:
    recipe = {"dataset": "x", "out_dir": "y"}
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(recipe), encoding="utf-8")
    env = _cli_subprocess_env()
    cp = subprocess.run(
        [sys.executable, "-m", "hypernix.cli", "brew", str(rp), "--set", "bad_format"],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert cp.returncode != 0
    assert "Traceback" not in cp.stderr


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_v045_exports() -> None:
    import hypernix

    for mod in ["blender", "coffee_maker", "espresso_maker", "food_processor",
                "microwave", "smoker", "toaster"]:
        assert getattr(hypernix, mod) is not None
