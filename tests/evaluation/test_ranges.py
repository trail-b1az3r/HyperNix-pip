"""Tests for new_range / old_range / industrial_range."""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# new_range
# ---------------------------------------------------------------------------

def test_new_range_default_passes_well_formed_responses() -> None:
    from hypernix import new_range

    r = new_range.new_range()
    assert r("Capital of France?", "Paris") == new_range.LABEL_GOOD
    assert r("2 + 2 = ?", "4") == new_range.LABEL_GOOD


def test_new_range_flags_empty() -> None:
    from hypernix import new_range

    r = new_range.new_range()
    assert r("Anything?", "") == new_range.LABEL_BAD
    assert r("Anything?", "   ") == new_range.LABEL_BAD
    assert r.last_failed_rule == "is_empty"


def test_new_range_flags_refusal() -> None:
    from hypernix import new_range

    r = new_range.new_range()
    assert r("How do I bake bread?", "I'm sorry, I cannot help with that.") == new_range.LABEL_BAD
    assert r.last_failed_rule == "is_refusal"


def test_new_range_flags_math_without_digits() -> None:
    from hypernix import new_range

    r = new_range.new_range()
    assert r("How many planets are in the solar system?", "lots of them") == new_range.LABEL_BAD
    assert r.last_failed_rule == "math_lacks_digit"
    # But same prompt with a digit answer passes:
    assert r("How many planets are in the solar system?", "8") == new_range.LABEL_GOOD


def test_new_range_flags_repetition() -> None:
    from hypernix import new_range

    r = new_range.new_range()
    assert r("Hello?", "aaaaaaaaaaaa") == new_range.LABEL_BAD
    assert r.last_failed_rule == "is_repetition"


def test_new_range_custom_rules() -> None:
    from hypernix import new_range

    def starts_with_no(prompt: str, response: str) -> bool:
        return response.lower().startswith("no")

    r = new_range.new_range(rules=[starts_with_no])
    assert r("Anything?", "no thanks") == new_range.LABEL_BAD
    assert r("Anything?", "yes please") == new_range.LABEL_GOOD


def test_new_range_first_fail_short_circuits() -> None:
    """When multiple rules would fire, the first one wins for last_failed_rule."""
    from hypernix import new_range

    r = new_range.new_range()
    # Empty AND a refusal-shaped response would both fire; empty is first.
    assert r("Q?", "") == new_range.LABEL_BAD
    assert r.last_failed_rule == "is_empty"


# ---------------------------------------------------------------------------
# old_range
# ---------------------------------------------------------------------------

def test_old_range_default_labels_known_good() -> None:
    from hypernix import old_range

    r = old_range.old_range()
    assert r("Capital of France?", "Paris") == old_range.LABEL_GOOD


def test_old_range_default_labels_empty_bad() -> None:
    from hypernix import old_range

    r = old_range.old_range()
    assert r("Q?", "") == old_range.LABEL_BAD


def test_old_range_breakdown_explains_why() -> None:
    from hypernix import old_range

    r = old_range.old_range()
    label, agg, breakdown = r.label_with_breakdown(
        "How many planets in the solar system?",
        "lots of them",
    )
    # The math rule scored 0.0 on a math-y prompt → fatal short-circuit
    # to BAD even though length / refusal / repetition all passed.
    assert label == old_range.LABEL_BAD
    names = [name for name, _, _ in breakdown]
    assert "math_digit_score" in names
    math_score = next(s for name, s, _ in breakdown if name == "math_digit_score")
    assert math_score == 0.0


def test_old_range_overlap_uses_references() -> None:
    from hypernix import old_range

    r = old_range.old_range(
        references={"What is the capital of France?": "Paris is the capital."},
    )
    good = r("What is the capital of France?", "Paris is the capital city of France.")
    bad = r("What is the capital of France?", "London is in England.")
    assert good == old_range.LABEL_GOOD
    assert bad == old_range.LABEL_BAD


def test_old_range_keywords_required() -> None:
    from hypernix import old_range

    r = old_range.old_range(
        keywords={"List a primary color.": ["red", "blue", "yellow"]},
        threshold=0.7,   # tighten the cutoff so partial matches fail
    )
    # All three keywords present:
    assert r("List a primary color.", "red, blue, and yellow") == old_range.LABEL_GOOD
    # None present → keyword score 0:
    assert r("List a primary color.", "purple and orange") == old_range.LABEL_BAD


def test_old_range_threshold_affects_label() -> None:
    from hypernix import old_range

    # Only one rule, returning a constant 0.4. With threshold=0.5 it's
    # BAD; with threshold=0.3 it's GOOD.
    def fixed(prompt: str, response: str, ctx: dict) -> tuple[float, str]:
        return 0.4, "fixed"

    r = old_range.OldRange(rules=[fixed], threshold=0.5)
    assert r.label("p", "r") == old_range.LABEL_BAD
    r2 = old_range.OldRange(rules=[fixed], threshold=0.3)
    assert r2.label("p", "r") == old_range.LABEL_GOOD


def test_old_range_weights_change_aggregate() -> None:
    from hypernix import old_range

    def good_rule(p, r, c):
        return 1.0, "good"

    def bad_rule(p, r, c):
        return 0.0, "bad"

    good_rule.__name__ = "good_rule"
    bad_rule.__name__ = "bad_rule"

    # fatal_rules=() opts out of the "0.0 short-circuits to BAD" default
    # so the weighted mean alone decides the label.
    r = old_range.OldRange(
        rules=[good_rule, bad_rule],
        weights={"good_rule": 9.0, "bad_rule": 1.0},
        threshold=0.5,
        fatal_rules=(),
    )
    # Weighted mean = (9*1 + 1*0)/10 = 0.9 -> GOOD.
    assert r.label("p", "r") == old_range.LABEL_GOOD


# ---------------------------------------------------------------------------
# industrial_range
# ---------------------------------------------------------------------------

class _FakeJudge:
    """Stand-in for a CodeOven; records calls and returns scripted replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    def complete(self, prompt: str, *, max_new_tokens: int,
                 temperature: float, stop: tuple[str, ...]) -> str:
        self.calls.append(prompt)
        return self._replies.pop(0)


def test_industrial_range_pointwise_good() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["GOOD"])
    r = industrial_range.industrial_range(judge=judge)
    assert r("Capital of France?", "Paris") == industrial_range.LABEL_GOOD
    assert len(judge.calls) == 1
    # Rubric is in the judge prompt:
    assert "PROMPT: Capital of France?" in judge.calls[0]
    assert "RESPONSE: Paris" in judge.calls[0]


def test_industrial_range_pointwise_bad() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["BAD"])
    r = industrial_range.industrial_range(judge=judge)
    assert r("Capital of France?", "London") == industrial_range.LABEL_BAD


def test_industrial_range_unparseable_falls_back_to_bad() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["who knows"])
    r = industrial_range.industrial_range(judge=judge)
    assert r("Q?", "R") == industrial_range.LABEL_BAD


def test_industrial_range_caches_repeats() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["GOOD"])  # only one reply available
    r = industrial_range.industrial_range(judge=judge)
    # First call hits the judge; second is served from cache.
    assert r("Q?", "R") == industrial_range.LABEL_GOOD
    assert r("Q?", "R") == industrial_range.LABEL_GOOD
    assert len(judge.calls) == 1


def test_industrial_range_pairwise() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["A is better", "B wins"])
    r = industrial_range.industrial_range(judge=judge)
    assert r.compare("Q?", "answer one", "answer two") == "A"
    assert r.compare("Q?", "answer one", "answer two-ish") == "B"


def test_industrial_range_pairwise_tie() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["Tie"])
    r = industrial_range.industrial_range(judge=judge)
    assert r.compare("Q?", "x", "y") == "T"


def test_industrial_range_label_batch() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["GOOD", "BAD", "GOOD"])
    r = industrial_range.industrial_range(judge=judge)
    out = r.label_batch([("p1", "r1"), ("p2", "r2"), ("p3", "r3")])
    assert out == ["GOOD", "BAD", "GOOD"]


def test_industrial_range_extracts_first_keyword_when_both_appear() -> None:
    from hypernix import industrial_range

    judge = _FakeJudge(["GOOD because the BAD label would be wrong"])
    r = industrial_range.industrial_range(judge=judge)
    assert r("Q?", "R") == industrial_range.LABEL_GOOD


# ---------------------------------------------------------------------------
# Cross-module smoke
# ---------------------------------------------------------------------------

def test_ranges_drop_in_as_label_rule() -> None:
    """All three ranges have the same (prompt, response) -> str signature."""
    from hypernix import industrial_range, new_range, old_range

    nr = new_range.new_range()
    or_ = old_range.old_range()
    ir = industrial_range.industrial_range(judge=_FakeJudge(["GOOD"]))

    for r in (nr, or_, ir):
        out = r("Capital of France?", "Paris")
        assert out in {"GOOD", "BAD"}


def test_ranges_exposed_on_package() -> None:
    import hypernix

    assert hypernix.new_range is not None
    assert hypernix.old_range is not None
    assert hypernix.industrial_range is not None


# ---------------------------------------------------------------------------
# Integration with mediocre_fridge.collect_responses_from
# ---------------------------------------------------------------------------

class _FakeOven:
    """Tiny CodeOven stand-in that .complete()s by echoing a canned response."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str, *, max_new_tokens: int = 32,
                 temperature: float = 0.7, stop: tuple = ()) -> str:
        return self._response


def test_collect_with_new_range_label_rule() -> None:
    """A NewRange instance is callable as a label_rule."""
    from hypernix import mediocre_fridge, new_range

    rule = new_range.new_range()
    oven = _FakeOven(response="42")
    examples = mediocre_fridge.collect_responses_from(
        oven, prompts=["What is 6 * 7?"], label_rule=rule,
    )
    assert len(examples) == 1
    assert examples[0].response == "42"
    assert examples[0].label == "GOOD"


def test_collect_with_industrial_range_label_rule() -> None:
    from hypernix import industrial_range, mediocre_fridge

    rule = industrial_range.industrial_range(judge=_FakeJudge(["GOOD"]))
    oven = _FakeOven(response="Paris")
    examples = mediocre_fridge.collect_responses_from(
        oven, prompts=["Capital of France?"], label_rule=rule,
    )
    assert examples[0].label == "GOOD"


# Quiet pytest's "captured" reminder for the FakeJudge fixture-style helpers.
@pytest.fixture(autouse=True)
def _no_warnings() -> None:
    return None
