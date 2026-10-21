"""0.72.8 updates to the ranges: parsing current judges, modern refusals,
the unused fallback, and a judge on an OpenAI-compatible server."""
from __future__ import annotations

import io
import json

import pytest

from hypernix.evaluation import industrial_range as ir
from hypernix.evaluation import new_range


class Judge:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt, max_new_tokens, temperature, stop):
        return self.reply


@pytest.mark.parametrize("reply, label", [
    ("GOOD", "GOOD"), ("bad.", "BAD"),
    ("not good", "BAD"), ("Not bad at all", "GOOD"),
    ("<think>it looks bad at first, but</think> GOOD", "GOOD"),
    ("<thinking>hmm good?</thinking>\nBAD", "BAD"),
])
def test_pointwise(reply, label):
    assert ir.IndustrialRange(Judge(reply), use_cache=False).label("p", "r") == label


def test_the_fallback_is_used_when_there_is_no_verdict():
    judge = ir.IndustrialRange(Judge("maybe?"), use_cache=False, fallback_label="GOOD")
    assert judge.label("p", "r") == "GOOD"


@pytest.mark.parametrize("reply, verdict", [
    ("A", "A"), ("B is better", "B"), ("T", "T"), ("It's a tie", "T"),
    ("The better one is A", "A"),          # used to be read as a tie
    ("<think>A or B... B</think> A", "A"),
])
def test_pairwise(reply, verdict):
    assert ir.IndustrialRange(Judge(reply), use_cache=False).compare("p", "x", "y") == verdict


@pytest.mark.parametrize("text", ["I can’t help with that.", "I'm unable to do that.",
                                  "I must decline this request.", "I apologize, but no."])
def test_modern_refusals(text):
    assert new_range.is_refusal("q", text)


def test_an_answer_is_not_a_refusal():
    assert not new_range.is_refusal("q", "Paris is the capital of France.")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_judge_on_an_openai_compatible_server(monkeypatch):
    sent = {}

    def fake(request, timeout=None):
        sent["url"], sent["body"] = request.full_url, json.loads(request.data)
        sent["auth"] = request.get_header("Authorization")
        return _Response(json.dumps({"choices": [{"message": {"content": "GOOD"}}]}).encode())

    monkeypatch.setattr("hypernix.security.safeurl.urllib.request.urlopen", fake)
    judge = ir.OpenAIJudge("http://127.0.0.1:8081/v1/", model="judge", api_key="k")
    rng = ir.industrial_range(judge, use_cache=False)
    assert rng.label("Capital of France?", "Paris") == "GOOD"
    assert sent["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert sent["body"]["model"] == "judge" and "\n" not in sent["body"].get("stop", [])
    assert sent["auth"] == "Bearer k"
