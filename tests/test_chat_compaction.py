"""Making a long conversation fit again.

A session that has run for a week does not fit in a context window, and
the default — dropping the oldest turns — loses exactly what a long
conversation is for. The decision made on Tuesday falls out first.

Most of this file is about the things compaction must *not* do, because
those are the ones that turn a helpful feature into data loss:

* never touch the recent exchange, which is what the next reply answers
* never delete, only mark, so the transcript a person scrolls survives
* never summarise a summary, which is how a conversation becomes a
  summary of a summary of a summary
* never replace real messages with an empty summary
"""
from __future__ import annotations

import logging

import pytest

from hypernix.hyperlink.compaction import (
    COMPACTED_KEY,
    KEEP_RECENT,
    SUMMARY_MARKER,
    plan,
    summarise_extractively,
    summary_prompt,
)
from hypernix.hyperlink.sessions import ChatMessage


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def message(
    role: str, content: str, *, seq: int = 0, metadata: dict | None = None
) -> ChatMessage:
    return ChatMessage(
        message_id=f"msg_{role}_{seq}",
        session_id="s",
        role=role,
        content=content,
        seq=seq,
        created_at=float(seq),
        metadata=metadata or {},
    )


def conversation(turns: int = 10, *, words: int = 40) -> list[ChatMessage]:
    """A thread long enough to be worth compacting."""
    history = [message("system", "You are helpful.", seq=0)]
    for index in range(turns):
        history.append(message("user", f"question {index} " + "q " * words, seq=index * 2 + 1))
        history.append(
            message("assistant", f"answer {index} " + "a " * words, seq=index * 2 + 2)
        )
    return history


class TestWhatEachScopeTakes:
    def test_prompts_takes_only_what_the_person_sent(self):
        proposed = plan(conversation(), "prompts")
        assert {m.role for m in proposed.targets} == {"user"}

    def test_responses_takes_only_the_model(self):
        proposed = plan(conversation(), "responses")
        assert {m.role for m in proposed.targets} == {"assistant"}

    def test_system_takes_only_the_system_prompt(self):
        proposed = plan(conversation(), "system")
        assert {m.role for m in proposed.targets} == {"system"}

    def test_all_takes_both_sides_and_not_the_system_prompt(self):
        """The system prompt has its own endpoint because it is a
        different decision: it is paid for on every turn, and somebody
        compacting a long thread rarely means to rewrite their
        instructions at the same time."""
        proposed = plan(conversation(), "all")
        assert {m.role for m in proposed.targets} == {"user", "assistant"}

    def test_an_unknown_scope_is_refused(self):
        with pytest.raises(ValueError, match="Unknown compaction scope"):
            plan(conversation(), "everything")


class TestTheRecentExchangeIsProtected:
    """What the next reply is a reply *to*. Summarising it is how
    compaction turns into "the assistant stopped following the thread"."""

    def test_the_last_few_messages_are_never_targets(self):
        history = conversation()
        proposed = plan(history, "all")
        recent = {m.message_id for m in history[-KEEP_RECENT:]}
        assert not recent & {m.message_id for m in proposed.targets}

    def test_the_question_just_asked_survives_a_prompt_compaction(self):
        history = conversation()
        latest = [m for m in history if m.role == "user"][-1]
        proposed = plan(history, "prompts")
        assert latest.message_id not in {m.message_id for m in proposed.targets}

    def test_the_protection_can_be_widened(self):
        wide = plan(conversation(turns=20), "all", keep_recent=12)
        narrow = plan(conversation(turns=20), "all", keep_recent=2)
        assert len(wide.targets) < len(narrow.targets)

    def test_the_system_prompt_is_not_shielded_by_recency(self):
        """It sits at the top of every request regardless of age, so
        "recent" means nothing for it — and shielding it would exempt it
        from the one scope that names it."""
        short = [
            message("system", "instructions", seq=0),
            message("user", "hi", seq=1),
            message("assistant", "hello", seq=2),
        ]
        assert plan(short, "system").targets


class TestItDoesNotEatItself:
    def test_an_already_compacted_message_is_not_a_target(self):
        history = conversation()
        history[1].metadata = {COMPACTED_KEY: "msg_summary"}
        proposed = plan(history, "all")
        assert history[1].message_id not in {m.message_id for m in proposed.targets}

    def test_a_summary_is_never_summarised(self):
        """Otherwise a conversation becomes a summary of a summary of a
        summary, losing a little more each time for less and less
        saving."""
        history = conversation()
        history.append(
            message("system", "[earlier…]", seq=99, metadata={SUMMARY_MARKER: True})
        )
        proposed = plan(history, "all")
        assert "msg_system_99" not in {m.message_id for m in proposed.targets}

    def test_an_empty_message_is_not_worth_compacting(self):
        history = [message("user", "   ", seq=i) for i in range(10)]
        assert plan(history, "prompts").targets == []


class TestWhenItIsNotWorthIt:
    def test_a_short_conversation_is_left_alone(self):
        """A summary has its own overhead; three short messages do not
        become two."""
        short = [
            message("user", "hi", seq=1),
            message("assistant", "hello", seq=2),
        ]
        assert not plan(short, "all").viable

    def test_that_is_not_an_error(self):
        """The session is simply already short. Raising would make every
        caller handle a non-problem."""
        proposed = plan([message("user", "hi", seq=1)], "all")
        assert proposed.viable is False
        assert proposed.reason

    def test_an_empty_history_is_fine(self):
        assert not plan([], "dynamic").viable


class TestDynamicMeasuresRatherThanGuesses:
    def test_it_picks_the_prompts_when_they_are_the_weight(self):
        """Somebody pasting logs into a model."""
        history = [message("system", "be helpful", seq=0)]
        for index in range(8):
            history.append(message("user", "x " * 400, seq=index * 2 + 1))
            history.append(message("assistant", "ok", seq=index * 2 + 2))
        assert plan(history, "dynamic").scope == "prompts"

    def test_it_picks_the_responses_when_they_are(self):
        """Somebody reading long answers."""
        history = [message("system", "be helpful", seq=0)]
        for index in range(8):
            history.append(message("user", "go on", seq=index * 2 + 1))
            history.append(message("assistant", "y " * 400, seq=index * 2 + 2))
        assert plan(history, "dynamic").scope == "responses"

    def test_a_huge_system_prompt_wins(self):
        """Paid for on every single turn, so it is the cheapest fix."""
        history = [message("system", "s " * 2000, seq=0)]
        for index in range(6):
            history.append(message("user", "short", seq=index * 2 + 1))
            history.append(message("assistant", "short", seq=index * 2 + 2))
        assert plan(history, "dynamic").scope == "system"

    def test_it_says_why(self):
        """Inspectable rather than magic: a caller showing "compacted
        your prompts" should be able to say what made that the answer."""
        proposed = plan(conversation(), "dynamic")
        assert proposed.reason

    def test_the_reported_scope_is_the_real_one(self):
        """`dynamic` resolves to a concrete scope; reporting "dynamic"
        back would tell the caller nothing."""
        assert plan(conversation(), "dynamic").scope in (
            "prompts", "responses", "system", "all"
        )


class TestTheExtractiveFallback:
    def test_it_quotes_rather_than_inventing(self):
        """A bad abstractive summary invents things the conversation did
        not contain, and a model reading an invented fact cannot tell it
        from a real one."""
        summary = summarise_extractively([
            message("user", "the database is on port 5433", seq=1),
            message("assistant", "noted", seq=2),
        ])
        assert "5433" in summary

    def test_it_attributes(self):
        summary = summarise_extractively([message("user", "hello there", seq=1)])
        assert "They asked" in summary

    def test_it_is_bounded(self):
        summary = summarise_extractively(
            [message("user", "x " * 500, seq=i) for i in range(50)], limit=800
        )
        assert len(summary) < 2000

    def test_nothing_in_is_nothing_out(self):
        assert summarise_extractively([]) == ""


class TestTheModelPrompt:
    def test_it_tells_the_model_what_must_survive(self):
        prompt = summary_prompt([message("user", "hi", seq=1)])
        instructions = prompt[0]["content"]
        for must in ("decision", "keep", "unsure"):
            assert must in instructions.lower()

    def test_the_transcript_is_attributed(self):
        prompt = summary_prompt([
            message("user", "a question", seq=1),
            message("assistant", "an answer", seq=2),
        ])
        assert "USER: a question" in prompt[1]["content"]
        assert "ASSISTANT: an answer" in prompt[1]["content"]


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch) -> TestClient:
    from hypernix.t1api.app import create_app

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    return TestClient(create_app(), client=("192.168.1.50", 5432))


@pytest.fixture
def session(client) -> str:
    """A session with enough in it to be worth compacting."""
    session_id = client.post(
        "/hyperlink/sessions",
        json={"title": "long one", "model_id": "", "system_prompt": ""},
    ).json()["session"]["session_id"]
    store = client.app.state.t1_session_store
    owner = client.get("/usage/current").json()["key_id"]
    for index in range(10):
        store.append(session_id, role="user", content=f"question {index} " + "q " * 60,
                     owner=owner)
        store.append(session_id, role="assistant", content=f"answer {index} " + "a " * 60,
                     owner=owner)
    return session_id


class TestTheEndpoints:
    def test_all_five_exist(self, client):
        paths = set(client.app.openapi()["paths"])
        for name in ("prompts", "prompts/system", "all", "responses", "dynamic"):
            assert f"/chat/compact/{name}" in paths

    def test_a_dry_run_changes_nothing(self, client, session):
        before = len(client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"])
        body = client.post(
            "/chat/compact/all", json={"session_id": session, "dry_run": True}
        ).json()
        assert body["applied"] is False
        assert body["plan"]["message_count"] > 0
        after = len(client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"])
        assert after == before

    def test_applying_writes_a_summary(self, client, session):
        body = client.post("/chat/compact/all", json={"session_id": session}).json()
        assert body["applied"] is True
        assert body["summary"]
        assert body["messages_compacted"] > 0

    def test_it_falls_back_to_quoting_with_no_model(self, client, session):
        """Compaction is most wanted exactly when a conversation has got
        long. Refusing would leave somebody stuck with a session that no
        longer fits and no way to shrink it."""
        body = client.post("/chat/compact/all", json={"session_id": session}).json()
        assert body["summarised_by"] == "extractive"

    def test_the_transcript_is_not_shortened(self, client, session):
        """Only the model's view changes. The person's scrollback is data
        they may care about more than the model does."""
        before = client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]
        client.post("/chat/compact/all", json={"session_id": session})
        after = client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]
        assert len(after) == len(before) + 1  # the summary
        originals = {m["message_id"] for m in before}
        assert originals <= {m["message_id"] for m in after}

    def test_the_models_view_is_shortened(self, client, session):
        """The point of the whole thing."""
        store = client.app.state.t1_session_store
        owner = client.get("/usage/current").json()["key_id"]
        before = store.context_for(session, owner=owner, token_budget=100_000)
        client.post("/chat/compact/all", json={"session_id": session})
        after = store.context_for(session, owner=owner, token_budget=100_000)
        assert len(after) < len(before)

    def test_compacting_twice_does_not_double_count(self, client, session):
        """Marking is idempotent, and the second pass has fewer eligible
        messages rather than re-summarising the first summary."""
        first = client.post("/chat/compact/all", json={"session_id": session}).json()
        second = client.post("/chat/compact/all", json={"session_id": session}).json()
        assert first["messages_compacted"] > 0
        assert second["messages_compacted"] < first["messages_compacted"]

    def test_dynamic_reports_what_it_decided(self, client, session):
        body = client.post("/chat/compact/dynamic", json={"session_id": session}).json()
        assert body["scope"] != "dynamic"
        assert body["plan"]["reason"]

    def test_another_owners_session_is_not_reachable(self, client):
        assert client.post(
            "/chat/compact/all", json={"session_id": "sess_not_yours"}
        ).status_code == 404
