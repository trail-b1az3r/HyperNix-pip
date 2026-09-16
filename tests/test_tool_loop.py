"""Letting a HyperLink model actually use the tools.

`/noodle/run` lets a *caller* run one tool. This is the other half: the
model asks, the server runs it, and the answer goes back into the
conversation — so "zip up the logs and tell me what is in them" is one
message rather than five.

Three rules, and each one is a failure designed out rather than handled.

**Bounded.** A model that calls a tool, reads the result and calls it
again can do that forever, and each round is a full inference over a
transcript that just grew. The limit ends it, and the model is *told* it
has been ended — a model that does not know it ran out of rounds writes
its last reply as if it were about to do more.

**Every call is answered.** A `tool_calls` message with no matching
`tool` reply is a malformed conversation, and the next turn either
errors or quietly drops it. So a missing tool, a raising tool and a
switched-off tool all produce a tool message saying so.

**A refusal is a result.** `allow_execute` off is the operator's answer,
not an error. Handing it back as a normal tool result lets the model say
so to the person, instead of the turn dying with a stack trace nobody
can act on.
"""
from __future__ import annotations

import json

import pytest

from hypernix.hyperlink.toolloop import (
    MAX_RESULT_CHARS,
    MAX_ROUNDS,
    openai_tools,
    run_tool_loop,
)
from hypernix.interfaces.noodle.tools import ToolContext


@pytest.fixture
def context(tmp_path):
    return ToolContext(root=tmp_path)


def envelope(content: str = "", calls: list | None = None) -> dict:
    message: dict = {"content": content}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def call(name: str, arguments, call_id: str = "c1") -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "function": {"name": name, "arguments": raw}}


def extract(env: dict) -> tuple[str, str]:
    return env["choices"][0]["message"].get("content", ""), "stop"


def scripted(*turns):
    """An `ask` that returns each envelope in order."""
    remaining = list(turns)
    seen: list[tuple[int, int]] = []

    def ask(messages, tools):
        seen.append((len(messages), len(tools)))
        return remaining.pop(0) if remaining else envelope("done")

    ask.seen = seen  # type: ignore[attr-defined]
    return ask


class TestWhatIsOffered:
    def test_execution_is_not_offered_when_it_is_off(self, context):
        """A model told it can execute files and then refused, every
        time, spends its rounds finding that out."""
        names = {t["function"]["name"] for t in openai_tools(context)}
        assert "execute_file" not in names

    def test_execution_is_offered_when_the_operator_allows_it(self, tmp_path):
        context = ToolContext(root=tmp_path, allow_execute=True)
        names = {t["function"]["name"] for t in openai_tools(context)}
        assert "execute_file" in names

    def test_the_file_tools_are_always_there(self, context):
        names = {t["function"]["name"] for t in openai_tools(context)}
        assert {"create_file", "edit_file", "read_file"} <= names

    def test_fish_and_zipping_are_offered(self, context):
        """The four things the request named for noodle."""
        names = {t["function"]["name"] for t in openai_tools(context)}
        assert "run_fish" in names
        assert "unzip" in names

    def test_every_tool_has_the_openai_shape(self, context):
        for tool in openai_tools(context):
            assert tool["type"] == "function"
            assert tool["function"]["name"]
            assert tool["function"]["description"]
            assert tool["function"]["parameters"]["type"] == "object"


class TestTheLoop:
    def test_no_tool_call_means_one_round(self, context):
        ask = scripted(envelope("Just an answer."))
        _messages, final, rounds = run_tool_loop(
            [{"role": "user", "content": "hi"}], context, ask=ask, extract=extract
        )
        assert rounds == []
        assert extract(final)[0] == "Just an answer."

    def test_a_tool_runs_and_the_model_answers(self, context, tmp_path):
        ask = scripted(
            envelope(calls=[call("create_file", {"path": "a.txt", "content": "hi"})]),
            envelope("Made a.txt."),
        )
        _messages, final, rounds = run_tool_loop(
            [{"role": "user", "content": "make a file"}],
            context, ask=ask, extract=extract,
        )
        assert [r.tool for r in rounds] == ["create_file"]
        assert rounds[0].ok
        assert (tmp_path / "a.txt").read_text() == "hi"
        assert extract(final)[0] == "Made a.txt."

    def test_the_transcript_pairs_every_call_with_a_reply(self, context):
        """A `tool_calls` message with no matching `tool` reply is a
        malformed conversation, and the *next* turn is built from it."""
        ask = scripted(
            envelope(calls=[call("read_file", {"path": "missing.txt"})]),
            envelope("It is not there."),
        )
        messages, _final, _rounds = run_tool_loop(
            [{"role": "user", "content": "read it"}],
            context, ask=ask, extract=extract,
        )
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool"]
        assert messages[-1]["tool_call_id"] == "c1"

    def test_several_calls_in_one_round_all_answer(self, context):
        ask = scripted(
            envelope(calls=[
                call("create_file", {"path": "a.txt", "content": "1"}, "c1"),
                call("create_file", {"path": "b.txt", "content": "2"}, "c2"),
            ]),
            envelope("Made both."),
        )
        messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "two files"}],
            context, ask=ask, extract=extract,
        )
        assert len(rounds) == 2
        replies = [m for m in messages if m["role"] == "tool"]
        assert {m["tool_call_id"] for m in replies} == {"c1", "c2"}

    def test_the_tools_are_offered_on_every_round(self, context):
        ask = scripted(
            envelope(calls=[call("create_file", {"path": "a.txt", "content": "1"})]),
            envelope("done"),
        )
        run_tool_loop([{"role": "user", "content": "x"}], context, ask=ask, extract=extract)
        assert all(tools > 0 for _messages, tools in ask.seen[:2])


class TestItCannotRunForever:
    def test_a_model_that_never_stops_is_stopped(self, context):
        """Each round is a full inference over a growing transcript. A
        loop here is a GPU nobody notices is busy."""
        def ask(messages, tools):
            return envelope(calls=[call("read_file", {"path": "a.txt"})])

        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "loop"}],
            context, ask=ask, extract=extract, max_rounds=3,
        )
        assert len(rounds) == 3

    def test_the_model_is_told_it_ran_out(self, context):
        """A model that does not know writes its last reply as if it
        were about to do more, and the person gets half a sentence about
        what it was going to check next."""
        def ask(messages, tools):
            if any("used all" in str(m.get("content", "")) for m in messages):
                return envelope("I ran out of steps.")
            return envelope(calls=[call("read_file", {"path": "a.txt"})])

        messages, final, _rounds = run_tool_loop(
            [{"role": "user", "content": "loop"}],
            context, ask=ask, extract=extract, max_rounds=2,
        )
        assert any("used all" in str(m.get("content", "")) for m in messages)
        assert extract(final)[0] == "I ran out of steps."

    def test_the_default_limit_is_small_enough_to_notice(self):
        assert 1 < MAX_ROUNDS <= 16


class TestEveryFailureIsAResult:
    def test_a_tool_that_does_not_exist(self, context):
        ask = scripted(
            envelope(calls=[call("teleport", {})]),
            envelope("I cannot do that."),
        )
        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        assert rounds[0].ok is False
        assert rounds[0].code == "no_such_tool"
        assert "teleport" in rounds[0].content

    def test_it_lists_what_does_exist(self, context):
        """So the model's next attempt is an informed one."""
        ask = scripted(envelope(calls=[call("teleport", {})]), envelope("ok"))
        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        assert "create_file" in rounds[0].content

    def test_arguments_that_are_not_json(self, context):
        """Models do emit broken JSON. The model can fix its own if it
        is told what was wrong with it."""
        ask = scripted(
            envelope(calls=[call("create_file", "{not json at all")]),
            envelope("Let me try again."),
        )
        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        assert rounds[0].ok is False
        assert rounds[0].code == "bad_arguments"

    def test_arguments_already_parsed_by_the_backend(self, context, tmp_path):
        """Some backends hand back an object rather than a string."""
        ask = scripted(
            envelope(calls=[{"id": "c1", "function": {
                "name": "create_file",
                "arguments": {"path": "a.txt", "content": "hi"},
            }}]),
            envelope("done"),
        )
        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        assert rounds[0].ok
        assert (tmp_path / "a.txt").exists()

    def test_empty_arguments_are_not_an_error(self, context):
        ask = scripted(
            envelope(calls=[call("create_todo", "")]),
            envelope("done"),
        )
        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        assert rounds[0].code != "bad_arguments"

    def test_a_tool_that_raises_does_not_lose_the_turn(self, context, monkeypatch):
        """A 500 here loses the person's question with it."""
        from hypernix.hyperlink import toolloop

        def explode(ctx, name, arguments):
            raise RuntimeError("the disk caught fire")

        monkeypatch.setattr(toolloop, "run_tool", explode)
        ask = scripted(
            envelope(calls=[call("read_file", {"path": "a.txt"})]),
            envelope("Something went wrong."),
        )
        _messages, final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        assert rounds[0].ok is False
        assert rounds[0].code == "tool_raised"
        assert extract(final)[0] == "Something went wrong."


class TestResultsAreBounded:
    def test_an_enormous_result_is_cut(self, context, monkeypatch):
        """A file read can be megabytes. Past the limit it is not
        context, it is the whole context — the conversation gets evicted
        to make room for a listing nobody asked to see."""
        from hypernix.hyperlink import toolloop
        from hypernix.interfaces.noodle.tools import ToolResult

        monkeypatch.setattr(
            toolloop, "run_tool",
            lambda ctx, name, args: ToolResult(True, "x" * (MAX_RESULT_CHARS * 3)),
        )
        ask = scripted(
            envelope(calls=[call("read_file", {"path": "big.txt"})]),
            envelope("done"),
        )
        messages, _final, _rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        reply = [m for m in messages if m["role"] == "tool"][0]
        assert len(reply["content"]) < MAX_RESULT_CHARS * 2

    def test_the_cut_says_it_was_cut(self, context, monkeypatch):
        """A model handed a silently truncated file will reason about
        the missing half as if it were not there."""
        from hypernix.hyperlink import toolloop
        from hypernix.interfaces.noodle.tools import ToolResult

        monkeypatch.setattr(
            toolloop, "run_tool",
            lambda ctx, name, args: ToolResult(True, "x" * (MAX_RESULT_CHARS * 2)),
        )
        ask = scripted(envelope(calls=[call("read_file", {"path": "big.txt"})]),
                       envelope("done"))
        messages, _final, _rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        reply = [m for m in messages if m["role"] == "tool"][0]
        assert "more characters were cut" in reply["content"]


class TestItReportsWhatHappened:
    def test_each_round_is_recorded(self, context):
        ask = scripted(
            envelope(calls=[call("create_file", {"path": "a.txt", "content": "1"})]),
            envelope("done"),
        )
        _messages, _final, rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context, ask=ask, extract=extract
        )
        record = rounds[0].to_dict()
        assert record["tool"] == "create_file"
        assert record["arguments"]["path"] == "a.txt"
        assert record["ok"] is True

    def test_a_listener_sees_them_as_they_happen(self, context):
        """For streaming: the app can show "running create_file…" rather
        than a spinner that means nothing."""
        seen = []
        ask = scripted(
            envelope(calls=[call("create_file", {"path": "a.txt", "content": "1"})]),
            envelope("done"),
        )
        run_tool_loop(
            [{"role": "user", "content": "x"}], context,
            ask=ask, extract=extract, on_round=seen.append,
        )
        assert [r.tool for r in seen] == ["create_file"]

    def test_a_listener_that_raises_does_not_fail_the_turn(self, context):
        def explode(round_):
            raise RuntimeError("the UI is on fire")

        ask = scripted(
            envelope(calls=[call("create_file", {"path": "a.txt", "content": "1"})]),
            envelope("done"),
        )
        _messages, final, _rounds = run_tool_loop(
            [{"role": "user", "content": "x"}], context,
            ask=ask, extract=extract, on_round=explode,
        )
        assert extract(final)[0] == "done"
