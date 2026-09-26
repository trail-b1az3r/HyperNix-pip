"""HyperLink tool calls on the models people actually run (0.72.6.post1).

"hyperlink still can't tool call." 0.72.6 offered the tools on both chat
routes, and three things still stood between a local model and a call:

1. **The format was taught only on the built-in runner.** On LM Studio
   the tools went in the ``tools`` field and nowhere else, and a model
   whose chat template drops that field (Gemma's does) never learned
   there were any. It said it had none.
2. **Half the conventions were unread.** LM Studio's own
   ``[TOOL_REQUEST]``, Gemma's ``call:name{...}`` and ``tool_code``,
   Llama's ``<function=...>``, DeepSeek's markers and a JSON call written
   after a sentence were all shown to the person as the answer.
3. **A backend that refuses ``tools`` failed the message.** llama-server
   without ``--jinja`` rejects the field outright.
"""
from __future__ import annotations

import json

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink.toolloop import CallGuard, ToolRound, run_tool_loop, stream_tool_loop
from hypernix.interfaces.noodle.tools import ToolContext
from hypernix.runtime.toolcalls import extract_calls

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "server_version", "description": "version",
        "parameters": {"type": "object", "properties": {}}}},
]


def _one(text):
    found = extract_calls(text, tools=TOOLS)
    assert len(found.calls) == 1, (text, found)
    return found.calls[0], found.text


class TestTheConventionsModelsWrite:
    def test_lm_studios_own(self):
        call, rest = _one('[TOOL_REQUEST]{"name": "get_weather", "arguments": '
                          '{"location": "Leeds"}}[END_TOOL_REQUEST]')
        assert (call.name, call.arguments, call.source) == ("get_weather", {"location": "Leeds"}, "lmstudio")
        assert rest == ""

    def test_gemma_function_call_with_escaped_strings(self):
        """A comma and a colon inside a value are not syntax."""
        call, _ = _one("<start_function_call>call:get_weather{location:<escape>Leeds, UK: north<escape>,"
                       "days:3}<end_function_call>")
        assert call.arguments == {"location": "Leeds, UK: north", "days": 3}
        assert call.source == "gemma"

    def test_gemma_4_markers(self):
        call, _ = _one("<|tool_call>call:server_version{}<tool_call|>")
        assert call.name == "server_version" and call.arguments == {}

    def test_gemma_3_tool_code(self):
        call, rest = _one('I will check.\n```tool_code\nget_weather(location="Leeds", days=2)\n```')
        assert call.arguments == {"location": "Leeds", "days": 2}
        assert rest == "I will check."

    def test_tool_code_wrapped_in_print_and_positional(self):
        call, _ = _one('```tool_code\nprint(get_weather("Leeds"))\n```')
        assert call.arguments == {"location": "Leeds"}

    def test_a_reply_that_is_only_a_python_call(self):
        call, _ = _one("server_version()")
        assert call.name == "server_version" and call.source == "python"

    def test_llama_function_tag(self):
        call, _ = _one('<function=get_weather>{"location": "Leeds"}</function>')
        assert call.arguments == {"location": "Leeds"} and call.source == "llama-function"

    def test_deepseek(self):
        call, _ = _one('<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n'
                       '{"location": "Leeds"}\n```<｜tool▁call▁end｜>')
        assert call.arguments == {"location": "Leeds"}

    def test_json_after_a_sentence(self):
        call, rest = _one('Let me check that. {"name": "get_weather", "arguments": {"location": "Leeds"}}')
        assert call.arguments == {"location": "Leeds"}
        assert rest == "Let me check that."


class TestWhatIsNotACall:
    def test_python_that_is_not_a_literal_is_never_run_or_read(self):
        found = extract_calls('```tool_code\nget_weather(location=__import__("os").system("x"))\n```',
                              tools=TOOLS)
        assert found.calls == []

    def test_an_unoffered_name_written_as_python(self):
        assert extract_calls("delete_everything()", tools=TOOLS).calls == []

    def test_prose_with_a_function_name_and_parentheses(self):
        text = "You can call server_version() to see it, and then carry on."
        assert extract_calls(text, tools=TOOLS).calls == []

    def test_json_that_is_the_answer(self):
        text = 'Here is the config: {"name": "my-app", "version": "1.0"}'
        assert extract_calls(text, tools=TOOLS).calls == []


class TestTheStreamHoldsThemBack:
    @pytest.mark.parametrize("marker", ["[TOOL_REQUEST]", "<function=", "<start_function_call>",
                                        "<|tool_call>", "```tool_code", '{"name"'])
    def test_marker(self, marker):
        guard = CallGuard()
        shown = guard.feed("Checking. ") + guard.feed(marker + "rest")
        assert marker not in shown


@pytest.fixture
def context(tmp_path):
    # Memory on, so there are tools to offer with no workspace or T1.
    return ToolContext(root=tmp_path, memory_enabled=True)


def _chunks(text, *, size=8):
    for i in range(0, len(text), size):
        yield {"model": "m", "choices": [{"delta": {"content": text[i:i + size]}}]}
    yield {"model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}]}


class RefusesTools:
    """A backend that errors on the `tools` field, as llama-server does
    without --jinja, and otherwise answers from a script."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked = []

    def __call__(self, messages, tools):
        self.asked.append({"messages": [dict(m) for m in messages], "tools": tools})

        def stream():
            if tools:
                raise RuntimeError('"tools" param requires --jinja flag')
            yield from _chunks(self.replies.pop(0))
        return stream()


class TestABackendThatRefusesTools:
    def test_the_turn_carries_on_in_text_and_the_call_runs(self, context, monkeypatch):
        from hypernix.hyperlink import toolloop

        monkeypatch.setattr(toolloop, "_run_call", lambda call, *a: ToolRound(
            call["function"]["name"], {}, True, "hypernix 0.72.6"))
        backend = RefusesTools(
            '<tool_call>{"name": "server_version", "arguments": {}}</tool_call>',
            "It is 0.72.6.",
        )
        events = list(stream_tool_loop([{"role": "user", "content": "version?"}], context,
                                       ask_stream=backend, teach_format=True, workspace=False))
        assert [v.tool for k, v in events if k == "tool"] == ["server_version"]
        assert "".join(v for k, v in events if k == "delta") == "It is 0.72.6."
        # Asked once with tools (refused), then never again with them.
        assert backend.asked[0]["tools"] and not any(a["tools"] for a in backend.asked[1:])
        # The second round's transcript has no tool role and no tool_calls field.
        second = backend.asked[-1]["messages"]
        assert all(m["role"] != "tool" for m in second)
        assert all("tool_calls" not in m for m in second)
        assert any("<tool_response" in str(m.get("content")) for m in second)

    def test_the_non_streamed_route_too(self, context, monkeypatch):
        from hypernix.hyperlink import toolloop

        monkeypatch.setattr(toolloop, "_run_call", lambda call, *a: ToolRound(
            call["function"]["name"], {}, True, "ok"))
        replies = ['<tool_call>{"name": "server_version", "arguments": {}}</tool_call>', "Done."]
        asked = []

        def ask(messages, tools):
            asked.append(tools)
            if tools:
                raise RuntimeError("tools not supported")
            return {"choices": [{"message": {"role": "assistant", "content": replies.pop(0)},
                                 "finish_reason": "stop"}]}

        _msgs, envelope, rounds = run_tool_loop(
            [{"role": "user", "content": "v"}], context, ask=ask,
            extract=lambda e: (e["choices"][0]["message"]["content"], "stop"),
            teach_format=True, workspace=False)
        assert [r.tool for r in rounds] == ["server_version"]
        assert envelope["choices"][0]["message"]["content"] == "Done."

    def test_a_backend_that_is_simply_down_still_fails(self, context):
        def down(messages, tools):
            def stream():
                raise ConnectionError("refused")
                yield  # pragma: no cover
            return stream()

        with pytest.raises(ConnectionError):
            list(stream_tool_loop([{"role": "user", "content": "x"}], context,
                                  ask_stream=down, workspace=False))


class TestTheFormatIsTaughtEverywhere:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        clear_t1_config(monkeypatch)
        monkeypatch.delenv("T1_HYPERLINK_TEACH_TOOLS", raising=False)

    class LMStudio:
        is_hypernix = False

    class Runner:
        is_hypernix = True

    def test_on_by_default_for_lm_studio_too(self):
        from hypernix.t1api.routers.hyperlink import _teach_tools

        assert _teach_tools(self.LMStudio()) and _teach_tools(self.Runner())

    def test_off_and_runner_only(self, monkeypatch):
        from hypernix.t1api.routers.hyperlink import _teach_tools

        monkeypatch.setenv("T1_HYPERLINK_TEACH_TOOLS", "0")
        assert not _teach_tools(self.Runner())
        monkeypatch.setenv("T1_HYPERLINK_TEACH_TOOLS", "runner")
        assert _teach_tools(self.Runner()) and not _teach_tools(self.LMStudio())

    def test_the_taught_prompt_reaches_the_model(self, context):
        seen = []

        def backend(messages, tools):
            seen.append(messages)
            return _chunks("hello")

        list(stream_tool_loop([{"role": "system", "content": "Be kind."},
                               {"role": "user", "content": "hi"}], context,
                              ask_stream=backend, teach_format=True, workspace=False))
        system = seen[0][0]["content"]
        assert system.startswith("Be kind.") and "<tool_call>" in system
        example = system.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        assert "name" in json.loads(example)


class TestTheTaughtPromptSaysWhenAndShowsTheWholeExchange:
    """Told only the call format, a small model treated the tools as
    optional: it said it had no access to things a tool could reach, or
    stopped at the call as though that were the answer."""

    def test_it_says_the_tools_are_real_and_when_to_use_them(self):
        from hypernix.runtime.toolcalls import tool_prompt

        text = tool_prompt(TOOLS)
        assert "These tools are real" in text
        assert "do not say you cannot access it" in text

    def test_it_shows_call_result_and_answer_with_a_real_tool(self):
        from hypernix.runtime.toolcalls import tool_prompt

        text = tool_prompt(TOOLS)
        first_call = text.split("<tool_call>")[1].split("</tool_call>")[0]
        assert json.loads(first_call)["name"] == "get_weather"
        assert '<tool_response name="get_weather">' in text
        assert "Assistant: (the answer" in text
        assert "never write a <tool_response> yourself" in text

    def test_a_result_the_model_invents_is_never_kept(self, context, monkeypatch):
        from hypernix.hyperlink import toolloop

        monkeypatch.setattr(toolloop, "_run_call", lambda call, *a: ToolRound(
            call["function"]["name"], {}, True, "real: 0.72.6.post1"))
        seen = []

        def backend(messages, tools):
            seen.append([dict(m) for m in messages])
            if len(seen) == 1:
                return _chunks('<tool_call>{"name": "server_version", "arguments": {}}</tool_call>\n'
                               '<tool_response name="server_version">made up 9.9</tool_response>')
            return _chunks("It is 0.72.6.post1.")

        events = list(stream_tool_loop([{"role": "user", "content": "version?"}], context,
                                       ask_stream=backend, teach_format=True, workspace=False))
        shown = "".join(v for k, v in events if k == "delta")
        assert "made up" not in shown and shown == "It is 0.72.6.post1."
        assert not any("made up" in str(m.get("content")) for m in seen[1])
