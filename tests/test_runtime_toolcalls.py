"""Reading tool calls out of whatever a model wrote, and T1 as tools.

What these tests are for
------------------------
Most of "local models are bad at calling tools" is a parsing problem.
A GGUF trained on Hermes' `<tool_call>` tags writes exactly that, and if
nothing reads it the call is shown to the person as the answer and no
tool runs. So the first half here is every convention models actually
use, and the JSON mistakes they actually make.

The second half is what must *not* happen: a JSON answer the person
asked for treated as a call, a misspelt tool name quietly corrected into
a different tool, a mutating tool run because the model named one it was
never offered. Each of those turns a parsing convenience into an action
nobody asked for.

The T1 tools are exercised against a real HTTP server, so the credential
forwarding and the query/body placement are checked on the wire.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hypernix.runtime import t1tools
from hypernix.runtime.toolcalls import (
    ToolCall,
    extract_calls,
    repair_json,
    tool_prompt,
    validate_arguments,
)

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web.",
        "parameters": {"type": "object", "properties": {
            "q": {"type": "string"}, "depth": {"type": "integer"},
            "safe": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["fast", "deep"]}},
            "required": ["q"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "runner_status", "description": "What is loaded.",
        "parameters": {"type": "object", "properties": {}}}},
]


class TestFormats:
    def test_structured(self):
        found = extract_calls({"content": "", "tool_calls": [{
            "id": "abc", "function": {"name": "runner_status", "arguments": "{}"}}]})
        assert [(c.name, c.id, c.source) for c in found.calls] == [
            ("runner_status", "abc", "structured")]

    def test_hermes_and_qwen(self):
        found = extract_calls('<tool_call>{"name": "web_search", "arguments": {"q": "x"}}</tool_call>')
        assert found.calls[0].name == "web_search"
        assert found.calls[0].source == "hermes"

    def test_llama_python_tag_with_parameters(self):
        """Llama says `parameters`, not `arguments`."""
        found = extract_calls('<|python_tag|>{"name": "web_search", "parameters": {"q": "y"}}<|eom_id|>')
        assert found.calls[0].arguments == {"q": "y"}

    def test_mistral_list(self):
        found = extract_calls('[TOOL_CALLS][{"name": "web_search", "arguments": {"q": "a"}},'
                              ' {"name": "runner_status", "arguments": {}}]')
        assert [c.name for c in found.calls] == ["web_search", "runner_status"]

    def test_several_hermes_calls_in_one_reply(self):
        text = ('<tool_call>{"name": "runner_status", "arguments": {}}</tool_call>\n'
                '<tool_call>{"name": "web_search", "arguments": {"q": "b"}}</tool_call>')
        assert len(extract_calls(text).calls) == 2

    def test_fenced_json_when_the_tool_is_offered(self):
        found = extract_calls('```json\n{"name": "web_search", "arguments": {"q": "c"}}\n```',
                              tools=TOOLS)
        assert found.calls[0].source == "fenced"

    def test_arguments_as_a_json_string(self):
        found = extract_calls('<tool_call>{"name": "web_search", "arguments": "{\\"q\\": \\"d\\"}"}</tool_call>')
        assert found.calls[0].arguments == {"q": "d"}

    def test_the_openai_function_wrapper(self):
        found = extract_calls('<tool_call>{"function": {"name": "runner_status", "arguments": {}}}</tool_call>')
        assert found.calls[0].name == "runner_status"

    def test_an_unclosed_hermes_tag_at_the_end(self):
        """A model cut off at max_tokens before writing </tool_call>."""
        found = extract_calls('<tool_call>{"name": "runner_status", "arguments": {}}')
        assert found.calls[0].name == "runner_status"


class TestWhatIsNotACall:
    def test_a_json_answer_is_not_executed(self):
        """The person asked for a config file. It has a `name` key. It is
        not a tool call, and executing it would be an action nobody asked
        for."""
        text = 'Here you go:\n```json\n{"name": "my-app", "arguments": {"x": 1}}\n```'
        assert extract_calls(text, tools=TOOLS).calls == []

    def test_fenced_json_without_a_tool_list_is_not_read(self):
        """With nothing offered there is nothing a fenced block could
        safely be calling."""
        text = '```json\n{"name": "web_search", "arguments": {"q": "c"}}\n```'
        assert extract_calls(text).calls == []

    def test_a_bare_object_naming_an_unoffered_tool(self):
        assert extract_calls('{"name": "delete_everything", "arguments": {}}', tools=TOOLS).calls == []

    def test_prose_mentioning_the_tag(self):
        assert extract_calls("I would use a <tool_call> here if I could.").calls == []


class TestRepair:
    @pytest.mark.parametrize("raw,expected", [
        ('{"a": 1,}', {"a": 1}),
        ('{"a": [1, 2,],}', {"a": [1, 2]}),
        ('{"a": True, "b": None}', {"a": True, "b": None}),
        ("{'a': 'b'}", {"a": "b"}),
        ('{"a": {"b": 1}', {"a": {"b": 1}}),
    ])
    def test_the_mistakes_models_make(self, raw, expected):
        value, repaired = repair_json(raw)
        assert value == expected and repaired

    def test_valid_json_is_not_marked_repaired(self):
        assert repair_json('{"a": 1}') == ({"a": 1}, False)

    def test_true_inside_a_string_is_left_alone(self):
        """Only literals outside strings are Python's. "True story" is
        somebody's text."""
        value, _ = repair_json('{"title": "True story", "x": 1,}')
        assert value["title"] == "True story"

    def test_mixed_quotes_are_not_guessed_at(self):
        """An apostrophe inside a double-quoted value is not a string
        boundary, and swapping every ' for " would make it one."""
        value, _ = repair_json('{"q": "it\'s fine",}')
        assert value == {"q": "it's fine"}

    def test_garbage_is_refused(self):
        with pytest.raises(ValueError):
            repair_json("not json at all")

    def test_an_unreadable_call_is_reported_not_dropped(self):
        """So the model can be told and try again."""
        found = extract_calls("<tool_call>{name: web_search arguments</tool_call>")
        assert found.calls == [] and found.failures


class TestValidation:
    def check(self, name, args):
        return validate_arguments(ToolCall(name, args), TOOLS)

    def test_good_arguments(self):
        assert self.check("web_search", {"q": "x"})[1] is None

    def test_a_missing_required_field_is_named(self):
        _, problem = self.check("web_search", {})
        assert "`q` is required" in problem
        assert problem.startswith("R2-02055.a2")

    def test_unambiguous_strings_are_coerced(self):
        args, problem = self.check("web_search", {"q": "x", "depth": "3", "safe": "true"})
        assert problem is None
        assert args == {"q": "x", "depth": 3, "safe": True}

    def test_a_wrong_type_says_what_arrived(self):
        """"invalid arguments" makes a model guess; this lets it fix it."""
        _, problem = self.check("web_search", {"q": "x", "depth": "ten"})
        assert "`depth` must be integer" in problem and "ten" in problem

    def test_a_boolean_is_not_an_integer(self):
        """True is an int in Python and not a valid integer in JSON."""
        _, problem = self.check("web_search", {"q": "x", "depth": True})
        assert problem and "depth" in problem

    def test_enum(self):
        _, problem = self.check("web_search", {"q": "x", "mode": "turbo"})
        assert "one of" in problem

    def test_an_unknown_field_is_refused_when_the_schema_says_so(self):
        _, problem = self.check("web_search", {"q": "x", "limit": 5})
        assert "`limit` is not a parameter" in problem

    def test_a_misspelt_tool_is_suggested_not_run(self):
        """Auto-correcting a tool name is how a typo becomes an action."""
        _, problem = self.check("web_serch", {"q": "x"})
        assert problem.startswith("R2-02050.a2")
        assert "Did you mean web_search" in problem


class TestPrompt:
    def test_it_lists_every_tool_and_shows_one_real_example(self):
        """A model copies an example more reliably than it follows a
        description, and an example using a real tool cannot teach it to
        call one that does not exist."""
        text = tool_prompt(TOOLS)
        assert "web_search(q: string" in text and "runner_status()" in text
        example = text.split("<tool_call>")[1].split("</tool_call>")[0]
        parsed = json.loads(example)
        assert parsed["name"] == "web_search" and "q" in parsed["arguments"]

    def test_the_example_round_trips_through_the_extractor(self):
        text = tool_prompt(TOOLS)
        example = "<tool_call>" + text.split("<tool_call>")[1].split("</tool_call>")[0] + "</tool_call>"
        assert extract_calls(example).calls[0].name == "web_search"

    def test_no_tools_no_prompt(self):
        assert tool_prompt([]) == ""


# ---------------------------------------------------------------------------
# T1 as tools, on the wire
# ---------------------------------------------------------------------------


class _Recorder(BaseHTTPRequestHandler):
    seen: list[dict] = []
    status = 200

    def _reply(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else ""
        type(self).seen.append({"method": self.command, "path": self.path,
                                "auth": self.headers.get("Authorization"),
                                "body": body})
        payload = json.dumps({"ok": True, "path": self.path, "request_id": "r1"})
        if type(self).status != 200:
            payload = json.dumps({"error": {"code": "AUTH_ADMIN_REQUIRED",
                                            "hx_code": "T1-05021.a3",
                                            "message": "needs admin"}})
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    do_GET = do_POST = _reply

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Recorder.seen = []
    _Recorder.status = 200
    httpd = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


class TestT1Tools:
    def test_the_callers_key_is_forwarded(self, server):
        """Never a server key: a model can do what the person could do by
        hand and no more."""
        t1tools.call_tool("runner_status", {}, base_url=server, token="T1_person")
        assert _Recorder.seen[0]["auth"] == "Bearer T1_person"

    def test_get_arguments_go_in_the_query(self, server):
        t1tools.call_tool("web_search", {"q": "cats", "depth": 2}, base_url=server)
        assert _Recorder.seen[0]["path"] == "/web/v1/search?q=cats&depth=2"

    def test_post_arguments_go_in_the_body(self, server):
        t1tools.call_tool("web_summarize", {"url": "https://a.test"}, base_url=server)
        assert json.loads(_Recorder.seen[0]["body"]) == {"url": "https://a.test"}

    def test_arguments_the_tool_does_not_take_are_dropped(self, server):
        t1tools.call_tool("web_search", {"q": "x", "admin": True}, base_url=server)
        assert "admin" not in _Recorder.seen[0]["path"]

    def test_request_id_is_not_shown_to_the_model(self, server):
        out = t1tools.call_tool("runner_status", {}, base_url=server)
        assert "request_id" not in out

    def test_an_http_error_is_a_result_not_an_exception(self, server):
        """A 403 is something the model can relay to the person."""
        _Recorder.status = 403
        out = t1tools.call_tool("runner_status", {}, base_url=server)
        assert out.startswith("HTTP 403") and "T1-05021.a3" in out

    def test_an_unreachable_server_is_a_result(self):
        out = t1tools.call_tool("runner_status", {}, base_url="http://127.0.0.1:9", timeout=2)
        assert out.startswith("T1-04140.d3")

    def test_mutating_tools_are_hidden_by_default(self):
        names = {t.name for t in t1tools.catalogue()}
        assert "runner_unload" not in names and "memory_create" not in names
        assert "runner_status" in names

    def test_and_refused_even_if_named(self, server):
        """Hiding is not enforcing: a model can name a tool it was never
        shown."""
        out = t1tools.call_tool("runner_unload", {}, base_url=server)
        assert out.startswith("R2-05060.a3")
        assert _Recorder.seen == []

    def test_allowed_when_enabled(self, server):
        t1tools.call_tool("runner_unload", {}, base_url=server, allow_mutating=True)
        assert _Recorder.seen[0]["method"] == "POST"

    def test_every_tool_names_a_route_the_server_has(self):
        """A catalogue entry pointing at a route that does not exist is a
        tool that 404s every time it is used."""
        pytest.importorskip("fastapi")
        from hypernix.t1api.routers import ALL_ROUTERS

        routes = {(m, r.path) for router in ALL_ROUTERS for r in router.routes
                  for m in getattr(r, "methods", []) or []}
        missing = [t.name for t in t1tools.TOOLS if (t.method, t.path) not in routes]
        assert missing == []

    def test_every_schema_is_strict(self):
        for tool in t1tools.openai_tools(allow_mutating=True):
            assert tool["function"]["parameters"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Through the loop
# ---------------------------------------------------------------------------


def _envelope(content="", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


class TestTheLoop:
    @pytest.fixture
    def context(self, tmp_path):
        from hypernix.interfaces.noodle.tools import ToolContext
        return ToolContext(root=str(tmp_path))

    def run(self, context, replies, **kwargs):
        from hypernix.hyperlink.toolloop import run_tool_loop

        queue = list(replies)
        asked = []

        def ask(messages, tools):
            asked.append((list(messages), tools))
            return queue.pop(0)

        result = run_tool_loop([{"role": "user", "content": "hi"}], context,
                               ask=ask, extract=lambda e: ("", "stop"), **kwargs)
        return result, asked

    def test_a_text_written_call_now_runs(self, context):
        """The whole point. Before this, the markup was the reply."""
        (messages, final, rounds), _ = self.run(context, [
            _envelope('<tool_call>{"name": "create_todo", "arguments": {"text": "buy milk"}}</tool_call>'),
            _envelope("Added it."),
        ])
        assert [r.tool for r in rounds] == ["create_todo"]
        assert rounds[0].ok
        assert final["choices"][0]["message"]["content"] == "Added it."

    def test_the_tool_reply_matches_the_call_id(self, context):
        (messages, _, _), _ = self.run(context, [
            _envelope('<tool_call>{"name": "create_todo", "arguments": {"text": "x"}}</tool_call>'),
            _envelope("ok"),
        ])
        call_id = messages[1]["tool_calls"][0]["id"]
        assert messages[2]["role"] == "tool" and messages[2]["tool_call_id"] == call_id

    def test_an_unreadable_call_is_sent_back_once(self, context):
        (messages, final, rounds), asked = self.run(context, [
            _envelope("<tool_call>{name: create_todo args</tool_call>"),
            _envelope("Sorry, never mind."),
        ])
        assert len(asked) == 2
        assert "could not be read" in asked[1][0][-1]["content"]
        assert final["choices"][0]["message"]["content"] == "Sorry, never mind."

    def test_a_schema_error_goes_back_to_the_model(self, context):
        (_, _, rounds), _ = self.run(context, [
            _envelope('<tool_call>{"name": "create_todo", "arguments": {}}</tool_call>'),
            _envelope("ok"),
        ])
        assert rounds[0].code == "invalid_arguments"
        assert "`text` is required" in rounds[0].content

    def test_teach_format_prepends_the_prompt(self, context):
        _, asked = self.run(context, [_envelope("hello")], teach_format=True)
        assert asked[0][0][0]["role"] == "system"
        assert "<tool_call>" in asked[0][0][0]["content"]

    def test_t1_tools_are_offered_and_called(self, context, server):
        from hypernix.hyperlink.toolloop import T1Access

        (_, _, rounds), asked = self.run(context, [
            _envelope('<tool_call>{"name": "runner_status", "arguments": {}}</tool_call>'),
            _envelope("It's running."),
        ], t1=T1Access(server, "T1_person"))
        names = {t["function"]["name"] for t in asked[0][1]}
        assert "runner_status" in names
        assert rounds[0].tool == "runner_status" and rounds[0].ok
        assert _Recorder.seen[0]["auth"] == "Bearer T1_person"

    def test_a_name_clash_keeps_the_existing_tool(self, context, server):
        """noodle and T1 both have web_search; a model shown two tools
        with one name calls whichever it saw last."""
        from hypernix.hyperlink.toolloop import T1Access

        context.allow_web_search = True
        _, asked = self.run(context, [_envelope("hi")], t1=T1Access(server))
        names = [t["function"]["name"] for t in asked[0][1]]
        assert names.count("web_search") == 1
