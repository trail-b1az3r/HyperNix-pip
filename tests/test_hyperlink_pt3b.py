"""HyperLink 0.72.6 pt3: edit resends, memories reach the screen, search finds things.

Three bugs, all of the same kind: the feature ran, reported success,
and produced nothing the person could see.

* **Editing never re-asked.** The edit rewrote the message and removed
  everything after it — and stopped, leaving the thread ending on an
  unanswered question.
* **Memories went somewhere no screen reads.** The model's memory tool
  wrote a JSON file in its workspace. It said "I'll remember that"; the
  Memories screen, which reads the real store, never showed it.
* **Web search searched almost nothing.** The tool used DuckDuckGo's
  instant-answer API, which answers "capital of France" and comes back
  empty for nearly every real question.

These run through the real API with a scripted model, because each bug
lived in the seam between two pieces that were fine on their own.
"""
from __future__ import annotations

import pytest
from conftest import clear_t1_config

pytest.importorskip("fastapi")


class Loaded:
    model_id = "local-gguf"


class FakeRunner:
    base_url = "http://127.0.0.1:1"
    current = Loaded()


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, *, model=None, tools=None, **kwargs):
        self.seen.append({"messages": list(messages), "tools": tools})
        content = self.replies.pop(0) if self.replies else "ok"
        return {"choices": [{"message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}], "model": model, "usage": {}}

    def stream_chat(self, messages, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")
    monkeypatch.setenv("T1_NOODLE_ENABLED", "1")
    from hypernix.t1api.app import create_app

    app = create_app()
    app.state.t1_runner = FakeRunner()
    return app, TestClient(app, client=("192.168.1.9", 5000))


def use_model(monkeypatch, model):
    from hypernix.hyperlink import inference

    monkeypatch.setattr(inference, "_openai_client",
                        lambda base_url, timeout=300.0: model)


def new_session(client) -> str:
    return client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]


def messages(client, session):
    return client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]


# ---------------------------------------------------------------------------
# Regenerate: what edit and resend both stand on
# ---------------------------------------------------------------------------


class TestRegenerate:
    def test_it_answers_the_last_message_without_adding_one(self, app_client, monkeypatch):
        _app, client = app_client
        use_model(monkeypatch, ScriptedModel("first answer", "second answer"))
        session = new_session(client)
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hello"})
        first = client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]
        user = next(m for m in first if m["role"] == "user")
        assistant = next(m for m in first if m["role"] == "assistant")

        # Edit with the same text: removes the reply, leaves the question last.
        edited = client.patch(
            f"/hyperlink/sessions/{session}/messages/{user['message_id']}",
            json={"content": "hello"})
        assert edited.status_code == 200, edited.text

        again = client.post(f"/hyperlink/sessions/{session}/chat",
                            json={"regenerate": True})
        assert again.status_code == 200, again.text
        after = messages(client, session)
        users = [m for m in after if m["role"] == "user"]
        replies = [m for m in after if m["role"] == "assistant"]
        # One question, one answer — the new one. Sending the text again
        # would have put the question in the thread twice.
        assert len(users) == 1
        assert [r["content"] for r in replies] == ["second answer"]
        assert assistant["message_id"] not in {r["message_id"] for r in replies}

    def test_it_refuses_when_the_thread_ends_in_a_reply(self, app_client, monkeypatch):
        """Answering again would stack a second reply under the first.
        The refusal says to edit instead."""
        _app, client = app_client
        use_model(monkeypatch, ScriptedModel("an answer"))
        session = new_session(client)
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hi"})
        got = client.post(f"/hyperlink/sessions/{session}/chat", json={"regenerate": True})
        assert got.status_code == 409
        assert "edit it" in got.text

    def test_an_empty_thread_has_nothing_to_regenerate(self, app_client, monkeypatch):
        _app, client = app_client
        use_model(monkeypatch, ScriptedModel())
        session = new_session(client)
        got = client.post(f"/hyperlink/sessions/{session}/chat", json={"regenerate": True})
        assert got.status_code == 409

    def test_without_regenerate_empty_content_is_still_refused(self, app_client, monkeypatch):
        _app, client = app_client
        use_model(monkeypatch, ScriptedModel())
        session = new_session(client)
        got = client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "  "})
        assert got.status_code in (400, 422)

    def test_the_streaming_route_regenerates_too(self, app_client, monkeypatch):
        """The app streams. Supporting it only on the plain route would
        have been a fix the phone never used."""
        _app, client = app_client
        use_model(monkeypatch, ScriptedModel("x"))
        session = new_session(client)
        got = client.post(f"/hyperlink/sessions/{session}/chat/stream",
                          json={"regenerate": True})
        assert got.status_code == 409   # nothing to answer yet — but it read the flag


# ---------------------------------------------------------------------------
# Memories the model writes are memories the screen shows
# ---------------------------------------------------------------------------


class TestModelMemories:
    def enable(self, client, auto_memory=True):
        got = client.patch("/hyperlink/preferences",
                           json={"tools_enabled": True, "auto_memory": auto_memory})
        assert got.status_code == 200, got.text

    def test_what_the_model_remembers_appears_in_memory_list(self, app_client, monkeypatch):
        _app, client = app_client
        use_model(monkeypatch, ScriptedModel(
            '<tool_call>{"name": "update_memory", "arguments": '
            '{"key": "favourite editor", "value": "Helix"}}</tool_call>',
            "Noted — Helix.",
        ))
        self.enable(client)
        session = new_session(client)
        reply = client.post(f"/hyperlink/sessions/{session}/chat",
                            json={"content": "I use Helix now, remember that"})
        assert reply.status_code == 200, reply.text

        listed = client.get("/memory/list").json()["memories"]
        found = [m for m in listed if m["category"] == "favourite editor"]
        assert found and found[0]["content"] == "Helix"
        # Marked as the model's, so "why does it think that" has an answer.
        assert found[0]["source"] == "auto"

    def test_remembering_again_updates_rather_than_duplicates(self, app_client, monkeypatch):
        _app, client = app_client
        call = ('<tool_call>{{"name": "update_memory", "arguments": '
                '{{"key": "editor", "value": "{v}"}}}}</tool_call>')
        use_model(monkeypatch, ScriptedModel(call.format(v="Vim"), "ok",
                                             call.format(v="Helix"), "ok"))
        self.enable(client)
        session = new_session(client)
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "a"})
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "b"})
        editors = [m for m in client.get("/memory/list").json()["memories"]
                   if m["category"] == "editor"]
        assert [m["content"] for m in editors] == ["Helix"]

    def test_the_person_switching_auto_memory_off_is_respected(self, app_client, monkeypatch):
        """Whether an assistant keeps notes about you is yours to decide."""
        _app, client = app_client
        model = ScriptedModel("hello")
        use_model(monkeypatch, model)
        self.enable(client, auto_memory=False)
        session = new_session(client)
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hi"})
        offered = {t["function"]["name"] for t in model.seen[0]["tools"] or []}
        assert "update_memory" not in offered

    def test_the_model_cannot_forget_what_the_person_wrote(self, tmp_path):
        from hypernix.hyperlink.memory import MemoryStore, ToolMemoryBackend
        from hypernix.t1api.db import SQLiteBackend

        store = MemoryStore(SQLiteBackend(tmp_path / "m.sqlite3"))
        store.create(owner="me", content="Allergic to peanuts", category="health")
        backend = ToolMemoryBackend(store, "me")
        assert backend.forget("health") is False
        assert [m.content for m in store.list(owner="me")] == ["Allergic to peanuts"]


# ---------------------------------------------------------------------------
# Web search that finds things
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def outcome(self, hits, status="ok"):
        from hypernix.t1api.websearch import SearchHit, SearchOutcome

        return SearchOutcome("q", [SearchHit(t, u, s, "duckduckgo") for t, u, s in hits],
                             engine="duckduckgo", status=status)

    def test_ranked_results_are_used(self, tmp_path):
        from hypernix.interfaces.noodle.tools import ToolContext, run_tool

        ctx = ToolContext(root=tmp_path, search_backend=lambda q: self.outcome(
            [("Helix editor", "https://helix-editor.com", "A post-modern editor")]))
        result = run_tool(ctx, "web_search", {"query": "helix"})
        assert result.ok
        assert "helix-editor.com" in result.content
        assert "post-modern" in result.content

    def test_no_ranked_results_falls_back(self, tmp_path, monkeypatch):
        """To the instant-answer API — not to an error. A thin answer is
        better than none, and the fallback says it is thin."""
        from hypernix.interfaces.noodle import tools

        called = {}

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return b'{"AbstractText": "Paris.", "AbstractURL": "u"}'

        def fake_urlopen(req, timeout=0):
            called["url"] = req.full_url
            return Resp()

        monkeypatch.setattr(tools.urllib.request, "urlopen", fake_urlopen)
        ctx = tools.ToolContext(root=tmp_path, search_backend=lambda q: self.outcome([], "failed"))
        result = tools.run_tool(ctx, "web_search", {"query": "capital of france"})
        assert "api.duckduckgo.com" in called["url"]
        assert "Paris" in result.content

    def test_a_backend_that_raises_falls_back_too(self, tmp_path, monkeypatch):
        from hypernix.interfaces.noodle import tools

        def explode(q):
            raise RuntimeError("DNS")

        monkeypatch.setattr(tools.urllib.request, "urlopen",
                            lambda req, timeout=0: (_ for _ in ()).throw(OSError("offline")))
        ctx = tools.ToolContext(root=tmp_path, search_backend=explode)
        result = tools.run_tool(ctx, "web_search", {"query": "x"})
        assert not result.ok        # both failed: reported, not raised

    def test_still_off_when_switched_off(self, tmp_path):
        from hypernix.interfaces.noodle.tools import ToolContext, run_tool

        ctx = ToolContext(root=tmp_path, allow_web_search=False,
                          search_backend=lambda q: self.outcome([("a", "b", "c")]))
        assert not run_tool(ctx, "web_search", {"query": "x"}).ok
