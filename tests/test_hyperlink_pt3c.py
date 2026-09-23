"""HyperLink: model titles, image capability, the opt-in shell, context
compression, and memory organisation (0.72.5.post16).

The chat-level tests go through the real API with a scripted model, so a
title or a compaction is what the app would actually receive.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink import capabilities, titles
from hypernix.hyperlink.memory import (
    DEFAULT_TOPIC,
    MemoryStore,
    ToolMemoryBackend,
    categories,
    organise,
    rename_category,
    suggest_category,
)

# ---------------------------------------------------------------------------
# Image capability
# ---------------------------------------------------------------------------


class TestCapabilities:
    @pytest.mark.parametrize("name", [
        "Qwen2.5-VL-7B-Instruct", "qwen3-vl-8b", "llava-v1.6-mistral-7b", "gemma-3-12b-it",
        "pixtral-12b", "MiniCPM-V-2_6", "moondream2", "Llama-3.2-11B-Vision-Instruct",
        "mistral-small-3.1-24b-instruct", "Phi-3.5-vision-instruct", "InternVL2-8B",
        "qwen3.5-9b",
    ])
    def test_vision_families(self, name):
        assert capabilities.supports_images(name) is True

    @pytest.mark.parametrize("name", ["gemma-3-1b-it", "gemma-3n-e4b", "qwen3_5_text"])
    def test_text_only_members_of_vision_families(self, name):
        assert capabilities.supports_images(name) is False

    @pytest.mark.parametrize("name", ["llama-3.1-8b-instruct", "gpt-oss-20b", "mistral-7b"])
    def test_unknown_is_unknown_not_no(self, name):
        """Guessing no would hide the button from a model that can see."""
        assert capabilities.supports_images(name) is None

    def test_the_runtime_wins(self):
        assert capabilities.supports_images("llama-3.1-8b", runtime_says=True) is True
        assert capabilities.supports_images("qwen2.5-vl-7b", runtime_says=False) is False

    def test_a_gguf_sees_only_with_its_projector(self, tmp_path):
        model = tmp_path / "Qwen2.5-VL-7B-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        assert capabilities.supports_images("Qwen2.5-VL-7B", path=model) is False
        (tmp_path / "mmproj-Qwen2.5-VL-7B-f16.gguf").write_bytes(b"GGUF")
        assert capabilities.supports_images("Qwen2.5-VL-7B", path=model) is True
        # A projector makes even an unrecognised name a vision model.
        other = tmp_path / "custom.gguf"
        other.write_bytes(b"GGUF")
        assert capabilities.supports_images("custom", path=other) is True

    def test_the_catalogue_reports_it(self):
        from hypernix.hyperlink.catalogue import CatalogueModel

        assert CatalogueModel("m", "m", "lmstudio", vision=True).to_dict()["supports_images"] is True
        assert CatalogueModel("qwen2-vl", "qwen2-vl", "registry").to_dict()["supports_images"] is True
        assert CatalogueModel("llama3", "llama3", "registry").to_dict()["supports_images"] is None


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


class TestTitles:
    @pytest.mark.parametrize(("raw", "expected"), [
        ('Title: "Fixing the Wi-Fi"', "Fixing the Wi-Fi"),
        ("**Rust borrow checker help**", "Rust borrow checker help"),
        ("<think>short</think>\nTrip to Lisbon.", "Trip to Lisbon"),
        ("# Sourdough starter tips\n\nMore text", "Sourdough starter tips"),
        ("“Quoted title”", "Quoted title"),
    ])
    def test_cleaning(self, raw, expected):
        assert titles.clean_title(raw) == expected

    @pytest.mark.parametrize("raw", [
        "", "   ", "Sure! Here is a title for you", "I think this conversation is about cooking",
        "As an AI I cannot", " ".join(["word"] * 20),
    ])
    def test_not_a_title(self, raw):
        assert titles.clean_title(raw) == ""

    def test_long_titles_are_cut(self):
        assert len(titles.clean_title("A " + "very " * 11 + "long")) <= titles.MAX_TITLE

    def test_the_model_title_is_used(self):
        seen = []

        def complete(messages):
            seen.append(messages)
            return "Planning a Lisbon trip"

        title, how = titles.make_title("where should I stay in lisbon?", "Alfama is lovely.", complete)
        assert (title, how) == ("Planning a Lisbon trip", "model")
        # The conversation is data in the prompt, not instructions.
        assert "<conversation>" in seen[0][-1]["content"]

    @pytest.mark.parametrize("complete", [None, lambda m: "Sure, here's a title!", lambda m: 1 / 0])
    def test_anything_unusable_falls_back_to_the_first_line(self, complete):
        title, how = titles.make_title("fix my wifi please\nit keeps dropping", "Try…", complete)
        assert (title, how) == ("fix my wifi please", "first-line")


# ---------------------------------------------------------------------------
# Memory organisation
# ---------------------------------------------------------------------------


@pytest.fixture
def memories(tmp_path, monkeypatch):
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "m.sqlite3"))
    from hypernix.t1api.storage import SQLiteBackend

    return MemoryStore(SQLiteBackend(str(tmp_path / "m.sqlite3")))


class TestMemoryOrganisation:
    @pytest.mark.parametrize(("text", "topic"), [
        ("favourite editor Helix", "Tech"),
        ("home city Lisbon", "Places"),
        ("works at Acme as a manager", "Work"),
        ("dog's name Rex", "About you"),
        ("allergic to peanuts", "Health"),
        ("likes short answers", "Preferences"),
        ("the sky was grey", DEFAULT_TOPIC),
    ])
    def test_topics(self, text, topic):
        assert suggest_category(text) == topic

    def test_the_model_files_facts_under_topics(self, memories):
        tool = ToolMemoryBackend(memories, owner="me", session_id="s1")
        tool.set("favourite editor", "Helix")
        tool.set("home city", "Lisbon")
        stored = {m.metadata["key"]: m for m in memories.list(owner="me")}
        assert stored["favourite editor"].category == "Tech"
        assert stored["favourite editor"].content == "favourite editor: Helix"
        assert stored["home city"].category == "Places"
        # Two facts, two topics — not one category per fact.
        assert {c["name"] for c in categories(memories, owner="me")} == {"Tech", "Places"}

    def test_a_fact_is_updated_in_place(self, memories):
        tool = ToolMemoryBackend(memories, owner="me")
        tool.set("favourite editor", "Vim")
        tool.set("favourite editor", "Helix")
        mine = memories.list(owner="me")
        assert len(mine) == 1 and mine[0].content == "favourite editor: Helix"
        assert tool.load()["favourite editor"]["value"] == "Helix"

    def test_an_old_per_key_memory_is_found_and_refiled(self, memories):
        # The filing before topics: key as category, bare value as content.
        memories.create(owner="me", content="Helix", category="favourite editor", source="auto")
        tool = ToolMemoryBackend(memories, owner="me")
        tool.set("favourite editor", "Zed")
        [only] = memories.list(owner="me")
        assert only.content == "favourite editor: Zed"
        # Under a topic now, not under its own key.
        assert only.category == suggest_category("favourite editor Zed") != "favourite editor"
        assert only.metadata["key"] == "favourite editor"

    def test_forget_only_removes_what_the_model_wrote(self, memories):
        memories.create(owner="me", content="favourite editor: Emacs", category="Tech",
                        source="manual", metadata={"key": "favourite editor"})
        tool = ToolMemoryBackend(memories, owner="me")
        tool.set("favourite editor", "Helix")
        assert tool.forget("favourite editor")
        assert [m.content for m in memories.list(owner="me")] == ["favourite editor: Emacs"]

    def test_organise_refiles_old_auto_memories_and_leaves_a_persons_alone(self, memories):
        memories.create(owner="me", content="Lisbon", category="home city", source="auto")
        memories.create(owner="me", content="loose note about python", source="manual")
        memories.create(owner="me", content="call mum on sundays", category="Family stuff", source="manual")
        preview = organise(memories, owner="me", dry_run=True)
        assert {c["to"] for c in preview} == {"Places", "Tech"}
        assert all(m.category in ("home city", "", "Family stuff") for m in memories.list(owner="me"))
        organise(memories, owner="me")
        by_content = {m.content: m for m in memories.list(owner="me")}
        assert by_content["home city: Lisbon"].category == "Places"
        assert by_content["loose note about python"].category == "Tech"
        # A category a person chose is theirs.
        assert by_content["call mum on sundays"].category == "Family stuff"
        assert organise(memories, owner="me") == []

    def test_rename_merges(self, memories):
        memories.create(owner="me", content="a", category="Work")
        memories.create(owner="me", content="b", category="Job")
        assert rename_category(memories, owner="me", old="Job", new="Work") == 1
        assert categories(memories, owner="me") == [{"name": "Work", "count": 2, "pinned": 0, "auto": 0}]

    def test_rename_needs_a_name(self, memories):
        from hypernix.t1api.errors import T1APIError

        with pytest.raises(T1APIError):
            rename_category(memories, owner="me", old="Work", new="  ")

    def test_owners_do_not_mix(self, memories):
        memories.create(owner="a", content="x", category="Work")
        assert categories(memories, owner="b") == []
        assert rename_category(memories, owner="b", old="Work", new="Other") == 0


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


class TestPreferences:
    def test_defaults_and_patch(self, tmp_path):
        from hypernix.hyperlink.preferences import PreferenceStore
        from hypernix.t1api.storage import SQLiteBackend

        store = PreferenceStore(SQLiteBackend(str(tmp_path / "p.sqlite3")))
        assert store.get(owner="me").auto_compact and store.get(owner="me").model_titles
        store.save(owner="me", auto_compact=False, model_titles=False)
        got = store.get(owner="me")
        assert not got.auto_compact and not got.model_titles
        assert got.to_dict()["auto_compact"] is False

    def test_an_old_database_is_migrated(self, tmp_path):
        """A server upgraded in place has a table without the columns."""
        from hypernix.hyperlink.preferences import PreferenceStore
        from hypernix.t1api.storage import SQLiteBackend

        path = tmp_path / "old.sqlite3"
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE hyperlink_preferences (
                owner TEXT PRIMARY KEY, display_name TEXT NOT NULL DEFAULT '',
                bio TEXT NOT NULL DEFAULT '', system_prompt TEXT NOT NULL DEFAULT '',
                effort TEXT NOT NULL DEFAULT 'medium', context_minimum INTEGER NOT NULL DEFAULT 0,
                context_maximum INTEGER NOT NULL DEFAULT 0, backup_model TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL DEFAULT '', tools_enabled INTEGER NOT NULL DEFAULT 0,
                auto_memory INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, metadata TEXT NOT NULL DEFAULT '{}');
            INSERT INTO hyperlink_preferences (owner, created_at, updated_at) VALUES ('me', 1, 1);
        """)
        conn.commit()
        conn.close()
        store = PreferenceStore(SQLiteBackend(str(path)))
        assert store.get(owner="me").auto_compact is True
        store.save(owner="me", model_titles=False)
        assert store.get(owner="me").model_titles is False


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------


pytest.importorskip("fastapi")


class Loaded:
    model_id = "local-gguf"


class FakeRunner:
    base_url = "http://127.0.0.1:1"
    current = Loaded()


class ScriptedModel:
    """Answers chat with scripted replies; titles and summaries by prompt."""

    base_url = "http://127.0.0.1:1/v1"

    def __init__(self, *replies, title="Planning a Lisbon trip", summary="Earlier: they chose Alfama."):
        self.replies = list(replies)
        self.title = title
        self.summary = summary
        self.calls = []

    def _answer(self, messages):
        system = str(messages[0].get("content", "")) if messages else ""
        if "You name conversations" in system:
            self.calls.append("title")
            return self.title
        last = str(messages[-1].get("content", ""))
        if "summar" in system.lower() or "summar" in last.lower()[:400]:
            self.calls.append("summary")
            return self.summary
        self.calls.append("chat")
        return self.replies.pop(0) if self.replies else "ok"

    def chat(self, messages, *, model=None, tools=None, **kwargs):
        content = self._answer(messages)
        return {"choices": [{"message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}], "model": model or "local-gguf", "usage": {}}

    def chat_stream(self, messages, **kwargs):
        content = self._answer(messages)

        def chunks():
            for word in content.split(" "):
                yield {"choices": [{"delta": {"content": word + " "}}], "model": "local-gguf"}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "local-gguf"}
        return chunks()


@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")

    def make(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from hypernix.t1api.app import create_app

        app = create_app()
        app.state.t1_runner = FakeRunner()
        return TestClient(app, client=("192.168.1.9", 5000))
    return make


def use_model(monkeypatch, model):
    from hypernix.hyperlink import inference

    monkeypatch.setattr(inference, "_openai_client", lambda base_url, timeout=300.0: model)


def session_title(client, session):
    return client.get(f"/hyperlink/sessions/{session}").json()["session"]["title"]


class TestTitlesThroughTheAPI:
    def test_the_model_names_a_new_chat(self, api, monkeypatch):
        client = api()
        model = ScriptedModel("Alfama, probably.")
        use_model(monkeypatch, model)
        session = client.post("/hyperlink/sessions", json={}).json()["session"]["session_id"]
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "where to stay in lisbon"})
        assert session_title(client, session) == "Planning a Lisbon trip"
        # Named once: the second turn does not rename it.
        model.title = "Something else"
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "and food?"})
        assert session_title(client, session) == "Planning a Lisbon trip"
        assert model.calls.count("title") == 1

    def test_turned_off_it_uses_the_first_line(self, api, monkeypatch):
        client = api()
        model = ScriptedModel("ok")
        use_model(monkeypatch, model)
        client.patch("/hyperlink/preferences", json={"model_titles": False})
        session = client.post("/hyperlink/sessions", json={}).json()["session"]["session_id"]
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "where to stay in lisbon"})
        assert session_title(client, session) == "where to stay in lisbon"
        assert "title" not in model.calls

    def test_a_named_chat_is_left_alone(self, api, monkeypatch):
        client = api()
        use_model(monkeypatch, ScriptedModel("ok"))
        session = client.post("/hyperlink/sessions", json={"title": "Mine"}).json()["session"]["session_id"]
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hi"})
        assert session_title(client, session) == "Mine"

    def test_retitle(self, api, monkeypatch):
        client = api()
        model = ScriptedModel("ok")
        use_model(monkeypatch, model)
        session = client.post("/hyperlink/sessions", json={"title": "Mine"}).json()["session"]["session_id"]
        client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "where to stay in lisbon"})
        got = client.post(f"/hyperlink/sessions/{session}/title")
        assert got.status_code == 200 and got.json()["session"]["title"] == "Planning a Lisbon trip"

    def test_the_stream_sends_the_title_after_done(self, api, monkeypatch):
        client = api()
        use_model(monkeypatch, ScriptedModel("Alfama is lovely"))
        session = client.post("/hyperlink/sessions", json={}).json()["session"]["session_id"]
        with client.stream("POST", f"/hyperlink/sessions/{session}/chat/stream",
                           json={"content": "where to stay in lisbon"}) as response:
            frames = [json.loads(line[5:]) for line in response.iter_lines()
                      if line.startswith("data:") and line[5:].strip() != "[DONE]"]
        kinds = [f.get("type") for f in frames]
        assert kinds.index("title") > kinds.index("done")
        assert frames[kinds.index("title")]["title"] == "Planning a Lisbon trip"


class TestCompressionThroughTheAPI:
    def fill(self, client, session, n=12):
        for i in range(n):
            client.post(f"/hyperlink/sessions/{session}/chat",
                        json={"content": f"message {i} " + "words " * 80, "token_budget": 100_000})

    def test_a_thread_that_no_longer_fits_is_summarised_not_dropped(self, api, monkeypatch):
        client = api()
        model = ScriptedModel(*(["answer " * 60] * 20))
        use_model(monkeypatch, model)
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        self.fill(client, session)
        got = client.post(f"/hyperlink/sessions/{session}/chat",
                          json={"content": "what did we decide?", "token_budget": 1500})
        assert got.status_code == 200, got.text
        history = client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]
        summaries = [m for m in history if m["role"] == "system" and m["metadata"].get("compaction_summary")]
        assert len(summaries) == 1
        assert got.json()["assistant_message"]["metadata"]["compacted_before"] > 0
        # Nothing was deleted: the transcript still has every message.
        assert len([m for m in history if m["role"] == "user"]) == 13

    def test_turned_off_it_does_not(self, api, monkeypatch):
        client = api()
        use_model(monkeypatch, ScriptedModel(*(["answer " * 60] * 20)))
        client.patch("/hyperlink/preferences", json={"auto_compact": False})
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        self.fill(client, session)
        client.post(f"/hyperlink/sessions/{session}/chat",
                    json={"content": "what did we decide?", "token_budget": 1500})
        history = client.get(f"/hyperlink/sessions/{session}/messages").json()["messages"]
        assert not any(m["metadata"].get("compaction_summary") for m in history)

    def test_a_short_thread_is_left_alone(self, api, monkeypatch):
        client = api()
        use_model(monkeypatch, ScriptedModel("hi"))
        session = client.post("/hyperlink/sessions", json={"title": "t"}).json()["session"]["session_id"]
        got = client.post(f"/hyperlink/sessions/{session}/chat", json={"content": "hello"})
        assert "compacted_before" not in got.json()["assistant_message"]["metadata"]


class TestShellThroughTheAPI:
    def test_off_by_default(self, api):
        client = api()
        assert client.get("/hyperlink/shell").json()["enabled"] is False
        got = client.post("/hyperlink/shell", json={"command": "echo hi"})
        assert got.status_code == 403
        assert "T1_HYPERLINK_SHELL" in got.text

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
    def test_on_it_runs_and_reports(self, api, tmp_path):
        client = api(T1_HYPERLINK_SHELL="1")
        got = client.post("/hyperlink/shell", json={"command": "echo out; echo err >&2; exit 4",
                                                    "cwd": str(tmp_path)})
        body = got.json()
        assert got.status_code == 200, got.text
        assert (body["stdout"], body["stderr"], body["exit_code"]) == ("out\n", "err\n", 4)
        assert body["cwd"] == str(tmp_path)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    def test_a_runaway_command_is_killed_with_its_children(self, api):
        client = api(T1_HYPERLINK_SHELL="1", T1_HYPERLINK_SHELL_TIMEOUT="1")
        started = time.monotonic()
        body = client.post("/hyperlink/shell", json={"command": "sleep 30 | cat"}).json()
        assert body["timed_out"] is True and body["exit_code"] is None
        assert time.monotonic() - started < 10

    def test_bad_requests(self, api, tmp_path):
        client = api(T1_HYPERLINK_SHELL="1")
        assert client.post("/hyperlink/shell", json={"command": "  "}).status_code in (400, 422)
        assert client.post("/hyperlink/shell", json={"command": "ls", "cwd": str(tmp_path / "nope")}).status_code in (400, 422)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
    def test_output_is_capped(self, api, monkeypatch):
        from hypernix.hyperlink import shell

        monkeypatch.setattr(shell, "MAX_OUTPUT", 100)
        client = api(T1_HYPERLINK_SHELL="1")
        body = client.post("/hyperlink/shell", json={"command": "yes | head -c 5000"}).json()
        assert body["truncated"] is True and len(body["stdout"]) == 100


class TestMemoryRoutes:
    def test_categories_rename_and_organise(self, api):
        client = api()
        client.post("/memory/create", json={"content": "Lisbon", "category": "home city", "source": "auto"})
        client.post("/memory/create", json={"content": "a", "category": "Job"})
        client.post("/memory/create", json={"content": "b", "category": "Work"})
        preview = client.post("/memory/organise", json={"dry_run": True}).json()
        assert preview["applied"] is False and [c["to"] for c in preview["changes"]] == ["Places"]
        client.post("/memory/organise", json={})
        assert client.post("/memory/categories/rename", json={"from": "Job", "to": "Work"}).json()["moved"] == 1
        cats = {c["name"]: c["count"] for c in client.get("/memory/categories").json()["categories"]}
        assert cats == {"Work": 2, "Places": 1}
