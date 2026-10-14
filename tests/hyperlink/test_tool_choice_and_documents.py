"""0.72.8: the model chooses when a tool is needed, saves a memory only
when told to, and can write a document the person downloads."""
from __future__ import annotations

import json

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink.default_prompt import DEFAULT_PROMPT
from hypernix.hyperlink.toolloop import MEMORY_WRITES, run_tool_loop, told_to_remember
from hypernix.interfaces.noodle.tools import ToolContext
from hypernix.runtime import t1tools


def envelope(content: str = "", calls: list | None = None) -> dict:
    message: dict = {"content": content}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def call(name: str, arguments: dict) -> dict:
    return {"id": "c1", "function": {"name": name, "arguments": json.dumps(arguments)}}


def extract(env: dict) -> tuple[str, str]:
    return env["choices"][0]["message"].get("content", ""), "stop"


def scripted(*turns):
    remaining = list(turns)
    return lambda messages, tools: remaining.pop(0) if remaining else envelope("done")


class TestThePrompt:
    @pytest.mark.parametrize("point", [
        "decide whether a tool is needed at all",
        "work out what to search for",
        "document tool",
        "Save a memory only when the",
        "Never save one on your own initiative",
    ])
    def test_it_says(self, point):
        assert point in DEFAULT_PROMPT

    def test_the_tools_say_so_too(self):
        tools = {t.name: t for t in t1tools.TOOLS}
        assert "Only when they ask" in tools["memory_create"].description
        assert "only when they ask for a file" in tools["create_document"].description.lower()
        assert not tools["create_document"].mutating  # offered by default: it writes only their own files


class TestToldToRemember:
    @pytest.mark.parametrize("text", [
        "Remember that my server is called atlas", "please memorise this", "forget my old address",
        "Make a note: I prefer metric", "save this to your memory", "keep in mind I'm vegetarian"])
    def test_asked(self, text):
        assert told_to_remember([{"role": "user", "content": text}])

    @pytest.mark.parametrize("text", ["What's the weather?", "My server is called atlas", "write me a poem"])
    def test_not_asked(self, text):
        assert not told_to_remember([{"role": "user", "content": text}])

    def test_only_the_latest_message_counts(self):
        assert not told_to_remember([{"role": "user", "content": "remember X"},
                                     {"role": "assistant", "content": "ok"},
                                     {"role": "user", "content": "now tell me a joke"}])


class TestTheGuard:
    @pytest.fixture
    def context(self, tmp_path):
        return ToolContext(root=tmp_path, memory_enabled=True, memory_path=tmp_path / "memory.json")

    def test_a_memory_the_person_did_not_ask_for_is_not_saved(self, context, tmp_path):
        ask = scripted(envelope(calls=[call("update_memory", {"key": "city", "value": "Leeds"})]),
                       envelope("Sure."))
        _m, _f, rounds = run_tool_loop([{"role": "user", "content": "I live in Leeds, what's near me?"}],
                                       context, ask=ask, extract=extract, workspace=False)
        assert rounds[0].ok is False and rounds[0].code == "memory_not_requested"
        assert not (tmp_path / "memory.json").exists()

    def test_one_they_asked_for_is(self, context, tmp_path):
        ask = scripted(envelope(calls=[call("update_memory", {"key": "city", "value": "Leeds"})]),
                       envelope("Noted."))
        _m, _f, rounds = run_tool_loop([{"role": "user", "content": "Remember that I live in Leeds"}],
                                       context, ask=ask, extract=extract, workspace=False)
        assert rounds[0].ok is True
        assert json.loads((tmp_path / "memory.json").read_text())["city"]["value"] == "Leeds"

    def test_the_t1_memory_tool_is_guarded_too(self):
        assert set(MEMORY_WRITES) == {"update_memory", "memory_create"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_HYPERLINK_SHARED_KEYLESS_CHATS", "1")
    monkeypatch.setenv("T1_HYPERLINK_FILES_DIR", str(tmp_path / "files"))
    from hypernix.t1api.app import create_app

    return TestClient(create_app(), client=("192.168.1.9", 5000))


class TestDocuments:
    def test_a_markdown_document_can_be_downloaded(self, client):
        made = client.post("/hyperlink/files/document", json={
            "title": "Release notes", "content": "Things got faster.", "format": "markdown"})
        assert made.status_code == 200, made.text
        body = made.json()
        assert body["file"]["filename"] == "Release notes.md"
        got = client.get(body["download_path"])
        assert got.status_code == 200 and "attachment" in got.headers["Content-Disposition"]
        assert got.text == "# Release notes\n\nThings got faster."

    def test_names_are_cleaned_and_text_gets_txt(self, client):
        body = client.post("/hyperlink/files/document", json={
            "filename": "../../etc/passwd", "content": "x", "format": "text"}).json()
        assert "/" not in body["file"]["filename"] and body["file"]["filename"].endswith(".txt")

    def test_an_empty_document_is_refused(self, client):
        assert client.post("/hyperlink/files/document", json={"content": ""}).status_code == 422

    def test_a_reply_that_wrote_one_lists_it(self):
        from hypernix.t1api.routers.hyperlink import _documents

        content = json.dumps({"file": {"file_id": "f1", "filename": "a.md", "size_bytes": 3},
                              "download_path": "/hyperlink/files/f1"})
        docs = _documents([{"tool": "create_document", "ok": True, "content": content},
                           {"tool": "create_document", "ok": False, "content": "HTTP 500"},
                           {"tool": "web_search", "ok": True, "content": "{}"}])
        assert docs == [{"file_id": "f1", "filename": "a.md", "size_bytes": 3, "content_type": "",
                         "download_path": "/hyperlink/files/f1"}]
