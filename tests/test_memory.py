"""What the assistant remembers between conversations.

A transcript is the wrong shape for "I use metric", "my GPU is a 1080",
"call me Mason". Those are true across every conversation and they fall
out of the context window exactly when a long session most needs them.

Two things make this need guard rails rather than a table:

**The model writes some of them.** So every memory records its source,
and auto-memory has a budget — without one, a model that writes a memory
per turn produces a prompt prefix longer than the conversation, and the
cost lands on somebody who never asked for any of it.

**They are durable and invisible.** A person has to be able to find out
why the assistant believes something and stop it believing that. Which
is why delete exists here even though the spec named four endpoints.
"""
from __future__ import annotations

import logging

import pytest

from hypernix.hyperlink.memory import AUTO_LIMIT, MAX_CONTENT, MemoryStore
from hypernix.t1api.errors import T1APIError


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    logging.disable(logging.CRITICAL)
    for name in list(__import__("os").environ):
        if name.startswith("T1_"):
            monkeypatch.delenv(name, raising=False)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    from hypernix.t1api.db import SQLiteBackend

    return MemoryStore(SQLiteBackend(str(tmp_path / "memories.db")))


class TestWritingAndReading:
    def test_a_memory_survives_being_written(self, store):
        record = store.create(owner="mason", content="Prefers metric units")
        assert store.get(record.memory_id, owner="mason").content == "Prefers metric units"

    def test_it_records_who_wrote_it(self, store):
        """A person deleting "you dislike Python" needs to see where the
        model got that idea."""
        auto = store.create(
            owner="mason", content="Dislikes Python", source="auto",
            session_id="sess_1",
        )
        assert auto.source == "auto"
        assert auto.session_id == "sess_1"

    def test_listing_puts_pinned_first(self, store):
        store.create(owner="mason", content="ordinary")
        store.create(owner="mason", content="important", pinned=True)
        assert store.list(owner="mason")[0].content == "important"

    def test_filtering_by_source_is_the_settings_screen_question(self, store):
        """"What has the model decided about me on its own" is a
        different question from "what did I tell it"."""
        store.create(owner="mason", content="told it this")
        store.create(owner="mason", content="it decided this", source="auto")
        assert len(store.list(owner="mason", source="auto")) == 1
        assert len(store.list(owner="mason", source="manual")) == 1

    def test_editing_changes_only_what_was_sent(self, store):
        record = store.create(owner="mason", content="a", category="prefs")
        edited = store.edit(record.memory_id, owner="mason", content="b")
        assert edited.content == "b"
        assert edited.category == "prefs"

    def test_deleting_forgets_it(self, store):
        record = store.create(owner="mason", content="wrong thing")
        assert store.delete(record.memory_id, owner="mason") is True
        with pytest.raises(T1APIError):
            store.get(record.memory_id, owner="mason")

    def test_deleting_twice_is_not_an_error(self, store):
        record = store.create(owner="mason", content="x")
        store.delete(record.memory_id, owner="mason")
        assert store.delete(record.memory_id, owner="mason") is False


class TestOwnership:
    """The phone and the desktop share a set; two people do not."""

    def test_another_owner_cannot_read_it(self, store):
        record = store.create(owner="mason", content="private")
        with pytest.raises(T1APIError):
            store.get(record.memory_id, owner="someone-else")

    def test_another_owner_cannot_edit_it(self, store):
        record = store.create(owner="mason", content="private")
        with pytest.raises(T1APIError):
            store.edit(record.memory_id, owner="someone-else", content="hijacked")
        assert store.get(record.memory_id, owner="mason").content == "private"

    def test_another_owner_cannot_delete_it(self, store):
        record = store.create(owner="mason", content="private")
        assert store.delete(record.memory_id, owner="someone-else") is False
        assert store.get(record.memory_id, owner="mason")

    def test_another_owner_cannot_list_it(self, store):
        store.create(owner="mason", content="private")
        assert store.list(owner="someone-else") == []

    def test_a_missing_memory_and_somebody_elses_look_the_same(self, store):
        """Distinguishing them would let a caller enumerate ids."""
        record = store.create(owner="mason", content="private")
        with pytest.raises(T1APIError) as theirs:
            store.get(record.memory_id, owner="someone-else")
        with pytest.raises(T1APIError) as absent:
            store.get("mem_doesnotexist", owner="someone-else")
        assert theirs.value.code == absent.value.code


class TestDuplicates:
    def test_the_same_fact_twice_is_one_memory(self, store):
        """"User's name is Mason" written forty times is forty memories
        that say one thing and cost forty times as much to carry."""
        first = store.create(owner="mason", content="User's name is Mason")
        second = store.create(owner="mason", content="User's name is Mason")
        assert first.memory_id == second.memory_id
        assert len(store.list(owner="mason")) == 1

    def test_case_and_punctuation_do_not_make_it_new(self, store):
        store.create(owner="mason", content="User's name is Mason.")
        store.create(owner="mason", content="users name is mason")
        assert len(store.list(owner="mason")) == 1

    def test_a_duplicate_refreshes_rather_than_erroring(self, store):
        """The caller asked for this to be remembered, and after the call
        it is. Making every auto-write handle a duplicate error means the
        model has to remember what it has remembered."""
        first = store.create(owner="mason", content="likes tea")
        second = store.create(owner="mason", content="likes tea", pinned=True)
        assert second.memory_id == first.memory_id
        assert second.pinned is True

    def test_two_owners_may_each_know_the_same_fact(self, store):
        store.create(owner="a", content="likes tea")
        store.create(owner="b", content="likes tea")
        assert len(store.list(owner="a")) == 1
        assert len(store.list(owner="b")) == 1


class TestTheAutoBudget:
    """Without a cap, a model that writes a memory per turn produces a
    prompt prefix longer than the conversation."""

    def test_auto_memories_are_capped(self, store):
        for index in range(AUTO_LIMIT + 10):
            store.create(owner="mason", content=f"auto fact {index}", source="auto")
        assert store.count(owner="mason", source="auto") == AUTO_LIMIT

    def test_the_oldest_go_first(self, store):
        for index in range(AUTO_LIMIT + 5):
            store.create(owner="mason", content=f"auto fact {index}", source="auto")
        remaining = {m.content for m in store.list(owner="mason", source="auto")}
        assert "auto fact 0" not in remaining
        assert f"auto fact {AUTO_LIMIT + 4}" in remaining

    def test_pinned_auto_memories_are_exempt(self, store):
        keep = store.create(
            owner="mason", content="pinned auto fact", source="auto", pinned=True
        )
        for index in range(AUTO_LIMIT + 10):
            store.create(owner="mason", content=f"auto fact {index}", source="auto")
        assert store.get(keep.memory_id, owner="mason")

    def test_manual_memories_are_never_evicted(self, store):
        """A person who wrote two hundred memories meant to. Evicting
        their notes to make room for the model's guesses is precisely
        backwards."""
        mine = [
            store.create(owner="mason", content=f"my note {i}") for i in range(20)
        ]
        for index in range(AUTO_LIMIT + 20):
            store.create(owner="mason", content=f"auto fact {index}", source="auto")
        for record in mine:
            assert store.get(record.memory_id, owner="mason")

    def test_one_owners_budget_does_not_touch_another(self, store):
        theirs = store.create(owner="other", content="their fact", source="auto")
        for index in range(AUTO_LIMIT + 10):
            store.create(owner="mason", content=f"auto {index}", source="auto")
        assert store.get(theirs.memory_id, owner="other")


class TestRefusals:
    def test_an_empty_memory_is_refused(self, store):
        with pytest.raises(T1APIError):
            store.create(owner="mason", content="   ")

    def test_a_document_is_refused(self, store):
        """A memory is a fact. Anything past the limit is a conversation
        that wanted to be a session."""
        with pytest.raises(T1APIError):
            store.create(owner="mason", content="x" * (MAX_CONTENT + 1))

    def test_an_unknown_source_is_refused(self, store):
        with pytest.raises(T1APIError):
            store.create(owner="mason", content="a", source="telepathy")

    def test_editing_to_nothing_is_refused(self, store):
        """Silently deleting on an empty edit would be a data-loss path
        that looks like a typo."""
        record = store.create(owner="mason", content="something")
        with pytest.raises(T1APIError):
            store.edit(record.memory_id, owner="mason", content="  ")


class TestThePromptBlock:
    def test_nothing_remembered_is_an_empty_string(self, store):
        """So a caller can concatenate without checking."""
        assert store.prompt_block(owner="mason") == ""

    def test_it_contains_the_memories(self, store):
        store.create(owner="mason", content="Prefers metric")
        block = store.prompt_block(owner="mason")
        assert "Prefers metric" in block

    def test_a_category_is_shown(self, store):
        store.create(owner="mason", content="1080", category="hardware")
        assert "[hardware]" in store.prompt_block(owner="mason")

    def test_pinned_memories_come_first(self, store):
        store.create(owner="mason", content="ordinary")
        store.create(owner="mason", content="pinned one", pinned=True)
        block = store.prompt_block(owner="mason")
        assert block.index("pinned one") < block.index("ordinary")

    def test_it_is_scoped_to_one_owner(self, store):
        store.create(owner="mason", content="mine")
        store.create(owner="other", content="theirs")
        assert "theirs" not in store.prompt_block(owner="mason")


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """An app with its own database.

    Not a detail: without it these tests share the real ~/.hypernix
    SQLite file, so they see each other's memories *and* write into
    whatever the developer running them actually has stored.
    """
    from hypernix.t1api.app import create_app

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
    return TestClient(create_app(), client=("192.168.1.50", 5432))


class TestTheEndpoints:
    def test_all_four_named_endpoints_exist(self, client):
        paths = set(client.app.openapi()["paths"])
        for name in ("create", "get", "list", "edit"):
            assert f"/memory/{name}" in paths

    def test_create_then_get(self, client):
        created = client.post(
            "/memory/create", json={"content": "Prefers metric units"}
        ).json()["memory"]
        fetched = client.get(
            "/memory/get", params={"memory_id": created["memory_id"]}
        ).json()["memory"]
        assert fetched["content"] == "Prefers metric units"

    def test_list_reports_the_auto_count(self, client):
        """Shown next to the budget on a settings screen: "the assistant
        has decided 64 things about me" is worth knowing without
        counting."""
        client.post("/memory/create", json={"content": "a", "source": "auto"})
        client.post("/memory/create", json={"content": "b"})
        body = client.get("/memory/list").json()
        assert body["count"] == 2
        assert body["auto_count"] == 1

    def test_edit_changes_it(self, client):
        created = client.post("/memory/create", json={"content": "old"}).json()["memory"]
        edited = client.post(
            "/memory/edit",
            json={"memory_id": created["memory_id"], "content": "new"},
        ).json()["memory"]
        assert edited["content"] == "new"

    def test_delete_forgets_it(self, client):
        created = client.post("/memory/create", json={"content": "wrong"}).json()["memory"]
        assert client.post(
            "/memory/delete", params={"memory_id": created["memory_id"]}
        ).json()["ok"] is True
        assert client.get("/memory/list").json()["count"] == 0

    def test_a_missing_memory_is_a_404(self, client):
        assert client.get(
            "/memory/get", params={"memory_id": "mem_nope"}
        ).status_code == 404

    def test_a_document_is_a_validation_error_not_a_500(self, client):
        response = client.post(
            "/memory/create", json={"content": "x" * (MAX_CONTENT + 1)}
        )
        assert response.status_code in (400, 422), response.text
        assert "limit" in response.text


class TestMemoriesReachTheModel:
    """A store the prompt never reads is a store that does nothing.

    The merge into the existing system message is the part worth testing:
    two system messages is not an error and is not reliably handled —
    some backends concatenate, some keep only the first — so a second one
    would silently drop either the session's instructions or everything
    known about the person, depending on which end the backend took.
    """

    def test_a_block_is_prepended_when_there_is_no_system_message(self):
        from hypernix.t1api.routers.hyperlink import _with_memories

        wire = _with_memories(
            [{"role": "user", "content": "hello"}], "What you know: likes tea"
        )
        assert wire[0]["role"] == "system"
        assert "likes tea" in wire[0]["content"]
        assert wire[1]["content"] == "hello"

    def test_it_merges_into_an_existing_system_message(self):
        from hypernix.t1api.routers.hyperlink import _with_memories

        wire = _with_memories(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hello"},
            ],
            "What you know: likes tea",
        )
        assert sum(1 for m in wire if m["role"] == "system") == 1
        assert "You are terse." in wire[0]["content"]
        assert "likes tea" in wire[0]["content"]

    def test_the_sessions_instructions_come_first(self):
        """The instructions are what the conversation is for; the
        memories are context for carrying them out."""
        from hypernix.t1api.routers.hyperlink import _with_memories

        wire = _with_memories(
            [{"role": "system", "content": "INSTRUCTIONS"}], "MEMORIES"
        )
        assert wire[0]["content"].index("INSTRUCTIONS") < wire[0]["content"].index(
            "MEMORIES"
        )

    def test_nothing_remembered_changes_nothing(self):
        from hypernix.t1api.routers.hyperlink import _with_memories

        original = [{"role": "user", "content": "hello"}]
        assert _with_memories(original, "") == original

    def test_the_original_list_is_not_mutated(self):
        """The caller built that list from stored history; editing it in
        place would change what a later read of the same objects sees."""
        from hypernix.t1api.routers.hyperlink import _with_memories

        original = [{"role": "system", "content": "keep me"}]
        _with_memories(original, "extra")
        assert original[0]["content"] == "keep me"

    def test_both_chat_routes_take_the_memory_store(self):
        """A dependency added to the streaming route and forgotten on the
        other is memory that works only when streaming."""
        import inspect

        from hypernix.t1api.routers import hyperlink

        for name in ("chat_turn", "chat_turn_stream"):
            signature = inspect.signature(getattr(hyperlink, name))
            assert "memories" in signature.parameters, name
