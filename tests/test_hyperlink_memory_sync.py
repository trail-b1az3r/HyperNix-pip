"""Keeping the phone's copy of its memories current (0.72.6).

The app used to fetch ``/memory/list`` whole: the first 200, whenever a
screen appeared or a reply ended, errors swallowed. A memory the model
wrote mid-chat reached the phone only if somebody was looking, the
201st never did, and offline the screen was empty.

Now every write is logged, deletions and evictions included, and
``/memory/sync`` answers "what changed since cursor N?" with a small
delta, or with the whole set when a delta cannot be trusted. A chat
turn that changes a memory says so in a ``memory`` frame after ``done``.
"""
from __future__ import annotations

import logging

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink.memory import (
    AUTO_LIMIT,
    MEMORY_LOG_TTL_SECONDS,
    MEMORY_SYNC_PAGE,
    MemoryStore,
)
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    logging.disable(logging.CRITICAL)
    clear_t1_config(monkeypatch)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(SQLiteBackend(str(tmp_path / "memories.db")))


class Phone:
    """What the app does with an answer: replace on full, else apply."""

    def __init__(self, store: MemoryStore, owner: str = "mason") -> None:
        self.store, self.owner = store, owner
        self.cursor = 0
        self.copy: dict[str, str] = {}
        self.pages = 0

    def sync(self) -> None:
        while True:
            page = self.store.sync(owner=self.owner, cursor=self.cursor, limit=50)
            self.pages += 1
            if page.full:
                self.copy = {}
            for memory in page.memories:
                self.copy[memory.memory_id] = memory.content
            for memory_id in page.deleted:
                self.copy.pop(memory_id, None)
            self.cursor = page.cursor
            if not page.more:
                return

    def truth(self) -> dict[str, str]:
        return {m.memory_id: m.content for m in self.store.list(owner=self.owner, limit=100_000)}


class TestTheFirstSync:
    def test_it_is_the_whole_set_and_says_so(self, store):
        store.create(owner="mason", content="Prefers metric units")
        page = store.sync(owner="mason", cursor=0)
        assert page.full and page.reason == "first"
        assert [m.content for m in page.memories] == ["Prefers metric units"]
        assert page.cursor >= store.head(owner="mason") > 0

    def test_nothing_remembered_is_an_empty_full_answer_with_a_real_cursor(self, store):
        """0 means "never synced", so an empty set must not be answered
        with it, or every sync after it is a full one."""
        page = store.sync(owner="mason", cursor=0)
        assert page.full and page.memories == [] and page.cursor > 0
        assert not store.sync(owner="mason", cursor=page.cursor).full

    def test_it_is_not_capped_at_two_hundred(self, store):
        """The old list stopped at 200 and the rest never reached the phone."""
        for n in range(250):
            store.create(owner="mason", content=f"fact number {n} is {n * 7}")
        phone = Phone(store)
        phone.sync()
        assert len(phone.copy) == 250


class TestADelta:
    def test_a_current_cursor_gets_nothing(self, store):
        store.create(owner="mason", content="a")
        cursor = store.sync(owner="mason").cursor
        page = store.sync(owner="mason", cursor=cursor)
        assert not page.full
        assert page.memories == [] and page.deleted == [] and page.cursor == cursor

    def test_created_edited_and_deleted_all_arrive(self, store):
        keep = store.create(owner="mason", content="keep me")
        change = store.create(owner="mason", content="change me")
        drop = store.create(owner="mason", content="drop me")
        phone = Phone(store)
        phone.sync()

        store.edit(change.memory_id, owner="mason", content="changed")
        store.delete(drop.memory_id, owner="mason")
        new = store.create(owner="mason", content="brand new")
        page = store.sync(owner="mason", cursor=phone.cursor)
        assert not page.full
        assert {m.memory_id for m in page.memories} == {change.memory_id, new.memory_id}
        assert page.deleted == [drop.memory_id]
        assert keep.memory_id not in {m.memory_id for m in page.memories}

        phone.sync()
        assert phone.copy == phone.truth()

    def test_a_memory_edited_many_times_is_sent_once(self, store):
        record = store.create(owner="mason", content="v0")
        cursor = store.sync(owner="mason").cursor
        for n in range(1, 41):
            store.edit(record.memory_id, owner="mason", content=f"v{n}")
        page = store.sync(owner="mason", cursor=cursor)
        assert [m.content for m in page.memories] == ["v40"]

    def test_a_memory_created_then_deleted_between_syncs_is_a_deletion(self, store):
        cursor = store.sync(owner="mason").cursor
        brief = store.create(owner="mason", content="brief")
        store.delete(brief.memory_id, owner="mason")
        page = store.sync(owner="mason", cursor=cursor)
        assert page.memories == [] and page.deleted == [brief.memory_id]

    def test_an_eviction_is_a_deletion_the_phone_hears_about(self, store):
        """Otherwise the phone keeps showing a fact the model no longer has."""
        first = store.create(owner="mason", content="the oldest auto fact", source="auto")
        phone = Phone(store)
        phone.sync()
        assert first.memory_id in phone.copy
        for n in range(AUTO_LIMIT):
            store.create(owner="mason", content=f"auto fact {n} about thing {n}", source="auto")
        phone.sync()
        assert first.memory_id not in phone.copy
        assert phone.copy == phone.truth()

    def test_a_big_delta_comes_in_pages(self, store):
        phone = Phone(store)
        phone.sync()
        for n in range(120):
            store.create(owner="mason", content=f"note {n} says {n * 3}")
        page = store.sync(owner="mason", cursor=phone.cursor, limit=50)
        assert page.more and len(page.memories) == 50
        phone.sync()
        assert phone.pages >= 4
        assert phone.copy == phone.truth()

    def test_the_page_cap_cannot_be_raised(self, store):
        cursor = store.sync(owner="mason").cursor
        for n in range(MEMORY_SYNC_PAGE + 5):
            store.create(owner="mason", content=f"n{n}x{n}")
        page = store.sync(owner="mason", cursor=cursor, limit=100_000)
        assert len(page.memories) == MEMORY_SYNC_PAGE and page.more


class TestWhenADeltaCannotBeTrusted:
    def test_a_cursor_this_server_never_issued_gets_the_whole_set(self, store):
        """The phone was paired to a server since restored or replaced."""
        store.create(owner="mason", content="a")
        page = store.sync(owner="mason", cursor=10_000)
        assert page.full and page.reason == "unknown_cursor"

    def test_a_cursor_older_than_the_tombstones_gets_the_whole_set(self, store):
        gone = store.create(owner="mason", content="gone soon")
        store.create(owner="mason", content="stays")
        phone = Phone(store)
        phone.sync()
        stale = phone.cursor
        store.delete(gone.memory_id, owner="mason")
        store.create(owner="mason", content="later")
        import time as _time
        assert store.prune(now=_time.time() + MEMORY_LOG_TTL_SECONDS + 1) == 1
        page = store.sync(owner="mason", cursor=stale)
        assert page.full and page.reason == "expired"
        assert gone.memory_id not in {m.memory_id for m in page.memories}

    def test_a_cursor_after_the_pruned_tombstones_still_gets_a_delta(self, store):
        gone = store.create(owner="mason", content="gone soon")
        store.delete(gone.memory_id, owner="mason")
        cursor = store.sync(owner="mason").cursor
        import time as _time
        store.prune(now=_time.time() + MEMORY_LOG_TTL_SECONDS + 1)
        later = store.create(owner="mason", content="later")
        page = store.sync(owner="mason", cursor=cursor)
        assert not page.full and [m.memory_id for m in page.memories] == [later.memory_id]

    def test_pruning_keeps_live_memories_in_the_log(self, store):
        store.create(owner="mason", content="alive")
        cursor_before = 1
        import time as _time
        assert store.prune(now=_time.time() + MEMORY_LOG_TTL_SECONDS + 1) == 0
        assert not store.sync(owner="mason", cursor=cursor_before).full

    def test_a_negative_cursor_is_refused(self, store):
        with pytest.raises(T1APIError):
            store.sync(owner="mason", cursor=-1)


class TestOwnership:
    def test_another_owners_changes_never_arrive(self, store):
        cursor = store.sync(owner="mason").cursor
        theirs = store.create(owner="someone", content="theirs")
        store.delete(theirs.memory_id, owner="someone")
        page = store.sync(owner="mason", cursor=cursor)
        assert page.memories == [] and page.deleted == []

    def test_a_full_answer_is_only_the_owners(self, store):
        store.create(owner="someone", content="theirs")
        assert store.sync(owner="mason").memories == []


class TestMemoriesFromBeforeTheLog:
    def test_they_are_logged_when_the_store_opens(self, tmp_path):
        """Otherwise the first full answer comes back with cursor 0, and
        0 means "send everything" every time."""
        backend = SQLiteBackend(str(tmp_path / "old.db"))
        MemoryStore(backend)
        with backend.connect() as conn:
            conn.execute(
                "INSERT INTO hyperlink_memories (memory_id, owner, content, created_at, "
                "updated_at) VALUES ('mem_old', 'mason', 'from 0.72.5', 1.0, 1.0)"
            )
            conn.execute("DELETE FROM hyperlink_memory_log")
        reopened = MemoryStore(backend)
        page = reopened.sync(owner="mason")
        assert [m.memory_id for m in page.memories] == ["mem_old"]
        assert page.cursor > 0
        assert not reopened.sync(owner="mason", cursor=page.cursor).full


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    from hypernix.t1api.app import create_app

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
    return TestClient(create_app(), client=("192.168.1.50", 5432))


class TestTheEndpoint:
    def test_first_then_delta_then_nothing(self, client):
        client.post("/memory/create", json={"content": "Prefers metric units"})
        first = client.get("/memory/sync").json()
        assert first["full"] is True and first["count"] == 1
        assert first["memories"][0]["content"] == "Prefers metric units"

        made = client.post("/memory/create", json={"content": "Has a 1080", "source": "auto"}).json()
        delta = client.get("/memory/sync", params={"cursor": first["cursor"]}).json()
        assert delta["full"] is False
        assert [m["memory_id"] for m in delta["memories"]] == [made["memory"]["memory_id"]]
        assert delta["auto_count"] == 1

        client.post("/memory/delete", params={"memory_id": made["memory"]["memory_id"]})
        gone = client.get("/memory/sync", params={"cursor": delta["cursor"]}).json()
        assert gone["deleted"] == [made["memory"]["memory_id"]]

        idle = client.get("/memory/sync", params={"cursor": gone["cursor"]}).json()
        assert idle["memories"] == [] and idle["deleted"] == [] and idle["cursor"] == gone["cursor"]

    def test_a_negative_cursor_is_a_validation_error(self, client):
        assert client.get("/memory/sync", params={"cursor": -1}).status_code == 422


# ---------------------------------------------------------------------------
# The chat stream says when a turn changed a memory
# ---------------------------------------------------------------------------

pytest.importorskip("uvicorn")
pytest.importorskip("httpx")

import httpx  # noqa: E402
import test_hyperlink_stream_tools as _stream_tools  # noqa: E402
from test_hyperlink_stream_tools import StreamingModel, _stream, _use  # noqa: E402

#: The live server those tests start: a real app on a port, a paired phone.
served = _stream_tools.served


def test_a_turn_that_remembers_something_sends_a_memory_frame(served, monkeypatch):
    base, auth = served()
    before = httpx.get(f"{base}/memory/sync", headers=auth).json()
    model = StreamingModel(
        '<tool_call>{"name": "update_memory", "arguments": '
        '{"key": "favourite editor", "value": "Helix"}}</tool_call>',
        "Noted: Helix.",
    )
    _use(monkeypatch, model)
    _session, frames = _stream(base, auth, "Remember that my editor is Helix")

    kinds = [f["type"] for f in frames]
    assert "memory" in kinds and kinds.index("done") < kinds.index("memory")
    memory = next(f for f in frames if f["type"] == "memory")
    assert memory["cursor"] > before["cursor"]

    delta = httpx.get(f"{base}/memory/sync", params={"cursor": before["cursor"]},
                      headers=auth).json()
    assert delta["full"] is False
    assert any("Helix" in m["content"] for m in delta["memories"])
    assert delta["cursor"] == memory["cursor"]


def test_a_turn_that_remembers_nothing_sends_no_memory_frame(served, monkeypatch):
    base, auth = served()
    _use(monkeypatch, StreamingModel("Just an answer."))
    _session, frames = _stream(base, auth, "hello")
    assert "memory" not in [f["type"] for f in frames]
