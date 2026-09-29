"""When a chat counts as "updated", and when it only counts as changed.

The bug this is about
---------------------
HyperLink's chat list shows `updated_at` as "updated 2m ago" and orders
by it. The store bumped that column on *any* write — a rename, a model
switch, an archive, a message deleted from a chat from last Tuesday —
so tidying up eleven old conversations sent all eleven to the top of
the list claiming they had just replied. The app was faithfully
reporting a column that meant something other than its name.

So there are two clocks now. `updated_at` is when the conversation last
said anything, which is what the phone shows. `touched_at` is when
anything about it last changed, which is what a sync needs. These tests
pin which write moves which, because getting it backwards produces a
list that looks right until somebody renames something.
"""
from __future__ import annotations

import time

import pytest

from hypernix.hyperlink.sessions import ChatSessionStore
from hypernix.t1api.db import SQLiteBackend


@pytest.fixture
def store(tmp_path) -> ChatSessionStore:
    return ChatSessionStore(SQLiteBackend(tmp_path / "t1.sqlite3"))


@pytest.fixture
def session(store):
    made = store.create(owner="me", title="A chat")
    time.sleep(0.01)
    return made


def _reload(store, session):
    return store.get(session.session_id, owner="me")


class TestWhatCountsAsUpdated:
    def test_a_new_session_has_both_clocks_at_once(self, store):
        made = store.create(owner="me")
        assert made.updated_at == made.touched_at

    def test_a_prompt_moves_updated_at(self, store, session):
        """The whole point of the column."""
        store.append(session.session_id, owner="me", role="user", content="hi")
        after = _reload(store, session)
        assert after.updated_at > session.updated_at

    def test_a_reply_moves_updated_at(self, store, session):
        """Including one that arrived while the app was in the
        background — which is the case the list most needs to show."""
        store.append(session.session_id, owner="me", role="assistant",
                     content="hello")
        assert _reload(store, session).updated_at > session.updated_at

    def test_a_rename_does_not(self, store, session):
        """The bug, stated directly. Rename eleven chats and eleven of
        them used to read "updated just now"."""
        store.update(session.session_id, owner="me", title="Renamed")
        assert _reload(store, session).updated_at == session.updated_at

    def test_a_model_switch_does_not(self, store, session):
        store.update(session.session_id, owner="me", model_id="qwen3-4b")
        assert _reload(store, session).updated_at == session.updated_at

    def test_archiving_does_not(self, store, session):
        store.update(session.session_id, owner="me", archived=True)
        assert _reload(store, session).updated_at == session.updated_at

    def test_a_system_prompt_change_does_not(self, store, session):
        store.update(session.session_id, owner="me", system_prompt="Be terse.")
        assert _reload(store, session).updated_at == session.updated_at

    def test_deleting_a_message_does_not(self, store, session):
        """Removing a pasted key from an old chat is housekeeping, and
        sending it to the top of the list under "updated just now" is
        the opposite of discreet."""
        message = store.append(session.session_id, owner="me", role="user",
                               content="my key is hunter2")
        moved = _reload(store, session)
        time.sleep(0.01)
        store.delete_message(session.session_id, message.message_id, owner="me")
        assert _reload(store, session).updated_at == moved.updated_at


class TestWhatCountsAsTouched:
    def test_a_rename_moves_touched_at(self, store, session):
        """A sync still has to know something changed — the point is
        the two facts being separable, not one of them being lost."""
        store.update(session.session_id, owner="me", title="Renamed")
        assert _reload(store, session).touched_at > session.touched_at

    def test_a_message_moves_both(self, store, session):
        store.append(session.session_id, owner="me", role="user", content="hi")
        after = _reload(store, session)
        assert after.touched_at > session.touched_at
        assert after.updated_at > session.updated_at

    def test_deleting_a_message_moves_touched_at(self, store, session):
        message = store.append(session.session_id, owner="me", role="user",
                               content="x")
        moved = _reload(store, session)
        time.sleep(0.01)
        store.delete_message(session.session_id, message.message_id, owner="me")
        assert _reload(store, session).touched_at > moved.touched_at


class TestOrdering:
    def test_the_list_orders_by_conversation_activity(self, store):
        """What the list is for. Renaming an old chat must not push a
        live one down it."""
        old = store.create(owner="me", title="old")
        time.sleep(0.01)
        live = store.create(owner="me", title="live")
        store.append(live.session_id, owner="me", role="user", content="hi")
        time.sleep(0.01)
        store.update(old.session_id, owner="me", title="old, renamed")

        listed = [s.session_id for s in store.list_sessions(owner="me")]
        assert listed[0] == live.session_id


class TestTheMigration:
    @staticmethod
    def _legacy(path) -> None:
        """A database in the shape this code shipped with before the
        column existed."""
        backend = SQLiteBackend(path)
        with backend.connect() as conn:
            conn.executescript(
                """CREATE TABLE hyperlink_sessions (
                       session_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                       owner TEXT NOT NULL, device_id TEXT NOT NULL DEFAULT '',
                       model_id TEXT NOT NULL DEFAULT '',
                       backend TEXT NOT NULL DEFAULT '',
                       system_prompt TEXT NOT NULL DEFAULT '',
                       created_at REAL NOT NULL, updated_at REAL NOT NULL,
                       archived INTEGER NOT NULL DEFAULT 0,
                       metadata TEXT NOT NULL DEFAULT '{}');"""
            )
            conn.execute(
                "INSERT INTO hyperlink_sessions (session_id, title, owner, "
                "created_at, updated_at) VALUES ('chat_old', 'Old', 'me', 1.0, 42.0)"
            )
    def test_a_database_written_before_the_column_still_opens(self, tmp_path):
        """`CREATE TABLE IF NOT EXISTS` does not alter an existing
        table, so the column in the schema only ever reaches new
        databases. Without the ALTER, every install that upgrades into
        this gets a store that raises on its first read."""
        path = tmp_path / "old.sqlite3"
        self._legacy(path)

        store = ChatSessionStore(SQLiteBackend(path))
        found = store.get("chat_old", owner="me")
        assert found.title == "Old"
        assert found.updated_at == 42.0

    def test_writing_to_a_migrated_database_works(self, tmp_path):
        """The half of the migration a read cannot show.

        Reading survives a missing column — the row mapper falls back
        to `updated_at`. Writing does not: `update()` sets `touched_at`
        by name, and against a table that never got the ALTER that is an
        OperationalError on the first rename after upgrading.
        """
        path = tmp_path / "old.sqlite3"
        self._legacy(path)
        store = ChatSessionStore(SQLiteBackend(path))
        renamed = store.update("chat_old", owner="me", title="New name")
        assert renamed.title == "New name"
        assert store.get("chat_old", owner="me").updated_at == 42.0

    def test_an_old_row_is_seeded_from_updated_at(self, tmp_path):
        """Read straight out of SQL, not through the row mapper.

        The mapper falls back to `updated_at` when `touched_at` is
        zero, so asking it would pass whether or not the seeding ran.
        What the seeding is for is the *column*: a default of 0 sorts
        every pre-existing session to the bottom of anything ordered by
        it, which on a machine with two years of chats is the whole
        list.
        """
        path = tmp_path / "old.sqlite3"
        self._legacy(path)

        store = ChatSessionStore(SQLiteBackend(path))
        with store.backend.connect() as conn:
            stored = conn.execute(
                "SELECT touched_at FROM hyperlink_sessions "
                "WHERE session_id = 'chat_old'"
            ).fetchone()[0]
        assert stored == 42.0

    def test_running_it_twice_is_harmless(self, tmp_path):
        path = tmp_path / "t.sqlite3"
        first = ChatSessionStore(SQLiteBackend(path))
        made = first.create(owner="me", title="A")
        second = ChatSessionStore(SQLiteBackend(path))
        assert second.get(made.session_id, owner="me").title == "A"


class TestTheWireShape:
    def test_touched_at_is_sent(self, store, session):
        assert "touched_at" in session.to_dict()

    def test_it_falls_back_to_updated_at_rather_than_zero(self):
        """A session built in memory without one — which is what a
        caller constructing a ChatSession by hand produces — must not
        report the epoch as its last change."""
        from hypernix.hyperlink.sessions import ChatSession

        made = ChatSession(session_id="c", title="t", owner="me",
                           created_at=1.0, updated_at=99.0)
        assert made.to_dict()["touched_at"] == 99.0
