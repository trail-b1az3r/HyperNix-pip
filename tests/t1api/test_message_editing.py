"""Edit mode: changing what you asked, and re-asking it.

The interesting decision here is not that an edit is possible. It is
what happens to everything *below* the edited message.

A conversation is not a list of independent lines. Every message after
one you edit was written in reply to the text that used to be there, and
— this is the part that matters — that same transcript is what gets sent
as context on the next turn. Keep it and the model is handed a record in
which it answered a question that was never asked, and reasons from it.
So an edit truncates, and the count is reported so a client can say what
it costs before spending it.

Deleting is the opposite and for the opposite reason. Removing one
message is usually about removing something that should not be stored —
a pasted key, somebody's name — and taking the rest of the conversation
with it would make people keep the secret rather than lose the thread.
"""
from __future__ import annotations

import tempfile

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink.sessions import ChatSessionStore
from hypernix.t1api.errors import T1APIError


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("T1_DB_PATH", str(tempfile.mkdtemp() + "/t.sqlite3"))
    return ChatSessionStore()


@pytest.fixture
def conversation(store):
    """A real shape: question, answer, question, answer."""
    session = store.create(owner="me", title="t")
    ids = []
    for role, content in (
        ("user", "what is a GGUF"),
        ("assistant", "a file format"),
        ("user", "how big is one"),
        ("assistant", "it depends"),
    ):
        ids.append(
            store.append(session.session_id, role=role, content=content, owner="me")
        )
    return session.session_id, ids


class TestAnEditTruncates:
    def test_everything_after_it_goes(self, store, conversation):
        session_id, messages = conversation
        removed = store.edit_message(
            session_id, messages[0].message_id,
            content="what is a safetensors file", owner="me",
        )
        assert [m.content for m in removed] == [
            "a file format", "how big is one", "it depends",
        ]
        left = store.messages(session_id, owner="me")
        assert [m.content for m in left] == ["what is a safetensors file"]

    def test_it_reports_what_it_removed(self, store, conversation):
        """So a client can say "this removes 3 messages" rather than
        removing them and letting somebody notice."""
        session_id, messages = conversation
        removed = store.edit_message(
            session_id, messages[0].message_id, content="changed", owner="me"
        )
        assert len(removed) == 3

    def test_editing_the_last_message_removes_nothing(self, store, conversation):
        """The fixture ends on an assistant message, which is not
        editable — so this appends the trailing user message that the
        case is actually about: a question typed, then corrected before
        anything answered it."""
        session_id, _ = conversation
        last = store.append(
            session_id, role="user", content="one more thing", owner="me"
        )
        removed = store.edit_message(
            session_id, last.message_id, content="one more thing, actually",
            owner="me",
        )
        assert removed == []
        assert len(store.messages(session_id, owner="me")) == 5

    def test_truncation_can_be_declined(self, store, conversation):
        """For fixing a typo with nothing after it, where re-running
        would be waste."""
        session_id, messages = conversation
        removed = store.edit_message(
            session_id, messages[0].message_id,
            content="typo fixed", owner="me", truncate=False,
        )
        assert removed == []
        assert len(store.messages(session_id, owner="me")) == 4

    def test_the_content_actually_changes(self, store, conversation):
        session_id, messages = conversation
        store.edit_message(
            session_id, messages[0].message_id, content="new text", owner="me"
        )
        assert store.messages(session_id, owner="me")[0].content == "new text"


class TestWhatItRefuses:
    def test_the_assistants_words_are_not_editable(self, store, conversation):
        """Rewriting them turns the transcript into a record of
        something that did not happen — and the next turn is built from
        it."""
        session_id, messages = conversation
        with pytest.raises(T1APIError) as refused:
            store.edit_message(
                session_id, messages[1].message_id, content="nope", owner="me"
            )
        assert "your own messages" in str(refused.value)

    def test_an_empty_edit_is_refused(self, store, conversation):
        """Saving an empty message is a delete wearing the wrong name,
        and it leaves a blank bubble in the transcript."""
        session_id, messages = conversation
        with pytest.raises(T1APIError) as refused:
            store.edit_message(
                session_id, messages[0].message_id, content="   ", owner="me"
            )
        assert "Delete it instead" in str(refused.value)

    def test_a_message_that_is_not_there(self, store, conversation):
        session_id, _ = conversation
        with pytest.raises(T1APIError):
            store.edit_message(session_id, "msg_nonsense", content="x", owner="me")

    def test_another_owners_session_is_not_reachable(self, store, conversation):
        session_id, messages = conversation
        with pytest.raises(T1APIError):
            store.edit_message(
                session_id, messages[0].message_id, content="x", owner="someone-else"
            )


class TestItKeepsWhatWasThere:
    def test_the_original_text_is_recorded(self, store, conversation):
        """Somebody reading the transcript later can see that this line
        is not what was originally sent. A silent rewrite hides it."""
        session_id, messages = conversation
        store.edit_message(
            session_id, messages[0].message_id, content="changed", owner="me"
        )
        edited = store.messages(session_id, owner="me")[0]
        assert edited.metadata.get("edited_from") == "what is a GGUF"

    def test_a_second_edit_keeps_the_first_original(self, store, conversation):
        """`edited_from` is what was *sent*, not what it said last time
        — otherwise two edits lose the only copy of the real text."""
        session_id, messages = conversation
        store.edit_message(
            session_id, messages[0].message_id, content="once", owner="me"
        )
        store.edit_message(
            session_id, messages[0].message_id, content="twice", owner="me"
        )
        edited = store.messages(session_id, owner="me")[0]
        assert edited.metadata["edited_from"] == "what is a GGUF"

    def test_the_time_is_recorded(self, store, conversation):
        session_id, messages = conversation
        store.edit_message(
            session_id, messages[0].message_id, content="changed", owner="me"
        )
        assert store.messages(session_id, owner="me")[0].metadata.get("edited_at")


class TestDeletingIsNotTruncating:
    def test_the_rest_of_the_conversation_stays(self, store, conversation):
        """The reason: deleting is usually about removing something that
        should not be stored, and losing the thread over it makes people
        keep the secret instead."""
        session_id, messages = conversation
        assert store.delete_message(session_id, messages[1].message_id, owner="me")
        left = [m.content for m in store.messages(session_id, owner="me")]
        assert left == ["what is a GGUF", "how big is one", "it depends"]

    def test_the_assistants_words_can_be_deleted(self, store, conversation):
        """Unlike editing. Removing a record is honest; rewriting one is
        not."""
        session_id, messages = conversation
        assert store.delete_message(session_id, messages[1].message_id, owner="me")

    def test_deleting_nothing_is_false_not_an_error(self, store, conversation):
        session_id, _ = conversation
        assert store.delete_message(session_id, "msg_gone", owner="me") is False

    def test_another_owner_cannot_delete(self, store, conversation):
        session_id, messages = conversation
        with pytest.raises(T1APIError):
            store.delete_message(
                session_id, messages[0].message_id, owner="someone-else"
            )


class TestThroughTheAPI:
    @pytest.fixture
    def app_and_client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        # Not every T1_* variable — the storage redirects in
        # conftest.py stay, or "a server with no configuration" quietly
        # becomes "a server writing to the real ~/.hypernix".
        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")

        from hypernix.t1api.app import create_app

        app = create_app()
        return app, TestClient(app, client=("192.168.1.9", 5000))

    @pytest.fixture
    def thread(self, app_and_client):
        app, client = app_and_client
        session_id = client.post(
            "/hyperlink/sessions", json={"title": "t"}
        ).json()["session"]["session_id"]
        store = app.state.t1_session_store
        first = store.append(session_id, role="user", content="first")
        store.append(session_id, role="assistant", content="reply")
        store.append(session_id, role="user", content="second")
        return client, session_id, first.message_id

    def test_an_edit_returns_what_it_removed(self, thread):
        client, session_id, message_id = thread
        response = client.patch(
            f"/hyperlink/sessions/{session_id}/messages/{message_id}",
            json={"content": "first, edited"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["removed_count"] == 2
        assert body["message"]["content"] == "first, edited"

    def test_the_conversation_is_actually_shorter(self, thread):
        client, session_id, message_id = thread
        client.patch(
            f"/hyperlink/sessions/{session_id}/messages/{message_id}",
            json={"content": "edited"},
        )
        left = client.get(f"/hyperlink/sessions/{session_id}/messages").json()
        assert left["count"] == 1

    def test_editing_the_assistant_is_a_422(self, thread, app_and_client):
        """VALIDATION_ERROR, not 500: it is a thing the caller asked for
        that is not allowed, and the message says why."""
        app, _ = app_and_client
        client, session_id, _ = thread
        assistant = [
            m for m in app.state.t1_session_store.messages(session_id)
            if m.role == "assistant"
        ][0]
        response = client.patch(
            f"/hyperlink/sessions/{session_id}/messages/{assistant.message_id}",
            json={"content": "nope"},
        )
        assert response.status_code == 422
        assert "your own messages" in response.json()["error"]["message"]

    def test_delete_leaves_the_rest(self, thread):
        client, session_id, message_id = thread
        assert client.delete(
            f"/hyperlink/sessions/{session_id}/messages/{message_id}"
        ).status_code == 200
        left = client.get(f"/hyperlink/sessions/{session_id}/messages").json()
        assert left["count"] == 2

    def test_deleting_twice_is_a_404(self, thread):
        client, session_id, message_id = thread
        client.delete(f"/hyperlink/sessions/{session_id}/messages/{message_id}")
        second = client.delete(
            f"/hyperlink/sessions/{session_id}/messages/{message_id}"
        )
        assert second.status_code == 404

    def test_an_empty_edit_is_refused_over_the_wire(self, thread):
        client, session_id, message_id = thread
        response = client.patch(
            f"/hyperlink/sessions/{session_id}/messages/{message_id}",
            json={"content": "  "},
        )
        assert response.status_code == 422
