"""``hypernix.hyperlink.sync`` — catching up, and not sending twice.

Two failures that a phone client cannot fix on its own:

A retry sends the message twice. The phone POSTs a turn, the connection
drops before the response arrives, and it cannot tell "the server never
saw it" from "the server saw it and the reply was lost". Retrying
produces two identical user messages and two model replies, one of which
cost real tokens for nothing.

A phone that was away does not know what it missed — least of all that
something was *deleted*, which is invisible when you are diffing against
a list you no longer trust.
"""
from __future__ import annotations

import pathlib
import threading
import time

import pytest

from hypernix.hyperlink.sync import (
    CLAIM_TTL_SECONDS,
    MAX_PAGE,
    TOMBSTONE_TTL_SECONDS,
    ChangeKind,
    SyncStore,
)
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode


@pytest.fixture
def store(tmp_path: pathlib.Path) -> SyncStore:
    return SyncStore(SQLiteBackend(db_path=tmp_path / "sync.sqlite3"))


def add(store: SyncStore, n: int, *, owner: str = "alice", session_id: str = "s1"):
    for i in range(n):
        store.record(
            kind=ChangeKind.CREATED, entity="message", entity_id=f"m{i}",
            owner=owner, session_id=session_id,
        )


class TestNotSendingItTwice:
    """The idempotency key, which is the client's to mint."""

    def test_a_first_claim_is_fresh(self, store):
        assert store.claim(
            device_id="dev1", client_msg_id="k1", owner="alice"
        ).fresh is True

    def test_a_retry_is_not_fresh_and_gets_the_original_answer(self, store):
        store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        store.settle(
            device_id="dev1", client_msg_id="k1", result={"message_id": "m1"}
        )
        again = store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        assert again.fresh is False
        assert again.settled is True
        assert again.result == {"message_id": "m1"}

    def test_a_retry_while_the_first_is_still_running_says_so(self, store):
        """Not settled, no result: the caller tells the client to wait.

        Answering "already done" with an empty result would look like a
        successful turn that produced nothing.
        """
        store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        again = store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        assert again.fresh is False
        assert again.settled is False
        assert again.result == {}

    def test_two_devices_do_not_collide_on_the_same_key(self, store):
        """Keys are per device, so two phones can both pick "1"."""
        assert store.claim(device_id="dev1", client_msg_id="k", owner="alice").fresh
        assert store.claim(device_id="dev2", client_msg_id="k", owner="alice").fresh

    def test_a_key_belonging_to_another_owner_is_a_conflict(self, store):
        store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        with pytest.raises(T1APIError) as caught:
            store.claim(device_id="dev1", client_msg_id="k1", owner="bob")
        assert caught.value.code == T1ErrorCode.CONFLICT

    def test_releasing_a_failed_claim_lets_the_client_try_again(self, store):
        """Otherwise the key is held for a day and every retry is told
        "already done" with nothing to show for it -- the work
        permanently unfinished and unaskable."""
        store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        store.release(device_id="dev1", client_msg_id="k1")
        assert store.claim(device_id="dev1", client_msg_id="k1", owner="alice").fresh

    def test_releasing_a_settled_claim_does_nothing(self, store):
        """A completed turn must stay non-repeatable."""
        store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        store.settle(device_id="dev1", client_msg_id="k1", result={"message_id": "m1"})
        store.release(device_id="dev1", client_msg_id="k1")
        again = store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        assert again.fresh is False and again.result == {"message_id": "m1"}

    def test_settling_something_unclaimed_says_so(self, store):
        with pytest.raises(T1APIError) as caught:
            store.settle(device_id="dev1", client_msg_id="nope", result={})
        assert caught.value.code == T1ErrorCode.NOT_FOUND

    def test_an_expired_key_is_treated_as_new(self, store):
        store.claim(device_id="dev1", client_msg_id="k1", owner="alice")
        store.settle(device_id="dev1", client_msg_id="k1", result={"message_id": "m1"})
        with store.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_claims SET created_at = ?",
                (time.time() - CLAIM_TTL_SECONDS - 10,),
            )
        assert store.claim(device_id="dev1", client_msg_id="k1", owner="alice").fresh

    @pytest.mark.parametrize(
        ("device_id", "client_msg_id"), [("", "k"), ("d", ""), ("d", "x" * 129)]
    )
    def test_a_malformed_claim_is_refused(self, store, device_id, client_msg_id):
        with pytest.raises(T1APIError) as caught:
            store.claim(
                device_id=device_id, client_msg_id=client_msg_id, owner="alice"
            )
        assert caught.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_minted_keys_are_unique(self, store):
        assert len({store.new_client_msg_id() for _ in range(200)}) == 200

    def test_pruning_drops_expired_claims_only(self, store):
        store.claim(device_id="dev1", client_msg_id="old", owner="alice")
        with store.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_claims SET created_at = ?",
                (time.time() - CLAIM_TTL_SECONDS - 10,),
            )
        store.claim(device_id="dev1", client_msg_id="new", owner="alice")
        assert store.prune()["claims"] == 1
        assert not store.claim(
            device_id="dev1", client_msg_id="new", owner="alice"
        ).fresh


class TestCatchingUp:
    def test_an_empty_log_returns_an_empty_page(self, store):
        page = store.since(0, owner="alice")
        assert page.changes == [] and page.more is False and page.cursor == 0

    def test_changes_come_back_in_order(self, store):
        add(store, 5)
        seqs = [c.seq for c in store.since(0, owner="alice").changes]
        assert seqs == sorted(seqs) == [1, 2, 3, 4, 5]

    def test_a_page_reports_whether_there_is_more(self, store):
        add(store, 5)
        page = store.since(0, owner="alice", limit=3)
        assert [c.seq for c in page.changes] == [1, 2, 3]
        assert page.more is True and page.cursor == 3

    def test_an_exactly_full_page_with_nothing_after_says_no_more(self, store):
        """Inferring `more` from a full page sends the client round
        again for an empty answer -- on a cellular link, for nothing."""
        add(store, 3)
        page = store.since(0, owner="alice", limit=3)
        assert len(page.changes) == 3 and page.more is False

    def test_the_cursor_walks_the_whole_log(self, store):
        add(store, 12)
        seen, cursor, guard = [], 0, 0
        while guard < 20:
            guard += 1
            page = store.since(cursor, owner="alice", limit=5)
            seen.extend(c.seq for c in page.changes)
            cursor = page.cursor
            if not page.more:
                break
        assert seen == list(range(1, 13))

    def test_the_returned_cursor_is_the_last_seq_not_an_arithmetic_guess(self, store):
        """A filtered feed skips rows. Advancing the caller's cursor by
        the page size would step over the rows it skipped."""
        store.record(kind=ChangeKind.CREATED, entity="session",
                     entity_id="s1", owner="alice")
        add(store, 5)
        page = store.since(0, owner="alice", limit=2, entities=["message"])
        assert page.cursor == page.changes[-1].seq

    def test_one_owner_never_sees_anothers_changes(self, store):
        add(store, 3, owner="alice")
        add(store, 3, owner="bob")
        assert len(store.since(0, owner="alice").changes) == 3
        assert all(c.owner == "alice" for c in store.since(0, owner="alice").changes)

    def test_it_can_be_filtered_to_one_session(self, store):
        add(store, 3, session_id="s1")
        add(store, 2, session_id="s2")
        page = store.since(0, owner="alice", session_id="s2")
        assert len(page.changes) == 2

    def test_it_can_be_filtered_by_entity(self, store):
        add(store, 3)
        store.record(kind=ChangeKind.CREATED, entity="device",
                     entity_id="d1", owner="alice")
        page = store.since(0, owner="alice", entities=["device"])
        assert [c.entity for c in page.changes] == ["device"]

    def test_the_page_size_is_capped(self, store):
        """The point of the feed is a bounded response."""
        page = store.since(0, owner="alice", limit=10_000)
        assert page.changes == []  # nothing recorded, but the cap applied
        add(store, 3)
        assert len(store.since(0, owner="alice", limit=10_000).changes) <= MAX_PAGE

    @pytest.mark.parametrize("limit", [0, -5])
    def test_a_nonsense_limit_still_returns_something(self, store, limit):
        add(store, 3)
        assert len(store.since(0, owner="alice", limit=limit).changes) == 1

    def test_a_negative_cursor_is_refused(self, store):
        with pytest.raises(T1APIError) as caught:
            store.since(-1, owner="alice")
        assert caught.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_an_unknown_entity_filter_is_refused(self, store):
        with pytest.raises(T1APIError):
            store.since(0, owner="alice", entities=["telepathy"])

    def test_head_lets_a_new_device_skip_the_history(self, store):
        """Fetch state, start the feed at head. Starting at 0 replays
        every change ever made to rebuild a state it already has."""
        add(store, 40)
        assert store.head(owner="alice") == 40
        assert store.since(store.head(owner="alice"), owner="alice").changes == []

    def test_head_of_an_empty_log_is_zero(self, store):
        assert store.head(owner="alice") == 0


class TestDeletionsAreVisible:
    """An absence cannot be diffed. A tombstone can."""

    def test_a_deletion_appears_in_the_feed(self, store):
        store.record(kind=ChangeKind.DELETED, entity="session",
                     entity_id="s9", owner="alice")
        page = store.since(0, owner="alice")
        assert [(c.kind, c.entity_id) for c in page.changes] == [("deleted", "s9")]

    def test_tombstones_expire(self, store):
        store.record(kind=ChangeKind.DELETED, entity="session",
                     entity_id="s9", owner="alice")
        with store.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_change_log SET created_at = ?",
                (time.time() - TOMBSTONE_TTL_SECONDS - 10,),
            )
        assert store.prune()["tombstones"] == 1

    def test_pruning_never_drops_a_create(self, store):
        """Dropping one would make its entity look like it never existed."""
        add(store, 3)
        with store.backend.connect() as conn:
            conn.execute("UPDATE hyperlink_change_log SET created_at = 0")
        assert store.prune()["tombstones"] == 0
        assert len(store.since(0, owner="alice").changes) == 3

    def test_a_cursor_that_fell_off_the_back_demands_a_resync(self, store):
        """The deletions it needs to hear about are gone. Replaying what
        is left would leave it holding a session the server forgot."""
        for i in range(3):
            store.record(kind=ChangeKind.DELETED, entity="session",
                         entity_id=f"s{i}", owner="alice")
        with store.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_change_log SET created_at = ?",
                (time.time() - TOMBSTONE_TTL_SECONDS - 10,),
            )
        store.prune()
        add(store, 2)
        assert store.since(1, owner="alice").resync_required is True

    def test_a_current_cursor_does_not(self, store):
        add(store, 5)
        assert store.since(3, owner="alice").resync_required is False

    def test_a_brand_new_device_is_not_told_to_resync(self, store):
        """Cursor 0 means "I have nothing", not "I fell behind"."""
        add(store, 5)
        assert store.since(0, owner="alice").resync_required is False

    def test_an_empty_log_is_not_a_gap(self, store):
        """Nothing has been forgotten; there was nothing to forget."""
        assert store.since(7, owner="alice").resync_required is False


class TestSequenceNumbers:
    def test_they_are_unique_under_concurrency(self, store):
        """Not `MAX(seq) + 1`: two writers reading the same maximum pick
        the same number, and the second insert either fails on the
        primary key or -- worse, on a backend without one -- succeeds,
        producing two rows a client can only ever see one of."""
        errors: list[str] = []

        def writer(tag: int) -> None:
            try:
                for i in range(30):
                    store.record(
                        kind=ChangeKind.CREATED, entity="message",
                        entity_id=f"t{tag}m{i}", owner="alice", session_id="s1",
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors[:3]
        seqs = [c.seq for c in store.since(0, owner="alice", limit=MAX_PAGE).changes]
        assert len(seqs) == 240
        assert len(set(seqs)) == 240, "duplicate sequence numbers"
        assert seqs == list(range(1, 241)), "gaps or reordering"

    def test_a_batch_is_written_in_one_transaction(self, store):
        """A burst of messages must not be observable half-applied."""
        changes = store.record_many([
            {"kind": ChangeKind.CREATED, "entity": "message",
             "entity_id": f"m{i}", "owner": "alice", "session_id": "s1"}
            for i in range(4)
        ])
        assert [c.seq for c in changes] == [1, 2, 3, 4]
        assert len(store.since(0, owner="alice").changes) == 4

    def test_a_batch_with_a_bad_entry_writes_nothing(self, store):
        """Validated up front, so a partial batch cannot be committed."""
        with pytest.raises(T1APIError):
            store.record_many([
                {"kind": ChangeKind.CREATED, "entity": "message",
                 "entity_id": "m0", "owner": "alice"},
                {"kind": "exploded", "entity": "message",
                 "entity_id": "m1", "owner": "alice"},
            ])
        assert store.since(0, owner="alice").changes == []


class TestRecordValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "exploded", "entity": "message", "entity_id": "m", "owner": "a"},
            {"kind": ChangeKind.CREATED, "entity": "telepathy",
             "entity_id": "m", "owner": "a"},
            {"kind": ChangeKind.CREATED, "entity": "message",
             "entity_id": "", "owner": "a"},
            {"kind": ChangeKind.CREATED, "entity": "message",
             "entity_id": "m", "owner": ""},
        ],
    )
    def test_a_malformed_change_is_refused(self, store, kwargs):
        with pytest.raises(T1APIError) as caught:
            store.record(**kwargs)
        assert caught.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_the_error_lists_what_was_expected(self, store):
        with pytest.raises(T1APIError) as caught:
            store.record(
                kind="exploded", entity="message", entity_id="m", owner="a"
            )
        assert ChangeKind.CREATED in str(caught.value)


class TestTheWireShape:
    """It crosses a JSON boundary to a Swift decoder and back."""

    def test_empty_optional_fields_are_omitted(self, store):
        """So a device change does not carry a blank session_id that a
        decoder has to treat as meaningful."""
        change = store.record(
            kind=ChangeKind.CREATED, entity="device", entity_id="d1", owner="alice"
        )
        out = change.to_dict()
        assert "session_id" not in out
        assert "device_id" not in out
        assert "payload" not in out

    def test_set_fields_are_present(self, store):
        change = store.record(
            kind=ChangeKind.CREATED, entity="message", entity_id="m1",
            owner="alice", session_id="s1", device_id="dev1",
            payload={"preview": "hi"},
        )
        out = change.to_dict()
        assert out["session_id"] == "s1"
        assert out["device_id"] == "dev1"
        assert out["payload"] == {"preview": "hi"}

    def test_the_owner_is_not_on_the_wire(self, store):
        """The client already authenticated as them; repeating it in
        every row is bytes on a cellular link for no information."""
        change = store.record(
            kind=ChangeKind.CREATED, entity="message", entity_id="m1", owner="alice"
        )
        assert "owner" not in change.to_dict()

    def test_a_page_serialises_whole(self, store):
        add(store, 3)
        out = store.since(0, owner="alice", limit=2).to_dict()
        assert set(out) == {"changes", "cursor", "more", "resync_required"}
        assert len(out["changes"]) == 2

    def test_a_payload_round_trips(self, store):
        nested = {"a": [1, 2, {"b": "こんにちは"}], "c": None}
        store.record(
            kind=ChangeKind.UPDATED, entity="session", entity_id="s1",
            owner="alice", payload=nested,
        )
        assert store.since(0, owner="alice").changes[0].payload == nested
