"""``hypernix.hyperlink.notify`` — telling a phone something happened.

Every interesting event on a HyperNix machine happens on a timescale a
phone is not awake for: a fine-tune runs for six hours, a 70B download
takes forty minutes, a chat turn against a large local model takes long
enough for iOS to suspend the app. This module is the machine's half of
saying so.

Two bugs in here were only findable by running it against real text, and
both are pinned below.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from hypernix.hyperlink.notify import (
    BACKOFF_SECONDS,
    DEFAULT_EVENTS,
    MAX_ATTEMPTS,
    MAX_PAYLOAD_BYTES,
    EventKind,
    NotificationStore,
    Notifier,
    PermanentDeliveryError,
    RecordingTransport,
    build_apns_payload,
    encode_payload,
    fingerprint_token,
    truncate_to_bytes,
)
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode

TOKEN = "a" * 64
OTHER_TOKEN = "b" * 64


@pytest.fixture
def store(tmp_path: pathlib.Path) -> NotificationStore:
    return NotificationStore(SQLiteBackend(db_path=tmp_path / "notify.sqlite3"))


@pytest.fixture
def registered(store: NotificationStore):
    reg = store.register(
        device_id="dev1", owner="alice", token=TOKEN, bundle_id="org.hypernix.hyperlink"
    )
    return store, reg


def payload(body: str = "hello", event: str = EventKind.CHAT_REPLY) -> dict:
    return build_apns_payload(title="Reply", body=body, event=event)


class TestThePayloadFits:
    """APNs refuses anything over 4096 bytes, and a reply is often more.

    The first version of the builder subtracted the skeleton's length
    from the limit and trimmed the body to the difference. That is wrong
    in two independent ways, and both shipped.
    """

    def test_the_encoding_is_utf8_not_escaped_ascii(self):
        """Bug one: `json.dumps` escapes non-ASCII by default.

        `ensure_ascii=True` turns each Japanese character into a
        six-byte `\\uXXXX` sequence where UTF-8 needs three. The
        builder measured UTF-8 and the payload went out at roughly
        double — 8069 bytes against a 4096-byte limit, refused by APNs
        for every reply that was not plain English.
        """
        text = "こんにちは"
        assert len(json.dumps(text).encode("utf-8")) > len(
            json.dumps(text, ensure_ascii=False).encode("utf-8")
        ), "the premise of this test no longer holds"
        built = build_apns_payload(title="t", body=text, event=EventKind.CHAT_REPLY)
        assert "\\u" not in encode_payload(built).decode("utf-8")

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            ("plain ascii", "hello world " * 2000),
            ("japanese", "こんにちは" * 3000),
            ("emoji", "🔥🧊" * 2000),
            ("nothing but quotes", '"' * 8000),
            ("nothing but backslashes", "\\" * 8000),
            ("nothing but newlines", "\n" * 8000),
            ("mixed scripts and escapes", 'a"\\\nこ🔥' * 2000),
            ("python source", 'def f(x):\n    return {"a": [1], "b": "c\\\\d"}\n' * 300),
            ("short", "fine"),
            ("empty", ""),
        ],
    )
    def test_it_fits_whatever_the_body_is(self, name, body):
        built = build_apns_payload(
            title="Reply", body=body, event=EventKind.CHAT_REPLY,
            data={"session_id": "s1", "message_id": "m1"},
        )
        wire = encode_payload(built)
        assert len(wire) <= MAX_PAYLOAD_BYTES, f"{name}: {len(wire)} bytes"
        # Still decodable: a cut through the middle of a multi-byte
        # sequence would make the whole payload invalid JSON.
        wire.decode("utf-8")

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            ("nothing but quotes", '"' * 8000),
            ("nothing but backslashes", "\\" * 8000),
            ("python source", 'def f(x):\n    return {"a": [1], "b": "c\\\\d"}\n' * 300),
        ],
    )
    def test_a_body_that_escapes_is_trimmed_not_discarded(self, name, body):
        """Bug two: subtracting the overflow over-corrects to nothing.

        A quote escapes to two bytes, so the first overflow on a body of
        8000 quotes is about as large as the whole budget; subtracting
        it drove the budget to zero and produced a 143-byte payload with
        an empty body. A model reply containing code arrived with no
        text in it.

        Binary search finds the real maximum instead, so these now fill
        the payload rather than emptying it.
        """
        built = build_apns_payload(title="Reply", body=body, event=EventKind.CHAT_REPLY)
        kept = built["aps"]["alert"]["body"]
        assert len(kept) > 500, f"{name}: only kept {len(kept)} characters"
        assert len(encode_payload(built)) > MAX_PAYLOAD_BYTES - 64, (
            f"{name}: left {MAX_PAYLOAD_BYTES - len(encode_payload(built))} bytes unused"
        )

    def test_a_body_that_already_fits_is_untouched(self):
        built = build_apns_payload(title="t", body="short", event=EventKind.CHAT_REPLY)
        assert built["aps"]["alert"]["body"] == "short"

    def test_a_title_longer_than_the_budget_cannot_squeeze_out_the_body(self):
        """No amount of body-trimming saves a payload whose title is 9 KB."""
        built = build_apns_payload(
            title="T" * 9000, body="the body", event=EventKind.CHAT_REPLY
        )
        assert len(encode_payload(built)) <= MAX_PAYLOAD_BYTES

    def test_the_custom_data_survives_trimming(self):
        """It is the part a client cannot reconstruct from the text."""
        built = build_apns_payload(
            title="Reply", body="x" * 9000, event=EventKind.CHAT_REPLY,
            data={"session_id": "s1", "message_id": "m1"},
        )
        assert built["hnx"]["data"] == {"session_id": "s1", "message_id": "m1"}
        assert built["hnx"]["event"] == EventKind.CHAT_REPLY

    def test_enqueue_measures_the_same_bytes_it_will_send(self, registered):
        """Or the queue accepts what APNs will reject."""
        store, _ = registered
        oversized = {"aps": {"alert": {"title": "t", "body": "x" * 5000}}}
        with pytest.raises(T1APIError) as caught:
            store.enqueue(
                owner="alice", event=EventKind.CHAT_REPLY, payload=oversized
            )
        assert caught.value.code == T1ErrorCode.PAYLOAD_TOO_LARGE
        assert "build_apns_payload" in str(caught.value), "does not say how to fix it"


class TestTruncateToBytes:
    def test_it_counts_bytes_not_characters(self):
        assert truncate_to_bytes("hello", 5) == "hello"
        # Five characters, fifteen bytes: a character-based cut would
        # let this through a byte-based limit three times over.
        assert len("こんにちは".encode()) == 15
        assert len(truncate_to_bytes("こんにちは", 10).encode()) <= 10

    def test_it_never_cuts_mid_sequence(self):
        for limit in range(1, 20):
            out = truncate_to_bytes("こんにちは", limit)
            out.encode("utf-8").decode("utf-8")  # raises if invalid
            assert len(out.encode()) <= limit

    def test_astral_plane_characters_survive_intact(self):
        """An emoji is four bytes and a surrogate pair in UTF-16.

        Cutting one in half is the classic way to produce a string that
        looks fine in Python and crashes a Swift decoder.
        """
        for limit in range(1, 24):
            out = truncate_to_bytes("🔥🧊🌋", limit)
            out.encode("utf-8").decode("utf-8")
            assert "�" not in out

    def test_it_marks_what_it_trimmed(self):
        assert truncate_to_bytes("hello world", 8).endswith("…")

    def test_the_marker_is_counted_against_the_limit(self):
        """Appending it after measuring is how a fit becomes an overflow."""
        out = truncate_to_bytes("hello world", 8)
        assert len(out.encode()) <= 8

    @pytest.mark.parametrize("limit", [0, -1, -100])
    def test_a_nonsense_limit_gives_nothing(self, limit):
        assert truncate_to_bytes("hello", limit) == ""


class TestTokensAreCredentials:
    """A device token lets its holder push to that device.

    So it is handled the way keys are handled everywhere else here:
    stored because delivery needs it, and never emitted.
    """

    def test_the_api_shape_has_no_token_in_it(self, registered):
        _, reg = registered
        assert "token" not in reg.to_dict()
        assert TOKEN not in json.dumps(reg.to_dict())

    def test_the_repr_has_no_token_in_it(self, registered):
        """A dataclass repr would put it in every traceback and log."""
        _, reg = registered
        assert TOKEN not in repr(reg)
        assert reg.fingerprint in repr(reg)

    def test_a_listing_has_no_token_in_it(self, registered):
        store, _ = registered
        assert TOKEN not in json.dumps([r.to_dict() for r in store.list_registrations(owner="alice")])

    def test_the_validation_error_does_not_echo_the_value(self, store):
        """An error string is a place secrets leak."""
        with pytest.raises(T1APIError) as caught:
            store.register(device_id="d", owner="o", token="not-hex-but-secret")
        assert "not-hex-but-secret" not in str(caught.value)

    def test_the_fingerprint_identifies_without_revealing(self):
        assert fingerprint_token(TOKEN) == fingerprint_token(TOKEN)
        assert fingerprint_token(TOKEN) != fingerprint_token(OTHER_TOKEN)
        assert len(fingerprint_token(TOKEN)) == 8
        assert TOKEN[:8] not in fingerprint_token(TOKEN)

    def test_the_test_double_records_the_fingerprint_only(self, registered):
        """A fixture must not become a place tokens are written down."""
        store, reg = registered
        transport = RecordingTransport()
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        Notifier(store, transport).drain()
        assert TOKEN not in json.dumps(transport.sent)
        assert transport.sent[0]["fingerprint"] == reg.fingerprint


class TestRegistration:
    def test_a_default_registration_gets_the_default_events(self, registered):
        _, reg = registered
        assert reg.events == frozenset(DEFAULT_EVENTS)

    def test_re_registering_the_same_token_does_not_duplicate(self, store):
        """iOS hands the app a token on every launch.

        A row per launch would send every notification once per day the
        app had been opened.
        """
        first = store.register(device_id="dev1", owner="alice", token=TOKEN)
        second = store.register(device_id="dev1", owner="alice", token=TOKEN)
        assert first.registration_id == second.registration_id
        assert len(store.list_registrations(owner="alice")) == 1

    def test_re_registering_clears_a_failure_count(self, store):
        """A new token from the same device is a fresh start."""
        reg = store.register(device_id="dev1", owner="alice", token=TOKEN)
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        transport = RecordingTransport()
        transport.fail_next = 1
        Notifier(store, transport).drain()
        assert store.get_registration(reg.registration_id).failure_count == 1
        again = store.register(device_id="dev1", owner="alice", token=TOKEN)
        assert again.failure_count == 0

    def test_a_second_device_is_a_second_registration(self, store):
        store.register(device_id="dev1", owner="alice", token=TOKEN)
        store.register(device_id="dev2", owner="alice", token=OTHER_TOKEN)
        assert len(store.list_registrations(owner="alice")) == 2

    def test_one_owner_cannot_see_anothers_registrations(self, store):
        store.register(device_id="dev1", owner="alice", token=TOKEN)
        store.register(device_id="dev2", owner="bob", token=OTHER_TOKEN)
        assert len(store.list_registrations(owner="alice")) == 1
        assert len(store.list_registrations(owner="bob")) == 1

    @pytest.mark.parametrize(
        "bad", ["", "short", "g" * 64, "a" * 63, "a" * 201, "  " + "a" * 64]
    )
    def test_a_token_that_is_not_a_token_is_refused(self, store, bad):
        with pytest.raises(T1APIError) as caught:
            store.register(device_id="d", owner="o", token=bad)
        assert caught.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_an_unknown_event_kind_is_refused_with_the_valid_list(self, store):
        with pytest.raises(T1APIError) as caught:
            store.register(
                device_id="d", owner="o", token=TOKEN, events=["chat.reply", "nope"]
            )
        assert "nope" in str(caught.value)
        assert EventKind.TRAINING_DONE in str(caught.value), "does not list the options"

    def test_an_unknown_environment_is_refused(self, store):
        """Sending a sandbox token to production APNs fails silently."""
        with pytest.raises(T1APIError):
            store.register(
                device_id="d", owner="o", token=TOKEN, environment="staging"
            )

    def test_events_can_be_narrowed_afterwards(self, registered):
        store, reg = registered
        updated = store.set_events(reg.registration_id, [EventKind.TRAINING_DONE])
        assert updated.events == frozenset({EventKind.TRAINING_DONE})
        assert not updated.wants(EventKind.CHAT_REPLY)

    def test_unregistering_takes_the_queue_with_it(self, registered):
        """Otherwise it keeps trying to reach a device that just said stop."""
        store, reg = registered
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        assert store.pending_count(owner="alice") == 1
        store.unregister(reg.registration_id)
        assert store.pending_count(owner="alice") == 0

    def test_unregistering_something_absent_says_so(self, store):
        with pytest.raises(T1APIError) as caught:
            store.unregister("push_nope")
        assert caught.value.code == T1ErrorCode.NOT_FOUND

    def test_disabling_keeps_the_row_so_an_operator_can_see_why(self, registered):
        store, reg = registered
        store.disable(reg.registration_id, reason="410 Unregistered")
        assert store.list_registrations(owner="alice") == []
        kept = store.list_registrations(owner="alice", include_disabled=True)
        assert len(kept) == 1 and not kept[0].enabled


class TestTheQueue:
    def test_an_event_nobody_subscribed_to_is_dropped_at_enqueue(self, registered):
        """So a queue length of zero means "nothing to say".

        Queueing it and discarding it at send time would make the queue
        length a measure of unwanted work instead.
        """
        store, reg = registered
        store.set_events(reg.registration_id, [EventKind.TRAINING_DONE])
        assert store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload()
        ) == []
        assert store.pending_count(owner="alice") == 0

    def test_one_event_fans_out_to_every_subscribed_device(self, store):
        store.register(device_id="dev1", owner="alice", token=TOKEN)
        store.register(device_id="dev2", owner="alice", token=OTHER_TOKEN)
        queued = store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload()
        )
        assert len(queued) == 2

    def test_an_unknown_event_kind_is_refused(self, registered):
        store, _ = registered
        with pytest.raises(T1APIError) as caught:
            store.enqueue(owner="alice", event="chat.telepathy", payload=payload())
        assert caught.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_a_collapse_id_supersedes_rather_than_stacks(self, registered):
        """Download progress at 40, 60 and 80 percent is one notification.

        Three would be three buzzes for one fact.
        """
        store, _ = registered
        for pct in (40, 60, 80):
            store.enqueue(
                owner="alice", event=EventKind.DOWNLOAD_DONE, collapse_id="dl-1",
                payload=payload(f"{pct}%", EventKind.DOWNLOAD_DONE),
            )
        due = [n for n in store.due(now=1e12) if n.collapse_id == "dl-1"]
        assert len(due) == 1
        assert "80%" in json.dumps(due[0].payload), "kept the stale one"

    def test_different_collapse_ids_do_not_supersede_each_other(self, registered):
        store, _ = registered
        for name in ("dl-1", "dl-2"):
            store.enqueue(
                owner="alice", event=EventKind.DOWNLOAD_DONE, collapse_id=name,
                payload=payload("x", EventKind.DOWNLOAD_DONE),
            )
        assert store.pending_count(owner="alice") == 2

    def test_nothing_is_due_before_its_time(self, registered):
        store, _ = registered
        store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload(), now=1000.0
        )
        assert store.due(now=999.0) == []
        assert len(store.due(now=1000.0)) == 1

    def test_purge_leaves_pending_work_alone(self, registered):
        store, _ = registered
        store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload(), now=0.0
        )
        assert store.purge(older_than=0, now=1e9) == 0
        assert store.pending_count(owner="alice") == 1


class TestDelivery:
    def test_a_successful_send_is_recorded_on_the_registration(self, registered):
        store, reg = registered
        # Enqueued at the same fake clock the drain uses: mixing real
        # time in with a fake `now` puts the notification's due time
        # about fifty years in the future.
        store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload(), now=5000.0
        )
        result = Notifier(store, RecordingTransport()).drain(now=5000.0)
        assert result == {"sent": 1, "retry": 0, "failed": 0, "dropped": 0}
        assert store.get_registration(reg.registration_id).last_delivery_at == 5000.0

    def test_a_transient_failure_is_retried_with_backoff(self, registered):
        store, _ = registered
        store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload(), now=0.0
        )
        transport = RecordingTransport()
        transport.fail_next = 1
        notifier = Notifier(store, transport)
        assert notifier.drain(now=0.0)["retry"] == 1
        # Not due again immediately -- that would be a spin, not a retry.
        assert store.due(now=1.0) == []
        assert len(store.due(now=BACKOFF_SECONDS[1] + 1)) == 1
        assert notifier.drain(now=BACKOFF_SECONDS[1] + 1)["sent"] == 1

    def test_the_backoff_grows(self):
        assert list(BACKOFF_SECONDS) == sorted(BACKOFF_SECONDS)
        assert BACKOFF_SECONDS[-1] > BACKOFF_SECONDS[1]

    def test_it_gives_up_after_max_attempts(self, registered):
        store, _ = registered
        store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload(), now=0.0
        )
        transport = RecordingTransport()
        transport.fail_next = MAX_ATTEMPTS
        notifier = Notifier(store, transport)
        moment = 0.0
        for _ in range(MAX_ATTEMPTS):
            notifier.drain(now=moment)
            moment += BACKOFF_SECONDS[-1] + 1
        assert store.pending_count(owner="alice") == 0
        assert transport.sent == []

    def test_a_permanent_failure_is_not_retried(self, registered):
        """APNs 410 means the token is gone. Retrying burns five attempts."""
        store, reg = registered
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        transport = RecordingTransport()
        transport.permanent_failures.add(TOKEN)
        assert Notifier(store, transport).drain()["failed"] == 1
        assert store.pending_count(owner="alice") == 0

    def test_a_permanent_failure_disables_the_registration(self, registered):
        """Or the next hundred notifications queue against a dead token."""
        store, reg = registered
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        transport = RecordingTransport()
        transport.permanent_failures.add(TOKEN)
        Notifier(store, transport).drain()
        assert not store.get_registration(reg.registration_id).enabled
        # And nothing new queues for it.
        assert store.enqueue(
            owner="alice", event=EventKind.CHAT_REPLY, payload=payload()
        ) == []

    def test_a_registration_deleted_mid_flight_does_not_raise(self, registered):
        store, reg = registered
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        with store.backend.connect() as conn:
            conn.execute(
                "DELETE FROM hyperlink_push_registrations WHERE registration_id = ?",
                (reg.registration_id,),
            )
        assert Notifier(store, RecordingTransport()).drain()["dropped"] == 1

    def test_a_disabled_registration_is_skipped_not_sent_to(self, registered):
        store, reg = registered
        store.enqueue(owner="alice", event=EventKind.CHAT_REPLY, payload=payload())
        store.disable(reg.registration_id)
        transport = RecordingTransport()
        Notifier(store, transport).drain()
        assert transport.sent == []

    def test_the_permanent_error_is_its_own_type(self):
        """So a transport can say "stop" and be believed."""
        assert issubclass(PermanentDeliveryError, Exception)

    def test_draining_an_empty_queue_is_a_no_op(self, store):
        assert Notifier(store, RecordingTransport()).drain() == {
            "sent": 0, "retry": 0, "failed": 0, "dropped": 0
        }


class TestTheEventVocabulary:
    def test_the_defaults_are_a_subset_of_what_exists(self):
        assert DEFAULT_EVENTS <= EventKind.ALL

    def test_the_defaults_leave_something_to_opt_into(self):
        """A default set equal to everything is not a default set."""
        assert DEFAULT_EVENTS < EventKind.ALL

    def test_failures_are_on_by_default(self):
        """The case where silence is the worst outcome."""
        assert EventKind.TRAINING_FAILED in DEFAULT_EVENTS
        assert EventKind.DOWNLOAD_FAILED in DEFAULT_EVENTS

    def test_every_kind_is_a_dotted_string(self):
        """They cross a JSON boundary to Swift and back."""
        for kind in EventKind.ALL:
            assert isinstance(kind, str) and "." in kind
