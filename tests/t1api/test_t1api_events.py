"""Tests for hypernix.t1api.events — the in-process pub/sub event bus
behind GET /events."""
from __future__ import annotations

from hypernix.t1api.events import EventBus


class TestPublishAndHistory:
    def test_publish_returns_event_with_type_and_data(self):
        bus = EventBus()
        e = bus.publish("job.created", {"job_id": "j1"}, source="jobs")
        assert e.type == "job.created"
        assert e.data == {"job_id": "j1"}
        assert e.source == "jobs"

    def test_recent_includes_events_published_before_any_subscriber(self):
        bus = EventBus()
        bus.publish("a", {}, source="x")
        assert len(bus.recent()) == 1

    def test_history_is_bounded(self):
        bus = EventBus(history_size=5)
        for i in range(10):
            bus.publish("noise", {"i": i}, source="x")
        assert len(bus.recent(limit=100)) == 5

    def test_since_id_filters_to_events_after_that_one(self):
        bus = EventBus()
        e1 = bus.publish("a", {}, source="x")
        bus.publish("b", {}, source="x")
        bus.publish("c", {}, source="x")
        remaining = bus.recent(since_id=e1.event_id)
        assert [e.type for e in remaining] == ["b", "c"]

    def test_event_type_filter(self):
        bus = EventBus()
        bus.publish("job.created", {}, source="x")
        bus.publish("module.synced", {}, source="x")
        bus.publish("job.created", {}, source="x")
        filtered = bus.recent(event_type="job.created")
        assert len(filtered) == 2
        assert all(e.type == "job.created" for e in filtered)

    def test_limit_returns_most_recent(self):
        bus = EventBus()
        for i in range(5):
            bus.publish("x", {"i": i}, source="s")
        recent = bus.recent(limit=2)
        assert [e.data["i"] for e in recent] == [3, 4]


class TestSubscribe:
    def test_subscriber_only_gets_events_after_subscribing(self):
        bus = EventBus()
        bus.publish("before", {}, source="x")
        sub = bus.subscribe()
        bus.publish("after", {}, source="x")
        received = list(sub.iter(timeout=0.1))
        assert [e.type for e in received] == ["after"]

    def test_multiple_subscribers_each_receive_independently(self):
        bus = EventBus()
        sub_a = bus.subscribe()
        sub_b = bus.subscribe()
        bus.publish("ping", {}, source="x")
        a = list(sub_a.iter(timeout=0.1))
        b = list(sub_b.iter(timeout=0.1))
        assert len(a) == 1
        assert len(b) == 1

    def test_unsubscribe_stops_future_delivery(self):
        bus = EventBus()
        sub = bus.subscribe()
        bus.unsubscribe(sub)
        bus.publish("ping", {}, source="x")
        received = list(sub.iter(timeout=0.05))
        assert received == []

    def test_slow_subscriber_drops_oldest_rather_than_blocking_publisher(self):
        bus = EventBus()
        sub = bus.subscribe(maxsize=2)
        for i in range(10):
            bus.publish("flood", {"i": i}, source="x")  # must not raise/block
        received = list(sub.iter(timeout=0.05))
        assert len(received) <= 2

    def test_subscriber_count_tracks_active_subscribers(self):
        bus = EventBus()
        assert bus.subscriber_count() == 0
        sub = bus.subscribe()
        assert bus.subscriber_count() == 1
        bus.unsubscribe(sub)
        assert bus.subscriber_count() == 0
