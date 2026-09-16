"""Stop, actually stopping.

The button cancelled the client's read task. Nothing told the server and
nothing told LM Studio, so the model finished the whole answer into a
socket nobody was reading — a minute and a half of GPU, and on a metered
deployment a bill for output the person had explicitly asked not to have.

Closing the connection is not the fix, which is the subtle part.
``chat_stream`` closes its upstream response in a ``finally``, and that
``finally`` runs when the generator is closed — but a *sync* generator
handed to ``StreamingResponse`` is iterated in a threadpool, and on
disconnect Starlette cancels the task awaiting the thread while the
thread keeps running ``next()`` to completion. A Python thread blocked in
a socket read cannot be interrupted. The cleanup is correct and
unreachable.

So it is cooperative: an event the streaming loop reads between chunks,
on the thread it is already on. These tests are mostly about the
boundaries of who may set it.
"""
from __future__ import annotations

import logging
import threading

import pytest

from hypernix.hyperlink.generation import GenerationRegistry


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)




@pytest.fixture
def registry() -> GenerationRegistry:
    return GenerationRegistry()


class TestTheRegistry:
    def test_a_generation_starts_uncancelled(self, registry):
        record = registry.begin("session-1", owner="mason")
        assert not record.cancelled
        assert len(registry) == 1

    def test_cancelling_sets_the_event_the_stream_reads(self, registry):
        record = registry.begin("session-1", owner="mason")
        assert registry.cancel(owner="mason", session_id="session-1") == [
            record.generation_id
        ]
        assert record.cancelled

    def test_finishing_removes_it(self, registry):
        record = registry.begin("session-1", owner="mason")
        registry.finish(record.generation_id)
        assert len(registry) == 0

    def test_finishing_twice_is_fine(self, registry):
        """The stream's `finally` and its GeneratorExit handler can both
        run. A second finish must not raise inside a `finally`."""
        record = registry.begin("session-1", owner="mason")
        registry.finish(record.generation_id)
        registry.finish(record.generation_id)

    def test_cancelling_nothing_is_not_an_error(self, registry):
        """The model finishing a quarter-second before Stop arrives is
        the common race. The person got what they asked for; putting an
        error on screen for a button that worked is the wrong answer."""
        assert registry.cancel(owner="mason", session_id="gone") == []

    def test_cancelling_twice_reports_the_second_as_nothing(self, registry):
        registry.begin("session-1", owner="mason")
        first = registry.cancel(owner="mason", session_id="session-1")
        second = registry.cancel(owner="mason", session_id="session-1")
        assert first and not second


class TestOwnership:
    """A guessed session id must reach nothing."""

    def test_another_owner_cannot_stop_it(self, registry):
        record = registry.begin("session-1", owner="mason")
        assert registry.cancel(owner="someone-else", session_id="session-1") == []
        assert not record.cancelled

    def test_another_owner_cannot_see_it(self, registry):
        registry.begin("session-1", owner="mason")
        assert registry.active(owner="someone-else") == []

    def test_another_owner_cannot_fetch_it_by_id(self, registry):
        record = registry.begin("session-1", owner="mason")
        assert registry.get(record.generation_id, owner="someone-else") is None
        assert registry.get(record.generation_id, owner="mason") is record

    def test_cancelling_everything_of_mine_leaves_yours_alone(self, registry):
        """What the app does when it goes to the background."""
        mine = registry.begin("a", owner="mason")
        yours = registry.begin("b", owner="someone-else")
        registry.cancel(owner="mason")
        assert mine.cancelled and not yours.cancelled


class TestTargeting:
    def test_one_session_does_not_stop_another(self, registry):
        first = registry.begin("session-1", owner="mason")
        second = registry.begin("session-2", owner="mason")
        registry.cancel(owner="mason", session_id="session-1")
        assert first.cancelled and not second.cancelled

    def test_a_generation_id_is_more_specific_than_a_session(self, registry):
        """Two devices open on one conversation. Stopping "whatever is
        running here" would stop the other phone's answer."""
        first = registry.begin("session-1", owner="mason")
        second = registry.begin("session-1", owner="mason")
        registry.cancel(
            owner="mason", session_id="session-1", generation_id=second.generation_id
        )
        assert second.cancelled and not first.cancelled

    def test_shutdown_stops_everything_regardless_of_owner(self, registry):
        mine = registry.begin("a", owner="mason")
        yours = registry.begin("b", owner="someone-else")
        assert registry.cancel_all() == 2
        assert mine.cancelled and yours.cancelled


class TestItIsThreadSafe:
    """The reader and the writer are different threads by construction:
    the stream runs in a threadpool and the Stop arrives on another
    request. That is why it is an Event and not a bool."""

    def test_a_concurrent_cancel_is_seen_by_the_reader(self, registry):
        record = registry.begin("session-1", owner="mason")
        seen = threading.Event()

        def reader():
            while not record.cancelled:
                pass
            seen.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        registry.cancel(owner="mason", session_id="session-1")
        assert seen.wait(timeout=5), "the streaming loop never saw the cancel"
        thread.join(timeout=5)

    def test_many_beginnings_at_once_all_register(self, registry):
        def start():
            for _ in range(20):
                registry.begin("s", owner="mason")

        threads = [threading.Thread(target=start) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert len(registry) == 80


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


def client(**env) -> TestClient:
    import os

    from hypernix.t1api.app import create_app


    # Clear the configuration, but not where the storage lives: the
    # _own_storage fixture set those, and wiping them here sent every
    # app built by this helper back to the real ~/.hypernix database.
    keep = {"T1_DB_PATH", "T1_BACKUP_DIR", "T1_MODULE_STORAGE_DIR", "T1_HYPERLINK_DIR"}
    for name in list(os.environ):
        if name.startswith("T1_") and name not in keep:
            os.environ.pop(name)
    os.environ["T1_TRUSTED_NETWORK"] = "1"
    for key, value in env.items():
        os.environ[key] = value
    return TestClient(create_app(), client=("192.168.1.50", 5432))


class TestTheEndpoint:
    def test_stopping_nothing_is_a_success(self):
        response = client().post("/hyperlink/sessions/nothing-here/chat/stop")
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_it_reports_what_it_stopped(self):
        app_client = client()
        registry = app_client.app.state.t1_generations
        # The owner a keyless LAN caller resolves to.
        owner = app_client.get("/usage/current").json()["key_id"]
        record = registry.begin("session-1", owner=owner)

        body = app_client.post("/hyperlink/sessions/session-1/chat/stop").json()
        assert body["stopped"] == [record.generation_id]
        assert record.cancelled

    def test_the_running_list_is_visible(self):
        app_client = client()
        registry = app_client.app.state.t1_generations
        owner = app_client.get("/usage/current").json()["key_id"]
        registry.begin("session-1", owner=owner)

        body = app_client.get("/hyperlink/generations").json()
        assert body["count"] == 1
        assert body["generations"][0]["session_id"] == "session-1"

    def test_it_does_not_stop_another_owners_generation(self):
        app_client = client()
        registry = app_client.app.state.t1_generations
        theirs = registry.begin("session-1", owner="somebody-else")

        body = app_client.post("/hyperlink/sessions/session-1/chat/stop").json()
        assert body["count"] == 0
        assert not theirs.cancelled


class TestItActuallyStopsTheModel:
    """The test the others exist to support.

    A bridge that never stops on its own: if Stop does not reach the
    upstream generator, this test hangs rather than failing, which is
    exactly what the bug did to a GPU.
    """

    @pytest.fixture
    def endless(self, monkeypatch):
        import itertools
        import time as clock

        import hypernix.t1api.routers.hyperlink as router

        state = {"produced": itertools.count(), "closed": threading.Event()}

        class Endless:
            base_url = "http://fake"

            def chat_stream(self, *args, **kwargs):
                try:
                    while True:
                        next(state["produced"])
                        clock.sleep(0.01)
                        yield {
                            "model": "endless",
                            "choices": [{"delta": {"content": "x"}}],
                        }
                finally:
                    state["closed"].set()

        monkeypatch.setattr(router, "_chat_bridge", lambda *a, **k: Endless())
        return state

    def test_the_upstream_generator_is_closed(self, endless):
        import time as clock

        app_client = client(T1_LMSTUDIO_ENABLED="1")
        session = app_client.post(
            "/hyperlink/sessions", json={"title": "t", "model_id": "", "system_prompt": ""}
        ).json()["session"]["session_id"]

        done: dict = {}

        def read():
            with app_client.stream(
                "POST",
                f"/hyperlink/sessions/{session}/chat/stream",
                json={"content": "go", "attachment_ids": []},
            ) as response:
                for line in response.iter_lines():
                    if '"done"' in line:
                        done["frame"] = line
                        break

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        clock.sleep(1.0)

        before = next(endless["produced"])
        stopped = app_client.post(f"/hyperlink/sessions/{session}/chat/stop").json()
        assert stopped["count"] == 1

        reader.join(timeout=15)
        assert not reader.is_alive(), "the stream never ended, so Stop did not stop it"
        clock.sleep(0.3)

        # The three things that make this a real stop rather than a
        # client-side one.
        assert endless["closed"].is_set(), "the upstream stream was left open"
        # A bound rather than zero: the flag is read at the top of the
        # loop, so a cancel arriving while the generator is mid-yield
        # costs one more chunk. What is being asserted is that it stops
        # *promptly* -- at 100 chunks a second, an endless stream that
        # ran even a second longer would be dozens.
        assert next(endless["produced"]) - before <= 3, "the model kept generating"
        assert '"cancelled": true' in done.get("frame", "")

    def test_what_streamed_before_the_stop_is_kept(self, endless):
        import json as jsonlib
        import time as clock

        app_client = client(T1_LMSTUDIO_ENABLED="1")
        session = app_client.post(
            "/hyperlink/sessions", json={"title": "t", "model_id": "", "system_prompt": ""}
        ).json()["session"]["session_id"]

        def read():
            with app_client.stream(
                "POST",
                f"/hyperlink/sessions/{session}/chat/stream",
                json={"content": "go", "attachment_ids": []},
            ) as response:
                for line in response.iter_lines():
                    if '"done"' in line:
                        break

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        clock.sleep(1.0)
        app_client.post(f"/hyperlink/sessions/{session}/chat/stop")
        reader.join(timeout=15)

        messages = app_client.get(f"/hyperlink/sessions/{session}/messages").json()
        assistant = [m for m in messages["messages"] if m["role"] == "assistant"][-1]
        # A stopped answer is still an answer. Throwing it away would
        # make Stop indistinguishable from a failure.
        assert assistant["content"]
        metadata = assistant.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = jsonlib.loads(metadata)
        assert metadata.get("truncated") is True
        assert metadata.get("finish_reason") == "cancelled"
