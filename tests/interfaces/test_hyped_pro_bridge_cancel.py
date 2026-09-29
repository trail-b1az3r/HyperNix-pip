"""hyped-pro bridge: cancellation, and the threading that makes it possible.

Requests used to be dispatched inline off the stdin loop, which made
``cancel`` undeliverable: a ninety-second ``chat`` held the loop for ninety
seconds, so a cancel sent at second two wasn't *read* until second
ninety-one. The TUI's Escape key therefore couldn't stop anything — it only
discarded an answer the model had already finished producing.

The tests here pin the two halves of the fix: that a control command is
answered while a long one is still running, and that the response says
honestly whether the backend could actually be interrupted.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from hypernix.interfaces import hyped_pro_bridge as bridge
from hypernix.interfaces import hyped_pro_core as core


class _Stdin:
    """A pipe the test writes lines into and ``serve()`` reads as a file.

    ``serve()`` iterates ``sys.stdin``, so this has to block on ``__next__``
    until a line is available rather than returning EOF — otherwise the loop
    would exit before the test could send its second command, which is the
    exact interleaving under test.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._closed = False
        self._cond = threading.Condition()

    def send(self, payload: dict) -> None:
        with self._cond:
            self._lines.append(json.dumps(payload) + "\n")
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        with self._cond:
            while not self._lines and not self._closed:
                self._cond.wait(timeout=5)
            if self._lines:
                return self._lines.pop(0)
            raise StopIteration


class _Responses:
    """Collects what the bridge emits and lets the test wait for one by id.

    This stands in for ``bridge._write`` rather than for ``sys.stdout``:
    pytest owns stdout during a test, so intercepting at the bridge's own
    write seam is both more reliable and a closer statement of what is
    being asserted — the responses the bridge produces, not the bytes that
    happen to reach a terminal.
    """

    def __init__(self) -> None:
        self._lines: list[dict] = []
        self._cond = threading.Condition()

    def __call__(self, payload: dict) -> None:
        with self._cond:
            self._lines.append(payload)
            self._cond.notify_all()

    def wait_for(self, id_: int, timeout: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for entry in self._lines:
                    if entry.get("id") == id_:
                        return entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"no response for id={id_} within {timeout}s; got {self._lines}"
                    )
                self._cond.wait(timeout=remaining)


@pytest.fixture
def served(monkeypatch):
    """Run ``serve()`` on a thread against a fake stdin, and clean up after."""
    stdin, responses = _Stdin(), _Responses()
    monkeypatch.setattr(bridge.sys, "stdin", stdin)
    monkeypatch.setattr(bridge, "_write", responses)
    with bridge._inflight_lock:
        bridge._inflight.clear()

    thread = threading.Thread(target=bridge.serve, daemon=True)
    thread.start()
    try:
        yield stdin, responses
    finally:
        stdin.close()
        thread.join(timeout=5)


MAX_FAKE_TOKENS = 400


@pytest.fixture
def slow_chat(monkeypatch):
    """A chat backend that polls should_stop, like the local model path.

    It blocks on an event rather than sleeping per token. A sleep-based
    loop makes the whole file hostage to timer granularity — Windows
    rounds a 5ms sleep up to ~15ms, so a 400-token loop that takes 2s on
    Linux takes 6s there and the tests turn into a race against their own
    timeouts. Waiting on an event the test controls is exact everywhere:
    the generation runs until it is cancelled, or until the test releases
    it, and never on a clock.
    """
    state = {
        "tokens": 0,
        "started": threading.Event(),
        "release": threading.Event(),
        "generated_one": threading.Event(),
    }

    def fake_chat(*, should_stop=None, **_kwargs):
        state["started"].set()
        for _ in range(MAX_FAKE_TOKENS):
            if should_stop is not None and should_stop():
                break
            state["tokens"] += 1
            state["generated_one"].set()
            # Returns as soon as the test releases it, or polls again.
            if state["release"].wait(timeout=0.01):
                break
        return {"content": "tok " * state["tokens"], "thinking": None}

    monkeypatch.setattr(core, "send_chat_message", fake_chat)
    try:
        yield state
    finally:
        # Never leave a worker spinning if the test failed mid-way.
        state["release"].set()


@pytest.fixture
def interruptible(monkeypatch):
    monkeypatch.setattr(bridge, "_interruptible", lambda _req: True)


@pytest.fixture
def uninterruptible(monkeypatch):
    monkeypatch.setattr(bridge, "_interruptible", lambda _req: False)


def _chat(id_: int = 1) -> dict:
    return {"id": id_, "cmd": "chat", "model": "qwen3-4b-gguf",
            "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# The loop stays responsive
# ---------------------------------------------------------------------------


class TestLoopStaysResponsive:
    def test_a_control_command_is_answered_during_a_long_chat(
        self, served, slow_chat, interruptible
    ):
        """The whole point: this used to be impossible."""
        stdin, stdout = served
        stdin.send(_chat(1))
        assert slow_chat["started"].wait(timeout=5)

        stdin.send({"id": 2, "cmd": "t1api_get_url"})
        answered = stdout.wait_for(2, timeout=5)

        assert answered["ok"] is True
        stdin.send({"id": 3, "cmd": "cancel", "target": 1})
        stdout.wait_for(1, timeout=10)

    def test_two_chats_do_not_serialize_behind_each_other(
        self, served, slow_chat, interruptible
    ):
        stdin, stdout = served
        stdin.send(_chat(1))
        stdin.send(_chat(2))
        assert slow_chat["started"].wait(timeout=5)
        stdin.send({"id": 3, "cmd": "cancel", "target": 1})
        stdin.send({"id": 4, "cmd": "cancel", "target": 2})
        assert stdout.wait_for(1, timeout=10)["ok"] is True
        assert stdout.wait_for(2, timeout=10)["ok"] is True


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_stops_an_interruptible_generation(
        self, served, slow_chat, interruptible
    ):
        stdin, stdout = served
        stdin.send(_chat(1))
        assert slow_chat["generated_one"].wait(timeout=5)

        stdin.send({"id": 2, "cmd": "cancel", "target": 1})
        assert stdout.wait_for(2)["data"]["result"] == "stopped"

        reply = stdout.wait_for(1, timeout=10)
        assert reply["ok"] is True
        assert reply["data"]["cancelled"] is True
        # It stopped well short of what it would otherwise have produced.
        assert slow_chat["tokens"] < MAX_FAKE_TOKENS

    def test_a_cancelled_turn_keeps_what_was_generated(
        self, served, slow_chat, interruptible
    ):
        """A partial answer is the work that was actually done; discarding
        it would waste the tokens the person already waited for."""
        stdin, stdout = served
        stdin.send(_chat(1))
        assert slow_chat["generated_one"].wait(timeout=5)
        stdin.send({"id": 2, "cmd": "cancel", "target": 1})

        reply = stdout.wait_for(1, timeout=10)
        assert reply["data"]["reply"].strip()

    def test_an_uninterruptible_backend_says_pending_not_stopped(
        self, served, slow_chat, uninterruptible
    ):
        """A cloud HTTP call in flight has no interruption point. Reporting
        "stopped" would be the same lie the TUI was already telling."""
        stdin, stdout = served
        stdin.send(_chat(1))
        assert slow_chat["started"].wait(timeout=5)

        stdin.send({"id": 2, "cmd": "cancel", "target": 1})
        assert stdout.wait_for(2)["data"]["result"] == "pending"
        stdout.wait_for(1, timeout=20)

    def test_cancelling_an_unknown_id_says_so(self, served):
        stdin, stdout = served
        stdin.send({"id": 1, "cmd": "cancel", "target": 999})
        assert stdout.wait_for(1)["data"]["result"] == "unknown"

    def test_cancelling_a_finished_request_says_unknown(
        self, served, slow_chat, interruptible
    ):
        stdin, stdout = served
        stdin.send(_chat(1))
        assert slow_chat["started"].wait(timeout=5)
        stdin.send({"id": 2, "cmd": "cancel", "target": 1})
        stdout.wait_for(1, timeout=10)

        stdin.send({"id": 3, "cmd": "cancel", "target": 1})
        assert stdout.wait_for(3)["data"]["result"] == "unknown"

    def test_an_uncancelled_turn_is_not_marked_cancelled(self, served, monkeypatch):
        monkeypatch.setattr(
            core, "send_chat_message",
            lambda **_kw: {"content": "done", "thinking": None},
        )
        stdin, stdout = served
        stdin.send(_chat(1))
        reply = stdout.wait_for(1)
        assert reply["data"]["cancelled"] is False
        assert reply["data"]["reply"] == "done"

    def test_cancel_targets_one_request_not_all_of_them(
        self, served, slow_chat, interruptible
    ):
        stdin, stdout = served
        stdin.send(_chat(1))
        stdin.send(_chat(2))
        assert slow_chat["started"].wait(timeout=5)

        stdin.send({"id": 3, "cmd": "cancel", "target": 1})
        assert stdout.wait_for(1, timeout=10)["data"]["cancelled"] is True

        stdin.send({"id": 4, "cmd": "cancel", "target": 2})
        stdout.wait_for(2, timeout=10)


# ---------------------------------------------------------------------------
# Which backends can be interrupted
# ---------------------------------------------------------------------------


class TestInterruptibility:
    def test_only_chat_is_interruptible(self):
        assert not bridge._interruptible({"cmd": "download", "model": "deepseek-r1"})

    def test_a_local_safetensors_model_is_interruptible(self):
        """NeoOven's generation loop takes a should_stop."""
        assert bridge._interruptible({"cmd": "chat", "model": "deepseek-r1"})

    def test_a_local_gguf_model_is_not(self):
        """It runs inside multilama/llama.cpp, which has no stop hook."""
        assert not bridge._interruptible({"cmd": "chat", "model": "qwen3-4b-gguf"})

    def test_a_cloud_model_is_not(self):
        assert not bridge._interruptible({"cmd": "chat", "model": "claude-sonnet-5"})

    def test_an_unknown_model_does_not_raise(self):
        """It fails later with a real error; this check must not pre-empt it."""
        assert not bridge._interruptible({"cmd": "chat", "model": "no-such-model"})


# ---------------------------------------------------------------------------
# Protocol robustness
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_bad_json_gets_an_error_line_not_a_crash(self, served):
        stdin, stdout = served
        with stdin._cond:
            stdin._lines.append("{not json\n")
            stdin._cond.notify_all()
        stdin.send({"id": 1, "cmd": "t1api_get_url"})
        assert stdout.wait_for(1)["ok"] is True

    def test_unknown_command_is_refused(self, served):
        stdin, stdout = served
        stdin.send({"id": 1, "cmd": "nonsense"})
        resp = stdout.wait_for(1)
        assert resp["ok"] is False
        assert resp["code"] == "HPB-PROTO-001"

    def test_a_worker_exception_still_answers(self, served, monkeypatch):
        def boom(**_kwargs):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(core, "send_chat_message", boom)
        stdin, stdout = served
        stdin.send(_chat(1))
        resp = stdout.wait_for(1)
        assert resp["ok"] is False
        assert "backend exploded" in resp["error"]

    def test_a_failed_request_is_not_left_in_flight(self, served, monkeypatch):
        monkeypatch.setattr(
            core, "send_chat_message",
            lambda **_kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        stdin, stdout = served
        stdin.send(_chat(1))
        stdout.wait_for(1)
        with bridge._inflight_lock:
            assert 1 not in bridge._inflight

    def test_shutdown_releases_everything_still_running(
        self, monkeypatch, slow_chat, interruptible
    ):
        """Closing stdin must not leave a generation running with nobody to
        send its answer to."""
        stdin, stdout = _Stdin(), _Responses()
        monkeypatch.setattr(bridge.sys, "stdin", stdin)
        monkeypatch.setattr(bridge, "_write", stdout)
        with bridge._inflight_lock:
            bridge._inflight.clear()

        thread = threading.Thread(target=bridge.serve, daemon=True)
        thread.start()
        stdin.send(_chat(1))
        assert slow_chat["started"].wait(timeout=5)
        stdin.close()
        thread.join(timeout=5)

        stdout.wait_for(1, timeout=10)
        assert slow_chat["tokens"] < 400
