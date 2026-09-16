"""Noodle, reachable from hyped-pro.

Noodle's docstring has called it "the autonomous executor inside Hyped
Pro" since it shipped, and hyped-pro had no command, no bridge verb and
no idea it existed. These tests cover the connection, and they are
weighted towards the parts that were actually broken rather than towards
the parts that are easy to assert.

The biggest of those is the key store. hyped-pro's ``/key`` writes to
``~/.hypernix/config.json``; Noodle's ``ProviderSpec.resolve_key`` reads
``os.environ`` and nothing else. So somebody would set a key in the TUI,
Noodle would report no provider with a key, and both were telling the
truth. :func:`adopt_stored_keys` is the bridge and most of what follows
is about it.

The second is that a swarm run is minutes long and produces nothing until
it ends. Running one inside a bridge request would freeze the TUI for the
duration, so it is a session with a polled event feed, and the tests here
check the session lifecycle with a fake swarm rather than by spending
money on a real one.

Nothing here makes a network call.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from hypernix.interfaces.noodle import hyped as noodle


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Sessions are process-global, so a test that leaves one behind
    changes what the next one sees."""
    noodle.clear_finished()
    yield
    noodle.clear_finished()


@pytest.fixture(autouse=True)
def _no_key_leakage():
    """Undo what adopt_stored_keys() puts in the environment.

    It sets real ``os.environ`` entries on purpose — that is the whole
    mechanism, since Noodle's ProviderSpec.resolve_key reads nothing
    else. But monkeypatch cannot restore a variable it never saw being
    set, so without this a test here leaves ANTHROPIC_API_KEY set for
    every test that runs afterwards in the same process. It did:
    tests/test_security_fixes.py asserts on the *file* store and was
    reading this module's leaked environment instead.
    """
    watched = {
        name for names in noodle.KEY_VENDORS.values() for name in names
    }
    before = {name: os.environ.get(name) for name in watched}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    """A machine with no provider keys anywhere.

    Both halves have to be cleared: the environment, and the config file
    that ``get_provider_key`` falls back to. Clearing one and not the
    other is how a test passes on a laptop and fails in CI.
    """
    for names in noodle.KEY_VENDORS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "hypernix.system.config._load_config", lambda: {}, raising=False
    )
    return tmp_path


class TestKeyAdoption:
    """The bug this whole module exists for."""

    def test_a_stored_key_reaches_noodle(self, no_keys, monkeypatch):
        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key",
            lambda vendor: "sk-ant-secret" if vendor == "anthropic" else None,
        )
        assert noodle.adopt_stored_keys() == ["anthropic"]
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"

    def test_the_environment_always_wins(self, no_keys, monkeypatch):
        """get_provider_key resolves env first, and a function that
        overwrote a key the operator exported into this shell would be
        the more surprising of the two behaviours by some margin."""
        monkeypatch.setenv("OPENAI_API_KEY", "from-the-shell")
        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key",
            lambda vendor: "from-the-config",
        )
        noodle.adopt_stored_keys()
        assert os.environ["OPENAI_API_KEY"] == "from-the-shell"

    def test_the_vendor_names_are_translated(self, no_keys, monkeypatch):
        """hyped-pro stores Moonshot under 'moonshot' and Noodle's
        provider is called 'kimi'. Neither spelling is wrong; the mapping
        is the point."""
        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key",
            lambda vendor: "key" if vendor == "moonshot" else None,
        )
        assert noodle.adopt_stored_keys() == ["moonshot"]
        # Both spellings, because the spec looks in either.
        assert os.environ["MOONSHOT_API_KEY"] == "key"
        assert os.environ["KIMI_API_KEY"] == "key"

    def test_a_corrupt_config_does_not_stop_a_run(self, no_keys, monkeypatch):
        def _explode(_vendor):
            raise json.JSONDecodeError("bad", "", 0)

        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key", _explode
        )
        assert noodle.adopt_stored_keys() == []

    def test_nothing_stored_adopts_nothing(self, no_keys, monkeypatch):
        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key", lambda _vendor: None
        )
        assert noodle.adopt_stored_keys() == []


class TestProviders:
    def test_every_provider_is_listed_ready_or_not(self):
        """"Noodle has no providers" and "Noodle has nine and you have
        set a key for none of them" are different problems, and only the
        second one is fixable from inside hyped-pro."""
        from hypernix.interfaces.noodle.providers import PROVIDERS

        info = noodle.providers()
        assert len(info["providers"]) == len(PROVIDERS)

    def test_a_provider_without_a_key_says_how_to_set_one(self, no_keys, monkeypatch):
        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key", lambda _vendor: None
        )
        info = noodle.providers()
        anthropic = next(
            p for p in info["providers"] if p["provider"] == "anthropic"
        )
        assert not anthropic["ready"]
        assert "/key anthropic" in anthropic["reason"]

    def test_the_fix_it_line_names_the_right_vendor(self, no_keys, monkeypatch):
        """Telling somebody to run `/key qwen` when the key is stored
        under `dashscope` sends them round the loop a second time."""
        monkeypatch.setattr(
            "hypernix.system.config.get_provider_key", lambda _vendor: None
        )
        info = noodle.providers()
        qwen = next(p for p in info["providers"] if p["provider"] == "qwen")
        assert qwen["key_vendor"] == "dashscope"
        assert "/key dashscope" in qwen["reason"]

    def test_local_backends_are_ready_without_a_key(self):
        info = noodle.providers()
        ollama = next(p for p in info["providers"] if p["provider"] == "ollama")
        assert ollama["ready"]
        assert not ollama["paid"]

    def test_ready_ones_sort_first(self):
        readiness = [p["ready"] for p in noodle.providers()["providers"]]
        assert readiness == sorted(readiness, reverse=True)


class TestTheRootGuard:
    def test_the_default_root_is_a_subdirectory(self, tmp_path, monkeypatch):
        """An agent with file tools pointed at the repo you are sitting
        in is a thing to opt into, not a thing to get by pressing
        enter."""
        monkeypatch.chdir(tmp_path)
        assert noodle._resolve_root(None, allow_outside=False) == tmp_path / ".noodle"

    def test_a_root_outside_the_tree_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(noodle.NoodleSessionError, match="outside"):
            noodle._resolve_root("/etc", allow_outside=False)

    def test_it_can_be_asked_for(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert noodle._resolve_root("/etc", allow_outside=True).name == "etc"

    def test_a_subdirectory_is_fine(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        inner = tmp_path / "work"
        assert noodle._resolve_root(inner, allow_outside=False) == inner


class TestStarting:
    def test_an_empty_task_is_refused(self):
        with pytest.raises(noodle.NoodleSessionError, match="something to do"):
            noodle.start("   ")

    def test_verify_requires_execution(self, tmp_path, monkeypatch):
        """A verifier is a shell command. Running one while claiming
        execution is off would be the same capability under a different
        name, which is worse than not having the flag."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(noodle.NoodleSessionError, match="same permission"):
            noodle.start("x", roster=["ollama:llama3.2"], verify="pytest")

    def test_no_usable_provider_says_what_to_do(self, no_keys, monkeypatch):
        monkeypatch.setattr(
            "hypernix.interfaces.noodle.providers.available_providers",
            lambda **_kwargs: [],
        )
        with pytest.raises(noodle.NoodleSessionError, match="/key"):
            noodle._default_roster()

    def test_the_default_roster_prefers_free_models(self, monkeypatch):
        """A default that silently picked a paid frontier model would
        turn `/noodle do a thing` into an invoice."""
        roster = noodle._default_roster()
        assert roster
        assert all("ollama" in entry or "hypernix" in entry or "vllm" in entry
                   for entry in roster), roster


class _FakeSwarm:
    """A swarm that emits events and finishes, without a network call."""

    def __init__(self, on_event, events=3, delay=0.0, fail=False):
        self.on_event = on_event
        self.events = events
        self.delay = delay
        self.fail = fail
        self.submitted: list[str] = []
        self.stop_requested = False

    def submit(self, prompt, **_kwargs):
        self.submitted.append(prompt)

    def request_stop(self):
        self.stop_requested = True

    def run(self, tasks=None):
        for index in range(self.events):
            if self.stop_requested:
                break
            self.on_event({"kind": "turn", "task_id": f"task{index}"})
            if self.delay:
                time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("the provider said no")
        return _FakeReport()


class _FakeReport:
    def to_dict(self):
        return {"ok": True, "results": [], "failed": []}


@pytest.fixture
def fake_swarm(monkeypatch):
    """Replace Swarm with one that does not make network calls.

    Patched where it is looked up rather than where it is defined:
    :func:`noodle.start` does ``from .swarm import Swarm`` inside the
    function, so patching the attribute on the swarm module is what that
    import resolves to.
    """
    made: list[_FakeSwarm] = []

    def _factory(**kwargs):
        swarm = _FakeSwarm(kwargs["on_event"], **_factory.options)
        made.append(swarm)
        return swarm

    _factory.options = {}
    _factory.made = made
    monkeypatch.setattr(
        "hypernix.interfaces.noodle.swarm.Swarm",
        lambda roster, root, **kwargs: _factory(**kwargs),
    )
    return _factory


def _await_finish(session_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    tick = noodle.poll(session_id)
    while tick["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
        tick = noodle.poll(session_id)
    return tick


class TestTheSessionLifecycle:
    def test_start_returns_before_the_run_finishes(self, fake_swarm, tmp_path,
                                                   monkeypatch):
        """The reason this is a session at all: a swarm run is minutes of
        work, and doing it inside a bridge request would hold the thread
        and leave the TUI with nothing to draw."""
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 3, "delay": 0.05}
        session = noodle.start("do a thing", roster=["ollama:llama3.2"])
        assert session["session_id"]
        assert session["report"] is None
        _await_finish(session["session_id"])

    def test_events_arrive_and_are_drained(self, fake_swarm, tmp_path, monkeypatch):
        """poll() removes what it returns. A long run produces thousands
        of events and a poll that returned the whole history every second
        would grow quadratically in a TUI that redraws on each one."""
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 5}
        session = noodle.start("x", roster=["ollama:llama3.2"])
        collected = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            tick = noodle.poll(session["session_id"], timeout=0.2)
            collected.extend(tick["events"])
            if not tick["running"]:
                break
        assert len([e for e in collected if e["kind"] == "turn"]) == 5
        # Drained: a second poll returns nothing new.
        assert noodle.poll(session["session_id"])["events"] == []

    def test_the_tail_survives_draining(self, fake_swarm, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 4}
        session = noodle.start("x", roster=["ollama:llama3.2"])
        _await_finish(session["session_id"])
        tail = noodle.summary(session["session_id"])["tail"]
        assert [e for e in tail if e["kind"] == "turn"]

    def test_the_report_lands_on_the_session(self, fake_swarm, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 1}
        session = noodle.start("x", roster=["ollama:llama3.2"])
        tick = _await_finish(session["session_id"])
        assert not tick["running"]
        assert tick["report"] == {"ok": True, "results": [], "failed": []}
        assert tick["error"] == ""

    def test_a_failing_run_reports_rather_than_vanishing(self, fake_swarm, tmp_path,
                                                        monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 1, "fail": True}
        session = noodle.start("x", roster=["ollama:llama3.2"])
        tick = _await_finish(session["session_id"])
        assert not tick["running"]
        assert "the provider said no" in tick["error"]

    def test_tasks_split_the_work(self, fake_swarm, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 1}
        session = noodle.start(
            "overall", roster=["ollama:llama3.2"], tasks=["one", "two", "three"],
        )
        _await_finish(session["session_id"])
        assert fake_swarm.made[0].submitted == ["one", "two", "three"]

    def test_without_tasks_the_prompt_is_the_task(self, fake_swarm, tmp_path,
                                                  monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 1}
        session = noodle.start("just this", roster=["ollama:llama3.2"])
        _await_finish(session["session_id"])
        assert fake_swarm.made[0].submitted == ["just this"]

    def test_stop_notifies_a_swarm_that_can_be_stopped(self, fake_swarm, tmp_path,
                                                      monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 50, "delay": 0.05}
        session = noodle.start("x", roster=["ollama:llama3.2"])
        result = noodle.stop(session["session_id"])
        assert result["cancelled"]
        assert result["swarm_notified"]
        _await_finish(session["session_id"])

    def test_stop_is_honest_when_it_cannot(self, fake_swarm, tmp_path, monkeypatch):
        """A "stop" that does not stop things is worth knowing about."""
        monkeypatch.chdir(tmp_path)

        class _Unstoppable(_FakeSwarm):
            request_stop = None
            cancel = None
            stop = None

        monkeypatch.setattr(
            "hypernix.interfaces.noodle.swarm.Swarm",
            lambda roster, root, **kwargs: _Unstoppable(kwargs["on_event"], events=1),
        )
        session = noodle.start("x", roster=["ollama:llama3.2"])
        assert not noodle.stop(session["session_id"])["swarm_notified"]
        _await_finish(session["session_id"])

    def test_an_unknown_session_is_a_clear_error(self):
        with pytest.raises(noodle.NoodleSessionError, match="No Noodle session"):
            noodle.poll("deadbeef")

    def test_sessions_are_listed_newest_first(self, fake_swarm, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 1}
        first = noodle.start("one", roster=["ollama:llama3.2"])
        _await_finish(first["session_id"])
        time.sleep(0.01)
        second = noodle.start("two", roster=["ollama:llama3.2"])
        _await_finish(second["session_id"])
        listed = noodle.sessions()
        assert listed[0]["session_id"] == second["session_id"]

    def test_a_running_session_is_never_evicted(self, fake_swarm, tmp_path,
                                                monkeypatch):
        """Evicting one would lose the handle to something still burning
        tokens."""
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 200, "delay": 0.02}
        long_running = noodle.start("long", roster=["ollama:llama3.2"])
        fake_swarm.options = {"events": 1}
        for index in range(noodle.MAX_SESSIONS + 4):
            done = noodle.start(f"quick{index}", roster=["ollama:llama3.2"])
            _await_finish(done["session_id"])
        assert long_running["session_id"] in {s["session_id"] for s in noodle.sessions()}
        noodle.stop(long_running["session_id"])
        _await_finish(long_running["session_id"], timeout=10.0)

    def test_the_event_tail_is_bounded(self, fake_swarm, tmp_path, monkeypatch):
        """A long-lived TUI must not accumulate a swarm's whole history."""
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": noodle.MAX_TAIL + 50}
        session = noodle.start("x", roster=["ollama:llama3.2"])
        _await_finish(session["session_id"], timeout=10.0)
        assert len(noodle.summary(session["session_id"])["tail"]) <= noodle.MAX_TAIL

    def test_a_bad_event_does_not_break_a_run(self, fake_swarm, tmp_path, monkeypatch):
        """record() is called from worker threads. An event that cannot
        be serialised must not take the swarm down with it."""
        monkeypatch.chdir(tmp_path)

        class _Weird:
            def to_dict(self):
                raise TypeError("not serialisable")

        class _Emitter(_FakeSwarm):
            def run(self, tasks=None):
                self.on_event(_Weird())
                return _FakeReport()

        monkeypatch.setattr(
            "hypernix.interfaces.noodle.swarm.Swarm",
            lambda roster, root, **kwargs: _Emitter(kwargs["on_event"]),
        )
        session = noodle.start("x", roster=["ollama:llama3.2"])
        tick = _await_finish(session["session_id"])
        assert tick["error"] == ""
        assert any("repr" in event for event in tick["events"])


class TestTheBridgeVerbs:
    """The TUI reaches all of this over one JSON line at a time."""

    def _dispatch(self, **req):
        from hypernix.interfaces.hyped_pro_bridge import dispatch

        return dispatch({"id": 1, **req})

    def test_providers(self):
        resp = self._dispatch(cmd="noodle_providers")
        assert resp["ok"]
        assert "providers" in resp["data"]

    def test_sessions_when_there_are_none(self):
        resp = self._dispatch(cmd="noodle_sessions")
        assert resp["ok"]
        assert resp["data"]["sessions"] == []

    def test_an_unknown_session_is_an_error_code_not_a_traceback(self):
        resp = self._dispatch(cmd="noodle_poll", session="nope")
        assert not resp["ok"]
        assert resp["code"] == "HPB-NOODLE-002"

    def test_a_refused_start_is_an_error_code(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resp = self._dispatch(
            cmd="noodle_start", prompt="x",
            roster=["ollama:llama3.2"], root="/etc",
        )
        assert not resp["ok"]
        assert resp["code"] == "HPB-NOODLE-001"
        assert "outside" in resp["error"]

    def test_a_full_round_trip(self, fake_swarm, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_swarm.options = {"events": 3}
        started = self._dispatch(
            cmd="noodle_start", prompt="x", roster=["ollama:llama3.2"],
        )
        assert started["ok"]
        session_id = started["data"]["session_id"]
        seen = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            tick = self._dispatch(cmd="noodle_poll", session=session_id, timeout=0.2)
            assert tick["ok"]
            seen.extend(tick["data"]["events"])
            if not tick["data"]["running"]:
                break
        assert len([e for e in seen if e["kind"] == "turn"]) == 3
        assert self._dispatch(cmd="noodle_sessions", session=session_id)["ok"]

    def test_poll_is_backgrounded(self):
        """It long-polls, so inline it would block the stdin loop and the
        TUI could not deliver a cancel while a swarm was running."""
        from hypernix.interfaces.hyped_pro_bridge import BACKGROUND_COMMANDS

        assert "noodle_poll" in BACKGROUND_COMMANDS
        # start returns immediately, so backgrounding it would buy a
        # thread to start a thread.
        assert "noodle_start" not in BACKGROUND_COMMANDS


class TestTheTuiAndGuiAreWiredUp:
    """The parts that are not Python, checked as text.

    Cheap, and it catches the thing that was actually wrong: a command
    implemented in Python that nothing in the UI ever calls.
    """

    def _read(self, relative):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        return (root / relative).read_text()

    def test_the_tui_registers_the_command(self):
        source = self._read("src/hypernix/hyped_pro.ts")
        assert '{ name: "/noodle"' in source
        assert 'case "/noodle"' in source

    def test_the_compiled_artifact_is_in_step(self):
        """hyped_pro.js is what hyped_pro.py actually spawns. A TS change
        that was never built is a command that does not exist."""
        assert 'case "/noodle"' in self._read("src/hypernix/hyped_pro.js")

    def test_the_tui_calls_every_verb_it_needs(self):
        source = self._read("src/hypernix/hyped_pro.ts")
        for verb in ("noodle_providers", "noodle_start", "noodle_poll",
                     "noodle_stop", "noodle_sessions"):
            assert f"'{verb}'" in source, verb

    def test_the_gui_has_a_tab(self):
        source = self._read("src/hypernix/interfaces/hyped_pro_gui.py")
        assert '"Noodle"' in source
        assert "_build_noodle_tab" in source
