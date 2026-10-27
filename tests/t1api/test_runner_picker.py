"""The arrow-key chooser behind `runner start` and `runner backend`."""
from __future__ import annotations

import os
import sys

import pytest

from hypernix.t1api import picker, runner_cli
from hypernix.t1api.picker import Choice, Picker, decode_keys, fuzzy_score

ITEMS = [Choice("qwen3-8b"), Choice("qwen3.5-27b"), Choice("HyperNix.3-mini", current=True), Choice("llama-3.2-3b")]


class TestMatching:
    def test_letters_in_order_anywhere(self):
        assert fuzzy_score("q38", "qwen3-8b") is not None
        assert fuzzy_score("8q", "qwen3-8b") is None

    def test_a_substring_beats_a_scattered_match(self):
        assert fuzzy_score("mini", "hypernix.3-mini") > fuzzy_score("mni", "hypernix.3-mini")

    def test_empty_matches_everything(self):
        assert fuzzy_score("", "x") == 0


class TestThePicker:
    def test_arrows_move_and_wrap(self):
        p = Picker("t", list(ITEMS))
        p.key("down")
        assert p.selected().value == "qwen3.5-27b"
        p.key("up")
        p.key("up")
        assert p.selected().value == "llama-3.2-3b"

    def test_typing_filters_and_best_match_is_first(self):
        p = Picker("t", list(ITEMS))
        for ch in "llama":
            p.key(ch)
        assert [c.value for c in p.visible()] == ["llama-3.2-3b"]
        p.key("backspace")
        assert p.query == "llam"

    def test_j_and_k_move_only_when_not_filtering(self):
        p = Picker("t", list(ITEMS))
        p.key("j")
        assert p.selected().value == "qwen3.5-27b" and p.query == ""
        p.key("q")
        p.key("j")
        assert p.query == "qj"

    def test_enter_picks_esc_cancels(self):
        p = Picker("t", list(ITEMS))
        p.key("enter")
        assert p.done and not p.cancelled and p.selected().value == "qwen3-8b"
        q = Picker("t", list(ITEMS))
        q.key("escape")
        assert q.done and q.cancelled

    def test_enter_with_nothing_matching_does_not_pick(self):
        p = Picker("t", list(ITEMS), query="zzz")
        p.key("enter")
        assert not p.done and "nothing matches" in p.note

    def test_a_refresh_keeps_the_selection_on_the_same_model(self):
        p = Picker("t", list(ITEMS))
        p.key("down")  # qwen3.5-27b
        p.set_items([Choice("new-model"), *ITEMS])
        assert p.selected().value == "qwen3.5-27b"

    def test_render_marks_selection_and_what_is_serving(self):
        lines = Picker("Load", list(ITEMS), index=2).render(60, 10)
        assert lines[0].startswith("Load")
        assert any(line.startswith("❯ HyperNix.3-mini ●") for line in lines)
        assert all(len(line) <= 60 for line in lines)

    def test_a_long_list_scrolls_around_the_selection(self):
        many = [Choice(f"m{i}") for i in range(50)]
        p = Picker("t", many, index=40)
        lines = p.render(80, 5)
        assert any("❯ m40" in line for line in lines)
        assert any("50 in all" in line for line in lines)


def test_decode_keys():
    assert decode_keys("\x1b[A\x1b[Bab\r") == ["up", "down", "a", "b", "enter"]
    assert decode_keys("\x1b") == ["escape"]
    assert decode_keys("\x1b[1;5Cq") == ["q"]  # unknown sequences are skipped whole
    assert decode_keys("\x7f\x03") == ["backspace", "ctrl-c"]


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_choose_in_a_real_terminal():
    """Arrow down twice and Enter, through a pseudo-terminal, as a person would."""
    import pty
    import select
    import time

    pid, fd = pty.fork()
    if pid == 0:  # the child: run the chooser, report the choice on its exit code
        from hypernix.t1api.picker import choose

        picked = choose("Pick", lambda: ITEMS, refresh_seconds=0.2, fd=0, out=os.fdopen(1, "w"))
        os._exit(10 + [c.value for c in ITEMS].index(picked.value) if picked else 99)
    output = b""

    def pump(seconds):
        nonlocal output
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                try:
                    output += os.read(fd, 4096)
                except OSError:
                    return

    pump(1.5)
    os.write(fd, b"\x1b[B")
    pump(0.3)
    os.write(fd, b"\x1b[B")
    pump(0.6)  # a refresh happens in here
    os.write(fd, b"\r")
    pump(1.0)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 12, output[-400:]
    assert b"Pick: HyperNix.3-mini" in output


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_a_slow_list_does_not_freeze_the_keys():
    """The server took seconds to answer the model list, and the picker
    asked in its key loop -- so every refresh froze it, and opening it
    froze it before anything was drawn. Here the first answer takes a
    second and every refresh five: the list must be drawn at once, and
    the keys must be read while a refresh is still out."""
    import pty
    import select
    import time

    pid, fd = pty.fork()
    if pid == 0:
        from hypernix.t1api.picker import choose

        asked = []

        def slow():
            time.sleep(1.0 if not asked else 5.0)
            asked.append(1)
            return ITEMS

        picked = choose("Pick", slow, refresh_seconds=0.1, fd=0, out=os.fdopen(1, "w"))
        os._exit(10 + [c.value for c in ITEMS].index(picked.value) if picked else 99)
    output = b""

    def pump(seconds):
        nonlocal output
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                try:
                    output += os.read(fd, 4096)
                except OSError:
                    return

    pump(0.5)
    assert b"loading the list" in output  # drawn before the list arrived
    pump(1.2)  # the list is in, and the first refresh is out for five seconds
    os.write(fd, b"\x1b[B")
    pump(0.2)
    os.write(fd, b"\x1b[B")
    pump(0.2)
    os.write(fd, b"\r")
    sent = time.time()
    _, status = os.waitpid(pid, 0)
    assert time.time() - sent < 2.5, "the keys waited for the refresh"
    assert os.WEXITSTATUS(status) == 12, output[-400:]


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_a_list_that_cannot_be_fetched_is_raised():
    import pty
    import time

    pid, fd = pty.fork()
    if pid == 0:
        from hypernix.t1api.picker import choose

        def refused():
            raise RuntimeError("the server refused the model list (403)")

        try:
            choose("Pick", refused, fd=0, out=os.fdopen(1, "w"))
        except RuntimeError as exc:
            os._exit(42 if "403" in str(exc) else 1)
        os._exit(2)
    end = time.time() + 5
    while time.time() < end:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        try:
            os.read(fd, 4096)
        except OSError:
            pass
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("the picker kept waiting after its list was refused")
    assert os.WEXITSTATUS(status) == 42


@pytest.fixture
def calls(monkeypatch):
    recorded, replies = [], []

    def fake(method, url, key, payload=None):
        recorded.append((method, url, payload))
        return replies.pop(0) if replies else (200, {})

    monkeypatch.setattr(runner_cli, "_request", fake)
    fake.recorded, fake.replies = recorded, replies
    return fake


CATALOGUE = {"models": [{"model_id": "qwen3-8b", "path": "/m/a.gguf"},
                        {"model_id": "qwen3.5-27b", "path": "/m/b.gguf"},
                        {"model_id": "llama-3.2-3b", "path": "/m/c.gguf"}]}


class TestStartWithPartOfAName:
    def test_one_match_is_loaded(self, calls, monkeypatch, capsys):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: False)
        calls.replies.extend([
            (200, {"loaded": False}),
            (404, {"error": {"code": "NOT_FOUND", "message": "No model 'llama'"}}),
            (200, CATALOGUE),
            (200, {"loaded": True, "model": {"model_id": "llama-3.2-3b"}}),
        ])
        assert runner_cli.main(["start", "llama"]) == 0
        assert calls.recorded[-1][2]["model_id"] == "llama-3.2-3b"
        assert "llama → llama-3.2-3b" in capsys.readouterr().err

    def test_several_matches_without_a_terminal_are_listed(self, calls, monkeypatch, capsys):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: False)
        calls.replies.extend([(200, {"loaded": False}), (404, {}), (200, CATALOGUE)])
        assert runner_cli.main(["start", "qwen"]) == 1
        err = capsys.readouterr().err
        assert "2 models match 'qwen'" in err and "qwen3-8b" in err

    def test_several_matches_on_a_terminal_open_the_list_filtered(self, calls, monkeypatch):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: True)
        seen = {}

        def fake_choose(title, fetch, **kwargs):
            seen["query"] = kwargs.get("initial_query")
            return Choice("qwen3.5-27b")

        monkeypatch.setattr(picker, "choose", fake_choose)
        calls.replies.extend([(200, {"loaded": False}), (404, {}), (200, CATALOGUE), (200, CATALOGUE),
                              (200, {"loaded": True, "model": {"model_id": "qwen3.5-27b"}})])
        assert runner_cli.main(["start", "qwen"]) == 0
        assert seen["query"] == "qwen"
        assert calls.recorded[-1][2]["model_id"] == "qwen3.5-27b"

    def test_no_name_on_a_terminal_opens_the_list(self, calls, monkeypatch):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: True)
        monkeypatch.setattr(picker, "choose", lambda title, fetch, **kw: fetch()[1])
        calls.replies.extend([(200, {"loaded": False}), (200, CATALOGUE),
                              (200, {"loaded": True, "model": {"model_id": "qwen3.5-27b"}})])
        assert runner_cli.main(["start"]) == 0
        assert calls.recorded[-1][2]["model_id"] == "qwen3.5-27b"

    def test_cancelling_the_list_loads_nothing(self, calls, monkeypatch):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: True)
        monkeypatch.setattr(picker, "choose", lambda title, fetch, **kw: None)
        calls.replies.append((200, {"loaded": False}))
        assert runner_cli.main(["start"]) == 1
        assert not any(url.endswith("/runner/load") for _m, url, _p in calls.recorded)

    def test_no_pick_keeps_the_old_default(self, calls, monkeypatch):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: True)
        monkeypatch.setattr(picker, "choose", lambda *a, **k: pytest.fail("asked"))
        calls.replies.extend([(200, {"loaded": False}), (200, {"loaded": True, "model": {}})])
        assert runner_cli.main(["start", "--no-pick"]) == 0


class TestBackend:
    STATUS = {"loaded": True, "model": {"model_id": "m", "backend": "cuda", "context_length": 8192},
              "backends": ["auto", "cuda", "vulkan", "cpu"]}

    def test_named(self, calls):
        calls.replies.extend([(200, self.STATUS), (200, {"loaded": True, "model": {"model_id": "m"}})])
        assert runner_cli.main(["backend", "vulkan"]) == 0
        payload = calls.recorded[-1][2]
        assert payload["model_id"] == "m" and payload["backend"] == "vulkan" and payload["context_length"] == 8192

    def test_chosen_from_the_list(self, calls, monkeypatch):
        monkeypatch.setattr(runner_cli, "_can_ask", lambda: True)
        monkeypatch.setattr(picker, "choose", lambda title, fetch, **kw: [c for c in fetch() if c.value == "cpu"][0])
        calls.replies.extend([(200, self.STATUS), (200, self.STATUS), (200, {"loaded": True, "model": {}})])
        assert runner_cli.main(["backend"]) == 0
        assert calls.recorded[-1][2]["backend"] == "cpu"

    def test_the_same_backend_is_a_no_op(self, calls, capsys):
        calls.replies.append((200, self.STATUS))
        assert runner_cli.main(["backend", "cuda"]) == 0
        assert "already on cuda" in capsys.readouterr().out

    def test_unknown_or_nothing_serving(self, calls, capsys):
        calls.replies.append((200, self.STATUS))
        assert runner_cli.main(["backend", "metal"]) == 1
        calls.replies.append((200, {"loaded": False, "backends": ["cpu"]}))
        assert runner_cli.main(["backend", "cpu"]) == 1
        assert "Nothing is serving" in capsys.readouterr().err
