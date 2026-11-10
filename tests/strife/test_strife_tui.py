"""``hnx strife chat MODEL`` (0.73.0 alpha 1): the chat screen.

:class:`ChatApp` draws onto any screen, so most of this runs it against a
character grid; the last class runs the real program in a pseudo-terminal
and types and clicks at it.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from hypernix.strife import Engine, SamplerConfig, StrifeError
from hypernix.strife.fixtures import make_model
from hypernix.strife.models import resolve_model
from hypernix.strife.tui import ChatApp, ChatStore, InputBox, split_thinking

GREEDY = SamplerConfig(temperature=0, repeat_penalty=1.0)


class Grid:
    """A fake curses screen: a grid of characters and the style of each row."""

    def __init__(self, height: int = 30, width: int = 100) -> None:
        self.h, self.w = height, width
        self.rows = [[" "] * width for _ in range(height)]
        self.styles: dict[tuple[int, int], str] = {}
        self.cursor = (0, 0)

    def getmaxyx(self):
        return self.h, self.w

    def addstr(self, y, x, text, style):
        for i, ch in enumerate(text):
            if 0 <= x + i < self.w:
                self.rows[y][x + i] = ch
                self.styles[(y, x + i)] = style

    def place_cursor(self, y, x):
        self.cursor = (y, x)

    def line(self, y: int) -> str:
        return "".join(self.rows[y])

    def find(self, text: str) -> tuple[int, int]:
        for y in range(self.h):
            x = self.line(y).find(text)
            if x >= 0:
                return y, x
        raise AssertionError(f"{text!r} not on screen:\n" + "\n".join(self.line(y) for y in range(self.h)))

    def text(self) -> str:
        return "\n".join(self.line(y) for y in range(self.h))


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    model = make_model(tmp_path_factory.mktemp("tui") / "tiny-qwen35.gguf", arch="qwen35", seed=9)
    with Engine(model["path"], n_ctx=512) as e:
        yield e


@pytest.fixture
def app(engine, tmp_path):
    return ChatApp(engine, ChatStore(tmp_path / "chats.json"), max_tokens=6, sampling=GREEDY)


def _type(app: ChatApp, text: str) -> None:
    for ch in text:
        app.key(ch)


def _finish(app: ChatApp, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while (app.busy or not app._events.empty()) and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    app.pump()


class TestTheTextArea:
    def test_editing(self):
        box = InputBox()
        box.insert("helo")
        box.move(d_col=-1)
        box.insert("l")
        assert box.text == "hello"
        box.end()
        box.newline()
        box.insert("world")
        assert box.text == "hello\nworld" and (box.row, box.col) == (1, 5)
        box.home()
        box.backspace()
        assert box.text == "helloworld" and (box.row, box.col) == (0, 5)
        box.delete()
        assert box.text == "helloorld"
        box.clear()
        assert box.text == "" and (box.row, box.col) == (0, 0)

    def test_cursor_stays_inside(self):
        box = InputBox()
        box.insert("ab\nc")
        box.move(d_row=-5)
        box.end()
        assert (box.row, box.col) == (0, 2)
        box.move(d_col=1)  # past the end of a line is the start of the next
        assert (box.row, box.col) == (1, 0)
        box.place(9, 9)
        assert (box.row, box.col) == (1, 1)


class TestThinking:
    @pytest.mark.parametrize("text,opened,expected", [
        ("Paris.", False, ("", "Paris.")),
        ("<think>hmm</think>Paris.", False, ("hmm", "Paris.")),
        ("hmm</think>Paris.", True, ("hmm", "Paris.")),
        ("still thinking", True, ("still thinking", "")),
    ])
    def test_split(self, text, opened, expected):
        assert split_thinking(text, opened) == expected


class TestTheScreen:
    def test_layout_has_chats_context_text_area_and_buttons(self, app):
        grid = Grid()
        app.render(grid)
        assert "Strife" in grid.line(0) and "ctx 0/512" in grid.line(0) and "%" in grid.line(0)
        grid.find("+ New chat")
        grid.find("[ Delete ]")
        grid.find("[ Send ]")
        grid.find("type a message")
        assert "Enter sends" in grid.line(29)

    def test_a_turn_streams_in_and_is_saved(self, app, tmp_path):
        _type(app, "hello world")
        app.key("enter")
        assert app.busy or not app._events.empty()
        grid = Grid()
        app.render(grid)
        grid.find("[ Stop ]")
        _finish(app)
        assert [m["role"] for m in app.chat.messages] == ["user", "assistant"]
        assert app.chat.title == "hello world"
        saved = json.loads((tmp_path / "chats.json").read_text())
        assert saved["chats"][0]["messages"][0]["content"] == "hello world"
        grid = Grid()
        app.render(grid)
        grid.find("You")
        grid.find("hello world")
        assert "ctx 0/" not in grid.line(0)  # the meter moved
        assert "tokens" in grid.line(29)

    def test_the_reply_is_the_engines(self, app, engine):
        _type(app, "the model")
        app.key("enter")
        _finish(app)
        expected = engine.generate(messages=[{"role": "user", "content": "the model"}],
                                   max_tokens=6, sampling=GREEDY)
        assert app.chat.messages[-1]["content"] == split_thinking(expected.text, False)[1]

    def test_esc_stops_a_reply(self, app):
        app.max_tokens = 400
        _type(app, "hello")
        app.key("enter")
        app.key("escape")
        _finish(app)
        assert app.chat.messages[-1].get("stopped") or len(app.chat.messages) == 2

    def test_new_switch_and_delete_chats(self, app):
        _type(app, "first chat")
        app.key("enter")
        _finish(app)
        first = app.chat
        app.key("ctrl-n")
        assert app.chat is not first and not app.chat.messages
        _type(app, "second chat")
        app.key("enter")
        _finish(app)
        assert [c.title for c in app.store.chats][:2] == ["second chat", "first chat"]
        app.open_chat(1)
        assert app.chat is first
        app.key("ctrl-d")
        assert app.chat in app.store.chats  # asks first
        app.key("ctrl-d")
        assert first not in app.store.chats

    def test_chats_survive_a_restart(self, app, engine, tmp_path):
        _type(app, "remember me")
        app.key("enter")
        _finish(app)
        again = ChatApp(engine, ChatStore(tmp_path / "chats.json"), max_tokens=6, sampling=GREEDY)
        assert again.chat.title == "remember me" and len(again.chat.messages) == 2

    def test_alt_enter_adds_a_line_instead_of_sending(self, app):
        _type(app, "line one")
        app.key("newline")
        _type(app, "line two")
        assert app.input.text == "line one\nline two" and not app.busy

    def test_a_long_conversation_drops_old_turns_to_fit(self, engine, tmp_path):
        app = ChatApp(engine, ChatStore(tmp_path / "c.json"), max_tokens=6, sampling=GREEDY)
        for i in range(40):
            app.chat.messages += [{"role": "user", "content": f"question number {i} " * 3},
                                  {"role": "assistant", "content": f"answer number {i} " * 3}]
        messages, tokens, dropped = app._prompt_messages(app.chat)
        assert dropped > 0 and tokens <= engine.n_ctx
        app._measure()
        grid = Grid()
        app.render(grid)
        grid.find("no longer fit the context")


class TestTheMouse:
    def test_clicks(self, app):
        _type(app, "a chat")
        app.key("enter")
        _finish(app)
        grid = Grid()
        app.render(grid)
        y, x = grid.find("+ New chat")
        app.click(y, x + 2)
        assert not app.chat.messages
        grid = Grid()
        app.render(grid)
        y, x = grid.find("a chat")
        app.click(y, x)
        assert app.chat.title == "a chat"
        y, x = grid.find("[ Delete ]")
        app.click(y, x + 3)
        assert app.confirm_delete

    def test_send_button_and_cursor_placement(self, app):
        _type(app, "hello there")
        grid = Grid()
        app.render(grid)
        y, x = grid.find("> hello there")
        app.click(y, x + 4)
        assert app.input.col == 2
        y, x = grid.find("[ Send ]")
        app.click(y, x + 2)
        assert app.busy or not app._events.empty()
        _finish(app)
        assert app.chat.messages[0]["content"] == "hello there"

    def test_the_wheel_scrolls(self, app):
        for i in range(30):
            app.chat.messages += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        grid = Grid()
        app.render(grid)
        app.wheel(10, 60, up=True)
        app.wheel(10, 60, up=True)
        assert app.scroll == 6
        grid = Grid()
        app.render(grid)
        grid.find("more")
        app.wheel(10, 60, up=False)
        assert app.scroll == 3


class TestModelIds:
    def test_a_path_an_id_and_a_part(self, tmp_path, monkeypatch):
        folder = tmp_path / "models"
        folder.mkdir()
        make_model(folder / "Tiny-LFM3-Q8_0.gguf", arch="lfm3")
        monkeypatch.setenv("T1_HF_DOWNLOAD_DIR", str(folder))
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = folder / "Tiny-LFM3-Q8_0.gguf"
        assert resolve_model(str(expected)) == expected
        assert resolve_model("tiny-lfm3-q8-0") == expected
        assert resolve_model("lfm3") == expected
        with pytest.raises(StrifeError) as info:
            resolve_model("LiquidAI/LFM3-1.7B-GGUF")
        assert info.value.code == "strife.model.not_found" and "hnx download" in info.value.remedy


@pytest.mark.skipif(os.name != "posix", reason="needs a pseudo-terminal")
class TestInARealTerminal:
    def test_type_click_and_quit(self, tmp_path):
        import pty
        import select
        import subprocess
        import sys

        model = make_model(tmp_path / "tiny.gguf", arch="lfm3", seed=2)
        chats = tmp_path / "chats.json"
        master, slave = pty.openpty()
        import fcntl
        import struct
        import termios

        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        env = dict(os.environ, TERM="xterm-256color", ESCDELAY="25")
        proc = subprocess.Popen([sys.executable, "-m", "hypernix.strife.cli", "chat", model["path"],
                                 "--chats", str(chats), "-n", "4", "--temp", "0", "--ctx", "256"],
                                stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
        os.close(slave)
        out = b""

        def read_for(seconds: float) -> None:
            nonlocal out
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        out += os.read(master, 65536)
                    except OSError:
                        return

        read_for(8)
        assert b"New chat" in out and b"Send" in out
        os.write(master, b"hello strife\r")
        read_for(8)
        # A mouse click (SGR 1006 and X10 forms) on "+ New chat" at row 2, col 3.
        os.write(master, b"\x1b[<0;3;2M\x1b[<0;3;2m")
        read_for(1)
        os.write(master, b"\x11")  # Ctrl+Q
        read_for(3)
        proc.wait(timeout=20)
        os.close(master)
        assert proc.returncode == 0, out[-2000:]
        saved = json.loads(chats.read_text())
        assert saved["chats"][0]["messages"][0]["content"] == "hello strife"
        assert saved["chats"][0]["messages"][1]["role"] == "assistant"
