"""``hnx strife chat MODEL`` — a terminal chat on Strife, with the mouse.

::

    ┌ Chats ─────────┬ Strife · LFM3 1.7B (lfm3)        ctx 1.2k/4.1k ▕███░░░░░▏ 29% ┐
    │ + New chat     │ You                                                         │
    │                │   What is the capital of France?                            │
    │ ▶ Capitals     │                                                             │
    │   Poems        │ LFM3 1.7B                                                   │
    │                │   thinking: The user asks for the capital…                  │
    │                │   The capital of France is Paris.                           │
    │                ├──────────────────────────────────────────────── [ Send ] ───┤
    │ [ Delete ]     │ > type here; Enter sends, Alt+Enter or Ctrl+O adds a line   │
    └────────────────┴ Enter send · Esc stop · ^N new · ^D delete · ^Q quit ───────┘

* **Chats** on the left: click one (or Tab to the list, arrows, Enter) to
  switch; ``+ New chat`` or Ctrl+N starts one; ``Delete`` or Ctrl+D removes
  the open one (asks first). Chats are saved in
  ``~/.hypernix/strife/chats.json`` and come back next time.
* **Context** in the header: how much of the context the open chat's
  prompt takes, as tokens and a bar. Older turns are left out of the
  prompt when it would not fit, and the status line says so.
* **The text area** at the bottom: multi-line editing with the arrow keys,
  Home/End, Backspace/Delete, Ctrl+U to clear; click to put the cursor.
* **The mouse**: clicks on chats and buttons, the wheel scrolls the
  conversation (or the chat list under the pointer). PgUp/PgDn scroll too.
* A model that thinks first (``<think>``) shows its thinking dimmed; only
  the answer is kept in the conversation it is sent next time.

The reply streams while the screen stays live: generation runs on a
thread, and Esc (or ``Stop``) cancels it at the next token.

Everything but the curses loop is plain Python (:class:`ChatApp` draws
onto any object with ``addstr``/``getmaxyx``), which is what the tests
drive.
"""
from __future__ import annotations

import json
import os
import queue
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Chat", "ChatApp", "ChatStore", "InputBox", "run"]

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


# ---------------------------------------------------------------------------
# Chats on disk
# ---------------------------------------------------------------------------

@dataclass
class Chat:
    chat_id: str
    title: str = "New chat"
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"chat_id": self.chat_id, "title": self.title, "model": self.model,
                "messages": self.messages, "created": self.created, "updated": self.updated}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chat:
        messages = [m for m in data.get("messages") or [] if isinstance(m, dict)
                    and isinstance(m.get("content"), str)]
        return cls(str(data.get("chat_id") or uuid.uuid4().hex), str(data.get("title") or "Chat"),
                   str(data.get("model") or ""), messages, float(data.get("created") or time.time()),
                   float(data.get("updated") or time.time()))


class ChatStore:
    """Every chat, newest first, in one JSON file written atomically."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.path.expanduser("~")) / ".hypernix" / "strife" / "chats.json"
        self.chats: list[Chat] = []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.chats = [Chat.from_dict(c) for c in data.get("chats", []) if isinstance(c, dict)]
        except (OSError, ValueError, AttributeError):
            self.chats = []
        self.chats.sort(key=lambda c: c.updated, reverse=True)

    def new(self, model: str) -> Chat:
        chat = Chat(uuid.uuid4().hex, model=model)
        self.chats.insert(0, chat)
        return chat

    def delete(self, chat: Chat) -> None:
        self.chats = [c for c in self.chats if c.chat_id != chat.chat_id]
        self.save()

    def touch(self, chat: Chat) -> None:
        chat.updated = time.time()
        self.chats.sort(key=lambda c: c.updated, reverse=True)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        keep = [c.to_dict() for c in self.chats if c.messages]
        tmp.write_text(json.dumps({"chats": keep}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# The text area
# ---------------------------------------------------------------------------

class InputBox:
    """A multi-line text buffer with a cursor."""

    def __init__(self) -> None:
        self.lines: list[str] = [""]
        self.row = self.col = 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def clear(self) -> None:
        self.lines, self.row, self.col = [""], 0, 0

    def insert(self, text: str) -> None:
        for ch in text:
            if ch == "\n":
                self.newline()
            elif ch.isprintable():
                line = self.lines[self.row]
                self.lines[self.row] = line[: self.col] + ch + line[self.col:]
                self.col += 1

    def newline(self) -> None:
        line = self.lines[self.row]
        self.lines[self.row: self.row + 1] = [line[: self.col], line[self.col:]]
        self.row += 1
        self.col = 0

    def backspace(self) -> None:
        if self.col:
            line = self.lines[self.row]
            self.lines[self.row] = line[: self.col - 1] + line[self.col:]
            self.col -= 1
        elif self.row:
            self.col = len(self.lines[self.row - 1])
            self.lines[self.row - 1] += self.lines.pop(self.row)
            self.row -= 1

    def delete(self) -> None:
        line = self.lines[self.row]
        if self.col < len(line):
            self.lines[self.row] = line[: self.col] + line[self.col + 1:]
        elif self.row + 1 < len(self.lines):
            self.lines[self.row] += self.lines.pop(self.row + 1)

    def move(self, d_row: int = 0, d_col: int = 0) -> None:
        if d_col:
            self.col += d_col
            if self.col < 0 and self.row:
                self.row -= 1
                self.col = len(self.lines[self.row])
            elif self.col > len(self.lines[self.row]) and self.row + 1 < len(self.lines):
                self.row += 1
                self.col = 0
        if d_row:
            self.row = max(0, min(len(self.lines) - 1, self.row + d_row))
        self.col = max(0, min(len(self.lines[self.row]), self.col))

    def home(self) -> None:
        self.col = 0

    def end(self) -> None:
        self.col = len(self.lines[self.row])

    def place(self, row: int, col: int) -> None:
        self.row = max(0, min(len(self.lines) - 1, row))
        self.col = max(0, min(len(self.lines[self.row]), col))


# ---------------------------------------------------------------------------
# The app (no curses in here)
# ---------------------------------------------------------------------------

def split_thinking(text: str, opened: bool) -> tuple[str, str]:
    """``(thinking, answer)`` for a reply that may open with a think block."""
    if opened and THINK_CLOSE not in text:
        return text, ""
    if opened:
        thought, _, answer = text.partition(THINK_CLOSE)
        return thought.strip(), answer.lstrip()
    if text.lstrip().startswith(THINK_OPEN):
        body = text.lstrip()[len(THINK_OPEN):]
        if THINK_CLOSE not in body:
            return body, ""
        thought, _, answer = body.partition(THINK_CLOSE)
        return thought.strip(), answer.lstrip()
    return "", text


def _short(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


@dataclass
class Region:
    y0: int
    x0: int
    y1: int
    x1: int
    action: str
    arg: Any = None

    def hit(self, y: int, x: int) -> bool:
        return self.y0 <= y <= self.y1 and self.x0 <= x <= self.x1


#: Attribute names a screen maps to curses attributes (see :func:`run`).
STYLES = ("normal", "header", "selected", "user", "assistant", "thinking", "status", "button",
          "meter_ok", "meter_warn", "meter_full", "border", "error")


class ChatApp:
    def __init__(self, engine: Any, store: ChatStore, *, model_name: str = "",
                 max_tokens: int = 1024, sampling: Any = None) -> None:
        self.engine = engine
        self.store = store
        self.model_name = model_name or engine.report.name or engine.path.stem
        self.model_key = engine.path.stem
        self.max_tokens = max_tokens
        self.sampling = sampling
        self.input = InputBox()
        self.focus = "input"            # "input" | "chats"
        self.cursor_chat = 0            # highlighted row in the chat list
        self.scroll = 0                 # transcript lines up from the bottom
        self.chat_scroll = 0
        self.status = "Enter sends · Alt+Enter or ^O new line · Esc stops · ^N new · ^D delete · ^Q quit"
        self.error = ""
        self.confirm_delete = False
        self.running = True
        self.regions: list[Region] = []
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel: threading.Event | None = None
        self.streaming = ""             # the reply as it arrives
        self.opened_thinking = False
        self.context_used = 0
        self.dropped_turns = 0
        self._sessions: dict[str, str] = {}
        self.speed = ""
        self.chat = store.chats[0] if store.chats else store.new(self.model_key)
        self._measure()

    # -- state ---------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _prompt_messages(self, chat: Chat, extra: list[dict[str, Any]] | None = None
                         ) -> tuple[list[dict[str, Any]], int, int]:
        """The messages to send, oldest turns dropped until they fit with
        room for a reply; ``(messages, tokens, dropped)``."""
        messages = [dict(role=m["role"], content=m["content"]) for m in chat.messages] + (extra or [])
        system = [m for m in messages if m["role"] == "system"][:1]
        rest = [m for m in messages if m["role"] != "system"]
        budget = max(16, self.engine.n_ctx - min(self.max_tokens, self.engine.n_ctx // 2))
        dropped = 0
        while True:
            convo = system + rest
            if not convo:
                return [], 0, dropped
            tokens = len(self.engine.prompt_tokens(messages=convo))
            if tokens <= budget or len(rest) <= 1:
                return convo, tokens, dropped
            rest = rest[1:]
            dropped += 1

    def _measure(self) -> None:
        if not self.chat.messages:
            self.context_used, self.dropped_turns = 0, 0
            return
        try:
            _, self.context_used, self.dropped_turns = self._prompt_messages(self.chat)
        except Exception as exc:  # noqa: BLE001 - a template problem must not take the screen down
            self.error = str(exc)

    def open_chat(self, index: int) -> None:
        if self.busy or not 0 <= index < len(self.store.chats):
            return
        self.chat = self.store.chats[index]
        self.cursor_chat = index
        self.scroll = 0
        self.confirm_delete = False
        self._measure()

    def new_chat(self) -> None:
        if self.busy:
            return
        if self.chat.messages or self.chat not in self.store.chats:
            self.chat = self.store.new(self.model_key)
        self.cursor_chat = self.store.chats.index(self.chat)
        self.scroll = 0
        self.error = ""
        self._measure()

    def delete_chat(self) -> None:
        if self.busy:
            return
        if not self.confirm_delete:
            self.confirm_delete = True
            self.error = f"Delete '{self.chat.title}'? Press ^D (or click Delete) again to confirm."
            return
        self.confirm_delete = False
        self.error = ""
        self._sessions.pop(self.chat.chat_id, None)
        self.store.delete(self.chat)
        if self.store.chats:
            self.open_chat(0)
        else:
            self.chat = self.store.new(self.model_key)
            self.cursor_chat = 0
            self._measure()

    def send(self) -> None:
        text = self.input.text.strip()
        if self.busy or not text:
            return
        self.error = ""
        self.confirm_delete = False
        if not self.chat.messages:
            self.chat.title = text.splitlines()[0][:40]
        try:
            messages, tokens, dropped = self._prompt_messages(
                self.chat, [{"role": "user", "content": text}])
            rendered = self.engine.tokenizer.apply_chat_template(messages)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Could not build the prompt: {exc}"
            return
        self.chat.messages.append({"role": "user", "content": text})
        self.input.clear()
        self.scroll = 0
        self.context_used, self.dropped_turns = tokens, dropped
        self.opened_thinking = rendered.rstrip().endswith(THINK_OPEN)
        self.streaming = ""
        self.speed = ""
        session = self._sessions.get(self.chat.chat_id)
        if session is None:
            session = self._sessions[self.chat.chat_id] = self.engine.new_session()
        self._cancel = threading.Event()
        cancel = self._cancel

        def work() -> None:
            try:
                for event in self.engine.stream(messages=messages, session=session,
                                                max_tokens=self.max_tokens,
                                                sampling=self.sampling, cancel=cancel):
                    self._events.put(event)
            except Exception as exc:  # noqa: BLE001 - reported on screen
                self._events.put({"type": "error", "message": str(exc)})

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    def pump(self) -> bool:
        """Take what the generation thread produced; True when anything changed."""
        changed = False
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return changed
            changed = True
            if event["type"] == "token":
                self.streaming += event["text"]
            elif event["type"] == "error":
                self.error = event["message"]
                self.streaming = ""
            elif event["type"] == "done":
                thought, answer = split_thinking(event["text"], self.opened_thinking)
                reply: dict[str, Any] = {"role": "assistant", "content": answer if (thought or answer) else ""}
                if thought:
                    reply["thinking"] = thought
                if event["stop_reason"] in ("cancelled",):
                    reply["stopped"] = True
                self.chat.messages.append(reply)
                self.streaming = ""
                stats = event["stats"]
                tps = stats.get("decode_tokens_per_second")
                self.speed = (f"{stats.get('generated_tokens', 0)} tokens"
                              + (f" · {tps:.1f} tok/s" if tps else "")
                              + f" · {event['stop_reason'].replace('_', ' ')}")
                self.store.touch(self.chat)
                self.cursor_chat = self.store.chats.index(self.chat)
                self._measure()

    # -- input -------------------------------------------------------------

    def key(self, key: Any) -> None:
        """One key: a ``str`` character, or a name (``"enter"``, ``"up"``...)."""
        if key in ("ctrl-q",):
            self.stop()
            self.running = False
            return
        if key == "ctrl-c":
            if self.busy:
                self.stop()
            else:
                self.running = False
            return
        if key == "escape":
            self.stop() if self.busy else setattr(self, "confirm_delete", False)
            return
        if key == "ctrl-n":
            return self.new_chat()
        if key == "ctrl-d":
            return self.delete_chat()
        if key == "tab":
            self.focus = "chats" if self.focus == "input" else "input"
            return
        if key in ("pageup", "pagedown"):
            self.scroll = max(0, self.scroll + (10 if key == "pageup" else -10))
            return
        if self.focus == "chats":
            if key == "up":
                self.cursor_chat = max(0, self.cursor_chat - 1)
            elif key == "down":
                self.cursor_chat = min(len(self.store.chats) - 1, self.cursor_chat + 1)
            elif key == "enter":
                self.open_chat(self.cursor_chat)
                self.focus = "input"
            return
        box = self.input
        actions = {"enter": self.send, "newline": box.newline, "backspace": box.backspace,
                   "delete": box.delete, "home": box.home, "end": box.end, "ctrl-u": box.clear,
                   "left": lambda: box.move(d_col=-1), "right": lambda: box.move(d_col=1),
                   "up": lambda: box.move(d_row=-1), "down": lambda: box.move(d_row=1)}
        if key in actions:
            actions[key]()
        elif isinstance(key, str) and len(key) == 1:
            box.insert(key)

    def click(self, y: int, x: int) -> None:
        for region in self.regions:
            if region.action == "chats-area":
                continue  # only for the wheel; the chats inside it take clicks
            if region.hit(y, x):
                act = region.action
                if act == "chat":
                    self.open_chat(region.arg)
                    self.focus = "input"
                elif act == "new":
                    self.new_chat()
                elif act == "delete":
                    self.delete_chat()
                elif act == "send":
                    self.stop() if self.busy else self.send()
                elif act == "input":
                    self.focus = "input"
                    row, col = region.arg(y, x)
                    self.input.place(row, col)
                return

    def wheel(self, y: int, x: int, up: bool) -> None:
        for region in self.regions:
            if region.hit(y, x) and region.action == "chats-area":
                self.chat_scroll = max(0, self.chat_scroll + (-1 if up else 1))
                return
        self.scroll = max(0, self.scroll + (3 if up else -3))

    # -- drawing -----------------------------------------------------------

    def _transcript(self, width: int) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        wrap = max(10, width - 3)

        def para(text: str, style: str, prefix: str = "  ") -> None:
            for raw in (text or "").split("\n"):
                pieces = textwrap.wrap(raw, wrap, replace_whitespace=False, drop_whitespace=False) or [""]
                for piece in pieces:
                    lines.append((prefix + piece, style))

        shown = list(self.chat.messages)
        if self.dropped_turns:
            lines.append((f"  ({self.dropped_turns} older message(s) no longer fit the context "
                          "and are not sent)", "thinking"))
        for message in shown:
            role = message.get("role")
            if role == "user":
                lines.append(("You", "user"))
                para(message["content"], "normal")
            elif role == "assistant":
                lines.append((self.model_name, "assistant"))
                if message.get("thinking"):
                    para("thinking: " + message["thinking"], "thinking")
                para(message["content"] + (" [stopped]" if message.get("stopped") else ""), "normal")
            lines.append(("", "normal"))
        if self.busy or self.streaming:
            lines.append((self.model_name, "assistant"))
            thought, answer = split_thinking(self.streaming, self.opened_thinking)
            if thought or (self.opened_thinking and not answer):
                para("thinking: " + (thought or "…"), "thinking")
            if answer:
                para(answer, "normal")
            elif not thought and not self.opened_thinking:
                para("…", "thinking")
        if not shown and not self.streaming and not self.busy:
            para(f"Ask {self.model_name} something. Running on Strife, on this machine.",
                 "thinking")
        return lines

    def render(self, screen: Any) -> None:
        height, width = screen.getmaxyx()
        self.regions = []
        put = _Painter(screen, height, width)
        if height < 8 or width < 40:
            put.text(0, 0, "Make the terminal bigger (40x8 at least).", "error")
            return
        side = min(28, width // 3) if width >= 70 else 0
        main_x = side + 1 if side else 0
        main_w = width - main_x

        # header with the context meter
        title = f" Strife · {self.model_name} ({self.engine.report.architecture}) "
        used = self.context_used + (len(self.streaming) // 4 if self.streaming else 0)
        frac = min(1.0, used / max(1, self.engine.n_ctx))
        bar_w = 12
        filled = int(round(frac * bar_w))
        meter = f"ctx {_short(used)}/{_short(self.engine.n_ctx)} "
        put.text(0, 0, " " * width, "header")
        put.text(0, 0, title, "header")
        mx = max(len(title) + 1, width - len(meter) - bar_w - 7)
        put.text(0, mx, meter, "header")
        style = "meter_ok" if frac < 0.7 else "meter_warn" if frac < 0.9 else "meter_full"
        put.text(0, mx + len(meter), "▕" + "█" * filled + "░" * (bar_w - filled) + "▏", style)
        put.text(0, mx + len(meter) + bar_w + 2, f" {int(frac * 100):>3}%", "header")

        # input area size
        input_rows = max(1, min(6, len(self.input.lines)))
        status_y = height - 1
        input_y0 = status_y - input_rows
        sep_y = input_y0 - 1
        body_y0, body_y1 = 1, sep_y - 1

        # sidebar
        if side:
            for y in range(1, status_y):
                put.text(y, side, "│", "border")
            put.text(1, 1, "+ New chat", "button")
            self.regions.append(Region(1, 0, 1, side - 1, "new"))
            list_y0, list_y1 = 3, status_y - 3
            self.regions.append(Region(list_y0, 0, list_y1, side - 1, "chats-area"))
            rows = list_y1 - list_y0 + 1
            self.chat_scroll = max(0, min(self.chat_scroll, max(0, len(self.store.chats) - rows)))
            if self.cursor_chat < self.chat_scroll:
                self.chat_scroll = self.cursor_chat
            elif self.cursor_chat >= self.chat_scroll + rows:
                self.chat_scroll = self.cursor_chat - rows + 1
            for i, chat in enumerate(self.store.chats[self.chat_scroll:self.chat_scroll + rows]):
                index = i + self.chat_scroll
                y = list_y0 + i
                is_open = chat is self.chat
                mark = "▶ " if is_open else "  "
                label = (mark + (chat.title or "New chat"))[: side - 1].ljust(side - 1)
                hl = index == self.cursor_chat and self.focus == "chats"
                put.text(y, 0, label, "selected" if (is_open or hl) else "normal")
                self.regions.append(Region(y, 0, y, side - 1, "chat", index))
            put.text(status_y - 2, 1, "[ Delete ]", "error" if self.confirm_delete else "button")
            self.regions.append(Region(status_y - 2, 0, status_y - 2, side - 1, "delete"))

        # transcript
        lines = self._transcript(main_w)
        visible = body_y1 - body_y0 + 1
        max_scroll = max(0, len(lines) - visible)
        self.scroll = min(self.scroll, max_scroll)
        start = max(0, len(lines) - visible - self.scroll)
        for i, (text, style_name) in enumerate(lines[start:start + visible]):
            put.text(body_y0 + i, main_x, text[:main_w], style_name)
        if self.scroll:
            put.text(body_y0, width - 12, f" ↑ {self.scroll} more ", "status")

        # separator with the Send / Stop button
        put.text(sep_y, main_x, "─" * main_w, "border")
        label = "[ Stop ]" if self.busy else "[ Send ]"
        bx = width - len(label) - 2
        put.text(sep_y, bx, label, "button")
        self.regions.append(Region(sep_y, bx, sep_y, bx + len(label) - 1, "send"))

        # the text area
        box = self.input
        first = max(0, box.row - input_rows + 1)
        for i in range(input_rows):
            row = first + i
            text = box.lines[row] if row < len(box.lines) else ""
            prefix = "> " if row == 0 else "  "
            shift = max(0, box.col - (main_w - 4)) if row == box.row else 0
            put.text(input_y0 + i, main_x, (prefix + text[shift:])[:main_w], "normal")
        if not box.text and not self.busy:
            put.text(input_y0, main_x + 2, "type a message", "thinking")

        def to_cursor(y: int, x: int, first: int = first) -> tuple[int, int]:
            return first + (y - input_y0), max(0, x - main_x - 2)

        self.regions.append(Region(input_y0, main_x, status_y - 1, width - 1, "input", to_cursor))

        # status line
        if self.error:
            put.text(status_y, 0, (" " + self.error)[:width].ljust(width), "error")
        else:
            note = (f"{self.dropped_turns} older message(s) no longer fit the context and are "
                    "not sent · ") if self.dropped_turns else ""
            info = (" generating… " + (self.speed or "") + "  Esc stops") if self.busy else (
                " " + note + (self.speed + " · " if self.speed else "") + self.status)
            put.text(status_y, 0, info[:width].ljust(width), "status")

        cursor_y = input_y0 + (box.row - first)
        cursor_x = main_x + 2 + min(box.col, main_w - 4)
        put.cursor(cursor_y if self.focus == "input" else -1, cursor_x)


class _Painter:
    """Bounds-checked writes, so a narrow terminal clips instead of raising."""

    def __init__(self, screen: Any, height: int, width: int) -> None:
        self.screen, self.h, self.w = screen, height, width

    def text(self, y: int, x: int, text: str, style: str) -> None:
        if not 0 <= y < self.h or x >= self.w or x < 0:
            return
        text = text[: self.w - x - (1 if y == self.h - 1 else 0)]
        if text:
            try:
                self.screen.addstr(y, x, text, style)
            except Exception:  # noqa: BLE001 - curses raises at the last cell
                pass

    def cursor(self, y: int, x: int) -> None:
        mover = getattr(self.screen, "place_cursor", None)
        if mover is not None:
            mover(y, x)


# ---------------------------------------------------------------------------
# curses
# ---------------------------------------------------------------------------

def _curses_keys(curses: Any) -> dict[int, str]:
    table = {curses.KEY_UP: "up", curses.KEY_DOWN: "down", curses.KEY_LEFT: "left",
             curses.KEY_RIGHT: "right", curses.KEY_HOME: "home", curses.KEY_END: "end",
             curses.KEY_BACKSPACE: "backspace", curses.KEY_DC: "delete", curses.KEY_PPAGE: "pageup",
             curses.KEY_NPAGE: "pagedown", curses.KEY_ENTER: "enter", 10: "enter", 13: "enter",
             9: "tab", 127: "backspace", 8: "backspace", 14: "ctrl-n", 4: "ctrl-d", 17: "ctrl-q",
             21: "ctrl-u", 15: "newline", 3: "ctrl-c", 12: "redraw"}
    return table


class _CursesScreen:
    def __init__(self, curses: Any, stdscr: Any, styles: dict[str, int]) -> None:
        self.curses, self.stdscr, self.styles = curses, stdscr, styles

    def getmaxyx(self) -> tuple[int, int]:
        return self.stdscr.getmaxyx()

    def addstr(self, y: int, x: int, text: str, style: str) -> None:
        self.stdscr.addstr(y, x, text, self.styles.get(style, 0))

    def place_cursor(self, y: int, x: int) -> None:
        try:
            if y < 0:
                self.curses.curs_set(0)
            else:
                self.curses.curs_set(1)
                self.stdscr.move(y, x)
        except Exception:  # noqa: BLE001 - some terminals cannot show a cursor
            pass


def _styles(curses: Any) -> dict[str, int]:
    out = {name: 0 for name in STYLES}
    out.update(header=curses.A_REVERSE, selected=curses.A_REVERSE, user=curses.A_BOLD,
               assistant=curses.A_BOLD, thinking=curses.A_DIM, status=curses.A_DIM,
               button=curses.A_BOLD, border=curses.A_DIM, error=curses.A_BOLD)
    if curses.has_colors():
        try:
            curses.start_color()
            curses.use_default_colors()
            pairs = {"user": curses.COLOR_CYAN, "assistant": curses.COLOR_GREEN,
                     "meter_ok": curses.COLOR_GREEN, "meter_warn": curses.COLOR_YELLOW,
                     "meter_full": curses.COLOR_RED, "button": curses.COLOR_BLUE,
                     "error": curses.COLOR_RED}
            for i, (name, color) in enumerate(pairs.items(), start=1):
                curses.init_pair(i, color, -1)
                out[name] = curses.color_pair(i) | (curses.A_BOLD if name in ("user", "assistant", "button") else 0)
        except Exception:  # noqa: BLE001 - monochrome terminals keep the plain styles
            pass
    return out


def run(app: ChatApp) -> int:
    try:
        import curses
    except ImportError:
        print("strife: the chat screen needs curses (on Windows: pip install windows-curses); "
              "use hnx strife chat --plain")
        return 2
    os.environ.setdefault("ESCDELAY", "25")

    def loop(stdscr: Any) -> int:
        curses.raw()
        stdscr.keypad(True)
        stdscr.timeout(80)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        curses.mouseinterval(0)
        screen = _CursesScreen(curses, stdscr, _styles(curses))
        keys = _curses_keys(curses)
        wheel_up = getattr(curses, "BUTTON4_PRESSED", 0x80000)
        wheel_down = getattr(curses, "BUTTON5_PRESSED", 0x200000)
        clicks = curses.BUTTON1_PRESSED | curses.BUTTON1_CLICKED | curses.BUTTON1_RELEASED
        dirty = True
        while app.running:
            if app.pump():
                dirty = True
            if dirty:
                stdscr.erase()
                app.render(screen)
                stdscr.refresh()
                dirty = False
            try:
                ch = stdscr.get_wch()
            except curses.error:
                if app.busy:
                    dirty = True  # keep the "generating" line alive
                continue
            dirty = True
            if ch == curses.KEY_MOUSE:
                try:
                    _id, x, y, _z, state = curses.getmouse()
                except curses.error:
                    continue
                if state & wheel_up:
                    app.wheel(y, x, up=True)
                elif state & wheel_down:
                    app.wheel(y, x, up=False)
                elif state & curses.BUTTON1_PRESSED or (state & clicks and not state & curses.BUTTON1_RELEASED):
                    app.click(y, x)
                continue
            if ch == curses.KEY_RESIZE:
                continue
            if ch in ("\x1b", 27):
                stdscr.timeout(30)
                try:
                    nxt = stdscr.get_wch()
                except curses.error:
                    nxt = None
                stdscr.timeout(80)
                app.key("newline" if nxt in ("\n", "\r", 10, 13) else "escape")
                continue
            if isinstance(ch, str):
                code = ord(ch) if len(ch) == 1 else -1
                name = keys.get(code)
                app.key(name if name and name != "redraw" else ch if code >= 32 else None)
            else:
                name = keys.get(ch)
                if name and name != "redraw":
                    app.key(name)
        app.stop()
        return 0

    return curses.wrapper(loop)
