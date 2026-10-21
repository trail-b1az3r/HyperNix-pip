"""An inline arrow-key chooser, like ``lms load``'s.

Drawn in place under the prompt -- not a full-screen TUI: the terminal's
scrollback stays as it was, and when the choice is made the list is
replaced by one line saying what was chosen. ↑/↓ (or Ctrl-P/Ctrl-N, k/j
when not filtering) move, typing filters (letters in order, anywhere in
the name: ``q38`` finds ``qwen3-8b``), Enter picks, Esc or Ctrl-C gives
up. The list is fetched again every couple of seconds while it is open,
so a model that finishes downloading appears without starting over.

The state machine (:class:`Picker`) is pure, so the tests drive it with
keys; :func:`choose` is the terminal around it.
"""
from __future__ import annotations

import os
import select
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Choice", "Picker", "choose", "fuzzy_score", "interactive"]


@dataclass(frozen=True)
class Choice:
    value: str
    label: str = ""
    detail: str = ""
    #: Shown with a marker, e.g. what is serving now.
    current: bool = False

    @property
    def text(self) -> str:
        return self.label or self.value


def fuzzy_score(query: str, text: str) -> int | None:
    """How well *query* matches *text*: None for no match, higher is better.

    Every character of the query must appear in the text in order. A
    whole substring scores highest, then a match at the start, then
    matches whose characters sit close together.
    """
    q = query.lower().strip()
    t = text.lower()
    if not q:
        return 0
    if q in t:
        return 1000 - t.index(q) * 2 - (len(t) - len(q))
    position, gaps, start = 0, 0, -1
    for ch in q:
        found = t.find(ch, position)
        if found < 0:
            return None
        if start < 0:
            start = found
        gaps += found - position
        position = found + 1
    return 500 - gaps * 5 - start


@dataclass
class Picker:
    title: str
    items: list[Choice] = field(default_factory=list)
    query: str = ""
    index: int = 0
    done: bool = False
    cancelled: bool = False
    note: str = ""

    def visible(self) -> list[Choice]:
        scored = [(fuzzy_score(self.query, c.text), n, c) for n, c in enumerate(self.items)]
        kept = [(s, n, c) for s, n, c in scored if s is not None]
        if self.query:
            kept.sort(key=lambda row: (-row[0], row[1]))
        return [c for _, _, c in kept]

    def selected(self) -> Choice | None:
        shown = self.visible()
        return shown[min(self.index, len(shown) - 1)] if shown else None

    def set_items(self, items: Sequence[Choice]) -> None:
        """New items from a refresh, keeping the selection on the same value."""
        keep = self.selected()
        self.items = list(items)
        shown = self.visible()
        if keep is not None:
            for n, c in enumerate(shown):
                if c.value == keep.value:
                    self.index = n
                    return
        self.index = min(self.index, max(0, len(shown) - 1))

    def key(self, key: str) -> None:
        shown = self.visible()
        if key in ("up", "ctrl-p") or (key == "k" and not self.query):
            self.index = (self.index - 1) % len(shown) if shown else 0
        elif key in ("down", "ctrl-n", "tab") or (key == "j" and not self.query):
            self.index = (self.index + 1) % len(shown) if shown else 0
        elif key == "pageup":
            self.index = max(0, self.index - 10)
        elif key == "pagedown":
            self.index = min(max(0, len(shown) - 1), self.index + 10)
        elif key == "enter":
            if shown:
                self.done = True
            else:
                self.note = "nothing matches; Backspace to widen the search"
        elif key in ("escape", "ctrl-c"):
            self.done = self.cancelled = True
        elif key == "backspace":
            self.query = self.query[:-1]
            self.index = 0
        elif len(key) == 1 and key.isprintable():
            self.query += key
            self.index = 0

    def render(self, width: int = 80, height: int = 12) -> list[str]:
        shown = self.visible()
        lines = [f"{self.title}  {('filter: ' + self.query) if self.query else '(type to filter)'}"[:width]]
        if not shown:
            lines.append("  nothing matches" if self.items else "  (nothing yet; refreshing…)")
        top = max(0, min(self.index - height // 2, len(shown) - height))
        for n, c in enumerate(shown[top:top + height], start=top):
            marker = "❯" if n == self.index else " "
            current = " ●" if c.current else ""
            detail = f"  {c.detail}" if c.detail else ""
            lines.append(f"{marker} {c.text}{current}{detail}"[:width])
        if len(shown) > height:
            lines.append(f"  {len(shown)} in all, {top + 1}-{min(top + height, len(shown))} shown"[:width])
        lines.append((self.note or "↑↓ move · type to filter · Enter choose · Esc cancel")[:width])
        return lines


# -- the terminal ------------------------------------------------------------------

_KEYS = {
    "\x1b[A": "up", "\x1bOA": "up", "\x1b[B": "down", "\x1bOB": "down",
    "\x1b[5~": "pageup", "\x1b[6~": "pagedown",
    "\r": "enter", "\n": "enter", "\x1b": "escape", "\x03": "ctrl-c",
    "\x7f": "backspace", "\x08": "backspace", "\x10": "ctrl-p", "\x0e": "ctrl-n", "\t": "tab",
}


def decode_keys(data: str) -> list[str]:
    """Raw terminal input as key names, several per read when typed fast."""
    out: list[str] = []
    i = 0
    while i < len(data):
        for size in (4, 3, 2):
            chunk = data[i:i + size]
            if chunk in _KEYS and len(chunk) == size:
                out.append(_KEYS[chunk])
                i += size
                break
        else:
            if data[i] == "\x1b" and i + 1 < len(data) and data[i + 1] == "[":
                # An escape sequence this does not know: skip it whole.
                j = i + 2
                while j < len(data) and not data[j].isalpha() and data[j] != "~":
                    j += 1
                i = j + 1
                continue
            out.append(_KEYS.get(data[i], data[i]))
            i += 1
    return out


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def choose(
    title: str,
    fetch: Callable[[], Sequence[Choice]],
    *,
    refresh_seconds: float = 2.0,
    initial_query: str = "",
    out: Any = None,
    fd: int | None = None,
) -> Choice | None:
    """Ask on the terminal. None if cancelled."""
    import shutil
    import termios
    import tty

    out = out or sys.stdout
    picker = Picker(title, list(fetch()), query=initial_query)
    fd = sys.stdin.fileno() if fd is None else fd
    saved = termios.tcgetattr(fd)
    drawn = 0

    def draw() -> None:
        nonlocal drawn
        size = shutil.get_terminal_size((80, 24))
        lines = picker.render(size.columns - 1, max(3, min(12, size.lines - 4)))
        # Back to where the list started, clear to the end, draw again.
        if drawn:
            out.write(f"\x1b[{drawn}F")
        out.write("\x1b[J" + "\r\n".join(lines) + "\r\n")
        out.flush()
        drawn = len(lines)

    try:
        tty.setcbreak(fd)
        out.write("\x1b[?25l")  # hide the cursor while the list is up
        draw()
        last = time.monotonic()
        while not picker.done:
            ready, _, _ = select.select([fd], [], [], 0.25)
            if ready:
                data = os.read(fd, 64).decode("utf-8", errors="ignore")
                for key in decode_keys(data):
                    picker.key(key)
                    if picker.done:
                        break
                draw()
            if refresh_seconds and time.monotonic() - last >= refresh_seconds and not picker.done:
                last = time.monotonic()
                try:
                    picker.set_items(fetch())
                    picker.note = ""
                except Exception as exc:  # noqa: BLE001 - a failed refresh keeps the old list
                    picker.note = f"refresh failed: {exc}"[:120]
                draw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        if drawn:
            out.write(f"\x1b[{drawn}F\x1b[J")
        out.write("\x1b[?25h")
        out.flush()
    if picker.cancelled:
        out.write(f"{title}: cancelled\n")
        return None
    chosen = picker.selected()
    out.write(f"{title}: {chosen.text if chosen else '-'}\n")
    return chosen
