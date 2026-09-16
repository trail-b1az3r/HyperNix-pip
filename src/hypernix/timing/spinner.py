"""spinner — Terminal loading animations for HyperNix CLI.

Provides a lightweight spinner/progress animation that works without
any external dependencies (pure stdlib + ANSI), with a rich fallback
for a richer experience when available.

Usage:
    from hypernix.spinner import Spinner, anime_print

    with Spinner("Loading model..."):
        load_heavy_stuff()

    anime_print("HyperNix", style="banner")
"""
from __future__ import annotations

import sys
import threading
import time

# ANSI helpers
_CSI = "\x1b["
_HIDE = f"{_CSI}?25l"
_SHOW = f"{_CSI}?25h"
_CYAN = f"{_CSI}96m"
_RESET = f"{_CSI}0m"
_GREEN = f"{_CSI}92m"
_YELLOW = f"{_CSI}93m"
_BOLD = f"{_CSI}1m"
_DIM = f"{_CSI}2m"


def _is_tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


# ---------------------------------------------------------------------------
# Spinner frames — several beautiful animation styles
# ---------------------------------------------------------------------------

SPINNERS: dict[str, list[str]] = {
    "dots":    ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
    "arc":     ["◜", "◠", "◝", "◞", "◡", "◟"],
    "bounce":  ["⠁", "⠂", "⠄", "⠂"],
    "pulse":   ["█", "▓", "▒", "░", "▒", "▓"],
    "bar":     ["[    ]", "[=   ]", "[==  ]", "[=== ]", "[====]", "[ ===]", "[  ==]", "[   =]"],
    "moon":    ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
    "clock":   ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"],
    "fire":    ["🔥", "💥", "✨", "💫", "⚡"],
    "star":    ["✶", "✸", "✹", "✺", "✹", "✷"],
    "line":    ["─", "╲", "│", "╱"],
    "grow":    ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", "▎"],
    "arrows":  ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    "matrix":  ["0", "1", "0", "1", "0", "1"],
}

# ---------------------------------------------------------------------------
# 3x3 grid spinner (no center) — cells light up to show estimated progress
# ---------------------------------------------------------------------------
#
# Eight cells are arranged clockwise around a deliberately blank center:
#
#     0 1 2
#     7 · 3
#     6 5 4
#
# `progress` (0..1) sets how many of the 8 outer cells are "filled" to
# represent estimated loaded progress; filled cells ramp red -> yellow ->
# green as progress increases, so the grid reads as a loading gauge at a
# glance. A bright highlight also rotates around the ring each tick so the
# grid keeps animating even when progress hasn't changed yet.

_GRID3X3_CELL_COUNT = 8
_GRID3X3_LAYOUT: tuple[tuple[int | None, ...], ...] = (
    (0, 1, 2),
    (7, None, 3),
    (6, 5, 4),
)
_GRID3X3_RAMP = [
    f"{_CSI}31m",  # red          — just started
    _YELLOW,       # yellow       — a little over a third done
    _GREEN,        # green        — over halfway
    f"{_CSI}92m",  # bright green — nearly / fully done
]
_GRID3X3_ACTIVE_COLOR = f"{_BOLD}{_CSI}96m"  # bright cyan highlight


def _grid3x3_ramp_color(progress: float) -> str:
    progress = max(0.0, min(1.0, progress))
    idx = min(len(_GRID3X3_RAMP) - 1, int(progress * len(_GRID3X3_RAMP)))
    return _GRID3X3_RAMP[idx]


def _grid3x3_frame(progress: float, tick: int, *, color: bool = True, label: str = "") -> list[str]:
    """Render one frame of the 3x3 (no-center) grid loader as text lines.

    Returns the 3 grid rows plus (if `label` is given) a trailing label
    row showing the text and numeric percentage, so callers can redraw
    the whole block in place each tick.
    """
    progress = max(0.0, min(1.0, progress))
    filled = round(progress * _GRID3X3_CELL_COUNT)
    active = tick % _GRID3X3_CELL_COUNT
    fill_color = _grid3x3_ramp_color(progress)

    lines: list[str] = []
    for row in _GRID3X3_LAYOUT:
        cells = []
        for pos in row:
            if pos is None:
                cells.append(" ")  # blank center — deliberately no cell here
                continue
            is_filled = pos < filled
            glyph = "█" if is_filled else "▫"
            if not color:
                cells.append(glyph)
                continue
            if pos == active:
                cells.append(f"{_GRID3X3_ACTIVE_COLOR}{glyph}{_RESET}")
            elif is_filled:
                cells.append(f"{fill_color}{glyph}{_RESET}")
            else:
                cells.append(f"{_DIM}{glyph}{_RESET}")
        lines.append(" ".join(cells))

    if label:
        pct = f"{progress * 100:.0f}%"
        if color:
            lines.append(f"{_BOLD}{label}{_RESET}  {_DIM}{pct}{_RESET}")
        else:
            lines.append(f"{label}  {pct}")
    return lines


BANNER_FRAMES = [
    "H Y P E R N I X",
    "H·Y·P·E·R·N·I·X",
    "HYPERNIX",
    "⚡HYPERNIX⚡",
    "▓HYPERNIX▓",
    "█HYPERNIX█",
]


# ---------------------------------------------------------------------------
# Spinner context manager
# ---------------------------------------------------------------------------

class Spinner:
    """Thread-based terminal spinner.

    Works with or without rich. Falls back to simple print on non-TTY.

    Example::

        with Spinner("Downloading model") as sp:
            do_work()
            sp.update("Still downloading...")
        # On exit, prints ✓ Done (or ✗ on exception)
    """

    def __init__(
        self,
        text: str = "Loading",
        *,
        style: str = "dots",
        fps: float = 10,
        color: bool = True,
        progress: float = 0.0,
    ) -> None:
        self.text = text
        self._style = style
        self._frames = SPINNERS.get(style, SPINNERS["dots"])
        self._interval = 1.0 / max(fps, 1)
        self._color = color and _is_tty()
        self._tty = _is_tty()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._succeeded: bool = True
        self._progress = max(0.0, min(1.0, progress))
        self._grid_lines_drawn = 0

    def update(self, text: str) -> None:
        with self._lock:
            self.text = text

    def set_progress(self, progress: float) -> None:
        """Update the estimated loaded progress (0..1) shown by the
        ``grid3x3`` style. No-op for other styles."""
        with self._lock:
            self._progress = max(0.0, min(1.0, progress))

    def _spin(self) -> None:
        idx = 0
        if self._tty:
            sys.stdout.write(_HIDE)
            sys.stdout.flush()
        try:
            while not self._stop.is_set():
                with self._lock:
                    msg = self.text
                    progress = self._progress
                if self._style == "grid3x3":
                    self._render_grid3x3_frame(progress, idx, msg)
                elif self._tty:
                    frame = self._frames[idx % len(self._frames)]
                    if self._color:
                        line = f"\r{_CYAN}{frame}{_RESET}  {_BOLD}{msg}{_RESET}  "
                    else:
                        line = f"\r{frame}  {msg}  "
                    sys.stdout.write(line)
                    sys.stdout.flush()
                idx += 1
                time.sleep(self._interval)
        finally:
            pass

    def _render_grid3x3_frame(self, progress: float, tick: int, msg: str) -> None:
        """Redraw the 3x3 (no-center) progress grid in place."""
        if not self._tty:
            return
        grid_lines = _grid3x3_frame(progress, tick, color=self._color, label=msg)
        if self._grid_lines_drawn:
            # Move the cursor back up to the top of the previously drawn
            # block so this frame overwrites it instead of scrolling.
            sys.stdout.write(f"{_CSI}{self._grid_lines_drawn}A")
        for line in grid_lines:
            sys.stdout.write(f"\r{_CSI}2K{line}\n")
        self._grid_lines_drawn = len(grid_lines)
        sys.stdout.flush()

    def start(self) -> Spinner:
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, succeeded: bool = True) -> None:
        self._succeeded = succeeded
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._tty:
            sys.stdout.write(_SHOW)
            icon = f"{_GREEN}✓{_RESET}" if succeeded else f"\x1b[91m✗{_RESET}"
            sys.stdout.write(f"\r{icon}  {_BOLD}{self.text}{_RESET}  \n")
            sys.stdout.flush()
        else:
            status = "✓" if succeeded else "✗"
            print(f"{status}  {self.text}")

    def __enter__(self) -> Spinner:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop(succeeded=exc_type is None)
        return False


# ---------------------------------------------------------------------------
# Animated banner / intro print
# ---------------------------------------------------------------------------

#: Characters the decode effect scrambles through. Deliberately narrow
#: and monospace-safe: a set with wide glyphs in it makes the line change
#: width between frames, which reads as jitter rather than as decoding.
_DECODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&@*+=<>/\\|"


def decode_frames(
    text: str, *, steps: int = 18, seed: int | None = None
) -> list[str]:
    """The frames of a ``decode`` animation, as plain strings.

    Separated from the printing so it can be tested — an animation that
    only exists inside a ``time.sleep`` loop is one nobody ever checks,
    and the property that matters (the last frame is the text, exactly)
    is trivial to assert and easy to get wrong.

    Characters resolve in a *random* order rather than left to right.
    That is the whole difference from ``glitch``: a left-to-right reveal
    reads as typing, and the thing being imitated here reads as
    decryption because the plaintext appears out of order.

    Spaces never scramble. A title whose word boundaries move around is
    one the eye cannot lock onto, so the shape of the text is visible
    from the first frame and only the letters resolve.
    """
    import random

    rng = random.Random(seed)
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    rng.shuffle(positions)
    steps = max(1, int(steps))

    frames: list[str] = []
    for step in range(steps):
        # Ceil, so the last frame before the final one has resolved
        # everything but at most one character rather than jumping.
        resolved = set(positions[: (len(positions) * (step + 1) + steps - 1) // steps])
        frames.append("".join(
            ch if (index in resolved or ch.isspace())
            else rng.choice(_DECODE_ALPHABET)
            for index, ch in enumerate(text)
        ))
    # The contract: whatever the arithmetic above did, the animation ends
    # showing exactly what was asked for.
    frames[-1] = text
    return frames


def anime_print(
    text: str = "HyperNix",
    *,
    style: str = "banner",
    color: bool = True,
    delay: float = 0.05,
) -> None:
    """Print an animated banner or text effect to the terminal.

    Styles:
      ``banner``  — slide in letter by letter then flash
      ``glitch``  — glitch effect (randomised chars) then settle
      ``decode``  — scrambled text resolving in random order, as if
                    decrypted. See :func:`_decode_frames`.
      ``typewriter`` — typewriter character reveal
      ``fade``    — fade in using block characters
    """
    if not _is_tty() or not color:
        print(text)
        return

    if style == "typewriter":
        sys.stdout.write(_HIDE)
        sys.stdout.flush()
        try:
            for i in range(1, len(text) + 1):
                sys.stdout.write(f"\r{_BOLD}{_CYAN}{text[:i]}{_RESET}{'▌' if i < len(text) else '  '}")
                sys.stdout.flush()
                time.sleep(delay)
        finally:
            sys.stdout.write(_SHOW + "\n")
            sys.stdout.flush()

    elif style == "glitch":
        import random
        chars = "!@#$%^&*<>?/\\|0123456789"
        sys.stdout.write(_HIDE)
        sys.stdout.flush()
        try:
            for step in range(12):
                glitched = ""
                for i, ch in enumerate(text):
                    if i < (step / 12) * len(text):
                        glitched += ch
                    elif ch == " ":
                        glitched += " "
                    else:
                        glitched += random.choice(chars)
                sys.stdout.write(f"\r{_BOLD}\x1b[95m{glitched}{_RESET}   ")
                sys.stdout.flush()
                time.sleep(delay * 1.5)
            sys.stdout.write(f"\r{_BOLD}{_CYAN}{text}{_RESET}   \n")
            sys.stdout.flush()
        finally:
            sys.stdout.write(_SHOW)
            sys.stdout.flush()

    elif style == "decode":
        sys.stdout.write(_HIDE)
        sys.stdout.flush()
        try:
            frames = decode_frames(text)
            for index, frame in enumerate(frames):
                # Unresolved characters stay dim and the resolved ones
                # brighten, so the eye follows the plaintext appearing
                # rather than the noise moving.
                shade = _CYAN if index == len(frames) - 1 else "\x1b[96;2m"
                sys.stdout.write(f"\r{_BOLD}{shade}{frame}{_RESET}   ")
                sys.stdout.flush()
                time.sleep(delay)
            sys.stdout.write(f"\r{_BOLD}{_CYAN}{text}{_RESET}   \n")
            sys.stdout.flush()
        finally:
            sys.stdout.write(_SHOW)
            sys.stdout.flush()

    elif style == "fade":
        blocks = ["░", "▒", "▓", "█"]
        sys.stdout.write(_HIDE)
        sys.stdout.flush()
        try:
            for block in blocks:
                sys.stdout.write(f"\r{_BOLD}{_CYAN}{block * len(text)}{_RESET}")
                sys.stdout.flush()
                time.sleep(delay * 2)
            sys.stdout.write(f"\r{_BOLD}{_CYAN}{text}{_RESET}   \n")
            sys.stdout.flush()
        finally:
            sys.stdout.write(_SHOW)
            sys.stdout.flush()

    else:  # banner (default): slide-in then flash
        sys.stdout.write(_HIDE)
        sys.stdout.flush()
        try:
            padded = " " * len(text) + text
            for i in range(len(text) + 1):
                shown = padded[i:i + len(text)]
                sys.stdout.write(f"\r{_BOLD}{_CYAN}{shown}{_RESET}")
                sys.stdout.flush()
                time.sleep(delay)
            # Flash effect
            for flash_color in ["\x1b[97m", _CYAN, "\x1b[97m", _CYAN]:
                sys.stdout.write(f"\r{_BOLD}{flash_color}{text}{_RESET}")
                sys.stdout.flush()
                time.sleep(delay * 2)
            sys.stdout.write(f"\r{_BOLD}{_CYAN}{text}{_RESET}   \n")
            sys.stdout.flush()
        finally:
            sys.stdout.write(_SHOW)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Convenience: print a styled startup header
# ---------------------------------------------------------------------------

def print_startup_header(version: str = "") -> None:
    """Print the HyperNix startup banner with version and animation."""
    if not _is_tty():
        print(f"HyperNix {version}")
        return

    logo = r"""
  ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ███╗   ██╗██╗██╗  ██╗
  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗████╗  ██║██║╚██╗██╔╝
  ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝██╔██╗ ██║██║ ╚███╔╝ 
  ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██║╚██╗██║██║ ██╔██╗ 
  ██║  ██║   ██║   ██║     ███████╗██║  ██║██║ ╚████║██║██╔╝ ██╗
  ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝
"""

    sys.stdout.write(_HIDE)
    sys.stdout.flush()
    try:
        lines = logo.strip("\n").split("\n")
        for i, line in enumerate(lines):
            # Animate each line sliding in from the left
            for end in range(0, len(line) + 1, 3):
                sys.stdout.write(f"\x1b[{i + 1};1H{_CYAN}{line[:end]}{_RESET}")
                sys.stdout.flush()
                time.sleep(0.003)
            sys.stdout.write(f"\x1b[{i + 1};1H{_CYAN}{line}{_RESET}")
            sys.stdout.flush()

        # Print version below logo
        ver_line = f"  {_DIM}v{version}{_RESET}" if version else ""
        sys.stdout.write(f"\n{ver_line}\n\n")
        sys.stdout.flush()
        time.sleep(0.1)
    finally:
        sys.stdout.write(_SHOW)
        sys.stdout.flush()
