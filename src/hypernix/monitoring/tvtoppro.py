"""hypernix.monitoring.tvtoppro — tvtop++'s numbers, btop++'s presentation.

    tvtoppro
    tvtoppro --theme gruvbox-dark
    tvtoppro --theme ~/.config/btop/themes/nord.theme
    tvtoppro --list-themes

Not built on cctvtop
--------------------
``cctvtop`` wraps a compiled C++ dashboard and shells out to it; when the
extension is not built for the running Python it falls back to importing
``tvtop_plus_plus`` and rendering that instead. So "run cctvtop" means
one of two different programs depending on how the wheel was built, and
the fallback is the one most people see.

This takes the other half: :class:`~hypernix.monitoring.tvtop_plus_plus.TVTopPlusPlus`
is used purely as a *stat source* — its ``latest_frame`` already collects
per-core CPU, the /proc/meminfo breakdown, the full ``nvidia-smi`` row
and the training-log tail — and everything drawn is new.

What "presented exactly like btop++" means here
-----------------------------------------------
btop++'s look is four specific things, and skipping any of them gives
something that is merely a boxed TUI:

1. **Boxes titled in the border.** ``┌─┤ cpu ├───────┐``, not a heading
   inside the box. The title sits in the rule with a bracket either side
   and the box number in the opposite corner.
2. **Braille graphs.** Two samples per character cell horizontally, four
   per cell vertically, so a 60x4 graph plots 120 points across 16
   levels. Block characters give 60 points across 4.
3. **Gradient meters.** A bar is not one colour that changes with the
   value — each *cell* takes its colour from its own position along a
   start/mid/end ramp, so a full bar shows the whole ramp and a quarter
   bar shows only the cool end. Getting this wrong is what makes a
   btop-alike look like a progress bar.
4. **Themes as data.** btop's ``.theme`` files are ``theme[key]="#hex"``
   lines, and this reads them directly — so a theme someone already has
   in ``~/.config/btop/themes`` works here without conversion. Six are
   built in.

Colour customization
--------------------
``--theme`` takes a built-in name, a path to a btop ``.theme`` file, or a
JSON file of the same keys. ``--dump-theme`` writes the active one out in
btop's format, which is the quickest way to start editing one. Unknown
keys are kept and unset ones inherit from the default, so a partial theme
is a valid theme rather than a crash.

Startup
-------
The ``decode`` animation and a spinner, the way ``tvtop-older`` opens.
Neither is decoration: the spinner runs during log autodetection, which
walks the working tree and is the one part of startup that can take a
noticeable moment, and a dashboard that shows nothing while it does that
looks hung. ``--no-intro`` skips both, and so does a non-TTY without
being asked — an animation in a CI log is noise.

Extra panels
------------
The five boxes here are the five that matter on a training box and not
the only five anyone wants. ``--modules disk,net,swap`` adds more, from
:mod:`hypernix.monitoring.tvtoppro_modules`: five ship, packages can
advertise their own through an entry point, and a ``.py`` file dropped in
``~/.config/hypernix/tvtoppro/modules/`` is picked up with no
installation at all. ``--list-modules`` shows what was found, including
what failed to load and why.

When the log stops moving
-------------------------
A dashboard tailing a ``train.log`` nobody has written to since last
Tuesday is not broken — it is faithfully reporting a dead file, and
nothing on screen says so. If the log has gone untouched for a week
(``--stale-after``), tvtoppro finds the busiest Python process on the
machine, reads the logs *it* has open, and offers the one with progress
lines in it. ``--find-run`` does the same on demand.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Theme",
    "THEMES",
    "load_theme",
    "parse_btop_theme",
    "gradient",
    "braille_graph",
    "meter",
    "box_top",
    "box_bottom",
    "TvTopPro",
    "play_intro",
    "cli_main",
    "main",
]

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{6})$")


def _rgb(value: str) -> tuple[int, int, int]:
    """A btop theme colour to ``(r, g, b)``.

    btop accepts ``#RRGGBB`` and a two-digit greyscale shorthand
    ``#XX``. The shorthand is not a truncated hex triplet — it is a grey
    level — and reading it as one turns every neutral in a real theme
    into a dark red.
    """
    text = str(value).strip().strip('"').strip("'")
    if not _HEX.match(text):
        raise ValueError(f"{value!r} is not a btop theme colour (#RRGGBB or #XX)")
    digits = text[1:]
    if len(digits) == 2:
        level = int(digits, 16)
        return (level, level, level)
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))


def _hex(rgb: tuple[int, int, int]) -> str:
    red, green, blue = (max(0, min(255, int(c))) for c in rgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


def gradient(start: str, mid: str, end: str, steps: int) -> list[str]:
    """``steps`` colours ramping start -> mid -> end.

    Two linear segments rather than one, because btop's ramps are not
    monotonic in any channel: the cyan-yellow-red CPU ramp passes
    *through* yellow, and interpolating cyan straight to red gives a
    muddy purple that looks nothing like it.
    """
    if steps <= 0:
        return []
    if steps == 1:
        return [_hex(_rgb(start))]
    first, middle, last = _rgb(start), _rgb(mid), _rgb(end)
    out: list[str] = []
    half = (steps - 1) / 2
    for index in range(steps):
        if index <= half:
            fraction = index / half if half else 0.0
            low, high = first, middle
        else:
            fraction = (index - half) / (steps - 1 - half)
            low, high = middle, last
        out.append(_hex(tuple(
            low[channel] + (high[channel] - low[channel]) * fraction
            for channel in range(3)
        )))
    return out


#: Every key a theme can set. The names are btop's, so its own theme
#: files load unchanged.
_THEME_KEYS = (
    "main_bg", "main_fg", "title", "hi_fg", "selected_bg", "selected_fg",
    "inactive_fg", "graph_text", "meter_bg", "proc_misc", "div_line",
    "cpu_box", "mem_box", "net_box", "proc_box",
    "cpu_start", "cpu_mid", "cpu_end",
    "free_start", "free_mid", "free_end",
    "used_start", "used_mid", "used_end",
    "available_start", "available_mid", "available_end",
    "temp_start", "temp_mid", "temp_end",
    "download_start", "download_mid", "download_end",
    "upload_start", "upload_mid", "upload_end",
)


@dataclass
class Theme:
    """A btop-compatible palette.

    Every field defaults, so a theme file that sets three keys is a valid
    theme with three keys changed. btop behaves the same way and the
    alternative — refusing a partial file — makes hand-editing one a
    guessing game about which keys are mandatory.
    """

    name: str = "hypernix"
    colors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = dict(_DEFAULT_COLORS)
        merged.update({k: v for k, v in self.colors.items() if v})
        self.colors = merged

    def __getitem__(self, key: str) -> str:
        return self.colors.get(key, _DEFAULT_COLORS.get(key, "#cccccc"))

    def ramp(self, prefix: str, steps: int) -> list[str]:
        """The ``<prefix>_start/_mid/_end`` gradient, ``steps`` long."""
        return gradient(
            self[f"{prefix}_start"], self[f"{prefix}_mid"], self[f"{prefix}_end"],
            steps,
        )

    def to_btop(self) -> str:
        """This theme as a btop ``.theme`` file."""
        lines = [f'# {self.name} — written by hypernix tvtoppro']
        lines += [f'theme[{key}]="{self.colors[key]}"' for key in _THEME_KEYS
                  if key in self.colors]
        return "\n".join(lines) + "\n"


_DEFAULT_COLORS: dict[str, str] = {
    "main_bg": "#00", "main_fg": "#cc", "title": "#ee", "hi_fg": "#7bd88f",
    "selected_bg": "#2f3b47", "selected_fg": "#ffffff", "inactive_fg": "#40",
    "graph_text": "#60", "meter_bg": "#40", "proc_misc": "#7bd88f",
    "div_line": "#30",
    "cpu_box": "#3d7b46", "mem_box": "#8a882e", "net_box": "#423ba5",
    "proc_box": "#923535",
    "cpu_start": "#50f0ff", "cpu_mid": "#f2e266", "cpu_end": "#fc2929",
    "free_start": "#223014", "free_mid": "#b5e685", "free_end": "#dcff85",
    "used_start": "#0b1a29", "used_mid": "#4c7cb0", "used_end": "#74e6fc",
    "available_start": "#292107", "available_mid": "#a3a10a",
    "available_end": "#fffa50",
    "temp_start": "#4897d4", "temp_mid": "#5474e8", "temp_end": "#ff40b6",
    "download_start": "#de6e6e", "download_mid": "#c72e2e", "download_end": "#ff0000",
    "upload_start": "#63c5b7", "upload_mid": "#26c9b0", "upload_end": "#00ffd0",
}


THEMES: dict[str, Theme] = {
    "hypernix": Theme("hypernix", {}),
    "gruvbox-dark": Theme("gruvbox-dark", {
        "main_bg": "#282828", "main_fg": "#ebdbb2", "title": "#fbf1c7",
        "hi_fg": "#fabd2f", "selected_bg": "#3c3836", "selected_fg": "#fbf1c7",
        "inactive_fg": "#665c54", "graph_text": "#a89984", "meter_bg": "#504945",
        "proc_misc": "#8ec07c", "div_line": "#504945",
        "cpu_box": "#98971a", "mem_box": "#d79921", "net_box": "#458588",
        "proc_box": "#cc241d",
        "cpu_start": "#83a598", "cpu_mid": "#fabd2f", "cpu_end": "#fb4934",
        "free_start": "#427b58", "free_mid": "#8ec07c", "free_end": "#b8bb26",
        "used_start": "#458588", "used_mid": "#83a598", "used_end": "#d3869b",
        "available_start": "#79740e", "available_mid": "#b57614",
        "available_end": "#fabd2f",
        "temp_start": "#458588", "temp_mid": "#d79921", "temp_end": "#fb4934",
    }),
    "nord": Theme("nord", {
        "main_bg": "#2e3440", "main_fg": "#d8dee9", "title": "#eceff4",
        "hi_fg": "#88c0d0", "selected_bg": "#434c5e", "selected_fg": "#eceff4",
        "inactive_fg": "#4c566a", "graph_text": "#7b88a1", "meter_bg": "#3b4252",
        "proc_misc": "#a3be8c", "div_line": "#434c5e",
        "cpu_box": "#81a1c1", "mem_box": "#8fbcbb", "net_box": "#b48ead",
        "proc_box": "#bf616a",
        "cpu_start": "#8fbcbb", "cpu_mid": "#ebcb8b", "cpu_end": "#bf616a",
        "free_start": "#4c566a", "free_mid": "#a3be8c", "free_end": "#d8dee9",
        "used_start": "#5e81ac", "used_mid": "#81a1c1", "used_end": "#88c0d0",
        "available_start": "#5e5b3a", "available_mid": "#d08770",
        "available_end": "#ebcb8b",
        "temp_start": "#5e81ac", "temp_mid": "#d08770", "temp_end": "#bf616a",
    }),
    "dracula": Theme("dracula", {
        "main_bg": "#282a36", "main_fg": "#f8f8f2", "title": "#ffffff",
        "hi_fg": "#bd93f9", "selected_bg": "#44475a", "selected_fg": "#f8f8f2",
        "inactive_fg": "#6272a4", "graph_text": "#8be9fd", "meter_bg": "#44475a",
        "proc_misc": "#50fa7b", "div_line": "#44475a",
        "cpu_box": "#bd93f9", "mem_box": "#50fa7b", "net_box": "#8be9fd",
        "proc_box": "#ff79c6",
        "cpu_start": "#8be9fd", "cpu_mid": "#f1fa8c", "cpu_end": "#ff5555",
        "free_start": "#2d4a35", "free_mid": "#50fa7b", "free_end": "#f8f8f2",
        "used_start": "#44475a", "used_mid": "#bd93f9", "used_end": "#ff79c6",
        "available_start": "#4a4a2d", "available_mid": "#ffb86c",
        "available_end": "#f1fa8c",
        "temp_start": "#8be9fd", "temp_mid": "#ffb86c", "temp_end": "#ff5555",
    }),
    "tokyo-night": Theme("tokyo-night", {
        "main_bg": "#1a1b26", "main_fg": "#c0caf5", "title": "#c0caf5",
        "hi_fg": "#7aa2f7", "selected_bg": "#33467c", "selected_fg": "#c0caf5",
        "inactive_fg": "#565f89", "graph_text": "#737aa2", "meter_bg": "#292e42",
        "proc_misc": "#9ece6a", "div_line": "#292e42",
        "cpu_box": "#7aa2f7", "mem_box": "#9ece6a", "net_box": "#bb9af7",
        "proc_box": "#f7768e",
        "cpu_start": "#7dcfff", "cpu_mid": "#e0af68", "cpu_end": "#f7768e",
        "free_start": "#2c3b2c", "free_mid": "#9ece6a", "free_end": "#c0caf5",
        "used_start": "#3d59a1", "used_mid": "#7aa2f7", "used_end": "#7dcfff",
        "available_start": "#4a3d2c", "available_mid": "#ff9e64",
        "available_end": "#e0af68",
        "temp_start": "#7dcfff", "temp_mid": "#ff9e64", "temp_end": "#f7768e",
    }),
    "monokai": Theme("monokai", {
        "main_bg": "#272822", "main_fg": "#f8f8f2", "title": "#f9f8f5",
        "hi_fg": "#a6e22e", "selected_bg": "#49483e", "selected_fg": "#f8f8f2",
        "inactive_fg": "#75715e", "graph_text": "#a59f85", "meter_bg": "#3e3d32",
        "proc_misc": "#a6e22e", "div_line": "#49483e",
        "cpu_box": "#a6e22e", "mem_box": "#e6db74", "net_box": "#66d9ef",
        "proc_box": "#f92672",
        "cpu_start": "#66d9ef", "cpu_mid": "#e6db74", "cpu_end": "#f92672",
        "free_start": "#2d3a1e", "free_mid": "#a6e22e", "free_end": "#e6db74",
        "used_start": "#1e3a3a", "used_mid": "#66d9ef", "used_end": "#ae81ff",
        "available_start": "#3a341e", "available_mid": "#fd971f",
        "available_end": "#e6db74",
        "temp_start": "#66d9ef", "temp_mid": "#fd971f", "temp_end": "#f92672",
    }),
    "mono": Theme("mono", {
        # For a terminal without truecolor, and for a screenshot that has
        # to survive being printed. Every ramp is grey.
        "main_bg": "#00", "main_fg": "#cc", "title": "#ff", "hi_fg": "#ee",
        "selected_bg": "#40", "selected_fg": "#ff", "inactive_fg": "#50",
        "graph_text": "#88", "meter_bg": "#30", "proc_misc": "#bb",
        "div_line": "#40",
        "cpu_box": "#88", "mem_box": "#88", "net_box": "#88", "proc_box": "#88",
        "cpu_start": "#55", "cpu_mid": "#aa", "cpu_end": "#ff",
        "free_start": "#33", "free_mid": "#88", "free_end": "#dd",
        "used_start": "#33", "used_mid": "#88", "used_end": "#dd",
        "available_start": "#33", "available_mid": "#88", "available_end": "#dd",
        "temp_start": "#55", "temp_mid": "#aa", "temp_end": "#ff",
    }),
}

_BTOP_LINE = re.compile(r'^\s*theme\[(?P<key>[a-z_]+)\]\s*=\s*"?(?P<value>#[0-9a-fA-F]+)"?')


def parse_btop_theme(text: str, *, name: str = "custom") -> Theme:
    """A btop ``.theme`` file's contents to a :class:`Theme`.

    Lines that are not ``theme[key]="#colour"`` are ignored rather than
    refused: real theme files carry comments and the occasional setting
    this does not use, and rejecting the file over one of them would
    mean none of them load.
    """
    colors: dict[str, str] = {}
    for line in text.splitlines():
        found = _BTOP_LINE.match(line)
        if not found:
            continue
        try:
            _rgb(found.group("value"))
        except ValueError:
            continue
        colors[found.group("key")] = found.group("value")
    if not colors:
        raise ValueError("no theme[...] lines found; is this a btop theme file?")
    return Theme(name, colors)


def load_theme(name_or_path: str | None) -> Theme:
    """A built-in theme by name, or a ``.theme``/``.json`` file by path."""
    if not name_or_path:
        return THEMES[os.environ.get("TVTOPPRO_THEME", "hypernix")] \
            if os.environ.get("TVTOPPRO_THEME") in THEMES else THEMES["hypernix"]
    key = str(name_or_path).strip()
    if key in THEMES:
        return THEMES[key]

    path = Path(key).expanduser()
    if not path.is_file():
        raise ValueError(
            f"Unknown theme {key!r}. Built in: {', '.join(sorted(THEMES))}. "
            f"Or give a path to a btop .theme or a JSON file."
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        payload = json.loads(text)
        colors = payload.get("colors", payload)
        return Theme(payload.get("name", path.stem), dict(colors))
    return parse_btop_theme(text, name=path.stem)


# ---------------------------------------------------------------------------
# btop's drawing primitives
# ---------------------------------------------------------------------------

#: Braille dot bit for ``(x, y)`` within a 2x4 cell, y counted from the
#: top. The fourth row is 0x40/0x80 rather than continuing the pattern,
#: which is the detail that makes hand-written braille graphs come out
#: with a gap in the bottom row.
_DOTS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


def braille_graph(values: list[float], width: int, height: int) -> list[str]:
    """A history as braille rows, newest on the right.

    Two samples per cell across and four levels per cell down, so a
    ``60 x 4`` graph plots 120 points over 16 levels. Values are
    fractions in ``[0, 1]``; anything outside is clamped rather than
    refused, because a percentage that briefly reads 100.4 is not worth
    ending a dashboard over.

    Returns ``height`` strings of exactly ``width`` characters, top row
    first, so a caller can colour each row from a gradient.
    """
    if width <= 0 or height <= 0:
        return []
    samples = width * 2
    tail = [max(0.0, min(1.0, float(v))) for v in values[-samples:]]
    tail = [0.0] * (samples - len(tail)) + tail

    levels = height * 4
    cells = [[0] * width for _ in range(height)]
    for index, value in enumerate(tail):
        filled = int(round(value * levels))
        if filled <= 0:
            continue
        column, half = divmod(index, 2)
        for level in range(filled):
            # level 0 is the bottom of the graph.
            from_top = levels - 1 - level
            row, within = divmod(from_top, 4)
            cells[row][column] |= _DOTS[half][within]
    return ["".join(chr(0x2800 + cell) for cell in row) for row in cells]


def meter(fraction: float, width: int, ramp: list[str], *,
          empty: str = "#404040") -> str:
    """A btop-style gradient meter as Rich markup.

    Each cell takes its colour from *its own* position along the ramp,
    not from the value — so a full bar shows the whole ramp and a quarter
    bar shows only its cool end. A bar that is one colour chosen by the
    value is a progress bar, and it is the single thing that most makes a
    btop-alike look like something else.
    """
    if width <= 0:
        return ""
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    colours = ramp if len(ramp) == width else gradient(
        ramp[0] if ramp else "#50f0ff",
        ramp[len(ramp) // 2] if ramp else "#f2e266",
        ramp[-1] if ramp else "#fc2929",
        width,
    )
    parts = [
        f"[{colours[i]}]█[/]" if i < filled else f"[{empty}]─[/]"
        for i in range(width)
    ]
    return "".join(parts)


def box_top(title: str, width: int, *, line: str, accent: str,
            number: str = "") -> str:
    """btop's top border: the title sits *in* the rule, bracketed.

    ``┌─┤ cpu ├────────────────────────────┤1├─┐``. A heading inside the
    box is the ordinary way to do this and it is the wrong look.
    """
    if width < 8:
        return f"[{line}]{'─' * max(0, width)}[/]"
    left = f"[{line}]┌─┤[/][{accent}]{title}[/][{line}]├[/]"
    if number:
        right = f"[{line}]┤[/][{accent}]{number}[/][{line}]├─┐[/]"
        used = (4 + len(title)) + (4 + len(number))
    else:
        right = f"[{line}]─┐[/]"
        used = (4 + len(title)) + 2
    return left + f"[{line}]{'─' * max(0, width - used)}[/]" + right


def box_bottom(width: int, *, line: str) -> str:
    return f"[{line}]└{'─' * max(0, width - 2)}┘[/]"


def _fmt_bytes(value: float | int | None) -> str:
    if value is None:
        return "  --  "
    size = float(value)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:5.1f}{unit}"
        size /= 1024
    return f"{size:5.1f}T"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# The dashboard
# ---------------------------------------------------------------------------


@dataclass
class TvTopPro:
    """tvtop++'s frame, drawn the way btop++ draws.

    Composition rather than inheritance: the stat source is a
    :class:`~hypernix.monitoring.tvtop_plus_plus.TVTopPlusPlus` held as an
    attribute, so the two can diverge without either dragging the other.
    Every method below is presentation.
    """

    log_path: Path | str | None = None
    refresh_seconds: float = 1.0
    theme: Theme = field(default_factory=lambda: THEMES["hypernix"])
    width: int | None = None
    show_processes: bool = True
    source: Any = field(default=None, repr=False)
    #: Extra panels, from :mod:`hypernix.monitoring.tvtoppro_modules`.
    modules: list[Any] = field(default_factory=list)
    #: Seconds of silence before the log is treated as dead and the
    #: busiest Python process is asked what it is doing instead.
    stale_after: float = 0.0
    #: Run that investigation whatever the log's age.
    always_find_run: bool = False

    def __post_init__(self) -> None:
        if self.source is None:
            from .tvtop_plus_plus import TVTopPlusPlus

            self.source = TVTopPlusPlus(
                log_path=self.log_path, refresh_seconds=self.refresh_seconds
            )
        # The investigation walks /proc and stats a pile of files, which
        # is far too much to do at every refresh. Cached, and refreshed
        # on the interval below -- a run that has been dead for a week is
        # not going to come back to life between two frames.
        self._stale_report: Any = None
        self._stale_checked: float = 0.0

    #: How often the stale-log investigation is redone, in seconds. Long,
    #: because its answer changes on the timescale of a training run
    #: starting, not on the timescale of a frame.
    STALE_RECHECK_SECONDS = 30.0

    def stale_report(self, *, now: float | None = None):
        """The cached stale-log investigation, refreshed on its own clock.

        ``None`` when neither :attr:`stale_after` nor
        :attr:`always_find_run` asked for one — the check costs a /proc
        walk and nobody should pay for it by default.
        """
        if not self.stale_after and not self.always_find_run:
            return None
        import time as _time

        now = now if now is not None else _time.monotonic()
        if (self._stale_report is not None
                and now - self._stale_checked < self.STALE_RECHECK_SECONDS):
            return self._stale_report
        from .stale_log import DEFAULT_STALE_SECONDS, investigate

        self._stale_report = investigate(
            self.log_path,
            threshold_seconds=self.stale_after or DEFAULT_STALE_SECONDS,
            force=self.always_find_run,
        )
        self._stale_checked = now
        return self._stale_report

    # -- pieces ---------------------------------------------------------

    def latest_frame(self):
        return self.source.latest_frame()

    @staticmethod
    def _cells(markup: str) -> int:
        """Printed width of some Rich markup.

        Measured rather than counted. The strings here carry colour tags
        that print as nothing and braille that prints as one cell each,
        so ``len()`` is wrong in both directions and a right border
        placed by ``len()`` lands in a different column on every row --
        which is exactly what a btop-alike must not look like.
        """
        from rich.text import Text

        return Text.from_markup(markup).cell_len

    def _row(self, markup: str, width: int) -> str:
        """One bordered line, padded *or truncated* so the edge lines up.

        Truncation is not the unlikely half. Every label here fits at
        100 columns and several do not at 60 -- and an over-long row does
        not wrap tidily, it pushes the right border onto the next line
        and every box below it looks broken. Rich does the cutting,
        because a naive slice of markup can cut a colour tag in two.
        """
        from rich.text import Text

        line = self.theme["div_line"]
        inner = max(0, width - 4)
        text = Text.from_markup(markup)
        if text.cell_len > inner:
            text.truncate(inner, overflow="ellipsis")
            markup = text.markup
        pad = max(0, inner - text.cell_len)
        return f"[{line}]│[/] {markup}{' ' * pad} [{line}]│[/]"

    def _graph_rows(self, history: list[float], width: int, height: int,
                    prefix: str) -> list[str]:
        """A braille graph, each row coloured by its own height.

        btop's graphs ramp bottom-to-top, so the ramp is indexed by row
        rather than by value — the top of a busy graph is red because it
        is the top, not because the latest sample was high.
        """
        rows = braille_graph(history, width, height)
        ramp = self.theme.ramp(prefix, max(height, 1))
        # ramp[0] is the cool end and row 0 is the top of the graph.
        return [f"[{ramp[height - 1 - i]}]{row}[/]" for i, row in enumerate(rows)]

    def _core_grid(self, per_core: list[float], width: int) -> list[str]:
        """Per-core meters, in as many columns as fit.

        btop lays cores out in columns and so does this; a 64-core
        machine in one column is a dashboard nobody can read.
        """
        if not per_core:
            return []
        cell = 20
        columns = max(1, min(4, width // cell))
        rows: list[str] = []
        ramp = self.theme.ramp("cpu", 8)
        per_row = -(-len(per_core) // columns)
        for row in range(per_row):
            parts = []
            for column in range(columns):
                index = column * per_row + row
                if index >= len(per_core):
                    continue
                value = per_core[index] or 0.0
                parts.append(
                    f"[{self.theme['inactive_fg']}]{index:>3}[/] "
                    f"{meter(value / 100.0, 8, ramp, empty=self.theme['meter_bg'])} "
                    f"[{self.theme['main_fg']}]{value:3.0f}%[/]"
                )
            rows.append("  ".join(parts))
        return rows

    def _panel_rows(self, panel, width: int, number: str) -> list[str]:
        """One module's :class:`~...tvtoppro_modules.Panel`, drawn.

        The module supplied numbers; every colour, meter and box
        character here is the active theme's. That split is why a module
        written against one theme looks right under all six.
        """
        theme = self.theme
        inner = width - 4
        out = [box_top(panel.title, width, line=theme["proc_box"],
                       accent=theme["title"], number=number)]
        if panel.error:
            out.append(self._row(
                f"[{theme['temp_end']}]! {panel.error}[/]", width,
            ))
        if not panel.readings:
            out.append(self._row(
                f"[{theme['inactive_fg']}]{panel.note or 'no readings'}[/]", width,
            ))
            out.append(box_bottom(width, line=theme["proc_box"]))
            return out

        label_width = min(12, max((len(r.label) for r in panel.readings), default=6))
        for reading in panel.readings:
            label = f"[{theme['inactive_fg']}]{reading.label[:label_width]:>{label_width}}[/]"
            if reading.fraction is None:
                out.append(self._row(
                    f"{label}  [{theme['main_fg']}]{reading.value}[/]", width,
                ))
            else:
                bar = max(8, inner - label_width - len(reading.value) - 6)
                out.append(self._row(
                    f"{label} "
                    f"{meter(reading.fraction, bar, theme.ramp(reading.ramp, bar), empty=theme['meter_bg'])} "
                    f"[{theme['main_fg']}]{reading.value}[/]",
                    width,
                ))
            if reading.history:
                for row in self._graph_rows(reading.history, inner, 2, reading.ramp):
                    out.append(self._row(row, width))
        out.append(box_bottom(width, line=theme["proc_box"]))
        return out

    def _stale_rows(self, report, width: int, number: str) -> list[str]:
        """The "your log is dead, here is what is actually running" box.

        Deliberately loud. The whole failure being addressed is that a
        stale log looks exactly like a quiet one, so this box uses the
        temperature ramp's hot end for its headline -- the same colour
        the dashboard uses for a GPU about to throttle.
        """
        theme = self.theme
        out = [box_top("stale log", width, line=theme["temp_end"],
                       accent=theme["temp_end"], number=number)]
        out.append(self._row(
            f"[{theme['temp_end']}]![/] "
            f"[{theme['main_fg']}]{report.log_path}[/] "
            f"[{theme['inactive_fg']}]last written "
            f"{_fmt_duration(report.log_age_seconds)} ago[/]",
            width,
        ))
        process = report.process
        if process is None:
            out.append(self._row(
                f"[{theme['inactive_fg']}]and no Python process here is busy, so "
                f"nothing is training[/]",
                width,
            ))
        else:
            out.append(self._row(
                f"[{theme['hi_fg']}]busiest[/] "
                f"[{theme['main_fg']}]pid {process.pid}[/] "
                f"[{theme['graph_text']}]{process.cpu_percent:.0f}% cpu · "
                f"{process.memory_percent:.0f}% mem · "
                f"up {_fmt_duration(process.age_seconds)}[/]",
                width,
            ))
            out.append(self._row(
                f"[{theme['inactive_fg']}]{process.command}[/]", width,
            ))
            if report.cwd:
                out.append(self._row(
                    f"[{theme['inactive_fg']}]cwd[/] "
                    f"[{theme['graph_text']}]{report.cwd}[/]",
                    width,
                ))
            if report.suggested_log:
                out.append(self._row(
                    f"[{theme['hi_fg']}]writing[/] "
                    f"[{theme['main_fg']}]{report.suggested_log}[/]",
                    width,
                ))
                if report.step is not None:
                    bar = max(8, (width - 4) - 28)
                    out.append(self._row(
                        f"[{theme['hi_fg']}]step[/] "
                        f"{meter(report.progress, bar, theme.ramp('free', bar), empty=theme['meter_bg'])} "
                        f"[{theme['main_fg']}]{report.step}/{report.total_steps or '?'}[/]",
                        width,
                    ))
                    if report.loss is not None:
                        out.append(self._row(
                            f"[{theme['inactive_fg']}]loss[/] "
                            f"[{theme['main_fg']}]{report.loss:.4f}[/]",
                            width,
                        ))
                out.append(self._row(
                    f"[{theme['title']}]tvtoppro --log {report.suggested_log}[/]",
                    width,
                ))
        for note in report.notes[:3]:
            out.append(self._row(f"[{theme['inactive_fg']}]{note}[/]", width))
        out.append(box_bottom(width, line=theme["temp_end"]))
        return out

    def render(self, frame=None, width: int | None = None) -> str:
        """One whole screen as Rich markup.

        Returned as a string rather than printed so it can be diffed in a
        test, which is the only way a TUI's layout gets checked at all.
        """
        frame = frame if frame is not None else self.latest_frame()
        width = max(40, width or self.width or 100)
        theme = self.theme
        title = theme["title"]
        inner = width - 4          # what fits between "│ " and " │"
        graph_w = inner
        out: list[str] = []

        # -- cpu ---------------------------------------------------------
        out.append(box_top("cpu", width, line=theme["cpu_box"], accent=title,
                           number="1"))
        cpu = frame.cpu_percent or 0.0
        bar = max(8, inner - 22)
        out.append(self._row(
            f"[{theme['hi_fg']}]CPU[/] "
            f"{meter(cpu / 100.0, bar, theme.ramp('cpu', bar), empty=theme['meter_bg'])} "
            f"[{theme['main_fg']}]{cpu:5.1f}%[/] "
            f"[{theme['graph_text']}]{len(frame.cpu_per_core) or '?'}c[/]",
            width,
        ))
        for row in self._graph_rows(
            [v / 100.0 for v in frame.cpu_history], graph_w, 3, "cpu"
        ):
            out.append(self._row(row, width))
        for row in self._core_grid(frame.cpu_per_core, inner)[:4]:
            out.append(self._row(row, width))
        out.append(box_bottom(width, line=theme["cpu_box"]))

        # -- mem ---------------------------------------------------------
        out.append(box_top("mem", width, line=theme["mem_box"], accent=title,
                           number="2"))
        memory = frame.memory or {}
        ram = frame.ram_percent or 0.0
        bar = max(8, inner - 32)
        out.append(self._row(
            f"[{theme['hi_fg']}]RAM[/] "
            f"{meter(ram / 100.0, bar, theme.ramp('used', bar), empty=theme['meter_bg'])} "
            f"[{theme['main_fg']}]{ram:5.1f}%[/] "
            f"[{theme['graph_text']}]{_fmt_bytes(memory.get('used_bytes'))}/"
            f"{_fmt_bytes(memory.get('total_bytes'))}[/]",
            width,
        ))
        for label, key, prefix in (
            ("available", "available_bytes", "available"),
            ("cached", "cached_bytes", "free"),
        ):
            value = memory.get(key)
            total = memory.get("total_bytes") or 0
            fraction = (value / total) if (value and total) else 0.0
            out.append(self._row(
                f"[{theme['inactive_fg']}]{label:>9}[/] "
                f"{meter(fraction, bar - 6, theme.ramp(prefix, bar - 6), empty=theme['meter_bg'])} "
                f"[{theme['main_fg']}]{_fmt_bytes(value)}[/]",
                width,
            ))
        out.append(box_bottom(width, line=theme["mem_box"]))

        # -- gpu ---------------------------------------------------------
        out.append(box_top("gpu", width, line=theme["net_box"], accent=title,
                           number="3"))
        if frame.gpu_util_percent is None and not frame.gpu_name:
            out.append(self._row(
                f"[{theme['inactive_fg']}]no nvidia-smi here — nothing rather "
                f"than zeroes, which read as an idle GPU[/]",
                width,
            ))
        else:
            util = frame.gpu_util_percent or 0.0
            bar = max(8, inner - 32)
            out.append(self._row(
                f"[{theme['hi_fg']}]GPU[/] "
                f"{meter(util / 100.0, bar, theme.ramp('cpu', bar), empty=theme['meter_bg'])} "
                f"[{theme['main_fg']}]{util:5.1f}%[/] "
                f"[{theme['graph_text']}]{(frame.gpu_name or '')[:16]}[/]",
                width,
            ))
            used, total = frame.gpu_mem_used_mib, frame.gpu_mem_total_mib
            fraction = (used / total) if (used and total) else 0.0
            out.append(self._row(
                f"[{theme['inactive_fg']}]     vram[/] "
                f"{meter(fraction, bar - 6, theme.ramp('used', bar - 6), empty=theme['meter_bg'])} "
                f"[{theme['main_fg']}]{used or 0}/{total or 0} MiB[/]",
                width,
            ))
            if frame.gpu_temp_c is not None:
                temp = frame.gpu_temp_c
                out.append(self._row(
                    f"[{theme['inactive_fg']}]     temp[/] "
                    f"{meter(min(temp, 100) / 100.0, bar - 6, theme.ramp('temp', bar - 6), empty=theme['meter_bg'])} "
                    f"[{theme['main_fg']}]{temp:.0f}°C[/] "
                    f"[{theme['graph_text']}]{frame.gpu_power_w or 0:.0f}/"
                    f"{frame.gpu_power_limit_w or 0:.0f}W[/]",
                    width,
                ))
            for row in self._graph_rows(
                [v / 100.0 for v in frame.gpu_util_history], graph_w, 2, "cpu"
            ):
                out.append(self._row(row, width))
        out.append(box_bottom(width, line=theme["net_box"]))

        # -- training ----------------------------------------------------
        out.append(box_top("training", width, line=theme["proc_box"],
                           accent=title, number="4"))
        if not frame.has_training_data:
            out.append(self._row(
                f"[{theme['inactive_fg']}]no training log found — pass --log, "
                f"or start a run[/]",
                width,
            ))
        else:
            bar = max(8, inner - 26)
            out.append(self._row(
                f"[{theme['hi_fg']}]step[/] "
                f"{meter(frame.progress, bar, theme.ramp('free', bar), empty=theme['meter_bg'])} "
                f"[{theme['main_fg']}]{frame.step}/{frame.total_steps or '?'}[/]",
                width,
            ))
            loss = frame.loss if frame.loss is not None else float("nan")
            out.append(self._row(
                f"[{theme['inactive_fg']}]loss[/] "
                f"[{theme['main_fg']}]{loss:.4f}[/]  "
                f"[{theme['inactive_fg']}]lr[/] "
                f"[{theme['main_fg']}]{frame.lr or 0:.2e}[/]  "
                f"[{theme['inactive_fg']}]tput[/] "
                f"[{theme['main_fg']}]{frame.throughput or 0:.2f}/s[/]  "
                f"[{theme['inactive_fg']}]eta[/] "
                f"[{theme['main_fg']}]{_fmt_duration(frame.eta_seconds)}[/]",
                width,
            ))
            if frame.recent_losses:
                span = max(frame.recent_losses) - min(frame.recent_losses) or 1.0
                floor = min(frame.recent_losses)
                normalised = [(v - floor) / span for v in frame.recent_losses]
                for row in self._graph_rows(normalised, graph_w, 3, "free"):
                    out.append(self._row(row, width))
        out.append(box_bottom(width, line=theme["proc_box"]))

        # -- stale log ---------------------------------------------------
        # Directly under training, because it is the box that explains
        # the one above it. Further down and it reads as unrelated.
        stale = self.stale_report()
        next_number = 5
        if stale is not None and stale.stale:
            out.extend(self._stale_rows(stale, width, str(next_number)))
            next_number += 1

        # -- modules -----------------------------------------------------
        if self.modules:
            from .tvtoppro_modules import poll_all

            for panel in poll_all(self.modules):
                out.extend(self._panel_rows(panel, width, str(next_number)))
                next_number += 1

        # -- processes ---------------------------------------------------
        if self.show_processes:
            out.append(box_top("proc", width, line=theme["proc_box"],
                               accent=title, number=str(next_number)))
            out.append(self._row(
                f"[{theme['title']}]{'pid':>7} {'user':<10} {'cpu%':>6} "
                f"{'mem%':>6}  command[/]",
                width,
            ))
            for process in self.source._get_active_processes():  # noqa: SLF001
                command = str(process["cmd"])[:max(0, inner - 34)]
                out.append(self._row(
                    f"[{theme['main_fg']}]{process['pid']:>7}[/] "
                    f"[{theme['inactive_fg']}]{str(process['user'])[:10]:<10}[/] "
                    f"[{theme['hi_fg']}]{process['cpu']:>6.1f}[/] "
                    f"[{theme['proc_misc']}]{process['mem']:>6.1f}[/]  "
                    f"[{theme['graph_text']}]{command}[/]",
                    width,
                ))
            out.append(box_bottom(width, line=theme["proc_box"]))

        out.append(
            f"[{theme['inactive_fg']}]tvtoppro[/] "
            f"[{theme['graph_text']}]theme {theme.name} · "
            f"up {_fmt_duration(frame.elapsed_seconds)} · ctrl-c to quit[/]"
        )
        return "\n".join(out)

    def run(self) -> None:
        """Draw until interrupted."""
        from rich.console import Console
        from rich.live import Live
        from rich.text import Text

        console = Console(force_terminal=True, width=self.width)
        try:
            with Live(console=console, refresh_per_second=max(1, int(1 / self.refresh_seconds)),
                      screen=True) as live:
                import time

                while True:
                    markup = self.render(width=self.width or console.width)
                    live.update(Text.from_markup(markup))
                    time.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


#: ``--stale-after`` suffixes. Seconds when bare, because that is what
#: every other duration in this package takes and a flag that silently
#: meant minutes would be the sort of thing found out a week later.
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _duration(text: str) -> float:
    """``"7d"`` to seconds. ``"off"``/``"0"``/``""`` disable, as 0.0."""
    value = str(text or "").strip().lower()
    if value in ("", "off", "none", "never", "0"):
        return 0.0
    scale = _DURATION_UNITS.get(value[-1:], 0)
    number, scale = (value[:-1], scale) if scale else (value, 1)
    try:
        seconds = float(number) * scale
    except ValueError:
        raise ValueError(
            f"{text!r} is not a duration. Use a number of seconds, or a "
            f"suffix: {', '.join(sorted(_DURATION_UNITS))}. 'off' disables it."
        ) from None
    if seconds < 0:
        raise ValueError(f"{text!r} is negative; a duration cannot be.")
    return seconds


def play_intro(theme: Theme, *, enabled: bool = True) -> None:
    """The ``tvtop-older`` opening: the title decoding out of noise.

    Silent on a non-TTY without being asked. An animation written into a
    CI log is a few hundred lines of carriage returns and escape codes,
    and the thing a CI log is for is reading afterwards.

    Wrapped in a bare ``except`` on purpose. This is the first thing
    ``tvtoppro`` does, and there is no failure of a decorative animation
    that should stop a monitoring tool from starting -- a terminal that
    cannot render braille, a closed stdout, an import that is not there.
    """
    if not enabled:
        return
    try:
        from ..timing.spinner import anime_print

        anime_print("tvtoppro", style="decode", delay=0.035)
    except Exception:  # noqa: BLE001 - see docstring
        logger = __import__("logging").getLogger(__name__)
        logger.debug("tvtoppro: intro animation failed", exc_info=True)


def _find_log(path: str | None, *, spinner: bool) -> Path | None:
    """The log to tail, with a spinner over the part that takes a moment.

    Autodetection walks the working tree looking for a file with
    ``step N/M loss=`` lines in it, which on a large repository is not
    instant. Without something on screen it looks like a hang, which is
    the specific reason the spinner is here rather than for decoration.
    """
    if path:
        return Path(path)
    from .tv import _autodetect_log

    if not spinner:
        return _autodetect_log()
    try:
        from ..timing.spinner import Spinner

        with Spinner("Looking for a training log", style="dots"):
            return _autodetect_log()
    except Exception:  # noqa: BLE001 - the search must happen either way
        return _autodetect_log()


def _list_modules() -> int:
    """``--list-modules``, including what failed to load and why."""
    from .tvtoppro_modules import (
        ENTRY_POINT_GROUP,
        Module,
        discover,
        user_module_dir,
    )

    found = discover()
    if not found:
        print("No modules found at all, which should not happen — the "
              "built-ins are part of the package.")
        return 1

    for name, module in sorted(found.items()):
        if isinstance(module, str):
            print(f"  {name:12} [failed to load] {module}")
            continue
        try:
            usable = module.available()
            reason = "" if usable else module.unavailable_reason()
        except Exception as exc:  # noqa: BLE001
            usable, reason = False, f"available() raised {type(exc).__name__}: {exc}"
        mark = " " if usable else "-"
        note = f"  ({reason})" if reason else ""
        print(f"{mark} {name:12} {module.description or module.title}{note}")
    print()
    print("  --modules disk,net        add those panels")
    print("  --modules all             every one marked available")
    print()
    print(f"  Drop a .py file in {user_module_dir()} to add your own,")
    print(f"  or ship one with a {ENTRY_POINT_GROUP!r} entry point.")
    print(f"  See {Module.__module__} for the three classes involved.")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tvtoppro",
        description=(
            "tvtop++'s stats, presented like btop++, with themes. "
            "Not built on cctvtop."
        ),
    )
    parser.add_argument("--log", default=None,
                        help="Training log to tail. Autodetected when omitted.")
    parser.add_argument("--theme", default=None,
                        help="Built-in name, or a path to a btop .theme or JSON file.")
    parser.add_argument("--list-themes", action="store_true")
    parser.add_argument("--dump-theme", action="store_true",
                        help="Print the active theme as a btop .theme file.")
    parser.add_argument("--refresh", type=float, default=1.0,
                        help="Seconds between frames.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--no-processes", dest="processes",
                        action="store_false", default=True)
    parser.add_argument("--once", action="store_true",
                        help="Draw one frame and exit. For scripts and screenshots.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="With --once, emit the frame's numbers instead.")
    parser.add_argument("--no-intro", dest="intro", action="store_false",
                        default=True,
                        help="Skip the startup animation and spinner. Implied "
                             "on a non-TTY.")
    parser.add_argument("--modules", default=None, metavar="A,B,C",
                        help="Extra stat panels. 'all' for everything this "
                             "machine supports; --list-modules to see them.")
    parser.add_argument("--list-modules", action="store_true",
                        help="Show every module found, including ones that "
                             "failed to load and why.")
    parser.add_argument(
        "--stale-after", default="7d", metavar="DURATION",
        help="Treat the log as dead after this long without a write, and "
             "show what is actually running instead. Accepts 30m, 6h, 7d. "
             "'off' disables the check. Default: 7d.",
    )
    parser.add_argument(
        "--find-run", action="store_true",
        help="Find the busiest Python process and the log it is writing, "
             "whatever the age of --log. Print it and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_modules:
        return _list_modules()

    if args.list_themes:
        for name, theme in sorted(THEMES.items()):
            print(f"  {name:14} {theme['cpu_start']} -> {theme['cpu_mid']} "
                  f"-> {theme['cpu_end']}")
        print()
        print("  Any btop .theme file works too: --theme ~/.config/btop/themes/x.theme")
        return 0

    try:
        theme = load_theme(args.theme)
    except (ValueError, OSError) as exc:
        print(f"tvtoppro: {exc}", file=__import__("sys").stderr)
        return 2

    if args.dump_theme:
        print(theme.to_btop(), end="")
        return 0

    import sys

    try:
        stale_after = _duration(args.stale_after)
    except ValueError as exc:
        print(f"tvtoppro: {exc}", file=sys.stderr)
        return 2

    modules, problems = [], []
    if args.modules:
        from .tvtoppro_modules import load_modules

        modules, problems = load_modules(args.modules)
        for name, reason in problems:
            # A warning, not a failure. One bad name in
            # `--modules disk,nte,swap` should cost the typo, not the
            # other two panels and the dashboard.
            print(f"tvtoppro: module {name!r}: {reason}", file=sys.stderr)

    # An animation only makes sense when a human is watching it render.
    interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
    intro = args.intro and interactive and not args.as_json and not args.find_run

    play_intro(theme, enabled=intro)
    log = _find_log(args.log, spinner=intro)

    if args.find_run:
        from .stale_log import investigate

        report = investigate(log, threshold_seconds=stale_after or 0, force=True)
        if args.as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.describe())
        # Non-zero when nothing was found, so a script can gate on it.
        return 0 if report.found_something else 1

    dashboard = TvTopPro(
        log_path=log, refresh_seconds=args.refresh, theme=theme,
        width=args.width, show_processes=args.processes,
        modules=modules, stale_after=stale_after,
    )

    if args.once:
        frame = dashboard.latest_frame()
        if args.as_json:
            from dataclasses import asdict

            print(json.dumps(asdict(frame), indent=2, default=str))
            return 0
        from rich.console import Console
        from rich.text import Text

        console = Console(width=args.width)
        console.print(Text.from_markup(
            dashboard.render(frame, width=args.width or console.width)
        ))
        return 0

    dashboard.run()
    return 0


def cli_main() -> None:
    """Console-script entry point.

    Without the ``__main__`` guard below, ``python -m`` on this module
    imports it, runs nothing and exits 0 — which looks exactly like a
    dashboard that drew an empty screen.
    """
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
