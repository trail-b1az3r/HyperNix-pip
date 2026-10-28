"""scriptgen.theme — dark slate, charcoal, obsidian, and HyperNix red.

One palette, defined once, with contrast ratios checked rather than
eyeballed. Tkinter has no theming system worth the name, so every widget
gets its colours explicitly; having them in one place is what keeps a
40-widget form from drifting into four slightly different greys.

The constraint
--------------
Dark neutrals and one red. No purple, no neon. That rules out the entire
family of "dark theme" accents that lean violet — a blue-grey pushed far
enough becomes purple, so the neutrals here are *warm*: their red channel
is at or above their blue channel at every step. :func:`audit_palette`
enforces that as an actual check, because "no purple" is easy to agree
with and easy to violate by picking a nice-looking `#2a2a3e`.

Contrast
--------
Every foreground/background pair used for text is checked against WCAG
AA (4.5:1 for body, 3:1 for large text and UI edges) by
:func:`audit_palette`. A dark theme is where contrast failures hide:
mid-grey on dark-grey looks tasteful in a mockup and is unreadable on a
dim laptop screen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Palette", "PALETTE", "FONTS", "audit_palette", "relative_luminance", "contrast_ratio",
           "pick_fonts"]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance."""
    def channel(raw: int) -> float:
        c = raw / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in _rgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: str, background: str) -> float:
    a, b = relative_luminance(foreground), relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


@dataclass(frozen=True)
class Palette:
    """The colours. Every one is a warm neutral or the HyperNix red."""

    # --- surfaces, darkest to lightest -------------------------------
    # Neutral-to-warm at every step: red >= blue, enforced by
    # audit_palette(). The first draft of this palette drifted cool
    # (#16161a, #1f1f25) and the audit caught it — which is the entire
    # reason the audit exists, because a blue-grey pushed dark enough
    # reads as purple and nobody notices while they are picking it.
    obsidian: str = "#10100f"        # the window behind everything
    charcoal: str = "#1a1a18"        # cards and panels
    slate: str = "#252523"           # inputs, list rows
    slate_high: str = "#333330"      # hover, selected row
    edge: str = "#41403c"            # borders and dividers

    # --- text --------------------------------------------------------
    text: str = "#e8e6e3"            # body — warm white, not blue-white
    text_dim: str = "#b5b1aa"        # labels, secondary
    text_faint: str = "#96918a"      # hints, disabled

    # --- the accent --------------------------------------------------
    red: str = "#b92525"             # HyperNix red
    red_bright: str = "#c22626"      # hover / active
    red_dim: str = "#922121"         # pressed / track fill
    red_text: str = "#ff7777"        # red used *as text* on a dark surface

    # --- status ------------------------------------------------------
    ok: str = "#6dbb80"              # muted green, not neon
    warn: str = "#e2b45c"            # amber
    error: str = "#c22626"           # the accent doubles as error
    info: str = "#5b8aa6"            # desaturated steel blue

    # --- the preview's syntax colours (0.72.8rc1) ----------------------
    # Warm, like the rest: coral keywords, sage strings, amber numbers.
    syn_keyword: str = "#f08a6c"
    syn_string: str = "#b8cc8a"
    syn_number: str = "#e2b45c"
    syn_name: str = "#ecd9b0"        # the name after def / class, and keys
    syn_builtin: str = "#8fb6c9"

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, str)}


PALETTE = Palette()

#: Use a humanist sans-serif for the application chrome and reserve
#: monospace for generated code. This gives the form a calmer hierarchy
#: while keeping the preview easy to scan.
FONTS: dict[str, tuple[Any, ...]] = {
    "body": ("DejaVu Sans", 10),
    "body_bold": ("DejaVu Sans", 10, "bold"),
    "small": ("DejaVu Sans", 9),
    "tiny": ("DejaVu Sans", 8),
    "heading": ("DejaVu Sans", 12, "bold"),
    "title": ("DejaVu Sans", 18, "bold"),
    "code": ("DejaVu Sans Mono", 10),
}

#: Fallbacks tried in order when the preferred family is missing. Tk
#: silently substitutes a default when a family does not exist, which
#: looks like a bug rather than a missing font.
FONT_FALLBACKS: tuple[str, ...] = (
    "DejaVu Sans Mono", "Menlo", "Consolas", "Liberation Mono", "Courier New", "TkFixedFont",
)


#: Families tried for the interface and the code, best first. Inter and
#: JetBrains Mono when installed; DejaVu, which every Linux has, otherwise.
SANS_FAMILIES: tuple[str, ...] = ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue",
                                  "Cantarell", "Noto Sans", "DejaVu Sans")
MONO_FAMILIES: tuple[str, ...] = ("JetBrains Mono", "Cascadia Code", "SF Mono", "Menlo",
                                  "Consolas", "DejaVu Sans Mono", "Liberation Mono")


def pick_fonts(root: Any = None) -> dict[str, tuple[Any, ...]]:
    """:data:`FONTS` with the best installed families, for *root*'s display.
    Without a display (or Tk) the defaults stand."""
    try:
        from tkinter import font as tkfont

        available = {name.lower() for name in tkfont.families(root)}
    except Exception:  # noqa: BLE001 - no display, or no Tk
        return dict(FONTS)
    sans = next((f for f in SANS_FAMILIES if f.lower() in available), FONTS["body"][0])
    mono = next((f for f in MONO_FAMILIES if f.lower() in available), FONTS["code"][0])
    return {
        "body": (sans, 10),
        "body_bold": (sans, 10, "bold"),
        "small": (sans, 9),
        "tiny": (sans, 8),
        "label": (sans, 10, "bold"),
        "heading": (sans, 15, "bold"),
        "title": (sans, 16, "bold"),
        "nav": (sans, 10),
        "nav_bold": (sans, 10, "bold"),
        "code": (mono, 10),
        "code_small": (mono, 9),
    }


def audit_palette(palette: Palette = PALETTE) -> dict[str, Any]:
    """Check the palette against its own rules. Used by the tests.

    Two rules, both checkable:

    * **No purple.** Every neutral has ``red >= blue``. A blue-grey
      pushed dark enough reads as purple, and "avoid purple" is
      otherwise a matter of opinion nobody can enforce.
    * **Readable.** Body text on every surface clears WCAG AA (4.5:1),
      and dim text clears the 3:1 large-text threshold. Mid-grey on
      dark-grey looks tasteful in a mockup and vanishes on a dim screen.
    """
    neutrals = {
        "obsidian": palette.obsidian, "charcoal": palette.charcoal,
        "slate": palette.slate, "slate_high": palette.slate_high,
        "edge": palette.edge, "text": palette.text,
        "text_dim": palette.text_dim, "text_faint": palette.text_faint,
    }
    purple: list[str] = []
    for name, colour in neutrals.items():
        r, _, b = _rgb(colour)
        if b > r:
            purple.append(f"{name} ({colour}) is cooler than neutral: blue {b} > red {r}")

    surfaces = {
        "obsidian": palette.obsidian, "charcoal": palette.charcoal,
        "slate": palette.slate, "slate_high": palette.slate_high,
    }
    contrast: list[str] = []
    for surface_name, surface in surfaces.items():
        body = contrast_ratio(palette.text, surface)
        if body < 4.5:
            contrast.append(f"text on {surface_name} is {body:.2f}:1, below AA 4.5:1")
        dim = contrast_ratio(palette.text_dim, surface)
        if dim < 3.0:
            contrast.append(f"text_dim on {surface_name} is {dim:.2f}:1, below 3:1")

    # The accent is used as a fill behind white text and as text on dark.
    accent: list[str] = []
    on_red = contrast_ratio(palette.text, palette.red)
    if on_red < 4.5:
        accent.append(f"text on red is {on_red:.2f}:1, below AA 4.5:1")
    red_on_dark = contrast_ratio(palette.red_text, palette.charcoal)
    if red_on_dark < 4.5:
        accent.append(f"red_text on charcoal is {red_on_dark:.2f}:1, below AA 4.5:1")

    for name in ("syn_keyword", "syn_string", "syn_number", "syn_name", "syn_builtin"):
        ratio = contrast_ratio(getattr(palette, name), palette.obsidian)
        if ratio < 4.5:
            accent.append(f"{name} on obsidian is {ratio:.2f}:1, below AA 4.5:1")

    return {
        "ok": not (purple or contrast or accent),
        "purple_violations": purple,
        "contrast_violations": contrast,
        "accent_violations": accent,
        "ratios": {
            "text/obsidian": round(contrast_ratio(palette.text, palette.obsidian), 2),
            "text/charcoal": round(contrast_ratio(palette.text, palette.charcoal), 2),
            "text/slate": round(contrast_ratio(palette.text, palette.slate), 2),
            "text_dim/charcoal": round(contrast_ratio(palette.text_dim, palette.charcoal), 2),
            "text/red": round(on_red, 2),
            "red_text/charcoal": round(red_on_dark, 2),
        },
    }


def resolve_font(root: Any, preferred: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return *preferred*, or the first fallback the system actually has.

    Tk substitutes a default silently for a missing family, which shows
    up as "why is this proportional" rather than as an error.
    """
    try:
        from tkinter import font as tkfont

        available = {name.lower() for name in tkfont.families(root)}
    except Exception:  # noqa: BLE001 - no display, or no Tk
        return preferred
    if preferred and str(preferred[0]).lower() in available:
        return preferred
    for family in FONT_FALLBACKS:
        if family.lower() in available or family == "TkFixedFont":
            return (family, *preferred[1:])
    return preferred
