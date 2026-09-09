#!/usr/bin/env python3
"""Build HyperLink's app icon from the HyperNix mark.

Committed as a script rather than as three mystery PNGs. An icon that
arrives as a binary with no recipe cannot be adjusted by whoever comes
next: they can only replace it, and the brand drifts one replacement at
a time. Run this and the PNGs regenerate byte-identically.

    python ios/scripts/make_appicon.py

The geometry is the mark's own, read from ``assets/logo-new`` — three
equal parallelograms staggered up and to the right, dark to red. They
are re-drawn here as polygons rather than rasterised from the SVG,
because the shapes are four points each and pulling in an SVG renderer
to draw twelve points would be the larger dependency by far. The
coordinates below are the ``d`` attributes of ``hypernix-icon.svg``,
verified against that file by ``tests/test_ios_appicon.py`` — so if the
mark ever changes, the test fails rather than the icon quietly going
stale.

Colours follow ``assets/logo-new/README.md``: the on-dark variant, since
an app icon on a home screen is closer to that case than to letterhead.
The red top bar is the one constant the brand doc asks never to change,
and it does not change here either — not between the light and dark
appearances, and in the tinted appearance it becomes the *brightest*
value rather than being dropped, because it is the bar that makes the
mark read as HyperNix.

Three variants, because iOS 18 asks for three:

* **any** — the default. Full-bleed and opaque; iOS applies its own
  rounded-rectangle mask, so this file must be a plain square with no
  corner rounding of its own (rounding it here produces dark wedges in
  the corners on the home screen).
* **dark** — the same mark on a deeper ground.
* **tinted** — greyscale on transparency. iOS maps this through the
  user's chosen tint and fills the transparent area with the dark end of
  it, so colour left here fights the tint and an opaque ground flattens
  it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MARK_SVG = ROOT / "assets" / "logo-new" / "hypernix-icon.svg"
OUT = ROOT / "ios" / "HyperLink" / "Resources" / "Assets.xcassets" / "AppIcon.appiconset"

SIZE = 1024
#: The mark's own coordinate space, from the SVG's viewBox.
VIEWBOX = 64.0
#: Fraction of the canvas the 64x64 viewBox maps onto. The art inside it
#: spans 6..54 horizontally, so it already carries its own margin; 0.76
#: here puts the drawn shapes at about 66% of the icon, which is where
#: Apple's own optical margins sit.
SCALE = 0.76

#: The three bars, bottom first, exactly as `hypernix-icon.svg` draws
#: them. Bottom-to-top dark-to-red is the whole idea of the mark, per
#: the brand notes; do not reorder these or recolour the top one.
BARS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((6, 52), (16, 40), (42, 40), (32, 52)),
    ((12, 34), (22, 22), (48, 22), (38, 34)),
    ((18, 16), (28, 4), (54, 4), (44, 16)),
)

#: The on-dark palette from assets/logo-new/README.md. An app icon lives
#: on a home screen, which is the dark-backdrop case that variant is for.
OFF_WHITE = (242, 242, 240)
LIGHT_GREY = (143, 143, 143)
ACCENT_RED = (200, 25, 46)
BAR_COLOURS = (OFF_WHITE, LIGHT_GREY, ACCENT_RED)

#: The ground. Near-black, matching the site, with a slight lift toward
#: the top so the icon is not a flat rectangle.
LIGHT_GROUND = ((38, 40, 44), (14, 15, 17))
DARK_GROUND = ((24, 25, 28), (7, 8, 9))

#: Greyscale for the tinted appearance. The middle bar sits at 72% of the
#: outer ones, which is the relationship `hypernix-icon-mono.svg` uses to
#: keep the layers legible in a single ink; the red bar becomes the
#: brightest value so it survives the tint.
TINTED_VALUES = (200, 144, 255)

#: Drawn at this multiple and downscaled. Pillow's polygon fill has no
#: antialiasing of its own, and these are all diagonals.
SUPERSAMPLE = 4

# Pillow is imported inside the drawing functions, not at the top. The
# geometry above is what tests/test_ios_appicon.py compares against
# hypernix-icon.svg, and that check is the one worth running everywhere
# -- CI installs the package without Pillow, and a top-level import made
# the most important assertion in that file the one that got skipped.


def _ground(top: tuple[int, int, int], bottom: tuple[int, int, int]):
    """A vertical gradient, drawn as a 1px column and stretched.

    Cheaper than per-pixel and — more usefully — exactly reproducible, so
    rerunning this script does not produce a diff.
    """
    from PIL import Image

    column = Image.new("RGB", (1, SIZE))
    pixels = column.load()
    for y in range(SIZE):
        t = y / (SIZE - 1)
        pixels[0, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return column.resize((SIZE, SIZE), Image.BILINEAR).convert("RGBA")


def _draw_mark(colours):
    """The three bars on transparency, at icon size."""
    from PIL import Image, ImageDraw

    big = SIZE * SUPERSAMPLE
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)

    span = big * SCALE
    offset = (big - span) / 2
    for bar, colour in zip(BARS, colours, strict=True):
        pen.polygon(
            [(offset + x / VIEWBOX * span, offset + y / VIEWBOX * span) for x, y in bar],
            fill=(*colour, 255),
        )
    return layer.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> int:
    if not MARK_SVG.exists():
        print(f"make_appicon: no source mark at {MARK_SVG}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    mark = _draw_mark(BAR_COLOURS)
    written = []
    for name, ground in (
        ("AppIcon-1024.png", LIGHT_GROUND),
        ("AppIcon-1024-dark.png", DARK_GROUND),
    ):
        icon = _ground(*ground)
        icon.alpha_composite(mark)
        # Opaque: a transparent default icon renders black on the home
        # screen, which is a bug report waiting to happen.
        icon.convert("RGB").save(OUT / name, "PNG", optimize=True)
        written.append(name)

    tinted = _draw_mark(tuple((v, v, v) for v in TINTED_VALUES))
    tinted.save(OUT / "AppIcon-1024-tinted.png", "PNG", optimize=True)
    written.append("AppIcon-1024-tinted.png")

    for name in written:
        path = OUT / name
        print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
