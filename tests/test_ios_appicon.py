"""The app icon is the brand mark, and stays the brand mark.

``ios/scripts/make_appicon.py`` re-draws the mark as polygons rather
than rasterising the SVG, because the shapes are four points each and an
SVG renderer would be a far larger dependency than the twelve points it
would draw. The cost of that choice is a second copy of the geometry,
and a second copy goes stale silently — the icon would keep building,
keep looking plausible, and quietly stop being the logo.

So the coordinates and the colours are checked against
``assets/logo-new/hypernix-icon.svg`` here. If the mark is redrawn, this
fails and names what moved.

The brand notes (``assets/logo-new/README.md``) also state two rules
worth pinning, because both are the kind of thing a later "small tidy"
breaks: the bottom-to-top dark-to-red progression is the whole idea, and
the red top bar is the one constant across every variant.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "logo-new" / "hypernix-icon.svg"
ICONSET = (
    ROOT / "ios" / "HyperLink" / "Resources" / "Assets.xcassets" / "AppIcon.appiconset"
)


def _svg_paths() -> list[tuple[tuple[float, float], ...]]:
    """The polygons the SVG draws, in file order (bottom bar first)."""
    text = SVG.read_text(encoding="utf-8")
    shapes = []
    for d in re.findall(r'\sd="([^"]+)"', text):
        points = re.findall(r"(-?[\d.]+),(-?[\d.]+)", d)
        shapes.append(tuple((float(x), float(y)) for x, y in points))
    return shapes


def _svg_fills() -> list[str]:
    return [f.lower() for f in re.findall(r'fill="(#[0-9a-fA-F]{6})"', SVG.read_text(encoding="utf-8"))]


def _make_appicon():
    """The generator module, imported by path.

    No ``importorskip`` here on purpose. It imports Pillow inside its
    drawing functions rather than at the top, precisely so the geometry
    check below runs on a machine without Pillow — which is what CI is.
    A top-level import made the most important assertion in this file
    the one that got skipped, and it skipped straight past a release.
    """
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "make_appicon", ROOT / "ios" / "scripts" / "make_appicon.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheGeometryHasNotDrifted:
    def test_the_script_draws_the_svg_shapes(self):
        assert list(_make_appicon().BARS) == _svg_paths()

    def test_it_needs_no_pillow_to_be_asked(self):
        """So the check above cannot quietly become a skip again."""
        import builtins

        real = builtins.__import__

        def without_pillow(name, *args, **kwargs):
            if name.split(".")[0] == "PIL":
                raise ImportError("Pillow is not installed")
            return real(name, *args, **kwargs)

        builtins.__import__ = without_pillow
        try:
            assert len(_make_appicon().BARS) == 3
        finally:
            builtins.__import__ = real

    def test_the_bars_are_ordered_bottom_to_top(self):
        """The dark-to-red progression up the stack is the mark's whole
        idea; reordering the bars is called out in the brand notes as
        the thing not to do."""
        bars = _svg_paths()
        lowest_y = [max(y for _, y in bar) for bar in bars]

        assert lowest_y == sorted(lowest_y, reverse=True)

    def test_the_top_bar_is_the_accent_red(self):
        """The one constant across every variant of the mark."""
        assert _svg_fills()[-1] == "#c8192e"


class TestTheBuiltIcons:
    @pytest.mark.parametrize(
        "name", ["AppIcon-1024.png", "AppIcon-1024-dark.png", "AppIcon-1024-tinted.png"]
    )
    def test_each_variant_exists_at_1024(self, name):
        pytest.importorskip("PIL")
        from PIL import Image

        with Image.open(ICONSET / name) as image:
            assert image.size == (1024, 1024)

    def test_the_default_icon_is_opaque(self):
        """A transparent default icon renders black on the home screen."""
        pytest.importorskip("PIL")
        from PIL import Image

        with Image.open(ICONSET / "AppIcon-1024.png") as image:
            assert image.mode == "RGB"

    def test_the_tinted_variant_keeps_its_transparency(self):
        """iOS fills the transparent area with the dark end of the
        user's tint. An opaque ground here flattens that."""
        pytest.importorskip("PIL")
        from PIL import Image

        with Image.open(ICONSET / "AppIcon-1024-tinted.png") as image:
            assert image.mode == "RGBA"
            assert image.getchannel("A").getextrema()[0] == 0

    def test_the_tinted_variant_is_greyscale(self):
        """Colour left here fights the tint iOS applies over it."""
        pytest.importorskip("PIL")
        from PIL import Image

        with Image.open(ICONSET / "AppIcon-1024-tinted.png") as image:
            colours = image.convert("RGBA").getcolors(maxcolors=1 << 20)

        assert colours, "more distinct colours than a greyscale icon can have"
        for _count, (r, g, b, alpha) in colours:
            if alpha:
                assert r == g == b, f"coloured pixel {(r, g, b)} in the tinted icon"

    def test_no_corner_rounding_of_its_own(self):
        """iOS applies its own mask. An icon that arrives pre-rounded
        shows dark wedges in the corners on the home screen."""
        pytest.importorskip("PIL")
        from PIL import Image

        with Image.open(ICONSET / "AppIcon-1024.png") as image:
            corners = [image.getpixel(p) for p in ((0, 0), (1023, 0), (0, 1023), (1023, 1023))]

        assert all(sum(c) > 0 for c in corners), "a corner is pure black — pre-masked?"


class TestTheCommittedPNGsAreCurrent:
    """The PNGs are build output committed to the repo, so they can go
    stale against the script that makes them and nothing would say so.

    Compared pixel by pixel rather than byte by byte: PNG encoding is
    not stable across Pillow versions, and a test that fails on a
    dependency bump is a test people learn to ignore.
    """

    def test_rerunning_the_script_would_change_nothing(self):
        pytest.importorskip("PIL")
        from PIL import Image, ImageChops

        make = _make_appicon()
        mark = make._draw_mark(make.BAR_COLOURS)

        expected = {}
        for name, ground in (
            ("AppIcon-1024.png", make.LIGHT_GROUND),
            ("AppIcon-1024-dark.png", make.DARK_GROUND),
        ):
            icon = make._ground(*ground)
            icon.alpha_composite(mark)
            expected[name] = icon.convert("RGB")
        expected["AppIcon-1024-tinted.png"] = make._draw_mark(
            tuple((v, v, v) for v in make.TINTED_VALUES)
        )

        for name, fresh in expected.items():
            with Image.open(ICONSET / name) as committed:
                committed = committed.convert(fresh.mode)
                difference = ImageChops.difference(committed, fresh)
            assert difference.getbbox() is None, (
                f"{name} does not match what make_appicon.py would write now — "
                f"rerun `python ios/scripts/make_appicon.py`"
            )


class TestTheAppMatchesItsOwnIcon:
    def test_the_accent_colour_is_the_brand_red(self):
        """The tint on every button in the app, and the top bar of the
        icon on the home screen, are the same red. They were not: the
        accent was a blue left over from before the mark existed, so the
        app opened looking like a different product than its icon."""
        import json

        catalog = json.loads(
            (
                ICONSET.parent / "AccentColor.colorset" / "Contents.json"
            ).read_text(encoding="utf-8")
        )
        default = next(
            c for c in catalog["colors"] if not c.get("appearances")
        )["color"]["components"]
        rgb = tuple(round(float(default[k]) * 255) for k in ("red", "green", "blue"))

        assert "#{:02x}{:02x}{:02x}".format(*rgb) == _svg_fills()[-1]

    def test_the_dark_variant_is_still_recognisably_that_red(self):
        """Lifted for contrast on a dark background, not re-hued."""
        import json

        catalog = json.loads(
            (
                ICONSET.parent / "AccentColor.colorset" / "Contents.json"
            ).read_text(encoding="utf-8")
        )
        dark = next(
            c for c in catalog["colors"] if c.get("appearances")
        )["color"]["components"]
        red, green, blue = (float(dark[k]) for k in ("red", "green", "blue"))

        assert red > 0.8
        assert red > green * 2 and red > blue * 2


class TestTheCatalogAgrees:
    def test_every_declared_file_is_present(self):
        import json

        catalog = json.loads((ICONSET / "Contents.json").read_text(encoding="utf-8"))
        for entry in catalog["images"]:
            assert (ICONSET / entry["filename"]).exists(), entry["filename"]

    def test_all_three_appearances_are_declared(self):
        """iOS 18 asks for default, dark and tinted. A missing tinted
        variant is not an error -- the system generates one -- but the
        generated one loses the red bar, which is the half that makes
        the mark recognisable."""
        import json

        catalog = json.loads((ICONSET / "Contents.json").read_text(encoding="utf-8"))
        appearances = set()
        for entry in catalog["images"]:
            values = [a["value"] for a in entry.get("appearances", [])]
            appearances.add(values[0] if values else "any")

        assert appearances == {"any", "dark", "tinted"}
