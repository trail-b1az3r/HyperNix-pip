"""The website on a phone.

Item 25: *fix the website's mobile view, since the sizing is off*. It
was, and driving the built site in a headless Chromium at 375x667 found
what and why:

**One grid track.** `repeat(auto-fit, minmax(480px, 1fr))` on the hero.
A `minmax` track cannot shrink below its minimum, so on a 375px phone
the column stayed 480px wide, and the section's `overflow: hidden`
clipped the right third of the hero — the subtitle cut off mid-word, the
third button gone, the terminal running off the edge.

The page never scrolled sideways, which is why it looked fine to a check
that only measured `documentElement.scrollWidth`. That is the trap worth
recording: a container wider than the viewport *inside* something that
hides overflow does not overflow the document. It just silently loses
the right side of itself.

**Tap targets.** Nav and footer links 15px tall, the burger and copy
buttons 22px, against Apple's 44px minimum and WCAG 2.2's 24px floor.
**Text** down to 9.5px.

Why these tests are static
--------------------------
The measurements above needed a browser and a build. Reproducing that in
this suite would mean npm, a Vite build and a Chromium download on every
run, for a page that changes far less often than the code around it.

So these check for the *class* of defect in the source instead: a
`minmax` minimum wide enough to break a phone, and the presence of the
touch-sizing rules. That is cheap, runs everywhere, and fails the moment
someone reintroduces the actual root cause — which is what a regression
test is for. The browser audit is in the commit message, where the
numbers belong.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
SRC = DOCS / "src"

#: The narrowest phone worth supporting. An iPhone SE is 375 CSS px, and
#: it is the width at which every "looks fine on mine" bug appears.
NARROWEST_PHONE = 375


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.tsx")) + sorted(SRC.rglob("*.css"))


def _without_comments(text: str) -> str:
    """Source with /* */ and // comments blanked out.

    Necessary, not fastidious: the comment next to the hero grid explains
    the bug by *quoting the broken value*, and a grep over raw source
    reads that as the bug still being present. The same trap has now
    caught three greps in this repository -- over ``cmd_index`` in
    bin/hypernix-t1, over Studio's ``ToolNames()``, and here -- each time
    because the comment that documents a rule names the thing the rule
    forbids.

    Blanked rather than deleted, so reported line numbers stay right.
    """
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                  text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


class TestNoGridTrackWiderThanAPhone:
    """The bug that was actually reported, and its whole family."""

    def test_no_minmax_minimum_exceeds_a_phone(self):
        offenders = []
        for source in _sources():
            text = _without_comments(source.read_text(encoding="utf-8"))
            for match in re.finditer(r"minmax\(\s*(\d+)px", text):
                width = int(match.group(1))
                if width >= NARROWEST_PHONE:
                    line = text[: match.start()].count("\n") + 1
                    offenders.append(f"{source.name}:{line} minmax({width}px…")

        assert not offenders, (
            "A minmax() track cannot shrink below its minimum, so on a "
            f"{NARROWEST_PHONE}px phone these columns stay wider than the "
            "screen — and inside an overflow:hidden ancestor they are "
            "clipped rather than scrolled, so nothing looks wrong until you "
            "read the page. Wrap the minimum in min(): "
            "minmax(min(480px,100%),1fr).\n  " + "\n  ".join(offenders)
        )

    def test_the_hero_uses_the_min_form(self):
        """Specifically, since this is the one that shipped broken."""
        home = (SRC / "pages" / "Home.tsx").read_text(encoding="utf-8")

        assert "minmax(min(480px,100%),1fr)" in home.replace(" ", "")

    def test_the_reason_is_written_down_next_to_it(self):
        """A `min()` wrapper looks like noise to whoever tidies it away."""
        home = (SRC / "pages" / "Home.tsx").read_text(encoding="utf-8")
        index = home.index("minmax(min(480px")
        preamble = home[max(0, index - 700) : index]

        assert "overflow" in preamble
        assert "shrink" in preamble


class TestTouchSizing:
    """The rules that raise 15px links and 22px buttons to a usable size."""

    @pytest.fixture(scope="class")
    def css(self) -> str:
        return (SRC / "index.css").read_text(encoding="utf-8")

    def test_there_is_a_coarse_pointer_block(self, css):
        """`pointer: coarse` rather than a width query: what makes a 15px
        link hard to hit is a finger, not a narrow window, and a desktop
        browser dragged narrow should keep its denser layout."""
        assert "@media (pointer: coarse)" in css

    def test_it_reaches_44px(self, css):
        block = css.split("@media (pointer: coarse)")[1]
        block = block[: block.index("\n}\n\n@media") if "\n}\n\n@media" in block else len(block)]

        assert "44px" in block, "Apple's minimum tap target is 44px"

    @pytest.mark.parametrize(
        "selector",
        [".footer-link", ".mob-btn", ".copy-btn", ".nav-drawer-link", ".tab-btn"],
    )
    def test_every_measured_offender_is_covered(self, css, selector):
        block = css.split("@media (pointer: coarse)")[1]

        assert selector in block, f"{selector} was measured too small and is not sized"

    def test_the_classes_exist_in_the_markup(self):
        """A rule for a selector nobody has is a rule that does nothing,
        and it reads as coverage. Two of these were written before the
        elements had the class."""
        markup = "\n".join(
            p.read_text(encoding="utf-8") for p in SRC.rglob("*.tsx")
        )
        for selector in ("footer-link", "copy-btn", "tab-btn", "filter-chip"):
            assert selector in markup, f"no element carries {selector!r}"

    def test_the_copy_button_keeps_its_size(self, css):
        """It sits over the code it offers to copy, so it is the one
        control that cannot simply be made bigger. Its hit area grows via
        a pseudo-element and the visible chip does not."""
        block = css.split("@media (pointer: coarse)")[1]

        assert ".copy-btn::after" in block
        assert ".press-btn:not(.copy-btn)" in block, (
            "without the :not(), .press-btn's min-height makes the chip 44px "
            "tall again and it covers the first line of the code block"
        )

    def test_inline_prose_links_are_deliberately_excluded(self, css):
        """Stretching a link inside a sentence to 44px breaks the line
        spacing around it, and WCAG 2.2 exempts inline targets for that
        reason. Recorded because the exclusion looks like an oversight."""
        block = css.split("@media (pointer: coarse)")[1]

        assert "inline" in block.lower()
        assert "WCAG" in block


class TestTextIsReadableOnAPhone:
    def test_nothing_is_set_to_ten_pixels(self):
        """9.5px and 10px labels are legible on a 27-inch monitor and not
        on a phone at arm's length. 11px is the floor here."""
        offenders = []
        for source in sorted(SRC.rglob("*.tsx")):
            text = _without_comments(source.read_text(encoding="utf-8"))
            for match in re.finditer(r"fontSize:\s*(\d+(?:\.\d+)?)\s*[,}]", text):
                size = float(match.group(1))
                if size < 11:
                    line = text[: match.start()].count("\n") + 1
                    offenders.append(f"{source.name}:{line} fontSize:{size}")

        assert not offenders, "text under 11px:\n  " + "\n  ".join(offenders)

    def test_the_terminal_scrolls_rather_than_wraps(self):
        """`.term-line` is `white-space: pre` deliberately — a wrapped
        shell command reads as two commands, which is worse than a
        scrollbar. So the block scrolls instead of being made to fit."""
        css = (SRC / "index.css").read_text(encoding="utf-8")
        narrow = css.split("@media (max-width: 640px)")[-1]

        assert "overflow-x: auto" in narrow
        assert "overscroll-behavior-inline" in narrow, (
            "without it, reaching the end of a command starts swiping the "
            "page sideways behind the code block"
        )


class TestTheViewportIsDeclared:
    def test_index_html_has_the_meta_tag(self):
        """Without it a phone renders at 980px and scales down, which
        makes every other fix here invisible."""
        html = (DOCS / "index.html").read_text(encoding="utf-8")

        assert 'name="viewport"' in html
        assert "width=device-width" in html
