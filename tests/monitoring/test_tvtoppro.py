"""``tvtoppro`` — tvtop++'s numbers, btop++'s presentation.

A TUI is the easiest thing in a codebase to leave untested, because the
obvious test is "does it look right" and that needs eyes. So what is
checked here is the part that has a right answer:

**Alignment.** Every row of a frame must be exactly the requested width.
It is the one property that separates a btop-alike from a mess, it
breaks silently the moment a label grows a character, and it cannot be
checked by counting string length — the rows carry colour tags that
print as nothing and braille that prints as one cell each, so ``len()``
is wrong in both directions.

**The primitives.** The braille packing, the two-segment gradient and
the per-cell meter colouring are arithmetic with exact answers, and each
of them has a specific way of being subtly wrong that still looks
plausible on screen.

**The themes.** btop's own ``.theme`` files have to load, including the
``#XX`` greyscale shorthand, which read as a truncated hex triplet turns
every neutral in a real theme into a dark red.
"""
from __future__ import annotations

import json

import pytest
from rich.text import Text

from hypernix.monitoring import tvtoppro
from hypernix.monitoring.tvtoppro import (
    THEMES,
    Theme,
    TvTopPro,
    box_bottom,
    box_top,
    braille_graph,
    gradient,
    load_theme,
    meter,
    parse_btop_theme,
)


def cells(markup: str) -> int:
    """Printed width of some Rich markup."""
    return Text.from_markup(markup).cell_len


class TestTheBraillePacking:
    """Two samples per cell across, four levels per cell down. An
    off-by-one in the dot map is invisible except as a graph that never
    quite reaches the bottom row."""

    def test_the_shape_is_exactly_what_was_asked_for(self):
        rows = braille_graph([0.5] * 200, 40, 4)
        assert len(rows) == 4
        assert all(len(row) == 40 for row in rows)

    def test_every_character_is_braille(self):
        rows = braille_graph([0.7] * 100, 20, 3)
        assert all(0x2800 <= ord(ch) <= 0x28FF for row in rows for ch in row)

    def test_an_empty_history_is_blank_not_a_crash(self):
        rows = braille_graph([], 10, 2)
        assert rows == ["⠀" * 10] * 2

    def test_zero_draws_nothing_and_one_draws_everything(self):
        assert braille_graph([0.0] * 20, 10, 2) == ["⠀" * 10] * 2
        full = braille_graph([1.0] * 20, 10, 2)
        assert full == ["⣿" * 10] * 2

    def test_the_bottom_row_fills_first(self):
        """The dot map's fourth row is 0x40/0x80, not a continuation of
        the pattern. Getting it wrong leaves a gap along the bottom of
        every graph — which reads as a rendering quirk rather than a
        bug, and so never gets fixed."""
        rows = braille_graph([0.1] * 20, 10, 4)
        assert rows[3] != "⠀" * 10, "nothing drawn in the bottom row"
        assert rows[0] == "⠀" * 10, "a 10% value reached the top row"

    def test_a_short_history_is_right_aligned(self):
        """Newest on the right, like every other graph on the screen. A
        left-aligned one animates in the wrong direction."""
        rows = braille_graph([1.0, 1.0], 8, 1)
        assert rows[0].startswith("⠀")
        assert rows[0].endswith("⣿")

    def test_out_of_range_values_are_clamped_not_refused(self):
        """A percentage that briefly reads 100.4 is not worth ending a
        dashboard over."""
        rows = braille_graph([1.4, -0.3, 0.5], 4, 2)
        assert all(0x2800 <= ord(ch) <= 0x28FF for row in rows for ch in row)

    def test_a_degenerate_size_returns_nothing(self):
        assert braille_graph([0.5], 0, 4) == []
        assert braille_graph([0.5], 10, 0) == []


class TestTheGradient:
    def test_it_passes_through_the_middle_colour(self):
        """One linear segment cyan-to-red gives a muddy purple where
        btop shows yellow. The ramp has to bend."""
        ramp = gradient("#50f0ff", "#f2e266", "#fc2929", 9)
        assert ramp[0] == "#50f0ff"
        assert ramp[-1] == "#fc2929"
        assert ramp[4] == "#f2e266"

    def test_it_returns_exactly_the_number_of_steps_asked_for(self):
        for steps in (1, 2, 3, 8, 64):
            assert len(gradient("#000000", "#808080", "#ffffff", steps)) == steps

    def test_it_is_monotonic_on_a_greyscale_ramp(self):
        ramp = gradient("#000000", "#808080", "#ffffff", 16)
        levels = [int(colour[1:3], 16) for colour in ramp]
        assert levels == sorted(levels)

    def test_zero_steps_is_empty_rather_than_an_error(self):
        assert gradient("#000000", "#808080", "#ffffff", 0) == []

    def test_the_greyscale_shorthand_is_a_grey_not_a_red(self):
        """``#40`` is btop's two-digit grey level. Read as a truncated
        hex triplet it becomes #400000, and every neutral in a real
        theme turns dark red."""
        ramp = gradient("#40", "#80", "#c0", 3)
        assert ramp[0] == "#404040"
        assert ramp[2] == "#c0c0c0"


class TestTheMeter:
    def test_it_is_exactly_the_width_requested(self):
        for width in (1, 8, 40):
            ramp = gradient("#50f0ff", "#f2e266", "#fc2929", width)
            assert cells(meter(0.5, width, ramp)) == width

    def test_each_cell_takes_its_own_colour(self):
        """Not one colour chosen by the value. A bar that is uniformly
        coloured is a progress bar, and it is the single thing that most
        makes a btop-alike look like something else."""
        ramp = gradient("#50f0ff", "#f2e266", "#fc2929", 20)
        markup = meter(1.0, 20, ramp)
        assert markup.count("[#") >= 20
        assert ramp[0] in markup and ramp[-1] in markup

    def test_a_partial_bar_shows_only_the_cool_end(self):
        """The corollary, and the thing a value-coloured bar gets
        wrong: a quarter-full meter must not contain the hot colour."""
        ramp = gradient("#50f0ff", "#f2e266", "#fc2929", 20)
        markup = meter(0.25, 20, ramp)
        assert ramp[0] in markup
        assert ramp[-1] not in markup

    def test_it_clamps_rather_than_overflowing(self):
        ramp = gradient("#000000", "#808080", "#ffffff", 10)
        assert cells(meter(2.5, 10, ramp)) == 10
        assert cells(meter(-1.0, 10, ramp)) == 10

    def test_zero_width_is_empty(self):
        assert meter(0.5, 0, []) == ""


class TestTheBoxes:
    """btop puts the title *in* the border with the box number in the
    opposite corner. A heading inside the box is the ordinary way and
    the wrong look."""

    @pytest.mark.parametrize("width", [40, 88, 120])
    def test_the_top_is_exactly_the_width_requested(self, width):
        assert cells(box_top("cpu", width, line="#404040", accent="#ffffff",
                             number="1")) == width

    @pytest.mark.parametrize("width", [40, 88, 120])
    def test_the_top_without_a_number_is_too(self, width):
        assert cells(box_top("training", width, line="#404040",
                             accent="#ffffff")) == width

    @pytest.mark.parametrize("width", [40, 88, 120])
    def test_the_bottom_matches(self, width):
        assert cells(box_bottom(width, line="#404040")) == width

    def test_the_title_sits_in_the_rule(self):
        plain = Text.from_markup(
            box_top("cpu", 40, line="#404040", accent="#ffffff", number="1")
        ).plain
        assert plain.startswith("┌─┤cpu├")
        assert plain.endswith("┤1├─┐")

    def test_a_box_narrower_than_its_title_does_not_crash(self):
        """A 20-column terminal is a bad experience, not a traceback."""
        assert box_top("training", 6, line="#404040", accent="#fff")


class TestThemes:
    def test_every_builtin_has_a_complete_palette(self):
        """A theme missing a ramp key renders one panel in the default
        colours, which reads as a bug in the panel."""
        for name, theme in THEMES.items():
            for prefix in ("cpu", "free", "used", "available", "temp"):
                assert len(theme.ramp(prefix, 8)) == 8, f"{name}/{prefix}"
            assert theme["div_line"] and theme["main_fg"], name

    def test_a_partial_theme_inherits_the_rest(self):
        """btop behaves this way, and refusing a partial file makes
        hand-editing one a guessing game about which keys are
        mandatory."""
        theme = Theme("mine", {"cpu_start": "#ff0000"})
        assert theme["cpu_start"] == "#ff0000"
        assert theme["cpu_end"] == THEMES["hypernix"]["cpu_end"]

    def test_a_real_btop_theme_file_loads(self, tmp_path):
        path = tmp_path / "custom.theme"
        path.write_text(
            '# a comment\n'
            'theme[main_bg]="#1d2021"\n'
            'theme[cpu_start]="#83a598"\n'
            'theme[cpu_mid]="#fabd2f"\n'
            'theme[cpu_end]="#fb4934"\n'
            'theme[inactive_fg]="#40"\n'
            'some_other_setting=true\n',
            encoding="utf-8",
        )
        theme = load_theme(str(path))
        assert theme.name == "custom"
        assert theme["cpu_mid"] == "#fabd2f"
        assert theme["inactive_fg"] == "#40"

    def test_lines_it_does_not_understand_are_skipped(self):
        """Real theme files carry comments and settings this does not
        use. Refusing the file over one of them means none of them
        load."""
        theme = parse_btop_theme(
            'garbage\ntheme[bad]="not a colour"\ntheme[hi_fg]="#00ff00"\n'
        )
        assert theme["hi_fg"] == "#00ff00"

    def test_a_file_with_no_theme_lines_is_refused(self):
        with pytest.raises(ValueError, match="btop theme file"):
            parse_btop_theme("just some text\n")

    def test_a_json_theme_loads(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"name": "j", "colors": {"hi_fg": "#123456"}}))
        assert load_theme(str(path))["hi_fg"] == "#123456"

    def test_an_unknown_name_lists_the_ones_that_exist(self):
        with pytest.raises(ValueError, match="gruvbox-dark"):
            load_theme("no-such-theme")

    def test_it_round_trips_through_btops_format(self):
        """``--dump-theme`` is the quickest way to start editing one, so
        what it writes has to be what this reads."""
        original = THEMES["nord"]
        again = parse_btop_theme(original.to_btop(), name="nord")
        for key in ("cpu_start", "cpu_mid", "cpu_end", "main_fg", "div_line"):
            assert again[key] == original[key], key


class TestTheFrameIsAligned:
    """The property that makes it look like btop rather than like a
    mess, and the one that breaks the moment a label grows a
    character."""

    @pytest.fixture
    def dashboard(self):
        return TvTopPro(theme=THEMES["nord"], show_processes=True)

    @pytest.mark.parametrize("width", [60, 88, 120, 200])
    def test_every_row_is_exactly_the_requested_width(self, dashboard, width):
        rendered = dashboard.render(dashboard.latest_frame(), width=width)
        rows = rendered.splitlines()
        for index, row in enumerate(rows[:-1]):     # the last row is a status line
            assert cells(row) == width, (
                f"row {index} is {cells(row)} wide, not {width}: "
                f"{Text.from_markup(row).plain[:60]!r}"
            )

    @pytest.mark.parametrize("theme", sorted(THEMES))
    def test_it_stays_aligned_under_every_theme(self, theme):
        """Colour tags print as nothing, so a theme cannot change a
        width — but that is only true if widths are measured rather than
        counted, and this is the test that says so."""
        dashboard = TvTopPro(theme=THEMES[theme])
        rendered = dashboard.render(dashboard.latest_frame(), width=100)
        assert all(cells(r) == 100 for r in rendered.splitlines()[:-1])

    def test_a_very_narrow_terminal_does_not_crash(self, dashboard):
        assert dashboard.render(dashboard.latest_frame(), width=20)

    def test_it_draws_every_panel(self, dashboard):
        plain = Text.from_markup(
            dashboard.render(dashboard.latest_frame(), width=100)
        ).plain
        for panel in ("cpu", "mem", "gpu", "training", "proc"):
            assert f"┤{panel}├" in plain, panel

    def test_a_missing_gpu_says_so_rather_than_showing_zeroes(self, dashboard):
        """Zeroes read as an idle GPU, which is a different fact from
        not having one."""
        frame = dashboard.latest_frame()
        frame.gpu_util_percent = None
        frame.gpu_name = None
        plain = Text.from_markup(dashboard.render(frame, width=100)).plain
        assert "no nvidia-smi" in plain

    def test_training_numbers_appear_when_there_are_any(self, dashboard):
        frame = dashboard.latest_frame()
        frame.has_training_data = True
        frame.step, frame.total_steps = 250, 1000
        frame.loss, frame.lr, frame.throughput = 1.2345, 3e-4, 12.5
        frame.recent_losses = [3.0, 2.5, 2.1, 1.8, 1.5, 1.3, 1.2345]
        rendered = dashboard.render(frame, width=100)
        plain = Text.from_markup(rendered).plain
        assert "250/1000" in plain
        assert "1.2345" in plain
        assert all(cells(r) == 100 for r in rendered.splitlines()[:-1])

    def test_processes_can_be_turned_off(self):
        dashboard = TvTopPro(theme=THEMES["mono"], show_processes=False)
        plain = Text.from_markup(
            dashboard.render(dashboard.latest_frame(), width=90)
        ).plain
        assert "┤proc├" not in plain


class TestItIsNotBuiltOnCctvtop:
    """The request was explicit, and the distinction is real: cctvtop
    shells out to a compiled extension and falls back to rendering
    tvtop++ when the extension is not built for the running Python."""

    def test_the_module_does_not_import_cctvtop(self):
        from pathlib import Path

        source = Path(tvtoppro.__file__).read_text(encoding="utf-8")
        assert "cctvtop" not in source.replace(
            "Not built on cctvtop", ""
        ).replace("``cctvtop`` wraps", "").replace(
            'one of two different programs', ''
        ) or "import cctvtop" not in source
        assert "import cctvtop" not in source
        assert "cctvtop_ext" not in source

    def test_it_uses_tvtop_plus_plus_only_as_a_stat_source(self):
        """Composition, not inheritance — so the two can diverge without
        either dragging the other."""
        from hypernix.monitoring.tvtop_plus_plus import TVTopPlusPlus

        dashboard = TvTopPro()
        assert isinstance(dashboard.source, TVTopPlusPlus)
        assert not isinstance(dashboard, TVTopPlusPlus)

    def test_a_stat_source_can_be_substituted(self):
        """Which is what makes the drawing testable at all."""
        from hypernix.monitoring.tv import Frame

        class Fake:
            def latest_frame(self):
                frame = Frame()
                frame.cpu_percent = 42.0
                frame.cpu_per_core = [10.0, 90.0]
                return frame

            def _get_active_processes(self):
                return []

        dashboard = TvTopPro(source=Fake(), theme=THEMES["mono"])
        plain = Text.from_markup(
            dashboard.render(dashboard.latest_frame(), width=90)
        ).plain
        assert "42.0%" in plain


class TestTheCommandLine:
    def _run(self, *argv):
        import contextlib
        import io

        from hypernix.monitoring.tvtoppro import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_list_themes_names_them_all(self):
        code, text = self._run("--list-themes")
        assert code == 0
        for name in THEMES:
            assert name in text
        assert ".theme" in text, "it should say that btop's own themes work"

    def test_dump_theme_writes_btops_format(self):
        code, text = self._run("--dump-theme", "--theme", "dracula")
        assert code == 0
        assert 'theme[cpu_start]="#8be9fd"' in text
        assert parse_btop_theme(text)["cpu_end"] == THEMES["dracula"]["cpu_end"]

    def test_once_draws_a_frame_and_exits(self):
        code, text = self._run("--once", "--width", "80", "--no-processes")
        assert code == 0
        assert "┤cpu├" in text

    def test_once_with_json_emits_the_numbers(self):
        code, text = self._run("--once", "--json")
        assert code == 0
        payload = json.loads(text)
        assert "cpu_percent" in payload and "memory" in payload

    def test_an_unknown_theme_exits_two_with_a_message(self):
        import contextlib
        import io

        from hypernix.monitoring.tvtoppro import main

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["--once", "--theme", "nonesuch"])
        assert code == 2
        assert "Built in:" in err.getvalue()

    def test_the_module_runs_under_dash_m(self):
        """Without a __main__ guard it imports, runs nothing and exits 0
        — which looks exactly like a dashboard that drew an empty
        screen."""
        from pathlib import Path

        source = Path(tvtoppro.__file__).read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in source
