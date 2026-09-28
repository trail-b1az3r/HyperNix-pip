"""What `hypernix` does when you type a command it does not have.

It used to print the whole forty-row menu and exit 1, without ever
mentioning what was typed. `hypernix downlod` produced fifty lines of
table, none of which contained the string "downlod", so a typo looked
exactly like asking for help on purpose — and the only way to notice was
that the thing you asked for had not happened.

With this many subcommands, and pairs like quantize/quantise and
fizzle/fiz, a suggestion is worth more than the whole menu.

The other half of this module is the menu itself. Nine commands
dispatched and were not listed, so `hypernix --help` was not a list of
what hypernix can do — four of them (devices, wakeup, websearch,
hyprslug-headers) had no entry anywhere and no way to be discovered.
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stderr, redirect_stdout

import pytest

from hypernix.interfaces import cli


def _run(*argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestAnUnknownCommandSaysSo:
    def test_it_names_what_was_typed(self):
        """The whole bug in one assertion: the old output never contained
        the thing the user had actually typed."""
        _code, _out, err = _run("downlod")
        assert "downlod" in err

    def test_it_exits_non_zero(self):
        code, _out, _err = _run("downlod")
        assert code != 0

    def test_it_does_not_dump_the_whole_menu(self):
        """Fifty lines of table is not an error message. The failure has
        to be readable in the last few lines of a terminal."""
        _code, out, err = _run("downlod")
        assert len((out + err).splitlines()) < 12

    def test_it_goes_to_stderr(self):
        """So `hypernix downlod | something` does not feed a menu into
        the pipe and call it output."""
        _code, out, err = _run("downlod")
        assert "unknown command" in err
        assert "unknown command" not in out

    def test_it_points_at_the_full_list(self):
        _code, _out, err = _run("xyzzy")
        assert "--help" in err


class TestSuggestions:
    @pytest.mark.parametrize("typo,expected", [
        ("downlod", "download"),
        ("donwload", "download"),
        ("convrt", "convert"),
        ("quantze", "quantize"),
        ("doctr", "doctor"),
        ("genrate", "generate"),
    ])
    def test_a_typo_gets_the_command_back(self, typo, expected):
        _code, _out, err = _run(typo)
        assert expected in err

    @pytest.mark.parametrize("prefix,expected", [
        ("down", "download"),
        ("conv", "convert"),
        ("gen", "generate"),
    ])
    def test_a_prefix_finds_the_command(self, prefix, expected):
        """Ranked before fuzzy matches on purpose: somebody who typed
        `down` meant `download`, and difflib's ratio does not reliably
        put a prefix above a similarly-sized edit."""
        assert cli._suggest(prefix)[0] == expected

    def test_nonsense_gets_no_suggestion_rather_than_a_wrong_one(self):
        """A confidently wrong "did you mean" is worse than none."""
        _code, _out, err = _run("xyzzy")
        assert "did you mean" not in err.lower()

    def test_suggestions_are_real_subcommands(self):
        for typo in ("downlod", "convrt", "trian", "chta"):
            for line in cli._suggest(typo):
                assert line in cli._SUBCOMMANDS, (typo, line)

    def test_it_suggests_at_most_a_few(self):
        """A list of eight guesses is the menu again."""
        assert len(cli._suggest("c")) <= 3


class TestAliases:
    @pytest.mark.parametrize("typed,target", [
        ("quantise", "quantize"),     # British spelling, used in this package's own docs
        ("dl", "download"),
        ("gguf", "convert"),
        ("keys", "gkey"),
        ("hyprslug", "quantize"),     # the tool's name, not the subcommand's
        ("tvtoppro", "tvtop"),        # a console script people expect to work here too
        ("diagnose", "doctor"),
    ])
    def test_a_known_other_spelling_is_named_exactly(self, typed, target):
        """Not a guess — these are the other correct spelling, or the
        name of the tool rather than the name of the subcommand, so the
        message says "You want" rather than "Did you mean"."""
        _code, _out, err = _run(typed)
        assert f"hypernix {target}" in err
        assert "You want" in err

    def test_every_alias_points_at_something_real(self):
        for typed, target in cli._COMMAND_ALIASES.items():
            assert target in cli._SUBCOMMANDS or target.startswith("-"), typed

    def test_no_alias_shadows_a_real_command(self):
        """An alias for a name that already dispatches would never fire,
        and would quietly document the wrong thing."""
        assert not (set(cli._COMMAND_ALIASES) & cli._SUBCOMMANDS)


class TestTheMenuMatchesTheDispatcher:
    @pytest.fixture
    def menu(self) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            cli._print_usage()
        return out.getvalue()

    #: Alternative spellings of a command that is listed under its other
    #: name. Not listing these keeps the table readable; they appear in
    #: the "Also accepted" line instead.
    KNOWN_ALIASES = {
        "camouflage", "protect", "fuse-box", "fiz", "tvtop",
    }

    def test_everything_dispatchable_is_discoverable(self, menu):
        """Four commands dispatched with no menu entry at all, so there
        was no way to find out they existed."""
        listed = set(re.findall(r"│ ([a-z][a-z0-9\-]+)\s+│", menu))
        undiscoverable = cli._SUBCOMMANDS - listed - self.KNOWN_ALIASES
        assert not undiscoverable, f"dispatch but are not in --help: {sorted(undiscoverable)}"

    @pytest.mark.parametrize("command", ["devices", "wakeup", "websearch",
                                         "hyprslug-headers"])
    def test_the_four_that_were_missing_are_listed(self, menu, command):
        assert command in menu

    def test_the_aliases_are_mentioned_somewhere(self, menu):
        for alias in self.KNOWN_ALIASES:
            assert alias in menu, alias

    def test_help_still_exits_zero(self):
        code, _out, _err = _run("--help")
        assert code == 0
