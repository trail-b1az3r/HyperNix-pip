"""``hnx list-all`` (0.72.8a4): every command, and kept honest by these tests.

Three tables could drift from what they describe: the hnx menus (read
straight from the CLI), the T1 manager's commands (a table here, checked
against bin/hypernix-t1's dispatcher), and the console scripts (read from
the installed entry points or pyproject.toml).
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from hypernix.interfaces import cli, list_all

ROOT = Path(__file__).resolve().parents[2]


def _rows(*argv: str, capsys) -> list[dict]:
    assert list_all.cli_main(["--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_every_hnx_subcommand_is_listed(capsys):
    names = {r["name"] for r in _rows(capsys=capsys) if r["section"] in ("hnx", "extra", "aliases")}
    assert cli._SUBCOMMANDS <= names


def test_typo_suggestions_are_not_listed_as_commands(capsys):
    """cli._COMMAND_ALIASES ("dl", "quantise") are printed after a typo;
    they do not run, so listing them would send people to an error."""
    names = {r["name"] for r in _rows("--section", "aliases", capsys=capsys)}
    assert not (set(cli._COMMAND_ALIASES) - cli._SUBCOMMANDS) & names


def test_every_console_script_is_listed(capsys):
    rows = _rows("--section", "other", capsys=capsys)
    shown = {r["name"] for r in rows} | {a for r in rows for a in r["also"]}
    scripts = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"])
    assert scripts - {"hypernix", "hnx", "hypernix-quantize"} <= shown


def test_the_t1_table_matches_the_dispatcher():
    source = (ROOT / "bin" / "hypernix-t1").read_text()
    body = source[source.index('  case "$cmd" in'):]
    body = body[:body.index("\n  esac")]
    labels = set()
    for line in body.splitlines():
        m = re.match(r"\s+([a-z|\-]+)\)\s", line)
        if m:
            labels |= set(m.group(1).split("|"))
    listed = set(list_all.T1_COMMANDS) | {a for also, _ in list_all.T1_COMMANDS.values() for a in also}
    assert labels and labels <= listed, labels - listed
    assert {"setup", "upgrade"} <= labels


def test_words_filter_and_sections(capsys):
    rows = _rows("quant", capsys=capsys)
    assert rows and all("quant" in (r["name"] + r["what"] + " ".join(r["also"])).lower() for r in rows)
    assert {r["section"] for r in _rows("--section", "t1", capsys=capsys)} == {"t1"}


def test_the_plain_listing_prints_and_nothing_matching_is_an_error(capsys):
    assert list_all.cli_main([]) == 0
    out = capsys.readouterr().out
    assert "hnx-t1 setup" in out and "hnx list-all" in out and "waiter" in out
    assert list_all.cli_main(["zzzz-no-such-thing"]) == 1


def test_it_is_a_subcommand():
    assert "list-all" in cli._SUBCOMMANDS
    assert any(name == "list-all" for name, _ in cli._MAIN_MENU)
