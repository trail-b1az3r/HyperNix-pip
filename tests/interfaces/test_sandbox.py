"""``hnx sandbox`` / ``hnx sb`` (0.73.0 pre-alpha 1).

Settings parsing and saving, the macOS profile, refusing rather than
pretending, and -- where this machine has unprivileged user namespaces --
real runs that check web access, writes and the home folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hypernix.interfaces import sandbox
from hypernix.interfaces.sandbox import SandboxSettings, SettingsError, parse_settings


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config"


class TestSettings:
    def test_defaults_are_the_careful_ones(self):
        s = SandboxSettings()
        assert (s.web, s.write, s.tmp, s.home, s.env) == ("off", "dir", "on", "hide", "clean")

    def test_short_forms_and_pairs(self):
        s = parse_settings("no-write, web timeout=60 memory=512")
        assert (s.write, s.web, s.timeout, s.memory) == ("none", "on", 60, 512)

    def test_on_top_of_what_was_there(self):
        s = parse_settings("show-home", SandboxSettings(web="on"))
        assert (s.web, s.home) == ("on", "show")

    @pytest.mark.parametrize("bad,match", [
        ("web=maybe", "web: must be one of"), ("colour=red", "unknown setting 'colour'"),
        ("timeout=-1", "between"), ("timeout=soon", "whole number"), ("nonsense", "unknown setting"),
    ])
    def test_mistakes_are_named(self, bad, match):
        with pytest.raises(SettingsError, match=match):
            parse_settings(bad)

    def test_what_needs_an_os_sandbox(self):
        assert SandboxSettings().needs_isolation() == ["web=off", "write=dir", "tmp=on", "home=hide"]
        loose = parse_settings("web=on write=all tmp=off home=show")
        assert loose.needs_isolation() == []

    def test_saved_per_directory_outside_it(self, tmp_path, config_home):
        work = tmp_path / "work"
        sandbox.save_settings(work, parse_settings("web"))
        assert sandbox.load_settings(work).web == "on"
        assert sandbox.load_settings(tmp_path / "other").web == "off"
        assert (config_home / "hypernix" / "sandbox.json").is_file()
        sandbox.save_settings(work, None)
        assert sandbox.load_settings(work).web == "off"

    def test_a_hand_edited_bad_value_keeps_the_default(self):
        assert SandboxSettings.from_dict({"web": "sideways", "timeout": 5}).web == "off"


class TestEnvironment:
    def test_secrets_stay_out(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_x")
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("SOME_PLAIN_THING", "1")
        env = sandbox._environment(SandboxSettings())
        assert "HF_TOKEN" not in env and "OPENAI_API_KEY" not in env
        assert env["SOME_PLAIN_THING"] == "1" and env["HNX_SANDBOX"] == "1" and "PATH" in env
        assert "HF_TOKEN" in sandbox._environment(SandboxSettings(env="keep"))


class TestMacProfile:
    def test_the_profile_denies_what_is_off(self, tmp_path):
        text = sandbox.seatbelt_profile(tmp_path, SandboxSettings(), "/Users/me")
        assert "(deny network*)" in text and "(deny file-write*)" in text
        assert f'(allow file-write* (subpath "{tmp_path}"))' in text
        assert '(deny file-read* (subpath "/Users/me"))' in text

    def test_no_write_allows_no_directory(self, tmp_path):
        text = sandbox.seatbelt_profile(tmp_path, parse_settings("no-write no-tmp"), "/Users/me")
        assert f'(subpath "{tmp_path}"))' not in text.split("(deny file-read*")[0]
        assert "/private/tmp" not in text

    def test_quotes_in_paths_cannot_break_out(self):
        text = sandbox.seatbelt_profile(Path('/x/") (allow network*'), SandboxSettings(), "")
        assert '\\")' in text

    def test_everything_loose_is_just_allow_default(self, tmp_path):
        text = sandbox.seatbelt_profile(tmp_path, parse_settings("web write=all show-home"), "/h")
        assert text.strip() == "(version 1)\n(allow default)"


class TestTheCommand:
    def test_show(self, tmp_path, capsys):
        assert sandbox.cli_main([str(tmp_path / "w"), "-s", "web", "--show"]) == 0
        out = capsys.readouterr().out
        assert "web       on" in out and (tmp_path / "w").is_dir()

    def test_refuses_rather_than_pretends(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sandbox, "backend", lambda: "")
        assert sandbox.cli_main([str(tmp_path), "--", "true"]) == 3
        assert "Refusing" in capsys.readouterr().err

    def test_loosened_runs_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sandbox, "backend", lambda: "")
        code = sandbox.cli_main([str(tmp_path), "-s", "web write=all tmp=off show-home",
                                 "--", sys.executable, "-c",
                                 "import os; open('made','w').write(os.environ['HNX_SANDBOX'])"])
        assert code == 0 and (tmp_path / "made").read_text() == "1"

    def test_the_timeout_kills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sandbox, "backend", lambda: "")
        code = sandbox.cli_main([str(tmp_path), "-s", "web write=all tmp=off show-home timeout=1",
                                 "--", sys.executable, "-c", "import time; time.sleep(30)"])
        assert code == 124

    def test_bad_settings_are_exit_2(self, tmp_path):
        assert sandbox.cli_main([str(tmp_path), "-s", "web=maybe"]) == 2

    def test_no_save_is_not_saved(self, tmp_path):
        sandbox.cli_main([str(tmp_path), "-s", "web", "--no-save", "--show"])
        assert sandbox.load_settings(tmp_path.resolve()).web == "off"

    def test_options_after_the_dashes_belong_to_the_command(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(sandbox, "backend", lambda: "")
        monkeypatch.setattr(sandbox, "plan_for", lambda d, s, argv, how: seen.setdefault("argv", argv)
                            or ["true"])
        sandbox.cli_main([str(tmp_path), "-s", "web write=all tmp=off show-home", "--no-save",
                          "--", "ls", "--show", "--", "-s"])
        assert seen["argv"] == ["ls", "--show", "--", "-s"]

    def test_hnx_and_sb_both_reach_it(self, tmp_path, capsys):
        from hypernix.interfaces.cli import main

        assert main(["sb", str(tmp_path), "--show"]) == 0
        assert main(["sandbox", str(tmp_path), "--show"]) == 0
        assert capsys.readouterr().out.count("hnx sandbox") == 2


linux_ns = pytest.mark.skipif(sandbox.backend() != "linux",
                              reason="needs Linux with unshare and unprivileged user namespaces")


@linux_ns
class TestReallySandboxed:
    def _run(self, work: Path, script: str, *settings: str) -> int:
        args = [str(work)]
        if settings:
            args += ["-s", " ".join(settings), "--no-save"]
        return sandbox.cli_main([*args, "--", sys.executable, "-c", script])

    def test_writes_inside_but_not_outside(self, tmp_path):
        work, outside = tmp_path / "work", tmp_path / "outside.txt"
        script = (f"open('in.txt','w').write('x')\n"
                  f"try:\n open({str(outside)!r},'w')\nexcept OSError:\n open('blocked','w')")
        assert self._run(work, script, "show-home", "no-tmp") == 0
        assert (work / "in.txt").exists() and (work / "blocked").exists() and not outside.exists()

    def test_no_write_means_no_file_creation_at_all(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "keep.txt").write_text("original")
        script = ("import sys\ntry:\n open('keep.txt','w').write('changed')\n"
                  "except OSError:\n sys.exit(7)")
        assert self._run(work, script, "no-write", "show-home") == 7
        assert (work / "keep.txt").read_text() == "original"

    def test_no_web(self, tmp_path):
        script = ("import socket, sys\ntry:\n socket.create_connection(('1.1.1.1', 443), 3)\n"
                  "except OSError:\n sys.exit(9)")
        assert self._run(tmp_path / "w", script, "no-web", "show-home") == 9

    def test_the_home_folder_is_hidden(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "secrets").mkdir(parents=True)
        (home / "secrets" / "key").write_text("s")
        monkeypatch.setenv("HOME", str(home))
        work = home / "project"
        script = ("import os, sys\nsys.exit(0 if not os.path.exists(os.path.expanduser('~/secrets'))"
                  " and os.path.isdir(os.getcwd()) else 5)")
        assert self._run(work, script) == 0

    def test_the_command_is_not_root(self, tmp_path):
        script = f"import os, sys\nsys.exit(0 if os.getuid() == {os.getuid()} else 4)"
        assert self._run(tmp_path / "w", script, "show-home") == 0
